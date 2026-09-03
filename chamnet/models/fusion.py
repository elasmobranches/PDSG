"""RGB-D 융합 모듈.

원본에서는 CrossModalGating 이 dual_mit.py 에 정의되어 SD·HD 8개 백본이 MiT 파일에서
게이트를 import 했고, BiGateGating 은 dual_resnet_bigate.py 와 dual_mit_bigate.py 에
docstring 만 다른 채로 복제돼 있었다. 여기로 모아 한 벌만 둔다.

Extracted verbatim from mmsegmentation/mmseg/models/backbones/dual_mit.py
(_DWBlock, DepthBranch, CrossModalGating) and dual_resnet_bigate.py
(BiGateGating; the dual_mit_bigate.py copy was discarded, docstring-only diff).
No class body was modified during the move.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Depth Branch
# ---------------------------------------------------------------------------

class _DWBlock(nn.Module):
    """Depthwise-separable conv block with optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class DepthBranch(nn.Module):
    """4-stage lightweight depth feature extractor.

    Outputs depth feature maps at the same spatial resolution and channel
    count as each MiT-B0 stage output:

      stage 1: (B, C0, H/4,  W/4)   matched to MiT PatchEmbed stride=4
      stage 2: (B, C1, H/8,  W/8)
      stage 3: (B, C2, H/16, W/16)
      stage 4: (B, C3, H/32, W/32)

    Default embed_dims=(32, 64, 160, 256) for MiT-B0.
    Total parameters: ~0.20 M
    """

    def __init__(self, embed_dims=(32, 64, 160, 256)):
        super().__init__()
        C0, C1, C2, C3 = embed_dims

        # Stage 1: 1ch → C0, effective stride=4 via two stride-2 operations.
        # Matches MiT-B0 stage-0 PatchEmbed (kernel=7, stride=4).
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, C0 // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C0 // 2),
            nn.GELU(),
            _DWBlock(C0 // 2, C0, stride=2),
        )
        # Stages 2-4: each halves spatial dimensions.
        self.stage2 = _DWBlock(C0, C1, stride=2)
        self.stage3 = _DWBlock(C1, C2, stride=2)
        self.stage4 = _DWBlock(C2, C3, stride=2)

    def forward(self, depth: torch.Tensor):
        """
        Args:
            depth: (B, 1, H, W)
        Returns:
            Tuple of 4 feature maps at decreasing resolutions.
        """
        f1 = self.stage1(depth)   # (B, C0, H/4,  W/4)
        f2 = self.stage2(f1)      # (B, C1, H/8,  W/8)
        f3 = self.stage3(f2)      # (B, C2, H/16, W/16)
        f4 = self.stage4(f3)      # (B, C3, H/32, W/32)
        return (f1, f2, f3, f4)


# ---------------------------------------------------------------------------
# Cross-Modal Gating
# ---------------------------------------------------------------------------

class CrossModalGating(nn.Module):
    """Depth-guided channel gating for RGB feature enhancement.

    Mechanism (CBAM-style dual pooling):
      f_avg  = GAP(depth_feat)                  shape: (B, C)   — global mean context
      f_max  = GMP(depth_feat)                  shape: (B, C)   — peak / edge context
      gate   = Sigmoid( FC2( ReLU( FC1( cat[f_avg, f_max] ) ) ) )  shape: (B, C, 1, 1)
      d_proj = Conv1x1 + BN ( depth_feat )      shape: (B, C, H, W)
      output = rgb_feat  +  d_proj * gate

    Why dual pooling instead of GAP-only:
      - GAP captures average depth level  → scene-level structure (e.g. near/far zones)
      - GMP captures peak responses       → local edges, boundaries, salient structures
        (GMP implicitly encodes high-gradient regions without explicit Sobel computation)
      Together they let the gate distinguish flat regions from structured ones,
      allowing depth to contribute more strongly where geometry is informative.

    NOTE on gate behaviour (measured, not assumed): the gate MLP has no bias
    and ends in a sigmoid, so a zero input yields exactly 0.5 — NOT 0.  On
    trained HD checkpoints the mean gate value is essentially invariant to
    test-time depth corruption severity (e.g. MiT-B0 stage gates stay within
    ±0.002 from s=0 to s=0.75 zero-masking; ResNet-18 deep-stage gates OPEN
    slightly, 0.348→0.365).  The gate therefore acts as a learned per-channel
    prior on depth-feature magnitude, not a dynamic corruption detector.
    Robustness to corrupted depth comes from the asymmetric additive topology
    (RGB features are never multiplied by depth-derived signals), not from
    gate closure.  See _scratch/gate_vs_severity.py.

    Added parameters vs. GAP-only: FC1 input doubles (C→2C), negligible overhead.
    """

    def __init__(self, channels: int, reduction: int = 4, use_gate: bool = True,
                 gate_type: str = 'channel', pool_mode: str = 'both',
                 depth_proj_zero_init: bool = False,
                 gate_cond: str = 'depth', gate_bias: bool = False,
                 gate_init=None, fixed_gate=None,
                 proj_type: str = 'conv1x1',
                 spatial_residual: bool = False,
                 lambda_c: float = 0.5, lambda_s: float = 0.5,
                 gate_dw: bool = False):
        super().__init__()
        assert gate_cond in {'depth', 'both'}, gate_cond
        assert proj_type in {'conv1x1', 'dwsep3x3'}, proj_type
        self.proj_type = proj_type
        self.depth_proj_zero_init = depth_proj_zero_init
        self.gate_cond = gate_cond
        self.gate_bias = gate_bias
        self.gate_init = gate_init
        if fixed_gate is not None and not 0.0 < float(fixed_gate) < 1.0:
            raise ValueError('fixed_gate must be in (0, 1)')
        self.fixed_gate = None if fixed_gate is None else float(fixed_gate)
        self.use_gate = use_gate
        self.gate_type = gate_type
        self.pool_mode = pool_mode  # 'avg' | 'max' | 'both' (default: CBAM-style dual)
        self.spatial_residual = bool(spatial_residual)

        if use_gate and gate_type == 'channel':
            assert pool_mode in {'avg', 'max', 'both'}, pool_mode
            mid = max(channels // reduction, 8)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global mean — scene-level
            self.max_pool = nn.AdaptiveMaxPool2d(1)  # global peak — edges/boundaries
            # Optional depthwise 3x3 BEFORE pooling (PDAM, Zhu et al. AAAI 2025,
            # does this in its channel branch).
            #
            # Why it may matter here specifically. GAP destroys all spatial
            # information: one number per channel. Greenhouse scenes have
            # similar depth statistics frame to frame, so GAP(d) and GMP(d) are
            # nearly constant across the dataset and the gate has almost nothing
            # to condition on. That is consistent with what was measured on
            # trained HD checkpoints -- the gate is invariant to depth
            # corruption severity (within +-0.002 from s=0 to s=0.75) and acts
            # as a learned constant.
            #
            # A depthwise 3x3 first lets each channel respond to local structure
            # -- edges, gradients -- and pools that response instead. The gate
            # then asks "how much structure of this kind is present" rather than
            # "how bright is this channel", which is a quantity that actually
            # varies. It also closes a gap in the design: depth's entire
            # measured benefit is its spatial arrangement (shuffling it costs
            # 3.2-4.2 pp, collapsing to the RGB-only baseline), yet every gate
            # variant tried so far conditions on pooled statistics that cannot
            # see arrangement at all.
            self.gate_dw = bool(gate_dw)
            if self.gate_dw:
                self.gate_dwconv = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )

            # FC1 input dim depends on pool_mode (1×C for single, 2×C for both)
            # and on gate_cond ('both' appends the same pooling of the RGB
            # feature, so the gate can ask "how much depth does this RGB need?"
            # rather than only "how strong is this depth?").
            in_dim = channels * 2 if pool_mode == 'both' else channels
            if gate_cond == 'both':
                in_dim *= 2
            self.gate = nn.Sequential(
                nn.Linear(in_dim, mid, bias=gate_bias),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels, bias=gate_bias),
                nn.Sigmoid(),
            )
            if gate_bias:
                # Zero the biases so the gate still starts at sigmoid(0)=0.5,
                # matching the bias-free variant at initialisation; training is
                # then free to move the operating point, which the bias-free
                # form cannot do (a zero input is pinned to 0.5).
                for m in self.gate:
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)
                if gate_init is not None:
                    if not 0.0 < float(gate_init) < 1.0:
                        raise ValueError('gate_init must be in (0, 1)')
                    # Make the initial gate exactly gate_init for every input;
                    # the second linear then learns input dependence after
                    # the first optimizer update.
                    nn.init.zeros_(self.gate[2].weight)
                    nn.init.constant_(
                        self.gate[2].bias,
                        math.log(float(gate_init) / (1.0 - float(gate_init))))
            elif gate_init is not None:
                raise ValueError('gate_init requires gate_bias=True')

            if self.spatial_residual:
                # Preserve the original global channel gate, then add a
                # learnable spatial correction in logit space.  The final
                # convolution is zero-initialised, so this option is exactly
                # the original HD module at step zero.
                spatial_mid = max(channels // reduction, 8)
                self.spatial_gate = nn.Sequential(
                    nn.Conv2d(2 * channels, spatial_mid, kernel_size=1,
                              bias=False),
                    nn.GroupNorm(1, spatial_mid),
                    nn.GELU(),
                    nn.Conv2d(spatial_mid, channels, kernel_size=3,
                              padding=1, bias=True),
                )
                nn.init.zeros_(self.spatial_gate[-1].weight)
                nn.init.zeros_(self.spatial_gate[-1].bias)
        elif use_gate and gate_type == 'csres':
            # Parallel channel + spatial attention combined by addition, on a
            # residual base -- PDAM's combination rule (Zhu et al., AAAI 2025)
            # ported into this module's asymmetric topology.
            #
            #   PDAM:  out_i = feat_i * (1 + Lc*Ac_i + Ls*As_i)   every branch
            #   here:  out   = rgb + d_proj * (1 + Lc*Ac + Ls*As) depth only
            #
            # RGB is never multiplied by a depth-derived signal, so the
            # asymmetry the paper's corruption-robustness argument rests on is
            # preserved. What is taken from PDAM is the *combination*: channel
            # and spatial attention computed in parallel and added, on top of an
            # identity path -- not its symmetric modulation of every branch.
            #
            # Why a residual base rather than a (0,1) gate. The gate range is
            # the thing being changed here, deliberately. In v12, NoGate --
            # which is exactly this module with the multiplier pinned to 1 --
            # scored within 0.02 pp of HD, and the trained channel gate was
            # measured to sit near a constant 0.35. So the ability to suppress
            # depth buys nothing on this dataset while costing about 65% of the
            # depth signal. This variant makes 1x the floor and lets attention
            # add up to 1x more, so it is a strict superset of NoGate: at
            # Ac = As = 0 it IS NoGate.
            #
            # Range is (1, 1 + Lc + Ls) = (1, 2) at the defaults. Note the
            # consequence at initialisation: both sigmoids start near 0.5, so
            # depth enters at ~1.5x against the channel gate's ~0.5x. That is a
            # real difference in starting regime and part of what is being
            # tested -- which is why spatial_residual, same (0,1) range with
            # spatial added, is run alongside to separate "spatial helps" from
            # "more depth helps".
            assert lambda_c >= 0 and lambda_s >= 0, (lambda_c, lambda_s)
            mid = max(channels // reduction, 8)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            # PDAM applies a depthwise 3x3 before pooling in its channel branch;
            # gate_dw=True reproduces that, gate_dw=False isolates the
            # combination rule from it. See the note in the 'channel' branch.
            self.gate_dw = bool(gate_dw)
            if self.gate_dw:
                self.gate_dwconv = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                              groups=channels, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            self.cs_channel = nn.Sequential(
                nn.Linear(channels * 2, mid, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid, channels, bias=False),
                nn.Sigmoid(),
            )
            # Spatial branch follows PDAM rather than CBAM: 1x1 convolutions
            # over the full channel content, not a 7x7 over avg/max-pooled
            # summaries. CBAM's form is already available as gate_type='spatial'
            # and is run as the contrast.
            self.cs_spatial = nn.Sequential(
                nn.Conv2d(channels, mid, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.lambda_c = float(lambda_c)
            self.lambda_s = float(lambda_s)
        elif use_gate and gate_type == 'agg':
            # SA-Gate (ECCV 2020) Feature Aggregation Part: per-pixel softmax
            # over the two modalities. Unlike the channel gate this is spatially
            # resolved and zero-sum (A_rgb + A_depth = 1 at every location), so
            # it expresses "trust depth *here*" rather than "scale channel c".
            # Motivation for trying it: the channel gate was measured to be
            # indistinguishable from a learned constant, whereas a per-pixel
            # zero-sum selection has no reason to collapse the same way.
            self.agg_rgb = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
            self.agg_depth = nn.Conv2d(channels * 2, 1, kernel_size=1, bias=True)
            self.agg_softmax = nn.Softmax(dim=1)
        elif use_gate and gate_type == 'spatial':
            # CBAM-style spatial gate: [avg,max along channel] → 7×7 conv → sigmoid
            # Output: (B, 1, H, W) — broadcasts across channels (location-wise modulation)
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.spatial_sigmoid = nn.Sigmoid()
        elif use_gate:
            raise ValueError(
                f"Unknown gate_type: {gate_type!r}. "
                "Expected 'channel', 'spatial', 'agg' or 'csres'.")

        # Spatial depth feature projection to RGB channel space (항상 사용)
        if proj_type == 'dwsep3x3':
            # A 1x1 projection cannot reorganise depth spatially: it mixes
            # channels pointwise only. Since the measured benefit of depth is
            # entirely its spatial structure (shuffling it collapses to the
            # RGB-only baseline), give the injection one depthwise 3x3 before
            # the pointwise mix. Cost: 9C extra parameters per stage.
            self.depth_proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                          groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.depth_proj = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        self._bn_index = 2 if proj_type == 'dwsep3x3' else 1
        if depth_proj_zero_init:
            # Start with zero depth contribution so the optimiser decides how
            # much depth to admit, the way early fusion's zero-init stem does.
            # Caveat: gamma gates the whole upstream depth encoder, so its
            # gradient is zero until gamma leaves zero - log the trajectory.
            nn.init.zeros_(self.depth_proj[self._bn_index].weight)

    def _global_channel_gate(self, depth: torch.Tensor,
                             rgb: torch.Tensor) -> torch.Tensor:
        """Return the original GAP/GMP channel gate without spatial terms."""
        B = depth.shape[0]
        # gate_dw pools the depthwise response instead of the raw feature, so
        # the gate conditions on local structure rather than channel brightness
        depth = self.gate_dwconv(depth) if getattr(self, 'gate_dw', False) \
            else depth
        if self.pool_mode == 'avg':
            feat = self.avg_pool(depth).view(B, -1)
        elif self.pool_mode == 'max':
            feat = self.max_pool(depth).view(B, -1)
        else:
            feat = torch.cat([
                self.avg_pool(depth).view(B, -1),
                self.max_pool(depth).view(B, -1)], dim=1)
        if self.gate_cond == 'both':
            if self.pool_mode == 'avg':
                rgb_feat = self.avg_pool(rgb).view(B, -1)
            elif self.pool_mode == 'max':
                rgb_feat = self.max_pool(rgb).view(B, -1)
            else:
                rgb_feat = torch.cat([
                    self.avg_pool(rgb).view(B, -1),
                    self.max_pool(rgb).view(B, -1)], dim=1)
            feat = torch.cat([feat, rgb_feat], dim=1)
        return self.gate(feat).view(B, -1, 1, 1)

    def _gate(self, depth: torch.Tensor, rgb: torch.Tensor,
              d_proj: torch.Tensor = None) -> torch.Tensor:
        global_gate = self._global_channel_gate(depth, rgb)
        if not self.spatial_residual:
            return global_gate
        if d_proj is None:
            d_proj = self.depth_proj(depth)
        global_logit = torch.logit(global_gate.clamp(1e-6, 1 - 1e-6))
        spatial_logit = self.spatial_gate(torch.cat([rgb, d_proj], dim=1))
        return torch.sigmoid(global_logit + spatial_logit)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:   (B, C, H, W)  RGB feature from MiT-B0 stage
            depth: (B, C, H, W)  depth feature from DepthBranch stage
        Returns:
            (B, C, H, W)  enhanced RGB feature
        """
        d_proj = self.depth_proj(depth)
        if not self.use_gate:
            # Ablation: no gating — depth contribution is always full
            return rgb + d_proj
        if self.fixed_gate is not None:
            # Fixed-strength residual ablation: projection and depth encoder
            # remain learnable, but the CMG decision itself is disabled.
            return rgb + d_proj * self.fixed_gate
        if self.gate_type == 'agg':
            cat = torch.cat([rgb, d_proj], dim=1)                # (B, 2C, H, W)
            a = self.agg_softmax(torch.cat(
                [self.agg_rgb(cat), self.agg_depth(cat)], dim=1))  # (B, 2, H, W)
            merge = rgb * a[:, 0:1] + d_proj * a[:, 1:2]
            # Mirrors SA-Gate's rgb_out = (rgb + merge)/2, so the RGB stream is
            # attenuated to between 0.5x and 1.0x rather than replaced.
            return (rgb + merge) / 2
        if self.gate_type == 'csres':
            b, c = depth.shape[0], depth.shape[1]
            src = self.gate_dwconv(depth) if self.gate_dw else depth
            pooled = torch.cat([self.avg_pool(src).flatten(1),
                                self.max_pool(src).flatten(1)], dim=1)
            a_c = self.cs_channel(pooled).view(b, c, 1, 1)   # (B, C, 1, 1)
            a_s = self.cs_spatial(depth)                     # (B, 1, H, W)
            # broadcasts to (B, C, H, W); each sigmoid is independent, so the
            # sum lies in (1, 1 + lambda_c + lambda_s)
            w = 1.0 + self.lambda_c * a_c + self.lambda_s * a_s
            return rgb + d_proj * w

        if self.gate_type == 'channel':
            w = self._gate(depth, rgb, d_proj=d_proj)
            return rgb + d_proj * w
        # spatial gate: collapse channel dim, mix with 7×7 conv, broadcast over C
        s_avg = depth.mean(dim=1, keepdim=True)         # (B, 1, H, W) — channel mean
        s_max = depth.max(dim=1, keepdim=True).values   # (B, 1, H, W) — channel peak
        M = self.spatial_sigmoid(
            self.spatial_conv(torch.cat([s_avg, s_max], dim=1)))  # (B, 1, H, W)
        return rgb + d_proj * M


# ---------------------------------------------------------------------------
# Bidirectional Gating (HD-BiGate)
# ---------------------------------------------------------------------------

class BiGateGating(nn.Module):
    """Bidirectional multiplicative channel attention.

    Each modality (RGB, depth) gets its own channel gate (GAP+GMP → MLP → sigmoid),
    then the gated features are summed: rgb*w_rgb + d_proj*w_d.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # RGB self-attention branch
        self.gate_rgb = nn.Sequential(
            nn.Linear(channels * 2, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

        # Depth self-attention branch
        self.gate_d = nn.Sequential(
            nn.Linear(channels * 2, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

        # Depth → RGB-channel projection (same form as CMG)
        self.depth_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _channel_gate(self, x: torch.Tensor, gate_module: nn.Module) -> torch.Tensor:
        B = x.shape[0]
        f_avg = self.avg_pool(x).view(B, -1)
        f_max = self.max_pool(x).view(B, -1)
        return gate_module(torch.cat([f_avg, f_max], dim=1)).view(B, -1, 1, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        d_proj = self.depth_proj(depth)
        w_rgb = self._channel_gate(rgb,   self.gate_rgb)   # (B, C, 1, 1)
        w_d   = self._channel_gate(depth, self.gate_d)     # (B, C, 1, 1)
        return rgb * w_rgb + d_proj * w_d

