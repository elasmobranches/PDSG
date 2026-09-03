"""Dual-encoder backbone for RGB + Depth semantic segmentation (MiT-B0).

Architecture overview
---------------------
                   ┌─────────────────────────────────────────┐
  RGB (3ch)  ──▶  │  MiT-B0 (pretrained, 4 stages)          │ ──▶ fused features ──▶ SegformerHead
                   │    stage i ──▶ CrossModalGating[i] ──▶  │
  Depth (1ch) ──▶ │  DepthBranch (4-stage DW-sep CNN)        │
                   └─────────────────────────────────────────┘

CrossModalGating (per stage):
  1. [GAP(depth) ; GMP(depth)] → FC → Sigmoid → gate w  (B,C,1,1)
       GAP: scene-level depth stats (near/far zones)
       GMP: edge/boundary stats (high-gradient regions)
  2. depth_feat → Conv1x1+BN → depth_proj  d  (B,C,H,W)
  3. output = rgb_feat  +  d * w     ← residual addition
     → if depth is zero / constant, both pools ≈ 0 → gate ≈ 0
     → graceful fallback to pure-RGB baseline

Design choices
--------------
* Asymmetric:  heavy transformer (MiT-B0) for semantics,
               lightweight CNN (DepthBranch) for geometric priors
* Non-invasive: pretrained MiT-B0 weights loaded as-is (no weight surgery)
* Residual:    additive fusion never hurts RGB stream
* Multi-scale: fusion at all 4 stages → low-level edges + high-level context

Parameter count (B0 config):
  MiT-B0            ~3.72 M
  DepthBranch        ~0.20 M
  CrossModalGating   ~0.15 M
  ──────────────────────────
  Total              ~4.07 M   (+9% over RGB baseline)

Ported verbatim from mmsegmentation/mmseg/models/backbones/dual_mit.py —
only the imports changed: DepthBranch and CrossModalGating (originally
defined in that file) now come from chamnet.models.fusion (this package's
single home for them — dual_mit.py was their one and only defining module
upstream, and SD/HD's other backbones already imported them from there
too), and MixVisionTransformer
comes from mmseg's own backbones module instead of a sibling file in the
same package. JointCMGFusion/BidirectionalCMGFusion, an unrelated
experimental fusion module also defined in the source file but never used
by DualMiTB0 (or any of the four SD backbones), was not ported. No class
body was modified during the move.

The HD (Dual+) sibling backbone, DualMiTB0LateFusion, is appended below
from dual_mit_late.py; see the section comment above it.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.utils import nlc_to_nchw

from chamnet.models.fusion import BiGateGating, CrossModalGating, DepthBranch
from chamnet.models.depth_pretrain import load_rgb_into_depth_encoder


# ---------------------------------------------------------------------------
# Dual Encoder: DualMiTB0
# ---------------------------------------------------------------------------

@MODELS.register_module()
class DualMiTB0(MixVisionTransformer):
    """Dual-encoder SegFormer backbone: MiT-B0 (RGB) + DepthBranch + fusion.

    Input tensor: (B, 4, H, W)  —  channels 0:3 = RGB,  channel 3 = depth.

    The RGB stream runs through the standard MiT-B0 transformer stages.
    After each stage, cross-modal gating injects depth-derived cues as a
    residual, enriching the RGB features without modifying pretrained weights.

    The depth branch is a tiny DW-sep CNN (~0.20 M params) trained from
    scratch.  MiT-B0 weights are loaded from the ImageNet pretrained
    checkpoint; depth branch and gating modules are randomly initialised.

    Usage in config::

        backbone=dict(
            type='DualMiTB0',
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.1,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
            fusion_reduction=4)
    """

    def __init__(self, fusion_reduction: int = 4, **kwargs):
        # Force in_channels=3 so MiT-B0 patch embed stays unchanged.
        # The 4th (depth) channel is handled by DepthBranch.
        kwargs['in_channels'] = 3
        super().__init__(**kwargs)

        # Channel dimensions per stage: embed_dims × num_heads[i]
        # MiT-B0 default: [32, 64, 160, 256]
        stage_dims = tuple(self.embed_dims * h for h in self.num_heads)

        self.depth_branch = DepthBranch(embed_dims=stage_dims)
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in stage_dims
        ])

    def init_weights(self):
        """Load pretrained MiT-B0 for RGB stream; randomly init depth modules."""
        if (self.init_cfg is not None
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint_path = self.init_cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # strict=False: depth_branch.* and fusions.* keys absent in
            # checkpoint → these are skipped and stay randomly initialised.
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualMiTB0] RGB encoder (MiT-B0) loaded from pretrained. '
                'DepthBranch and CrossModalGating modules: random init.',
                logger='current')
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — RGB + depth concatenated by LoadDepthAsChannel
        Returns:
            List of feature maps at out_indices stages.
        """
        rgb = x[:, :3]    # (B, 3, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)

        # Pre-compute all depth features (cheap, single forward pass)
        depth_feats = self.depth_branch(depth)  # tuple of 4

        outs = []
        feat = rgb  # start as NCHW image; PatchEmbed accepts NCHW

        for i, layer in enumerate(self.layers):
            # 1. PatchEmbed: NCHW → token sequence NLC
            feat, hw_shape = layer[0](feat)
            # 2. Transformer blocks (EfficientMHA + MixFFN)
            for block in layer[1]:
                feat = block(feat, hw_shape)
            # 3. LayerNorm, then convert back to NCHW
            feat = layer[2](feat)
            feat = nlc_to_nchw(feat, hw_shape)

            # 4. Depth-guided cross-modal gating (residual)
            feat = self.fusions[i](feat, depth_feats[i])

            if i in self.out_indices:
                outs.append(feat)
            # feat is NCHW here → feeds next stage's PatchEmbed directly

        return outs


# ---------------------------------------------------------------------------
# HD (Dual+, heavy depth-branch): full MiT-B0 on the depth stream
#
# Ported verbatim from
# mmsegmentation/mmseg/models/backbones/dual_mit_late.py — only the imports
# changed (CrossModalGating from chamnet.models.fusion,
# load_rgb_into_depth_encoder from chamnet.models.depth_pretrain, and mmseg's
# own MixVisionTransformer/nlc_to_nchw). Of that file's other two classes,
# DualMiTB0LateFusionRGB is ported in the control-arm section at the bottom of
# this module and DualMiTB0BiCMG is not (no paper run used it). No class body
# was modified during the move.
# ---------------------------------------------------------------------------


@MODELS.register_module()
class DualMiTB0LateFusion(MixVisionTransformer):
    """Full dual-encoder MiT-B0 backbone with Serial Fusion.

    Both RGB and Depth streams use the full MiT-B0 architecture.
    After each RGB stage, CrossModalGating injects depth features as a
    residual; the gated output feeds directly into the next stage
    (stage-by-stage serial injection).

    DualMiTB0 대비 차이: depth 스트림을 경량 DepthBranch 대신 full MiT-B0로 처리.
    주입 메커니즘(serial CrossModalGating)은 동일.

    Args:
        fusion_reduction (int): Reduction ratio in CrossModalGating. Default: 4.
        **kwargs: All other kwargs forwarded to MixVisionTransformer (RGB backbone).
                  Must include embed_dims, num_heads, num_layers, etc.
    """

    FIRST_CONV_KEY = 'layers.0.0.projection.weight'

    def __init__(self, fusion_reduction: int = 4, fusion_use_gate: bool = True,
                 fusion_gate_type: str = 'channel', fusion_pool_mode: str = 'both',
                 fusion_stages=(0, 1, 2, 3), depth_proj_zero_init: bool = False,
                 fusion_gate_cond: str = 'depth',
                 fusion_gate_bias: bool = False,
                 fusion_gate_init=None,
                 fusion_gate_fixed=None,
                 fusion_proj_type: str = 'conv1x1',
                 fusion_spatial_residual: bool = False,
                 fusion_lambda_c: float = 0.5,
                 fusion_lambda_s: float = 0.5,
                 fusion_gate_dw: bool = False,
                 depth_pretrained: bool = False,
                 init_cfg=None, **kwargs):
        kwargs.pop('in_channels', None)  # RGB always 3ch
        self.depth_pretrained = depth_pretrained
        super().__init__(in_channels=3, init_cfg=init_cfg, **kwargs)

        stage_dims = tuple(self.embed_dims * h for h in self.num_heads)

        # Full depth MiT-B0 (1ch, random init) — same structure as RGB
        # Force out_indices=(0,1,2,3) so all 4 stage features are available
        depth_kwargs = dict(kwargs)
        depth_kwargs['out_indices'] = (0, 1, 2, 3)
        self.depth_backbone = MixVisionTransformer(
            in_channels=1,
            init_cfg=None,
            **depth_kwargs,
        )

        # Serial CrossModalGating × 4 (use_gate=False면 단순 additive fusion)
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction,
                             use_gate=fusion_use_gate,
                             gate_type=fusion_gate_type,
                             pool_mode=fusion_pool_mode,
                             depth_proj_zero_init=depth_proj_zero_init,
                             gate_cond=fusion_gate_cond,
                             gate_bias=fusion_gate_bias,
                             gate_init=(None if fusion_gate_init is None
                                        else fusion_gate_init[i]),
                             fixed_gate=fusion_gate_fixed,
                             proj_type=fusion_proj_type,
                             # was missing here while present in the ResNet
                             # sibling, so gate_type='channel' with a spatial
                             # residual was only reachable on ResNet-18
                             spatial_residual=fusion_spatial_residual,
                             lambda_c=fusion_lambda_c,
                             lambda_s=fusion_lambda_s,
                             gate_dw=fusion_gate_dw)
            for i, dim in enumerate(stage_dims)
        ])
        self.fusion_stages = tuple(fusion_stages)

        print_log(
            '[DualMiTB0LateFusion] RGB MiT-B0: pretrained init. '
            f'Depth MiT-B0 (1ch): '
            f'{"pretrained (RGB ckpt)" if depth_pretrained else "random"} init. '
            'Serial stage-by-stage injection. '
            f'Stage dims: {stage_dims}, use_gate={fusion_use_gate}, '
            f'gate_type={fusion_gate_type}, pool_mode={fusion_pool_mode}',
            logger='current')

    def init_weights(self):
        """Load pretrained MiT-B0 for RGB; randomly init depth + fusion modules."""
        if (self.init_cfg is not None
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint_path = self.init_cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strict=False: depth_backbone.* and fusions.* keys absent in checkpoint
            # → skipped, remain randomly initialised
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualMiTB0LateFusion] RGB MiT-B0 loaded from pretrained.',
                logger='current')
            if self.depth_pretrained:
                load_rgb_into_depth_encoder(
                    self.depth_backbone, state_dict, self.FIRST_CONV_KEY,
                    'DualMiTB0LateFusion')
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — BGR + depth concatenated by LoadDepthAsChannel
        Returns:
            List of feature maps at out_indices stages (depth-injected).
        """
        rgb   = x[:, :3]    # (B, 3, H, W)
        depth = x[:, 3:4]   # (B, 1, H, W)

        # Pre-compute all depth features (full MiT-B0 forward, all 4 stages)
        depth_feats = self.depth_backbone(depth)  # [d0, d1, d2, d3]

        # RGB stage-by-stage serial injection (same loop as DualMiTB0)
        feat = rgb
        outs = []
        for i, layer in enumerate(self.layers):
            # PatchEmbed: NCHW → token sequence NLC
            feat, hw_shape = layer[0](feat)
            # Transformer blocks
            for block in layer[1]:
                feat = block(feat, hw_shape)
            # LayerNorm + reshape to NCHW
            feat = layer[2](feat)
            feat = nlc_to_nchw(feat, hw_shape)

            # Depth-guided serial gating (residual); stages outside
            # fusion_stages stay pure RGB.
            if i in self.fusion_stages:
                feat = self.fusions[i](feat, depth_feats[i])

            if i in self.out_indices:
                outs.append(feat)

        return outs


# ---------------------------------------------------------------------------
# Control arms for HD (see chamnet/config/combos.py for the enabled set)
#
#   DualMiTB0BiGate — ported verbatim from
#       mmsegmentation/mmseg/models/backbones/dual_mit_bigate.py; only the
#       import changed (BiGateGating from chamnet.models.fusion — the source
#       file defined its own copy, identical to the ResNet one bar the
#       docstring, and this package keeps a single definition instead).
#   DualMiTB0LateFusionRGB — ported verbatim from dual_mit_late.py, the same
#       source file DualMiTB0LateFusion above came from; no import changed.
#
# The third class in dual_mit_late.py, DualMiTB0BiCMG, is an experimental
# fusion variant that no paper run used and is not ported. No class body was
# modified during either move.
# ---------------------------------------------------------------------------


@MODELS.register_module()
class DualMiTB0BiGate(DualMiTB0LateFusion):
    """HD-BiGate variant for MiT-B0: replaces CMG fusion with bidirectional gating.

    Inherits the full RGB+Depth dual MiT-B0 structure from DualMiTB0LateFusion.
    Only the per-stage fusion modules are replaced, isolating the fusion mechanism.

    Usage in config::

        backbone=dict(
            type='DualMiTB0BiGate',
            embed_dims=32, num_heads=[1, 2, 5, 8], mlp_ratio=4,
            qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.1, num_layers=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            out_indices=(0, 1, 2, 3),
            fusion_reduction=4,
            init_cfg=dict(type='Pretrained', checkpoint=...))
    """

    def __init__(self, fusion_reduction: int = 4, init_cfg=None, **kwargs):
        # NOTE: parent uses fusion_use_gate; force True so CMG is constructed
        # then replaced. (We replace self.fusions below regardless.)
        kwargs.pop('fusion_use_gate', None)
        super().__init__(fusion_reduction=fusion_reduction,
                         fusion_use_gate=True, init_cfg=init_cfg, **kwargs)

        stage_dims = tuple(self.embed_dims * h for h in self.num_heads)

        # Replace CMG with bidirectional channel gating
        self.fusions = nn.ModuleList([
            BiGateGating(dim, reduction=fusion_reduction)
            for dim in stage_dims
        ])

        print_log(
            '[DualMiTB0BiGate] Fusion replaced with bidirectional '
            'multiplicative channel gating. '
            f'Stage dims: {stage_dims}',
            logger='current')


@MODELS.register_module()
class DualMiTB0LateFusionRGB(DualMiTB0LateFusion):
    """HD-RGB ablation (MiT-B0): depth encoder receives RGB (3ch) instead of pseudo-depth.

    두 인코더 모두 동일한 RGB 입력을 받습니다:
      - RGB  인코더: pretrained MiT-B0 (3ch)
      - Depth 인코더: random init MiT-B0 (3ch, RGB 입력)

    capacity 통제군: depth 공간 구조 없이 모델 용량만 동일하게 유지.
    depth 인코더 초기화는 ``depth_pretrained`` 를 따른다 — HD와 같은 값을 주어야 통제된다.

    입력: (B, 3, H, W) — depth 채널 불필요, LoadDepthAsChannel 사용 안 함.

    Mirror of DualResNetV1c18LateFusionRGB but for MiT-B0 backbone.
    """

    def __init__(self, fusion_reduction: int = 4, fusion_use_gate: bool = True,
                 depth_pretrained: bool = False, init_cfg=None, **kwargs):
        # depth_pretrained 는 여기서 이름으로 받아야 한다. **kwargs 로 흘려보내면
        # 아래 depth_kwargs 에 남아 MixVisionTransformer 생성자로 새어 들어간다.
        super().__init__(fusion_reduction=fusion_reduction,
                         fusion_use_gate=fusion_use_gate,
                         depth_pretrained=depth_pretrained,
                         init_cfg=init_cfg, **kwargs)

        # depth_backbone을 3ch (RGB 입력) MiT-B0로 교체
        depth_kwargs = dict(kwargs)
        depth_kwargs.pop('in_channels', None)
        depth_kwargs['out_indices'] = (0, 1, 2, 3)
        self.depth_backbone = MixVisionTransformer(
            in_channels=3,
            init_cfg=None,        # 적재는 init_weights() 에서 처리
            **depth_kwargs,
        )

        print_log(
            '[DualMiTB0LateFusionRGB] Depth encoder: 3ch RGB input, '
            f'{"pretrained (RGB ckpt)" if self.depth_pretrained else "random"} '
            'init (ablation: capacity control without depth geometry).',
            logger='current')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — RGB only (depth 채널 없음)
        Returns:
            List of depth-injected feature maps at out_indices stages.
        """
        rgb   = x[:, :3]   # (B, 3, H, W)
        depth = x[:, :3]   # (B, 3, H, W) — same RGB fed to depth encoder

        depth_feats = self.depth_backbone(depth)  # [d0, d1, d2, d3]

        feat = rgb
        outs = []
        for i, layer in enumerate(self.layers):
            feat, hw_shape = layer[0](feat)
            for block in layer[1]:
                feat = block(feat, hw_shape)
            feat = layer[2](feat)
            feat = nlc_to_nchw(feat, hw_shape)

            feat = self.fusions[i](feat, depth_feats[i])

            if i in self.out_indices:
                outs.append(feat)

        return outs
