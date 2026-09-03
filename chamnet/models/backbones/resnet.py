"""Dual-encoder backbone for RGB + Depth semantic segmentation (ResNet-18).

Architecture overview
---------------------
  RGB (3ch)   ──▶  ResNetV1c-18 (pretrained, 4 stages)   ─┐
                     stage i ──▶ CrossModalGating[i] ──▶    ├──▶ fused features ──▶ UPerHead
  Depth (1ch) ──▶  DepthBranchResNet (4-stage DW-sep CNN) ─┘

Channel/spatial dimensions (ResNet-18, dilations=(1,1,2,4), strides=(1,2,1,1)):
  Stage 0:  64ch,  H/4  × W/4
  Stage 1: 128ch,  H/8  × W/8
  Stage 2: 256ch,  H/8  × W/8  (dilated conv — no spatial reduction)
  Stage 3: 512ch,  H/8  × W/8  (dilated conv — no spatial reduction)

DepthBranchResNet matches these spatial resolutions:
  Stage 0: stride×4 → H/4   (two stride-2 ops)
  Stage 1: stride×2 → H/8
  Stage 2: stride×1 → H/8   (same spatial as stage 1)
  Stage 3: stride×1 → H/8   (same spatial)

CrossModalGating: identical to DualMiTB0 (reused from dual_mit.py)
  gate   = Sigmoid( FC( cat[GAP(d), GMP(d)] ) )   (B, C, 1, 1)
  output = rgb_feat + Conv1x1(depth_feat) * gate

Parameter count (ResNet-18 config):
  ResNetV1c-18      ~11.2 M
  DepthBranchResNet  ~0.3 M
  CrossModalGating   ~0.2 M
  ─────────────────────────
  Total             ~11.7 M   (+4% over RGB baseline)

Ported verbatim from mmsegmentation/mmseg/models/backbones/dual_resnet.py —
only the imports changed: CrossModalGating and _DWBlock now come from
chamnet.models.fusion (this package's single home for them, moved there out
of dual_mit.py), and
ResNetV1c comes from mmseg's own backbones module instead of a sibling file
in the same package. No class body was modified during the move.

The HD (Dual+) sibling backbone, DualResNetV1c18LateFusion, is appended
below from dual_resnet_late.py; see the section comment above it.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.resnet import ResNetV1c

from chamnet.models.fusion import BiGateGating, CrossModalGating, _DWBlock


# ---------------------------------------------------------------------------
# Depth Branch (ResNet spatial resolution variant)
# ---------------------------------------------------------------------------

class DepthBranchResNet(nn.Module):
    """Depth feature extractor matched to ResNetV1c-18 spatial resolutions.

    Stages 0-1 halve spatial resolution; stages 2-3 keep H/8 × W/8
    to match dilated ResNet (dilations=(1,1,2,4)).

    Args:
        stage_dims (tuple): Output channels per stage. Default: (64,128,256,512).
        stage_strides (tuple): stride of stages 1..3 (stage 0 is always 4).
            The default (2, 1, 1) matches the dilated RGB backbone, which holds
            H/8 from stage 1 onward. A non-dilated RGB backbone emits
            H/8, H/16, H/32 instead, and the depth features must follow or the
            gate cannot add them — pass (2, 2, 2) for that case.
    """

    def __init__(self, stage_dims=(64, 128, 256, 512),
                 stage_strides=(2, 1, 1)):
        super().__init__()
        C0, C1, C2, C3 = stage_dims
        s1, s2, s3 = stage_strides

        # Stage 0: 1ch → C0, effective stride=4 via two stride-2 ops
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, C0 // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C0 // 2),
            nn.GELU(),
            _DWBlock(C0 // 2, C0, stride=2),
        )
        # Stages 1-3: stride follows the RGB backbone's schedule
        self.stage2 = _DWBlock(C0, C1, stride=s1)
        self.stage3 = _DWBlock(C1, C2, stride=s2)
        self.stage4 = _DWBlock(C2, C3, stride=s3)

    def forward(self, depth: torch.Tensor):
        """
        Args:
            depth: (B, 1, H, W)
        Returns:
            Tuple of 4 feature maps matching ResNetV1c-18 stage outputs.
        """
        f1 = self.stage1(depth)   # (B, C0, H/4,  W/4)
        f2 = self.stage2(f1)      # (B, C1, H/8,  W/8)
        f3 = self.stage3(f2)      # (B, C2, H/8,  W/8)
        f4 = self.stage4(f3)      # (B, C3, H/8,  W/8)
        return (f1, f2, f3, f4)


# ---------------------------------------------------------------------------
# Dual Encoder: DualResNetV1c18
# ---------------------------------------------------------------------------

@MODELS.register_module()
class DualResNetV1c18(ResNetV1c):
    """Dual-encoder ResNet-18 backbone: ResNetV1c-18 (RGB) + DepthBranch + fusion.

    Input tensor: (B, 4, H, W) — channels 0:3 = BGR, channel 3 = depth.

    The RGB stream runs through standard ResNetV1c-18 stages.
    After each stage, CrossModalGating injects depth-derived cues as a
    residual, enriching RGB features without modifying pretrained weights.

    The depth branch is a lightweight DW-sep CNN trained from scratch.
    ResNetV1c-18 weights are loaded from the ImageNet pretrained checkpoint;
    depth branch and gating modules are randomly initialised.

    Usage in config::

        backbone=dict(
            type='DualResNetV1c18',
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            fusion_reduction=4,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))
    """

    STAGE_DIMS = (64, 128, 256, 512)

    def __init__(self, fusion_reduction: int = 4, stage_dims=None,
                 depth_stage_strides=(2, 1, 1), **kwargs):
        """
        Args:
            stage_dims: per-stage channel widths for the depth branch and the
                gates. Defaults to ``STAGE_DIMS``, which is ResNet-18's
                (64, 128, 256, 512). A deeper RGB backbone needs this passed
                explicitly -- ResNet-50 emits (256, 512, 1024, 2048) and the
                gate adds the depth feature to the RGB feature, so a mismatch
                is a shape error at the first forward, not a silent one.
        """
        # RGB stream always 3ch — depth handled by DepthBranchResNet
        kwargs['in_channels'] = 3
        super().__init__(**kwargs)

        dims = tuple(stage_dims) if stage_dims is not None else self.STAGE_DIMS
        self.depth_branch = DepthBranchResNet(
            stage_dims=dims, stage_strides=tuple(depth_stage_strides))
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in dims
        ])

    def init_weights(self):
        """Load pretrained ResNet-18 for RGB stream; randomly init depth modules."""
        cfg = self.init_cfg
        if isinstance(cfg, list):
            cfg = cfg[0] if cfg else None
        if (cfg is not None and cfg.get('type') == 'Pretrained'):
            checkpoint_path = cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # strict=False: depth_branch.* and fusions.* absent in checkpoint
            # → these keys are skipped and stay randomly initialised.
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualResNetV1c18] RGB encoder (ResNetV1c-18) loaded from pretrained. '
                'DepthBranchResNet and CrossModalGating modules: random init.',
                logger='current')
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — BGR + depth concatenated by LoadDepthAsChannel
        Returns:
            Tuple of feature maps at out_indices stages.
        """
        rgb = x[:, :3]     # (B, 3, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)

        # Pre-compute all depth features (single forward pass)
        depth_feats = self.depth_branch(depth)  # tuple of 4

        # RGB stream: stem + maxpool
        if self.deep_stem:
            rgb = self.stem(rgb)
        else:
            rgb = self.relu(self.norm1(self.conv1(rgb)))
        rgb = self.maxpool(rgb)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rgb = res_layer(rgb)
            # Depth-guided cross-modal gating (residual addition)
            rgb = self.fusions[i](rgb, depth_feats[i])
            if i in self.out_indices:
                outs.append(rgb)

        return tuple(outs)


# ---------------------------------------------------------------------------
# HD (Dual+, heavy depth-branch): full ResNetV1c on the depth stream
#
# Ported verbatim from
# mmsegmentation/mmseg/models/backbones/dual_resnet_late.py — only the imports
# changed (CrossModalGating from chamnet.models.fusion, ResNetV1c from mmseg's
# own backbones module). Of that file's other two classes,
# DualResNetV1c18LateFusionRGB is ported in the control-arm section at the
# bottom of this module and DualResNetV1c18BiCMG is not (no paper run used
# it). No class body was modified during the move.
# ---------------------------------------------------------------------------

_RESNET_STAGE_DIMS = {
    18:  (64, 128, 256, 512),
    34:  (64, 128, 256, 512),
    50:  (256, 512, 1024, 2048),
    101: (256, 512, 1024, 2048),
    152: (256, 512, 1024, 2048),
}


@MODELS.register_module()
class DualResNetV1c18LateFusion(ResNetV1c):
    """Full dual-encoder ResNet-18 backbone with Serial Fusion.

    Both RGB and Depth streams use the full ResNetV1c-18 architecture.
    After each RGB ResLayer, CrossModalGating injects depth features as a
    residual; the gated output feeds directly into the next layer
    (stage-by-stage serial injection).

    DualResNetV1c18 대비 차이: depth 스트림을 DepthBranchResNet 대신
    full ResNetV1c(1ch)로 처리. 주입 메커니즘(serial)은 동일.

    Usage in config::

        backbone=dict(
            type='DualResNetV1c18LateFusion',
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            contract_dilation=True,
            fusion_reduction=4,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))
    """

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
                 depth_in_channels: int = 1,
                 **kwargs):
        # RGB stream always 3ch
        kwargs['in_channels'] = 3
        super().__init__(**kwargs)

        depth_val = kwargs.get('depth', 18)
        self.STAGE_DIMS = _RESNET_STAGE_DIMS[depth_val]
        self.depth_pretrained = depth_pretrained
        # 1 for a raw metric depth map, 3 for HHA. Stays 1 by default so the
        # v12 configs are untouched. forward() slices the geometry channels off
        # the input using this, so it has to match what the pipeline stacked on.
        self.depth_in_channels = depth_in_channels

        # Full depth ResNetV1c (1ch stem). Random init by default; when
        # depth_pretrained=True, init_weights() below loads the same
        # ImageNet checkpoint as the RGB stream into depth_backbone too,
        # channel-averaging stem.0 (3ch->1ch) since the checkpoint was
        # trained on RGB.
        depth_kwargs = dict(kwargs)
        depth_kwargs['in_channels'] = depth_in_channels
        depth_kwargs['init_cfg'] = None        # weight loading handled in init_weights()
        depth_kwargs['out_indices'] = (0, 1, 2, 3)  # all stages
        self.depth_backbone = ResNetV1c(**depth_kwargs)

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
                             spatial_residual=fusion_spatial_residual,
                             lambda_c=fusion_lambda_c,
                             lambda_s=fusion_lambda_s,
                             gate_dw=fusion_gate_dw)
            for i, dim in enumerate(self.STAGE_DIMS)
        ])
        self.fusion_stages = tuple(fusion_stages)

        print_log(
            '[DualResNetV1c18LateFusion] RGB ResNetV1c: pretrained init. '
            f'Depth ResNetV1c ({depth_in_channels}ch): '
            f'{"pretrained (RGB ckpt)" if depth_pretrained else "random"} init. '
            'Serial stage-by-stage injection. '
            f'Stage dims: {self.STAGE_DIMS}, gate_type={fusion_gate_type}, '
            f'pool_mode={fusion_pool_mode}',
            logger='current')

    def init_weights(self):
        """Load pretrained ResNetV1c for RGB; randomly init depth + fusion
        (unless depth_pretrained=True, see below)."""
        cfg = self.init_cfg
        if isinstance(cfg, list):
            cfg = cfg[0] if cfg else None
        if cfg is not None and cfg.get('type') == 'Pretrained':
            checkpoint_path = cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strict=False: depth_backbone.* and fusions.* absent in checkpoint
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualResNetV1c18LateFusion] RGB ResNetV1c loaded from pretrained. '
                'CrossModalGating: random init. Depth ResNetV1c: '
                + ('see next line.' if self.depth_pretrained else 'random init.'),
                logger='current')

            if self.depth_pretrained:
                self._load_pretrained_depth(state_dict)
        else:
            super().init_weights()

    def _load_pretrained_depth(self, state_dict):
        """Load the same ImageNet checkpoint into depth_backbone.

        stem.0.weight is (out,3,3,3) in the checkpoint. How it transfers
        depends on what the depth branch takes:

        depth_in_channels=1 (a raw metric depth map): average across the input
            channels so the 1ch conv starts from the mean of the three
            pretrained RGB filters, which preserves activation scale. The
            standard RGB->1ch transfer trick.
        depth_in_channels=3 (HHA): copy straight across. HHA is a 3-channel
            image-like input, so this is the same kind of init the RGB stream
            gets, with no adaptation.

        Worth noting when reading HHA-versus-raw results: the 3-channel case
        gets a truer pretrained init than the channel-averaged 1-channel case.
        A 1ch input cannot use a 3ch stem directly, so this asymmetry is
        inherent to the comparison rather than a choice -- it is much smaller
        than a random-versus-pretrained difference would be, but it does mean
        HHA benefits marginally more from pretraining than raw depth does.
        """
        n_geo = self.depth_in_channels
        if n_geo not in (1, 3):
            raise ValueError(
                f'depth_pretrained=True supports depth_in_channels of 1 or 3, '
                f'got {n_geo}: there is no sensible transfer of a 3-channel '
                f'stem to {n_geo} channels.')

        depth_state_dict = {}
        for k, v in state_dict.items():
            if k == 'stem.0.weight' and v.dim() == 4 and v.shape[1] == 3 \
                    and n_geo == 1:
                v = v.mean(dim=1, keepdim=True)  # (out,3,3,3) -> (out,1,3,3)
            depth_state_dict[k] = v

        norm_before = self.depth_backbone.stem[0].weight.detach().norm().item()
        missing, unexpected = self.depth_backbone.load_state_dict(
            depth_state_dict, strict=False)
        norm_after = self.depth_backbone.stem[0].weight.detach().norm().item()

        # depth_backbone has no 'fc' (classification head) -> expected unexpected keys.
        real_unexpected = [k for k in unexpected if not k.startswith('fc.')]
        matched = len(state_dict) - len(unexpected)
        if norm_before == norm_after or matched < 50:
            raise RuntimeError(
                f'[DualResNetV1c18LateFusion] depth_pretrained=True but weight '
                f'load looks like a no-op (stem.0 norm {norm_before:.4f} -> '
                f'{norm_after:.4f}, matched {matched}/{len(state_dict)} keys, '
                f'missing={len(missing)}, unexpected={real_unexpected[:5]}). '
                f'Refusing to silently train on random-init depth weights.')
        how = ('stem.0 3ch->1ch channel-averaged' if n_geo == 1
               else 'stem.0 3ch copied directly')
        print_log(
            f'[DualResNetV1c18LateFusion] Depth ResNetV1c: pretrained init '
            f'({how}, depth_in_channels={n_geo}). matched={matched}/'
            f'{len(state_dict)} keys, stem.0 norm {norm_before:.4f} -> '
            f'{norm_after:.4f}.',
            logger='current')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3+depth_in_channels, H, W) — BGR followed by the geometry
                channels the pipeline stacked on: depth (1) or HHA (3).
        Returns:
            Tuple of feature maps at out_indices stages (depth-injected).
        """
        n_geo = self.depth_in_channels
        expected = 3 + n_geo
        if x.shape[1] != expected:
            raise ValueError(
                f'{self.__class__.__name__} expects {expected} input channels '
                f'(3 RGB + {n_geo} geometry) but got {x.shape[1]}. Check that '
                f'depth_in_channels matches the loader in the config.')
        rgb   = x[:, :3]              # (B, 3, H, W)
        depth = x[:, 3:3 + n_geo]     # (B, n_geo, H, W)

        # Pre-compute all depth features (full ResNetV1c forward, all 4 stages)
        depth_feats = self.depth_backbone(depth)  # tuple of 4 NCHW tensors

        # RGB stem + maxpool
        if self.deep_stem:
            rgb = self.stem(rgb)
        else:
            rgb = self.relu(self.norm1(self.conv1(rgb)))
        rgb = self.maxpool(rgb)

        # RGB stage-by-stage serial injection
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rgb = res_layer(rgb)
            # Depth-guided serial gating (residual); stages outside
            # fusion_stages stay pure RGB.
            if i in self.fusion_stages:
                rgb = self.fusions[i](rgb, depth_feats[i])
            if i in self.out_indices:
                outs.append(rgb)

        return tuple(outs)


# ---------------------------------------------------------------------------
# Control arms for HD (see chamnet/config/combos.py for the enabled set)
#
#   DualResNetV1c18BiGate      — ported verbatim from
#       mmsegmentation/mmseg/models/backbones/dual_resnet_bigate.py; only the
#       import changed (BiGateGating from chamnet.models.fusion, this
#       package's single home for the gate modules).
#   DualResNetV1c18LateFusionRGB — ported verbatim from
#       dual_resnet_late.py, the same source file DualResNetV1c18LateFusion
#       above came from; no import changed.
#
# The third class in dual_resnet_late.py, DualResNetV1c18BiCMG, is an
# experimental fusion variant that no paper run used and is not ported.
# No class body was modified during either move.
# ---------------------------------------------------------------------------


@MODELS.register_module()
class DualResNetV1c18BiGate(DualResNetV1c18LateFusion):
    """HD-BiGate variant: replaces CMG fusion with bidirectional multiplicative gating.

    Inherits the full RGB+Depth dual-encoder structure from
    DualResNetV1c18LateFusion. Only the per-stage fusion modules are replaced.

    All other behaviour (stem, depth_backbone init, forward flow, weight loading)
    is unchanged so the comparison isolates the fusion mechanism.

    Usage in config::

        backbone=dict(
            type='DualResNetV1c18BiGate',
            depth=18, num_stages=4,
            out_indices=(0, 1, 2, 3),
            dilations=(1, 1, 2, 4),
            strides=(1, 2, 1, 1),
            norm_cfg=norm_cfg,
            norm_eval=False, style='pytorch', contract_dilation=True,
            fusion_reduction=4,
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))
    """

    def __init__(self, fusion_reduction: int = 4, **kwargs):
        super().__init__(fusion_reduction=fusion_reduction, **kwargs)

        self.fusions = nn.ModuleList([
            BiGateGating(dim, reduction=fusion_reduction)
            for dim in self.STAGE_DIMS
        ])

        print_log(
            '[DualResNetV1c18BiGate] Fusion replaced with bidirectional '
            'multiplicative channel gating. '
            f'Stage dims: {self.STAGE_DIMS}',
            logger='current')


@MODELS.register_module()
class DualResNetV1c18LateFusionRGB(DualResNetV1c18LateFusion):
    """HD-RGB ablation: depth encoder receives RGB (3ch) instead of pseudo-depth.

    두 인코더 모두 동일한 RGB 입력을 받습니다:
      - RGB  인코더: pretrained ResNetV1c-18 (3ch)
      - Depth 인코더: random init ResNetV1c-18 (3ch, RGB 입력)

    capacity 통제군: depth 공간 구조 없이 모델 용량만 동일하게 유지.
    depth 인코더 초기화는 ``depth_pretrained`` 를 따른다 — HD와 같은 값을 주어야
    초기화가 통제된다. v12+ recipe 는 HD 를 depth_pretrained=True 로 돌리므로
    이 통제군도 True 로 맞춘다.

    입력: (B, 3, H, W) — depth 채널 불필요, LoadDepthAsChannel 사용 안 함.
    """

    def __init__(self, fusion_reduction: int = 4, fusion_use_gate: bool = True,
                 fusion_gate_type: str = 'channel',
                 depth_pretrained: bool = False, **kwargs):
        # depth_pretrained 는 여기서 이름으로 받아야 한다. **kwargs 로 흘려보내면
        # 아래 depth_kwargs 에 남아 ResNetV1c 생성자로 새어 들어간다.
        super().__init__(fusion_reduction=fusion_reduction,
                         fusion_use_gate=fusion_use_gate,
                         fusion_gate_type=fusion_gate_type,
                         depth_pretrained=depth_pretrained, **kwargs)

        # 부모와 동일한 방식으로 depth_kwargs 구성 (fusion_* 제외, in_channels=3 오버라이드)
        depth_kwargs = dict(kwargs)
        for _k in ('depth_in_channels',):
            depth_kwargs.pop(_k, None)
        depth_kwargs['in_channels'] = 3   # RGB 3ch
        depth_kwargs['init_cfg'] = None   # 적재는 init_weights() 에서 처리
        depth_kwargs['out_indices'] = (0, 1, 2, 3)
        self.depth_backbone = ResNetV1c(**depth_kwargs)
        # The parent set this to 1 for its own 1ch depth encoder. The RGB
        # control replaced that encoder with a 3ch one, so init_weights()'s
        # pretrained transfer must know to copy the stem straight across.
        self.depth_in_channels = 3

        print_log(
            '[DualResNetV1c18LateFusionRGB] Depth encoder: 3ch RGB input, '
            f'{"pretrained (RGB ckpt)" if self.depth_pretrained else "random"} '
            'init (ablation: capacity control without depth geometry).',
            logger='current')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) — RGB only (depth 채널 없음)
        Returns:
            Tuple of depth-injected feature maps.
        """
        rgb   = x[:, :3]   # (B, 3, H, W)
        depth = x[:, :3]   # (B, 3, H, W) — same RGB fed to depth encoder

        depth_feats = self.depth_backbone(depth)

        if self.deep_stem:
            rgb = self.stem(rgb)
        else:
            rgb = self.relu(self.norm1(self.conv1(rgb)))
        rgb = self.maxpool(rgb)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            rgb = res_layer(rgb)
            rgb = self.fusions[i](rgb, depth_feats[i])
            if i in self.out_indices:
                outs.append(rgb)

        return tuple(outs)
