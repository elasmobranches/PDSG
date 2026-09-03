"""Serial-fusion dual-encoder backbone for ConvNeXt-Atto.

DualConvNeXtAttoSerial (Dual): RGB ConvNeXt-Atto + lightweight DepthBranch,
serial CMG.

Architecture
------------
  RGB (3ch) ──▶ stem
                  │
              [stage_0] ──▶ CrossModalGating[0](rgb, depth[0]) ──▶ feat_0
                  │
              [stage_1] ──▶ CrossModalGating[1](rgb, depth[1]) ──▶ feat_1
                  │
              [stage_2] ──▶ CrossModalGating[2](rgb, depth[2]) ──▶ feat_2
                  │
              [stage_3] ──▶ CrossModalGating[3](rgb, depth[3]) ──▶ feat_3
                                                                        │
  Depth(1ch) ──▶ DepthBranch ──▶ [d0, d1, d2, d3]              UPerHead

Stage dims (ConvNeXt-Atto): [40, 80, 160, 320]
  stem:    stride=4  → H/4,  W/4,  40ch
  stage_0: stride=1  → H/4,  W/4,  40ch  (no downsample, atto 특성)
  stage_1: stride=2  → H/8,  W/8,  80ch
  stage_2: stride=2  → H/16, W/16, 160ch
  stage_3: stride=2  → H/32, W/32, 320ch

파라미터:
  ConvNeXt-Atto (stem+stages)  ~3.7M  (timm ImageNet pretrained)
  DepthBranch                  ~0.4M  (random init)
  CrossModalGating ×4          ~0.3M  (random init)
  ─────────────────────────────────────
  Total backbone               ~4.4M

Ported verbatim from
mmsegmentation/mmseg/models/backbones/dual_convnext_serial.py — only the
imports changed: DepthBranch and CrossModalGating now come from
chamnet.models.fusion. Its HD (Dual+) sibling from the same source file,
DualConvNeXtAttoPlusSerial, is ported below it. No class body was modified
during the move.
"""

try:
    import timm
except ImportError:
    timm = None

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.model import BaseModule

from mmseg.registry import MODELS

from chamnet.models.fusion import CrossModalGating, DepthBranch


@MODELS.register_module()
class DualConvNeXtAttoSerial(BaseModule):
    """Serial depth-injection dual-encoder backbone.

    Depth features are injected into the RGB backbone at every stage via
    CrossModalGating, so each stage's depth-modulated output feeds directly
    into the next stage as input.

    Args:
        fusion_reduction (int): Channel reduction ratio in CrossModalGating.
            Default: 4.

    Usage in config::

        backbone=dict(
            type='DualConvNeXtAttoSerial',
            fusion_reduction=4)
    """

    STAGE_DIMS = (40, 80, 160, 320)

    def __init__(self, fusion_reduction: int = 4, init_cfg=None):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super().__init__(init_cfg=init_cfg)

        # ── RGB stream: ConvNeXt-Atto, timm pretrained ──────────────────────
        # features_only=True로 생성하면 FeatureListNet이 되어
        # stem_0, stem_1, stages_0~3 속성으로 접근 가능
        _m = timm.create_model(
            'convnext_atto',
            features_only=True,
            pretrained=True,
            in_chans=3,
            out_indices=(0, 1, 2, 3),
        )
        # stem: Conv2d(3→40, k=4, s=4) + LayerNorm2d
        self.stem = nn.Sequential(_m.stem_0, _m.stem_1)
        # 각 stage를 ModuleList로 분리해 forward에서 루프 가능하게
        self.rgb_stages = nn.ModuleList([
            getattr(_m, f'stages_{i}') for i in range(4)
        ])
        self._is_init = True  # timm이 이미 pretrained 로드

        # ── Depth stream: lightweight DW-sep CNN ────────────────────────────
        self.depth_branch = DepthBranch(embed_dims=self.STAGE_DIMS)

        # ── Per-stage CrossModalGating ───────────────────────────────────────
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in self.STAGE_DIMS
        ])

        print_log(
            '[DualConvNeXtAttoSerial] RGB ConvNeXt-Atto: timm pretrained (serial stage loop). '
            'DepthBranch and CrossModalGating: random init. '
            f'Stage dims: {self.STAGE_DIMS}',
            logger='current')

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — BGR + depth

        Returns:
            List of 4 depth-modulated feature maps:
            [(B,40,H/4,W/4), (B,80,H/8,W/8), (B,160,H/16,W/16), (B,320,H/32,W/32)]

        Flow:
            rgb  → stem → stage_0 → CMG[0] → stage_1 → CMG[1] → ...
                                ↑               ↑
            depth → DepthBranch → [d0, d1, d2, d3]
        """
        rgb   = x[:, :3]    # (B, 3, H, W)
        depth = x[:, 3:4]   # (B, 1, H, W)

        # Depth features: 모든 scale 한번에 추출 (경량)
        depth_feats = self.depth_branch(depth)  # (d0, d1, d2, d3)

        # RGB: stem 후 stage별로 루프 → 각 stage 직후 depth gating → 다음 stage 입력
        feat = self.stem(rgb)       # (B, 40, H/4, W/4)
        outs = []
        for i, stage in enumerate(self.rgb_stages):
            feat = stage(feat)                              # ConvNeXtStage (downsampling 포함)
            feat = self.fusions[i](feat, depth_feats[i])   # depth → RGB gating (잔차 구조)
            outs.append(feat)

        return outs  # [(B,40,H/4), (B,80,H/8), (B,160,H/16), (B,320,H/32)]


# ---------------------------------------------------------------------------
# HD (Dual+, heavy depth-branch): full ConvNeXt-Atto on the depth stream
#
# Ported verbatim from the same source file as DualConvNeXtAttoSerial above,
# mmsegmentation/mmseg/models/backbones/dual_convnext_serial.py — only the
# imports changed (CrossModalGating from chamnet.models.fusion). No class body
# was modified during the move.
# ---------------------------------------------------------------------------


@MODELS.register_module()
class DualConvNeXtAttoPlusSerial(BaseModule):
    """Dual+ serial backbone: full ConvNeXt-Atto for both RGB and Depth.

    RGB stream:   ConvNeXt-Atto (timm pretrained, 3ch) — stage-by-stage serial loop
    Depth stream: ConvNeXt-Atto (random init, 1ch)     — full forward → [d0,d1,d2,d3]
    Fusion:       CrossModalGating at every stage (serial injection)

    DualConvNeXtAttoSerial 대비 차이:
      - Dual  (Serial):      DepthBranch (~0.4M DW-sep)
      - Dual+ (PlusSerial):  Full ConvNeXt-Atto (~3.7M, 1ch)

    파라미터:
      RGB ConvNeXt-Atto   ~3.7M  (timm pretrained)
      Depth ConvNeXt-Atto ~3.7M  (random init, 1ch stem)
      CrossModalGating ×4 ~0.3M  (random init)
      ─────────────────────────────────────
      Total backbone      ~7.7M
    """

    STAGE_DIMS = (40, 80, 160, 320)

    def __init__(self, fusion_reduction: int = 4, depth_pretrained: bool = False,
                 init_cfg=None):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super().__init__(init_cfg=init_cfg)
        self.depth_pretrained = depth_pretrained

        # ── RGB stream: ConvNeXt-Atto, timm pretrained, serial stage loop ───
        _rgb = timm.create_model(
            'convnext_atto',
            features_only=True,
            pretrained=True,
            in_chans=3,
            out_indices=(0, 1, 2, 3),
        )
        self.stem = nn.Sequential(_rgb.stem_0, _rgb.stem_1)
        self.rgb_stages = nn.ModuleList([
            getattr(_rgb, f'stages_{i}') for i in range(4)
        ])
        self._is_init = True  # timm이 이미 pretrained 로드

        # ── Depth stream: full ConvNeXt-Atto, 1ch ───────────────────────────
        # With depth_pretrained, timm loads the ImageNet weights and adapts the
        # 4x4 stem itself. timm SUMS the three input channels where the other
        # backbones here average them; the stem is immediately followed by a
        # LayerNorm, which normalises away a constant scale, so the two
        # conventions differ only marginally. Using timm's own adaptation
        # avoids reimplementing its checkpoint loader.
        self.depth_backbone = timm.create_model(
            'convnext_atto',
            features_only=True,
            pretrained=depth_pretrained,
            in_chans=1,
            out_indices=(0, 1, 2, 3),
        )
        if depth_pretrained:
            _n = self.depth_backbone.stem_0.weight.detach().norm().item()
            if _n == 0:
                raise RuntimeError(
                    '[DualConvNeXtAttoPlusSerial] depth_pretrained=True but the '
                    'depth stem is all zeros; the pretrained load failed.')

        # ── Per-stage CrossModalGating ───────────────────────────────────────
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in self.STAGE_DIMS
        ])

        print_log(
            '[DualConvNeXtAttoPlusSerial] RGB ConvNeXt-Atto: timm pretrained (serial stage loop). '
            f'Depth ConvNeXt-Atto (1ch): '
            f'{"timm pretrained" if depth_pretrained else "random"} init. '
            f'Stage dims: {self.STAGE_DIMS}',
            logger='current')

    def forward(self, x: torch.Tensor):
        rgb   = x[:, :3]    # (B, 3, H, W)
        depth = x[:, 3:4]   # (B, 1, H, W)

        # Depth: full ConvNeXt-Atto forward → 4 scale features
        depth_feats = self.depth_backbone(depth)  # [d0, d1, d2, d3]

        # RGB: stem → stage_i → CMG[i] → stage_i+1 → ...
        feat = self.stem(rgb)
        outs = []
        for i, stage in enumerate(self.rgb_stages):
            feat = stage(feat)
            feat = self.fusions[i](feat, depth_feats[i])
            outs.append(feat)

        return outs