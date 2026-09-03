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
defined in that file) now come from chamnet.models.fusion (Task 7 moved
them there — this was their one and only defining module, SD/HD's other
backbones already imported them from here too), and MixVisionTransformer
comes from mmseg's own backbones module instead of a sibling file in the
same package. JointCMGFusion/BidirectionalCMGFusion, an unrelated
experimental fusion module also defined in the source file but never used
by DualMiTB0 (or any of the four SD backbones), was not ported. No class
body was modified during the move.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.utils import nlc_to_nchw

from chamnet.models.fusion import CrossModalGating, DepthBranch


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
