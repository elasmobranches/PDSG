"""Dual-encoder backbone for RGB + Depth semantic segmentation (SegNeXt/MSCAN).

Architecture overview
---------------------
  RGB (3ch)   ──▶  MSCAN (pretrained, 4 stages)          ─┐
                     stage i ──▶ CrossModalGating[i] ──▶    ├──▶ fused features ──▶ LightHamHead
  Depth (1ch) ──▶  DepthBranch (4-stage DW-sep CNN)       ─┘

Channel/spatial dimensions (MSCAN-T, embed_dims=[32,64,160,256]):
  Stage 0: StemConv (stride=4)            →  32ch, H/4
  Stage 1: OverlapPatchEmbed (stride=2)   →  64ch, H/8
  Stage 2: OverlapPatchEmbed (stride=2)   → 160ch, H/16
  Stage 3: OverlapPatchEmbed (stride=2)   → 256ch, H/32

DepthBranch matches these resolutions using the same DW-sep pattern as DualMiTB0.
CrossModalGating: identical to DualMiTB0 (reused from dual_mit.py).

Ported verbatim from mmsegmentation/mmseg/models/backbones/dual_mscan.py —
only the imports changed: DepthBranch and CrossModalGating now come from
chamnet.models.fusion (Task 7 moved them there), and MSCAN comes from
mmseg's own backbones module instead of a sibling file in the same package.
No class body was modified during the move.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.mscan import MSCAN

from chamnet.models.fusion import CrossModalGating, DepthBranch


@MODELS.register_module()
class DualMSCAN(MSCAN):
    """Dual-encoder MSCAN backbone: MSCAN (RGB) + DepthBranch + CrossModalGating.

    Input tensor: (B, 4, H, W) — channels 0:3 = BGR, channel 3 = depth.

    The RGB stream runs through standard MSCAN stages.
    After each stage output (NCHW), CrossModalGating injects depth-derived
    cues as a residual, enriching RGB features without modifying pretrained weights.

    Usage in config::

        backbone=dict(
            type='DualMSCAN',
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='BN', requires_grad=True),
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
            fusion_reduction=4)
    """

    def __init__(self, embed_dims=[32, 64, 160, 256], fusion_reduction=4, **kwargs):
        # RGB stream always 3ch — depth handled by DepthBranch
        kwargs['in_channels'] = 3
        super().__init__(embed_dims=embed_dims, **kwargs)

        stage_dims = tuple(embed_dims)
        self.depth_branch = DepthBranch(embed_dims=stage_dims)
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in stage_dims
        ])

    def init_weights(self):
        """Load pretrained MSCAN for RGB stream; randomly init depth modules."""
        if (self.init_cfg is not None
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint_path = self.init_cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # strict=False: depth_branch.* and fusions.* absent in checkpoint
            # → these keys are skipped and stay randomly initialised.
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualMSCAN] RGB encoder (MSCAN) loaded from pretrained. '
                'DepthBranch and CrossModalGating modules: random init.',
                logger='current')
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — BGR + depth concatenated by LoadDepthAsChannel
        Returns:
            List of 4 fused feature maps at decreasing spatial resolutions.
        """
        rgb = x[:, :3]     # (B, 3, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)

        # Pre-compute all depth features (single forward pass)
        depth_feats = self.depth_branch(depth)  # tuple of 4

        B = rgb.shape[0]
        outs = []
        feat = rgb  # starts as NCHW image

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            feat, H, W = patch_embed(feat)   # NCHW → NLC
            for blk in block:
                feat = blk(feat, H, W)       # NLC → NLC
            feat = norm(feat)                 # LayerNorm on NLC
            feat = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # NCHW

            # Depth-guided cross-modal gating (residual addition)
            feat = self.fusions[i](feat, depth_feats[i])
            outs.append(feat)

        return outs
