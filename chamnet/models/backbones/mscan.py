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
chamnet.models.fusion (this package's single home for them), and MSCAN comes
from
mmseg's own backbones module instead of a sibling file in the same package.
No class body was modified during the move.

The HD (Dual+) sibling backbone, DualMSCANLateFusion, is appended below
from dual_mscan_late.py; see the section comment above it.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.mscan import MSCAN, StemConv

from chamnet.models.fusion import CrossModalGating, DepthBranch
from chamnet.models.depth_pretrain import load_rgb_into_depth_encoder


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


# ---------------------------------------------------------------------------
# HD (Dual+, heavy depth-branch): full MSCAN on the depth stream
#
# Ported verbatim from
# mmsegmentation/mmseg/models/backbones/dual_mscan_late.py — only the imports
# changed (CrossModalGating from chamnet.models.fusion,
# load_rgb_into_depth_encoder from chamnet.models.depth_pretrain, and mmseg's
# own MSCAN/StemConv). No class body was modified during the move.
# ---------------------------------------------------------------------------


@MODELS.register_module()
class DualMSCANLateFusion(MSCAN):
    """Full dual-encoder MSCAN backbone with Serial Fusion.

    Both RGB and Depth streams use the full MSCAN architecture.
    After each RGB stage, CrossModalGating injects depth features as a
    residual; the gated output feeds directly into the next stage
    (stage-by-stage serial injection).

    DualMSCAN 대비 차이: depth 스트림을 경량 DepthBranch 대신 full MSCAN(1ch)으로
    처리. 주입 메커니즘(serial CrossModalGating)은 동일.

    Usage in config::

        backbone=dict(
            type='DualMSCANLateFusion',
            embed_dims=[32, 64, 160, 256],
            mlp_ratios=[8, 8, 4, 4],
            drop_rate=0.0,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
            attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='BN', requires_grad=True),
            init_cfg=dict(type='Pretrained', checkpoint='...mscan_t....pth'),
            fusion_reduction=4)
    """

    FIRST_CONV_KEY = 'patch_embed1.proj.0.weight'

    def __init__(self, embed_dims=(32, 64, 160, 256), fusion_reduction=4,
                 depth_pretrained: bool = False, **kwargs):
        # RGB stream always 3ch
        kwargs['in_channels'] = 3
        super().__init__(embed_dims=list(embed_dims), **kwargs)

        self.depth_pretrained = depth_pretrained
        stage_dims = tuple(embed_dims)

        # Full depth MSCAN (1ch StemConv, random init)
        # Build with default 3ch then replace StemConv to accept 1ch
        depth_kwargs = dict(kwargs)
        depth_kwargs['in_channels'] = 3   # placeholder; overwritten below
        depth_kwargs['init_cfg'] = None   # no pretrained for depth
        self.depth_backbone = MSCAN(embed_dims=list(embed_dims), **depth_kwargs)

        # StemConv is hardcoded to in_channels=3 inside MSCAN.__init__
        # Replace with 1-channel version of the same architecture
        norm_cfg = kwargs.get('norm_cfg', dict(type='SyncBN', requires_grad=True))
        act_cfg  = kwargs.get('act_cfg',  dict(type='GELU'))
        self.depth_backbone.patch_embed1 = StemConv(
            in_channels=1,
            out_channels=embed_dims[0],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        # Serial CrossModalGating × 4
        self.fusions = nn.ModuleList([
            CrossModalGating(dim, reduction=fusion_reduction)
            for dim in stage_dims
        ])

        print_log(
            '[DualMSCANLateFusion] RGB MSCAN: pretrained init. '
            f'Depth MSCAN (1ch StemConv): '
            f'{"pretrained (RGB ckpt)" if depth_pretrained else "random"} init. '
            'Serial stage-by-stage injection. '
            f'Stage dims: {stage_dims}',
            logger='current')

    def init_weights(self):
        """Load pretrained MSCAN for RGB; randomly init depth + fusion."""
        if (self.init_cfg is not None
                and self.init_cfg.get('type') == 'Pretrained'):
            checkpoint_path = self.init_cfg['checkpoint']
            checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # strict=False: depth_backbone.* and fusions.* absent in checkpoint
            load_state_dict(self, state_dict, strict=False, logger='current')
            print_log(
                '[DualMSCANLateFusion] RGB MSCAN loaded from pretrained.',
                logger='current')
            if self.depth_pretrained:
                load_rgb_into_depth_encoder(
                    self.depth_backbone, state_dict, self.FIRST_CONV_KEY,
                    'DualMSCANLateFusion')
        else:
            super().init_weights()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 4, H, W) — BGR + depth concatenated by LoadDepthAsChannel
        Returns:
            List of 4 depth-injected feature maps.
        """
        rgb   = x[:, :3]    # (B, 3, H, W)
        depth = x[:, 3:4]   # (B, 1, H, W)

        # Pre-compute all depth features (full MSCAN forward, all 4 stages)
        depth_feats = self.depth_backbone(depth)  # [d0, d1, d2, d3]

        # RGB stage-by-stage serial injection (same loop as DualMSCAN)
        B = rgb.shape[0]
        outs = []
        feat = rgb

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block       = getattr(self, f'block{i + 1}')
            norm        = getattr(self, f'norm{i + 1}')

            feat, H, W = patch_embed(feat)          # NCHW → NLC
            for blk in block:
                feat = blk(feat, H, W)              # NLC → NLC
            feat = norm(feat)                        # LayerNorm
            feat = feat.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # NCHW

            # Depth-guided serial gating (residual)
            feat = self.fusions[i](feat, depth_feats[i])
            outs.append(feat)

        return outs