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
chamnet.models.fusion (Task 7 moved them there out of dual_mit.py), and
ResNetV1c comes from mmseg's own backbones module instead of a sibling file
in the same package. No class body was modified during the move.
"""

import torch
import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.resnet import ResNetV1c

from chamnet.models.fusion import CrossModalGating, _DWBlock


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
