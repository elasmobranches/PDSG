"""Early-fusion (4-channel input) backbones: RGB + depth concatenated at the stem.

Where the dual-encoder backbones (chamnet/models/backbones/{resnet,mit,mscan,
convnext}.py) run depth through a second encoder and fuse the two streams with
CrossModalGating, early fusion does the cheapest possible thing instead: it
widens the backbone's very first convolution from 3 input channels to 4 and
feeds it the RGB-D tensor directly. Nothing else about the architecture
changes, so the whole added capacity is one extra input plane of stem filter
weights -- 288 parameters on ResNet-18, 144 on SegNeXt-T, 640 on
ConvNeXt-Atto, 1568 on MiT-B0 (out_channels x kernel_h x kernel_w in each
case). That is the point of the arm: whatever it gains over the RGB-only
baseline cannot be explained by model size.

The four classes here each subclass the stock mmseg backbone they widen and
differ only in where that first convolution lives and how their pretrained
checkpoint has to be adapted on the way in:

    ResNetV1c4Ch              stem.0                     (deep-stem 3x3)
    MixVisionTransformer4Ch   layers.0.0.projection      (7x7 patch embed)
    MSCAN4Ch                  patch_embed1.proj.0        (StemConv 3x3)
    TIMMBackbone4Ch           timm_model.stem_0          (4x4 patch stem)

All four take `extra_channel_init`, which selects what the new depth filter
starts as, and all four default it to 'zero'. The paper's runs pass 'mean' --
see chamnet/models/input_weight_adaptation.py for what the two settings mean
and chamnet/config/builder.py's `_ef_stem` for where that value is supplied.

Ported verbatim from mmsegmentation/mmseg/models/backbones/{resnet,mit,mscan,
convnext}_4ch.py, merged into one module (each was 60-90 lines). Only the
imports changed: expand_rgb_weight now comes from
chamnet.models.input_weight_adaptation, and the four base classes come from
mmseg's own backbones package instead of sibling files in the same package.
No class body was modified during the move.
"""

import torch.nn as nn
from mmengine.logging import print_log
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

from mmseg.registry import MODELS
from mmseg.models.backbones.mit import MixVisionTransformer
from mmseg.models.backbones.mscan import MSCAN
from mmseg.models.backbones.resnet import ResNetV1c
from mmseg.models.backbones.timm_backbone import TIMMBackbone

from chamnet.models.input_weight_adaptation import expand_rgb_weight


# ---------------------------------------------------------------------------
# ResNetV1c -- deep-stem 3x3 conv (stem.0), from resnet_4ch.py
#
# NOTE, and it applies to the ported class docstring below: where it says the
# depth channel is "zero-init", that describes `extra_channel_init`'s *default*
# value, not what this package emits. Every config `build_config` produces for
# this arm passes `extra_channel_init='mean'`, so channel 3 starts at the mean
# of the pretrained RGB filters, not at zeros. The class body is reproduced
# verbatim from the research fork, docstring included, so the correction lives
# here rather than in the text it corrects.
# ---------------------------------------------------------------------------

@MODELS.register_module()
class ResNetV1c4Ch(ResNetV1c):
    """ResNetV1c with 4-channel input (RGB + pseudo-depth).

    deep_stem=True 구조에서 stem의 첫 Conv2d(stem[0])를 3ch→4ch로 확장.
    Pretrained checkpoint는 RGB 3ch 기준으로 로드되며,
    depth 채널(index 3) weight는 zero-init됩니다.

    Usage in config::

        backbone=dict(
            type='ResNetV1c4Ch',
            depth=18,
            ...
            init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'))
    """

    # deep_stem 첫 번째 Conv2d의 state_dict key
    FIRST_CONV_KEY = 'stem.0.weight'

    def __init__(self, extra_channels: int = 1,
                 extra_channel_init: str = 'zero', **kwargs):
        """extra_channels: geometry planes appended after RGB. 1 = depth
        (4 channels total, the original behaviour and the default, so existing
        configs are untouched); 3 = HHA (6 total)."""
        if extra_channels < 1:
            raise ValueError('extra_channels must be at least 1')
        self.extra_channels = extra_channels
        self.extra_channel_init = extra_channel_init
        super().__init__(**kwargs)
        # ResNetV1c: deep_stem=True
        # self.stem = nn.Sequential(Conv2d, BN, ReLU, Conv2d, BN, ReLU, Conv2d, BN, ReLU)
        # stem[0] = Conv2d(in_channels, stem_channels//2, 3, 2, 1, bias=False)
        old = self.stem[0]
        n_in = 3 + self.extra_channels
        new = nn.Conv2d(
            in_channels=n_in,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        new.weight.data.copy_(expand_rgb_weight(
            old.weight.data, n_in, self.extra_channel_init))
        self.stem[0] = new

    def init_weights(self):
        if (self.init_cfg is None
                or self.init_cfg.get('type') != 'Pretrained'):
            super().init_weights()
            return

        checkpoint_path = self.init_cfg['checkpoint']
        checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # stem 첫 Conv2d 확장: (C//2, 3, 3, 3) → (C//2, 3+extra, 3, 3)
        key = self.FIRST_CONV_KEY
        if key in state_dict:
            old_w = state_dict[key]
            n_in = 3 + self.extra_channels
            state_dict[key] = expand_rgb_weight(
                old_w, n_in, self.extra_channel_init)

        load_state_dict(self, state_dict, strict=False, logger='current')
        print_log(
            f'[ResNetV1c4Ch] Loaded pretrained 3-ch weights; '
            f'{self.extra_channels} geometry channel(s) initialized with '
            f'{self.extra_channel_init!r} '
            f'(input {3 + self.extra_channels}ch).',
            logger='current',
        )


# ---------------------------------------------------------------------------
# MixVisionTransformer -- 7x7 overlap patch embed, from mit_4ch.py
#
# NOTE, and it applies to the ported class docstring below: where it says the
# depth channel is "zero-init", that describes `extra_channel_init`'s *default*
# value, not what this package emits. Every config `build_config` produces for
# this arm passes `extra_channel_init='mean'`, so channel 3 starts at the mean
# of the pretrained RGB filters, not at zeros. The class body is reproduced
# verbatim from the research fork, docstring included, so the correction lives
# here rather than in the text it corrects.
# ---------------------------------------------------------------------------

@MODELS.register_module()
class MixVisionTransformer4Ch(MixVisionTransformer):
    """MixVisionTransformer with 4-channel input (RGB + pseudo-depth).

    Always initialized with in_channels=4. When a Pretrained checkpoint is
    given, the first PatchEmbed Conv2d weight is expanded 3→4ch during load:
    - Channels 0-2 (RGB): pretrained weights
    - Channel 3 (depth): zero-initialized

    Usage in config::

        backbone=dict(
            type='MixVisionTransformer4Ch',
            in_channels=4,
            embed_dims=32,
            ...
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))
    """

    def __init__(self, extra_channel_init='zero', **kwargs):
        self.extra_channel_init = extra_channel_init
        kwargs['in_channels'] = 4  # always 4ch regardless of config
        super().__init__(**kwargs)
        proj = self.layers[0][0].projection
        proj.weight.data.copy_(expand_rgb_weight(
            proj.weight.data[:, :3].clone(), 4, self.extra_channel_init))

    def init_weights(self):
        if (self.init_cfg is None
                or self.init_cfg.get('type') != 'Pretrained'):
            super().init_weights()
            return

        checkpoint_path = self.init_cfg['checkpoint']
        checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Expand first PatchEmbed conv weight: (C_out, 3, k, k) → (C_out, 4, k, k)
        key = 'layers.0.0.projection.weight'
        if key in state_dict:
            old_w = state_dict[key]
            state_dict[key] = expand_rgb_weight(
                old_w, 4, self.extra_channel_init)

        load_state_dict(self, state_dict, strict=False, logger='current')
        print_log(
            '[MixVisionTransformer4Ch] Loaded pretrained 3-ch weights; '
            f'depth channel initialized with {self.extra_channel_init!r}.',
            logger='current',
        )


# ---------------------------------------------------------------------------
# MSCAN -- StemConv 3x3, from mscan_4ch.py
#
# NOTE, and it applies to the ported class docstring below: where it says the
# depth channel is "zero-init", that describes `extra_channel_init`'s *default*
# value, not what this package emits. Every config `build_config` produces for
# this arm passes `extra_channel_init='mean'`, so channel 3 starts at the mean
# of the pretrained RGB filters, not at zeros. The class body is reproduced
# verbatim from the research fork, docstring included, so the correction lives
# here rather than in the text it corrects.
# ---------------------------------------------------------------------------

@MODELS.register_module()
class MSCAN4Ch(MSCAN):
    """MSCAN with 4-channel input (RGB + pseudo-depth).

    StemConv(patch_embed1.proj[0])의 첫 Conv2d를 3ch→4ch로 확장.
    Pretrained checkpoint는 RGB 3ch 기준으로 로드되며,
    depth 채널(index 3) weight는 zero-init됩니다.

    Usage in config::

        backbone=dict(
            type='MSCAN4Ch',
            embed_dims=[32, 64, 160, 256],
            ...
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file))
    """

    # 첫 번째 Conv2d의 state_dict key (StemConv.proj[0])
    FIRST_CONV_KEY = 'patch_embed1.proj.0.weight'

    def __init__(self, extra_channel_init='zero', **kwargs):
        self.extra_channel_init = extra_channel_init
        super().__init__(**kwargs)
        # patch_embed1 = StemConv, proj = nn.Sequential
        # proj[0] = Conv2d(3, embed_dims[0]//2, 3, 2, 1)
        old = self.patch_embed1.proj[0]
        new = nn.Conv2d(
            in_channels=4,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        new.weight.data.copy_(expand_rgb_weight(
            old.weight.data, 4, self.extra_channel_init))
        self.patch_embed1.proj[0] = new

    def init_weights(self):
        if (self.init_cfg is None
                or self.init_cfg.get('type') != 'Pretrained'):
            super().init_weights()
            return

        checkpoint_path = self.init_cfg['checkpoint']
        checkpoint = _load_checkpoint(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # StemConv 첫 번째 Conv2d 확장: (C//2, 3, 3, 3) → (C//2, 4, 3, 3)
        key = self.FIRST_CONV_KEY
        if key in state_dict:
            old_w = state_dict[key]
            state_dict[key] = expand_rgb_weight(
                old_w, 4, self.extra_channel_init)

        load_state_dict(self, state_dict, strict=False, logger='current')
        print_log(
            '[MSCAN4Ch] Loaded pretrained 3-ch weights; '
            f'depth channel initialized with {self.extra_channel_init!r}.',
            logger='current',
        )


# ---------------------------------------------------------------------------
# TIMMBackbone -- timm patch stem, from convnext_4ch.py.
#
# Unlike the three above, this one has no init_weights(): timm loads the
# pretrained 3-channel stem itself during __init__, and the widening happens
# straight afterwards. The rationale for doing it that way is the source
# module's own, reproduced here because merging the four files into one left
# it without a module docstring to live in:
#
#   timm.create_model(..., in_chans=4, pretrained=True) would let timm's own
#   ``adapt_input_conv`` widen the stem conv automatically. That routine tiles
#   the RGB filters and rescales the *whole* tensor by 3/in_chans, so the RGB
#   weights end up scaled to 0.75x their pretrained values and the new depth
#   filter is a scaled copy of the R-channel filter -- not zero, and not equal
#   to the pretrained stem response.
#
#   Instead, the underlying timm model is built with in_channels=3 (so the
#   pretrained stem loads unmodified), and only then is the first stem Conv2d
#   widened to 4 input channels: the RGB filters keep their exact pretrained
#   weights and the depth filter is initialised by expand_rgb_weight, exactly
#   as in ResNetV1c4Ch / MixVisionTransformer4Ch / MSCAN4Ch.
#
# (The source module's wording said "zero-initialized" throughout, describing
# the shared default rather than what the paper's configs pass -- those pass
# extra_channel_init='mean'. The property that actually holds either way, and
# the one the argument above depends on, is that the RGB filters keep their
# exact pretrained values instead of being rescaled to 0.75x.)
# ---------------------------------------------------------------------------

@MODELS.register_module()
class TIMMBackbone4Ch(TIMMBackbone):
    """timm backbone with a 4-channel (RGB + depth) stem.

    Args:
        stem_conv_attr (str): Attribute name of the first stem Conv2d on
            ``self.timm_model``. For ``features_only=True`` timm models the
            stem's ``nn.Sequential`` is flattened, so the first conv is
            typically exposed as ``'stem_0'`` (e.g. ConvNeXt). Defaults to
            ``'stem_0'``.

    Usage in config::

        backbone=dict(
            type='TIMMBackbone4Ch',
            model_name='convnext_atto',
            features_only=True,
            pretrained=True,
            out_indices=(0, 1, 2, 3))
    """

    def __init__(self, stem_conv_attr='stem_0',
                 extra_channel_init='zero', **kwargs):
        kwargs['in_channels'] = 3
        super().__init__(**kwargs)

        old = getattr(self.timm_model, stem_conv_attr)
        new = nn.Conv2d(
            in_channels=4,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        new.weight.data.copy_(expand_rgb_weight(
            old.weight.data, 4, extra_channel_init))
        if old.bias is not None:
            new.bias.data = old.bias.data.clone()
        setattr(self.timm_model, stem_conv_attr, new)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
