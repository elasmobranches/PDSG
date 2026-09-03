"""Initialise a 1-channel depth encoder from an RGB-pretrained checkpoint.

Only ResNet's late-fusion class offered this; MSCAN, MiT and ConvNeXt always
started their depth encoder from scratch. That made the backbone comparison
unfair whenever ResNet ran with ``depth_pretrained=True``: HD's depth encoder
is a full backbone copy, so pretrained-versus-random is a far larger
difference than the fusion design being studied.

Whether pretraining actually helps is a separate, open question. On the
greenhouse data it does not -- ResNet-18 HD scores 82.44 Pillar IoU with a
random depth encoder and 79.29 with an ImageNet one (-3.14, t=-4.84, n=5),
while mIoU is unchanged. The point of this module is to make the choice a
controlled variable rather than an accident of which class you instantiate.

Transfer rule: every layer is copied verbatim; only the first convolution is
adapted, and only when the depth encoder takes one channel -- by averaging its
three input channels into one. A 3-channel depth-slot encoder (the HD-RGB
capacity control) gets the filters copied straight across, no adaptation. Averaging (rather
than summing) preserves activation scale for a single-channel input, which is
the usual RGB-to-grayscale transfer. It matches what
``DualResNetV1c18LateFusion._load_pretrained_depth`` already does.

Ported verbatim from
mmsegmentation/mmseg/models/backbones/depth_pretrain.py. Nothing in the
module body was modified during the move; only its location changed
(chamnet keeps it beside the backbones that call it rather than inside
mmseg's own backbones package).
"""
import torch.nn as nn
from mmengine.logging import print_log

__all__ = ['load_rgb_into_depth_encoder']


def _get_by_path(module, dotted):
    obj = module
    for part in dotted.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def load_rgb_into_depth_encoder(depth_backbone, state_dict, first_conv_key, tag):
    """Copy ``state_dict`` into ``depth_backbone``, adapting the first conv.

    Args:
        depth_backbone: the 1-channel encoder to initialise.
        state_dict: the RGB checkpoint's state dict.
        first_conv_key: key of the first convolution's weight, e.g.
            ``'patch_embed1.proj.0.weight'``. Its module path is derived by
            dropping the trailing ``'.weight'``.
        tag: class name, for log and error messages.

    Raises:
        RuntimeError: if the load looks like a no-op. A silent failure here
            leaves the arm training on random weights while the config and the
            log both claim otherwise, which is the failure mode this guard
            exists to prevent.
    """
    conv_path = first_conv_key[:-len('.weight')]
    try:
        conv = _get_by_path(depth_backbone, conv_path)
    except (AttributeError, IndexError, KeyError) as e:
        raise RuntimeError(
            f'[{tag}] depth_pretrained=True but the first conv path '
            f'{conv_path!r} does not exist on the depth encoder: {e}')
    if not isinstance(conv, nn.Conv2d):
        raise RuntimeError(
            f'[{tag}] {conv_path!r} is {type(conv).__name__}, not Conv2d')
    if conv.in_channels not in (1, 3):
        raise RuntimeError(
            f'[{tag}] depth encoder first conv takes {conv.in_channels} '
            f'channels, expected 1 or 3')

    # 1ch (raw metric depth): average the three RGB filters, preserving scale.
    # 3ch (HD-RGB capacity control): copy straight across, no adaptation.
    adapted = {}
    for k, v in state_dict.items():
        if k == first_conv_key and hasattr(v, 'dim') and v.dim() == 4 \
                and v.shape[1] == 3 and conv.in_channels == 1:
            v = v.mean(dim=1, keepdim=True)      # (out,3,k,k) -> (out,1,k,k)
        adapted[k] = v

    before = conv.weight.detach().norm().item()
    missing, unexpected = depth_backbone.load_state_dict(adapted, strict=False)
    after = conv.weight.detach().norm().item()
    matched = len(state_dict) - len(unexpected)

    if before == after or matched < 20:
        raise RuntimeError(
            f'[{tag}] depth_pretrained=True but the load looks like a no-op '
            f'({conv_path} norm {before:.4f} -> {after:.4f}, matched '
            f'{matched}/{len(state_dict)} keys, missing={len(missing)}, '
            f'unexpected={list(unexpected)[:5]}). Refusing to silently train '
            f'on random-init depth weights.')

    how = ('3ch->1ch channel-averaged' if conv.in_channels == 1
           else '3ch copied directly')
    print_log(
        f'[{tag}] Depth encoder: pretrained init from the RGB checkpoint '
        f'({conv_path} {how}). matched={matched}/'
        f'{len(state_dict)} keys, norm {before:.4f} -> {after:.4f}.',
        logger='current')
    return matched
