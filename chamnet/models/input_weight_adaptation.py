"""Utilities for adapting RGB pretrained input convolutions.

Widening a pretrained 3-channel input convolution to take extra geometry
planes (depth, or HHA) is the whole of what "early fusion" does to a
backbone: everything after the stem is untouched, so the only decision is
what the new input channel's filter starts as.

Two initialisations are offered and they are not interchangeable.
``'zero'`` makes the widened stem reproduce the pretrained RGB-only response
exactly at step 0 -- the depth channel contributes nothing until training
moves it. ``'mean'`` starts the depth filter at the channel-wise average of
the pretrained RGB filters, i.e. the grayscale-transfer convention, so the
depth channel is read from the first step with a filter of the right
activation scale. The paper's early-fusion runs all use ``'mean'``; the
function's own default is ``'zero'``, which is what the four backbone
classes inherit when a config says nothing, so a config that means to
reproduce those runs has to say so explicitly.

Ported verbatim from
mmsegmentation/mmseg/models/backbones/input_weight_adaptation.py. The
function body was not modified during the move; only its location changed
(chamnet keeps it beside the backbones that call it, next to
depth_pretrain.py, rather than inside mmseg's own backbones package).
"""


def expand_rgb_weight(weight, out_channels, extra_init='zero'):
    """Expand a 3-input-channel convolution while preserving RGB weights."""
    if weight.ndim != 4 or weight.shape[1] != 3:
        raise ValueError('weight must have shape (C_out, 3, K_h, K_w)')
    if out_channels < 3:
        raise ValueError('out_channels must be at least 3')
    if extra_init not in ('zero', 'mean'):
        raise ValueError("extra_init must be 'zero' or 'mean'")

    expanded = weight.new_zeros(
        weight.shape[0], out_channels, *weight.shape[2:])
    expanded[:, :3] = weight
    if out_channels > 3 and extra_init == 'mean':
        expanded[:, 3:] = weight.mean(dim=1, keepdim=True)
    return expanded
