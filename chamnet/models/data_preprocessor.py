# Copyright (c) OpenMMLab. All rights reserved.
"""ChamNetSegDataPreProcessor — fixes a >3-channel bgr_to_rgb no-op.

Ported verbatim (docstring, body, everything) from the research fork the
paper's checkpoints were trained in, which patches vanilla mmseg's own
mmseg/models/data_preprocessor.py in place. The two differ in exactly
one place, called out in the comment inside `forward` below: vanilla mmseg
checks ``inputs[0].size(0) == 3`` before doing the BGR->RGB channel swap, so
bgr_to_rgb silently becomes a no-op for any 4+ channel input (sd/hd/ef's
RGB+D tensors). Depth then rides through the pipeline in a model expecting
RGB-ordered colour channels while receiving BGR ones — exactly the class of
bug this file's own comment traces to "runs 1-25 of the paper-v8 campaign".

This release imports mmsegmentation as a normal pip dependency (see tests/
and README), not as a modified fork, so the same fix can't be made by
editing mmseg's installed file in place the way the original repo did.
Registered here under its own name, `ChamNetSegDataPreProcessor`, rather
than overriding vanilla's `SegDataPreProcessor` in place:
`chamnet.register_all()` is documented (tests/test_smoke.py) to stay
side-effect-free for callers, and silently swapping out a registry entry
whose type name a caller might reasonably expect to mean mmseg's own class
is exactly that kind of side effect — worse, an exported config that says
`SegDataPreProcessor` but actually means a different class is the one thing
a reproducibility release cannot ship (a config replayed outside chamnet
would silently resolve to the broken vanilla class instead of erroring).
`build_config` therefore writes `type='ChamNetSegDataPreProcessor'`
for every method, 3-channel bl included (verified behaviour-identical to
vanilla for exactly 3 channels — tests/test_data_preprocessor.py), so a
config that names this class always means this class, a config missing it
fails loudly (`KeyError`) rather than quietly losing the fix, and there is
only ever one preprocessor class in play for chamnet's own configs.

Verified in tools/replay.py's checkpoint replay: without this fix, every
SD backbone's `chamoe` class IoU collapses from ~65-80 to ~0 on the real
test set (BGR<->RGB swapped, its yellow signature reads as cyan), while
mIoU for BL (3-channel, unaffected by the >=3 vs ==3 distinction) stays
correct either way.
"""
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class ChamNetSegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) >= 3:
            # Convert the three colour channels and pass any extra channels
            # through untouched. Extra channels are geometry stacked on by the
            # loaders -- depth (1) or HHA (3) -- and must not be permuted.
            #
            # This was previously written as separate `== 3` and `== 4` cases.
            # Anything wider fell through both, so bgr_to_rgb became a silent
            # no-op and the colour channels stayed BGR while mean/std were
            # RGB-ordered. That is the same defect that contaminated runs 1-25
            # of the paper-v8 campaign at 4 channels; fixing only the 4-channel
            # case let it return the moment a 6-channel RGB+HHA input appeared.
            # Written generally so the next channel count cannot repeat it.
            inputs = [
                torch.cat([_input[[2, 1, 0], ...], _input[3:, ...]], dim=0)
                for _input in inputs
            ]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)
