"""Online RGB/depth/mask augmentation for ChamNet training."""

from __future__ import annotations

import cv2
import albumentations as A
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ChamNetOnlineAugmentation(BaseTransform):
    """Apply augmentation.py-style geometry to image/depth/mask together.

    Geometry sees the complete RGB+D image, while photometric transforms are
    applied only to RGB channels so depth values are never corrupted.

    Rotation is disabled by default since 2026-08-18: ``A.Rotate`` pads the
    rotated border with a constant, which injects synthetic 0 m depth and
    background label along image edges.  Set ``rotate_prob`` above zero to
    restore the previous ``limit=15, p=0.3`` behaviour.

    Args:
        rotate_limit (int): Maximum absolute rotation in degrees.
            Defaults to ``15``.
        rotate_prob (float): Probability of applying rotation.  ``0.0``
            removes the transform from the geometry pipeline entirely.
            Defaults to ``0.0``.
        photometric (str): which photometric stack to apply.

            ``'full'`` is the six-transform policy every v12 run used and is
            the default, so existing configs are unaffected.

            ``'light'`` keeps only brightness/contrast and gamma. Those are the
            two that fire most often (p=0.6 and p=0.5 against 0.2-0.3 for the
            rest) and the two that correspond to something real in a
            greenhouse, where illumination genuinely varies through the day.
            The four dropped -- JPEG compression, motion blur, saturation and
            colour jitter -- simulate sensor artefacts instead. This is the
            same pair the module-level ``strong_photometric_transform`` in
            ``tools/online_augmentation.py`` has always described.

            Photometry is not removed entirely: with 318 training frames it is
            doing real work against overfitting.

            ``'brightness_contrast'`` keeps only the RGB brightness/contrast
            operation (p=0.6). It does not alter depth or masks.
    """

    def __init__(self, rotate_limit: int = 15, rotate_prob: float = 0.0,
                 photometric: str = 'full'):
        assert photometric in ('full', 'light', 'brightness_contrast'), (
            'photometric must be full, light, or brightness_contrast, '
            f'got {photometric!r}')
        self.rotate_limit = rotate_limit
        self.rotate_prob = rotate_prob
        self.photometric_policy = photometric
        geometry = [A.HorizontalFlip(p=0.5)]
        if rotate_prob > 0:
            geometry.append(
                A.Rotate(
                    limit=rotate_limit,
                    p=rotate_prob,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                ))
        self.geometry = A.Compose(geometry)
        photometric_ops = [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.45, 0.45),
                contrast_limit=(-0.20, 0.20),
                p=0.6,
            ),
        ]
        if photometric in ('full', 'light'):
            photometric_ops.append(
                A.RandomGamma(gamma_limit=(60, 150), p=0.5))
        if photometric == 'full':
            photometric_ops += [
                A.ImageCompression(quality_range=(80, 100), p=0.3),
                A.MotionBlur(blur_limit=5, p=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=0, sat_shift_limit=20, val_shift_limit=15,
                    p=0.2,
                ),
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0,
                    p=0.2,
                ),
            ]
        self.photometric = A.Compose(photometric_ops)

    def transform(self, results: dict) -> dict:
        transformed = self.geometry(
            image=results['img'],
            mask=results['gt_seg_map'],
        )
        image = transformed['image']
        results['gt_seg_map'] = transformed['mask'].astype(
            results['gt_seg_map'].dtype, copy=False)

        rgb_source = image[:, :, :3]
        # Appending float32 depth promotes RGB from uint8 to float32 while its
        # numeric range remains [0, 255]. Albumentations interprets float
        # images as [0, 1], so feed photometric transforms an explicit uint8
        # RGB view and cast back afterwards.
        if np.issubdtype(rgb_source.dtype, np.floating):
            rgb_input = np.clip(rgb_source, 0, 255).astype(np.uint8)
        else:
            rgb_input = rgb_source
        rgb = self.photometric(image=rgb_input)['image']
        image = image.copy()
        image[:, :, :3] = rgb.astype(image.dtype, copy=False)
        results['img'] = image
        results['img_shape'] = image.shape[:2]
        return results
