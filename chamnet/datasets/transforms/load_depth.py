"""Transform for loading pseudo-depth as a 4th image channel (RGB+D)."""
import os.path as osp
from typing import Optional

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDepthAsChannel(BaseTransform):
    """Load a pseudo-depth map and append it as the 4th channel of the image.

    The depth channel is converted to float32 without changing its unit.
    Set ``max_depth=None`` to preserve the complete pseudo-depth range, or
    provide an explicit upper bound when a calibrated task cutoff is intended.
    Standardisation happens downstream in ``SegDataPreProcessor``.

    History: earlier revisions rescaled depth to [0, 255] before
    standardisation.  Because a fixed affine map followed by dataset-level
    standardisation is equivalent to standardising the metric values
    directly (verified to ±1e-6 in tests/test_depth_metric_refactor.py),
    removing the rescale changes nothing numerically — it only removes an
    arbitrary intermediate convention.

    Depth path is derived from the RGB image path by replacing the images
    sub-directory name with the depth sub-directory name and swapping the
    file extension.

    Args:
        depth_dir_name (str): Sub-directory holding depth files.
            Defaults to ``'depth'``.
        img_dir_name (str): Sub-directory of RGB images, used to derive the
            depth path.  Defaults to ``'images'``.
        depth_suffix (str): File suffix appended after the image stem.
            e.g. ``'.npy'`` (train) or ``'_depth.npy'`` (val/test).
            Defaults to ``'.npy'``.
        max_depth (float, optional): Explicit clip maximum. ``None`` disables
            clipping and preserves finite raw values. Defaults to ``10000.0``
            for backward compatibility with v8 configs.
    """

    def __init__(
        self,
        depth_dir_name: str = 'depth',
        img_dir_name: str = 'images',
        depth_suffix: str = '.npy',
        max_depth: Optional[float] = 10000.0,
    ):
        self.depth_dir_name = depth_dir_name
        self.img_dir_name = img_dir_name
        self.depth_suffix = depth_suffix
        self.max_depth = max_depth

    def _derive_depth_path(self, img_path: str) -> str:
        depth_path = img_path.replace(
            osp.sep + self.img_dir_name + osp.sep,
            osp.sep + self.depth_dir_name + osp.sep,
        )
        return osp.splitext(depth_path)[0] + self.depth_suffix

    def _load_depth(self, depth_path: str) -> np.ndarray:
        if depth_path.endswith('.npy'):
            raw = np.load(depth_path)
        else:
            raw = mmcv.imread(depth_path, flag='unchanged')
            if raw is None:
                raise FileNotFoundError(
                    f'[LoadDepthAsChannel] File not found: {depth_path}'
                )
        return raw

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Return finite float32 depth, optionally using the legacy clip."""
        raw = raw.astype(np.float32)
        if not np.isfinite(raw).all():
            raise ValueError('[LoadDepthAsChannel] depth contains non-finite values')
        if self.max_depth is None:
            return raw
        return np.clip(raw, 0.0, self.max_depth)

    def transform(self, results: dict) -> dict:
        depth_path = self._derive_depth_path(results['img_path'])
        raw = self._load_depth(depth_path)
        depth_mm = self._normalize(raw)

        img = results['img']  # (H, W, 3)

        if depth_mm.shape[:2] != img.shape[:2]:
            depth_mm = cv2.resize(
                depth_mm,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        depth_ch = depth_mm[:, :, np.newaxis]  # (H, W, 1)
        results['img'] = np.concatenate([img, depth_ch], axis=2)  # (H, W, 4)
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'depth_dir_name={self.depth_dir_name!r}, '
            f'img_dir_name={self.img_dir_name!r}, '
            f'depth_suffix={self.depth_suffix!r}, '
            f'max_depth={self.max_depth!r})'
        )


@TRANSFORMS.register_module()
class ShuffleDepthChannel(BaseTransform):
    """Depth 채널(4번째)의 픽셀 순서를 이미지별로 랜덤 셔플.

    통계(평균, 분산, 히스토그램)는 원본과 완전히 동일하게 유지하면서
    공간 구조(기하학적 정보)만 제거합니다.

    LoadDepthAsChannel 이후에 pipeline에 삽입하여 사용합니다::

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadDepthAsChannel', ...),
            dict(type='ShuffleDepthChannel'),   # ← 여기
            ...
        ]

    ablation 목적: depth 공간 구조의 기여도 측정.
    학습·검증·테스트 모두 동일하게 적용해야 합니다.
    """

    def transform(self, results: dict) -> dict:
        img = results['img']          # (H, W, 4)
        h, w = img.shape[:2]
        depth = img[:, :, 3].copy()  # (H, W)

        perm = np.random.permutation(h * w)
        img[:, :, 3] = depth.ravel()[perm].reshape(h, w)

        results['img'] = img
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
