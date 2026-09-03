import numpy as np
import pytest
import chamnet
from mmseg.registry import TRANSFORMS

chamnet.register_all()

CLASSES = ('background', 'chamoe', 'heatpipe', 'path',
           'pillar', 'topdownfarm', 'ceiling', 'duct')


def test_dataset_metainfo():
    from mmseg.registry import DATASETS
    meta = DATASETS.get('ChamNet').METAINFO
    assert meta['classes'] == CLASSES
    assert len(meta['palette']) == 8
    assert meta['palette'][4] == [0, 0, 255]        # pillar = blue


def _rgbd(h=32, w=64):
    rgb = np.random.randint(0, 256, (h, w, 3)).astype(np.float32)
    depth = np.random.uniform(0.5, 8.0, (h, w, 1)).astype(np.float32)
    return np.concatenate([rgb, depth], axis=2)


def test_shuffle_preserves_depth_values_and_leaves_rgb_alone():
    t = TRANSFORMS.build(dict(type='ShuffleDepthChannel'))
    img = _rgbd()
    out = t.transform(dict(img=img.copy()))['img']
    assert np.array_equal(np.sort(out[:, :, 3].ravel()),
                          np.sort(img[:, :, 3].ravel()))   # 값 다중집합 보존
    assert not np.array_equal(out[:, :, 3], img[:, :, 3])  # 배치는 파괴
    assert np.array_equal(out[:, :, :3], img[:, :, :3])    # RGB 불변


def test_photometric_never_touches_depth():
    t = TRANSFORMS.build(dict(type='ChamNetOnlineAugmentation',
                              photometric='brightness_contrast', rotate_prob=0.0))
    img = _rgbd()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for _ in range(200):
        out = t.transform(dict(img=img.copy(), gt_seg_map=mask.copy()))['img']
        assert np.array_equal(np.sort(out[:, :, 3].ravel()),
                              np.sort(img[:, :, 3].ravel()))
