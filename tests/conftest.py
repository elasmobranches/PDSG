import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope='session')
def synthetic_data(tmp_path_factory):
    """스펙 DATA_FORMAT 대로 생성한 최소 데이터셋. 실데이터 없이 코드 경로를 검증한다."""
    root = tmp_path_factory.mktemp('greenhouse')
    for split, n in [('train', 4), ('valid', 2), ('test', 2)]:
        for sub in ('images', 'masks', 'depth'):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            name = f'{split}_{i:03d}'
            rgb = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root / split / 'images' / f'{name}.jpg')
            mask = np.random.randint(0, 8, (64, 128), dtype=np.uint8)
            Image.fromarray(mask).save(root / split / 'masks' / f'{name}.png')
            depth = np.random.uniform(0.5, 8.0, (64, 128)).astype(np.float32)
            np.save(root / split / 'depth' / f'{name}.npy', depth)
    return root
