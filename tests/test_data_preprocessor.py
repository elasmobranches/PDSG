"""chamnet.register_all() 이 ChamNetSegDataPreProcessor 를 실제로 등록하는지,
그리고 4채널 이상 입력에서 BGR<->RGB 스왑이 실제로 일어나는지 확인한다.

배경: 논문 재생(tools/replay.py) 에서 vanilla mmseg 의 SegDataPreProcessor 는
`inputs[0].size(0) == 3` 일 때만 채널을 뒤집어, RGB+D(4채널) 입력에서는
bgr_to_rgb 가 조용히 무시된다 — SD 체크포인트 재생에서 chamoe 클래스 IoU 가
~65%에서 0으로 무너지는 것으로 실측 확인됨. chamnet 은 vanilla 클래스를 덮어쓰지
않고 별도 이름(ChamNetSegDataPreProcessor)으로 등록한다 — 이름을 바꿔치기하면
`chamnet.register_all()` 이 "호출자에게 부작용이 없어야 한다"는 규칙(Task 6,
test_smoke.py 참고)을 깨고, 내보낸 config 의 `type='SegDataPreProcessor'` 가
실제로는 다른 클래스를 뜻하게 되어 재현성 릴리스가 감당할 수 없는 모호함이
생긴다.
"""
import subprocess
import sys
from pathlib import Path

import torch

from chamnet.models.data_preprocessor import ChamNetSegDataPreProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_register_all_registers_chamnet_seg_data_preprocessor():
    """등록이 register_all() 자체에서 나오는지 검증한다 — 다른 top-level import
    가 먼저 실행돼 우연히 등록된 것이 아니라.

    이 테스트 파일 안에서 `from chamnet.models.data_preprocessor import
    ChamNetSegDataPreProcessor` 를 이미 썼으므로, 같은 프로세스에서
    MODELS.get(...) 를 확인하면 그 top-level import 가 이미
    @MODELS.register_module() 를 실행해놓은 뒤라 register_all() 이 한 일인지
    알 수 없다 (register_all() 을 호출하지 않아도 통과함 — 장식적 검증). 이
    테스트가 실제로 무언가를 검증하게 하려면, 이 모듈을 아직 아무도 import
    하지 않은 새 인터프리터에서 register_all() 만 호출해 확인해야 한다.

    깨뜨려서 검증함: chamnet/__init__.py 의 register_all() 에서
    `from chamnet.models import data_preprocessor` 줄을 지우고 재실행하면
    이 테스트가 실패한다 (등록되지 않음) — 지우기 전에는 통과.
    """
    script = (
        'import chamnet\n'
        'from mmseg.registry import MODELS\n'
        "assert 'ChamNetSegDataPreProcessor' not in MODELS.module_dict, "
        "'이미 등록됨 -- 격리 실패'\n"
        'chamnet.register_all()\n'
        "assert 'ChamNetSegDataPreProcessor' in MODELS.module_dict, "
        "'register_all() 이 등록하지 않음'\n"
        "print('OK')\n"
    )
    # cwd=REPO_ROOT: without it, `import chamnet` in the child only
    # resolves if chamnet happens to be pip-installed or the suite is run
    # from the repo root by coincidence -- pin it so the test doesn't
    # depend on the caller's working directory.
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == 'OK'


def test_bgr_to_rgb_swaps_first_three_channels_and_passes_the_rest():
    """4채널(RGB+D) 입력에서도 첫 3채널만 뒤집히고 depth 는 그대로여야 한다."""
    pre = ChamNetSegDataPreProcessor(mean=[0.0] * 4, std=[1.0] * 4, bgr_to_rgb=True)
    h, w = 4, 4
    img = torch.zeros(4, h, w)
    img[0], img[1], img[2], img[3] = 1.0, 2.0, 3.0, 99.0  # (B, G, R, depth)

    out = pre({'inputs': [img], 'data_samples': None}, training=False)
    result = out['inputs'][0]

    assert torch.equal(result[0], torch.full((h, w), 3.0))   # R moved to slot 0
    assert torch.equal(result[1], torch.full((h, w), 2.0))   # G unchanged
    assert torch.equal(result[2], torch.full((h, w), 1.0))   # B moved to slot 2
    assert torch.equal(result[3], torch.full((h, w), 99.0))  # depth untouched


def test_three_channel_behaviour_is_unchanged():
    """3채널(BL) 입력에서는 vanilla 와 동일하게 동작해야 한다 — 회귀 아님."""
    pre = ChamNetSegDataPreProcessor(mean=[0.0] * 3, std=[1.0] * 3, bgr_to_rgb=True)
    h, w = 4, 4
    img = torch.zeros(3, h, w)
    img[0], img[1], img[2] = 1.0, 2.0, 3.0  # (B, G, R)

    out = pre({'inputs': [img], 'data_samples': None}, training=False)
    result = out['inputs'][0]

    assert torch.equal(result[0], torch.full((h, w), 3.0))
    assert torch.equal(result[1], torch.full((h, w), 2.0))
    assert torch.equal(result[2], torch.full((h, w), 1.0))
