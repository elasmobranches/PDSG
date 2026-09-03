"""Two things: that chamnet.register_all() really does register
ChamNetSegDataPreProcessor, and that the BGR<->RGB swap really does happen on
inputs with more than three channels.

Background: while replaying the published checkpoints (tools/replay.py),
vanilla mmseg's SegDataPreProcessor was found to swap channels only when
`inputs[0].size(0) == 3`, so on RGB+D (4-channel) input bgr_to_rgb silently
does nothing -- measured, in the SD checkpoint replay, as the `chamoe` class
IoU collapsing from ~65% to 0. chamnet registers its corrected copy under its
own name (ChamNetSegDataPreProcessor) rather than overriding vanilla's. Taking
the name over would break the rule that `chamnet.register_all()` has no side
effects for a caller (see test_smoke.py), and would leave an exported config
whose `type='SegDataPreProcessor'` actually meant a different class -- an
ambiguity a reproducibility release cannot carry.
"""
import subprocess
import sys
from pathlib import Path

import torch

from chamnet.models.data_preprocessor import ChamNetSegDataPreProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_register_all_registers_chamnet_seg_data_preprocessor():
    """The registration must come from register_all() itself, not from some
    other top-level import having happened to run first.

    This file already does `from chamnet.models.data_preprocessor import
    ChamNetSegDataPreProcessor` at the top, so checking MODELS.get(...) in the
    same process says nothing: that import has already executed
    @MODELS.register_module(), and the check would pass even without calling
    register_all() at all -- a decorative assertion. For it to verify
    anything, register_all() has to be the only thing called, in a fresh
    interpreter where nothing has imported the module yet.

    Verified by breaking it: delete the `from chamnet.models import
    data_preprocessor` line from register_all() in chamnet/__init__.py and
    this test fails (nothing registered); it passes before the deletion.
    """
    script = (
        'import chamnet\n'
        'from mmseg.registry import MODELS\n'
        "assert 'ChamNetSegDataPreProcessor' not in MODELS.module_dict, "
        "'already registered -- the isolation failed'\n"
        'chamnet.register_all()\n'
        "assert 'ChamNetSegDataPreProcessor' in MODELS.module_dict, "
        "'register_all() did not register it'\n"
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
    """On 4-channel (RGB+D) input only the first three channels swap; depth passes through."""
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
    """On 3-channel (BL) input the behaviour must match vanilla's exactly -- no regression."""
    pre = ChamNetSegDataPreProcessor(mean=[0.0] * 3, std=[1.0] * 3, bgr_to_rgb=True)
    h, w = 4, 4
    img = torch.zeros(3, h, w)
    img[0], img[1], img[2] = 1.0, 2.0, 3.0  # (B, G, R)

    out = pre({'inputs': [img], 'data_samples': None}, training=False)
    result = out['inputs'][0]

    assert torch.equal(result[0], torch.full((h, w), 3.0))
    assert torch.equal(result[1], torch.full((h, w), 2.0))
    assert torch.equal(result[2], torch.full((h, w), 1.0))
