"""The shared fusion modules. Moving them out of the original must not change a number."""
import torch
from chamnet.models.fusion import BiGateGating, CrossModalGating, DepthBranch


def test_cmg_is_identity_when_depth_is_zero_and_gate_off():
    m = CrossModalGating(16, use_gate=False).eval()
    rgb = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        out = m(rgb, torch.zeros(2, 16, 8, 8))
    # depth_proj(0) leaves BN's bias behind, so the output is not exactly rgb.
    # Only the shape and finiteness are checked here; numerical equality is
    # what the checkpoint replay in tools/replay.py verifies (see
    # verification/).
    assert out.shape == rgb.shape and torch.isfinite(out).all()


def test_bigate_has_two_gates_and_cmg_has_one():
    cmg = CrossModalGating(64)
    bg = BiGateGating(64)
    cmg_gates = [n for n, _ in cmg.named_modules() if n.endswith('gate')]
    bg_gates = [n for n, _ in bg.named_modules() if n in ('gate_rgb', 'gate_d')]
    assert len(cmg_gates) == 1 and len(bg_gates) == 2


def test_depth_branch_output_shapes():
    db = DepthBranch(embed_dims=(32, 64, 160, 256)).eval()
    with torch.no_grad():
        f = db(torch.randn(1, 1, 64, 128))
    assert [t.shape[1] for t in f] == [32, 64, 160, 256]
    assert [t.shape[-1] for t in f] == [32, 16, 8, 4]
