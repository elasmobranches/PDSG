"""빌더 위의 얇은 껍데기. 로직은 전부 chamnet.config.builder 에 있다."""
from __future__ import annotations

import argparse
import sys

import chamnet
from chamnet.config.backbones import BACKBONES
from chamnet.config.builder import build_config
from chamnet.config.combos import VALID

# Derived from the combination table rather than restated, so `chamnet list`
# and `--method` cannot advertise a method the builder no longer serves.
METHODS = sorted({m for m, _ in VALID})

# ABLATIONS stays hand-written on purpose: it is deliberately a *superset* of
# what is currently valid, so `--ablation nogate` reaches combos.validate() and
# gets its explanatory message instead of argparse's bare "invalid choice".
# `list` filters it through VALID (and appends anything VALID knows about that
# this vocabulary doesn't), so nothing dead is printed and nothing live is
# omitted.
ABLATIONS = ['shuffled', 'rgb', 'nogate', 'bigate']


def _common(sp):
    sp.add_argument('--method', required=True, choices=METHODS)
    sp.add_argument('--backbone', required=True, choices=sorted(BACKBONES))
    sp.add_argument('--ablation', default=None, choices=ABLATIONS)
    sp.add_argument('--recipe', default='paper')
    sp.add_argument('--data', default=None, help='데이터 루트. 레시피 값을 덮어쓴다')
    sp.add_argument('--seed', type=int, default=31)


def _names(spec: str) -> list[str]:
    """Split a comma-separated CLI list, tolerating spaces after the commas."""
    return [name.strip() for name in spec.split(',') if name.strip()]


def _cfg(a):
    return build_config(method=a.method, backbone=a.backbone, ablation=a.ablation,
                        recipe=a.recipe, data_root=a.data, seed=a.seed,
                        work_dir=getattr(a, 'out', None))


def main(argv: list[str] | None = None) -> int:
    chamnet.register_all()
    ap = argparse.ArgumentParser(prog='chamnet')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('train'); _common(p); p.add_argument('--out', default=None)
    p = sub.add_parser('test');  _common(p)
    p.add_argument('--checkpoint', required=True,
                   help='checkpoint to score. Loading it executes pickled '
                        'code, so load only checkpoints you trust: reading a '
                        'checkpoint written by mmengine needs a wider '
                        'unpickler than torch enables by default, and this '
                        'command turns that default off for the load.')
    p = sub.add_parser('export-config'); _common(p)
    p.add_argument('-o', '--out', required=True)
    p = sub.add_parser('sweep')
    p.add_argument('--recipe', default='paper')
    p.add_argument('--methods', default=','.join(METHODS))
    p.add_argument('--backbones', default=','.join(sorted(BACKBONES)))
    # One ablation per sweep, applied to every method named in --methods. The
    # control arms exist on one or two methods each, so `--methods hd
    # --ablation nogate` is how you ask for one; mixing an ablation with a
    # method that has no such arm is refused up front rather than after the
    # earlier combinations have spent hours on the GPU.
    p.add_argument('--ablation', default=None, choices=ABLATIONS)
    p.add_argument('--seeds', default=None, help='예: 31-40 또는 31,32')
    p.add_argument('--data', default=None)
    p.add_argument('--out', default='runs/sweep')
    p.add_argument('--eval-seed', type=int, default=None,
                   help='score every run at this randomness.seed instead of '
                        'at its own. Reproducing the paper\'s recorded '
                        'numbers for SegNeXt-T and the shuffled control arms '
                        'needs it: their evaluation consumes randomness, and '
                        'the original runs were all scored at one fixed seed '
                        'rather than at the seed they were trained at (see '
                        'RECORDED_EVAL_SEED in tools/replay.py). Leave unset '
                        'for a fresh sweep.')
    p.add_argument('--git-hash', default=None,
                   help='value for the results CSV\'s git_hash column. '
                        'Defaults to the commit of this checkout (plus '
                        '"-dirty" when tracked files differ from it); pass it '
                        'explicitly when running from a copy with no git '
                        'history, where it would otherwise read "unknown".')
    p = sub.add_parser('smoke'); p.add_argument('--all', action='store_true')
    sub.add_parser('list')

    a = ap.parse_args(argv)

    if a.cmd == 'list':
        vocab = ABLATIONS + sorted({x for _, x in VALID
                                    if x is not None and x not in ABLATIONS})
        for m in METHODS:
            ok = [x or '(없음)' for x in vocab + [None] if (m, x) in VALID]
            print(f'{m:3s} × {", ".join(sorted(BACKBONES))}  ablation: {", ".join(ok)}')
        return 0

    if a.cmd == 'export-config':
        cfg = _cfg(a)
        cfg.dump(a.out)
        print(a.out)
        return 0

    if a.cmd == 'train':
        from mmengine.runner import Runner
        Runner.from_cfg(_cfg(a)).train()
        return 0

    if a.cmd == 'test':
        from mmengine.runner import Runner

        from chamnet.checkpoint import mmengine_checkpoint_loading
        cfg = _cfg(a)
        cfg.load_from = a.checkpoint
        # torch >= 2.6 cannot unpickle an mmengine checkpoint's logging state
        # under its default safe loader -- see chamnet.checkpoint.
        with mmengine_checkpoint_loading():
            Runner.from_cfg(cfg).test()
        return 0

    if a.cmd == 'sweep':
        from chamnet.sweep import parse_seeds, run_sweep
        csv_path = run_sweep(recipe=a.recipe, methods=_names(a.methods),
                             backbones=_names(a.backbones),
                             seeds=parse_seeds(a.seeds, a.recipe),
                             data_root=a.data, out_dir=a.out,
                             ablation=a.ablation, git_hash=a.git_hash,
                             eval_seed=a.eval_seed)
        print(csv_path)
        return 0

    if a.cmd == 'smoke':
        import subprocess
        # --all runs the whole suite (every layout/dtype/registry check the
        # release ships), not just the BL synthetic-data smoke test; plain
        # `chamnet smoke` stays the fast single-file check.
        target = 'tests/' if a.all else 'tests/test_smoke.py'
        args = ['-m', 'pytest', target, '-v']
        return subprocess.call([sys.executable, *args])

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
