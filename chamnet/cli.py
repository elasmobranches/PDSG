"""빌더 위의 얇은 껍데기. 로직은 전부 chamnet.config.builder 에 있다."""
from __future__ import annotations

import argparse
import sys

import chamnet
from chamnet.config.backbones import BACKBONES
from chamnet.config.builder import build_config

METHODS = ['bl', 'ef', 'sd', 'hd']
ABLATIONS = ['shuffled', 'rgb', 'nogate', 'bigate']


def _common(sp):
    sp.add_argument('--method', required=True, choices=METHODS)
    sp.add_argument('--backbone', required=True, choices=sorted(BACKBONES))
    sp.add_argument('--ablation', default=None, choices=ABLATIONS)
    sp.add_argument('--recipe', default='paper_v13')
    sp.add_argument('--data', default=None, help='데이터 루트. 레시피 값을 덮어쓴다')
    sp.add_argument('--seed', type=int, default=31)


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
    p.add_argument('--checkpoint', required=True)
    p = sub.add_parser('export-config'); _common(p)
    p.add_argument('-o', '--out', required=True)
    p = sub.add_parser('sweep')
    p.add_argument('--recipe', default='paper_v13')
    p.add_argument('--methods', default=','.join(METHODS))
    p.add_argument('--backbones', default=','.join(sorted(BACKBONES)))
    p.add_argument('--seeds', default=None, help='예: 31-40 또는 31,32')
    p.add_argument('--data', default=None)
    p.add_argument('--out', default='runs/sweep')
    p = sub.add_parser('smoke'); p.add_argument('--all', action='store_true')
    sub.add_parser('list')

    a = ap.parse_args(argv)

    if a.cmd == 'list':
        from chamnet.config.combos import VALID
        for m in METHODS:
            ok = [x or '(없음)' for x in ABLATIONS + [None] if (m, x) in VALID]
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
        cfg = _cfg(a)
        cfg.load_from = a.checkpoint
        Runner.from_cfg(cfg).test()
        return 0

    if a.cmd == 'sweep':
        from chamnet.sweep import run_sweep, parse_seeds
        run_sweep(recipe=a.recipe, methods=a.methods.split(','),
                  backbones=a.backbones.split(','), seeds=parse_seeds(a.seeds, a.recipe),
                  data_root=a.data, out_dir=a.out)
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
