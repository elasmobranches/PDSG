"""Replay the published checkpoints through this package's code and compare
the result against the recorded metrics.

The weights are left alone and only the model code changes, so equal numbers
mean the port did not change the forward pass. Verifying the refactor (pulling
the fusion modules out into fusion.py) is what this exists for.

**Read verification/README.md before reading this file** -- for how to run it
and for what it cannot do, and in particular for the two reasons it exits 1
**on purpose**: SegNeXt-T's LightHamHead runs with `rand_init=True`, and
`bl/resnet18`'s Pillar sits on the two-decimal rounding boundary at 79.995.

`--all` replays every row of WORK -- nine arms x four backbones, 36 rows. With
no arguments it replays only the four methods without ablations, 16 rows.

The checkpoints it loads were **trained** at seed 37, but the comparison runs
at the seed the recorded evaluations actually ran at (RECORDED_EVAL_SEED).
That constant's own comment explains why the two differ: the original sweep
never passed a seed to its evaluation processes.
"""
import argparse
import csv
import datetime
import glob
import os

import mmseg
import pandas as pd
import torch
from mmengine.runner import Runner

import chamnet
from chamnet.checkpoint import mmengine_checkpoint_loading
from chamnet.config.builder import build_config
from chamnet.config.combos import FLOW, VALID


# (method, ablation) -> (work_dir for resnet18/mit_b0,
#                        work_dir for segnext_t/convnext_atto)
#
# The arm's *name* is not repeated here: it comes from
# chamnet.config.combos.FLOW, the same table the sweep names its work_dirs and
# CSV rows from. Written out twice, a rename would land in one and miss the
# other, and the replay would then look for checkpoints under a directory the
# sweep never wrote.
#
# Every arm was swept twice: first on resnet18 + mit_b0, then -- as a separate,
# later sweep -- on segnext_t + convnext_atto, into a differently named
# directory. The second column is written out per entry rather than derived by
# appending '_ext', because the naming is not regular and the irregularities
# are silent: `hd/rgb`'s extended sweep went to `..._controls_rgb_ext` (not
# `..._controls_ext`), and `ef/shuffled`'s went to `..._controls_ext` -- the
# *hd* controls directory -- rather than to an `..._ef_controls_ext` that does
# not exist. A rule like `wd + '_ext'` gets the second of those wrong and fails
# with a plain FileNotFoundError that reads like missing data rather than a
# wrong path. Verified against the directory listing on the training server.
WORK = {
    ('bl', None):       ('work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('ef', None):       ('work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('sd', None):       ('work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('hd', None):       ('work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('hd', 'nogate'):   ('work_dirs_v12+_512_gate_ablation',
                         'work_dirs_v12+_512_gate_ablation_ext'),
    ('hd', 'bigate'):   ('work_dirs_v12+_512_gate_ablation',
                         'work_dirs_v12+_512_gate_ablation_ext'),
    ('hd', 'shuffled'): ('work_dirs_v12+_512_controls',
                         'work_dirs_v12+_512_controls_ext'),
    ('hd', 'rgb'):      ('work_dirs_v12+_512_controls',
                         'work_dirs_v12+_512_controls_rgb_ext'),
    ('ef', 'shuffled'): ('work_dirs_v12+_512_ef_controls',
                         'work_dirs_v12+_512_controls_ext'),
}
# `--methods` names the training methods to replay without ablations, which is
# the quick check; `--all` replays every WORK entry -- all nine arms on all four
# backbones, 36 rows -- and is what verification/replay.csv is generated with.
IMPLEMENTED_METHODS = ('bl', 'ef', 'sd', 'hd')
EXT_BACKBONES = ('segnext_t', 'convnext_atto')      # backbones of the extended sweep
BACKBONES = ('resnet18', 'mit_b0', 'segnext_t', 'convnext_atto')

# The seed the replayed checkpoints were *trained* at. It selects which run
# directory to read a checkpoint from and which results_v8.csv row to compare
# against; it is not the seed the comparison runs at. See below.
RUN_SEED = 37

# The seed the recorded evaluations actually ran at -- 42 for every row of
# results_v8.csv, whatever seed the run was trained at.
#
# Not a typo, and not the run's seed. The campaign's sweep script
# (`run_paper_sweep.sh`) trained with `--cfg-options randomness.seed="$run"`,
# then scored the resulting checkpoint in two further `tools/test.py`
# processes -- one on test, one on val -- and passed neither of those the
# seed. Both therefore fell back to the value in the config chain's base
# runtime file, `randomness = dict(seed=42, ...)`. The campaign's own console
# logs show exactly that: in every run directory `train_console.log` reports
# the run's seed (37, 39, ...) while `test_console.log` and `val_console.log`
# both report 42.
#
# It matters for the two arms whose *evaluation* consumes randomness --
# SegNeXt-T's LightHamHead (`rand_init=True` draws `torch.rand` on every
# forward pass) and the shuffled controls (a fresh depth permutation per
# sample). Replaying those at the training seed puts them on a different RNG
# trajectory from the recorded numbers, and no amount of re-running closes
# the gap. Measured, same checkpoints and worker count, one fresh process per
# row, changing only this seed:
#
#     row                      eval at 37   eval at 42   recorded
#     hd/shuffled/resnet18     80.96        80.97        80.97
#     hd/shuffled/mit_b0       76.69        76.71        76.71
#     bl/segnext_t             80.76        80.81        80.80
#
# Every other arm is unaffected: nothing else in the release consumes RNG at
# test time, which is why replaying them at either seed gave the same answer.
#
# A consequence worth stating for anyone reading numbers out of
# results_v8.csv: because every recorded evaluation ran at one fixed seed,
# the ten training seeds differ purely in training, and evaluation-time
# randomness was held constant across all conditions and seeds. That makes
# the between-condition comparisons cleaner. It also means a shuffled arm's
# number is conditional on one permutation realisation rather than averaged
# over permutations.
RECORDED_EVAL_SEED = 42


def _use_original_val_test_layout(dataloader_cfg):
    """Point val/test at the exact files the paper's own test run read.

    build_config's dataloader always points at 'masks' + depth_suffix='.npy'
    — correct for the release's own unified data layout, which normalises
    away the paper's inconsistent depth_suffix/mask-folder naming
    on the assumption that a properly migrated copy carries the same content
    under the new names. The dataset copy the recorded metrics were computed
    against is not that migrated copy — it's the original, un-migrated paper
    dataset: val/test depth files are suffixed '_depth.npy' (only train is
    '.npy'), and val/test labels live in 'masks_gray'/'*_mask_gray.png' rather
    than 'masks'/'*.png'.

    The mask-folder swap isn't cosmetic: comparing all 45 test images'
    masks/*.png against masks_gray/*_mask_gray.png byte-for-byte found one
    real mismatch — 250807output_video_250421_chamoe_vertical_flower_0940_
    frame_000012 differs in 14,670 of 518,400 pixels (~2.8%) between the two
    folders, a real full-height greenhouse post that masks/ dropped (verified
    against the RGB — see verification/README.md). Every other one of the 45
    test images' two label sources are identical, and results_v8.csv was
    generated against masks_gray, so the replay reads the same file the
    recorded metrics did rather than 'masks' (which is otherwise correct for
    anyone using a properly migrated dataset copy — this is a data quality
    gap in this specific un-migrated mount, not a bug in the release code).
    """
    for step in dataloader_cfg['dataset']['pipeline']:
        if step.get('type') == 'LoadDepthAsChannel':
            step['depth_suffix'] = '_depth.npy'
    prefix = dataloader_cfg['dataset']['data_prefix']
    prefix['seg_map_path'] = prefix['seg_map_path'].rsplit('/', 1)[0] + '/masks_gray'
    dataloader_cfg['dataset']['seg_map_suffix'] = '_mask_gray.png'


def replay(data_root, src, method, backbone, ablation, seed=RUN_SEED,
           eval_seed=RECORDED_EVAL_SEED):
    """Replay one recorded run and compare against what it recorded.

    `seed` picks the run -- its directory and its results_v8.csv row.
    `eval_seed` is what the comparison evaluates at, and the two are
    deliberately different numbers: see RECORDED_EVAL_SEED.
    """
    flow = FLOW[(method, ablation)]
    wd, wd_ext = WORK[(method, ablation)]
    wd = wd_ext if backbone in EXT_BACKBONES else wd
    run = f'{src}/{wd}/{seed}/chamnet_{flow}_{backbone}'
    matches = glob.glob(f'{run}/best_mIoU_iter_*.pth')
    if not matches:
        raise FileNotFoundError(
            f'no checkpoint found in {run} (expected exactly one '
            'best_mIoU_iter_*.pth)')
    if len(matches) > 1:
        # max_keep_ckpts=1 during training should make this impossible; if
        # it isn't, guessing via a lexicographic sort risks silently picking
        # a stale checkpoint ("iter_100" < "iter_20" as strings) instead of
        # the run's actual best -- refuse instead of guessing.
        raise RuntimeError(
            f'expected exactly one checkpoint in {run}, found '
            f'{len(matches)}: {matches}')
    ckpt = matches[0]

    # seed=eval_seed, not seed: build_config puts this into
    # `randomness.seed`, and what has to be reproduced here is the RNG state
    # the recorded *evaluation* ran at, not the one its training ran at.
    cfg = build_config(method=method, backbone=backbone, ablation=ablation,
                       recipe='paper', data_root=data_root, seed=eval_seed)
    cfg.load_from = ckpt
    cfg.work_dir = f'/tmp/replay/{flow}_{backbone}'
    _use_original_val_test_layout(cfg.val_dataloader)
    _use_original_val_test_layout(cfg.test_dataloader)
    # Swap in the per-class-emitting metric (chamnet.metrics, registered by
    # chamnet.register_all()) so `metrics['IoU.pillar']` below is populated;
    # iou_metrics stays as the builder set it, only the registered type
    # changes. The sweep does the same, for the same reason.
    cfg.test_evaluator['type'] = 'IoUMetricWithPerClass'
    # These checkpoints carry pickled mmengine logging state that torch
    # >= 2.6's default safe unpickler refuses -- see chamnet.checkpoint,
    # which is also what the sweep uses to read the ones it writes.
    with mmengine_checkpoint_loading():
        metrics = Runner.from_cfg(cfg).test()

    rec = pd.read_csv(f'{src}/{wd}/results_v8.csv')
    row = rec[(rec.flow == flow) & (rec.backbone == backbone) & (rec.seed == seed)].iloc[0]
    # float() before round(): IoUMetric returns numpy scalars, and leaving
    # them numpy makes this dict silently un-JSON-able for anything that wants
    # to serialise it. The CSV writer str()s them either way.
    return dict(method=method, backbone=backbone, ablation=ablation or '-',
                replay_mIoU=round(float(metrics['mIoU']), 4),
                recorded_mIoU=round(float(row.test_mIoU), 4),
                replay_pillar=round(float(metrics['IoU.pillar']), 4),
                recorded_pillar=round(float(row.test_IoU_pillar), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, metavar='ROOT',
                    help='root of the dataset copy the recorded metrics were '
                         'computed against. Note this is NOT the layout '
                         'docs/DATA_FORMAT.md describes: the recorded runs '
                         'read an un-migrated copy whose val/test depth files '
                         "carry a '_depth' infix and whose val/test labels "
                         "live under 'masks_gray'. This script reproduces "
                         'those paths itself -- see '
                         '_use_original_val_test_layout.')
    ap.add_argument('--src', required=True, metavar='ROOT',
                    help='root holding the recorded runs\' work_dirs and '
                         'their results_v8.csv. Required for the same reason '
                         'as --data: it locates both the checkpoints to '
                         'replay and the metrics to compare them against, so '
                         'a wrong value scores the wrong weights against the '
                         'wrong reference, and any default would name one '
                         "machine's filesystem. The checkpoints found under "
                         'it are loaded with the pickle enabled (see '
                         'chamnet.checkpoint), so point it only at '
                         'checkpoints you trust.')
    ap.add_argument('--methods', default=','.join(IMPLEMENTED_METHODS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default='verification/replay.csv')
    ap.add_argument('--eval-seed', type=int, default=RECORDED_EVAL_SEED,
                    help='randomness.seed to evaluate at. The default is the '
                         'seed the recorded evaluations ran at, which is not '
                         'the seed the checkpoints were trained at -- see '
                         'RECORDED_EVAL_SEED in this file. Override it to '
                         'reproduce the measurement that established that.')
    ap.add_argument('--commit', default='unknown',
                    help="short git commit hash this replay ran against. The "
                         "GPU container this script runs in doesn't have a "
                         ".git directory (not synced), so pass it from the "
                         'caller\'s side, e.g. --commit "$(git rev-parse '
                         '--short HEAD)" -- this stamps replay.csv so it is '
                         'citable rather than a bare, undated snapshot.')
    a = ap.parse_args()
    chamnet.register_all()

    # WORK is the recorded side and combos.VALID is the shipped side; if they
    # ever disagree, --all silently stops covering an arm the package still
    # advertises (or asks for a checkpoint that was never trained). Neither
    # failure announces itself, so check rather than assume.
    if set(WORK) != VALID:
        raise SystemExit(
            f'WORK and chamnet.config.combos.VALID disagree: '
            f'only in WORK {sorted(set(WORK) - VALID, key=str)}, '
            f'only in VALID {sorted(VALID - set(WORK), key=str)}')

    combos = list(WORK) if a.all else [(m, None) for m in a.methods.split(',')]
    rows = [replay(a.data, a.src, m, bb, ab, eval_seed=a.eval_seed)
            for m, ab in combos
            for bb in BACKBONES]
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    provenance = (f'# commit={a.commit} '
                 f'date={datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                 f'run_seed={RUN_SEED} eval_seed={a.eval_seed} '
                 f'gpu={gpu!r} torch={torch.__version__} mmseg={mmseg.__version__}')
    with open(a.out, 'w', newline='') as f:
        f.write(provenance + '\n')
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Gate on Pillar as well as mIoU: it's the paper's headline metric, and
    # it moves further than mIoU on the same rows -- e.g. hd/nogate/segnext_t
    # is 0.01 off on mIoU and 0.44 off on Pillar. Gating on mIoU alone would
    # let that pass unnoticed.
    #
    # The gate stays at 1e-3 (i.e. "equal at the 2 decimals results_v8.csv
    # stores") deliberately. Two row classes are expected to fail it and are
    # documented in verification/README.md rather than tolerated here:
    #
    #   * SegNeXt-T rows. Its decode head draws `torch.rand` on every forward
    #     pass, and evaluating at the recorded evaluation seed narrowed these
    #     but did not close them: 8 of the 9 still differ, by 0.00-0.07 mIoU
    #     and 0.00-0.44 Pillar. Why is an open question; see the README.
    #   * bl/resnet18's Pillar, whose raw value straddles the 79.995 rounding
    #     boundary so the second decimal flips between runs while the quantity
    #     itself moves by ~6 pixels in 1.46M.
    #
    # The shuffled arms used to be a third class and are not any more: all six
    # non-SegNeXt shuffled rows now match exactly on both metrics. What was
    # missing was the evaluation seed (see RECORDED_EVAL_SEED), not anything
    # about the arms.
    #
    # So the printed pass count under --all is 28/36, plus or minus the rows
    # the ~0.01 GPU flutter happens to land on and one for the boundary row.
    # Widening the gate to absorb any of this would also stop it catching a
    # real regression of the same size.
    bad = [r for r in rows
           if abs(r['replay_mIoU'] - r['recorded_mIoU']) > 1e-3
           or abs(r['replay_pillar'] - r['recorded_pillar']) > 1e-3]
    print(f'{len(rows) - len(bad)}/{len(rows)} rows match the recorded metrics')
    raise SystemExit(1 if bad else 0)
