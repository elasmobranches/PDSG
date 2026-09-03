"""논문 체크포인트를 새 패키지 코드로 재생해 기록 수치와 대조한다.

가중치는 그대로 두고 모델 코드만 바꿔 같은 숫자가 나오면, 이식이 forward 를
바꾸지 않았다는 뜻이다. 리팩터링(fusion.py 분리)을 검증하는 것이 주 목적이다.

사용법과 알려진 한계는 verification/README.md 를 먼저 읽을 것 — 특히 이
스크립트가 **의도적으로 exit 1** 하는 두 이유(SegNeXt-T LightHamHead 의
rand_init=True, 그리고 bl/resnet18 Pillar 가 소수점 2자리 반올림 경계인
79.995 에 걸쳐 있는 것).

`--all` 은 WORK 의 9개 arm × 4 백본 = 36행 전부를 재생한다. 인자 없이 돌리면
ablation 없는 4개 method 만 (16행) 돈다.
"""
import argparse
import contextlib
import csv
import datetime
import glob
import os

import mmseg
import pandas as pd
import torch
from mmengine.runner import Runner
from mmseg.evaluation import IoUMetric
from mmseg.registry import METRICS

import chamnet
from chamnet.config.builder import build_config


@contextlib.contextmanager
def _load_with_full_pickle():
    """Scope torch>=2.6's weights_only=False fallback to one checkpoint load.

    These checkpoints carry a pickled mmengine `HistoryBuffer` (optimizer/
    logging state) that torch>=2.6's default `weights_only=True` safe
    unpickler doesn't cover, so loading them needs the old full-pickle
    behaviour. Forcing that *process-wide from import time* (as an earlier
    version of this script did) is an ACE footgun on any other .pth path
    that runs through torch.load in this process afterwards — an
    attacker-controlled checkpoint would silently get full unpickling
    instead of the restricted loader torch>=2.6 defaults to. Monkeypatching
    torch.load only for the duration of the one call that needs it, then
    restoring it, keeps that exposure to exactly this block.
    """
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
    try:
        yield
    finally:
        torch.load = orig


@METRICS.register_module()
class IoUMetricWithPerClass(IoUMetric):
    """IoUMetric that also puts each class's IoU into the returned dict.

    Vanilla IoUMetric.compute_metrics only returns the aggregate keys
    (aAcc, mIoU, mAcc, mDice, mFscore, ...) — per-class numbers are printed
    to the log table but never handed back to the caller. The replay needs
    `IoU.pillar` specifically to compare against results_v8.csv's recorded
    per-class column, so recompute the same per-class breakdown IoUMetric
    already does internally (`total_area_to_metrics`) and add it to the dict
    this method returns.
    """

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        areas = tuple(zip(*results))
        per_class = self.total_area_to_metrics(
            sum(areas[0]), sum(areas[1]), sum(areas[2]), sum(areas[3]),
            self.metrics, self.nan_to_num, self.beta)
        per_class.pop('aAcc', None)
        class_names = self.dataset_meta['classes']
        for key, values in per_class.items():
            for name, v in zip(class_names, values):
                metrics[f'{key}.{name}'] = round(float(v) * 100, 2)
        return metrics


# (method, ablation) -> (flow, work_dir for resnet18/mit_b0,
#                                work_dir for segnext_t/convnext_atto)
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
    ('bl', None):       ('baseline',
                         'work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('ef', None):       ('proposed',
                         'work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('sd', None):       ('dual',
                         'work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('hd', None):       ('dual_plus',
                         'work_dirs_v12+_512bc_pretrained',
                         'work_dirs_v12+_512bc_pretrained'),
    ('hd', 'nogate'):   ('dual_plus_nogate',
                         'work_dirs_v12+_512_gate_ablation',
                         'work_dirs_v12+_512_gate_ablation_ext'),
    ('hd', 'bigate'):   ('dual_plus_bigate',
                         'work_dirs_v12+_512_gate_ablation',
                         'work_dirs_v12+_512_gate_ablation_ext'),
    ('hd', 'shuffled'): ('dual_plus_shuffled',
                         'work_dirs_v12+_512_controls',
                         'work_dirs_v12+_512_controls_ext'),
    ('hd', 'rgb'):      ('dual_plus_rgb',
                         'work_dirs_v12+_512_controls',
                         'work_dirs_v12+_512_controls_rgb_ext'),
    ('ef', 'shuffled'): ('proposed_shuffled',
                         'work_dirs_v12+_512_ef_controls',
                         'work_dirs_v12+_512_controls_ext'),
}
# `--methods` names the training methods to replay without ablations, which is
# the quick check; `--all` replays every WORK entry -- all nine arms on all four
# backbones, 36 rows -- and is what verification/replay.csv is generated with.
IMPLEMENTED_METHODS = ('bl', 'ef', 'sd', 'hd')
EXT_BACKBONES = ('segnext_t', 'convnext_atto')      # 확장 sweep 백본
BACKBONES = ('resnet18', 'mit_b0', 'segnext_t', 'convnext_atto')


def _use_original_val_test_layout(dataloader_cfg):
    """Point val/test at the exact files the paper's own test run read.

    build_config's dataloader always points at 'masks' + depth_suffix='.npy'
    — correct for the release's own unified data layout, which normalises
    away the paper's inconsistent depth_suffix/mask-folder naming
    on the assumption that a properly migrated copy carries the same content
    under the new names. /data/real_dataset_v12_plus is not that migrated
    copy — it's the original, un-migrated paper dataset, val/test depth
    files are suffixed '_depth.npy' (only train is '.npy'), and val/test
    labels live in 'masks_gray'/'*_mask_gray.png' rather than 'masks'/'*.png'.

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


def replay(data_root, src, method, backbone, ablation, seed=37):
    flow, wd, wd_ext = WORK[(method, ablation)]
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

    cfg = build_config(method=method, backbone=backbone, ablation=ablation,
                       recipe='paper_v13', data_root=data_root, seed=seed)
    cfg.load_from = ckpt
    cfg.work_dir = f'/tmp/replay/{flow}_{backbone}'
    _use_original_val_test_layout(cfg.val_dataloader)
    _use_original_val_test_layout(cfg.test_dataloader)
    # Swap in the per-class-emitting metric (see IoUMetricWithPerClass above)
    # so `metrics['IoU.pillar']` below is populated; iou_metrics stays as the
    # builder set it, only the registered type changes.
    cfg.test_evaluator['type'] = 'IoUMetricWithPerClass'
    with _load_with_full_pickle():
        metrics = Runner.from_cfg(cfg).test()

    rec = pd.read_csv(f'{src}/{wd}/results_v8.csv')
    row = rec[(rec.flow == flow) & (rec.backbone == backbone) & (rec.seed == seed)].iloc[0]
    return dict(method=method, backbone=backbone, ablation=ablation or '-',
                replay_mIoU=round(metrics['mIoU'], 4),
                recorded_mIoU=round(float(row.test_mIoU), 4),
                replay_pillar=round(metrics['IoU.pillar'], 4),
                recorded_pillar=round(float(row.test_IoU_pillar), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/data/real_dataset_v12_plus')
    ap.add_argument('--src', default='/data')
    ap.add_argument('--methods', default=','.join(IMPLEMENTED_METHODS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default='verification/replay.csv')
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
    from chamnet.config.combos import VALID
    if set(WORK) != VALID:
        raise SystemExit(
            f'WORK and chamnet.config.combos.VALID disagree: '
            f'only in WORK {sorted(set(WORK) - VALID, key=str)}, '
            f'only in VALID {sorted(VALID - set(WORK), key=str)}')

    combos = list(WORK) if a.all else [(m, None) for m in a.methods.split(',')]
    rows = [replay(a.data, a.src, m, bb, ab)
            for m, ab in combos
            for bb in BACKBONES]
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    provenance = (f'# commit={a.commit} '
                 f'date={datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                 f'gpu={gpu!r} torch={torch.__version__} mmseg={mmseg.__version__}')
    with open(a.out, 'w', newline='') as f:
        f.write(provenance + '\n')
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Gate on Pillar as well as mIoU: it's the paper's headline metric, and
    # it moves further than mIoU on the same rows -- e.g. bl/segnext_t was
    # 0.04 off on mIoU but 0.20 off on Pillar. Gating on mIoU alone would
    # have let that pass unnoticed.
    #
    # The gate stays at 1e-3 (i.e. "equal at the 2 decimals results_v8.csv
    # stores") deliberately. Three row classes are expected to fail it and are
    # documented in verification/README.md rather than tolerated here:
    #
    #   * SegNeXt-T's RNG-dependent decode head -- one row per arm, 9 of the 36
    #     under --all.
    #   * the shuffled arms, which draw their depth permutation at test time --
    #     8 rows, of which 2 are SegNeXt-T rows already counted, so 6 more.
    #   * bl/resnet18's Pillar, whose raw value straddles the 79.995 rounding
    #     boundary so the second decimal flips between runs while the quantity
    #     itself moves by ~6 pixels in 1.46M.
    #
    # That is 15 rows that do not match plus one coin flip, so the printed pass
    # count under --all is 20/36 or 21/36, less one for each row the ~0.01 GPU
    # flutter happens to land on (it took two in the committed run, hence
    # 19/36). Widening the gate to absorb any of this would also stop it
    # catching a real regression of the same size.
    bad = [r for r in rows
           if abs(r['replay_mIoU'] - r['recorded_mIoU']) > 1e-3
           or abs(r['replay_pillar'] - r['recorded_pillar']) > 1e-3]
    print(f'{len(rows) - len(bad)}/{len(rows)} 일치')
    raise SystemExit(1 if bad else 0)
