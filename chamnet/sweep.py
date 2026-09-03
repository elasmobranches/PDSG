"""Resumable sweep over seed × method × backbone.

The paper's numbers are means over ten training seeds, so producing them means
running one combination after another for days on a single GPU. A sweep that
long is going to be interrupted — a crash, a reboot, a machine needed for
something else — so being able to relaunch the identical command and have it
carry on is not a convenience here, it is the only way the campaign finished.

The resumption rule is a completion marker per run: a `DONE` file inside the
run's own work_dir. Two properties make it trustworthy, and both are about
*when* it is written rather than about the marker itself:

* It is written **last**. A run is trained, scored on test and val, and its
  complete CSV row is written to `metrics.json` and flushed — only then does
  the marker appear. A run that died anywhere before that has no marker, so a
  resume re-runs it. A half-trained run is never mistaken for a finished one.
* The results CSV is **rebuilt from the markers**, not appended to. Every
  finished run keeps its own row on disk next to its checkpoint, and the CSV
  is a rendering of those. So an interrupted sweep cannot leave a torn row
  behind, resuming cannot duplicate a row already written, and deleting the
  CSV entirely costs nothing — the next resume writes it back in full.

The CSV's columns are the campaign's own recorded schema (`FIELDNAMES`), so
the analysis that was written against those files reads these unchanged.
"""
from __future__ import annotations

import copy
import csv
import glob
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from chamnet.config.backbones import BACKBONES
from chamnet.config.combos import FLOW, validate
from chamnet.config.schema import load_recipe

#: Class order of the recorded schema's per-class columns — the dataset's own
#: label order, which is also the order IoUMetric reports them in.
CLASSES = ('background', 'chamoe', 'heatpipe', 'path', 'pillar',
           'topdownfarm', 'ceiling', 'duct')

#: Aggregate metrics recorded per split. All seven come out of a single
#: IoUMetric configured with iou_metrics=['mIoU', 'mDice', 'mFscore'] — the
#: builder's setting, and the paper's.
SUMMARY_KEYS = ('aAcc', 'mIoU', 'mAcc', 'mDice', 'mFscore', 'mPrecision',
                'mRecall')

METRIC_KEYS = tuple(f'IoU_{c}' for c in CLASSES) + SUMMARY_KEYS

#: The recorded results schema, column for column and in order. Matched
#: against a real campaign CSV rather than transcribed from a description.
FIELDNAMES = (['run', 'flow', 'backbone', 'seed', 'git_hash', 'best_iter']
              + [f'val_{k}' for k in METRIC_KEYS]
              + [f'test_{k}' for k in METRIC_KEYS])

#: Written last, and only once everything else about a run is on disk.
MARKER = 'DONE'
#: One finished run's CSV row, kept beside its checkpoints.
METRICS_FILE = 'metrics.json'
RESULTS = 'results.csv'


def parse_seeds(spec: str | None, recipe: str) -> list[int]:
    """Turn a `--seeds` argument into the list of seeds to run.

    `None` means "the recipe's own seeds", which is what reproduces the paper:
    the paper recipe lists the ten it was run with. `'31-40'` is an inclusive range,
    `'31,32'` an explicit list.
    """
    if spec is None or spec == '':
        return [int(s) for s in load_recipe(recipe).runtime.seeds]
    text = spec.strip()
    try:
        if '-' in text:
            lo, hi = text.split('-')
            seeds = list(range(int(lo), int(hi) + 1))
        else:
            seeds = [int(s) for s in text.split(',')]
    except ValueError:
        raise ValueError(
            f'cannot read {spec!r} as seeds; expected a range like "31-40" or '
            'a list like "31,32,33"') from None
    if not seeds:
        raise ValueError(f'{spec!r} selects no seeds')
    return seeds


def detect_git_hash(repo: str | Path | None = None) -> str:
    """Identify the code a sweep ran, for the recorded `git_hash` column.

    The campaign wrote placeholders here (`uncommitted-`), which records
    nothing: a row stamped that way cannot be traced to the code that produced
    it. What this writes instead is the commit of the checkout the running
    `chamnet` package lives in, abbreviated to the twelve characters the column
    has always held, with `-dirty` appended when tracked files differ from that
    commit. A dirty tree is not refused — mid-campaign edits happen — but it is
    never recorded as if it were the commit, because "this row came from
    <hash>, plus edits nobody kept" is the honest statement and "this row came
    from <hash>" would be false.

    Untracked files are ignored, matching what the campaign's own guard
    considered dirty: they cannot change what the package computes.

    Returns `'unknown'` when the package is not in a git checkout at all — an
    installed wheel, or a copy exported to a machine without its history. Pass
    `run_sweep(git_hash=...)` in that case; a stamp handed in from a caller
    that does know is worth more than one this function has to guess.
    """
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parent

    def _git(*args: str) -> str | None:
        try:
            done = subprocess.run(('git', '-C', str(repo)) + args,
                                  capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    head = _git('rev-parse', 'HEAD')
    if head is None:
        return 'unknown'
    changed = _git('status', '--porcelain', '--untracked-files=no')
    return head[:12] + ('-dirty' if changed else '')


def work_dir_for(out_dir: str | Path, seed: int, method: str,
                 backbone: str, ablation: str | None = None) -> Path:
    """`<out_dir>/<seed>/chamnet_<flow>_<backbone>` — the campaign's layout.

    Kept identical to what the campaign produced so a resumed or extended
    sweep lands beside the runs it is extending, and so `tools/replay.py`,
    which reads that layout, can read these too.
    """
    return Path(out_dir) / str(seed) / f'chamnet_{FLOW[(method, ablation)]}_{backbone}'


def _best_checkpoint(work_dir: Path, save_best: str) -> Path:
    """The single best-metric checkpoint a finished training run leaves behind."""
    matches = sorted(glob.glob(str(work_dir / f'best_{save_best}_iter_*.pth')))
    if not matches:
        raise FileNotFoundError(
            f'training left no best_{save_best}_iter_*.pth in {work_dir}; the '
            'run cannot be scored or recorded')
    if len(matches) > 1:
        # max_keep_ckpts=1 should make this impossible. If it happens anyway,
        # picking one by sorting would silently prefer a stale checkpoint
        # ('iter_100' sorts before 'iter_20' as text), so refuse instead.
        raise RuntimeError(
            f'expected exactly one best_{save_best}_iter_*.pth in {work_dir}, '
            f'found {len(matches)}: {matches}')
    return Path(matches[0])


def _best_iter(checkpoint: Path) -> int:
    match = re.search(r'_iter_(\d+)\.pth$', checkpoint.name)
    if match is None:
        raise ValueError(f'cannot read an iteration number out of {checkpoint}')
    return int(match.group(1))


def _metrics_of(raw: dict) -> dict:
    """Pull the recorded schema's metric names out of an evaluator's output.

    The per-class keys arrive as `'IoU.pillar'` (see chamnet.metrics) and the
    column is `IoU_pillar`; the aggregates already match. Keys the evaluator
    did not produce are left out rather than defaulted, so `_row` can refuse
    the row instead of writing a plausible-looking zero.

    This is also where the numbers stop being numpy's. IoUMetric accumulates
    float32 pixel-count tensors and returns `np.float32`/`np.float64` scalars,
    which are not JSON-serialisable at all — and a plain `float()` on a
    float32 would widen `96.35` into `96.3499984741211`, because the value is
    already the 2-decimal quantity IoUMetric rounded it to and float32 cannot
    hold that exactly. The campaign's own recorder never met this: it read its
    numbers back out of the printed log as text, so `float('96.35')` was
    exact. Rounding to the two decimals the value already carries reproduces
    that, and leaves a row of ordinary Python floats.
    """
    def clean(value):
        return round(float(value), 2)

    out = {f'IoU_{c}': clean(raw[f'IoU.{c}']) for c in CLASSES
           if f'IoU.{c}' in raw}
    out.update({k: clean(raw[k]) for k in SUMMARY_KEYS if k in raw})
    return out


def _evaluate(cfg, checkpoint: Path, dataloader, work_dir: Path,
              eval_seed: int | None = None) -> dict:
    """Score `checkpoint` on one split.

    A fresh `Runner` per split, rather than reusing the one that trained.
    `Runner.__init__` re-seeds every generator an evaluation can consume from
    `randomness.seed` (`set_randomness` -> `set_random_seed`: `random`,
    `numpy`, `torch`, `torch.cuda`), and dataloader worker streams are seeded
    from `(num_workers, rank, worker_id, runner.seed)` rather than inherited
    from the parent — so a new Runner starts an evaluation from exactly the
    RNG state a newly launched process would, and the campaign did score its
    checkpoints in two separate processes. Which is why this runs test and
    val in one process without that changing either: neither carries RNG
    history into the other.

    That equivalence is not only an argument from the source. The replay
    builds one Runner per row in a single process and reproduces its
    RNG-sensitive rows byte-identically across invocations that ran a
    different number of rows in a different order (verification/README.md),
    which is the same claim measured.

    `eval_seed` overrides the seed used for scoring only. It defaults to the
    run's own seed, which is the coherent choice for a fresh sweep. The
    campaign's evaluations did not do that: its two scoring processes were
    never passed a seed and so all evaluated at the base config's fixed
    value, whatever seed had been trained. Reproducing a recorded number for
    the two arms whose evaluation consumes randomness — SegNeXt-T's decode
    head, and the shuffled controls — therefore needs that value passed here
    explicitly. See `tools/replay.py`'s `RECORDED_EVAL_SEED`, which documents
    it and the measurement that established it.
    """
    from mmengine.runner import Runner

    from chamnet.checkpoint import mmengine_checkpoint_loading

    cfg = copy.deepcopy(cfg)
    cfg.load_from = str(checkpoint)
    cfg.work_dir = str(work_dir)
    cfg.test_dataloader = copy.deepcopy(dataloader)
    if eval_seed is not None:
        cfg.randomness = dict(cfg.randomness, seed=eval_seed)
    # Same metric, plus the per-class breakdown the schema's IoU_* columns
    # need; iou_metrics stays exactly as the builder set it.
    cfg.test_evaluator = dict(cfg.test_evaluator, type='IoUMetricWithPerClass')
    # `Runner.test()` loads cfg.load_from itself, and on torch >= 2.6 that
    # load fails on the checkpoint training just wrote -- see
    # chamnet.checkpoint.
    with mmengine_checkpoint_loading():
        return _metrics_of(Runner.from_cfg(cfg).test())


def _train_one(*, method, backbone, ablation, seed, recipe, data_root,
               work_dir, eval_seed=None) -> dict:
    """Train one combination and score its best checkpoint on test and val.

    Returns ``{'best_iter': int, 'val': {...}, 'test': {...}}``, the metric
    dicts keyed by `METRIC_KEYS`. This is the one function in the module that
    needs a GPU, which is why it is a separate function: the tests replace it
    to exercise everything around it.

    Order matches the campaign's: train, then score the best checkpoint on
    test, then on val, then record. Both evaluations use the same checkpoint —
    the best-mIoU one training selected, not the last.
    """
    import chamnet
    from chamnet.config.builder import build_config
    from mmengine.runner import Runner

    chamnet.register_all()
    work_dir = Path(work_dir)
    cfg = build_config(method=method, backbone=backbone, ablation=ablation,
                       recipe=recipe, data_root=data_root, seed=seed,
                       work_dir=str(work_dir))
    Runner.from_cfg(cfg).train()

    checkpoint = _best_checkpoint(work_dir,
                                  cfg.default_hooks['checkpoint']['save_best'])
    return dict(
        best_iter=_best_iter(checkpoint),
        test=_evaluate(cfg, checkpoint, cfg.test_dataloader, work_dir / 'test',
                       eval_seed),
        val=_evaluate(cfg, checkpoint, cfg.val_dataloader,
                      work_dir / 'val_eval', eval_seed),
    )


def _row(*, seed: int, flow: str, backbone: str, stamp: str,
         metrics: dict) -> dict:
    """Assemble one complete CSV row, or refuse.

    Refusing is the point. A row is what tells a later resume this run is
    finished, so a row missing a metric would freeze that gap permanently: the
    marker goes down, the combination is never re-run, and the column stays
    empty in every table built from the file. The campaign's own recorder
    checked the same thing before it wrote.
    """
    row = dict(run=seed, flow=flow, backbone=backbone, seed=seed,
               git_hash=stamp, best_iter=metrics.get('best_iter'))
    for split in ('val', 'test'):
        values = metrics.get(split) or {}
        row.update({f'{split}_{k}': values.get(k) for k in METRIC_KEYS})
    _require_complete(row, f'{flow}/{backbone} seed {seed}')
    return row


def _require_complete(row: dict, where: str) -> None:
    missing = [name for name in FIELDNAMES if row.get(name) is None]
    if missing:
        raise ValueError(
            f'refusing to record {where}: {len(missing)} of {len(FIELDNAMES)} '
            f'columns have no value ({", ".join(missing)})')


def _write_atomically(path: Path, write) -> None:
    """Replace `path` in one step, so a crash leaves either version, not half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f'.{path.name}.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _completed_rows(out_dir: Path) -> list[dict]:
    """Read back the row of every finished run anywhere under `out_dir`.

    Every run that ever finished here, not only the ones this invocation was
    asked for. The campaign accumulated one results file per work root across
    many separate invocations — main arms first, control arms later — and
    rebuilding from only the current request would silently shorten the file
    every time someone resumed with a narrower `--methods`.

    Sorted by (seed, flow, backbone) so the file is byte-identical whichever
    order the runs were actually finished in.
    """
    rows = []
    for run_dir in sorted(p for p in out_dir.glob('*/*') if p.is_dir()):
        if not (run_dir / MARKER).exists():
            continue
        path = run_dir / METRICS_FILE
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f'{run_dir} is marked complete but its {METRICS_FILE} cannot be '
                f'read ({exc}). This sweep writes that file before the '
                f'{MARKER} marker, so a marker without it did not come from '
                f'here. Delete {run_dir / MARKER} to re-run the combination, '
                f'or delete {run_dir}.') from exc
        _require_complete(row, str(path))
        rows.append({name: row[name] for name in FIELDNAMES})
    return sorted(rows, key=lambda r: (r['seed'], r['flow'], r['backbone']))


def _write_results(csv_path: Path, rows: list[dict]) -> None:
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    _write_atomically(csv_path, write)


def run_sweep(*, recipe, methods, backbones, seeds, data_root, out_dir,
              ablation=None, git_hash=None, eval_seed=None) -> Path:
    """Train every requested combination that is not already finished.

    Returns the path of the results CSV, which is rewritten from the finished
    runs on disk after each one completes — so it is complete and readable at
    every moment, including after an interruption, and a resume neither
    duplicates nor loses a row.

    Failures are not swallowed. If a run raises, the sweep stops there with
    that error: everything finished up to that point keeps its marker and its
    row, and relaunching the same command picks up from the failure. Carrying
    on past it would bury the reason a run failed under however many hours of
    output came after.

    `eval_seed` scores every run at one fixed seed instead of at its own,
    which is what reproducing the campaign's recorded numbers requires for
    the two arms whose evaluation consumes randomness. See `_evaluate`.
    """
    out_dir = Path(out_dir)
    # Duplicates in the request are left in rather than filtered out: the
    # completion marker already makes the second visit to a combination a
    # skip, and the CSV is rebuilt from the directories rather than from this
    # list, so neither the training nor the recording can happen twice. A
    # de-duplication pass here would be a second mechanism for a property the
    # marker already guarantees — and an untested one, since no test can
    # distinguish it.
    combos = [(int(seed), method, backbone)
              for seed in seeds
              for method in methods
              for backbone in backbones]
    if not combos:
        raise ValueError('nothing to sweep: seeds, methods and backbones must '
                         'each name at least one value')

    # Check every combination before training any of them. A combination the
    # builder will refuse is worth finding in the first second rather than
    # after the previous ones have spent a day on the GPU.
    for _, method, backbone in combos:
        validate(method, ablation)
        if backbone not in BACKBONES:
            raise ValueError(f'unknown backbone {backbone!r}; choose one of '
                             f'{sorted(BACKBONES)}')

    stamp = detect_git_hash() if git_hash is None else git_hash
    csv_path = out_dir / RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed, method, backbone in combos:
        run_dir = work_dir_for(out_dir, seed, method, backbone, ablation)
        if (run_dir / MARKER).exists():
            continue
        metrics = _train_one(method=method, backbone=backbone,
                             ablation=ablation, seed=seed, recipe=recipe,
                             data_root=data_root, work_dir=run_dir,
                             eval_seed=eval_seed)
        row = _row(seed=seed, flow=FLOW[(method, ablation)], backbone=backbone,
                   stamp=stamp, metrics=metrics)
        # Row first, marker second, CSV last. The marker is the claim that
        # this run is finished, so nothing may make that claim before the
        # evidence for it is on disk.
        _write_atomically(run_dir / METRICS_FILE,
                          lambda handle, row=row: json.dump(row, handle, indent=2))
        (run_dir / MARKER).touch()
        _write_results(csv_path, _completed_rows(out_dir))

    # Also on the way out, so a resume that trains nothing still leaves the
    # CSV present and current (e.g. after it was deleted, or after the run
    # that finished last was interrupted before it could rewrite it).
    _write_results(csv_path, _completed_rows(out_dir))
    return csv_path
