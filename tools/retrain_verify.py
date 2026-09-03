"""Compare runs trained by this package against the recorded distribution.

`tools/replay.py` answers "does this code compute the same thing, given the
same weights?". It cannot answer "does this code *arrive* at the same place,
starting from nothing", because it loads weights somebody else's code created.
Everything that only happens while weights are being made -- initialisation,
the optimiser, the schedule, checkpoint selection, and writing a checkpoint
that can be read back -- is invisible to it. This tool covers that layer:
train from scratch with `chamnet sweep`, then compare the result to the
recorded runs of the same condition.

Training is not run from here. `chamnet sweep` already does it, is already
tested, and takes long enough that scoring has to be separable from it -- one
combination at a time, resumable, and re-readable afterwards. So the two steps
are:

    chamnet sweep --methods hd --backbones resnet18 --seeds 37 --eval-seed 42 \\
                  --data <dataset> --out <runs>
    python tools/retrain_verify.py --runs <runs> --src <recorded>

`--seed 37` / `--eval-seed 42` is not a mixed pair by accident: the campaign
trained at the run's seed and scored every checkpoint at a fixed 42, because
its two scoring processes were never passed a seed. Reproducing a recorded
number therefore needs both values, and for the shuffled arm -- whose
evaluation draws a fresh depth permutation per sample -- it is the difference
between matching and not. See `RECORDED_EVAL_SEED` in tools/replay.py and
verification/README.md.

**The criterion, and why it is not a tolerance.** Training runs
`deterministic=False`, so a fresh run does not reproduce an earlier run's
digits and no fixed tolerance around one earlier run is defensible: on these
conditions the seed-to-seed SD is 0.30-0.71 mIoU and 0.64-1.26 Pillar, so a
tolerance tighter than that fails on correct code, and one loose enough to
pass reliably would be too loose to catch anything. What is available instead
is the recorded *distribution*: ten seeds per condition.

**mIoU is gated against the min-max range of those ten. Pillar is not
gated, and that is a conclusion from measurement rather than a concession.**
The two metrics are not gateable at the same resolution, and the difference
was measured rather than assumed by running the same condition, at the same
seed, more than once:

    bl/resnet18, three runs   mIoU 80.33 80.99 80.75   span 0.66
                              Pillar 78.89 79.65 79.63  span 0.76
    hd/resnet18, three runs   mIoU 81.65 80.56 81.01   span 1.09
                              Pillar 83.70 78.54 83.51  span 5.16

On mIoU the same-seed span (0.66-1.09) is the size of the recorded
seed-to-seed SD (0.55-0.60), so the ten-seed range is a meaningful envelope
for a single fresh run and the gate means something. On Pillar it is not:
hd/resnet18's same-seed span of 5.16 is nearly twice the whole width of that
condition's recorded Pillar range (2.74) and larger than its 95% prediction
interval (5.19) can usefully bound -- one of those three legitimate runs
lands outside *both*. No interval derived from the recorded ten seeds can
gate a single Pillar draw, so this tool reports Pillar against both the
range and the prediction interval and lets the exit code alone.

Why Pillar behaves that way is known and is not a property of this code:
Pillar appears in only 14 of the 45 test images and the top five carry 55%
of its ground-truth pixels, so one image predicted badly moves the aggregate
by points. verification/README.md documents a recorded run sitting at
z = -4.45 on Pillar with its mIoU at z = +0.11, traced to a single image.
The 78.54 above is that same phenomenon in a fresh training run.

Read the mIoU gate for what it is, too. The range of ten draws is not a
tolerance interval: for one fresh exchangeable draw its coverage is
9/11 = 81.8%, so even on mIoU an entirely faithful implementation lands
outside it about one time in five. A failure is a reason to investigate, not
proof of a defect -- and the honest response is to diagnose it, not to widen
the range. The 95% prediction interval for a new draw
(mean +- t(.975, n-1) * sd * sqrt(1 + 1/n)) is reported beside every row as
the interval that does have a stated coverage.

None of this is the strongest evidence available, and it should not be read
as though it were. Because the training seed, the worker count and every
pipeline setting match the recorded run's, a retrain starts from the same
initial weights and sees the same batches in the same order -- so its *loss
curve* can be compared against the recorded run's step by step, and that is
a far tighter check on the training path than any comparison of final
metrics. See docs/VERIFICATION.md.

Two further signals are computed and printed rather than gated, because each
is weak alone and neither should be rationalised away after the fact:

* **Every run deviating the same way.** A systematic defect can sit inside
  every individual range while moving all four conditions in one direction.
  Under a faithful implementation all four agreeing in sign happens 12.5% of
  the time per metric, so this is a prompt to look, not a verdict.
* **A Pillar drop with a normal mIoU.** That is the signature of one class
  failing rather than the model being worse, and it is documented in this
  project: a single test image moved one recorded run's Pillar to z = -4.45
  while its mIoU sat at z = +0.11 (verification/README.md, "Pillar on this
  split rests on a handful of images").
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import math
import os
import statistics
from pathlib import Path

# The combinations retrained, and why each is here. Four, not thirty-six:
# every one is a full training run, and the point is to cover the distinct
# things training can get wrong rather than to re-run the campaign.
#
#   hd/resnet18            the paper's headline arm, on the backbone it quotes
#   bl/resnet18            the same backbone with no depth at all, so a
#                          difference that is really about the dataset or the
#                          schedule shows up on both rather than on one
#   hd/resnet18 shuffled   a control whose *training input* is drawn per
#                          sample, so it also exercises the seeding of the
#                          dataloader workers over a whole run
#   ef/mit_b0              a second method on a second backbone family, whose
#                          entire difference from bl is one widened stem
#                          convolution and how its fourth channel is
#                          initialised -- a thing only training can check
#
# SegNeXt-T is deliberately absent. Its decode head draws random numbers on
# every forward pass, which gives its *evaluation* a reproducible offset
# between one way of driving a test loop and another (verification/README.md).
# That offset is understood, but it is the size of the thing this tool
# measures, and mixing the two would make a training-fidelity result
# unreadable.
COMBOS = (
    ('hd', 'resnet18', None),
    ('bl', 'resnet18', None),
    ('hd', 'resnet18', 'shuffled'),
    ('ef', 'mit_b0', None),
)

#: The seed the retrained runs were trained at, which selects the recorded row
#: they are compared against. Evaluation ran at a different seed -- see the
#: module docstring and `RECORDED_EVAL_SEED` in tools/replay.py.
RUN_SEED = 37

#: (column stem in the recorded CSV and in a run's metrics.json, name in
#: output). Prefixed with a split to make the actual column: `test_mIoU`.
METRICS = (('mIoU', 'mIoU'), ('IoU_pillar', 'pillar'))

#: Both splits are compared and both are written out; one cell is gated.
#:
#: Split: the gate is the test split, because that is the split the published
#: tables report and a criterion has to be about the number being defended.
#: Val is reported rather than dropped for the opposite reason -- it is
#: measured either way, and showing only the split that happens to pass would
#: be selection by split. Read a val row knowing what it is: the checkpoint
#: was *selected* on val mIoU over roughly a hundred evaluations, so that
#: metric is a maximum rather than a draw and its recorded spread is
#: correspondingly narrower (SD 0.19-0.39 against 0.30-0.71 on test). The
#: selection is identical on both sides, so it does not bias the comparison;
#: it does make the val range tighter than the noise it would have to
#: accommodate.
#:
#: Metric: only mIoU is gated. See the module docstring -- the same-seed span
#: on Pillar (measured at 5.16 on hd/resnet18) is wider than anything the
#: recorded ten seeds can bound, so a single Pillar draw is not gateable.
#: Pillar rows still carry both verdicts and a miss is still printed and
#: written to the CSV; it just does not decide the exit code.
SPLITS = ('test', 'val')
GATED_SPLIT = 'test'
GATED_METRIC = 'mIoU'

#: The measurement that retired the Pillar gate, kept beside the decision it
#: justifies. Same condition, same seed, same command, separate processes.
#: Three runs each, so each span is a floor on the band, not the band.
MEASURED_SAME_SEED_SPAN = {
    ('bl', 'resnet18', None): {'mIoU': 0.66, 'pillar': 0.76},
    ('hd', 'resnet18', None): {'mIoU': 1.09, 'pillar': 5.16},
}

#: Student-t 0.975 quantile by degrees of freedom. Only what this tool can
#: need is listed: the recorded campaign ran ten seeds per condition. An
#: unlisted size raises instead of being approximated, so a differently sized
#: recorded set cannot silently get a normal-approximation interval reported
#: as a t interval.
T_975 = {9: 2.2622}

#: A Pillar excursion this many SDs below the recorded mean, while mIoU sits
#: within this many SDs, is the "one class failed" signature.
PILLAR_Z = -2.0
MIOU_Z = -1.0


_REPLAY = None


def _replay_module():
    """Load tools/replay.py for its recorded-work_dir table.

    Imported rather than restated: `WORK` and `EXT_BACKBONES` say where each
    arm's recorded runs and `results_v8.csv` live, and the naming is
    irregular enough that a second copy of it would drift (see the comment on
    `WORK` itself). replay.py has no import-time side effects.

    Loaded once and kept: it pulls torch and mmseg in, which is worth paying
    for a single time and not once per combination.
    """
    global _REPLAY
    if _REPLAY is None:
        path = Path(__file__).with_name('replay.py')
        spec = importlib.util.spec_from_file_location('_chamnet_replay', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _REPLAY = module
    return _REPLAY


def recorded_rows(src: str, method: str, backbone: str,
                  ablation: str | None) -> list[dict]:
    """Every recorded seed's row for one condition, as dicts of strings."""
    replay = _replay_module()
    flow = replay.FLOW[(method, ablation)]
    work, work_ext = replay.WORK[(method, ablation)]
    work = work_ext if backbone in replay.EXT_BACKBONES else work
    path = f'{src}/{work}/results_v8.csv'
    with open(path, newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['flow'] == flow and row['backbone'] == backbone]
    if not rows:
        raise SystemExit(f'{path} has no rows for {flow}/{backbone}')
    return rows


def retrained_row(runs: str, method: str, backbone: str,
                  ablation: str | None, seed: int = RUN_SEED) -> dict:
    """One retrained run's recorded row, read from the sweep's own output.

    `chamnet sweep` writes `metrics.json` beside each run's checkpoints and
    only then marks the run done, so this reads the same numbers the sweep's
    results CSV was rendered from rather than re-deriving them.
    """
    from chamnet.config.combos import FLOW
    path = (Path(runs) / str(seed)
            / f'chamnet_{FLOW[(method, ablation)]}_{backbone}' / 'metrics.json')
    if not path.exists():
        raise SystemExit(
            f'{path} is missing: no finished run for '
            f'{method}/{backbone}/{ablation or "-"} at seed {seed} under '
            f'{runs}. Train it first with `chamnet sweep --seeds {seed} '
            '--eval-seed 42`.')
    return json.loads(path.read_text())


def prediction_interval(values: list[float]) -> tuple[float, float]:
    """95% interval for *one further draw* from the same condition.

    mean +- t(.975, n-1) * sd * sqrt(1 + 1/n) -- the extra 1/n is what makes
    this an interval for a new observation rather than for the mean.
    """
    n = len(values)
    try:
        t = T_975[n - 1]
    except KeyError:
        raise SystemExit(
            f'no t quantile for n={n}; add it to T_975 rather than letting '
            'this fall back to an approximation') from None
    mean = statistics.fmean(values)
    half = t * statistics.stdev(values) * math.sqrt(1 + 1 / n)
    return mean - half, mean + half


def compare(runs: str, src: str) -> list[dict]:
    """One output row per (combination, split, metric)."""
    out = []
    for method, backbone, ablation in COMBOS:
        recorded = recorded_rows(src, method, backbone, ablation)
        retrained = retrained_row(runs, method, backbone, ablation)
        same_seed = [r for r in recorded if int(r['seed']) == RUN_SEED]
        if len(same_seed) != 1:
            raise SystemExit(
                f'{method}/{backbone}: expected exactly one recorded seed '
                f'{RUN_SEED} row, found {len(same_seed)}')
        for split in SPLITS:
            for stem, name in METRICS:
                column = f'{split}_{stem}'
                values = [float(r[column]) for r in recorded]
                mean = statistics.fmean(values)
                sd = statistics.stdev(values)
                got = float(retrained[column])
                seed_37 = float(same_seed[0][column])
                low, high = prediction_interval(values)
                out.append(dict(
                    method=method, backbone=backbone,
                    ablation=ablation or '-', split=split, metric=name,
                    gated=(split == GATED_SPLIT
                           and name == GATED_METRIC),
                    retrain=round(got, 2),
                    recorded_seeds=len(values),
                    recorded_min=round(min(values), 2),
                    recorded_max=round(max(values), 2),
                    recorded_mean=round(mean, 3), recorded_sd=round(sd, 3),
                    in_recorded_range=min(values) <= got <= max(values),
                    z=round((got - mean) / sd, 2),
                    recorded_seed37=round(seed_37, 2),
                    delta_vs_seed37=round(got - seed_37, 2),
                    predict95_lo=round(low, 2), predict95_hi=round(high, 2),
                    in_predict95=low <= got <= high,
                    retrain_best_iter=int(retrained['best_iter']),
                    recorded_best_iter=int(same_seed[0]['best_iter']),
                ))
    return out


def signals(rows: list[dict]) -> list[str]:
    """The two non-gating failure signals, declared in advance of results.

    Computed on the gated split only -- both metrics, since the Pillar
    signal is the whole point of the second one. Counting both splits would
    double every condition and make "all four agree" look like eight
    agreeing.
    """
    rows = [row for row in rows if row.get('split', GATED_SPLIT)
            == GATED_SPLIT]
    found = []
    for _, name in METRICS:
        signs = {row['z'] > 0 for row in rows if row['metric'] == name}
        if len(signs) == 1:
            direction = 'above' if signs == {True} else 'below'
            found.append(
                f'SIGNAL: all {len([r for r in rows if r["metric"] == name])} '
                f'conditions land {direction} the recorded mean on {name}. '
                'Same-direction agreement happens 12.5% of the time by '
                'chance, so this is a prompt to look for something '
                'systematic, not a verdict.')
    by_combo: dict[tuple, dict] = {}
    for row in rows:
        by_combo.setdefault(
            (row['method'], row['backbone'], row['ablation']), {}
        )[row['metric']] = row['z']
    for combo, z in by_combo.items():
        if z.get('pillar', 0) <= PILLAR_Z and z.get('mIoU', 0) >= MIOU_Z:
            found.append(
                f'SIGNAL: {"/".join(combo)} drops on Pillar (z={z["pillar"]}) '
                f'while mIoU is normal (z={z["mIoU"]}). That is the signature '
                'of one class failing rather than of a worse model; on this '
                'split Pillar rests on a handful of images (see '
                'verification/README.md).')
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True,
                    help="a `chamnet sweep --out` directory holding the "
                         'retrained runs, in the campaign layout '
                         '<runs>/<seed>/chamnet_<flow>_<backbone>/')
    ap.add_argument('--src', required=True, metavar='ROOT',
                    help="root holding the recorded runs' work_dirs and their "
                         'results_v8.csv')
    ap.add_argument('--out', default='verification/retrain.csv')
    ap.add_argument('--commit', default='unknown',
                    help='short git commit the retrained runs were produced '
                         'by. Passed in rather than detected: the GPU '
                         'container this runs in has no .git directory, and '
                         'an undated, uncommitted table cannot be cited.')
    ap.add_argument('--note', default='',
                    help='free-text provenance appended to the header '
                         'comment, e.g. the dataset copy used')
    a = ap.parse_args()

    rows = compare(a.runs, a.src)
    directory = os.path.dirname(a.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance = (f'# commit={a.commit} date={stamp} train_seed={RUN_SEED} '
                  f'eval_seed=42 gate=recorded-10-seed-range '
                  f'gated={GATED_SPLIT}/{GATED_METRIC}'
                  + (f' {a.note}' if a.note else ''))
    with open(a.out, 'w', newline='') as handle:
        handle.write(provenance + '\n')
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        verdict = 'in range' if row['in_recorded_range'] else 'OUT OF RANGE'
        if not row['gated']:
            verdict += ' (reported, not gated)'
        print(f'{row["method"]}/{row["backbone"]}/{row["ablation"]:8s} '
              f'{row["split"]:4s} {row["metric"]:6s} {row["retrain"]:6.2f}  '
              f'recorded [{row["recorded_min"]:.2f}, {row["recorded_max"]:.2f}]'
              f'  z={row["z"]:+.2f}  seed37 {row["recorded_seed37"]:.2f} '
              f'({row["delta_vs_seed37"]:+.2f})  {verdict}')
    for line in signals(rows):
        print(line)
    gated = [row for row in rows if row['gated']]
    bad = [row for row in gated if not row['in_recorded_range']]
    print(f'{len(gated) - len(bad)}/{len(gated)} gated '
          f'({GATED_SPLIT} {GATED_METRIC}) values inside the recorded range')
    ungated_bad = [row for row in rows
                   if not row['gated'] and not row['in_recorded_range']]
    for row in ungated_bad:
        print(f'NOTE: {row["method"]}/{row["backbone"]}/{row["ablation"]} '
              f'{row["split"]} {row["metric"]} {row["retrain"]} is outside '
              f'its recorded range [{row["recorded_min"]}, '
              f'{row["recorded_max"]}] (z={row["z"]:+.2f}, '
              f'{"inside" if row["in_predict95"] else "outside"} the 95% '
              'prediction interval). Reported, not gated -- see the module '
              'docstring for the measurement that retired the Pillar gate.')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
