#!/usr/bin/env python3
"""Pick the representative seed, from validation metrics only.

The campaign trained every condition at ten seeds (31-40) and the published
tables are ten-seed means. A few things still need *one* seed: the qualitative
figures, the checkpoints in `docs/MODEL_ZOO.md`, `tools/replay.py`'s
comparison against the recorded per-run metrics, and the retraining check in
`docs/VERIFICATION.md`. All four use seed 37. This tool is why -- it recomputes
that choice from a stated rule so it can be checked instead of taken on trust.

**The rule reads validation metrics and nothing else, and that is the whole
point.** Choosing a seed by how good its test numbers look is choosing a
result; the literature on this is blunt about it (Nunez et al., "Sources of
Irreproducibility in Machine Learning: A Review", 2021), and so is the
arithmetic here. Seed 37's `hd/convnext_atto` run sits 4.45 SD below the other
nine seeds on *test* Pillar -- traced in `verification/README.md` to one badly
predicted image out of the fourteen in the split that contain pillar at all --
while the same run sits at a routine 1.69 SD below on *validation* Pillar. A
test-aware rule would have had that single image as an input. So this tool
cannot read a test column: `read_validation` discards every one of them before
any number is used, and naming a non-`val_` metric is an error rather than a
lookup.

The rule, in full, stated before any data is read:

1. **Scope: the four training methods on the four backbones -- 16
   configurations.** Not the control arms. They are diagnostics of one method,
   and putting all five HD variants in would weight HD five times against BL's
   one; the four methods are what the published comparison is between.
2. **Centre: each configuration's own median**, not its mean, so that one
   anomalous run cannot drag the target the other nine are measured against.
3. **Scale: that configuration's SD across the ten seeds**, so configurations
   with different spreads contribute comparably. `z = (x - median) / sd`.
4. **Per criterion, score each seed by `mean |z|`** over the configurations in
   scope, and rank the seeds by it, 1 = most typical. Ties take the average
   rank.
5. **Four criteria**: {all 16 configurations, ResNet-18 only} x {val mIoU, val
   Pillar IoU}. ResNet-18 gets its own criterion because it is the backbone the
   published headline numbers quote; Pillar gets one because it is the class
   the paper argues about and it is the noisiest.
6. **Select on the worst of the four ranks**, tie-broken by the mean of them
   and then by the lower seed number. Worst-rank rather than average-rank on
   purpose: the seed has to be typical under every criterion, and an average
   would let a seed be first on three criteria and last on the fourth.

On the recorded campaign this selects **seed 37**, whose worst rank is 2 and
whose mean rank is 1.50 -- first or second on all four criteria. No other seed
has a worst rank better than 5. `--expect 37` turns that into a check with an
exit code.

The identical rule reading test columns instead of validation ones selects
seed 35, and seed 37 drops to a worst rank of 6. That is not an argument for
either seed; it is the measurement that says the choice is not invariant to
which split it is allowed to see, which is exactly why the split it is allowed
to see has to be the one nothing is being claimed about.

Three honest limits, because the alternative is a rule that looks stronger
than it is.

*The margins are small.* First and second place on each of the four criteria
are separated by 0.04 to 0.13 of an SD.

*Steps 2 and 6 do not matter; step 1 does.* Swapping the median for the mean,
or the worst rank for the average of the ranks, selects seed 37 in all four
combinations -- so the choice is not an artefact of those two decisions. The
*scope* is a different matter: run the same rule over all nine arms x four
backbones, controls included, and it selects seed 33 with 37 fifth. Step 1 is
therefore load-bearing and has to stand on its own reasoning rather than on
its result, which is why that reasoning is stated there and not here. What
seed 37 is, precisely, is the most typical seed of the four-method comparison
the paper makes -- not of the ablation grid.

*Nobody outside the training server can run this.* The recorded per-run CSV is
not distributed. What is checkable is the rule, which is all of it above and
all of the code below.

Usage (the CSV is the campaign's own per-run metrics file):

    python tools/select_seed.py --results <results_v8.csv> [--expect 37]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from chamnet.config.combos import FLOW

#: The four training methods, on the four backbones: 16 configurations.
METHODS = ('bl', 'ef', 'sd', 'hd')
BACKBONES = ('resnet18', 'mit_b0', 'segnext_t', 'convnext_atto')

#: The metrics the criteria are built from. Every one of these must name a
#: validation column; `read_validation` keeps no other kind, so a test column
#: here fails loudly rather than being read.
METRICS = ('val_mIoU', 'val_IoU_pillar')

#: (label, backbones in scope) for each half of the criterion grid.
SCOPES = (('all16', BACKBONES), ('resnet18', ('resnet18',)))

#: The seeds the campaign ran. An incomplete configuration is an error rather
#: than a shorter series quietly given the same weight as a full one.
SEEDS = tuple(range(31, 41))


def read_validation(path: str | Path) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    """`{(method, backbone): {seed: {val_column: value}}}` -- validation only.

    Every column that is not a `val_` column is dropped here, before any value
    is looked at. That is the mechanism, not a convention: nothing downstream
    has a test number available to be tempted by.
    """
    for metric in METRICS:
        if not metric.startswith('val_'):
            raise SystemExit(
                f'{metric!r} is not a validation column. This tool selects a '
                'seed from validation metrics only -- see its module '
                'docstring for why.')

    wanted = {FLOW[(method, None)]: method for method in METHODS}
    table: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            method = wanted.get(row['flow'])
            if method is None or row['backbone'] not in BACKBONES:
                continue
            values = {key: float(value) for key, value in row.items()
                      if key.startswith('val_')}
            seed = int(row['seed'])
            per_seed = table.setdefault((method, row['backbone']), {})
            if seed in per_seed:
                raise SystemExit(
                    f'{path}: {method}/{row["backbone"]} has more than one '
                    f'row for seed {seed}')
            per_seed[seed] = values

    expected = {(m, b) for m in METHODS for b in BACKBONES}
    if set(table) != expected:
        missing = sorted(expected - set(table))
        raise SystemExit(f'{path} is missing configurations: {missing}')
    for key, per_seed in table.items():
        if tuple(sorted(per_seed)) != SEEDS:
            raise SystemExit(
                f'{path}: {key[0]}/{key[1]} has seeds '
                f'{tuple(sorted(per_seed))}, expected {SEEDS}')
    return table


def scores(table, backbones, metric: str) -> dict[int, float]:
    """`mean |z|` per seed for one criterion, z about each configuration's median."""
    total = {seed: 0.0 for seed in SEEDS}
    configurations = [key for key in table if key[1] in backbones]
    for key in configurations:
        per_seed = table[key]
        try:
            values = [per_seed[seed][metric] for seed in SEEDS]
        except KeyError:
            raise SystemExit(
                f'{metric!r} is not present in the validation columns of '
                f'{key[0]}/{key[1]}. Only validation columns are read.'
            ) from None
        median = statistics.median(values)
        sd = statistics.stdev(values)
        if sd == 0:
            raise SystemExit(
                f'{key[0]}/{key[1]} has zero spread on {metric}; the rule '
                'divides by it')
        for seed in SEEDS:
            total[seed] += abs(per_seed[seed][metric] - median) / sd
    return {seed: value / len(configurations) for seed, value in total.items()}


def ranks(scores_by_seed: dict[int, float]) -> dict[int, float]:
    """1 = most typical. Ties share the average of the ranks they span."""
    ordered = sorted(scores_by_seed, key=lambda seed: scores_by_seed[seed])
    out: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while (stop + 1 < len(ordered)
               and scores_by_seed[ordered[stop + 1]] == scores_by_seed[ordered[index]]):
            stop += 1
        shared = statistics.fmean(range(index + 1, stop + 2))
        for position in range(index, stop + 1):
            out[ordered[position]] = shared
        index = stop + 1
    return out


def criteria(table) -> dict[str, dict[int, float]]:
    """`{criterion label: {seed: rank}}` for the four criteria of the rule."""
    return {f'{label}/{metric}': ranks(scores(table, backbones, metric))
            for label, backbones in SCOPES for metric in METRICS}


def select(table) -> tuple[int, dict[str, dict[int, float]]]:
    """The representative seed: lowest worst rank, then lowest mean rank."""
    by_criterion = criteria(table)
    def key(seed: int):
        row = [by_criterion[name][seed] for name in by_criterion]
        return (max(row), statistics.fmean(row), seed)
    return min(SEEDS, key=key), by_criterion


def report(table) -> tuple[int, str]:
    seed, by_criterion = select(table)
    names = list(by_criterion)
    header = ['seed'] + names + ['worst', 'mean']
    rows = []
    for candidate in SEEDS:
        row = [by_criterion[name][candidate] for name in names]
        rows.append([str(candidate)] + [f'{value:g}' for value in row]
                    + [f'{max(row):g}', f'{statistics.fmean(row):.2f}'])
    rows.sort(key=lambda row: (float(row[-2]), float(row[-1]), int(row[0])))
    widths = [max(len(cell) for cell in column)
              for column in zip(header, *rows)]
    lines = ['  '.join(cell.rjust(width) for cell, width in zip(header, widths))]
    lines += ['  '.join(cell.rjust(width) for cell, width in zip(row, widths))
              for row in rows]
    lines.append('')
    lines.append(f'representative seed: {seed}  '
                 '(lowest worst rank across the four criteria; 1 = most '
                 'typical)')
    return seed, '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Select the representative seed from validation metrics.')
    parser.add_argument('--results', required=True,
                        help="the campaign's per-run metrics CSV")
    parser.add_argument('--expect', type=int, default=None,
                        help='exit non-zero unless the rule selects this seed')
    args = parser.parse_args(argv)

    seed, text = report(read_validation(args.results))
    print(text)
    if args.expect is not None and seed != args.expect:
        print(f'\nFAIL: the rule selects seed {seed}, not {args.expect}.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
