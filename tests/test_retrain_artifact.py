"""The retrain artifact has to satisfy the criterion it reports against.

`verification/retrain.csv` is the only place the retraining layer's result
survives: the greenhouse dataset and the recorded per-run metrics are not
distributed, so nobody outside the training server can regenerate the file,
and a reader has to take the table itself as the evidence. That makes two
things worth pinning down here, neither of which needs a GPU:

* the file says what it is -- which commit, which training seed, which
  evaluation seed -- rather than being an undated snapshot, and it covers
  exactly the combinations `tools/retrain_verify.py` claims to cover;
* its verdict columns agree with its own numbers, and those numbers are
  inside the recorded ranges. A row saying `in_recorded_range=True` beside a
  value outside its own min-max is the one failure mode a hand-edited or
  stale table produces, and it would otherwise be invisible.

The tool's two pieces of arithmetic -- the prediction interval and the
non-gating signals -- are exercised directly, because the CSV only carries
their output and a wrong formula would look like a fact about the runs.
"""
import csv
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'verification' / 'retrain.csv'


def _tool():
    """Import tools/retrain_verify.py without putting tools/ on the path.

    Its module scope is stdlib-only and it registers nothing, so importing it
    has no effect on any registry. `replay.py` (which it reads the recorded
    work_dir table from) is loaded lazily, inside the one function that needs
    the private data, so this import does not pull pandas or torch in.

    Compiled from the source text rather than through `SourceFileLoader`, on
    purpose. That loader consults `__pycache__`, and it validates a cached
    `.pyc` against the source's *size* and its mtime **in whole seconds** --
    so an edit that changes neither (any same-length edit made within the
    same second, which is what a mutation test does) silently runs the
    previous bytecode. Observed while checking these tests: a one-character
    mutation kept passing, and the restored tree kept failing, because both
    runs were reading a stale `.pyc`. A test that can read code other than
    the code on disk is not load-bearing, so this reads the bytes.
    """
    path = ROOT / 'tools' / 'retrain_verify.py'
    module = types.ModuleType('chamnet_retrain_tool')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture(scope='module')
def artifact():
    text = ARTIFACT.read_text().splitlines()
    assert text[0].startswith('#'), (
        f'{ARTIFACT} must open with a provenance comment; it starts '
        f'{text[0]!r}')
    rows = list(csv.DictReader(text[1:]))
    return text[0], rows


def test_provenance_says_which_code_and_which_seeds(artifact):
    """An artifact nobody can regenerate has to carry its own conditions.

    The two seeds in particular: the campaign trained at the run's seed and
    scored at a fixed 42, and a reader who assumes one number for both would
    misread the shuffled arm's row entirely.
    """
    header, _ = artifact
    for field in ('commit=', 'date=', 'train_seed=37', 'eval_seed=42',
                  'gate=recorded-10-seed-range', 'gated=test/mIoU'):
        assert field in header, f'{field!r} missing from {header!r}'
    assert 'commit=unknown' not in header, (
        'the artifact was written without --commit, so it cannot be traced '
        'to the code that produced it')


def test_every_retrained_combination_is_present(artifact):
    """Coverage comes from the tool's own table, not from a second copy of it.

    Restated here, this test would keep passing after someone added a
    combination to `COMBOS` and forgot to re-run the artifact.
    """
    tool = _tool()
    _, rows = artifact
    expected = {(method, backbone, ablation or '-', split, name)
                for method, backbone, ablation in tool.COMBOS
                for split in tool.SPLITS
                for _, name in tool.METRICS}
    got = {(r['method'], r['backbone'], r['ablation'], r['split'], r['metric'])
           for r in rows}
    assert got == expected, (
        f'artifact covers {sorted(got)}, tool claims {sorted(expected)}')


def test_everything_is_reported_and_only_test_miou_is_gated(artifact):
    """Reporting a number and gating on it are separate decisions here.

    Both splits and both metrics are in the file -- showing only the split or
    the metric that happens to pass would be selection dressed as a result.
    The exit code rests on test mIoU alone, because that is the only cell
    whose same-seed spread is smaller than the recorded range it is compared
    against; the `gated` column says so rather than leaving a reader to infer
    it. See tools/retrain_verify.py for the measurement behind that.
    """
    tool = _tool()
    _, rows = artifact
    for row in rows:
        expected = (row['split'] == tool.GATED_SPLIT
                    and row['metric'] == tool.GATED_METRIC)
        assert (row['gated'] == 'True') == expected, (
            f'{row["method"]}/{row["split"]}/{row["metric"]}: '
            f'gated={row["gated"]}, but the gate is '
            f'{tool.GATED_SPLIT}/{tool.GATED_METRIC}')
    assert {r['split'] for r in rows} == set(tool.SPLITS)
    assert {r['metric'] for r in rows} == {name for _, name in tool.METRICS}
    assert sum(1 for r in rows if r['gated'] == 'True') == len(tool.COMBOS)


def test_the_gated_cell_is_the_one_whose_spread_supports_a_gate(artifact):
    """The reason the gate is mIoU-only, asserted against the measurement.

    `MEASURED_SAME_SEED_SPAN` records what repeat runs of the same condition
    at the same seed actually spanned. A gate is only meaningful where that
    span is no wider than the recorded range being compared against; this
    pins that relation for the gated metric and its failure for Pillar, so
    the choice cannot quietly invert later.
    """
    tool = _tool()
    _, rows = artifact
    by_key = {(r['method'], r['backbone'], r['ablation'], r['split'],
               r['metric']): r for r in rows}
    checked = 0
    for (method, backbone, ablation), spans in \
            tool.MEASURED_SAME_SEED_SPAN.items():
        for metric, span in spans.items():
            row = by_key[(method, backbone, ablation or '-',
                          tool.GATED_SPLIT, metric)]
            width = float(row['recorded_max']) - float(row['recorded_min'])
            if metric == tool.GATED_METRIC:
                assert span <= width, (
                    f'{method}/{backbone} {metric}: same-seed span {span} '
                    f'exceeds the recorded range width {width:.2f}, so '
                    'gating on that range is not supportable any more')
            checked += 1
    assert checked == 4
    hd = tool.MEASURED_SAME_SEED_SPAN[('hd', 'resnet18', None)]
    pillar = by_key[('hd', 'resnet18', '-', tool.GATED_SPLIT, 'pillar')]
    width = float(pillar['recorded_max']) - float(pillar['recorded_min'])
    assert hd['pillar'] > width, (
        'the Pillar gate was retired because the same-seed span exceeds the '
        f'recorded range width; that is no longer true ({hd["pillar"]} vs '
        f'{width:.2f}) and the decision should be revisited')


def test_the_range_verdict_agrees_with_the_numbers_beside_it(artifact):
    """`in_recorded_range` is recomputed from the row's own three numbers."""
    _, rows = artifact
    for row in rows:
        low, high = float(row['recorded_min']), float(row['recorded_max'])
        got = float(row['retrain'])
        claimed = row['in_recorded_range'] == 'True'
        assert claimed == (low <= got <= high), (
            f'{row["method"]}/{row["backbone"]}/{row["ablation"]} '
            f'{row["split"]} {row["metric"]}: retrain {got} against '
            f'[{low}, {high}] but the row claims '
            f'in_recorded_range={row["in_recorded_range"]}')


def test_the_prediction_interval_verdict_agrees_too(artifact):
    _, rows = artifact
    for row in rows:
        low, high = float(row['predict95_lo']), float(row['predict95_hi'])
        got = float(row['retrain'])
        claimed = row['in_predict95'] == 'True'
        assert claimed == (low <= got <= high), (
            f'{row["method"]}/{row["backbone"]}/{row["ablation"]} '
            f'{row["split"]} {row["metric"]}: retrain {got} against '
            f'[{low}, {high}] but the row claims '
            f'in_predict95={row["in_predict95"]}')


def test_the_same_seed_delta_is_the_difference_it_claims_to_be(artifact):
    """Reported, not gated -- which is exactly why it needs checking here."""
    _, rows = artifact
    for row in rows:
        expected = float(row['retrain']) - float(row['recorded_seed37'])
        assert abs(float(row['delta_vs_seed37']) - expected) <= 0.005, (
            f'{row["method"]}/{row["split"]}/{row["metric"]}: '
            f'delta_vs_seed37 {row["delta_vs_seed37"]} is not '
            f'{expected:.2f}')


def test_the_gate_is_a_ten_seed_range_on_every_row(artifact):
    """The criterion is only as good as the distribution behind it.

    A row built from fewer seeds would have a narrower range for a reason
    that has nothing to do with the code being verified.
    """
    _, rows = artifact
    for row in rows:
        assert int(row['recorded_seeds']) == 10, (
            f'{row["method"]}/{row["split"]}/{row["metric"]} compares '
            f'against {row["recorded_seeds"]} recorded seeds, not 10')


#: Exactly the rows outside their recorded range in the committed artifact.
#:
#: Not a tolerance and not an allowance: an explicit tripwire. The gate in
#: `tools/retrain_verify.py` reports these as failures and exits non-zero,
#: which is how they should be reported -- but the committed artifact is a
#: fixed table, and what a test can usefully hold is that *these* rows and no
#: others are out. A different row going out, or one of these coming in,
#: fails here and forces whoever regenerated the artifact to say so in
#: docs/VERIFICATION.md rather than letting the change pass unremarked.
#:
#: Why these three are not a defect is measured, not asserted, and is set out
#: in docs/VERIFICATION.md: two same-seed retrains of `bl/resnet18` differ by
#: 0.66 mIoU and 0.76 Pillar, which is the size of the seed-to-seed spread
#: these ranges are drawn from, and the recorded run's own loss curve is
#: reproduced iteration by iteration.
KNOWN_EXCURSIONS = {
    ('bl', 'resnet18', '-', 'test', 'pillar'),
    ('ef', 'mit_b0', '-', 'test', 'pillar'),
    ('hd', 'resnet18', '-', 'val', 'pillar'),
}


def test_exactly_the_documented_rows_are_outside_their_recorded_range(artifact):
    """The tripwire. Equality, deliberately, in both directions.

    A subset check would let a new excursion through; a superset check would
    let a fixed one pass unnoticed. The artifact is a frozen table, so the
    set of rows that miss is a fact about it, and any change to that fact is
    a change the document has to describe.
    """
    _, rows = artifact
    outside = {(r['method'], r['backbone'], r['ablation'], r['split'],
                r['metric'])
               for r in rows if r['in_recorded_range'] != 'True'}
    assert outside == KNOWN_EXCURSIONS, (
        f'outside the recorded range: {sorted(outside)}; '
        f'documented: {sorted(KNOWN_EXCURSIONS)}. Do not edit '
        'KNOWN_EXCURSIONS to match without updating docs/VERIFICATION.md '
        'with why the new row misses.')


def test_the_documented_excursions_are_all_small_and_one_sided(artifact):
    """What makes them diagnosable rather than alarming, pinned numerically.

    Each miss is a fraction of the condition's own seed-to-seed SD outside
    the range, and each is comfortably inside the 95% prediction interval for
    a new draw. An excursion of a different character -- several SDs out, or
    outside the prediction interval too -- would not be covered by the
    diagnosis in docs/VERIFICATION.md, and should not inherit its allowance.
    """
    _, rows = artifact
    for row in rows:
        key = (row['method'], row['backbone'], row['ablation'], row['split'],
               row['metric'])
        if key not in KNOWN_EXCURSIONS:
            continue
        got = float(row['retrain'])
        sd = float(row['recorded_sd'])
        margin = min(abs(got - float(row['recorded_min'])),
                     abs(got - float(row['recorded_max'])))
        assert margin < 0.5 * sd, (
            f'{key} misses by {margin:.2f}, which is {margin / sd:.2f} of '
            f'its recorded SD {sd:.2f} -- larger than the diagnosis covers')
        assert row['in_predict95'] == 'True', (
            f'{key} is outside the 95% prediction interval as well as the '
            'recorded range; that is not the excursion this is documenting')


def test_prediction_interval_is_for_a_new_draw_not_for_the_mean():
    """mean +- t * sd * sqrt(1 + 1/n), with the 1/n a confidence interval lacks.

    Ten values whose squared deviations from a mean of 10 sum to 10, so the
    sample SD is sqrt(10/9) = 1.054093. With t(.975, 9) = 2.2622 and
    sqrt(1 + 1/10) = 1.048809 the half-width is 2.5010, computed by hand
    here rather than by re-running the formula under test.

    A confidence interval for the *mean* would use t * sd / sqrt(n) = 0.7541
    -- narrower by sqrt(n + 1) = 3.3166 -- and would answer a different
    question: how well ten runs pin down the condition's average, not where
    an eleventh run should land.
    """
    tool = _tool()
    values = [10 - 1.5, 10 - 1.5, 10 - 0.5, 10 - 0.5, 10.0,
              10.0, 10 + 0.5, 10 + 0.5, 10 + 1.5, 10 + 1.5]
    assert abs(sum(values) / len(values) - 10.0) < 1e-9
    low, high = tool.prediction_interval(values)
    half = (high - low) / 2
    assert abs(half - 2.5010) < 0.001, half
    assert abs(half / 0.7541 - 3.3166) < 0.01, half / 0.7541


def test_prediction_interval_refuses_a_sample_size_it_has_no_quantile_for():
    """Rather than approximating one and reporting it as a t interval."""
    tool = _tool()
    with pytest.raises(SystemExit, match='no t quantile'):
        tool.prediction_interval([1.0, 2.0, 3.0])


def _row(metric, z, method='hd', ablation='-'):
    return dict(method=method, backbone='resnet18', ablation=ablation,
                metric=metric, z=z)


def test_the_same_direction_signal_needs_every_condition_to_agree():
    tool = _tool()
    agree = [_row('mIoU', z, method=m)
             for m, z in zip('abcd', (0.4, 0.1, 1.2, 0.8))]
    assert any('all 4 conditions land above' in line
               for line in tool.signals(agree))
    one_dissents = [_row('mIoU', z, method=m)
                    for m, z in zip('abcd', (0.4, -0.1, 1.2, 0.8))]
    assert tool.signals(one_dissents) == []


def test_a_pillar_drop_with_a_normal_miou_is_flagged():
    """The z = -4.45 / +0.11 pattern this project has already met once."""
    tool = _tool()
    flagged = tool.signals([_row('mIoU', 0.11), _row('pillar', -4.45)])
    assert any('drops on Pillar' in line for line in flagged), flagged
    # Both metrics down together is a worse model, not a failed class, and
    # must not be reported as the same thing.
    assert not any('drops on Pillar' in line for line in
                   tool.signals([_row('mIoU', -2.6), _row('pillar', -4.45)]))
