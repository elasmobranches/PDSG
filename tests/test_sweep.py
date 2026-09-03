"""Tests for the resumable sweep.

What is and is not covered here. Everything except one function is exercised
for real: the completion-marker logic, the CSV rebuild, the schema, the flow
vocabulary, the row assembly, the checkpoint selection, the metric-name
translation, the git stamp and the CLI wiring. The exception is `_train_one`,
which trains a network and scores it — that needs a GPU and the greenhouse
dataset, so these tests replace it with a recorder and assert on what the
sweep does around it. `_evaluate` and the `Runner` calls inside `_train_one`
are therefore *not* covered by this file; `tests/test_smoke.py` is what checks
that every combination's config actually builds, loads data and trains a step.

`_metrics_of` sits on the boundary and is covered against the real metric
class rather than against a hand-written dict, because the one thing a mocked
trainer can never catch is the evaluator and the CSV disagreeing about what a
metric is called.
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from chamnet import sweep as sweep_module
from chamnet.config.builder import build_config
from chamnet.config.combos import FLOW, VALID
from chamnet.metrics import IoUMetricWithPerClass
from chamnet.sweep import (CLASSES, FIELDNAMES, MARKER, METRIC_KEYS, METRICS_FILE,
                           SUMMARY_KEYS, detect_git_hash, parse_seeds,
                           run_sweep, work_dir_for)

# The header of a real campaign results CSV, copied from one of them verbatim.
# The release claims this schema so the analysis written against those files
# keeps working; the claim is only worth something if it is checked against
# the actual bytes rather than against a description of them.
RECORDED_HEADER = (
    'run,flow,backbone,seed,git_hash,best_iter,'
    'val_IoU_background,val_IoU_chamoe,val_IoU_heatpipe,val_IoU_path,'
    'val_IoU_pillar,val_IoU_topdownfarm,val_IoU_ceiling,val_IoU_duct,'
    'val_aAcc,val_mIoU,val_mAcc,val_mDice,val_mFscore,val_mPrecision,'
    'val_mRecall,'
    'test_IoU_background,test_IoU_chamoe,test_IoU_heatpipe,test_IoU_path,'
    'test_IoU_pillar,test_IoU_topdownfarm,test_IoU_ceiling,test_IoU_duct,'
    'test_aAcc,test_mIoU,test_mAcc,test_mDice,test_mFscore,test_mPrecision,'
    'test_mRecall')

# The nine arms' recorded names, written out rather than read from FLOW: this
# is the fact the release has to agree with the campaign about, so taking it
# from the table under test would assert nothing. Confirmed against the `flow`
# column of the campaign's own CSVs.
RECORDED_FLOWS = {
    ('bl', None): 'baseline',
    ('ef', None): 'proposed',
    ('sd', None): 'dual',
    ('hd', None): 'dual_plus',
    ('ef', 'shuffled'): 'proposed_shuffled',
    ('hd', 'shuffled'): 'dual_plus_shuffled',
    ('hd', 'rgb'): 'dual_plus_rgb',
    ('hd', 'nogate'): 'dual_plus_nogate',
    ('hd', 'bigate'): 'dual_plus_bigate',
}

STAMP = 'abcdef123456'


def metrics_payload(nth: int) -> dict:
    """A complete `_train_one` result whose every number is distinct.

    Distinct on purpose: a row assembled with the val and test dicts swapped,
    or with one class's IoU written into another's column, would look perfectly
    healthy if every value were the same placeholder.
    """
    def split(offset):
        out = {f'IoU_{c}': offset + i for i, c in enumerate(CLASSES)}
        out.update({k: offset + 50 + i for i, k in enumerate(SUMMARY_KEYS)})
        return out

    return dict(best_iter=100 * nth, val=split(1000 * nth),
                test=split(1000 * nth + 500))


def read_rows(csv_path) -> list[dict]:
    with Path(csv_path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def identify(rows) -> list[tuple]:
    return [(int(r['seed']), r['flow'], r['backbone']) for r in rows]


@pytest.fixture
def trained(monkeypatch):
    """Replace the one GPU-bound function; collect what it was asked to train."""
    calls = []

    def fake_train_one(**kwargs):
        calls.append((kwargs['seed'], kwargs['method'], kwargs['backbone'],
                      kwargs['ablation']))
        Path(kwargs['work_dir']).mkdir(parents=True, exist_ok=True)
        return metrics_payload(len(calls))

    monkeypatch.setattr(sweep_module, '_train_one', fake_train_one)
    return calls


def sweep(tmp_path, methods, backbones, seeds, **extra):
    return run_sweep(recipe='quick', methods=methods, backbones=backbones,
                     seeds=seeds, data_root='/dataset-not-read-by-these-tests',
                     out_dir=tmp_path, git_hash=STAMP, **extra)


# --------------------------------------------------------------------------
# resumability
# --------------------------------------------------------------------------

def test_a_second_identical_sweep_trains_nothing(tmp_path, trained):
    """The property the campaign depended on: relaunching is free."""
    args = dict(methods=['bl', 'hd'], backbones=['resnet18'], seeds=[31, 32])
    csv_path = sweep(tmp_path, **args)
    assert trained == [(31, 'bl', 'resnet18', None), (31, 'hd', 'resnet18', None),
                       (32, 'bl', 'resnet18', None), (32, 'hd', 'resnet18', None)]
    first = csv_path.read_bytes()

    trained.clear()
    assert sweep(tmp_path, **args) == csv_path
    assert trained == []
    # Not merely "no training happened": the file is also unchanged, so the
    # second sweep neither re-stamped nor re-ordered nor appended anything.
    assert csv_path.read_bytes() == first
    assert identify(read_rows(csv_path)) == [
        (31, 'baseline', 'resnet18'), (31, 'dual_plus', 'resnet18'),
        (32, 'baseline', 'resnet18'), (32, 'dual_plus', 'resnet18')]


def test_a_resume_trains_exactly_the_unfinished_combinations(tmp_path, trained):
    """Some done, some not — assert which, not how many.

    The half-and-half case is the one a real resume is always in, and a
    skip rule that is subtly wrong (keyed on the seed only, or on the method
    only) still passes a test that counts calls.
    """
    sweep(tmp_path, methods=['bl'], backbones=['resnet18', 'mit_b0'], seeds=[31])
    assert len(trained) == 2
    trained.clear()

    csv_path = sweep(tmp_path, methods=['bl', 'hd'],
                     backbones=['resnet18', 'mit_b0'], seeds=[31, 32])
    assert trained == [
        (31, 'hd', 'resnet18', None), (31, 'hd', 'mit_b0', None),
        (32, 'bl', 'resnet18', None), (32, 'bl', 'mit_b0', None),
        (32, 'hd', 'resnet18', None), (32, 'hd', 'mit_b0', None)]
    assert identify(read_rows(csv_path)) == [
        (31, 'baseline', 'mit_b0'), (31, 'baseline', 'resnet18'),
        (31, 'dual_plus', 'mit_b0'), (31, 'dual_plus', 'resnet18'),
        (32, 'baseline', 'mit_b0'), (32, 'baseline', 'resnet18'),
        (32, 'dual_plus', 'mit_b0'), (32, 'dual_plus', 'resnet18')]


def test_a_crashed_run_is_not_marked_done(tmp_path, monkeypatch):
    """A run that dies partway must be re-run, and must leave no row.

    Ordering is what makes this hold: the marker goes down only after the
    row is on disk. Written the other way round, the crashed combination
    would be skipped forever and its columns would stay empty in every
    table built from the file — and nothing would ever say so.
    """
    calls = []

    def crash_on_mit(**kwargs):
        calls.append((kwargs['seed'], kwargs['method'], kwargs['backbone']))
        Path(kwargs['work_dir']).mkdir(parents=True, exist_ok=True)
        if kwargs['backbone'] == 'mit_b0':
            raise RuntimeError('CUDA out of memory')
        return metrics_payload(len(calls))

    monkeypatch.setattr(sweep_module, '_train_one', crash_on_mit)
    backbones = ['resnet18', 'mit_b0', 'segnext_t']
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        sweep(tmp_path, methods=['bl'], backbones=backbones, seeds=[31])

    # It stopped at the failure rather than carrying on past it.
    assert calls == [(31, 'bl', 'resnet18'), (31, 'bl', 'mit_b0')]
    done = work_dir_for(tmp_path, 31, 'bl', 'resnet18')
    crashed = work_dir_for(tmp_path, 31, 'bl', 'mit_b0')
    assert (done / MARKER).exists() and (done / METRICS_FILE).exists()
    assert not (crashed / MARKER).exists(), 'a crashed run was marked complete'
    assert not (crashed / METRICS_FILE).exists()

    csv_path = tmp_path / 'results.csv'
    assert identify(read_rows(csv_path)) == [(31, 'baseline', 'resnet18')]
    finished_row = read_rows(csv_path)[0]

    # Resume: exactly the two that did not finish, and no duplicate of the one
    # that did.
    calls.clear()

    def succeed(**kwargs):
        calls.append((kwargs['seed'], kwargs['method'], kwargs['backbone']))
        Path(kwargs['work_dir']).mkdir(parents=True, exist_ok=True)
        return metrics_payload(9)

    monkeypatch.setattr(sweep_module, '_train_one', succeed)
    sweep(tmp_path, methods=['bl'], backbones=backbones, seeds=[31])
    assert calls == [(31, 'bl', 'mit_b0'), (31, 'bl', 'segnext_t')]
    rows = read_rows(csv_path)
    assert identify(rows) == [(31, 'baseline', 'mit_b0'),
                              (31, 'baseline', 'resnet18'),
                              (31, 'baseline', 'segnext_t')]
    assert [r for r in rows if r['backbone'] == 'resnet18'] == [finished_row], (
        'the run that had already finished was rewritten by the resume')


def test_a_torn_results_csv_is_rebuilt_rather_than_appended_to(tmp_path, trained):
    """The CSV is a rendering of the finished runs, so damage to it is repaired.

    An append-only file cannot recover from this: whatever half-line an
    interruption left behind stays in the middle of the file, and every later
    row lands after it.
    """
    args = dict(methods=['bl'], backbones=['resnet18', 'mit_b0'], seeds=[31])
    csv_path = sweep(tmp_path, **args)
    intact = csv_path.read_bytes()

    csv_path.write_bytes(intact[:len(intact) - 40] + b'31,baseline,resn')
    trained.clear()
    sweep(tmp_path, **args)
    assert trained == []
    assert csv_path.read_bytes() == intact

    csv_path.unlink()
    sweep(tmp_path, **args)
    assert trained == []
    assert csv_path.read_bytes() == intact


def test_resuming_a_narrower_sweep_keeps_the_other_arms_rows(tmp_path, trained):
    """Rebuilding must not shorten the file to whatever was asked for today.

    The campaign accumulated one results file per work root over many separate
    invocations — the four main arms first, the control arms weeks later. A
    rebuild scoped to the current request would delete the earlier arms' rows
    every time.
    """
    csv_path = sweep(tmp_path, methods=['bl', 'hd'], backbones=['resnet18'],
                     seeds=[31])
    everything = csv_path.read_bytes()

    trained.clear()
    sweep(tmp_path, methods=['bl'], backbones=['resnet18'], seeds=[31])
    assert trained == []
    assert csv_path.read_bytes() == everything
    assert identify(read_rows(csv_path)) == [(31, 'baseline', 'resnet18'),
                                             (31, 'dual_plus', 'resnet18')]


def test_a_marker_without_its_metrics_file_is_refused(tmp_path, trained):
    """Only this sweep's own markers mean "finished, and here is the row".

    A directory carrying a marker but no row cannot be rendered into the CSV
    and cannot be re-run either (the marker says not to), so it is a silent
    hole by construction. Refusing names the directory and says which file to
    delete instead.
    """
    stale = work_dir_for(tmp_path, 31, 'bl', 'resnet18')
    stale.mkdir(parents=True)
    (stale / MARKER).touch()

    with pytest.raises(RuntimeError, match='marked complete'):
        sweep(tmp_path, methods=['bl'], backbones=['resnet18'], seeds=[31])
    assert trained == []

    (stale / METRICS_FILE).write_text('{"run": 31}')   # present but not a full row
    with pytest.raises(ValueError, match='columns have no value'):
        sweep(tmp_path, methods=['bl'], backbones=['resnet18'], seeds=[31])


def test_the_same_combination_asked_for_twice_is_run_and_recorded_once(
        tmp_path, trained):
    csv_path = sweep(tmp_path, methods=['bl', 'bl'], backbones=['resnet18'],
                     seeds=[31, 31])
    assert trained == [(31, 'bl', 'resnet18', None)]
    assert identify(read_rows(csv_path)) == [(31, 'baseline', 'resnet18')]


def test_an_invalid_arm_stops_the_sweep_before_any_training(tmp_path, trained):
    """Refuse in the first second, not after the earlier arms have run.

    `--methods hd,bl --ablation shuffled` is a plausible typo and its first
    combination is valid, so a sweep that validated lazily would train
    hd/shuffled for hours and then refuse bl/shuffled.
    """
    with pytest.raises(ValueError, match='no depth input to shuffle'):
        sweep(tmp_path, methods=['hd', 'bl'], backbones=['resnet18'],
              seeds=[31], ablation='shuffled')
    assert trained == []

    with pytest.raises(ValueError, match='unknown backbone'):
        sweep(tmp_path, methods=['bl'], backbones=['resnet18', 'resnet101'],
              seeds=[31])
    assert trained == []

    with pytest.raises(ValueError, match='at least one value'):
        sweep(tmp_path, methods=[], backbones=['resnet18'], seeds=[31])
    assert trained == []


# --------------------------------------------------------------------------
# the recorded schema
# --------------------------------------------------------------------------

def test_fieldnames_are_the_recorded_schema():
    assert FIELDNAMES == RECORDED_HEADER.split(',')


def test_results_csv_header_is_the_recorded_schema(tmp_path, trained):
    csv_path = sweep(tmp_path, methods=['bl'], backbones=['resnet18'],
                     seeds=[31])
    assert csv_path.name == 'results.csv'
    assert csv_path.read_text().splitlines()[0] == RECORDED_HEADER


def test_row_carries_this_runs_own_numbers(tmp_path, monkeypatch):
    """Assert the values, not that a row appeared.

    The numbers are one real recorded row (run 31, baseline/resnet18), so a
    val/test mix-up or an off-by-one across the eight class columns shows up
    as a wrong number here rather than as a row that merely exists.
    """
    val = dict(zip(METRIC_KEYS, [79.3, 68.17, 68.05, 91.15, 76.48, 97.42,
                                 89.4, 89.86, 96.23, 82.48, 91.85, 90.04,
                                 90.04, 88.5, 91.85]))
    test = dict(zip(METRIC_KEYS, [74.95, 63.23, 70.25, 90.78, 79.18, 97.79,
                                  87.3, 82.3, 96.35, 80.72, 90.8, 88.95,
                                  88.95, 87.4, 90.8]))
    monkeypatch.setattr(sweep_module, '_train_one',
                        lambda **kw: dict(best_iter=1620, val=val, test=test))

    csv_path = sweep(tmp_path, methods=['bl'], backbones=['resnet18'],
                     seeds=[31])
    row, = read_rows(csv_path)
    assert row['run'] == '31' and row['seed'] == '31'
    assert row['flow'] == 'baseline' and row['backbone'] == 'resnet18'
    assert row['git_hash'] == STAMP
    assert row['best_iter'] == '1620'
    assert row['val_IoU_background'] == '79.3'
    assert row['val_IoU_pillar'] == '76.48'
    assert row['val_mIoU'] == '82.48'
    assert row['val_mRecall'] == '91.85'
    assert row['test_IoU_pillar'] == '79.18'
    assert row['test_aAcc'] == '96.35'
    assert row['test_mIoU'] == '80.72'
    # The per-run copy beside the checkpoints is the whole row, not a summary:
    # it is what a later resume renders the CSV from, so anything it drops is
    # gone for good.
    expected = dict(run=31, flow='baseline', backbone='resnet18', seed=31,
                    git_hash=STAMP, best_iter=1620)
    expected.update({f'val_{k}': v for k, v in val.items()})
    expected.update({f'test_{k}': v for k, v in test.items()})
    beside = work_dir_for(tmp_path, 31, 'bl', 'resnet18') / METRICS_FILE
    assert json.loads(beside.read_text()) == expected
    assert sorted(expected) == sorted(FIELDNAMES)


def test_a_run_missing_a_metric_is_neither_recorded_nor_marked(
        tmp_path, monkeypatch):
    """A half-filled row would freeze the gap: marked done, never re-run."""
    payload = metrics_payload(1)
    del payload['val']['IoU_pillar']
    monkeypatch.setattr(sweep_module, '_train_one', lambda **kw: payload)

    with pytest.raises(ValueError, match='val_IoU_pillar'):
        sweep(tmp_path, methods=['hd'], backbones=['resnet18'], seeds=[31])
    run_dir = work_dir_for(tmp_path, 31, 'hd', 'resnet18')
    assert not (run_dir / MARKER).exists()
    assert not (run_dir / METRICS_FILE).exists()

    payload = metrics_payload(1)
    payload['best_iter'] = None
    monkeypatch.setattr(sweep_module, '_train_one', lambda **kw: payload)
    with pytest.raises(ValueError, match='best_iter'):
        sweep(tmp_path, methods=['hd'], backbones=['resnet18'], seeds=[31])


# --------------------------------------------------------------------------
# the campaign's flow vocabulary
# --------------------------------------------------------------------------

@pytest.mark.parametrize('method,ablation',
                         sorted(RECORDED_FLOWS, key=str))
def test_each_arm_uses_the_campaigns_name(method, ablation, tmp_path, trained):
    """The `flow` column and the work_dir name, for all nine arms.

    Both have to match what the campaign wrote, or new rows cannot be pooled
    with the recorded ones and `tools/replay.py` cannot find a checkpoint the
    sweep produced.
    """
    flow = RECORDED_FLOWS[(method, ablation)]
    expected = tmp_path / '37' / f'chamnet_{flow}_mit_b0'
    assert work_dir_for(tmp_path, 37, method, 'mit_b0', ablation) == expected

    csv_path = sweep(tmp_path, methods=[method], backbones=['mit_b0'],
                     seeds=[37], ablation=ablation)
    row, = read_rows(csv_path)
    assert row['flow'] == flow
    assert (expected / MARKER).exists()


def test_the_flow_table_covers_exactly_the_valid_arms():
    assert set(FLOW) == VALID
    assert FLOW == RECORDED_FLOWS


def test_the_replay_takes_its_flow_names_from_the_shared_table():
    """One definition, so a rename cannot land in the sweep and miss the replay.

    Checked by absence: the replay must not contain a flow name of its own.
    Its own table used to carry them, and the sweep would then have been a
    second copy.
    """
    source = (Path(__file__).resolve().parents[1] / 'tools' / 'replay.py'
              ).read_text()
    assert 'from chamnet.config.combos import FLOW' in source
    restated = [flow for flow in RECORDED_FLOWS.values()
                if f"'{flow}'" in source]
    assert restated == [], (
        f'tools/replay.py restates flow names {restated} instead of reading '
        'them from chamnet.config.combos.FLOW')


def test_hd_control_arms_get_distinct_default_work_dirs():
    """Keyed by method alone, all five HD arms shared one default directory.

    Two of them run without `--out` would then have the second overwrite the
    first's checkpoints. The sweep always passes an explicit work_dir, so this
    is about `chamnet train`, and it is the same table's doing.
    """
    arms = [None, 'nogate', 'bigate', 'rgb', 'shuffled']
    dirs = [build_config(method='hd', backbone='resnet18', ablation=arm,
                         recipe='quick', seed=31).work_dir for arm in arms]
    assert dirs == ['runs/dual_plus_resnet18_31',
                    'runs/dual_plus_nogate_resnet18_31',
                    'runs/dual_plus_bigate_resnet18_31',
                    'runs/dual_plus_rgb_resnet18_31',
                    'runs/dual_plus_shuffled_resnet18_31']


# --------------------------------------------------------------------------
# pieces of a real run that can be checked without a GPU
# --------------------------------------------------------------------------

def test_metric_names_survive_the_trip_from_the_evaluator_to_the_columns():
    """Close the loop between the metric class and the CSV's column names.

    The evaluator emits `IoU.pillar` and the column is `val_IoU_pillar`; the
    translation between them is the one link a mocked trainer can never
    exercise, and getting it wrong writes a file whose eight per-class columns
    are empty while every aggregate looks right. So run the real metric on
    hand-made pixel counts and translate its actual output.
    """
    metric = IoUMetricWithPerClass(iou_metrics=['mIoU', 'mDice', 'mFscore'])
    metric.dataset_meta = {'classes': list(CLASSES)}
    # Class i predicted and labelled 100 px, of which (i + 1) * 10 intersect.
    intersect = torch.tensor([(i + 1) * 10.0 for i in range(8)])
    total = torch.full((8,), 100.0)
    union = total + total - intersect
    raw = metric.compute_metrics([(intersect, union, total, total)])

    assert raw['IoU.pillar'] == 33.33      # 50 / (100 + 100 - 50)
    assert raw['IoU.background'] == 5.26   # 10 / (100 + 100 - 10)
    row = sweep_module._metrics_of(raw)
    assert sorted(row) == sorted(METRIC_KEYS), (
        'the evaluator and the CSV schema disagree about the metric names')
    assert row['IoU_pillar'] == raw['IoU.pillar']
    assert row['IoU_background'] == raw['IoU.background']
    assert row['mIoU'] == pytest.approx(raw['mIoU'], abs=0.005)

    # And they stop being numpy's here. IoUMetric hands back float32 scalars:
    # json.dump refuses them outright, and a bare float() widens a value that
    # is already rounded to two decimals into 96.3499984741211. Both faults
    # surface only after a run has trained, so they are pinned here.
    assert any(type(v) is not float for v in raw.values()), (
        'this check assumes the evaluator returns numpy scalars; if it no '
        'longer does, it is asserting nothing')
    assert all(type(v) is float for v in row.values()), (
        {k: type(v).__name__ for k, v in row.items()})
    assert row['aAcc'] == 45.0        # 360 intersecting px of 800 labelled
    text = json.dumps(row)
    assert not re.search(r'\d+\.\d{3,}', text), text
    assert json.loads(text) == row


def test_register_all_supplies_the_metric_type_the_sweep_asks_for():
    """`register_all()` must register the type the sweep names as a string.

    The sweep and the replay both reach it by name (`cfg.test_evaluator['type']
    = 'IoUMetricWithPerClass'`), so a missing import here is invisible until a
    real evaluation runs — hours in, on a machine with a GPU.

    In a fresh interpreter, because this module's own
    `from chamnet.metrics import IoUMetricWithPerClass` runs the registration
    decorator: checked in-process, the assertion holds whether or not
    `register_all()` does anything. Verified by breaking it — with that import
    removed from `register_all()` the in-process version still passed, and
    this one fails. Same pattern, and same original defect, as
    tests/test_data_preprocessor.py's registration test.
    """
    script = (
        'import chamnet\n'
        'from mmseg.registry import METRICS\n'
        "assert 'IoUMetricWithPerClass' not in METRICS.module_dict, "
        "'already registered -- this check is not isolated'\n"
        'chamnet.register_all()\n'
        "found = METRICS.get('IoUMetricWithPerClass')\n"
        "assert found is not None, 'register_all() did not register it'\n"
        "assert found.__module__ == 'chamnet.metrics', found.__module__\n"
        "print('OK')\n"
    )
    done = subprocess.run([sys.executable, '-c', script], capture_output=True,
                          text=True, cwd=Path(__file__).resolve().parents[1])
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == 'OK'


def test_evaluate_scores_the_given_checkpoint_on_the_given_split(
        tmp_path, monkeypatch):
    """The five edits `_evaluate` makes to a training config, without a GPU.

    Runner is replaced, so nothing is trained or loaded — but the config
    handed to it is the real one `build_config` emits, and every edit that
    decides *what gets scored* is asserted: which checkpoint, which split,
    which evaluator, where the logs go. Scoring the val phase against the
    test split would silently make every `val_*` column a copy of the
    `test_*` ones.
    """
    scored, plain_torch_load = [], torch.load

    class FakeRunner:
        @staticmethod
        def from_cfg(cfg):
            scored.append(cfg)
            return FakeRunner()

        def test(self):
            # A real Runner.test() loads cfg.load_from here, which on
            # torch >= 2.6 only succeeds inside the relaxed loader.
            assert torch.load is not plain_torch_load, (
                'the checkpoint is loaded outside chamnet.checkpoint\'s block, '
                'so a real run fails after it has finished training')
            raw = {f'IoU.{c}': 10.0 + i for i, c in enumerate(CLASSES)}
            raw.update({k: 90.0 + i for i, k in enumerate(SUMMARY_KEYS)})
            return raw

    monkeypatch.setattr('mmengine.runner.Runner', FakeRunner)
    cfg = build_config(method='hd', backbone='resnet18', recipe='quick',
                       seed=31)
    checkpoint = Path('/somewhere/best_mIoU_iter_1620.pth')

    out = sweep_module._evaluate(cfg, checkpoint, cfg.val_dataloader,
                                 tmp_path / 'val_eval')

    used, = scored
    assert used.load_from == str(checkpoint)
    assert used.work_dir == str(tmp_path / 'val_eval')
    assert used.test_evaluator['type'] == 'IoUMetricWithPerClass'
    # Only the type changes; what it measures is left as the builder set it.
    assert used.test_evaluator['iou_metrics'] == ['mIoU', 'mDice', 'mFscore']
    prefix = used.test_dataloader['dataset']['data_prefix']
    assert prefix['img_path'] == 'valid/images'
    assert used.test_dataloader['num_workers'] == (
        cfg.val_dataloader['num_workers'])
    # No eval_seed: score at the run's own seed, unchanged from the config.
    assert used.randomness['seed'] == 31
    # The caller's config is left alone, so the next split starts from it clean.
    assert cfg.test_evaluator['type'] == 'IoUMetric'
    assert cfg.test_dataloader['dataset']['data_prefix']['img_path'] == \
        'test/images'
    assert 'load_from' not in cfg
    assert out['IoU_pillar'] == 14.0 and out['mIoU'] == 91.0


def test_the_best_checkpoint_and_its_iteration_are_read_off_disk(tmp_path):
    with pytest.raises(FileNotFoundError, match='best_mIoU_iter'):
        sweep_module._best_checkpoint(tmp_path, 'mIoU')

    (tmp_path / 'best_mIoU_iter_1620.pth').touch()
    (tmp_path / 'iter_2000.pth').touch()          # a plain periodic checkpoint
    (tmp_path / 'best_mDice_iter_20.pth').touch()  # a different save_best key
    found = sweep_module._best_checkpoint(tmp_path, 'mIoU')
    assert found == tmp_path / 'best_mIoU_iter_1620.pth'
    assert sweep_module._best_iter(found) == 1620

    (tmp_path / 'best_mIoU_iter_20.pth').touch()
    with pytest.raises(RuntimeError, match='expected exactly one'):
        sweep_module._best_checkpoint(tmp_path, 'mIoU')


# --------------------------------------------------------------------------
# seeds and the git stamp
# --------------------------------------------------------------------------

def test_parse_seeds_reads_ranges_lists_and_the_recipe():
    assert parse_seeds('31-40', 'quick') == [31, 32, 33, 34, 35, 36, 37, 38,
                                             39, 40]
    assert parse_seeds('31,33,37', 'quick') == [31, 33, 37]
    assert parse_seeds('37', 'quick') == [37]
    assert parse_seeds(None, 'paper') == [31, 32, 33, 34, 35, 36, 37, 38,
                                              39, 40]
    assert parse_seeds(None, 'quick') == [31]
    for bad in ('abc', '31-', '31,,32', '40-31'):
        with pytest.raises(ValueError):
            parse_seeds(bad, 'quick')


def _git(repo, *args):
    return subprocess.run(('git', '-C', str(repo), '-c', 'user.name=t',
                           '-c', 'user.email=t@example.invalid') + args,
                          capture_output=True, text=True, check=True).stdout


def test_git_stamp_is_the_commit_and_says_when_the_tree_is_dirty(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    tracked = repo / 'module.py'
    tracked.write_text('x = 1\n')
    _git(repo, 'add', 'module.py')
    _git(repo, 'commit', '-qm', 'first')
    head = _git(repo, 'rev-parse', 'HEAD').strip()

    assert detect_git_hash(repo) == head[:12]
    assert len(head[:12]) == 12

    tracked.write_text('x = 2\n')
    assert detect_git_hash(repo) == head[:12] + '-dirty', (
        'an edited tree was recorded as if it were the commit'
    )

    tracked.write_text('x = 1\n')
    (repo / 'notes.txt').write_text('scratch\n')      # untracked
    assert detect_git_hash(repo) == head[:12]


def test_git_stamp_is_unknown_outside_a_checkout(tmp_path):
    plain = tmp_path / 'exported'
    plain.mkdir()
    assert detect_git_hash(plain) == 'unknown'
    assert detect_git_hash(tmp_path / 'does-not-exist') == 'unknown'


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_sweep_passes_every_argument_through(tmp_path, trained, capsys):
    from chamnet.cli import main

    assert main(['sweep', '--methods', 'hd', '--backbones', 'resnet18, mit_b0',
                 '--ablation', 'nogate', '--seeds', '31-32',
                 '--recipe', 'quick', '--data', '/dataset-not-read',
                 '--out', str(tmp_path), '--git-hash', STAMP]) == 0

    assert capsys.readouterr().out.strip() == str(tmp_path / 'results.csv')
    assert trained == [(31, 'hd', 'resnet18', 'nogate'),
                       (31, 'hd', 'mit_b0', 'nogate'),
                       (32, 'hd', 'resnet18', 'nogate'),
                       (32, 'hd', 'mit_b0', 'nogate')]
    rows = read_rows(tmp_path / 'results.csv')
    assert {r['flow'] for r in rows} == {'dual_plus_nogate'}
    assert {r['git_hash'] for r in rows} == {STAMP}
    assert identify(rows) == [(31, 'dual_plus_nogate', 'mit_b0'),
                              (31, 'dual_plus_nogate', 'resnet18'),
                              (32, 'dual_plus_nogate', 'mit_b0'),
                              (32, 'dual_plus_nogate', 'resnet18')]


def test_cli_sweep_defaults_to_every_method_and_the_recipes_seeds(
        tmp_path, trained):
    from chamnet.cli import main

    assert main(['sweep', '--recipe', 'quick', '--backbones', 'resnet18',
                 '--out', str(tmp_path), '--git-hash', STAMP]) == 0
    assert trained == [(31, 'bl', 'resnet18', None),
                       (31, 'ef', 'resnet18', None),
                       (31, 'hd', 'resnet18', None),
                       (31, 'sd', 'resnet18', None)]


def test_cli_sweep_rejects_an_ablation_it_cannot_run(tmp_path, trained):
    from chamnet.cli import main

    with pytest.raises(ValueError, match='no depth input to shuffle'):
        main(['sweep', '--methods', 'bl', '--backbones', 'resnet18',
              '--ablation', 'shuffled', '--recipe', 'quick',
              '--out', str(tmp_path)])
    assert trained == []


# --------------------------------------------------------------------------
# loading the checkpoints this package writes
# --------------------------------------------------------------------------

def test_an_mmengine_checkpoint_cannot_be_read_without_the_helper(tmp_path):
    """torch >= 2.6 refuses an mmengine checkpoint; the helper reads it.

    Both halves matter. The second is the fix; the first is the evidence that
    the fix is load-bearing rather than decorative, and it is the exact
    failure a real sweep hit after training had already finished — a run could
    not read back the checkpoint it had just written in order to score it.

    Built here from the pieces mmengine actually saves (a state dict plus a
    message hub carrying HistoryBuffer scalar histories) rather than from a
    trained checkpoint, so it needs no GPU and no dataset.
    """
    from mmengine.logging import HistoryBuffer

    from chamnet.checkpoint import mmengine_checkpoint_loading

    buffer = HistoryBuffer()
    buffer.update(0.5)
    buffer.update(0.25)
    path = tmp_path / 'best_mIoU_iter_200.pth'
    torch.save({'state_dict': {'w': torch.zeros(3)},
                'message_hub': {'log_scalars': {'train/loss': buffer}},
                'meta': {'iter': 200}}, path)

    with pytest.raises(Exception, match='Weights only load failed'):
        torch.load(path, map_location='cpu')

    with mmengine_checkpoint_loading():
        loaded = torch.load(path, map_location='cpu')
    assert loaded['meta']['iter'] == 200
    assert torch.equal(loaded['state_dict']['w'], torch.zeros(3))
    restored = loaded['message_hub']['log_scalars']['train/loss']
    assert isinstance(restored, HistoryBuffer)
    assert restored.current() == 0.25


def test_the_relaxed_loader_does_not_outlive_the_block(tmp_path):
    """It is a scoped patch, not a process-wide switch — even on the way out.

    Left on, it would apply full unpickling to every later `.pth` the process
    opens, including ones the caller did not get from this package. An earlier
    revision of `tools/replay.py` did exactly that from import time, and the
    reason it was narrowed is the reason this asserts the restore.
    """
    from mmengine.logging import HistoryBuffer

    from chamnet.checkpoint import mmengine_checkpoint_loading

    path = tmp_path / 'ckpt.pth'
    torch.save({'buf': HistoryBuffer()}, path)
    before = torch.load

    with mmengine_checkpoint_loading():
        torch.load(path, map_location='cpu')
    assert torch.load is before
    with pytest.raises(Exception, match='Weights only load failed'):
        torch.load(path, map_location='cpu')

    # And when the load itself raises, which is when it matters most.
    with pytest.raises(RuntimeError, match='boom'):
        with mmengine_checkpoint_loading():
            raise RuntimeError('boom')
    assert torch.load is before


def test_cli_test_command_loads_the_checkpoint_under_the_relaxed_loader(
        monkeypatch):
    """`chamnet test` had the same defect the sweep did, and the same fix.

    It points `cfg.load_from` at a checkpoint this package wrote, so on
    torch >= 2.6 it failed at load time on every one of them.
    """
    seen, plain_torch_load = [], torch.load

    class FakeRunner:
        @staticmethod
        def from_cfg(cfg):
            seen.append(('built', cfg.load_from, torch.load))
            return FakeRunner()

        def test(self):
            seen.append(('tested', None, torch.load))

    monkeypatch.setattr('mmengine.runner.Runner', FakeRunner)
    from chamnet.cli import main

    assert main(['test', '--method', 'bl', '--backbone', 'resnet18',
                 '--recipe', 'quick',
                 '--checkpoint', '/ckpt/best_mIoU_iter_1620.pth']) == 0

    assert [step for step, _, _ in seen] == ['built', 'tested']
    assert seen[0][1] == '/ckpt/best_mIoU_iter_1620.pth'
    assert seen[1][2] is not plain_torch_load, (
        'the checkpoint is loaded outside chamnet.checkpoint\'s block'
    )
    assert torch.load is plain_torch_load


def test_evaluate_can_score_at_a_fixed_seed_instead_of_the_runs_own(
        tmp_path, monkeypatch):
    """`eval_seed` changes the scoring seed and nothing else.

    Needed to reproduce a recorded number for the two arms whose evaluation
    consumes randomness — the campaign scored every run at one fixed seed
    rather than at the seed it trained at. Training must keep the run's seed,
    so this asserts the override reaches `randomness` while the seed the
    config was *built* with is untouched.
    """
    scored = []

    class FakeRunner:
        @staticmethod
        def from_cfg(cfg):
            scored.append(cfg)
            return FakeRunner()

        def test(self):
            raw = {f'IoU.{c}': 10.0 for c in CLASSES}
            raw.update({k: 90.0 for k in SUMMARY_KEYS})
            return raw

    monkeypatch.setattr('mmengine.runner.Runner', FakeRunner)
    cfg = build_config(method='hd', backbone='resnet18', ablation='shuffled',
                       recipe='quick', seed=37)
    assert cfg.randomness['seed'] == 37, 'precondition: trained at 37'

    sweep_module._evaluate(cfg, Path('/x/best_mIoU_iter_1.pth'),
                           cfg.test_dataloader, tmp_path, 42)

    used, = scored
    assert used.randomness['seed'] == 42
    # Everything else about randomness is carried over, not rebuilt.
    assert used.randomness['deterministic'] is False
    assert used.randomness['diff_rank_seed'] is False
    # And the caller's config still describes the run that was trained.
    assert cfg.randomness['seed'] == 37


def test_cli_sweep_passes_the_eval_seed_through(tmp_path, monkeypatch):
    """The flag has to reach `_train_one`, or reproduction is impossible."""
    seen = []

    def fake_train_one(**kwargs):
        seen.append((kwargs['seed'], kwargs['eval_seed']))
        Path(kwargs['work_dir']).mkdir(parents=True, exist_ok=True)
        return metrics_payload(1)

    monkeypatch.setattr(sweep_module, '_train_one', fake_train_one)
    from chamnet.cli import main

    assert main(['sweep', '--methods', 'hd', '--backbones', 'resnet18',
                 '--ablation', 'shuffled', '--seeds', '37',
                 '--recipe', 'quick', '--out', str(tmp_path),
                 '--git-hash', STAMP, '--eval-seed', '42']) == 0
    assert seen == [(37, 42)]

    seen.clear()
    assert main(['sweep', '--methods', 'hd', '--backbones', 'resnet18',
                 '--ablation', 'shuffled', '--seeds', '38',
                 '--recipe', 'quick', '--out', str(tmp_path),
                 '--git-hash', STAMP]) == 0
    assert seen == [(38, None)], 'default must be the run\'s own seed'


# --------------------------------------------------------------------------
# the replay's two seeds
# --------------------------------------------------------------------------

def _load_replay_module():
    """Import tools/replay.py without it being on the path.

    Safe to import now: it registers nothing at module scope (the per-class
    metric it used to define with a decorator lives in chamnet.metrics), so
    importing it has no effect on any registry.

    Compiled from the source text rather than through `spec.loader`, which
    consults `__pycache__` and validates a cached `.pyc` against the source's
    size and its mtime *in whole seconds*. A same-length edit made inside one
    second -- which is what a mutation test does, and `>= 3` to `== 3` is
    same-length -- would silently run the previous bytecode, so a mutation
    review of `tools/replay.py` could pass while reading code that is no
    longer on disk. Observed live while this project was checking another
    tool: the mutation kept passing and the restored tree kept failing.
    """
    import types

    path = Path(__file__).resolve().parents[1] / 'tools' / 'replay.py'
    module = types.ModuleType('chamnet_replay_tool')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def test_the_replay_reads_the_run_seed_but_evaluates_at_the_recorded_seed(
        tmp_path, monkeypatch):
    """The two seeds are different numbers and must not be confused.

    The checkpoints were trained at 37; every recorded evaluation ran at 42,
    because the campaign's two scoring processes were never passed a seed.
    Replaying at 37 puts SegNeXt-T and the shuffled arms on a different RNG
    trajectory than the numbers they are being compared against — which is
    what kept eight rows of verification/replay.csv unmatched, with the cause
    recorded as unidentified. So: the run seed picks the directory and the
    recorded row, and the evaluation seed is what `randomness.seed` gets.

    Nothing else guards this. `tools/replay.py` produces a committed artifact
    that a paper claim rests on, and a regression here would quietly put it
    back on the wrong seed while still writing a plausible-looking CSV.
    """
    replay_tool = _load_replay_module()
    assert replay_tool.RUN_SEED == 37
    assert replay_tool.RECORDED_EVAL_SEED == 42

    work_dir_name = replay_tool.WORK[('bl', None)][0]
    src = tmp_path / 'src'
    run_dir = src / work_dir_name / '37' / 'chamnet_baseline_resnet18'
    run_dir.mkdir(parents=True)
    (run_dir / 'best_mIoU_iter_100.pth').touch()
    (src / work_dir_name / 'results_v8.csv').write_text(
        'flow,backbone,seed,test_mIoU,test_IoU_pillar\n'
        'baseline,resnet18,37,80.95,80.0\n')

    scored = []

    class FakeRunner:
        @staticmethod
        def from_cfg(cfg):
            scored.append(cfg)
            return FakeRunner()

        def test(self):
            raw = {f'IoU.{c}': 1.0 for c in CLASSES}
            raw.update({k: 2.0 for k in SUMMARY_KEYS})
            return raw

    monkeypatch.setattr(replay_tool, 'Runner', FakeRunner)
    row = replay_tool.replay(str(tmp_path / 'data'), str(src), 'bl',
                             'resnet18', None)

    cfg, = scored
    assert cfg.randomness['seed'] == 42, (
        'the replay evaluated at the training seed, not the seed the recorded '
        'evaluations ran at')
    assert cfg.load_from == str(run_dir / 'best_mIoU_iter_100.pth'), (
        'the checkpoint must come from the run seed\'s directory')
    assert row['recorded_mIoU'] == 80.95 and row['recorded_pillar'] == 80.0

    # And the override still works, which is how the measurement that
    # established 42 was taken.
    scored.clear()
    replay_tool.replay(str(tmp_path / 'data'), str(src), 'bl', 'resnet18',
                       None, eval_seed=37)
    assert scored[0].randomness['seed'] == 37


def test_the_replay_row_is_plain_floats(tmp_path, monkeypatch):
    """`replay()`'s dict must be serialisable, not numpy scalars.

    Same boundary as the sweep's: the CSV writer str()s them either way, so
    nothing in the tool fails, but anything that wants to serialise a row —
    a one-off diagnostic, a later JSON artifact — dies on it.
    """
    replay_tool = _load_replay_module()
    work_dir_name = replay_tool.WORK[('bl', None)][0]
    src = tmp_path / 'src'
    run_dir = src / work_dir_name / '37' / 'chamnet_baseline_resnet18'
    run_dir.mkdir(parents=True)
    (run_dir / 'best_mIoU_iter_100.pth').touch()
    (src / work_dir_name / 'results_v8.csv').write_text(
        'flow,backbone,seed,test_mIoU,test_IoU_pillar\n'
        'baseline,resnet18,37,80.95,80.0\n')

    class FakeRunner:
        @staticmethod
        def from_cfg(cfg):
            return FakeRunner()

        def test(self):
            import numpy as np
            raw = {f'IoU.{c}': np.float32(1.5) for c in CLASSES}
            raw.update({k: np.float32(2.5) for k in SUMMARY_KEYS})
            return raw

    monkeypatch.setattr(replay_tool, 'Runner', FakeRunner)
    row = replay_tool.replay(str(tmp_path / 'data'), str(src), 'bl',
                             'resnet18', None)
    assert all(type(v) is float for k, v in row.items()
               if k not in ('method', 'backbone', 'ablation')), (
        {k: type(v).__name__ for k, v in row.items()})
    json.dumps(row)


def test_the_verification_tools_take_their_roots_from_the_caller():
    """Neither tool may carry a path off the machine that trained the runs.

    They can only run where the private dataset and checkpoints are, so a
    default pointing straight at them is the convenient thing to write -- and
    both shipped that way, `--data` at an absolute path naming a directory on
    that machine and `--src` at its parent. Two reasons that is wrong, and the
    second is the one that bites.

    A published repository should not describe someone's filesystem. And a
    default here is not merely cosmetic: the dataset copy `replay.py` needs is
    *not* the layout `docs/DATA_FORMAT.md` documents, so a `--data` default
    invites running it against a migrated copy and scoring on different label
    files (`_use_original_val_test_layout` explains which); and `--src`
    locates both the checkpoints to load and the recorded metrics to compare
    them against, so a wrong value scores the wrong weights against the wrong
    reference. Either way the run produces numbers rather than an error.

    Checked on the source text because both parsers live under
    `if __name__ == '__main__'` and are never built by an import.
    """
    import ast

    for name in ('replay.py', 'retrain_verify.py'):
        path = Path(__file__).resolve().parents[1] / 'tools' / name
        tree = ast.parse(path.read_text())

        declared = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', None) == 'add_argument'
                    and node.args
                    and isinstance(node.args[0], ast.Constant)):
                declared.setdefault(node.args[0].value, []).append(
                    {kw.arg: kw.value for kw in node.keywords})

        for flag in ('--data', '--src'):
            if flag not in declared:
                continue            # retrain_verify.py takes no --data
            assert len(declared[flag]) == 1, (
                f'{name}: expected one {flag}, found {len(declared[flag])}')
            keywords = declared[flag][0]
            assert 'default' not in keywords, (
                f"{name}'s {flag} must not have a default: the only path it "
                'could sensibly default to is one on the training machine')
            assert isinstance(keywords.get('required'), ast.Constant) and \
                keywords['required'].value is True, \
                f'{name}: {flag} must be required'
        assert '--src' in declared, f'{name}: no --src to check'

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith('/data/'), (
                    f'{name} names a path on the training machine: '
                    f'{node.value!r}')
