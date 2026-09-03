"""The representative-seed rule has to be a rule, not a preference.

`tools/select_seed.py` exists so that one number in this release -- seed 37,
which picks the staged checkpoints, the qualitative figures, the replay rows
and the retraining check -- is the output of a stated procedure rather than a
choice somebody liked. The recorded per-run CSV it reads is not distributed,
so these tests cannot check *its answer on the real data*. What they can check
is everything that makes the answer meaningful: that the rule reads validation
metrics and is structurally unable to read test ones, that the pieces of
arithmetic are the ones the docstring describes, and that an incomplete input
is refused rather than silently scored on fewer runs.

Every fixture here is synthetic and constructed so that the right answer is
known before the tool is run.
"""
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

HEADER = ('run,flow,backbone,seed,best_iter,'
          'val_mIoU,val_IoU_pillar,test_mIoU,test_IoU_pillar')

FLOWS = {'bl': 'baseline', 'ef': 'proposed', 'sd': 'dual', 'hd': 'dual_plus'}
BACKBONES = ('resnet18', 'mit_b0', 'segnext_t', 'convnext_atto')
SEEDS = tuple(range(31, 41))


def _tool():
    """Import tools/select_seed.py from its source text.

    Compiled from the bytes rather than loaded through `SourceFileLoader`,
    which consults `__pycache__` and validates a cached `.pyc` against the
    source's size and its mtime *in whole seconds*. A same-length edit made
    within one second -- what a mutation test does -- would silently run the
    previous bytecode, and a test that can read code other than the code on
    disk is not load-bearing. Same reasoning as `tests/test_retrain_artifact`.

    The module imports only the stdlib and `chamnet.config.combos`, which
    registers nothing, so this has no effect on any registry.
    """
    path = ROOT / 'tools' / 'select_seed.py'
    module = types.ModuleType('chamnet_select_seed_tool')
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def _csv(tmp_path, values, *, test_values=None, name='results.csv'):
    """Write a recorded-shaped CSV.

    `values[(method, backbone)][seed] = (val_mIoU, val_pillar)`. Test columns
    default to something wildly different from the validation ones, so a rule
    that read them could not accidentally agree with one that does not.
    """
    lines = [HEADER]
    for (method, backbone), per_seed in values.items():
        for seed, (miou, pillar) in per_seed.items():
            if test_values is None:
                t_miou, t_pillar = 100.0 - miou, 100.0 - pillar
            else:
                t_miou, t_pillar = test_values[(method, backbone)][seed]
            lines.append(
                f'{seed},{FLOWS[method]},{backbone},{seed},1000,'
                f'{miou},{pillar},{t_miou},{t_pillar}')
    path = tmp_path / name
    path.write_text('\n'.join(lines) + '\n')
    return path


def _spread(offsets):
    """`{seed: (val_mIoU, val_pillar)}` from per-seed offsets off a common base.

    Both metrics get the same offset, so the four criteria see the same
    ordering and the expected winner is unambiguous.
    """
    return {seed: (80.0 + offset, 70.0 + offset)
            for seed, offset in zip(SEEDS, offsets)}


def _uniform(offsets):
    return {(method, backbone): _spread(offsets)
            for method in FLOWS for backbone in BACKBONES}


# A spread whose median is 0.0 and in which seed 35 sits exactly on it.
CENTRED_ON_35 = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_the_seed_on_every_configuration_median_is_the_one_selected(tmp_path):
    """The base case, with the answer fixed by construction.

    Seed 35 carries offset 0.0 in all sixteen configurations, which is each
    one's median, so its `mean |z|` is 0 and no other seed's can be.
    """
    tool = _tool()
    seed, _ = tool.select(tool.read_validation(
        _csv(tmp_path, _uniform(CENTRED_ON_35))))
    assert seed == 35


def test_the_rule_never_reads_a_test_column(tmp_path):
    """The whole point of the tool, so it is checked two ways.

    First structurally: what `read_validation` returns contains no test key at
    all, so there is nothing downstream for a future edit to reach for.

    Then behaviourally: a CSV whose test columns say seed 31 is the typical
    one, while its validation columns say seed 35 is, still selects 35.
    """
    tool = _tool()
    values = _uniform(CENTRED_ON_35)
    # Test columns centred on seed 31 instead -- the opposite answer.
    flipped = tuple(reversed(CENTRED_ON_35))
    test_values = {key: _spread(flipped) for key in values}

    table = tool.read_validation(
        _csv(tmp_path, values, test_values=test_values))
    for per_seed in table.values():
        for row in per_seed.values():
            assert row, 'no validation columns were kept'
            assert not [key for key in row if not key.startswith('val_')], (
                f'read_validation kept a non-validation column: {sorted(row)}')

    seed, _ = tool.select(table)
    assert seed == 35


def test_naming_a_non_validation_metric_is_an_error_not_a_lookup(tmp_path):
    """The guard has to fire before any file is opened.

    `METRICS` is the one place a future edit could point the rule at the test
    split, so pointing it there must fail loudly rather than work.
    """
    tool = _tool()
    path = _csv(tmp_path, _uniform(CENTRED_ON_35))

    tool.METRICS = ('test_mIoU',)
    with pytest.raises(SystemExit) as raised:
        tool.read_validation(path)
    assert 'validation' in str(raised.value)

    # And if the guard were somehow passed, the lookup itself has no test
    # column to find, because read_validation kept none.
    tool.METRICS = ('val_mIoU',)
    table = tool.read_validation(path)
    tool.METRICS = ('test_mIoU',)
    with pytest.raises(SystemExit):
        tool.scores(table, tool.BACKBONES, 'test_mIoU')


def test_the_worst_rank_decides_not_the_average(monkeypatch):
    """The two aggregators are made to disagree, and the worst one must win.

    Seed 31 here is first on two criteria and seventh on the other two: worst
    rank 7, average 4.0. Seed 35 is fifth and fourth: worst rank 4, average
    4.5. So the average of the ranks prefers 31 and the worst of them prefers
    35, and the rule is the worst of them -- a seed has to be typical under
    every criterion, not on average across them.

    Driven through a substituted `criteria` so the case is the aggregation and
    nothing else; the scoring it normally consumes is checked separately.
    """
    tool = _tool()
    filler = {seed: 9.0 for seed in tool.SEEDS if seed not in (31, 35)}
    by_criterion = {
        'all16/val_mIoU':          {**filler, 31: 1.0, 35: 5.0},
        'all16/val_IoU_pillar':    {**filler, 31: 1.0, 35: 5.0},
        'resnet18/val_mIoU':       {**filler, 31: 7.0, 35: 4.0},
        'resnet18/val_IoU_pillar': {**filler, 31: 7.0, 35: 4.0},
    }
    monkeypatch.setattr(tool, 'criteria', lambda table: by_criterion)

    rank = lambda seed: [by_criterion[name][seed] for name in by_criterion]
    assert max(rank(35)) < max(rank(31)), 'precondition: worst prefers 35'
    assert sum(rank(31)) < sum(rank(35)), 'precondition: average prefers 31'

    seed, _ = tool.select(object())
    assert seed == 35, 'the average of the ranks decided instead of the worst'


def test_tied_scores_share_an_average_rank(tmp_path):
    """Two seeds equally typical must not both be handed rank 1.

    If they were, a seed tied for first on every criterion would be
    indistinguishable from one that won outright, and the tie-break would
    never run.
    """
    tool = _tool()
    tied = (-4.0, -3.0, -2.0, -1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 5.0)
    table = tool.read_validation(_csv(tmp_path, _uniform(tied)))
    by_criterion = tool.criteria(table)
    for name, per_seed in by_criterion.items():
        assert per_seed[35] == per_seed[36] == 1.5, (name, per_seed)
    # The tie is broken by the lower seed number, deterministically.
    assert tool.select(table)[0] == 35


def test_z_is_measured_about_the_median_not_the_mean(tmp_path):
    """One outlier must not move the centre the other nine are judged against.

    Nine seeds sit at 0.0 and one sits 90 points away. About the median the
    nine are all equally typical; about the mean they are not, and the
    would-be winner changes. The rule uses the median.
    """
    tool = _tool()
    outlier = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 90.0)
    table = tool.read_validation(_csv(tmp_path, _uniform(outlier)))
    scored = tool.scores(table, tool.BACKBONES, 'val_mIoU')
    assert scored[31] == 0.0, scored
    assert scored[39] > 0.0
    assert tool.select(table)[0] == 31


def test_an_incomplete_configuration_is_refused(tmp_path):
    """A short series must not be scored as though it were a full one.

    Nine seeds instead of ten changes every median and SD in that
    configuration, and it is exactly what a partly finished sweep looks like.
    """
    tool = _tool()
    values = _uniform(CENTRED_ON_35)
    del values[('hd', 'resnet18')][40]
    with pytest.raises(SystemExit) as raised:
        tool.read_validation(_csv(tmp_path, values))
    assert 'seeds' in str(raised.value)


def test_a_missing_configuration_is_refused(tmp_path):
    tool = _tool()
    values = _uniform(CENTRED_ON_35)
    del values[('sd', 'segnext_t')]
    with pytest.raises(SystemExit) as raised:
        tool.read_validation(_csv(tmp_path, values))
    assert 'missing configurations' in str(raised.value)


def test_a_duplicated_row_is_refused(tmp_path):
    """Two rows for one (configuration, seed) is a corrupted CSV, not data."""
    tool = _tool()
    path = _csv(tmp_path, _uniform(CENTRED_ON_35))
    with path.open('a') as handle:
        handle.write('37,baseline,resnet18,37,1000,80.0,70.0,20.0,30.0\n')
    with pytest.raises(SystemExit) as raised:
        tool.read_validation(path)
    assert 'more than one row' in str(raised.value)


def test_the_scope_is_the_four_published_methods_on_four_backbones(tmp_path):
    """Sixteen configurations, and the control arms are not among them.

    The scope is the one decision the answer is sensitive to, so it is pinned
    here: rows for a control arm in the same file must be ignored rather than
    quietly widening the grid.
    """
    tool = _tool()
    assert sorted(tool.METHODS) == ['bl', 'ef', 'hd', 'sd']
    assert len(tool.BACKBONES) == 4

    path = _csv(tmp_path, _uniform(CENTRED_ON_35))
    with path.open('a') as handle:
        for seed in SEEDS:
            handle.write(f'{seed},dual_plus_nogate,resnet18,{seed},1000,'
                         f'{99.0 - seed},{99.0 - seed},1.0,1.0\n')

    table = tool.read_validation(path)
    assert len(table) == 16, sorted(table)
    assert all(method in tool.METHODS for method, _ in table)


def test_expect_gates_the_answer(tmp_path, capsys):
    """`--expect` has to fail on the wrong seed, or it documents nothing."""
    tool = _tool()
    path = _csv(tmp_path, _uniform(CENTRED_ON_35))
    assert tool.main(['--results', str(path), '--expect', '35']) == 0
    assert tool.main(['--results', str(path), '--expect', '37']) == 1
    assert 'FAIL' in capsys.readouterr().out


def test_the_report_lists_every_seed_and_names_the_winner(tmp_path):
    tool = _tool()
    seed, text = tool.report(tool.read_validation(
        _csv(tmp_path, _uniform(CENTRED_ON_35))))
    assert f'representative seed: {seed}' in text
    for candidate in SEEDS:
        assert any(line.split()[:1] == [str(candidate)]
                   for line in text.splitlines()), candidate
