from itertools import product

import pytest

from chamnet.cli import ABLATIONS
from chamnet.config.combos import VALID, _WHY, validate


@pytest.mark.parametrize('method,ablation', sorted(VALID, key=str))
def test_valid_combos_do_not_raise(method, ablation):
    validate(method, ablation)


def test_the_nine_valid_combinations_are_the_paper_grid():
    """Pin the table itself, not just that its entries pass.

    `chamnet list`, the CLI's `--method` vocabulary and the replay's coverage
    are all derived from VALID, so a combination silently added or dropped here
    changes what the package advertises and what the verification artifact
    covers, with nothing else failing. These nine are the arms the paper
    actually ran: four methods, plus HD's four controls, plus early fusion's
    shuffled control.
    """
    assert VALID == {
        ('bl', None), ('ef', None), ('sd', None), ('hd', None),
        ('ef', 'shuffled'),
        ('hd', 'shuffled'), ('hd', 'rgb'), ('hd', 'nogate'), ('hd', 'bigate'),
    }


def test_invalid_combo_raises_with_reason():
    with pytest.raises(ValueError, match='no depth input to shuffle'):
        validate('bl', 'shuffled')


def test_message_names_the_offending_combo():
    with pytest.raises(ValueError, match=r"method='bl'.*ablation='shuffled'"):
        validate('bl', 'shuffled')


@pytest.mark.parametrize('method,ablation',
                         sorted(set(product(['bl', 'ef', 'sd', 'hd'], ABLATIONS))
                                - VALID, key=str))
def test_every_rejectable_combination_gives_a_real_reason(method, ablation):
    """No combination a user can type may fall through to 'unknown combination'.

    `--ablation` accepts a fixed vocabulary and `--method` a fixed set, so the
    product of the two is exactly what a user can ask for. Anything in it that
    is not valid must be refused with a sentence explaining why that arm makes
    no sense (or was not run), because the generic fallback tells the user
    nothing and reads like a bug in the package. ('ef', 'shuffled') sat in this
    hole for several revisions -- valid neither in VALID nor in _WHY -- and
    nothing failed; it is valid now, and this parametrisation is what stops the
    next one going unnoticed.
    """
    with pytest.raises(ValueError) as excinfo:
        validate(method, ablation)
    assert 'unknown combination' not in str(excinfo.value), (
        f'({method!r}, {ablation!r}) is rejected without a reason')
    assert (method, ablation) in _WHY


def test_truly_unrecognised_combo_falls_back_to_unknown():
    with pytest.raises(ValueError, match='unknown combination'):
        validate('bl', 'not-a-real-ablation')
