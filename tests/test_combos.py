import pytest

from chamnet.config.combos import validate


@pytest.mark.parametrize('method,ablation', [
    ('bl', None), ('ef', None), ('sd', None), ('hd', None),
])
def test_valid_combos_do_not_raise(method, ablation):
    validate(method, ablation)


def test_invalid_combo_raises_with_reason():
    with pytest.raises(ValueError, match='no depth input to shuffle'):
        validate('bl', 'shuffled')


def test_message_names_the_offending_combo():
    with pytest.raises(ValueError, match=r"method='bl'.*ablation='shuffled'"):
        validate('bl', 'shuffled')


def test_not_yet_enabled_hd_ablations_say_so_not_unknown():
    for ablation in ('shuffled', 'rgb', 'nogate', 'bigate'):
        with pytest.raises(ValueError, match='not enabled yet'):
            validate('hd', ablation)


def test_truly_unrecognised_combo_falls_back_to_unknown():
    with pytest.raises(ValueError, match='unknown combination'):
        validate('bl', 'not-a-real-ablation')
