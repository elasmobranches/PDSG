"""method × ablation 유효 조합. 스펙 §4.2 의 표가 유일한 출처."""

VALID: set[tuple[str, str | None]] = {
    ('bl', None), ('ef', None), ('sd', None), ('hd', None),
    # Task 11 에서 추가:
    # ('ef', 'shuffled'),
    # ('hd', 'shuffled'), ('hd', 'rgb'), ('hd', 'nogate'), ('hd', 'bigate'),
}

_WHY = {
    ('bl', 'shuffled'): 'BL has no depth input to shuffle',
    ('bl', 'rgb'):      'BL has no depth-slot encoder',
    ('bl', 'nogate'):   'BL has no fusion gate',
    ('bl', 'bigate'):   'BL has no fusion gate',
    ('ef', 'rgb'):      'EF has no separate depth encoder, so there is nothing to '
                        'swap for RGB; its depth channel adds ~0.5K parameters, '
                        'which makes a capacity control meaningless',
    ('ef', 'nogate'):   'EF concatenates at the input and has no gate',
    ('ef', 'bigate'):   'EF concatenates at the input and has no gate',
    ('sd', 'shuffled'): 'not run in the paper',
    ('sd', 'rgb'):      'not run in the paper',
    ('sd', 'nogate'):   'not run in the paper',
    ('sd', 'bigate'):   'not run in the paper',
    ('hd', 'shuffled'): 'not enabled yet — arrives in a later task',
    ('hd', 'rgb'):      'not enabled yet — arrives in a later task',
    ('hd', 'nogate'):   'not enabled yet — arrives in a later task',
    ('hd', 'bigate'):   'not enabled yet — arrives in a later task',
}


def validate(method: str, ablation: str | None) -> None:
    if (method, ablation) in VALID:
        return
    why = _WHY.get((method, ablation), 'unknown combination')
    raise ValueError(
        f"method={method!r} with ablation={ablation!r} is not available: {why}. "
        f"Run `chamnet list` to see valid combinations."
    )
