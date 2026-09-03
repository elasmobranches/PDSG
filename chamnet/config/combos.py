"""method × ablation 유효 조합. 이 표가 유일한 출처."""

VALID: set[tuple[str, str | None]] = {
    ('bl', None), ('ef', None), ('sd', None), ('hd', None),
    # 후속 작업에서 추가:
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
    # Run in the paper, but their backbones are not part of this release yet.
    # The wording is what a user sees, so it says what is true of the package
    # rather than describing the work queue that will change it.
    ('hd', 'shuffled'): 'not implemented in this release yet',
    ('hd', 'rgb'):      'not implemented in this release yet',
    ('hd', 'nogate'):   'not implemented in this release yet',
    ('hd', 'bigate'):   'not implemented in this release yet',
}


def validate(method: str, ablation: str | None) -> None:
    if (method, ablation) in VALID:
        return
    why = _WHY.get((method, ablation), 'unknown combination')
    raise ValueError(
        f"method={method!r} with ablation={ablation!r} is not available: {why}. "
        f"Run `chamnet list` to see valid combinations."
    )
