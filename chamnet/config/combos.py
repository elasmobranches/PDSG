"""method × ablation 유효 조합. 이 표가 유일한 출처."""

VALID: set[tuple[str, str | None]] = {
    ('bl', None), ('ef', None), ('sd', None), ('hd', None),
    # HD's four control arms, plus early fusion's depth-structure control.
    # ('ef', 'shuffled') exists and ('sd', 'shuffled') does not because the
    # paper ran the shuffled control on the two arms it argues about, not on
    # all four.
    ('ef', 'shuffled'),
    ('hd', 'shuffled'), ('hd', 'rgb'), ('hd', 'nogate'), ('hd', 'bigate'),
}

# Why each *invalid* combination is invalid, in the words a user sees. Every
# (method, ablation) pair outside VALID that a user could plausibly type has an
# entry here -- with four methods and four ablations there are twenty pairs,
# nine valid and eleven listed below, so nothing a `--ablation` choice can
# produce falls through to the bare "unknown combination" fallback. That
# fallback is left in place for a genuinely unrecognised ablation name.
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
}


def validate(method: str, ablation: str | None) -> None:
    if (method, ablation) in VALID:
        return
    why = _WHY.get((method, ablation), 'unknown combination')
    raise ValueError(
        f"method={method!r} with ablation={ablation!r} is not available: {why}. "
        f"Run `chamnet list` to see valid combinations."
    )
