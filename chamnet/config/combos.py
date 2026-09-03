"""method × ablation 유효 조합. 이 표가 유일한 출처."""

# Every valid combination, mapped to the name the original campaign gave it.
#
# That name -- the "flow" -- is not decoration. It is the `flow` column of the
# recorded per-run CSV, and it is the directory name a run's work_dir is
# built from (`<work_root>/<seed>/chamnet_<flow>_<backbone>/`). Anything that
# has to line up with a recorded run therefore has to agree on it: the sweep
# that writes new rows, and the replay that reads the old ones. Both take the
# name from here, so a rename cannot land in one and miss the other.
#
# The vocabulary is the campaign's, not the package's: `bl` is written
# `baseline`, `ef` is written `proposed` (the name early fusion had while the
# experiments were being run), `sd` is `dual` and `hd` is `dual_plus`. It is
# kept as recorded rather than modernised, because renaming it would orphan
# every metrics row and work_dir the campaign produced.
#
# HD's four control arms, plus early fusion's depth-structure control.
# ('ef', 'shuffled') exists and ('sd', 'shuffled') does not because the paper
# ran the shuffled control on the two arms it argues about, not on all four.
FLOW: dict[tuple[str, str | None], str] = {
    ('bl', None):       'baseline',
    ('ef', None):       'proposed',
    ('sd', None):       'dual',
    ('hd', None):       'dual_plus',
    ('ef', 'shuffled'): 'proposed_shuffled',
    ('hd', 'shuffled'): 'dual_plus_shuffled',
    ('hd', 'rgb'):      'dual_plus_rgb',
    ('hd', 'nogate'):   'dual_plus_nogate',
    ('hd', 'bigate'):   'dual_plus_bigate',
}

# Derived, not restated: a combination is valid exactly when the campaign has
# a name for it. Written the other way round these two tables could disagree,
# and the disagreement would be silent -- a combination valid but unnameable
# would crash only once something asked for its work_dir.
VALID: set[tuple[str, str | None]] = set(FLOW)

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
