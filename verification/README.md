# verification/

## What this is

`tools/replay.py` loads a set of previously-trained checkpoints into models
built by this package's own code, runs them against the greenhouse test
split they were originally evaluated on, and compares the result to
`results_v8.csv` — the per-run metrics recorded when those checkpoints were
originally trained and tested. Same weights, different (ported) forward-pass
code. If the numbers match, porting the code didn't change what the model
computes. `replay.csv` in this directory is the output of the most recent
such run (see its first line for exactly which run — commit, date, GPU,
and library versions; that line is a `#`-prefixed comment, not a data row).

The greenhouse dataset and the trained checkpoints are **not distributed
with this repository** — they live on a private training server with GPU
access, so this tool can only be run there, not from a plain checkout. Once
on that server, with the dataset and checkpoints in place, it runs as:

```bash
python3 tools/replay.py --out verification/replay.csv
```

`--methods` defaults to the three training methods this package currently
implements: `bl`, `sd` and `hd`. The fourth the underlying project defines,
`ef`, and every `hd`/`ef` ablation in the script's `WORK` table, aren't
implemented in this package yet. `--all` replays every entry in `WORK`
regardless, so it will fail past `bl`/`sd`/`hd` until those land — this is a
real, known limitation, not a bug to silently paper over.

## The tool exits 1 by design

`tools/replay.py` gates on **both** mIoU and Pillar (Pillar is the paper's
headline metric, and it can move further than mIoU on the same row — e.g.
`bl/segnext_t` was 0.04 off on mIoU but 0.20 off on Pillar; gating on mIoU
alone would have missed that). Of the 12 `bl`/`sd`/`hd` combinations, three
never match and one is a coin flip, for two separate and separately
documented reasons: the three SegNeXt-T rows (`bl/segnext_t`,
`sd/segnext_t`, `hd/segnext_t`), explained immediately below, and
`bl/resnet18`'s Pillar, which lands on a 2-decimal rounding boundary — see
"The `bl/resnet18` Pillar boundary" further down. Neither is a defect, and
the tool reports both as failures rather than being tuned to hide them.

**The printed pass count is therefore not deterministic.** It is 8/12 or
9/12 depending purely on which side of that boundary the ResNet row happens
to land on in a given run; the committed `replay.csv` shows one of the two
and a fresh run may legitimately show the other, with no code change and no
significance. The eight rows that always match are the eight that carry the
evidence.

Every `hd` row except SegNeXt-T matches exactly on both metrics
(`hd/resnet18` 81.88/84.01, `hd/mit_b0` 80.52/79.02, `hd/convnext_atto`
81.18/79.19 mIoU/Pillar), which is the evidence that porting the HD
backbones and their pretrained depth transfer did not change what those
models compute.

`LightHamHead`'s `NMF2D._build_bases` (`mmseg/models/decode_heads/ham_head.py`,
lines 89-90 and 123) calls `torch.rand((B * S, D, R))` **unconditionally on
every forward pass** whenever `rand_init=True` — which is the setting in
every SegNeXt-T config in this project, identically in the original merged
config and this package's builder output. SegNeXt-T's segmentation logits
are therefore RNG-state dependent by design, not solely a function of
weights + input; this is mmseg's own (unmodified, vanilla) `LightHamHead`
implementation, not something this port changed. Concretely: rerunning
`bl/segnext_t` with `randomness.seed=42` (the seed the original recorded run
used, vs. this replay's `seed=37`) moved the observed gap from 0.04/0.20
(mIoU/Pillar) to 0.01/0.11 — closer, in the expected direction, but still
not exact, because seeding `torch.manual_seed` doesn't reconstruct the exact
sequence of prior RNG calls a *different process* had made by the time
`_build_bases` runs. Consistent with that explanation, the SegNeXt-T rows
are stable *replay-to-replay* — two consecutive replays reproduced all three
to the last digit (`bl` 80.76/77.63, `sd` 80.54/80.47, `hd` 80.52/82.21
mIoU/Pillar) while still differing from the recorded numbers by 0.03-0.10
mIoU and 0.10-0.20 Pillar. Two fresh processes doing the same work in the
same order draw the same random bases; the process that produced
`results_v8.csv` ran its test loop at the end of a training run, with a very
different amount of RNG already consumed.

What this artifact can and can't support: it is a code-equivalence check —
evidence that this port computes the same thing the original code did, for
the same fixed weights — not a statement about the statistical reliability
of any published result. It says nothing about how a ±0.2 Pillar difference
compares to the spread across independently trained runs; that would need
its own measurement against however many training seeds actually back the
numbers being defended, which is outside what a single-checkpoint replay
can establish. What it does support: **SegNeXt-T evaluation is not exactly
reproducible even with fixed weights**, so nobody should expect a future
replay run's SegNeXt-T row to match `results_v8.csv` bit-for-bit — a small,
bounded gap on those three rows specifically is the expected outcome, and
the tool's exit code reflects that honestly rather than being tuned to
always report success.

## The `bl/resnet18` Pillar boundary

This row reads 79.99 against a recorded 80.00 in some runs and matches at
80.00 in others, with no code change in between. That is not a 0.01
regression that comes and goes; the underlying number barely moves at all,
and it is worth being precise about what does.

`IoUMetric` reports IoU as a ratio of integer pixel counts accumulated over
the split, rounded to two decimals. Those raw counts are not in the metric's
output — `tools/replay.py` records the rounded value, like `results_v8.csv`
does — so the table below came from a throwaway subclass of
`IoUMetricWithPerClass` that printed the accumulated `area_intersect` /
`area_union` before rounding. **That instrumentation is not shipped**: it
was a one-off diagnostic, and this table cannot be reproduced from
`tools/replay.py` as it stands without re-adding it. Over six *separate
processes* running the identical replay:

```
intersect  union     IoU (%)       rounded
1456721    1821017   79.99491493   79.99
1456720    1821015   79.99494787   79.99
1456721    1821015   79.99500279   80.00
1456720    1821015   79.99494787   79.99
1456721    1821017   79.99491493   79.99
1456726    1821014   79.99532129   80.00
```

Two things follow. First, the run-to-run spread is about six pixels in 1.46
million and 0.0004 percentage points of IoU — an order of magnitude below
even the ~0.01 jitter described below, and far below anything that could
matter. Second, that tiny band straddles 79.995, the point where `round(x,
2)` switches between 79.99 and 80.00. So the rounded value flips (here, four
runs to two) while the quantity being rounded is effectively constant.

How far apart the replay and the original actually are cannot be pinned down
from the artifact, and it is worth stating the bound honestly rather than
the flattering half of it. A recorded 80.00 means the original run's raw
value lay somewhere in `[79.995, 80.005)`. Against the lowest value observed
here, 79.99491, the true disagreement could be anywhere from **0 up to
0.0101 percentage points** — 0 because two of the six observations are
themselves above 79.995 and would agree with the original exactly at two
decimals, and 0.0101 because the original could have sat just under 80.005.
(An earlier version of this section quoted 0.005, which is the distance to
the *rounding boundary*, not to the widest admissible original — an
understatement, in the reassuring direction.) `results_v8.csv` stores two
decimals, so there is no more precise recorded number to compare against and
the range cannot be narrowed further. What can be said is that the bound is
comparable to the ~0.01 cudnn jitter described below and nowhere near the
SegNeXt-T gap.

Repeated runs were used to characterise the flutter, **not** to pick one.
`bl` is also untouched by the HD code added alongside this `replay.csv`, and
the config `build_config` emits for it is unchanged
(`tests/test_matches_paper.py` compares it to the paper's own merged config
key for key), so this row's inputs did not change.

Anything genuinely larger than this on `resnet18`, `mit_b0` or
`convnext_atto` would not fit the pattern. Those three backbones have shown
small single-row differences in some replay runs — up to about 0.01 on
either metric, on one row at a time — that later reruns did not reproduce
(the same row landing at exactly 0.00 on a subsequent run with no code
change in between). This is ordinary GPU/cudnn non-determinism under
`deterministic=False` and `cudnn_benchmark=True`: forcing full determinism
(`cudnn.deterministic=True`, `cudnn.benchmark=False`) and disabling TF32
did not remove it, which shows that bit-exact determinism isn't available
in this setup at all — it does not show that a given 0.01 gap is safe to
ignore on its own. The basis for treating it as jitter rather than a defect
is comparative: it is at least an order of magnitude smaller than the
SegNeXt-T gap above, it appears on different rows on different runs rather
than persistently on the same one, and it vanishes on rerun without any
code change. If a future run shows a gap larger than ~0.01 on one of these
three backbones, or the same row missing on repeated reruns, that would not
fit this pattern and would be worth investigating as a real regression.

## The mask finding

Early replay attempts (before the `_use_original_val_test_layout` fix in
`tools/replay.py`) showed small (~0.01-0.4 point) gaps on **every**
backbone, `bl` included. Root cause: `results_v8.csv` was generated against
label files under a `masks_gray/` folder, but `build_config`'s dataloader —
correctly, for this package's own unified data layout — reads a `masks/`
folder instead. Both folders exist for every split of the dataset this
replay runs against:

| split | `masks/` files | `masks_gray/` files | mismatches |
|---|---|---|---|
| train | 318 | 318 | 0 |
| valid | 91 | 91 | 0 |
| test | 45 | 45 | **1** |

Comparing the mismatched pair byte-for-byte:

```
one test image, disputed region x 902-959, y 0-537 (full-height right-edge strip):
  masks/       labels it [0, 2, 5]
  masks_gray/  labels it [4] = Pillar
  Pillar pixel count: masks/ 4,342  vs.  masks_gray/ 19,012
```

**`masks_gray/` is correct.** The disputed strip is a real structural post
standing the full height of the frame's right edge (visible directly in the
source image) — `masks_gray/` covers it precisely; `masks/` dropped it
entirely. A parallel check of a second, independent copy of this dataset
found every one of its test images byte-identical between the two folders,
confirming `masks/` is *normally* a straight rename of `masks_gray/` and
this one file is a stale artifact specific to this particular dataset copy,
not a systematic error in how `masks/` gets produced from `masks_gray/` in
general.

Conclusion: `results_v8.csv`'s recorded numbers were computed against the
correct labels. `tools/replay.py` reading `masks_gray/` for val/test is the
right call for reproducing those numbers against this dataset copy
specifically — not because `masks/` doesn't exist there (it does, for every
split, and matches everywhere except the one file above), but because that
is what the recorded metrics were actually computed against, and it sidesteps
the one stale file. **Action item, not yet done:** `build_config` itself
still reads `masks/` for training and evaluation on this dataset copy, which
means the one bad file above silently affects real training runs, not just
this replay — regenerating `masks/` from `masks_gray/` for this dataset copy
(or otherwise fixing the one stale file) is tracked as follow-up work, not
resolved by anything in this directory.
