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

`--methods` defaults to the two training methods this package currently
implements, `bl` and `sd`. The other two the underlying project defines,
`ef` and `hd` (and every `hd`/`ef` ablation in the script's `WORK` table),
aren't implemented in this package yet. `--all` replays every entry in
`WORK` regardless, so it will fail past `bl`/`sd` until those land — this is
a real, known limitation, not a bug to silently paper over.

## The tool exits 1 by design

`tools/replay.py` gates on **both** mIoU and Pillar (Pillar is the paper's
headline metric, and it can move further than mIoU on the same row — e.g.
`bl/segnext_t` was 0.04 off on mIoU but 0.20 off on Pillar; gating on mIoU
alone would have missed that). As of the run recorded in `replay.csv`, 6 of
8 `bl`/`sd` combinations match `results_v8.csv` exactly (both metrics, to
the file's own 2-decimal precision); the SegNeXt-T rows (`bl/segnext_t`,
`sd/segnext_t`) do not, and **this is expected, not a defect**:

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
`_build_bases` runs.

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
bounded gap on those two rows specifically is the expected outcome, and the
tool's exit code reflects that honestly rather than being tuned to always
report success.

The three other backbones (`resnet18`, `mit_b0`, `convnext_atto`) have also
shown small single-row differences in some replay runs — up to about 0.01
on either metric, on one row at a time — that later reruns did not
reproduce (the same row landing at exactly 0.00 on a subsequent run with no
code change in between). This is ordinary GPU/cudnn non-determinism under
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
