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

Every row replays a checkpoint **trained at a single seed** (37) and
**evaluates it at a different one** (42). That is not a slip. 42 is the seed
the recorded evaluations themselves ran at: the campaign trained with the
run's seed and then scored the checkpoint in two further processes it never
passed a seed to, so both fell back to the base config's
`randomness.seed = 42`, and every number in `results_v8.csv` was produced
there whatever seed had been trained. `tools/replay.py`'s
`RECORDED_EVAL_SEED` carries the log evidence and the measurement that
established it. It matters only for the arms whose *evaluation* consumes
randomness, and getting it right is what closed the shuffled family below.

Before reading a Pillar number out of this file as a result rather than as a
code-equivalence check, see "Pillar on this split rests on a handful of
images" below: pillar appears in only 14 of the 45 test images, so aggregate
Pillar IoU here is sensitive to a few of them, and one row is visibly
affected. The paper's numbers are ten-seed means.

The greenhouse dataset and the trained checkpoints are **not distributed
with this repository** — they live on a private training server with GPU
access, so this tool can only be run there, not from a plain checkout. Once
on that server, with the dataset and checkpoints in place, it runs as:

```bash
python3 tools/replay.py --all --out verification/replay.csv \
    --commit "$(git rev-parse --short HEAD)"
```

`--all` replays every entry in the script's `WORK` table: the nine arms this
package implements (`bl`, `ef`, `sd`, `hd`, plus HD's four controls and early
fusion's shuffled control) on all four backbones, 36 rows. That is what
`replay.csv` here contains. Without `--all`, `--methods` replays just the four
training methods without ablations, 16 rows, which is the quicker check.

`WORK`'s keys are checked against `chamnet.config.combos.VALID` at startup, so
a combination the package advertises but the table does not carry (or the
reverse) is an error rather than a silently missing row. The table itself
holds only the two work_dir names each arm's checkpoints live under; the arm's
recorded *name* — the `flow` in a work_dir path and in `results_v8.csv` — comes
from `chamnet.config.combos.FLOW`, so the replay and the package's own sweep
cannot disagree about it.

## The tool exits 1 by design

`tools/replay.py` gates on **both** mIoU and Pillar (Pillar is the paper's
headline metric, and it can move further than mIoU on the same row — e.g.
`hd/nogate/segnext_t` is 0.01 off on mIoU and 0.44 off on Pillar; gating on
mIoU alone would miss that). Of the 36 rows `--all` produces, **28 match
exactly on both metrics** and eight do not.

**All eight are SegNeXt-T rows.** `LightHamHead` draws random numbers on
every forward pass, so this arm's evaluation output is a function of RNG
state as well as of weights and input. Evaluating at the seed the recorded
evaluations actually used narrowed the family and took one row of it to an
exact match (`hd/bigate/segnext_t`), but the other eight still differ, by
0.01-0.07 mIoU and 0.00-0.44 Pillar. Three fresh processes return *identical*
values, so what is left is a reproducible offset rather than noise.

It is **not** a discrepancy in what the ported model computes, and that is
measured rather than argued: the campaign's own invocation reproduces its
recorded numbers exactly on both metrics, the two model files involved are
byte-identical to vanilla mmsegmentation's, and both harnesses reach the test
loop with the generator in identical state. On this row a single extra RNG
draw is worth ~0.01 mIoU and ~0.05 Pillar, so a 0.01 / 0.11 residual is a
couple of draws' worth of offset. What is left is a draw counter differing
between two ways of driving an evaluation — not a fork patch, not the
mmsegmentation version, and not the model. See "SegNeXt-T's decode head draws
random numbers on every forward pass" below.

Two things that used to be on this list are not any more, and they came off it
for different reasons:

- **The eight shuffled rows.** Those arms permute the depth channel at test
  time, so the evaluation input is drawn rather than read, and for a long
  while they neither matched nor had an identified cause. All six of them that
  are not *also* SegNeXt-T rows now match exactly on both metrics. The missing
  input was the evaluation seed: a dataloader worker's numpy stream is seeded
  from `(num_workers, rank, worker_id, seed)`, so replaying at the training
  seed drew different permutations from the ones the recording saw. See
  "Shuffled depth is drawn at evaluation time".
- **Two `mit_b0` rows** (`sd/mit_b0`, `hd/nogate/mit_b0`), each previously off
  by 0.01. They match here, but **not** because of the seed — measured, not
  assumed; see "The 0.01 jitter" near the end of this document.

**`bl/resnet18`'s Pillar** lands on a 2-decimal rounding boundary and can fall
on either side — see "The `bl/resnet18` Pillar boundary" further down. It
matched in the run committed here.

None of this is a defect, and the tool reports all of it as failures rather
than being tuned to hide it. **The printed pass count is not perfectly
deterministic**: it reads 28/36 in the committed run, and moves by one for the
boundary row and for each row the ~0.01-0.02 GPU jitter happens to land on.
The rows that match carry the evidence.

Every `hd` row except SegNeXt-T matches exactly on both metrics
(`hd/resnet18` 81.88/84.01, `hd/mit_b0` 80.52/79.02, `hd/convnext_atto`
81.18/79.19 mIoU/Pillar), which is the evidence that porting the HD
backbones and their pretrained depth transfer did not change what those
models compute.

The same holds for `ef`, the early-fusion arm, whose non-SegNeXt rows match
exactly on both metrics too (`ef/resnet18` 80.48/82.00, `ef/mit_b0`
79.31/77.87, `ef/convnext_atto` 80.54/82.59 mIoU/Pillar). That is worth
stating separately because early fusion's entire difference from the
baseline is one widened stem convolution, whose fourth input channel is
initialised from the mean of the pretrained RGB filters — a detail that
changes no config key, no parameter count and no log line. Replaying the
original checkpoints through the ported code is the end-to-end check that it
was reproduced: three arms landing on the recorded numbers to the last digit
is not something a subtly different stem would produce.

## The control arms

The same holds for HD's three non-shuffled controls, on all nine of their
non-SegNeXt rows:

| arm | resnet18 | mit_b0 | convnext_atto |
|---|---|---|---|
| `hd/nogate` | 81.53 / 82.62 | 80.09 / 78.56 | 80.77 / 83.69 |
| `hd/bigate` | 81.23 / 79.97 | 79.76 / 73.71 | 81.11 / 83.19 |
| `hd/rgb` | 80.12 / 78.01 | 78.66 / 76.33 | 80.02 / 79.25 |

(mIoU / Pillar.) Every entry equals its `results_v8.csv` value to the last
digit. One of them did not in the previous committed run —
`hd/nogate/mit_b0`'s Pillar read 78.55 there against a recorded 78.56 — which
is the ~0.01 jitter described in "The 0.01 jitter" near the end of this
document, measured there and not attributable to anything about the arm.
Their SegNeXt-T rows are the exception, for the reason every SegNeXt-T row is.

This is the end-to-end check on the three mechanisms these arms rest on, none
of which changes an output shape or a parameter count in a way a smoke test
could catch:

- `nogate` is `fusion_use_gate=False` reaching `CrossModalGating`. Two of the
  four HD backbone classes did not forward that argument at all before this
  package; a version that accepted and ignored it would build a fully gated
  HD and land on HD's numbers, not on these.
- `bigate` is a different fusion module in the same dual-encoder skeleton. Its
  checkpoint has `gate_rgb`/`gate_d` tensors that CMG has no slot for, so a
  wrong class would not even load.
- `rgb` puts the RGB image into the depth-slot encoder, which is rebuilt to
  take three channels and initialised from the RGB checkpoint *unadapted* —
  where HD's own rule averages three channels into one. A stem that averaged
  here, or a 1-channel encoder, would be a different network from the one the
  checkpoint was trained as.

## Shuffled depth is drawn at evaluation time

The `hd/shuffled` and `ef/shuffled` arms measure what depth contributes beyond
its per-image statistics: `ShuffleDepthChannel` permutes the depth channel's
pixels, preserving the value multiset exactly and destroying the spatial
arrangement. The paper's configs apply it to the **train, val and test**
pipelines alike (all three dataloaders — the top-level `val_test_pipeline`
variable in those merged configs omits it, but no dataloader refers to that
variable).

Applying it at test time means the evaluation input is *drawn*, not read. The
permutation comes from `np.random.permutation` inside the dataset pipeline,
which runs in the dataloader's **worker processes**. mmengine seeds each
worker's numpy stream in `worker_init_fn`
(`mmengine/dataset/utils.py`) as

```python
worker_seed = num_workers * rank + worker_id + seed
np.random.seed(worker_seed)
```

so which permutations a run sees is a function of the worker count, the
worker id and the run seed — and of how many samples each worker has already
handled, since samples are dealt round-robin. Change the worker count and the
model is scored on different images.

It is measurable. On `hd/shuffled/resnet18`, recorded 80.97 / 80.40:

```
num_workers   mIoU    Pillar   note
8             80.91   80.52    run 1
8             80.91   80.52    run 2, separate process — identical
4             80.96   80.43    the value the paper's val/test used
0             80.93   80.42    no worker processes at all
```

The row is **deterministic** at a fixed worker count — two separate processes
agreed to the last digit — so this is not run-to-run noise.

**This package used to emit one worker count for every split, and now does
not.** The paper's configs use 8 for training and 4 for val/test; the builder
emitted 8 everywhere, and `tests/test_matches_paper.py` listed `num_workers`
among the fields it deliberately did not compare, on the grounds that the
difference was numerically harmless. It was — until an arm arrived that draws
randomness per sample. `chamnet/recipes/*.yaml` now carry `num_workers_eval`
beside `num_workers`, and the equivalence test compares every dataloader field
on all three splits.

### The worker count was one of two missing inputs; the other was the seed

Matching the worker count moved the rows and did not land them. At the time
that was as far as it went, and it was worth stating plainly, because
`hd/shuffled/resnet18` came so close that it invited the wrong conclusion:

| row | 8 workers | 4 workers (the paper's) | recorded |
|---|---|---|---|
| `hd/shuffled/resnet18` | 80.91 / 80.52 | 80.96 / 80.43 | 80.97 / 80.40 |
| `hd/shuffled/mit_b0` | 76.72 / 73.80 | 76.69 / 73.65 | 76.71 / 73.79 |
| `hd/shuffled/segnext_t` | 79.66 / 78.38 | 79.57 / 78.21 | 79.66 / 78.34 |
| `hd/shuffled/convnext_atto` | 81.10 / 79.88 | 81.07 / 79.82 | 81.10 / 79.68 |
| `ef/shuffled/resnet18` | 79.73 / 78.05 | 79.68 / 77.92 | 79.80 / 78.17 |
| `ef/shuffled/mit_b0` | 78.09 / 74.08 | 78.07 / 74.00 | 78.05 / 73.96 |
| `ef/shuffled/segnext_t` | 80.12 / 75.43 | 80.17 / 75.41 | 80.16 / 75.53 |
| `ef/shuffled/convnext_atto` | 78.81 / 76.37 | 78.79 / 76.45 | 78.85 / 76.40 |

Mean absolute gap across the eight: 0.032 mIoU / 0.092 Pillar at 8 workers,
0.045 / 0.113 at 4. The differences moved; on that measure they grew
slightly. So the worker count was demonstrably *a* cause — it changes the
numbers, reproducibly — but not a sufficient one, and one row landing nearly
on its recorded value was one row, not the family.

The change was made anyway, and would have been made even if every row had
moved further away. What the release emits has to be the config the paper
ran; choosing a dataloader setting by which numbers it flatters is the same
error as re-running until a rounding boundary falls the right way.

Both columns above were also measured at the *training* seed. Fixing that —
see the next subsection — is what turned every one of these rows exact.

### The remaining cause was the evaluation seed

The subsection this replaces said the cause was unidentified, and listed the
candidates it had ruled out. It is identified now, and it was none of them:
the replay was evaluating at the seed each checkpoint had been **trained** at,
while every recorded evaluation ran at a fixed 42 (see `RECORDED_EVAL_SEED`,
and the intro). The seed is one of the four inputs to `worker_seed` in the
`worker_init_fn` above, so a mismatched one draws different permutations — the
mechanism was in this document all along, applied to the wrong number.

With the worker count *and* the seed matched, all six non-SegNeXt shuffled
rows match their recorded values exactly on both metrics:

| row | at the training seed (37) | at the recorded evaluation seed (42) | recorded |
|---|---|---|---|
| `hd/shuffled/resnet18` | 80.96 / 80.43 | **80.97 / 80.40** | 80.97 / 80.40 |
| `hd/shuffled/mit_b0` | 76.69 / 73.65 | **76.71 / 73.79** | 76.71 / 73.79 |
| `hd/shuffled/convnext_atto` | 81.07 / 79.82 | **81.10 / 79.68** | 81.10 / 79.68 |
| `ef/shuffled/resnet18` | 79.68 / 77.92 | **79.80 / 78.17** | 79.80 / 78.17 |
| `ef/shuffled/mit_b0` | 78.07 / 74.00 | **78.05 / 73.96** | 78.05 / 73.96 |
| `ef/shuffled/convnext_atto` | 78.79 / 76.45 | **78.85 / 76.40** | 78.85 / 76.40 |

The two shuffled rows still unmatched are `hd/shuffled/segnext_t` and
`ef/shuffled/segnext_t`, and they are unmatched for the SegNeXt-T reason
rather than this one. `hd/shuffled/segnext_t` is now **exact on Pillar**
(78.34 against 78.34) and off by 0.07 on mIoU; `ef/shuffled/segnext_t` is off
by 0.01 mIoU and 0.03 Pillar. Both residuals are the size of the other
SegNeXt-T rows', not the size of the shuffled gaps they used to carry.
Nothing about the shuffle itself remains open.

Two notes on how to read this. Nothing in the release was changed to achieve
it beyond passing the right seed — the arms' code, their configs and the
worker counts are what they already were. And it upgrades what these rows are
evidence *for*: see "What these rows support" below.

### Pillar on this split rests on a handful of images

This section used to weigh an unexplained residual against the effect these
arms measure. The residual is gone, so what remains is the part that was
never about the residual: how much a *single-seed* Pillar number can be
moved by one image, which is a caution about reading any row of this file —
matched or not — as a result.

The effect each shuffled arm measures, comparing it against its own
unshuffled arm *within this same replay*, so no cross-run comparison is
involved:

| row | effect of shuffling (mIoU / Pillar) |
|---|---|
| `ef/shuffled/convnext_atto` | −1.69 / −6.19 |
| `hd/shuffled/mit_b0` | −3.81 / −5.23 |
| `ef/shuffled/segnext_t` | −0.65 / −5.03 |
| `ef/shuffled/mit_b0` | −1.26 / −3.91 |
| `hd/shuffled/segnext_t` | −1.00 / −3.86 |
| `ef/shuffled/resnet18` | −0.68 / −3.83 |
| `hd/shuffled/resnet18` | −0.91 / −3.61 |
| **`hd/shuffled/convnext_atto`** | **−0.08 / +0.49** |

On the first seven, shuffling costs 3.6-6.2 points of Pillar. **The last row
is the exception**: at this seed the arm shows no effect at all.

The cause is known, and it is not the shuffled arm. Across the ten recorded
seeds, `hd/convnext_atto` minus `hd/shuffled/convnext_atto` on test Pillar is
−4.50 ± 2.76 (p = 0.0006), and **seed 37 is the only one of the ten where the
sign flips**; the other nine run −2.68 to −9.36. It is the HD *baseline* at
seed 37 that is anomalous: against its own nine-seed distribution that run
sits at z = −4.45 on Pillar, where fifteen of the sixteen flow×backbone
combinations sit between z −0.87 and +1.35, and the anomaly is confined to
that one class (every other class z +0.26 to +1.50; val mIoU z = +0.74, test
mIoU z = +0.11 — the model itself is fine). So this row's near-zero effect is
a property of the checkpoint the replay happens to load, not evidence about
ConvNeXt-Atto's depth dependence, and the ten-seed result for that arm points
the same way as the other seven.

That is worth reading past this one row, because of *why* the baseline moves
so far. **Pillar appears in only 14 of the 45 test images**, and it is
concentrated even among those: the largest single image holds 13.5% of the
split's pillar ground-truth pixels and the top five hold 55%. One image,
`0526_rfv7_069s`, is 7.4% of the pillar ground truth but 8.6% of seed 37's
accumulated union — inflated precisely because that checkpoint predicted it
badly, scoring 46.80 on it against seed 39's 66.46. Excluding that one image
takes seed 37's aggregate Pillar from 79.19 to 83.52 and seed 39's from 84.45
to 86.14; the 4.33-point drag it puts on seed 37 is larger than that run's
entire 3.66-point deficit against the other nine. Aggregate Pillar IoU on this
split therefore rests on a handful of images, and **a single-seed Pillar
number read out of `replay.csv` — any row, not just this one — carries that
sensitivity**. The paper's numbers are ten-seed means, which is the level at
which the comparison is stable.

None of this bears on whether the rows reproduce. They do; it bears on what a
single row can be used to argue.

### One root cause, two routes — and it was never RNG *history*

The guess this section used to attack was that both un-matchable families had
one root cause: that the process which wrote `results_v8.csv` *trained before
testing*, so it reached every random draw with a different amount of RNG
already consumed. It turns out to have been half right and wrong about the
mechanism, and both halves are worth keeping straight.

**The root cause was shared.** It was the evaluation seed, for both families
at once. Where the two differ is only in how the seed reaches the draw:
SegNeXt-T's is a `torch` draw in the main process, seeded directly by
`randomness.seed`; the shuffle's is a `numpy` draw in a worker, seeded from
`(num_workers, rank, worker_id, seed)`. One number, two routes.

**But it was not RNG history in either.** The recorded evaluations did not
train first — they were separate `tools/test.py` processes that only ever ran
a test loop (see the intro). And for the shuffle, main-process history could
not have reached the draw even if they had. That was tested rather than
argued, with a positive control: consuming 10 million numpy draws in the main
process immediately before `runner.test()`, on `hd/shuffled/resnet18`:

```
num_workers   burn   mIoU    Pillar   changed?
4             none   80.96   80.43
4             1e7    80.96   80.43    no
0             none   80.93   80.42
0             1e7    80.97   80.41    yes
```

At `num_workers=0` the transform runs in the main process, so the burn moves
the result — which is what makes the negative result above informative rather
than a broken experiment. At the worker counts any real run uses, main-process
RNG history cannot reach the shuffle at all. (Those numbers were taken at the
training seed, before the seed was corrected; the control they establish is
about *history*, not about which seed, so correcting the seed does not
disturb it.)

The measurements in this section are one-off diagnostics on the same footing
as the rounding table further down, and **are not shipped**: they reused
`tools/replay.py`'s own pieces, overriding
`cfg.test_dataloader['num_workers']` and, for the burn, calling
`np.random.random(n)` between `Runner.from_cfg` and `runner.test()`.

### What these rows support

They used to be the weakest rows in the file: scored on a differently
permuted version of the same data, so not evidence that the ported code
computes the same thing. That is no longer the case. Six of the eight land on
their recorded values exactly, on both metrics, which means the replay is
drawing the *same permutation* the recording drew — so these rows now carry
the same code-equivalence weight as the unshuffled arms, and they carry it
over a longer chain: the transform, its seeding, the worker layout and the
dataloader settings all have to agree, not just the model.

The two that do not land are the two SegNeXt-T ones, for the SegNeXt-T
reason.

It is also worth writing down that this closed *because* the tool kept
reporting these rows as failures. They had matched, at 0.03-0.25 of a point,
closely enough that widening the gate or writing them off as noise would have
been easy and would have looked reasonable. Nothing would have failed
afterwards, and the evaluation seed — which is a fact about how every recorded
number in the campaign was produced, not just about eight replay rows — would
not have been found.

## SegNeXt-T's decode head draws random numbers on every forward pass

`LightHamHead`'s `NMF2D._build_bases` (`mmseg/models/decode_heads/ham_head.py`,
lines 89-90 and 123) calls `torch.rand((B * S, D, R))` **unconditionally on
every forward pass** whenever `rand_init=True` — which is the setting in
every SegNeXt-T config in this project, identically in the original merged
config and this package's builder output. SegNeXt-T's segmentation logits
are therefore a function of RNG state as well as of weights and input; this
is mmseg's own (unmodified, vanilla) `LightHamHead` implementation, not
something this port changed.

That draw is taken **on the CPU from the global default generator** — the
tensor is created and only then moved to the device, with no device
generator and no local `Generator` anywhere in the path — so it is fully
determined by the last `torch.manual_seed` and by how many draws have been
taken since. Two consequences, and they point in opposite directions.

**Replay-to-replay it does reproduce, exactly.** `tools/replay.py` builds a
fresh `Runner` for every row, and `Runner.__init__` re-seeds the global RNG
from `randomness.seed`, so each row is a pure function of (seed, weights,
data) and inherits nothing from the rows before it. That is stronger than
"two processes did the same work in the same order": the committed
`replay.csv` runs replayed a **different number of rows in a different
order** — 12, then 16, then 36 — with `sd/segnext_t` moving from the 7th
model built to the 11th, four early-fusion models' worth of intervening
draws further in, and still agreed exactly. Per-row re-seeding is what
explains that; process identity has nothing to do with it.

**Against the recorded numbers it now nearly does, and the gap that is left
is a reproducible offset.** The previous version of this section explained
the gap by saying the recording's process "seeded once and then *trained*",
reaching the head's draw after thousands of intervening draws. **That was
wrong on the facts.** The recorded evaluations were separate `tools/test.py`
processes that never trained; what differed was the *seed* they ran at (see
the intro and `RECORDED_EVAL_SEED`). Replaying at that seed narrows every
SegNeXt-T row and closes one of the nine exactly:

| row | at 37 (mIoU / Pillar) | at 42 | recorded |
|---|---|---|---|
| `hd/bigate/segnext_t` | 80.88 / 82.73 | **80.84 / 82.67** | 80.84 / 82.67 |
| `bl/segnext_t` | 80.76 / 77.63 | 80.81 / 77.94 | 80.80 / 77.83 |
| `ef/segnext_t` | 80.86 / 80.67 | 80.82 / 80.59 | 80.84 / 80.59 |
| `sd/segnext_t` | 80.54 / 80.47 | 80.57 / 80.42 | 80.51 / 80.29 |
| `hd/segnext_t` | 80.52 / 82.21 | 80.59 / 82.20 | 80.62 / 82.31 |
| `hd/nogate/segnext_t` | 79.64 / 79.20 | 79.68 / 79.30 | 79.69 / 79.74 |
| `hd/shuffled/segnext_t` | 79.57 / 78.21 | 79.59 / 78.34 | 79.66 / 78.34 |
| `hd/rgb/segnext_t` | 80.83 / 78.89 | 80.83 / 78.79 | 80.84 / 78.81 |
| `ef/shuffled/segnext_t` | 80.17 / 75.41 | 80.17 / 75.56 | 80.16 / 75.53 |

The eight that remain are off by 0.01-0.07 mIoU and 0.00-0.44 Pillar. The
largest is `hd/nogate/segnext_t`'s Pillar; the widest gap on any row was 0.54
before the seed was corrected, so the family narrowed but did not close.

### Reproducible offset, or jitter? — measured

Those residuals are the size of the ~0.01-0.02 GPU jitter documented near the
end of this file, so the two have to be told apart rather than assumed. They
are distinguishable: jitter moves between fresh runs at a fixed
configuration, while an RNG-trajectory offset is exactly reproducible. Three
fresh processes per row, nothing changed between them:

```
bl/segnext_t          80.81 / 77.94    80.81 / 77.94    80.81 / 77.94
hd/nogate/segnext_t   79.68 / 79.28    79.68 / 79.30    79.68 / 79.30
```

`bl/segnext_t` is **identical three times out of three**, against a recorded
80.80 / 77.83. Its +0.01 / +0.11 is therefore not noise: it is a reproducible
offset, and something about this evaluation still differs from the recorded
one. `hd/nogate/segnext_t` shows both components at once — its mIoU is
identical three times, and its Pillar moves by 0.02 across the three, riding
on top of a −0.44 to −0.46 offset that is twenty times larger than that
movement.

So the honest state of this family: **the evaluation seed was the whole story
for the shuffled arms and only part of it for SegNeXt-T.** What is left is
smaller and sharper than the question this document started with — the seed
is matched, the worker count is matched, the weights are the recorded ones,
and the residual is reproducible. The next three sections narrow it to a
draw counter and say what that does and does not leave open.

### The recorded numbers reproduce exactly — by the command that recorded them

Stated first, because it is the strongest fidelity result in this document and
it bounds everything below it. Running the campaign's own invocation — its
config file, its checkpoint, its `--cfg-options`, in the research repository
and against the mmsegmentation fork that repository carries — returns the
recorded numbers on **both** metrics, to every digit mmengine prints:

```
mIoU: 80.8000        per-class table:  | pillar | 77.83 |
recorded             80.80                       77.83
```

So the residual on the eight rows is **not** evidence that anything about the
ported model differs. The model code is the same code: `ham_head.py` (the file
that contains the RNG-consuming `_build_bases`) and `mscan.py` are
**byte-identical** between the fork and vanilla mmsegmentation 1.2.2, and the
config the release builds matches the campaign's key for key on every
evaluation-relevant key (`tests/test_matches_paper.py`).

### What the residual is, and is not

Six things have been measured. Taken together they put the residual in one
place and take it out of every other.

**Not the model, and not the metric.** `models/decode_heads/ham_head.py` (the
file containing the RNG-consuming `_build_bases`), `models/backbones/mscan.py`
and `evaluation/metrics/iou_metric.py` are all **byte-identical** between the
research fork and vanilla mmsegmentation 1.2.2.

**Not the mmsegmentation version.** Driven the same way — `Runner.from_cfg`,
`load_from`, `.test()` — the fork and vanilla both return **80.81**. So
fork-versus-vanilla is not what separates 80.81 from the recorded 80.80.

**Not differing RNG consumption before the loop.** This was the candidate this
section used to carry, and it is falsified. `torch.rand(1)` drawn immediately
after `Runner.from_cfg` and immediately before `.test()` returns the **same
value to sixteen digits** (`0.924892008304596`) in both harnesses. Nothing
about model construction, weight initialisation or config routing consumes a
different number of draws.

**Not the visualization hook.** Same config, same seed, same worker count,
with `default_hooks.visualization` popped and with it kept: **both 80.81**.

**Not the pipeline transforms, and this is the argument that matters.**
`datasets/transforms/{loading,formatting,transforms}.py` — which supply
`LoadAnnotations`, `PackSegInputs` and `Resize` — do differ between the fork
and vanilla, and were the obvious remaining suspects. They are ruled out by
**which rows differ**: those three transforms are on the evaluation path of
*every one of the 36 rows*, and 28 rows match exactly, including every
non-SegNeXt row — all six non-SegNeXt shuffled arms, and every BL/EF/SD/HD
row on the other three backbones. A patch that changed the computation could
not confine its effect to one backbone family; every row would move. They
don't.

**It is the driving path, and it acts inside the loop.** With the config
controlled (the campaign's base config plus its one `--cfg-options` differs
from the dumped merged config only in `randomness.seed`, which was matched),
the same fork returns 80.80 driven by `tools/test.py` and 80.81 driven by
`Runner.from_cfg`. That is the one variable left standing — and since the
generator state is identical at loop entry, whatever it does, it does after
that point.

### So the residual is RNG bookkeeping, not a difference in what is computed

SegNeXt-T is the only architecture here that reads randomness during a forward
pass, which is the only mechanism that can produce a difference confined to
SegNeXt-T rows. And the row is sensitive to single draws: inserting that one
probe `torch.rand(1)` before `.test()` moved `bl/segnext_t` from 80.81 / 77.94
to **80.80 / 77.99** — so **one draw is worth about 0.01 mIoU and 0.05 Pillar
here** (a single observation of the perturbation; the unperturbed value is the
one confirmed three times). A residual of 0.01 / 0.11 is a couple of draws'
worth of offset, which is exactly the size observed.

So the two driving paths differ in how many draws are taken from the CPU
default generator somewhere inside the test loop, and nothing else. **No
evaluation-path fork patch needs replicating for numerical fidelity**, and
bisecting the four differing files is not worth the GPU time: it would
identify which file shifts a draw counter, and change no number the release
computes. The only fork patch established as necessary for these methods
remains the 4-channel `bgr_to_rgb` fix the release ships itself (see
`chamnet/models/data_preprocessor.py`).

**The limit of that conclusion, stated plainly.** It is an inference from
*which rows differ*, not a direct isolation: nobody has pointed at the line
that takes the extra draw. Two observations would falsify it — a non-SegNeXt
row developing a residual, or a SegNeXt-T residual appearing on a quantity
that cannot depend on the head's draws. Either would mean a fork patch does
change the computation after all, and would reopen this.

### Why the replay is not re-driven through `tools/test.py`

It might well close these eight rows: that path is the one that returns 80.80.
The reason not to is not that it wouldn't work.

The replay exists to exercise **the release's own** evaluation path — the one
`chamnet test` and `chamnet sweep` actually use. Driving it with the
campaign's script instead would buy eight rows of cosmetic agreement and stop
the artifact testing the code the release ships, which is the whole point of
having it. A code-equivalence check that runs different code than the package
is not a check.

And the thing it would be papering over is, by everything above, a draw
counter rather than a computation. So the rows stay reported as failures, and
the tool's exit code keeps saying so.

What this artifact can and can't support: it is a code-equivalence check —
evidence that this port computes the same thing the original code did, for
the same fixed weights — not a statement about the statistical reliability
of any published result. It says nothing about how a ±0.1 Pillar difference
compares to the spread across independently trained runs; that would need
its own measurement against however many training seeds actually back the
numbers being defended, which is outside what a single-checkpoint replay
can establish. The SegNeXt-T residual in particular is not a sample from a
distribution at all — it is a fixed offset, now measured to be reproducible,
so its magnitude is **one observation per row, not a spread**, and reading it
as an error bar would be a mistake in either direction. What this does
support: **SegNeXt-T's evaluation output depends on RNG state, not on weights
and input alone** — which is why its rows needed the recorded evaluation seed
to come this close, and why the tool's exit code reports the remainder
honestly rather than being tuned to report success.

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

## The 0.01 jitter

Anything genuinely larger than this on `resnet18`, `mit_b0` or
`convnext_atto` would not fit the pattern. Those three backbones have shown
small single-row differences in some replay runs — up to about 0.01 on
either metric, on one row at a time — that later reruns did not reproduce
(the same row landing at exactly 0.00 on a subsequent run with no code
change in between). The **previous** committed `replay.csv` contained two,
both on mit_b0: `sd` read 79.33 mIoU against a recorded 79.32, and
`hd/nogate` read 78.55 Pillar against 78.56. Each was replayed again, twice,
immediately afterwards, and every one of those four re-runs returned the
recorded values to the last digit (79.32/80.40 and 80.09/78.56) with no code
change in between — which is the pattern rather than an exception to it.
Neither arm's code moved either: `sd`'s four backbone classes, and the
`nogate` path on mit_b0, are unchanged in the commit this replay ran against.

Both rows match in the `replay.csv` committed now, which was also the run
that corrected the evaluation seed — and it would be easy, and wrong, to
credit the seed for it. Neither arm has any consumer of test-time randomness:
mit_b0 has no `rand_init` decode head and neither row is a shuffled arm, so
the seed should make no difference to them at all. Measured, rather than
argued from that: `sd/mit_b0` replayed twice at each seed, in four fresh
processes.

```
eval seed 37    79.32 / 80.40      79.32 / 80.40
eval seed 42    79.32 / 80.40      79.32 / 80.40
recorded        79.32 / 80.40
```

Four out of four on the recorded value, and **no dependence on the seed**.
So the 79.33 in the earlier file was the jitter excursion and 79.32 is the
row's value; the seed correction did not fix it, because there was nothing
for the seed to fix. The same reading applies to `hd/nogate/mit_b0` by the
same argument and by its own four earlier re-runs, though that row was not
re-measured under seed control in this batch.

This is ordinary GPU/cudnn non-determinism under
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
