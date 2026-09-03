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

Every row is a **single seed** (37). Before reading a Pillar number out of it
as a result rather than as a code-equivalence check, see "The residual's size,
in proportion" below: pillar appears in only 14 of the 45 test images, so
aggregate Pillar IoU on this split is sensitive to a handful of them, and one
of the rows here is visibly affected. The paper's numbers are ten-seed means.

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
reverse) is an error rather than a silently missing row.

## The tool exits 1 by design

`tools/replay.py` gates on **both** mIoU and Pillar (Pillar is the paper's
headline metric, and it can move further than mIoU on the same row — e.g.
`bl/segnext_t` was 0.04 off on mIoU but 0.20 off on Pillar; gating on mIoU
alone would have missed that). Of the 36 rows `--all` produces, **15 do not
match**, for two different reasons that are worth keeping apart, and one more
is a coin flip:

- **The nine SegNeXt-T rows** — one per arm. `LightHamHead` draws random
  numbers on every forward pass, from the *main* process's generator, and the
  recorded values were produced after a full training run's worth of draws.
  That trajectory is gone; these rows **cannot** be matched by a
  test-only replay, however carefully it is configured. Explained below.
- **The eight shuffled rows** (`hd/shuffled` and `ef/shuffled`, all four
  backbones each) — those arms shuffle the depth channel *at test time*, so
  the input is drawn rather than read. This is a weaker statement than the
  one above and should not be read as the same: the draw happens in the
  dataloader's worker processes, which mmengine re-seeds from
  `(num_workers, rank, worker_id, seed)`, and `ShuffleDepthChannel` is the
  only source of randomness in the test pipeline — so each permutation is a
  deterministic function of quantities that are all in the config. **These
  rows are reproducible in principle; they have not been reproduced, and what
  else differs is not identified.** See "Shuffled depth is drawn at evaluation
  time" below. Two rows are in both sets (`hd/shuffled/segnext_t`,
  `ef/shuffled/segnext_t`), hence 9 + 8 − 2 = 15.
- **`bl/resnet18`'s Pillar** lands on a 2-decimal rounding boundary and can
  fall on either side — see "The `bl/resnet18` Pillar boundary" further down.
  It matched in the run committed here.

None of these is a defect, and the tool reports all of them as failures rather
than being tuned to hide them.

**The printed pass count is therefore not deterministic.** It is 20/36 or
21/36 depending on which side of that boundary the ResNet row lands on, minus
one for each row the ~0.01 GPU jitter described at the end of this section
happens to land on. The committed run reads **19/36**: the boundary row
matched, and the jitter took two rows (`sd/mit_b0` and `hd/nogate/mit_b0`,
both re-verified — see the end of this section). The rows that matched carry
the evidence.

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

The same holds for HD's three non-shuffled controls, on eight of their nine
non-SegNeXt rows:

| arm | resnet18 | mit_b0 | convnext_atto |
|---|---|---|---|
| `hd/nogate` | 81.53 / 82.62 | 80.09 / **78.55** | 80.77 / 83.69 |
| `hd/bigate` | 81.23 / 79.97 | 79.76 / 73.71 | 81.11 / 83.19 |
| `hd/rgb` | 80.12 / 78.01 | 78.66 / 76.33 | 80.02 / 79.25 |

(mIoU / Pillar.) Every entry equals its `results_v8.csv` value to the last
digit except the bold one, `hd/nogate/mit_b0`'s Pillar, which reads 78.55
against a recorded 78.56 — the ~0.01 flutter described at the end of the
previous section, confirmed by two immediate re-runs that both returned 78.56.

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

### Matching the worker count did not reproduce the recorded permutations

Worth stating plainly, because the row above invites the opposite conclusion.
`hd/shuffled/resnet18` does move onto the recording when the worker count is
matched — from (−0.06, +0.12) to (−0.01, +0.03). The other seven do not:

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
0.045 / 0.113 at 4. The differences moved; they did not shrink, and on this
measure they grew slightly. So the worker count is demonstrably *a* cause of
the differences — it changes the numbers, reproducibly — but matching it is
not sufficient to recover the recorded permutations, and one row landing
nearly on its recorded value was a single row, not the family.

The change was made anyway, and would have been made even if every row had
moved further away. What the release emits has to be the config the paper ran;
choosing a dataloader setting by which numbers it flatters is the same error
as re-running until a rounding boundary falls the right way.

### The remaining cause is unidentified

Stated plainly, because the paragraphs above could be read as having settled
it. Every input to the permutation is known — `worker_seed` is a pure function
of `num_workers`, `rank`, `worker_id` and `seed`; `ShuffleDepthChannel` is the
only draw anywhere in the test pipeline; the sampler is `DefaultSampler` with
`shuffle=False`, so each worker sees the same samples in the same order — and
all four are now matched to the paper's configs. The rows still differ.
**Why is not known.** It is not main-process RNG history (falsified below with
a positive control), and it is not the worker count alone (the table above).
Candidates not yet eliminated include the rank the recorded run used and
anything in the recorded pipeline that consumed a worker's numpy stream
without appearing in the merged config. Until someone rules those out, this is
an open question, not a closed one, and the sentence above about these rows
being reproducible in principle is a claim about the mechanism rather than a
result.

### The residual's size, in proportion — with one exception

The open question matters less than it might, on seven of the eight rows.
These arms exist to show what depth's spatial arrangement is worth, and the
effect they measure is one to two orders of magnitude larger than the residual.
Comparing each shuffled arm against its own unshuffled arm *within this same
replay*, so no cross-run comparison is involved:

| row | effect of shuffling (mIoU / Pillar) | unexplained residual |
|---|---|---|
| `ef/shuffled/convnext_atto` | −1.75 / −6.14 | −0.06 / +0.05 |
| `ef/shuffled/segnext_t` | −0.69 / −5.26 | +0.01 / −0.12 |
| `hd/shuffled/mit_b0` | −3.83 / −5.37 | −0.02 / −0.14 |
| `ef/shuffled/resnet18` | −0.80 / −4.08 | −0.12 / −0.25 |
| `hd/shuffled/segnext_t` | −0.95 / −4.00 | −0.09 / −0.13 |
| `ef/shuffled/mit_b0` | −1.24 / −3.87 | +0.02 / +0.04 |
| `hd/shuffled/resnet18` | −0.92 / −3.58 | −0.01 / +0.03 |
| **`hd/shuffled/convnext_atto`** | **−0.11 / +0.63** | **−0.03 / +0.14** |

On the first seven, shuffling costs 3.6-6.1 points of Pillar and the residual
is 0.03-0.25. A conclusion of the form "destroying depth's spatial arrangement
costs several points of Pillar" is not in any danger from a hundredth-scale
disagreement about which permutation was drawn.

**The last row is the exception, and it is not protected by that argument.**
On `hd/shuffled/convnext_atto` shuffling *raised* Pillar by 0.63 and moved mIoU
by −0.11 — at this seed the arm shows no effect at all — and the residual
(0.03 / 0.14) is the same order as the "effect". Nothing about that row can be
supported by this replay in either direction: the quantity being measured and
the quantity that is unexplained are the same size.

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
the same way as the other seven. It remains true that *this replay row*
supports nothing either way.

That is worth reading past this one row, because of why the baseline moves so
far. **Pillar appears in only 14 of the 45 test images**, and it is
concentrated even among those: the largest single image holds 13.5% of the
split's pillar ground-truth pixels and the top five hold 55%. One image,
`0526_rfv7_069s`, is 7.4% of the pillar ground truth but 8.6% of seed 37's
accumulated union — inflated precisely because that checkpoint predicted it
badly, scoring 46.80 on it against seed 39's 66.46. Excluding that one image
takes seed 37's aggregate Pillar from 79.19 to 83.52 and seed 39's from 84.45
to 86.14; the 4.33-point drag it puts on seed 37 is larger than that run's
entire 3.66-point deficit against the other nine. Aggregate Pillar IoU on this
split therefore rests on a handful of images, and **a single-seed Pillar number
read out of `replay.csv` — any row, not just this one — carries that
sensitivity**. The paper's numbers are ten-seed means, which is the level at
which the comparison is stable.

None of this touches the unidentified residual above. It explains why one
shuffled row shows no effect; it says nothing about why the shuffled rows do
not land on their recorded values, which stays open.

### It is not the same phenomenon as the SegNeXt-T offset

A natural guess is that both un-matchable families have one root cause: the
process that wrote `results_v8.csv` *trained before testing*, so it reached
every random draw with a different amount of RNG already consumed. That is
exactly right for SegNeXt-T, whose draw is taken in the main process (next
section). **It is false for the shuffle**, and the reason is the
`worker_init_fn` above: a worker's numpy stream is re-seeded from
`(num_workers, rank, worker_id, seed)` and inherits nothing from the parent
process's generator state.

Tested rather than argued from the source, with a positive control. Consuming
10 million numpy draws in the main process immediately before `runner.test()`,
on `hd/shuffled/resnet18`:

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
RNG history cannot reach the shuffle at all. (The `0 / 1e7` row happens to land
on the recorded mIoU. That is not evidence of anything: with the transform
in-process the score is a function of an arbitrary RNG offset, so *some* offset
hits any given value, and the paper's runs used 4 workers, where this mechanism
is inactive.)

So the two families stay two: a main-process draw inside the model whose
trajectory cannot be recovered, and a worker-process draw over the input whose
inputs are known but whose recorded realisation is not.

### What these rows can and cannot support

They are **not** evidence that the ported code computes the same thing, because
the input is not the same tensor. What they show is that the arms build, load
their checkpoints and score within roughly a tenth of a point of the recording
on a differently-permuted version of the same data — which is the expected
behaviour of a control whose whole premise is that the arrangement carries no
information, and is worth having, but it is a weaker claim than the exact
agreement the other arms reach.

Nor are they a closed question. The other un-matched family, SegNeXt-T's, is
closed in the sense that matters: the mechanism is known and the recorded
trajectory is unrecoverable, so no future run will do better. These eight are
not like that. Every input to the permutation is in the config, so a run that
matched them is possible; nobody has produced one, and the reason is
unidentified. If a later change makes these rows land exactly, that is a
finding to write down, not a coincidence to pass over.

The measurements in this section are one-off diagnostics on the same footing
as the rounding table further down, and **are not shipped**: they reused
`tools/replay.py`'s own pieces, overriding `cfg.test_dataloader['num_workers']`
and, for the burn, calling `np.random.random(n)` between `Runner.from_cfg` and
`runner.test()`.

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
from `randomness.seed` (37 here), so each row is a pure function of (seed,
weights, data) and inherits nothing from the rows before it. Five replays
have now produced identical SegNeXt-T numbers to the last digit (`bl`
80.76/77.63, `sd` 80.54/80.47, `hd` 80.52/82.21 mIoU/Pillar), and
`ef/segnext_t` (80.86/80.67) has now reproduced itself across three of them.
That is stronger than "two processes did the same work in the same order":
the committed `replay.csv` runs replayed a **different number of rows in a
different order** — 12, then 16, then 36 — with `sd/segnext_t` moving from
the 7th model built to the 11th, four early-fusion models' worth of
intervening draws further in, and still agreed exactly. Per-row re-seeding is
what explains that; process identity has nothing to do with it.
`hd/bigate/segnext_t` and `hd/rgb/segnext_t` have now reproduced across two
runs as well. (`hd/nogate/segnext_t` moved 79.22 -> 79.20 between those two
runs and `hd/rgb/segnext_t` 80.82 -> 80.83, which is the ~0.01-0.02 GPU
flutter riding on top of the RNG offset, not a second RNG effect: the worker
count changed between those runs and cannot reach a draw taken in the main
process.)

**Against the recorded numbers it cannot.** The process that wrote
`results_v8.csv` seeded once and then *trained*, reaching `_build_bases` in
its test loop after many thousands of intervening draws. It is on a
different RNG trajectory, and re-running the replay does not move onto that
trajectory — which is why these nine rows differ, why the difference is a
fixed offset rather than run-to-run noise, and why no amount of re-running
will close it. Consistent with that: rerunning `bl/segnext_t` with
`randomness.seed=42` (the seed the original recorded run used, versus this
replay's 37) moved the observed gap from 0.04/0.20 (mIoU/Pillar) to
0.01/0.11 — closer, in the expected direction, but still not exact, because
matching the seed does not match how much of that seed's sequence had been
consumed before `_build_bases` was reached.

What this artifact can and can't support: it is a code-equivalence check —
evidence that this port computes the same thing the original code did, for
the same fixed weights — not a statement about the statistical reliability
of any published result. It says nothing about how a ±0.2 Pillar difference
compares to the spread across independently trained runs; that would need
its own measurement against however many training seeds actually back the
numbers being defended, which is outside what a single-checkpoint replay
can establish. The SegNeXt-T gap in particular is not a sample from a
distribution at all — it is the fixed offset between two RNG trajectories,
so its magnitude is **one observation per row, not a spread**, and reading it
as an error bar would be a mistake in either direction. Across the seven
SegNeXt-T rows not also confounded by a shuffled input, it currently runs
0.01-0.10 on mIoU and 0.06-0.54 on Pillar; the 0.54 is `hd/nogate/segnext_t`
(79.20 replayed against 79.74 recorded) and is the largest offset seen on any
row so far, larger than the 0.20 that was the maximum when only four
SegNeXt-T rows existed. Nothing distinguishes that
row mechanically — it is the same head, the same seed and the same
trajectory mismatch — so the honest reading is that the offset's size is
simply not bounded by the four values that happened to be measured first.
What this does support: **SegNeXt-T's evaluation output depends on RNG state,
not on weights and input alone** — so nobody should expect a replay's
SegNeXt-T row to match a number that `results_v8.csv` recorded at the end of
a training run, even though that same replay row reproduces itself exactly. A
gap on those nine rows specifically is the expected outcome, and the tool's
exit code reflects that honestly rather than being tuned to always report
success.

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
change in between). The `replay.csv` committed here contains two, both on
mit_b0: `sd` reads 79.33 mIoU against a recorded 79.32, and `hd/nogate` reads
78.55 Pillar against 78.56. Each was replayed again, twice, immediately
afterwards, and every one of those four re-runs returned the recorded values
to the last digit (79.32/80.40 and 80.09/78.56) with no code change in
between — which is the pattern rather than an exception to it. Neither arm's
code moved either: `sd`'s four backbone classes, and the `nogate` path on
mit_b0, are unchanged in the commit this replay ran against. This is ordinary GPU/cudnn non-determinism under
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
