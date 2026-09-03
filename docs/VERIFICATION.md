# Verification

The greenhouse dataset and the trained checkpoints behind the published
numbers are not distributed with this repository — they live on a private
training server. So the question this document exists to answer is the one an
outside reader is left with: *given that I cannot run any of this, what was
checked, how, and what would it have failed to notice?*

Three layers were used. They are listed in the order they were built, which
is also the order of increasing cost and decreasing blindness.

| | layer | what it compares | artifact |
|---|---|---|---|
| 1 | Config equivalence | the config this package emits vs the merged config each published run was produced from | `tests/test_matches_paper.py`, `tests/fixtures/paper/` |
| 2 | Checkpoint replay | published checkpoints, loaded into models this package builds, scored on the same split against the recorded metrics | `tools/replay.py`, `verification/replay.csv`, `verification/README.md` |
| 3 | Retraining | models trained from scratch by this package, scored against the recorded distribution of the same condition | `tools/retrain_verify.py`, `verification/retrain.csv`, this document |

## Layer 1 — config equivalence

36 merged configs, one per published arm × backbone, dumped from the runs
that produced the numbers, are committed under `tests/fixtures/paper/`. For
each of them `build_config` is asked for the same combination and the two
are compared key by key: `model`, `optim_wrapper`, `param_scheduler`,
`train_cfg`, `custom_hooks` (including the early-stopping patience),
`randomness`, `val_evaluator`, `test_evaluator`, `default_hooks`,
`default_scope`, and all three dataloaders including their full pipelines.
The comparison is a deny-list rather than an allow-list, so a key added later
is compared unless someone deliberately exempts it.

One field is normalised — the depth file suffix, which the paper's own
configs wrote two different ways — and both sides are asserted *before* the
normalisation, so normalising it does not quietly remove it from coverage.

**What this catches:** a setting renamed, dropped, defaulted, or given a
different value. Widening the comparison to the list above — it began as four
keys — turned up four real divergences between what this package emitted and
what the published runs used, among them a missing `default_scope`, a missing
visualization hook, and an evaluator configured for one metric where the
published runs asked for three.

**What it cannot catch:** anything about what the code *does* with those
settings. Two different classes registered under the same type string are
indistinguishable to it, and so is a class that accepts an argument and
ignores it.

## Layer 2 — checkpoint replay

`tools/replay.py` loads a published checkpoint into a model built by this
package's code, runs it over the same test split, and compares the result to
the metrics recorded when that checkpoint was originally scored. Same
weights, different (ported) code. 36 rows; the gate is equality at the two
decimals the recorded file stores, on **both** mIoU and Pillar. 28 of the 36
match exactly on both. Which rows do not, and why, is documented at length in
`verification/README.md` — the tool deliberately exits non-zero rather than
being tuned to report success.

**What this catches:** any change in the forward pass, the fusion module, the
stem channel adaptation, the transforms, the dataloader layout, and the
seeding that feeds them. It is how the fusion refactor, the four heavy-depth
backbones, the early-fusion stem's fourth-channel initialisation and the
depth-structure controls were checked — several of which change no config
key, no parameter count and no log line, and so are invisible to layer 1.

**What it cannot catch:** anything that only happens while weights are being
*created*. Initialisation, the optimiser, the learning-rate schedule, which
checkpoint gets selected, and whether a checkpoint this package writes can be
read back again are all outside it, because it loads checkpoints that another
program wrote.

## Layer 3 — retraining from scratch

Four combinations trained from nothing by this package's own sweep, then
scored against the recorded distribution of the same condition. That is what
the rest of this document is about.

### Why this layer exists, concretely

Both earlier layers were green — repeatedly, over the whole course of the
port — while a defect sat in this release that made **every checkpoint it
writes unreadable**. A run would train to completion and then die at its
first evaluation.

The mechanism: torch 2.6 made `weights_only=True` the default for
`torch.load`; an mmengine checkpoint carries pickled logging state
(`HistoryBuffer` objects wrapping numpy arrays) alongside its tensors, which
the safe unpickler refuses; and mmengine 0.10.7 calls `torch.load` with no
`weights_only` argument at all, so there was no setting to pass through.

Neither earlier layer could see it:

* Layer 1 was blind because no config key encodes it.
* Layer 2 was blind because it loads checkpoints *another program wrote*, and
  it had already relaxed the unpickler for its own loads — process-wide, from
  import time, in an early revision — so it never exercised the path the
  package's own commands take.

Only training a model, writing a checkpoint, and reading it back reaches that
layer. The fix is `chamnet/checkpoint.py`, now used by both `chamnet test`
and the sweep, narrowed to the single load rather than left on process-wide.

The lesson cuts both ways, and it applies to this layer too. **A green result
here means no defect large enough to move a condition's test mIoU outside the
band ten recorded seeds occupy, and no divergence in the optimisation path
big enough to show up against its recorded loss curve. It does not mean there
is none.** What each layer cannot see is as much a part of the result as what
it can — and this layer's own blind spot, measured below, is that a single
run's Pillar cannot be held to that kind of band at all.

### The criterion, stated before the results

Training runs with `deterministic=False`, so a fresh run does not reproduce
an earlier run's digits, and no fixed tolerance around one earlier run is
defensible. This is what the recorded runs actually do across the ten
training seeds behind every published mean (test split, seeds 31–40):

| condition | metric | seed 37 | mean | SD | min | max |
|---|---|---:|---:|---:|---:|---:|
| `hd/resnet18` | mIoU | 81.88 | 81.655 | 0.600 | 80.15 | 82.20 |
| | Pillar | 84.01 | 83.085 | 1.093 | 81.54 | 84.28 |
| `bl/resnet18` | mIoU | 80.95 | 80.589 | 0.553 | 79.82 | 81.51 |
| | Pillar | 80.00 | 79.867 | 0.873 | 79.03 | 81.21 |
| `hd/resnet18` shuffled | mIoU | 80.97 | 80.658 | 0.301 | 80.13 | 80.97 |
| | Pillar | 80.40 | 80.037 | 0.640 | 78.60 | 80.85 |
| `ef/mit_b0` | mIoU | 79.31 | 79.442 | 0.714 | 77.92 | 80.24 |
| | Pillar | 77.87 | 78.880 | 1.263 | 77.23 | 80.62 |

A tolerance of ±0.3 — the figure this layer was originally specified with —
is tighter than the seed-to-seed SD on three of the four conditions for mIoU
and on all four for Pillar. It would fail on correct code, and the predictable
response to a gate that fails spuriously is that somebody widens it.

The val split is measured too, and both splits are in `retrain.csv`:

| condition | metric | seed 37 | mean | SD | min | max |
|---|---|---:|---:|---:|---:|---:|
| `hd/resnet18` | mIoU | 83.72 | 83.674 | 0.334 | 82.88 | 84.01 |
| | Pillar | 80.83 | 81.192 | 0.680 | 80.31 | 82.13 |
| `bl/resnet18` | mIoU | 82.65 | 82.763 | 0.379 | 82.33 | 83.51 |
| | Pillar | 78.28 | 78.024 | 1.299 | 76.48 | 79.96 |
| `hd/resnet18` shuffled | mIoU | 82.95 | 82.937 | 0.189 | 82.51 | 83.23 |
| | Pillar | 79.17 | 78.517 | 0.788 | 76.74 | 79.30 |
| `ef/mit_b0` | mIoU | 82.48 | 82.702 | 0.393 | 81.83 | 83.22 |
| | Pillar | 79.36 | 79.169 | 0.898 | 77.65 | 80.38 |

**The gate.** Each retrained run must land inside the recorded ten-seed
min–max range for its condition, **on test mIoU**. That is the claim worth
making — this code trains models of the same quality as the runs behind the
paper — and both the range and the choice of cell are measured rather than
invented. Pillar and val are compared against the same ranges and every
verdict is written into the artifact; they do not decide the exit code, for
the reason set out two paragraphs below.

The gate is the test split because that is the split the published tables
report. Val is written into the artifact and checked against its own recorded
range, but not gated, and showing only one split would have been selection by
split. Read a val number knowing what it is: the checkpoint was *selected* on
val mIoU over roughly a hundred evaluations, so that metric is a maximum
rather than a draw, which is why its recorded spread is markedly narrower
(SD 0.19–0.39, against 0.30–0.71 on test). The selection is identical on both
sides of the comparison, so it does not bias the comparison — it does make
the val range tighter than the run-to-run noise it would be asked to
accommodate.

**What the gate is not.** The range of ten draws is not a tolerance interval.
For one further exchangeable draw its coverage is 9/11 = 81.8%, so a
perfectly faithful implementation lands outside it roughly one time in five.
An excursion is a reason to *investigate*, not proof of a defect — and the
honest response is to diagnose it, not to widen the range and not to re-run
until it lands. Both of those would make the gate the thing that was tuned.
Reported beside it for every row is the 95% prediction interval for a new
draw (mean ± t(.975, n−1)·SD·√(1+1/n)), which is the interval that does have
a stated coverage.

**And the gate covers only mIoU, which is a correction made on evidence
rather than a concession made on results.** The criterion this layer was
specified with gated both metrics against the recorded range. That is wrong
for Pillar, and the reason is measurable: run the same condition, at the same
seed, more than once. Doing so (the numbers are in "The same-seed band"
below) gives, on the test split:

| condition | runs | mIoU span | recorded mIoU range width | Pillar span | recorded Pillar range width |
|---|---:|---:|---:|---:|---:|
| `bl/resnet18` | 3 | 0.66 | 1.69 | 0.76 | 2.18 |
| `hd/resnet18` | 4 | 1.09 | 2.05 | **5.16** | **2.74** |

On mIoU the spread between runs that differ in nothing an experimenter
controls stays comfortably inside the recorded range, so that range is a
meaningful envelope for one fresh run and a gate on it means something. On
Pillar, `hd/resnet18` spans 5.16 across four identical invocations — nearly
twice the entire width of the recorded Pillar range, and wider than its 95%
prediction interval (5.19) can usefully bound. One of those four legitimate
runs falls outside *both*. **No interval derivable from the recorded ten
seeds can gate a single Pillar draw**, so Pillar is reported against both
intervals, a miss is printed and written to the artifact, and the exit code
does not depend on it.

Why Pillar behaves that way is known, is documented in this project, and is
not a property of this code: Pillar appears in only 14 of the 45 test images
and the top five carry 55% of its ground-truth pixels, so one badly predicted
image moves the aggregate by points. `verification/README.md` records an
archived run sitting at z = −4.45 on Pillar — against the other nine seeds,
leaving itself out — with its mIoU at z = +0.11, traced to a single image. One
of the same-seed repeats measured below reads 78.54 on Pillar with a normal
mIoU, which is that same phenomenon observed in a freshly trained model rather
than an archived one — itself a small piece of evidence that the release
reproduces the recipe's behaviour, tail included.

**Reported, not gated:** the difference against the same seed's recorded run,
*and* the difference against the condition's ten-seed mean. Which reference
is used changes the sign of the answer, so both are in the artifact; see the
results below.

**Three failure signals, declared in advance** so they cannot be
rationalised after the numbers are in:

1. **Any value outside its recorded range.** The gate, on test mIoU; printed
   and recorded, but not gated, for Pillar and for val.
2. **All four conditions deviating in the same direction.** A systematic
   defect can sit inside every individual range while moving everything one
   way. Weak on its own — four signs agreeing happens 12.5% of the time by
   chance — so it is a prompt to look, not a verdict.
3. **A Pillar drop with a normal mIoU.** That is the signature of one class
   failing rather than of a worse model, at z = −4.45 with mIoU at z = +0.11
   in the case this project has already met.

Signals 2 and 3 are computed and printed by `tools/retrain_verify.py`
alongside the gate.

### Procedure

Each combination was trained by the package's own sweep, one at a time on one
GPU, and then compared:

```
chamnet sweep --recipe paper --methods hd --backbones resnet18 \
              --seeds 37 --eval-seed 42 --data <dataset> --out <runs>
python tools/retrain_verify.py --runs <runs> --src <recorded> --commit <sha>
```

Nothing was overridden: the recipe is `paper` — the same one the equivalence
tests compare against the published configs — and the sweep trains, selects
the best-mIoU checkpoint, scores it on test and on val, and records the row.

**`--seeds 37 --eval-seed 42` is not a mixed pair by mistake.** The campaign
trained at each run's seed and then scored the checkpoint in two further
processes it never passed a seed to, so every recorded number was produced at
the base config's fixed `randomness.seed = 42` whatever seed had been
trained. For most arms that makes no difference. For the shuffled control it
decides the result: it permutes the depth channel per sample inside the
dataloader workers, whose numpy streams are seeded from
`(num_workers, rank, worker_id, seed)`, so the seed selects which permutation
realisation the model is scored on. Training at 37 and evaluating at 42
mirrors what the recorded runs did. See `RECORDED_EVAL_SEED` in
`tools/replay.py`.

`chamnet train` is not the right vehicle for this and was not used: it trains
and validates but never scores the test split, and it has no `--eval-seed`.
The sweep is also resumable, which for runs of this length is not a
convenience.

**Environment.** Python 3.12.3, torch 2.9.0a0 (NGC 25.09), mmcv 2.1.0 built
from source, mmengine 0.10.7, **vanilla `mmsegmentation` 1.2.2 installed from
PyPI**, timm 1.0.19, on one NVIDIA RTX PRO 6000 Blackwell. The vanilla
dependency is deliberate: the published runs used a patched fork, every one
of its patches that touches this method was audited, and the single one that
changes a number here is reproduced inside this package
(`chamnet/models/data_preprocessor.py` — vanilla's `bgr_to_rgb` silently
does nothing on a 4-channel input, worth about −9.5 mIoU on every arm that
uses depth).

**Data layout.** The runs read the dataset copy the published evaluations
were computed against, presented under the layout this package documents
(`<split>/images/<name>.jpg`, `<split>/depth/<name>.npy`,
`<split>/masks/<name>.png`) by a symlink tree over that copy — content
byte-identical, names normalised. That copy is only half migrated to the
documented layout: its train split matches, while its val and test depth
files carry a `_depth` infix and its val/test labels live in a differently
named folder with a differently suffixed file name. **Without that step the
package's own config cannot train on it at all** — the three depth-using
combinations die at their first validation on a missing depth file — which
is worth stating plainly rather than absorbing, because it is the same shape
of defect as the one that made this layer necessary: something no
config-level or replay-level check can see, reachable only by running the
thing end to end.

Two further notes on the labels, for exactness. The label folder the symlinks
point at is the one the recorded metrics were computed against. It differs
from the other folder on exactly one of the 45 test images, where the other
copy dropped a real full-height structural post; that folder is correct and
the discrepancy is a stale file in that dataset copy rather than a defect in
this code (`verification/README.md`, "The mask finding"). Using it therefore
scores the retrained models on the same labels the recorded numbers used.

### Results

Nine trainings were run: the four conditions of this layer, plus five repeats
of two of them at the same seed to measure the run-to-run band (below).
`verification/retrain.csv` is the machine-written form of the four, carrying
both reference columns, both intervals and both splits.

**Test split.** `Δ37` is against the same seed's recorded run; `Δmean`
against the condition's ten-seed mean. Both are reported because the choice
of reference changes the sign of the answer.

| condition | metric | retrain | recorded range | z | Δ37 | Δmean | range verdict |
|---|---|---:|---|---:|---:|---:|---|
| `hd/resnet18` | mIoU | 81.65 | [80.15, 82.20] | −0.01 | −0.23 | −0.01 | in (**gated**) |
| | Pillar | 83.70 | [81.54, 84.28] | +0.56 | −0.31 | +0.62 | in |
| `bl/resnet18` | mIoU | 80.33 | [79.82, 81.51] | −0.47 | −0.62 | −0.26 | in (**gated**) |
| | Pillar | 78.89 | [79.03, 81.21] | −1.12 | −1.11 | −0.98 | **outside by 0.14** |
| `hd/resnet18` shuffled | mIoU | 80.82 | [80.13, 80.97] | +0.54 | −0.15 | +0.16 | in (**gated**) |
| | Pillar | 80.23 | [78.60, 80.85] | +0.30 | −0.17 | +0.19 | in |
| `ef/mit_b0` | mIoU | 79.08 | [77.92, 80.24] | −0.51 | −0.23 | −0.36 | in (**gated**) |
| | Pillar | 77.01 | [77.23, 80.62] | −1.48 | −0.86 | −1.87 | **outside by 0.22** |

**Val split** (reported, not gated). `hd/resnet18`'s Pillar is 0.19 under its
range; everything else is inside, including all four mIoU.

| condition | metric | retrain | recorded range | z | Δ37 | Δmean |
|---|---|---:|---|---:|---:|---:|
| `hd/resnet18` | mIoU | 83.71 | [82.88, 84.01] | +0.11 | −0.01 | +0.04 |
| | Pillar | 80.12 | [80.31, 82.13] | −1.58 | −0.71 | −1.07 |
| `bl/resnet18` | mIoU | 82.35 | [82.33, 83.51] | −1.09 | −0.30 | −0.41 |
| | Pillar | 77.40 | [76.48, 79.96] | −0.48 | −0.88 | −0.62 |
| `hd/resnet18` shuffled | mIoU | 83.09 | [82.51, 83.23] | +0.81 | +0.14 | +0.15 |
| | Pillar | 77.93 | [76.74, 79.30] | −0.74 | −1.24 | −0.59 |
| `ef/mit_b0` | mIoU | 82.94 | [81.83, 83.22] | +0.61 | +0.46 | +0.24 |
| | Pillar | 78.37 | [77.65, 80.38] | −0.89 | −0.99 | −0.80 |

### The verdict

**mIoU passes cleanly.** All four gated values are inside their recorded
ranges, and so are all nine when the five repeats are included — 9/9 across
every training run produced for this document. Against the ten-seed mean the
four deviations are −0.26, −0.36, −0.01 and +0.16: mixed in sign and all
inside a third of a seed-to-seed SD of centre.

**Pillar is consistent with the recorded distribution but cannot be gated at
one run per condition.** Two of four land just outside the recorded range, by
0.14 and 0.22; both are inside the 95% prediction interval, and both are a
fraction of the same-seed band measured below (0.76 and, on the other arm,
5.16). Six of the nine runs are inside the range on Pillar.

**All eight test-split deltas against seed 37 are negative, and that is the
declared same-direction signal firing.** It dissolves against the mean,
because seed 37 is an above-average draw for three of the four conditions:
its recorded test mIoU sits +0.23, +0.36 and +0.31 above the condition mean
for `hd`, `bl` and `hd/shuffled` (and −0.13 below it for `ef`), and for
`hd/shuffled` the recorded 80.97 *is* the ten-seed maximum, tied. Comparing
a fresh run against the highest of ten draws and finding it lower is
regression toward the mean, not a defect — which is exactly why both columns
are reported and why the reference matters more than the number.

### The training path is reproduced iteration by iteration

This is the strongest evidence here, and it is not a comparison of final
metrics at all.

Because the training seed, the worker count and every pipeline setting match
the recorded run's, a retrain does not merely start from the same
*distribution* — it starts from the same initial weights and sees the same
batches in the same order with the same augmentation draws. So its loss curve
is directly comparable to the recorded run's, step by step:

| iteration | 0 | 1 | 10 | 100 | 500 | 1000 |
|---|---:|---:|---:|---:|---:|---:|
| `bl/resnet18` recorded | 3.6559 | 3.6566 | 3.5590 | 0.7933 | 0.2040 | 0.1706 |
| retrain | 3.6552 | 3.6584 | 3.5570 | 0.8002 | 0.2027 | 0.1767 |
| `hd/resnet18` recorded | 3.6574 | 3.6664 | 3.5528 | 0.8215 | 0.1848 | 0.1379 |
| retrain | 3.6592 | 3.6624 | 3.5527 | 0.8070 | 0.1817 | 0.1338 |
| `ef/mit_b0` recorded | 2.9572 | 2.9443 | 2.8545 | 0.3544 | 0.1653 | 0.1326 |
| retrain | 2.9521 | 2.9406 | 2.8542 | 0.3242 | 0.1821 | 0.1208 |

The agreement needs a scale to mean anything, so here is one: across the ten
recorded seeds — runs that differ in both initialisation and batch order —
the iteration-0 loss has SD 0.018 (`bl`), 0.018 (`hd`) and 0.045 (`ef`). The
retrain-versus-recorded differences at that iteration are 0.0007, 0.0018 and
0.0050, i.e. **0.04, 0.10 and 0.11 of one seed-to-seed SD**. A different
initialisation or a different first batch would show up an order of magnitude
larger than that.

So initialisation, data order, the augmentation stream, the loss, the
optimiser, the schedule and the backward pass are all reproduced — every
training-side mechanism the earlier two layers are blind to. The
final-metric differences are what non-determinism grows into over two
thousand steps, and by the end the two runs are no longer neighbours, which
is why a retrain behaves like a fresh draw from its condition rather than
like a copy of seed 37.

### The same-seed band

Two conditions were re-run at the same seed, with the identical command, in
separate processes. `bl/resnet18` because it is fast, carries one of the two
Pillar misses, and runs entirely on vanilla mmsegmentation plus this
package's config; `hd/resnet18` because it is the headline arm.

| run | best iter | test mIoU | test Pillar | val mIoU | val Pillar |
|---|---:|---:|---:|---:|---:|
| `bl` — the run in `retrain.csv` | 1260 | 80.33 | **78.89** | 82.35 | 77.40 |
| `bl` repeat | 1300 | 80.99 | 79.65 | 82.69 | 77.41 |
| `bl` repeat | 1480 | 80.75 | 79.63 | 82.78 | 77.56 |
| *span* | | *0.66* | *0.76* | *0.43* | *0.16* |
| `hd` — the run in `retrain.csv` | 1980 | 81.65 | 83.70 | 83.71 | 80.12 |
| `hd` repeat | 860 | 80.56 | **78.54** | 82.85 | 80.12 |
| `hd` repeat | 860 | 81.01 | 83.51 | 83.20 | 80.75 |
| `hd` repeat | 1660 | 81.59 | 81.74 | 83.41 | 80.14 |
| *span* | | *1.09* | *5.16* | *0.86* | *0.63* |

Four things follow.

**The band is metric-dependent, and that is what retired the Pillar gate.**
On mIoU the same-seed span is 0.66 and 1.09, against recorded seed-to-seed
SDs of 0.55 and 0.60 and range widths of 1.69 and 2.05 — comparable, so the
range is a usable envelope. On Pillar the `hd` span is 5.16 against a range
width of 2.74. A criterion cannot discriminate at a resolution finer than a
repeat of the same run.

**The two misses are low draws, not levels.** `bl`'s further runs land at
79.65 and 79.63, inside the range; had either been the run in the artifact
nothing would have fired. The first run stays in the artifact regardless.
Repeats were run to measure the band, not to choose a number, and choosing
among them would be the same error as re-running until a rounding boundary
falls the right way.

**`hd`'s 78.54 is the single-image phenomenon, caught live.** That run's
mIoU is 80.56 — inside its range — while its Pillar is 4.2 SD below the
recorded mean. Normal mIoU with collapsed Pillar is the pattern this project
already documented at z = −4.45 in an archived run, traced to one test image;
seeing it arise in a fresh training run is the mechanism reproducing, not a
new defect. It is also the observation that makes the wide Pillar band
expected rather than alarming.

**One pair bounds nothing, and the spans above are floors.** Three and four
runs give a minimum for each band, not the band itself; the true spread is at
least this wide and may be wider. In particular the 4.97 between `hd`'s two
860-iteration repeats is a single pair, and establishes only that a gap that
large occurs — not how often.

### What was checked and eliminated along the way

The four conditions initially looked like they carried a systematic signal:
mean recall ran +1.05 SD above the recorded mean (7 of 8 measurements) and
mean precision −0.59 below it, while mIoU (−0.00), mDice (−0.02) and aAcc
(−0.01) sat exactly on their means. That is a coherent recall-leaning shift,
and coherent with Pillar — small, thin, most exposed to false positives —
being the class that showed it. It was worth chasing, and four causes were
eliminated by measurement rather than by argument:

* **The environment.** The recorded runs' own logs report the same Python
  3.12.3, the same torch 2.9.0a0, the same torchvision, CUDA, NVCC, mmengine
  and the same GPU. The recorded campaign ran on this machine and this image.
* **The evaluation code.** The per-class metric this package swaps in
  delegates every aggregate to vanilla `IoUMetric` before adding its own
  keys, so the aggregates are the vanilla ones. Independently: recall and
  precision are fixed by the intersection and union counts together with the
  ground-truth area, so replay rows matching on IoU to two decimals pin them
  too.
* **Checkpoint selection.** The retrains' selected iterations sit inside the
  recorded per-condition spreads (recorded `best_iter` ranges 780–3360; means
  1500–1832, SD 297–796).
* **Initialisation, data order and the optimiser** — by the loss curves
  above.

Then the repeats settled it: recall and precision land on the *opposite* side
of the recorded mean from the same seed and the same command (`bl` recall
z = +1.06, −0.73, −0.48; precision z = −0.90, +1.03, +0.53), averaging to
z = −0.05 and +0.22 over three runs. The pattern was a fluctuation of a
statistic whose same-seed span (0.80 recall, 1.25 precision on `bl`; 1.40
recall on `hd`) exceeds its recorded seed-to-seed SD (0.45, 0.65, 0.16). On
`hd` that mismatch is stark: one repeat reads z = −7.12 on mean recall, which
says far more about how tight the recorded spread of that statistic is than
about the run.

Two further checks were made because they were declared in advance:

**Per-class profile.** Pillar has the most negative mean z of the eight
classes across the four runs and both splits (−0.68), but it is not
isolated: `path` swings from −3.21 to +3.15 over the same runs, Pillar's
worst is −1.58, and the most negative of eight class means is expected near
−0.5 by chance. The declared signature — one class collapsing while the rest
hold — is not present in the four artifact runs. It *is* present in the `hd`
repeat discussed above, where it is explained.

**What actually drives the variance.** In the recorded data, `best_iter`
spans 780–3360 at fixed recipe and correlates with test Pillar at r = +0.78
(`hd`), +0.82 (`bl`), +0.59 (`hd/shuffled`) and +0.47 (`ef`). Early stopping
with patience of 20 evaluations and `min_delta` 0.01 on a noisy val mIoU
makes the training length effectively random over a threefold range, and
Pillar follows it. That is a property of the published recipe, reproduced
here rather than introduced: every retrain's `best_iter` sits inside its
condition's recorded spread.

### What this layer does not cover

* **Four combinations, not thirty-six.** They were chosen to cover distinct
  things training can get wrong — the headline arm, the same backbone with no
  depth at all, a control whose training input is drawn per sample, and a
  second method on a second backbone family whose entire difference from the
  baseline is one widened stem convolution. The remaining 32 are covered by
  layers 1 and 2 and by a smoke test that builds each one, runs a batch
  through the real pipeline and backpropagates; none of that is a training
  run.
* **SegNeXt-T is deliberately absent.** Its decode head draws random numbers
  on every forward pass, which gives its *evaluation* a small reproducible
  offset between one way of driving a test loop and another. The offset is
  understood and documented, but it is the size of the thing this layer
  measures, and including it would make a training-fidelity result
  unreadable.
* **One seed per condition.** This is not a re-derivation of the published
  ten-seed means, and it is not evidence about the statistical reliability of
  any published comparison. It asks whether one run from this code lands
  where runs of that condition land.
* **A single run's Pillar is not gateable here, and no amount of care fixes
  that.** The same-seed band measured above (5.16 on `hd/resnet18`) exceeds
  every interval the recorded ten seeds can supply. Pillar agreement in this
  layer is therefore evidence only in aggregate and only weakly; the
  published Pillar claims rest on ten-seed means, which is the level at which
  they are stable, and on the checkpoint replay, which reproduces the
  recorded Pillar for fixed weights to two decimals.
* **The repeats vary numerics, not seeds.** They were all run at seed 37, so
  they measure how far identical invocations diverge, not how far the
  condition varies across seeds. Where a metric is dominated by the seed
  rather than by numerics the repeats cluster and say little: `hd/resnet18`'s
  four val Pillar values are 80.12, 80.12, 80.75 and 80.14, three of them
  0.17–0.19 below the recorded ten-seed minimum. Whether that small offset is
  a property of the seed, of this code, or of a val split whose Pillar is
  concentrated in a few images is **not settled here**; four same-seed runs
  cannot separate those, and it is recorded as open rather than explained.
* **Nothing here is bit-exact, and cannot be.** Forcing full cuDNN
  determinism and disabling TF32 does not remove the ~0.01 run-to-run
  flutter this setup shows even at fixed weights, so equality was never an
  available criterion at any layer.
* **Nobody outside the training server can reproduce layers 2 and 3.** The
  dataset and the checkpoints are not distributed. The artifacts in
  `verification/` and the tools that produced them are what is offered
  instead: each gates its own comparison and fails its exit code rather than
  printing a summary, and each writes a provenance line naming the commit,
  the date, the seeds and the library versions. The replay's exit code is
  currently non-zero on purpose, for reasons its own README sets out.
* **Layer 3 says nothing about the data itself.** One stale label file in the
  dataset copy described above is a data-quality item on that copy, tracked
  in `verification/README.md`; it is not resolved by anything here.
