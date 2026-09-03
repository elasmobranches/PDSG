# ChamNet — RGB-D fusion for greenhouse semantic segmentation

Eight-class semantic segmentation of a Korean melon (chamoe) greenhouse.
The input is RGB plus a monocular depth estimate.

There are four ways of using the depth channel and five control arms that test
what the depth actually contributes. Each runs on four backbone families. It is
built on `mmsegmentation` 1.2.2, unmodified.

## Dataset availability

The greenhouse dataset is not published. Access is possible for research use —
please contact shinds@sju.ac.kr with a short note on the intended use.

Two trained checkpoints *are* published, in the
[`v1.0.0` release](https://github.com/elasmobranches/PDSG/releases/tag/v1.0.0):
the ResNet-18 baseline and its RGB-D counterpart. `docs/MODEL_ZOO.md` has their
checksums and what they score.

Two things work without the data. `chamnet smoke` builds a synthetic dataset in
the documented layout and runs it through the real pipeline. `pytest
tests/test_matches_paper.py` compares the config this package emits against the
merged config each of the 36 published runs was trained from.

The checks that needed the data were run by the authors. Their output is in
`verification/`, and `docs/VERIFICATION.md` explains it.

## Environment

`mmcv` 2.1.0 is the awkward dependency. No prebuilt wheel exists for the torch
build these results came from, so it has to be compiled. That is why a
`Dockerfile` ships:

```bash
docker build -t chamnet .
# narrow the CUDA build to your own card to cut the compile time proportionally
docker build -t chamnet --build-arg TORCH_CUDA_ARCH_LIST=8.6 --build-arg MAX_JOBS=16 .
docker run --gpus all -it -v "$PWD:/workspace" chamnet
```

It starts from the public `nvcr.io/nvidia/pytorch:25.09-py3` and installs
`mmsegmentation` as an ordinary package. `mmcv` is compiled in a layer *before*
the source is copied in. Editing the package therefore rebuilds in about a
second.

A full build takes roughly twelve minutes at `MAX_JOBS=16` on a 32-core host and
comes to 29.9 GB. The suite passes inside it. Check that rather than believe it:

```bash
docker run --rm -w /opt/chamnet chamnet python -m pytest tests -q
```

Pinned versions, which are the ones the reported numbers came from:

| | |
|---|---|
| Python | 3.12.3 |
| torch | 2.9.0a0+nv25.09 (NGC 25.09) |
| mmcv | 2.1.0, from source with `MMCV_WITH_OPS=1 FORCE_CUDA=1` |
| mmengine | 0.10.7 |
| mmsegmentation | 1.2.2, from PyPI, unmodified |
| timm | 1.0.19 |
| albumentations | 2.0.8 |

Without Docker: build `mmcv` from source, then `pip install -e '.[test]'`. The
test extra is not optional if you want `chamnet smoke`, which shells out to
`pytest`.

## Quick start

```bash
chamnet smoke                                    # synthetic data through the real pipeline
chamnet list                                     # every valid method × ablation combination
chamnet export-config --method hd --backbone resnet18 -o hd.py
chamnet train  --method hd --backbone resnet18 --data <root> --out runs/hd
chamnet test   --method hd --backbone resnet18 --data <root> --checkpoint <file>
chamnet sweep  --methods bl,hd --backbones resnet18 --seeds 31-40 \
               --data <root> --out runs/sweep
```

Use `chamnet sweep` for anything long. It trains, picks the best checkpoint and
scores both splits. Each run's metrics are written before the run is marked
done, and the results CSV is rebuilt atomically. A crash five runs in loses no
finished row, and a rerun resumes instead of restarting.

Read `docs/DATA_FORMAT.md` before pointing `--data` anywhere. The package accepts
exactly one layout. A nearly-right tree fails partway into a run rather than at
startup.

`chamnet test` loads a checkpoint. Loading an mmengine checkpoint executes
pickled code, so load only ones you trust.

## The methods

Every method shares one recipe: 512×512, batch 16, AdamW, up to 3760 iterations
with early stopping on validation mIoU, and per-class-weighted cross-entropy plus
Dice. They differ only in how the depth channel reaches the network. The recipe
is `chamnet/recipes/paper.yaml`.

| id | input | how depth is used |
|---|---|---|
| **BL** | RGB | Not at all. The baseline everything is measured against. |
| **EF** | RGB + D | Early fusion. One encoder, with its stem convolution widened to four input channels. The fourth is initialised from the mean of the pretrained RGB filters. No extra module — 288 to 1568 extra parameters, depending on the backbone. |
| **SD** | RGB + D | A shallow second encoder of four depthwise-separable stages. It feeds a gate at each stage of the RGB backbone. |
| **HD** | RGB + D | A *full* second backbone on the single-channel depth, feeding the same gate. It starts from the same pretrained RGB checkpoint, with the first convolution averaged across its input channels. |

On the command line those four are lowercase — `--method hd`.

**SD** and **HD** fuse through `CrossModalGating`. The depth feature is projected
1×1 and scaled by a per-channel sigmoid gate computed from its pooled context.
The result is *added* to the RGB feature. Nothing flows the other way.

The campaign called these arms `baseline`, `proposed`, `dual` and `dual_plus`.
Those are its directory names and the `flow` column of its metrics files.
`chamnet/config/combos.py` maps one naming to the other.

### The control arms

Each removes one thing and leaves the rest, so a difference can be attributed.

| arm | what it removes | what a difference tells you |
|---|---|---|
| **HD** + `nogate` | The gate. The depth projection is added at full strength. | Whether the *gating* does anything, as opposed to the extra depth features. |
| **HD** + `bigate` | `CrossModalGating`, replaced by a bidirectional multiplicative gate in the same skeleton. | Whether this gate design matters, or gating in general. |
| **HD** + `rgb` | The depth. The depth-slot encoder is rebuilt for three channels and fed the RGB image. | How much of **HD**'s gain is depth, and how much is a second encoder's capacity. |
| **HD** + `shuffled` | Depth's spatial arrangement. Pixels are permuted per sample, in train, validation and test alike. | Whether depth contributes geometry, or only its per-image statistics. |
| **EF** + `shuffled` | The same, on early fusion. | The same question, on the arm whose only depth machinery is one wider convolution. |

`shuffled` applies to all three splits deliberately. The arm claims that depth's
value is its arrangement. Testing that claim means selecting and scoring a model
that never sees arranged depth.

## Recorded results

Means and standard deviations over ten training seeds (31-40) on the test split,
read from the recorded per-run metrics. These are not reproducible from this
repository. The paper is the reference for the published figures and for any
claim about them.

**Test mIoU**

| | resnet18 | mit_b0 | segnext_t | convnext_atto |
|---|---|---|---|---|
| **BL** | 80.59 ± 0.55 | 79.05 ± 0.42 | 80.18 ± 0.52 | 79.88 ± 0.51 |
| **EF** | 80.85 ± 0.80 | 79.44 ± 0.71 | 81.03 ± 0.25 | 81.01 ± 0.40 |
| **SD** | 81.19 ± 0.54 | 79.80 ± 0.67 | 80.27 ± 0.48 | 80.69 ± 0.54 |
| **HD** | 81.66 ± 0.60 | 80.12 ± 0.29 | 80.56 ± 0.56 | 81.13 ± 0.43 |

**Test Pillar IoU.** Pillar is the class the comparison is about. It is thin,
vertical and small in area, and it is where depth should help most.

| | resnet18 | mit_b0 | segnext_t | convnext_atto |
|---|---|---|---|---|
| **BL** | 79.87 ± 0.87 | 75.08 ± 1.54 | 77.30 ± 1.14 | 78.01 ± 1.27 |
| **EF** | 82.39 ± 0.72 | 78.88 ± 1.26 | 81.09 ± 0.86 | 82.77 ± 0.54 |
| **SD** | 83.51 ± 0.69 | 79.03 ± 1.93 | 80.71 ± 0.81 | 82.62 ± 1.37 |
| **HD** | 83.09 ± 1.09 | 78.44 ± 1.03 | 80.55 ± 1.50 | 82.48 ± 1.39 |

## Repository map

```
chamnet/
  cli.py                     chamnet train|test|sweep|export-config|smoke|list
  config/                    builder, per-backbone facts, the combination table, recipe schema
  recipes/paper.yaml         the published recipe. quick.yaml is a fast stand-in.
  models/
    fusion.py                CrossModalGating, BiGateGating, the depth branch
    backbones/               sd, hd and their controls, per backbone family; early_fusion.py
    depth_pretrain.py        loading an RGB checkpoint into a depth encoder
    data_preprocessor.py     a corrected copy of upstream's; see its docstring
  datasets/                  the 8-class dataset and the depth/augmentation transforms
  sweep.py                   resumable multi-run training
  checkpoint.py              the narrow unpickler mmengine checkpoints need
docs/                        DATA_FORMAT, MODEL_ZOO, VERIFICATION
tools/                       replay.py, retrain_verify.py, select_seed.py
verification/                the artifacts, and a README explaining every unmatched row
tests/fixtures/paper/        the 36 merged configs the published runs used
```

`pytest` runs the whole suite. `pytest -m "not network"` leaves out the tests
that download pretrained checkpoints. Those stay in the default run on purpose.
Observing that pretrained weights actually landed in a depth encoder requires
having the weights.

This release's own prose is English. Korean remains in nine modules whose class
bodies were ported byte-for-byte from the research repository. Editing it would
give up the property those bodies exist to have.
`tests/test_language_policy.py` enforces both halves.

## Licence and citation

Apache License 2.0. The full text is in `LICENSE`, and `NOTICE` records what this
package derives from.

Parts come from OpenMMLab code, which is also Apache-2.0.
`chamnet/models/data_preprocessor.py` is a modified copy of MMSegmentation's own
and keeps its copyright notice. So does `chamnet/datasets/dataset.py`. The
backbones, the metric and the transforms subclass MMSegmentation's and MMCV's.

The pretrained ImageNet weights these runs start from come from OpenMMLab and
`timm` under their own terms. They are not redistributed here.

The paper these numbers belong to is in preparation; the citation goes here when
it exists.
