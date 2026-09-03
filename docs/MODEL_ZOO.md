# Model zoo

Two checkpoints are attached to the [`v1.0.0`
release](https://github.com/elasmobranches/PDSG/releases/tag/v1.0.0). Verify the
SHA-256 below against what you download — the checksums are of the bytes as they
sit on the machine that trained them, and a download was checked against them.

Two checkpoints, both ResNet-18 at seed 37: the RGB baseline and the
heavy-depth model, the pair the published comparison turns on.

They are the files the recorded metrics were computed from — trained by the
research code, not by this package. That is what makes them useful here rather
than decorative: loading them into models this package builds and getting the
recorded numbers back is the checkpoint-replay layer of
`docs/VERIFICATION.md`, and both of these rows land exactly. Scored through
`chamnet test` on the split they were originally evaluated on, `hd` returns all
eight per-class IoUs and all six aggregates identical to the values recorded
when it was trained. (That needs the dataset, so it is not something a reader
can repeat; the equality of the two headline metrics is in
`verification/replay.csv`, and the architecture check below needs no data.)

## Loading a checkpoint runs pickled code

Reading an mmengine checkpoint needs a wider unpickler than torch enables by
default — the file carries pickled logging state (`HistoryBuffer` objects over
numpy arrays) beside its tensors — and this package turns that default off for
the load (`chamnet/checkpoint.py`, used by `chamnet test` and by the sweep).
So **loading a checkpoint through this package executes code from the file**.
Load only checkpoints you trust, and verify the SHA-256 below before you load
one you downloaded.

## The files

Loading a checkpoint executes pickled code — load only ones you trust, and
check the hash of anything you downloaded before you load it.

| file | method | backbone | seed | selected iter | bytes | SHA-256 |
|---|---|---|---:|---:|---:|---|
| [`chamnet_bl_resnet18_seed37.pth`](https://github.com/elasmobranches/PDSG/releases/download/v1.0.0/chamnet_bl_resnet18_seed37.pth) | `bl` (RGB only) | resnet18 | 37 | 1300 | 56,144,019 | `7557577f62e142565054be7ec78c9f179ea66212bec71887a41e5c3e233d817f` |
| [`chamnet_hd_resnet18_seed37.pth`](https://github.com/elasmobranches/PDSG/releases/download/v1.0.0/chamnet_hd_resnet18_seed37.pth) | `hd` (RGB + depth, heavy depth encoder) | resnet18 | 37 | 1720 | 103,633,995 | `127648c8702123360226219f7012a51f9229e5416f3f4d314a12096e501b3fda` |

Each is the campaign's own `best_mIoU_iter_<n>.pth` under a name that says what
it is; the checksums are of those bytes, unmodified, so they verify against
either name. "Selected iter" is where the run's best validation mIoU landed —
validation ran every 20 iterations and training stopped after 20 evaluations
without a 0.01 improvement, so the number varies widely between runs of the
same condition and is not a property of the method.

The config each was trained under is not shipped as a file, because this
package emits it. These two commands write it, and the emitted config is
compared key by key against the merged config the run actually used
(`tests/test_matches_paper.py`):

```bash
chamnet export-config --method bl --backbone resnet18 --seed 37 -o bl_resnet18.py
chamnet export-config --method hd --backbone resnet18 --seed 37 -o hd_resnet18.py
```

## What they scored

Recorded when they were trained, and reproduced by `tools/replay.py` through
this package's code. Both rows are exact on both metrics, to the two decimals
the recorded file stores — the full 36-row table is `verification/replay.csv`.

| | test mIoU | test Pillar | val mIoU | val Pillar |
|---|---:|---:|---:|---:|
| `bl` | 80.95 | 80.00 | 82.65 | 78.28 |
| `hd` | 81.88 | 84.01 | 83.72 | 80.83 |

**These are single runs and the published tables are not.** Every published
figure is a mean over ten training seeds, and the run-to-run spread on this
dataset is not small: 0.55-0.60 SD on test mIoU for these two conditions and
0.87-1.09 on test Pillar, with a same-seed span — identical command, separate
processes — measured at 5.16 points of Pillar on `hd/resnet18`
(`docs/VERIFICATION.md`). Pillar appears in only 14 of the 45 test images and
the top five of those carry 55% of its ground-truth pixels, so one image
predicted badly moves the aggregate by points (`verification/README.md`). The
`hd` minus `bl` difference in the table above is one draw of a difference the
paper reports as a ten-seed mean. Do not read it as the effect size.

## Why seed 37

Seed 37 is the representative seed, chosen by a rule that reads validation
metrics only and never test ones: `tools/select_seed.py` implements it, states
it in full before touching any data, and selects 37 on the recorded campaign
with a worst rank of 2 across its four criteria where no other seed does
better than 5. The same rule reading test columns selects a different seed,
which is why it is not allowed to.

```bash
python tools/select_seed.py --results <recorded results CSV> --expect 37
```

The recorded per-run CSV is not distributed, so that command needs the private
data. The rule is in the file and is checkable without it.

## Why only these two

Nine arms × four backbones at ten seeds each is 360 selected checkpoints and
16.8 GB, nearly all of which nobody would open. These two are the pair the
argument is about, on the backbone the headline numbers quote, at the
representative seed. What the other 358 would show is in
`verification/replay.csv` — what all 36 combinations scored when replayed
through this code — and in the config-equivalence tests, which need no
checkpoints at all.

## What you can do with them without the dataset

The greenhouse dataset is not distributed, so the table above cannot be
regenerated outside the training server. What a download does support:

* **Check that the architecture matches, without any data at all.** They are
  ordinary mmengine checkpoints, and loading one into the model this package
  builds reports **no missing and no unexpected keys** — 378 tensors for `hd`.
  That is a real check on the port and it needs nothing but the file:

  ```python
  import torch, chamnet
  from chamnet.checkpoint import mmengine_checkpoint_loading
  from chamnet.config.builder import build_config
  from mmseg.registry import MODELS

  chamnet.register_all()
  cfg = build_config(method='hd', backbone='resnet18')
  with chamnet.scoped(cfg):
      model = MODELS.build(cfg.model)
  with mmengine_checkpoint_loading():                     # executes pickled code
      state = torch.load('chamnet_hd_resnet18_seed37.pth', map_location='cpu')
  print(model.load_state_dict(state['state_dict'], strict=False))
  ```

  Both lists empty means every parameter the checkpoint carries has a slot in
  the model this code builds, and the model has no slot the checkpoint does
  not fill. A subtly different fusion module, stem width or gate would show up
  here.
* **Score them on your own data**, laid out as `docs/DATA_FORMAT.md` requires,
  with the eight greenhouse classes:

  ```bash
  chamnet test --method hd --backbone resnet18 --data <your data root> \
               --checkpoint chamnet_hd_resnet18_seed37.pth
  ```

  Expect the numbers to be poor unless your scene resembles a Korean melon
  greenhouse; the class list, the loss weights and the depth standardisation
  constants are all specific to this dataset (`docs/DATA_FORMAT.md`).
* **Fine-tune from them**, by passing the file as `load_from` on an exported
  config.

## Where they came from

Each is the campaign's own `best_mIoU_iter_<n>.pth`, copied under a name that
says which arm it is:

```
<run dir>/chamnet_baseline_resnet18/best_mIoU_iter_1300.pth  ->  chamnet_bl_resnet18_seed37.pth
<run dir>/chamnet_dual_plus_resnet18/best_mIoU_iter_1720.pth ->  chamnet_hd_resnet18_seed37.pth
```

The SHA-256 values in the table were computed on that machine, before the
upload, and re-checked against an anonymous download of the release afterwards.
A mismatch on either side would mean the wrong run directory or a copy that did
not complete.
