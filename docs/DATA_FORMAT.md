# Data format

This package accepts **one** dataset layout. There is no option to point it at
a different one, and it does not probe for alternatives. Read this before
pointing `--data` at anything, because a directory tree that is nearly right
fails at the moment the pipeline first touches the part that is wrong, which
can be a long way into a run.

## The layout

```
<data_root>/
  train/
    images/<name>.jpg      RGB, 8-bit, 3 channels
    depth/<name>.npy       depth, float32, (H, W)
    masks/<name>.png       label indices, 8-bit, 1 channel, values 0-7
  valid/
    images/  depth/  masks/
  test/
    images/  depth/  masks/
```

The three split directories are named `train`, `valid` and `test` — `valid`,
not `val`. They come from `data.splits` in `chamnet/recipes/paper.yaml`, which
is the only place they are written down.

**The three files of one sample share a stem.** `images/0526_rfv7_069s.jpg`,
`depth/0526_rfv7_069s.npy` and `masks/0526_rfv7_069s.png` are one sample. The
image list is built by scanning `images/` for `*.jpg`; the other two paths are
then *derived* from each image path — `LoadDepthAsChannel` swaps `images` for
`depth` in the directory part and `.jpg` for `.npy` in the file part, and the
dataset swaps in `masks` and `.png`. Nothing is matched by fuzzy name, so a
suffix or an infix anywhere in the stem is a different sample that does not
exist.

| | `images/` | `depth/` | `masks/` |
|---|---|---|---|
| extension | `.jpg` | `.npy` | `.png` |
| read by | `LoadImageFromFile` | `LoadDepthAsChannel` | `LoadAnnotations` |
| dtype on disk | uint8 | float32 (any numpy dtype `np.load` returns, cast to float32) | uint8 |
| shape | `(H, W, 3)` | `(H, W)` | `(H, W)` |

Depth and mask do not have to match the image's resolution: depth is resized
to the image with bilinear interpolation when it differs, and everything is
resized to 512×512 (`keep_ratio: false`) by the pipeline before it reaches the
model. Matching resolutions is still the sane choice — a resized label map is
not what a resized image is.

## What the values have to be

**Masks are class indices, not colours.** A PNG whose pixel values are `0`-`7`
in a single channel. A palette (`P` mode) PNG counts — the index plane is what
gets read, so the colours a viewer shows are irrelevant. A genuine
three-channel `RGB` PNG of the same picture does not count, and the failure is
worth recognising because it arrives in two steps: packing warns

```
UserWarning: Please pay attention your ground truth segmentation map, usually
the segmentation map is 2D, but got (512, 512, 3)
```

and then the first iteration stops in the loss with

```
RuntimeError: only batches of spatial targets supported (3D tensors) but got
targets of size: : [2, 512, 512, 3]
```

Nothing silently trains on colours, but the warning is the last chance to
notice before the traceback.

| index | class |
|---:|---|
| 0 | background |
| 1 | chamoe (Korean melon) |
| 2 | heatpipe |
| 3 | path |
| 4 | **pillar** |
| 5 | topdownfarm (vertical growing rack) |
| 6 | ceiling |
| 7 | duct |

`255` is the ignore index and is what padding is filled with, so do not use it
as a class. Pillar is the class the published comparison is about.

**Depth is a metric map, and the standardisation constants assume the
greenhouse's unit.** `LoadDepthAsChannel` casts to float32, rejects a file
containing any non-finite value, and appends it as a fourth channel with no
rescaling — the value in the `.npy` file is the value the model sees, before
standardisation. The preprocessor then standardises that channel with a fixed
mean 2.2638 and standard deviation 2.5189, computed once from the greenhouse
training split, alongside the ImageNet constants used for RGB. Those two
numbers live in `chamnet/config/backbones.py` (`depth_mean_std`).

If your depth is in different units, or is disparity, or is normalised to
`[0, 1]`, or is a 16-bit millimetre PNG converted without dividing, those two
constants are wrong for it and the depth channel arrives at the model badly
scaled. Nothing will complain. Recompute them from your own training split.

Measured or predicted depth both work; the package cannot tell. The loader's
own name for it is pseudo-depth because the published runs feed a monocular
depth estimator's approximately-metric output rather than a range sensor's.
All that is required here is one finite float32 map per image.

## A different layout crashes, and here is what that looks like

This is not hypothetical. The dataset copy the published evaluations were
computed against does **not** satisfy the layout above, and running this
package's own config against it unmodified fails. It is half migrated: its
train split matches, while its `valid` and `test` depth files carry a `_depth`
infix before the extension and their labels live in a differently named folder
under a differently suffixed file name. Getting the retraining check in
`docs/VERIFICATION.md` to run at all needed a symlink tree that presented that
copy under the layout above.

So: **a copy whose train split is right and whose val split is not will train
normally and then die at its first validation**, roughly twenty iterations in
with this recipe. The two failures look like this.

Depth file, from a stem that carries an extra infix:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '<data_root>/valid/depth/<name>.npy'
```

Label file, when the labels are under a different folder or suffix:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '<data_root>/valid/masks/<name>.png'
```

Both name the path that was *derived* and not found, which is the useful half:
compare it against what is actually on disk and the difference is the answer.
Note what the second one is not — the dataset object **builds** successfully
and reports the right length, because the length comes from scanning
`images/`. The label paths are never checked until a sample is fetched. A
green startup means nothing about whether `masks/` is right.

There is no flag for this, deliberately. One layout, documented here, is
easier to defend than a set of heuristics that quietly accept several; and the
fix — symlink or copy your files into the shape above — is a few lines of shell
that leave the original untouched:

```bash
# example: valid/ depth named <name>_depth.npy, labels under masks_gray/<name>_mask_gray.png
mkdir -p out/valid/images out/valid/depth out/valid/masks
ln -sr src/valid/images/*.jpg out/valid/images/
for f in src/valid/depth/*_depth.npy; do
    ln -sr "$f" "out/valid/depth/$(basename "$f" _depth.npy).npy"
done
for f in src/valid/masks_gray/*_mask_gray.png; do
    ln -sr "$f" "out/valid/masks/$(basename "$f" _mask_gray.png).png"
done
```

## Checking a layout without training

`chamnet smoke` builds a synthetic dataset in exactly this layout, runs it
through the real pipeline and backpropagates, so it tells you the code works
but nothing about your files. To check *your* files, build the dataset and
fetch one sample — the cheapest thing that exercises every derived path:

```python
import chamnet
from chamnet.config.builder import build_config

chamnet.register_all()
cfg = build_config(method='hd', backbone='resnet18', data_root='<data_root>')
with chamnet.scoped(cfg):
    from mmseg.registry import DATASETS
    for split in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataset = DATASETS.build(cfg[split]['dataset'])
        sample = dataset[0]
        print(split, len(dataset), sample['inputs'].shape, sample['inputs'].dtype)
```

`method='hd'` because it is a 4-channel method, so the depth path is
exercised; `bl` would not touch `depth/` at all. On a correct layout each of
the three lines prints `torch.Size([4, 512, 512]) torch.float32` (`bl` and the
`hd`/`rgb` control print `[3, 512, 512] torch.uint8` — the depth channel is
what upcasts the tensor, so a 4-channel arm printing `uint8` would mean the
depth loader silently did nothing; the suite asserts both). If a split's files are wrong, this
is where it says so, in a second rather than after twenty minutes of training.

## Using your own data

Point `--data` (or `build_config(data_root=...)`) at a root laid out as above.
Everything else — image size, batch size, schedule — comes from the recipe.

**A different number of classes needs three edits, and they must agree.**
Nothing derives one from another, and a mismatch surfaces as a shape error in
the loss or, worse, as a silently wrong `class_weight`:

1. `chamnet/datasets/dataset.py` — `ChamNet.METAINFO`: the `classes` tuple and
   a `palette` entry per class. The palette is used for visualisation only.
2. `chamnet/config/builder.py` — `CLASSES`, which sets `num_classes` on both
   the decode head and the auxiliary head.
3. `chamnet/recipes/paper.yaml` — `loss.cross_entropy.class_weight`, which is
   one weight per class, in index order. The published weights are
   `[0.5, 3.0, 3.0, 1.0, 3.0, 0.5, 1.0, 1.0]`: three times the weight on
   chamoe, heatpipe and pillar, half on background and topdownfarm. They are
   specific to this dataset's class imbalance and are not a default worth
   keeping for another one.

Two things to know before comparing your numbers to anything published. The
recipe stops early — patience 20 evaluations on validation mIoU with
`min_delta` 0.01 — which on this dataset makes the effective training length
vary over roughly a threefold range from run to run and is the dominant source
of run-to-run variance in per-class IoU (`docs/VERIFICATION.md`). And the
published figures are means over ten training seeds, not single runs. A single
run of your own is one draw.

The class weights and the depth constants are the two places this package
carries greenhouse-specific numbers. Everything else is a method.
