import torch
import chamnet
from chamnet.config.builder import build_config
from chamnet.config.schema import load_recipe
from mmengine.registry import DefaultScope
from mmseg.registry import DATASETS, MODELS

chamnet.register_all()


def test_bl_builds_and_backprops(synthetic_data):
    cfg = build_config(method='bl', backbone='resnet18',
                       recipe='quick', data_root=str(synthetic_data), seed=31)
    # cfg.default_scope='mmseg' (set by build_config) is what `Runner.from_cfg`
    # reads to establish this scope for `chamnet train`/`chamnet test`. This
    # test builds the model directly, bypassing Runner, so nothing reads that
    # key on its own — without an active scope, mmengine's BaseModel.__init__
    # builds the nested `data_preprocessor` via mmengine's *root* MODELS
    # registry, which can't resolve mmseg-only types like SegDataPreProcessor.
    # Scope the call explicitly instead of mutating global state at import
    # time (chamnet.register_all() must stay side-effect-free for callers).
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        model = MODELS.build(cfg.model)
    x = torch.randn(2, 3, 64, 128)
    feats = model.backbone(x)
    logits = model.decode_head(feats)
    assert logits.shape[1] == 8
    logits.sum().backward()
    assert any(p.grad is not None for p in model.backbone.parameters())


def test_dataset_loads_synthetic_image_and_mask(synthetic_data):
    """Exercise the real data path — LoadImageFromFile -> LoadAnnotations ->
    Resize -> PackSegInputs -> ChamNet.__getitem__ — against the synthetic
    fixture, not a hand-made tensor. Since the greenhouse dataset is not
    public, this is the only check anyone who clones the repo has that the
    .jpg decodes, the mask .png round-trips as integer class indices 0-7, and
    the shapes/dtypes PackSegInputs hands to the model preprocessor are what
    the model expects.

    Scope note: 'bl' is RGB-only (data_channels=3 in build_config), so its
    pipeline has no LoadDepthAsChannel step — this test covers the image and
    mask paths only. Depth (.npy) round-trip coverage needs a 4-channel
    method (sd/hd/ef) and lands with Task 8, which is when those builders
    exist; not overlooked, just not in scope for the BL-only smoke harness
    Task 6 delivers.
    """
    cfg = build_config(method='bl', backbone='resnet18',
                       recipe='quick', data_root=str(synthetic_data), seed=31)
    size = tuple(load_recipe('quick').data.size)  # quick.yaml's data.size, e.g. (512, 512)

    # Building the dataset also builds its pipeline (mmcv.transforms.Compose),
    # which resolves each step through mmcv's *own* TRANSFORMS registry, not
    # mmseg's — the exact same nested-registry gap that affects the model's
    # data_preprocessor build above. Without an active 'mmseg' scope here,
    # 'ChamNetOnlineAugmentation' fails to resolve at all (confirmed by
    # temporarily dropping this `with` block: KeyError, not in the
    # mmengine::transform registry) because it's only registered on mmseg's
    # child TRANSFORMS registry.
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        dataset = DATASETS.build(cfg.train_dataloader['dataset'])
        item = dataset[0]

    inputs = item['inputs']
    assert inputs.shape == (3, *size)   # decoded from the 64x128 synthetic .jpg, then resized
    assert inputs.dtype == torch.uint8  # raw decoded image; SegDataPreProcessor normalises to float later

    gt = item['data_samples'].gt_sem_seg.data
    assert gt.shape == (1, *size)
    assert gt.dtype == torch.int64
    # The synthetic mask .png was written with np.random.randint(0, 8, ...,
    # dtype=np.uint8) — i.e. only values 0-7 ever exist. If the mask suffix,
    # LoadAnnotations' decode path, or Resize's interpolation for the label
    # map were wrong (e.g. bilinear-interpolating class indices, or reading
    # the wrong file), this range check or the dtype check above would catch
    # it.
    assert int(gt.min()) >= 0
    assert int(gt.max()) <= 7


def test_cli_export_config_writes_readable_python(tmp_path):
    from chamnet.cli import main
    out = tmp_path / 'cfg.py'
    assert main(['export-config', '--method', 'bl', '--backbone', 'resnet18',
                 '-o', str(out)]) == 0
    from mmengine.config import Config
    cfg = Config.fromfile(str(out))
    assert cfg.train_cfg['max_iters'] == 3760
