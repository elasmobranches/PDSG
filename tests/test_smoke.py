import pytest
import torch

import chamnet
from chamnet.config.builder import build_config
from chamnet.config.schema import load_recipe
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
    with chamnet.scoped(cfg):
        model = MODELS.build(cfg.model)
    x = torch.randn(2, 3, 64, 128)
    feats = model.backbone(x)
    logits = model.decode_head(feats)
    assert logits.shape[1] == 8
    logits.sum().backward()
    assert any(p.grad is not None for p in model.backbone.parameters())


def test_sd_builds_backprops_and_round_trips_depth(synthetic_data):
    """SD (Dual, shallow depth-branch) coverage: the depth .npy round trip
    through LoadDepthAsChannel/PackSegInputs, and a real forward+backward
    pass through DualResNetV1c18. This is the only test that builds a real
    4-channel input against the ported SD classes — before this, they had
    zero in-repo forward-pass coverage; tools/replay.py's checkpoint replay
    is a one-off verification artifact run against real data on the GPU
    wrapper, not a test that runs in this suite.

    A forward pass here is itself a meaningful check on `_sd_stem`'s
    `depth_stage_strides` derivation: get that wrong (e.g. leave
    DualResNetV1c18's dilated-backbone default `(2, 1, 1)` instead of this
    project's non-dilated `(2, 2, 2)`) and CrossModalGating's
    `rgb + d_proj * gate` raises a shape mismatch the moment the RGB and
    depth streams' spatial sizes disagree at stage 2 or 3 — there's no way
    to silently get the gate shapes wrong and still have this pass.
    """
    cfg = build_config(method='sd', backbone='resnet18',
                       recipe='quick', data_root=str(synthetic_data), seed=31)
    size = tuple(load_recipe('quick').data.size)

    with chamnet.scoped(cfg):
        dataset = DATASETS.build(cfg.train_dataloader['dataset'])
        item = dataset[0]

    inputs = item['inputs']
    # 4 channels: 3 RGB + 1 depth, appended by LoadDepthAsChannel. The
    # concatenation upcasts the whole array to the depth channel's float32
    # (contrast BL's uint8 RGB-only inputs, asserted below) — an all-uint8
    # result here would itself mean LoadDepthAsChannel silently no-opped.
    assert inputs.shape == (4, *size)
    assert inputs.dtype == torch.float32

    with chamnet.scoped(cfg):
        model = MODELS.build(cfg.model)
    x = torch.randn(2, 4, 64, 128)
    feats = model.backbone(x)
    logits = model.decode_head(feats)
    assert logits.shape[1] == 8
    logits.sum().backward()
    # Both streams must receive gradient: the RGB backbone (as BL's test
    # above already checks) *and* the depth branch + fusion gates, which BL
    # has none of. A depth path that forward()s fine but was never actually
    # wired into the computation graph (e.g. depth_branch computed and
    # discarded) would still pass BL-style coverage while training nothing
    # on the depth side.
    assert any(p.grad is not None for p in model.backbone.depth_branch.parameters())
    assert any(p.grad is not None for p in model.backbone.fusions.parameters())


def test_ef_builds_backprops_and_round_trips_four_channel_input(synthetic_data):
    """EF (early fusion) coverage: one encoder, four input channels.

    Where SD and HD send depth down a second encoder, EF concatenates it onto
    the RGB tensor and hands the whole thing to a single backbone whose stem
    conv was widened by one input channel. That makes the pipeline's output
    width and the backbone's input width the same number, and a mismatch
    between them is a hard RuntimeError on the very first conv — which is
    exactly what `build_config(method='ef', ...)` used to produce before the
    4-channel backbone classes existed, since it emitted a plain 3-channel
    `ResNetV1c` for a 4-channel pipeline. A dict-comparison test cannot see
    that; only running a real tensor through can.

    So this builds the dataset the config describes, takes a real item
    through LoadDepthAsChannel/PackSegInputs, and separately runs a forward
    and backward pass. The gradient check is the part that would catch a stem
    that was widened but left disconnected: the 4th input channel's weights
    have to receive gradient like any other, and a stem conv that quietly
    stayed 3-channel would have failed the forward pass above it.
    """
    cfg = build_config(method='ef', backbone='resnet18',
                       recipe='quick', data_root=str(synthetic_data), seed=31)
    size = tuple(load_recipe('quick').data.size)

    with chamnet.scoped(cfg):
        dataset = DATASETS.build(cfg.train_dataloader['dataset'])
        item = dataset[0]
    assert item['inputs'].shape == (4, *size)
    assert item['inputs'].dtype == torch.float32

    with chamnet.scoped(cfg):
        model = MODELS.build(cfg.model)
    stem_conv = model.backbone.stem[0]
    assert stem_conv.in_channels == 4

    x = torch.randn(2, 4, 64, 128)
    feats = model.backbone(x)
    logits = model.decode_head(feats)
    assert logits.shape[1] == 8
    logits.sum().backward()
    assert stem_conv.weight.grad is not None
    assert stem_conv.weight.grad[:, 3].abs().sum() > 0, (
        'the depth input channel received no gradient — it is not actually '
        'wired into the forward pass')


def test_hd_builds_and_backprops_through_both_encoders(synthetic_data):
    """HD (Dual+, heavy depth-branch) forward/backward coverage.

    HD replaces SD's small depthwise-separable depth branch with a second
    full copy of the RGB architecture, so the two streams' stage resolutions
    have to line up on their own rather than by construction. Wrong strides,
    a wrong stage-dim table, or a dropped stem key that turned out not to be
    the class default would all show up as a shape mismatch the first time
    CrossModalGating computes `rgb + d_proj * gate`. Only a real forward pass
    catches that; the fixture-equivalence tests compare dicts and would not.

    Backbones: resnet18 for the stride/dilation geometry, and mit_b0 because
    its config is the one whose backbone dict omits keys BL states explicitly
    (see HD_STEM_DELTA) — if any of those omissions were not the class
    default, this is where it would surface. segnext_t and convnext_atto are
    left out because building the ConvNeXt one downloads timm weights inside
    __init__; their depth-encoder behaviour is covered in
    tests/test_ablation_semantics.py.
    """
    for backbone in ('resnet18', 'mit_b0'):
        cfg = build_config(method='hd', backbone=backbone, recipe='quick',
                           data_root=str(synthetic_data), seed=31)
        with chamnet.scoped(cfg):
            model = MODELS.build(cfg.model)
        x = torch.randn(2, 4, 64, 128)
        feats = model.backbone(x)
        logits = model.decode_head(feats)
        assert logits.shape[1] == 8, backbone
        logits.sum().backward()
        # The depth encoder is a whole second backbone; if it were computed
        # and then discarded (or never wired into the gates), the forward
        # pass would still succeed and only the gradient check would notice.
        assert any(p.grad is not None
                   for p in model.backbone.depth_backbone.parameters()), backbone
        assert any(p.grad is not None
                   for p in model.backbone.fusions.parameters()), backbone


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
    mask paths only. Depth (.npy) round-trip coverage is
    `test_sd_builds_backprops_and_round_trips_depth`, above.
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
    with chamnet.scoped(cfg):
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


@pytest.mark.parametrize('backbone', ['resnet18', 'mit_b0', 'segnext_t',
                                      'convnext_atto'])
def test_cli_export_config_produces_a_usable_ef_config(backbone, tmp_path):
    """`chamnet list` advertises `ef`; `chamnet export-config --method ef`
    has to actually produce a config for it, on every backbone.

    Both CLI paths read the same table (`chamnet.config.combos.VALID`), so
    `list` advertised `ef` from the first commit — but for several revisions
    `export-config --method ef` emitted a plain 3-channel backbone for a
    4-channel pipeline, i.e. a file that parsed fine and then died on the
    first convolution. `list` naming a combination the exporter cannot serve
    is the failure this pins shut: a user's first two commands must agree.

    The exported file is re-read from disk rather than inspected as an
    in-memory Config, because that round trip through `Config.dump` /
    `Config.fromfile` is what a user actually gets. The `in_channels`
    assertions are per backbone on purpose — the paper's own EF configs
    disagree about whether to state the key at all, and
    tests/test_matches_paper.py covers why that is safe.
    """
    from mmengine.config import Config

    from chamnet.cli import main
    from chamnet.config.backbones import EF_TYPE
    from chamnet.config.combos import VALID

    assert ('ef', None) in VALID, "precondition: `chamnet list` advertises ef"
    out = tmp_path / f'ef_{backbone}.py'
    assert main(['export-config', '--method', 'ef', '--backbone', backbone,
                 '-o', str(out)]) == 0

    cfg = Config.fromfile(str(out))
    stem = cfg.model['backbone']
    assert stem['type'] == EF_TYPE[backbone]
    assert stem['extra_channel_init'] == 'mean'
    # The pipeline loads 4 channels, the preprocessor normalises 4, and the
    # backbone is one of the 4-channel-native classes: the three have to agree
    # or the config is the crashing kind described above.
    assert len(cfg.model['data_preprocessor']['mean']) == 4
    assert len(cfg.model['data_preprocessor']['std']) == 4
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        types = [s['type'] for s in cfg[name]['dataset']['pipeline']]
        assert 'LoadDepthAsChannel' in types, name
    if backbone == 'convnext_atto':
        assert 'in_channels' not in stem
    elif 'in_channels' in stem:
        assert stem['in_channels'] == 4


def test_cli_list_only_advertises_combinations_the_builder_can_serve(capsys):
    """`chamnet list` must not name a method whose config would crash.

    Both `list` and `export-config` read `combos.VALID`, so `list` advertised
    `ef` from the first commit — while `build_config(method='ef', ...)` still
    emitted a plain 3-channel backbone class for a 4-channel pipeline, i.e. a
    config that parsed cleanly and then died on the first convolution. A
    user's first two commands disagreed, and nothing in the suite noticed.

    So this checks the agreement itself, on two levels. Structurally: what
    `list` prints must *be* `combos.VALID`, method for method and ablation
    for ablation — parsed back out of the printed lines and compared as a
    set, so a combination that is advertised but dead, or valid but never
    advertised, fails here instead of being found by a user. (`cli.METHODS`
    is derived from `VALID` precisely so the first of those cannot happen;
    this keeps it that way, and covers the ablation axis too, where the
    CLI's accepted vocabulary is deliberately a superset.)

    Then, for every advertised method on every backbone: the emitted backbone
    type must actually be registered, and its input width must match the
    width the pipeline and preprocessor produce — a 4-channel run has to
    name one of the 4-channel-capable classes (early fusion's widened stems,
    or a dual encoder that splits the tensor), and a 3-channel run has to
    name the plain baseline class. Building every one of those models for
    real is the all-combination smoke test's job; this is the cheap,
    hermetic half that pins the table-level mismatch.
    """
    from chamnet.cli import main
    from chamnet.config.backbones import BACKBONES, EF_TYPE, HD_TYPE, SD_TYPE
    from chamnet.config.combos import VALID

    four_channel_types = set(EF_TYPE.values()) | set(SD_TYPE.values()) | set(
        HD_TYPE.values())

    assert main(['list']) == 0
    listed = capsys.readouterr().out
    methods = sorted({m for m, _ in VALID})
    assert methods == ['bl', 'ef', 'hd', 'sd']

    # Parse the printed table back: 'bl  × <backbones>  ablation: (없음), rgb'
    printed = {}
    for line in listed.splitlines():
        if '  ablation: ' not in line:
            continue
        head, tail = line.split('  ablation: ', 1)
        printed[head.split()[0]] = {None if x == '(없음)' else x
                                    for x in tail.split(', ')}
    assert printed == {m: {a for mm, a in VALID if mm == m} for m in methods}, (
        f'`chamnet list` printed {printed!r}, which is not combos.VALID')

    for method in methods:
        for backbone in sorted(BACKBONES):
            cfg = build_config(method=method, backbone=backbone, seed=31)
            stem_type = cfg.model['backbone']['type']
            assert MODELS.get(stem_type) is not None, (
                f'{method}/{backbone}: backbone type {stem_type!r} is not '
                'registered — register_all() does not import it')
            channels = len(cfg.model['data_preprocessor']['mean'])
            assert channels == len(cfg.model['data_preprocessor']['std'])
            if channels == 4:
                assert stem_type in four_channel_types, (
                    f'{method}/{backbone}: the pipeline loads 4 channels but '
                    f'the backbone is {stem_type!r}, a 3-channel class — this '
                    'config crashes on the first convolution')
            else:
                assert stem_type == BACKBONES[backbone]['stem']['type'], (
                    f'{method}/{backbone}: 3-channel run should use the plain '
                    f'baseline backbone, got {stem_type!r}')
