import pytest
import torch

import chamnet
from chamnet.config.builder import build_config
from chamnet.config.combos import VALID
from chamnet.config.schema import load_recipe
from mmseg.registry import DATASETS, MODELS

chamnet.register_all()


ALL_COMBINATIONS = [
    # convnext_atto is the one backbone whose classes call timm.create_model
    # with pretrained=True inside __init__, so merely *building* one downloads
    # a checkpoint. Marked rather than skipped, per the `network` marker's note
    # in pyproject.toml -- deselect with -m "not network" and the record of
    # what went unchecked is explicit.
    pytest.param(method, backbone, ablation,
                 marks=[pytest.mark.network] if backbone == 'convnext_atto' else [])
    for method, ablation in sorted(VALID, key=str)
    for backbone in ['resnet18', 'mit_b0', 'segnext_t', 'convnext_atto']
]

# Where each method keeps the modules that consume depth. A forward and
# backward pass has to put gradient on them, or the run is training an RGB
# model with a decorative second encoder attached.
DEPTH_MODULES = {
    'bl': (),
    'ef': (),                             # one widened stem; checked separately
    'sd': ('depth_branch', 'fusions'),
    'hd': ('depth_backbone', 'fusions'),
}


@pytest.mark.parametrize('method,backbone,ablation', ALL_COMBINATIONS)
def test_every_combination_builds_and_backprops(method, backbone, ablation,
                                                synthetic_data):
    """Every combination `chamnet list` advertises must build and train.

    This is the broad half of the smoke coverage: all 36 of them, on real
    synthetic data, through the real pipeline. It replaces four
    method-specific tests (bl/sd/ef/hd on resnet18) and keeps every check they
    made, generalised so it applies to each arm rather than to one:

    * The pipeline actually runs. `dataset[0]` goes through
      LoadImageFromFile -> LoadDepthAsChannel (where the arm has depth) ->
      LoadAnnotations -> Resize -> ChamNetOnlineAugmentation ->
      ShuffleDepthChannel (where the arm shuffles) -> PackSegInputs. A config
      whose pipeline and backbone disagree about width is a hard error on the
      first convolution -- which is exactly what `method='ef'` produced before
      the 4-channel backbone classes existed, and what `ablation='rgb'` would
      produce if its depth loader were not removed.
    * The depth `.npy` round trip. A 4-channel arm's item must come back with
      four channels and float32 dtype (the concatenation upcasts the uint8
      RGB); an all-uint8 result would mean LoadDepthAsChannel silently
      no-opped. A 3-channel arm's must stay uint8.
    * Forward, 8 output classes, backward.
    * Gradient reaches the depth pathway. Depth features that are computed and
      then dropped -- an encoder never wired into the gates, a stem widened but
      disconnected -- pass a forward test and fail here. What counts as "the
      depth pathway" differs per method (DEPTH_MODULES, and EF's widened stem
      below), so it is looked up rather than assumed.

    A forward pass is also the only check on the two streams' stage
    resolutions lining up: SD's `depth_stage_strides`, HD's second full
    backbone, and every ablation's inherited version of both. Get any of them
    wrong and CrossModalGating's `rgb + d_proj * gate` raises a shape mismatch
    here. The fixture-equivalence tests compare dicts and cannot see it.
    """
    cfg = build_config(method=method, backbone=backbone, ablation=ablation,
                       recipe='quick', data_root=str(synthetic_data), seed=31)
    size = tuple(load_recipe('quick').data.size)
    in_channels = len(cfg.model['data_preprocessor']['mean'])

    # `chamnet.scoped(cfg)` rather than a global scope: `Runner.from_cfg` reads
    # cfg.default_scope itself, but building a model or dataset directly
    # bypasses Runner and mmengine then resolves nested types (the model's
    # data_preprocessor, an mmseg-only pipeline step) against its own root
    # registry and fails. Scoping at the call site keeps `register_all()`
    # side-effect-free for anyone importing chamnet alongside another
    # mmengine-based library -- see that function's docstring.
    with chamnet.scoped(cfg):
        dataset = DATASETS.build(cfg.train_dataloader['dataset'])
        item = dataset[0]
    assert item['inputs'].shape == (in_channels, *size)
    assert item['inputs'].dtype == (torch.float32 if in_channels == 4
                                    else torch.uint8)

    with chamnet.scoped(cfg):
        model = MODELS.build(cfg.model)
    feats = model.backbone(torch.randn(2, in_channels, 64, 128))
    logits = model.decode_head(feats)
    assert logits.shape[1] == 8
    logits.sum().backward()
    assert any(p.grad is not None for p in model.backbone.parameters())

    for name in DEPTH_MODULES[method]:
        module = getattr(model.backbone, name)
        assert any(p.grad is not None for p in module.parameters()), (
            f'{method}/{backbone}/{ablation}: no gradient reached '
            f'backbone.{name}')

    if method == 'ef':
        # Early fusion has no second encoder: its entire use of depth is the
        # stem conv's fourth input plane, so that plane is the thing that must
        # receive gradient. Located by width rather than by a path table --
        # "exactly one 4-input-channel module" is what early fusion means.
        widened = [m for m in model.backbone.modules()
                   if getattr(m, 'in_channels', None) == 4]
        assert len(widened) == 1, widened
        assert widened[0].weight.grad[:, 3].abs().sum() > 0, (
            'the depth input channel received no gradient -- it is not '
            'actually wired into the forward pass')


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
    `test_every_combination_builds_and_backprops` above, on each of the 32
    combinations whose pipeline loads four channels.
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

    # Parse the printed table back: 'bl  × <backbones>  ablation: (none), rgb'
    printed = {}
    for line in listed.splitlines():
        if '  ablation: ' not in line:
            continue
        head, tail = line.split('  ablation: ', 1)
        printed[head.split()[0]] = {None if x == '(none)' else x
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
