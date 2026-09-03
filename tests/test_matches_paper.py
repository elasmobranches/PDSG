"""빌더 산출물이 논문을 만든 병합 config 와 같은지 확인한다.

fixture 는 work_dirs_v12+_512bc_pretrained 등에서 가져온 실제 실행 config 다.
데이터 경로 관행(depth_suffix 등)은 논문 실행에서도 split 마다 달랐다 (train은
'.npy', valid/test는 '_depth.npy') — 릴리스가 모든 split 을 '.npy' 하나로
통일했다. 이 차이는 무시하지 않고, 양쪽이 실제로 기대한 값을 말하는지 확인한
뒤에만 비교에서 지운다 — `_normalize_pipeline` 참고.
"""
import pytest
from mmengine.config import Config

import chamnet
from chamnet.config.builder import build_config

chamnet.register_all()

BACKBONES = ['resnet18', 'mit_b0', 'segnext_t', 'convnext_atto']


def _fixture(name):
    # import_custom_modules=False: the fixture's custom_imports point at modules
    # from the original mmsegmentation fork ('hooks', 'mmseg.datasets.chamoe', ...)
    # that don't exist in this release's environment. They only matter for
    # registering classes at train time, not for the literal dict values under
    # test here, so skip trying to import them.
    return Config.fromfile(f'tests/fixtures/paper/{name}.merged.py',
                           import_custom_modules=False)


def _normalize_pipeline(built_pipeline, ref_pipeline, ref_depth_suffix):
    """Assert, then blank, LoadDepthAsChannel's depth_suffix on both sides.

    The paper's own runs did NOT use one depth_suffix — they used both,
    split-dependently. Confirmed directly in the fixtures (e.g.
    hd_mit_b0.merged.py defines `_depth_cfg_train` with `depth_suffix='.npy'`
    and `_depth_cfg_valtest` with `depth_suffix='_depth.npy'`, and
    train_pipeline / val_test_pipeline / test_pipeline inherit them
    respectively): train's depth files are `NAME.npy`, valid's and test's
    are `NAME_depth.npy`. That's a half-migrated dataset — train converted
    to a new naming, valid/test left on the old — not a single paper-wide
    convention. This release picks one convention, `.npy`, for every
    split's pipeline (a defensible simplification, but a real change, not a
    no-op).

    `ref_depth_suffix` is which of the two the *fixture* is expected to say
    for this particular pipeline ('.npy' for train, '_depth.npy' for
    val/test) — the caller must know which pipeline it's normalizing and
    pass the right one. Blanking depth_suffix without asserting what either
    side actually says first would let either half of that story silently
    drift — build_config regressing to per-split suffixes again, or the
    fixture itself being edited — with a fully green suite; the same hole
    NEW-1 found in the preprocessor-type normalizer, one field over. So:
    assert the built side is always '.npy' and the ref side matches
    `ref_depth_suffix`, *then* blank the field so the rest of the pipeline
    is compared exactly.
    """
    def _check_and_blank(pipeline, expected_suffix):
        out = []
        for step in pipeline:
            step = dict(step)
            if step.get('type') == 'LoadDepthAsChannel':
                assert step['depth_suffix'] == expected_suffix, (
                    f'expected depth_suffix={expected_suffix!r}, got '
                    f"{step['depth_suffix']!r}")
                step['depth_suffix'] = None
            out.append(step)
        return out

    return (_check_and_blank(built_pipeline, '.npy'),
            _check_and_blank(ref_pipeline, ref_depth_suffix))


def _normalize_preprocessor_type(built_model, ref_model):
    """Assert, then blank, data_preprocessor's registered type *name*.

    The fixture's literal string 'SegDataPreProcessor' names the production
    fork's own patched class (it never used vanilla's broken one — see
    chamnet/models/data_preprocessor.py). chamnet's builder instead emits
    'ChamNetSegDataPreProcessor', a distinct name for the same corrected
    forward() body (see that module's docstring for why it isn't registered
    under vanilla's name). Comparing the literal strings would fail on a
    difference that isn't a difference in what gets computed, so both names
    are blanked before the caller compares the rest of `model` — but not
    before checking, explicitly, what each one actually says.

    That check is the point of this function existing as more than a blank:
    blanking both sides unconditionally, with no assertion on what was
    there, would silently pass if `build_config` regressed to emitting
    vanilla's own 'SegDataPreProcessor' — restoring its silent >3-channel
    bgr_to_rgb no-op with a fully green suite, since the type name is the
    only place that regression would ever show up in this comparison.
    """
    built_type = built_model['data_preprocessor']['type']
    ref_type = ref_model['data_preprocessor']['type']
    assert built_type == 'ChamNetSegDataPreProcessor', (
        f'build_config must emit ChamNetSegDataPreProcessor, got {built_type!r}')
    assert ref_type == 'SegDataPreProcessor', (
        f"fixture's own preprocessor type changed unexpectedly: {ref_type!r}")

    def _blank(model):
        model = dict(model)
        dp = dict(model['data_preprocessor'])
        dp['type'] = None
        model['data_preprocessor'] = dp
        return model

    return _blank(built_model), _blank(ref_model)


def _assert_matches_paper(built, ref):
    """Compare a builder's output against the merged config it must reproduce.

    Covers: model architecture (backbone / decode_head / auxiliary_head /
    data_preprocessor, modulo the preprocessor's registered type *name* —
    see `_normalize_preprocessor_type`), optim_wrapper, param_scheduler,
    train_cfg, custom_hooks, randomness, val/test_evaluator, default_hooks,
    default_scope, and every train/val/test pipeline *step* (modulo
    LoadDepthAsChannel's depth_suffix — see `_normalize_pipeline`).

    Also covers the dataloaders' own settings — batch_size, num_workers,
    persistent_workers, sampler, and the dataset type/img_suffix — on all
    three splits. `num_workers` in particular was previously excluded as a
    deliberate, numerically harmless change (the release emitted one value
    for every split where the paper used 8 for train and 4 for val/test).
    That was true until the shuffled control arms landed: they draw a fresh
    permutation per sample inside the worker processes, so the worker count
    decides which permutations the model is scored on, and the divergence
    stopped being cosmetic without anything failing. It is asserted now, and
    `_dataloader` emits the paper's per-split values — see that function and
    verification/README.md for the measurements.

    Does NOT cover: two label-path fields and the data root — val/test's
    `data_prefix['seg_map_path']` and `seg_map_suffix`, and `data_root`. The
    release deliberately changed those: the paper's val/test read
    'masks_gray/*_mask_gray.png' where the release's own layout unifies every
    split under 'masks/*.png' (see verification/README.md for the one dataset
    copy where that unification turned out to be stale for a single image),
    and data_root is a path the caller supplies. Comparing those literally
    would fail on an intentional, documented layout change rather than a real
    divergence. `data_prefix['img_path']` is *not* excused with them —
    excluding the whole dict would drop a field that does match — so nothing
    else about the dataloaders is left unasserted. That list used to be
    longer, and `num_workers` is what it cost.
    """
    normalized_built, normalized_ref = _normalize_preprocessor_type(built.model, ref.model)
    assert normalized_built == normalized_ref
    assert built.optim_wrapper == ref.optim_wrapper
    assert built.param_scheduler == ref.param_scheduler
    assert built.train_cfg == ref.train_cfg
    assert built.custom_hooks == ref.custom_hooks
    assert built.randomness == ref.randomness
    assert built.val_evaluator == ref.val_evaluator
    assert built.test_evaluator == ref.test_evaluator
    assert built.default_hooks == ref.default_hooks
    assert built.default_scope == ref.default_scope
    built_train, ref_train = _normalize_pipeline(
        built.train_dataloader['dataset']['pipeline'],
        ref.train_dataloader['dataset']['pipeline'], '.npy')
    assert built_train == ref_train
    built_val, ref_val = _normalize_pipeline(
        built.val_dataloader['dataset']['pipeline'],
        ref.val_dataloader['dataset']['pipeline'], '_depth.npy')
    assert built_val == ref_val
    built_test, ref_test = _normalize_pipeline(
        built.test_dataloader['dataset']['pipeline'],
        ref.test_dataloader['dataset']['pipeline'], '_depth.npy')
    assert built_test == ref_test
    _assert_dataloader_settings_match(built, ref)


def _assert_dataloader_settings_match(built, ref):
    """Compare every dataloader field except the three data-layout ones.

    Named separately from the pipeline comparison because these are the
    fields that decide *how* the pipeline is run rather than what it does,
    and because one of them — num_workers — spent several revisions in the
    'deliberately different, numerically harmless' list and then stopped
    being harmless when a per-sample random transform arrived. The lesson
    generalises past that one key, so the loop is written the other way
    round: every key present on either side is compared unless it is one of
    the three the release genuinely changed, so a field added to the builder
    or appearing in a future fixture is covered by default instead of
    needing to be remembered.
    """
    # val/test read 'masks_gray/*_mask_gray.png' in the paper's own runs and
    # 'masks/*.png' in the release's unified layout; data_root is whatever the
    # caller passed. Only those two label-path fields are excused -- not the
    # whole `data_prefix` dict, whose other entry, img_path, is the same on
    # both sides and stays compared. See this function's docstring and the
    # module header.
    layout_fields = {'seg_map_suffix', 'data_root'}
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        b, r = built[name], ref[name]
        assert set(b) == set(r), (
            f'{name}: keys differ, built {sorted(set(b) - set(r))} / '
            f'fixture {sorted(set(r) - set(b))}')
        for key in sorted(b):
            if key == 'dataset':
                continue
            assert b[key] == r[key], f'{name}.{key}: {b[key]!r} != {r[key]!r}'
        bd, rd = b['dataset'], r['dataset']
        assert set(bd) == set(rd), f'{name}.dataset: keys differ'
        for key in sorted(bd):
            if key == 'pipeline' or key in layout_fields:
                continue
            if key == 'data_prefix':
                assert set(bd[key]) == set(rd[key]), f'{name}.dataset.data_prefix keys'
                for pk in sorted(bd[key]):
                    if pk == 'seg_map_path':
                        continue
                    assert bd[key][pk] == rd[key][pk], (
                        f'{name}.dataset.data_prefix.{pk}: '
                        f'{bd[key][pk]!r} != {rd[key][pk]!r}')
                continue
            assert bd[key] == rd[key], (
                f'{name}.dataset.{key}: {bd[key]!r} != {rd[key]!r}')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_bl_matches_paper(backbone):
    built = build_config(method='bl', backbone=backbone,
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'bl_{backbone}'))


@pytest.mark.parametrize('backbone', BACKBONES)
def test_ef_matches_paper(backbone):
    built = build_config(method='ef', backbone=backbone,
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'ef_{backbone}'))


@pytest.mark.parametrize('backbone', BACKBONES)
def test_sd_matches_paper(backbone):
    built = build_config(method='sd', backbone=backbone,
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'sd_{backbone}'))


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_matches_paper(backbone):
    built = build_config(method='hd', backbone=backbone,
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'hd_{backbone}'))


# ---------------------------------------------------------------------------
# The five control arms. Each is HD (or EF) with exactly one thing changed, and
# the fixture is what says which thing -- so these run through the same
# `_assert_matches_paper` as the four training methods, comparing the whole
# config rather than the key the arm is named after. A control arm that also
# moved the learning rate, the augmentation or the evaluator would not be a
# control, and only a whole-config comparison can see that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_nogate_matches_paper(backbone):
    built = build_config(method='hd', backbone=backbone, ablation='nogate',
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'hd-nogate_{backbone}'))
    assert built.model['backbone']['fusion_use_gate'] is False


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_bigate_matches_paper(backbone):
    built = build_config(method='hd', backbone=backbone, ablation='bigate',
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'hd-bigate_{backbone}'))


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_rgb_matches_paper(backbone):
    """The capacity control: RGB into the depth slot, and no depth read at all.

    The last part is the one a reader is most likely to get wrong, so it is
    asserted here as well as being implied by the pipeline comparison: an
    hd-rgb config has *no* LoadDepthAsChannel step in any split and a
    three-element mean/std. Depth is never loaded, not loaded and discarded.
    """
    built = build_config(method='hd', backbone=backbone, ablation='rgb',
                         recipe='paper_v13', seed=37)
    ref = _fixture(f'hd-rgb_{backbone}')
    _assert_matches_paper(built, ref)

    for cfg, who in ((built, 'build_config'), (ref, 'the paper fixture')):
        assert len(cfg.model['data_preprocessor']['mean']) == 3, who
        assert len(cfg.model['data_preprocessor']['std']) == 3, who
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            types = [step['type'] for step in cfg[name]['dataset']['pipeline']]
            assert 'LoadDepthAsChannel' not in types, f'{who}: {name}'
    # ...and the depth-slot encoder still starts from HD's initialisation, or
    # the arm would be controlling for two things at once.
    assert built.model['backbone']['depth_pretrained'] is True


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_shuffled_matches_paper(backbone):
    _assert_shuffled_matches_paper('hd', backbone)


@pytest.mark.parametrize('backbone', BACKBONES)
def test_ef_shuffled_matches_paper(backbone):
    _assert_shuffled_matches_paper('ef', backbone)


def _assert_shuffled_matches_paper(method, backbone):
    """Shared body for the two shuffled arms, including where the step goes.

    ShuffleDepthChannel must be in **all three** dataloader pipelines --
    train, val and test. That is what the paper's runs did, and it is easy to
    read the fixtures as saying otherwise: they also define a top-level
    `val_test_pipeline` that has no shuffle step, but no dataloader refers to
    it. Asserting the placement on both sides makes which one is authoritative
    explicit, so a later "fix" that drops val back to unshuffled depth fails
    here with a reason instead of just failing a dict comparison.
    """
    built = build_config(method=method, backbone=backbone, ablation='shuffled',
                         recipe='paper_v13', seed=37)
    ref = _fixture(f'{method}-shuffled_{backbone}')
    _assert_matches_paper(built, ref)

    for cfg, who in ((built, 'build_config'), (ref, 'the paper fixture')):
        for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            types = [step['type'] for step in cfg[name]['dataset']['pipeline']]
            assert types.count('ShuffleDepthChannel') == 1, f'{who}: {name} {types}'
            assert types.index('ShuffleDepthChannel') == len(types) - 2, (
                f'{who}: {name} must shuffle immediately before PackSegInputs, '
                f'got {types}')


def test_hd_mit_b0_stem_omissions_are_mixvisiontransformer_defaults():
    """The paper's HD MiT-B0 backbone dict omits three keys its BL sibling sets.

    `chamnet.config.backbones.HD_STEM_DELTA['mit_b0']` drops in_channels,
    num_stages and patch_sizes when deriving the HD stem from the BL one,
    because the paper's HD MiT config (hand-written from the upstream
    SegFormer reference rather than derived from this project's BL config)
    doesn't carry them. `test_hd_matches_paper` only checks that the emitted
    dict matches that config literally — it would pass just as happily if
    dropping those keys silently changed the architecture. This test is the
    other half: the omitted keys are only harmless because
    MixVisionTransformer's own defaults are the same values BL passes
    explicitly, so HD's RGB stream is the same network as BL's. Read the
    defaults out of the class's actual signature rather than restating them,
    so an mmseg upgrade that changed one fails here instead of quietly
    training a different backbone.

    in_channels is checked against 3 rather than against the BL value for a
    different reason: DualMiTB0LateFusion pops the key and hardcodes 3, since
    the 4th (depth) channel goes to a separate encoder.
    """
    import inspect

    from mmseg.models.backbones.mit import MixVisionTransformer

    from chamnet.config.backbones import BACKBONES, HD_STEM_DELTA

    defaults = {name: param.default for name, param
                in inspect.signature(MixVisionTransformer.__init__).parameters.items()}
    bl_stem = BACKBONES['mit_b0']['stem']
    dropped = HD_STEM_DELTA['mit_b0']['drop']
    assert set(dropped) == {'in_channels', 'num_stages', 'patch_sizes'}
    assert defaults['in_channels'] == 3 == bl_stem['in_channels']
    for key in ('num_stages', 'patch_sizes'):
        assert defaults[key] == bl_stem[key], (
            f'MixVisionTransformer default for {key!r} is {defaults[key]!r}, but '
            f'the BL config passes {bl_stem[key]!r}; dropping the key for HD '
            f'would change the architecture, not just the spelling')
    # The two keys HD *adds* are likewise the class defaults, so the same
    # argument runs in the other direction.
    for key, value in HD_STEM_DELTA['mit_b0']['add'].items():
        assert defaults[key] == value


def test_ef_convnext_in_channels_drop_relies_on_timms_own_default():
    """EF's ConvNeXt backbone dict drops the in_channels key its BL sibling sets.

    `chamnet.config.backbones.EF_STEM_DELTA['convnext_atto']` removes it when
    deriving the EF stem, because the paper's EF ConvNeXt config doesn't carry
    it — and `test_ef_matches_paper` only checks that the emitted dict matches
    that config literally, so it would pass just as happily if dropping the key
    silently changed how many channels timm builds. This test is the other
    half.

    TIMMBackbone4Ch is the one EF class that must NOT hand its 4 channels
    straight to the underlying model: it deliberately builds timm at 3 channels
    so the pretrained stem loads unmodified, then widens the stem conv itself
    (`kwargs['in_channels'] = 3` in its __init__ — see
    chamnet/models/backbones/early_fusion.py for why timm's own
    adapt_input_conv is avoided). Dropping the config key is therefore harmless
    *only because* TIMMBackbone's own default is already 3. Read that default
    out of the class's actual signature rather than restating it, so an mmseg
    upgrade that changed it fails here instead of quietly building a stem of a
    different width and then widening the wrong thing.

    The complementary check — that the emitted config really does yield a
    4-channel stem on all four backbones, key or no key — is
    tests/test_ablation_semantics.py::test_ef_stem_takes_four_channels_
    initialised_from_the_rgb_mean, which looks at the built conv itself.
    """
    import inspect

    from mmseg.models.backbones.timm_backbone import TIMMBackbone

    from chamnet.config.backbones import BACKBONES, EF_STEM_DELTA

    assert set(EF_STEM_DELTA) == {'convnext_atto'}, (
        'EF_STEM_DELTA grew an entry without a matching defaults check here')
    assert set(EF_STEM_DELTA['convnext_atto']['drop']) == {'in_channels'}
    assert BACKBONES['convnext_atto']['stem']['in_channels'] == 3, (
        "precondition: BL's ConvNeXt stem is the 3-channel one EF derives from")

    default = inspect.signature(TIMMBackbone.__init__).parameters['in_channels'].default
    assert default == 3, (
        f'TIMMBackbone default in_channels is {default!r}, not 3; dropping the '
        'key for EF would build the timm model at a different width than the '
        'pretrained stem expects, not just omit a redundant spelling')


def test_hd_mit_b0_explicit_betas_are_the_adamw_defaults():
    """HD MiT-B0's config states AdamW's betas; BL's and SD's don't.

    `chamnet.config.backbones.HD_OPTIM_EXTRA` reproduces that so
    `test_hd_matches_paper` can compare optim_wrapper literally. The reason
    that is a spelling difference and not a training difference is that the
    stated value is torch's own default — which is an assumption about
    torch, so check it against torch rather than asserting it in a comment.
    If a future torch changed the default, HD MiT-B0 and BL/SD MiT-B0 would
    genuinely stop training with the same optimizer and this fails.
    """
    import inspect

    import torch

    from chamnet.config.backbones import HD_OPTIM_EXTRA

    default = inspect.signature(torch.optim.AdamW.__init__).parameters['betas'].default
    assert tuple(default) == HD_OPTIM_EXTRA['mit_b0']['betas']
    assert set(HD_OPTIM_EXTRA) == {'mit_b0'}, (
        'HD_OPTIM_EXTRA grew an entry without a matching defaults check here')


def test_unknown_backbone_raises_clear_error():
    with pytest.raises(ValueError, match='unknown backbone'):
        build_config(method='bl', backbone='nonexistent', recipe='paper_v13')


def test_timm_default_tag_for_convnext_atto_is_pinned_by_assumption():
    """The paper's fixture spells the backbone plain 'convnext_atto', not a
    tagged variant, so the checkpoint it actually loaded was whatever timm
    resolves as that model's *default* pretrained tag — currently 'd2_in1k'.
    chamnet's backbone table follows the fixture (see backbones.py) rather than
    pinning a tag explicitly, which means the released code depends on timm
    never changing that default. If timm ever does, `build_config` would keep
    running and silently load different weights than the paper used, with no
    error anywhere. This test exists to fail loudly the day that assumption
    breaks, so someone notices and pins the tag explicitly instead.
    """
    import timm
    model = timm.create_model('convnext_atto', pretrained=False)
    assert model.default_cfg['tag'] == 'd2_in1k'
