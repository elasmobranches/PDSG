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

    Does NOT cover: non-pipeline dataloader fields — num_workers, batch_size,
    data_prefix/seg_map_path, seg_map_suffix, img_suffix. The release
    deliberately changed these: the paper's runs used num_workers=4 for
    val/test vs. 8 for train, and val/test read
    'masks_gray/*_mask_gray.png' where the release's own layout unifies
    every split under 'masks/*.png' (see verification/README.md for the one
    dataset copy where that unification turned out to be stale for a single
    image). Comparing those fields
    literally would fail on an intentional, documented layout change, not a
    real divergence — what determines the paper's numbers is the model and
    how each image moves through it, and that is checked in full.
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


@pytest.mark.parametrize('backbone', BACKBONES)
def test_bl_matches_paper(backbone):
    built = build_config(method='bl', backbone=backbone,
                         recipe='paper_v13', seed=37)
    _assert_matches_paper(built, _fixture(f'bl_{backbone}'))


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
