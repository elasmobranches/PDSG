"""빌더 산출물이 논문을 만든 병합 config 와 같은지 확인한다.

fixture 는 work_dirs_v12+_512bc_pretrained 등에서 가져온 실제 실행 config 다.
데이터 경로 관행(depth_suffix 등)은 릴리스에서 통일했으므로 비교하지 않는다 — 스펙 §4.3.
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


def _normalize_pipeline(pipeline):
    """Blank out the one pipeline detail the release deliberately changed:
    LoadDepthAsChannel's depth_suffix (the paper's runs used `_depth.npy`;
    the release normalised the data layout to plain `.npy`) — spec §4.3.
    Every other step and key must match exactly, or the equivalence claim
    doesn't hold.
    """
    out = []
    for step in pipeline:
        step = dict(step)
        if step.get('type') == 'LoadDepthAsChannel':
            step['depth_suffix'] = None
        out.append(step)
    return out


@pytest.mark.parametrize('backbone', BACKBONES)
def test_bl_matches_paper(backbone):
    built = build_config(method='bl', backbone=backbone,
                         recipe='paper_v13', seed=37)
    ref = _fixture(f'bl_{backbone}')
    assert built.model == ref.model
    assert built.optim_wrapper == ref.optim_wrapper
    assert built.param_scheduler == ref.param_scheduler
    assert built.train_cfg == ref.train_cfg
    assert built.custom_hooks == ref.custom_hooks
    assert built.randomness == ref.randomness
    assert built.val_evaluator == ref.val_evaluator
    assert built.test_evaluator == ref.test_evaluator
    assert built.default_hooks == ref.default_hooks
    assert built.default_scope == ref.default_scope
    assert _normalize_pipeline(built.train_dataloader['dataset']['pipeline']) == \
        _normalize_pipeline(ref.train_dataloader['dataset']['pipeline'])
    assert _normalize_pipeline(built.val_dataloader['dataset']['pipeline']) == \
        _normalize_pipeline(ref.val_dataloader['dataset']['pipeline'])


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
