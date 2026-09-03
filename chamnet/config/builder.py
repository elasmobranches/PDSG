"""레시피(어떻게 학습하는가) + 백본표(이 백본이 요구하는 것) → mmengine Config."""
from __future__ import annotations

import copy
from pathlib import Path

from mmengine.config import Config

from chamnet.config.backbones import (BACKBONES, EF_EXTRA_CHANNEL_INIT,
                                      EF_STEM_DELTA, EF_TYPE, HD_BIGATE_TYPE,
                                      HD_OPTIM_EXTRA, HD_RGB_TYPE,
                                      HD_STEM_DELTA, HD_TYPE, NORM, SD_TYPE)
from chamnet.config.combos import validate
from chamnet.config.schema import Recipe, load_recipe

CLASSES = 8
FLOW = {'bl': 'baseline', 'ef': 'proposed', 'sd': 'dual', 'hd': 'dual_plus'}
RGB_MEAN, RGB_STD = [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]


def _preprocessor(size, data_channels, depth_mean_std):
    mean, std = list(RGB_MEAN), list(RGB_STD)
    if data_channels == 4:
        mean.append(depth_mean_std[0])
        std.append(depth_mean_std[1])
    return dict(type='ChamNetSegDataPreProcessor', mean=mean, std=std, bgr_to_rgb=True,
                pad_val=0, seg_pad_val=255, size=tuple(size),
                test_cfg=dict(size_divisor=32))


def _losses(r: Recipe):
    return [
        dict(type='CrossEntropyLoss', use_sigmoid=False,
             loss_weight=r.loss.ce_weight, class_weight=list(r.loss.class_weight)),
        dict(type='DiceLoss', use_sigmoid=False,
             loss_weight=r.loss.dice_weight, eps=r.loss.dice_eps),
    ]


def _decode_head(spec, r: Recipe):
    head = dict(spec['decode_head'])
    # Most heads use SyncBN; a head can pin its own norm_cfg (e.g. LightHamHead
    # uses GroupNorm) — only fill in the default when the head didn't specify one.
    head.setdefault('norm_cfg', NORM)
    # All four BL backbones use every stage ([0,1,2,3]) confirmed against all
    # four fixtures, but this is a fact about the *decoder*, not a universal
    # constant — v12's SegNeXt config dropped stage 0 ([1,2,3]) before the
    # v12+ 512 campaign changed it. Giving it a spec key with this default
    # means a future backbone/method that needs to drop a stage again just
    # sets 'in_index' on its own spec entry instead of branching here.
    head.update(in_channels=list(spec['in_channels']),
                in_index=spec.get('in_index', [0, 1, 2, 3]),
                dropout_ratio=0.1, num_classes=CLASSES,
                align_corners=False, loss_decode=_losses(r))
    return head


def _aux_head(spec, r: Recipe):
    if spec['aux'] is None:
        return None
    aux = dict(spec['aux'])
    aux.update(dropout_ratio=0.1, num_classes=CLASSES, norm_cfg=NORM,
               concat_input=False, align_corners=False,
               loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False,
                                loss_weight=r.loss.aux_weight,
                                class_weight=list(r.loss.class_weight)))
    return aux


def _sd_stem(backbone: str, bl_stem: dict, pretrained, stem_channels: int):
    """Build the SD (Dual, shallow depth-branch) backbone stem for `backbone`.

    Derived from the BL stem for the same backbone rather than written from
    scratch, since three of the four share every RGB-side architecture
    argument with BL verbatim. Confirmed field-for-field against
    tests/fixtures/paper/sd_*.merged.py — per this module's header rule, the
    fixture is the truth if this ever disagrees.

    `stem_channels` isn't threaded into the returned stem dict — every SD
    class hardcodes its RGB stream to 3ch internally and doesn't expose it as
    a config key at all (that's what the `in_channels` pop below is for), so
    there's nothing to set it *to*. It's still taken as a parameter and
    checked here rather than silently ignored: it documents and enforces the
    invariant `build_config` already relies on (stem_channels == 3 for `sd`)
    instead of leaving that fact implicit in the caller alone. A `raise`
    rather than `assert`, since this is library code that must still enforce
    the invariant under `python -O` (which strips asserts), not a test.
    """
    if stem_channels != 3:
        raise ValueError(
            f'SD backbones always take a 3-channel RGB stem; got {stem_channels}')
    if backbone == 'convnext_atto':
        # DualConvNeXtAttoSerial calls timm.create_model itself — model_name,
        # features_only, pretrained and out_indices are all hardcoded inside
        # __init__, so none of TIMMBackbone's config keys (the BL stem) apply.
        return dict(type=SD_TYPE[backbone], fusion_reduction=4)
    stem = copy.deepcopy(bl_stem)
    stem.pop('in_channels', None)  # RGB stream is hardcoded to 3ch inside the class
    stem['type'] = SD_TYPE[backbone]
    stem['fusion_reduction'] = 4
    if backbone == 'resnet18':
        # This project's ResNet-18 is non-dilated (strides=(1,2,2,2)), so the
        # depth branch must halve resolution at every stage too — the
        # class's own default (2,1,1) assumes a dilated RGB backbone instead.
        stem['depth_stage_strides'] = (2, 2, 2)
    if pretrained:
        stem['init_cfg'] = dict(type='Pretrained', checkpoint=pretrained)
    return stem


def _ef_stem(backbone: str, bl_stem: dict, pretrained, stem_channels: int):
    """Build the EF (early fusion, 4-channel input) backbone stem for `backbone`.

    Same shape of derivation as `_sd_stem`/`_hd_stem`: an EF backbone *is* the
    BL backbone with its first convolution widened by one input channel, so
    every other architecture argument is BL's verbatim. Only `type`,
    `extra_channel_init`, `init_cfg` and the per-backbone `in_channels`
    handling in `EF_STEM_DELTA` differ. Confirmed field-for-field against
    tests/fixtures/paper/ef_*.merged.py, which is the truth if this ever
    disagrees.

    Unlike SD and HD, `stem_channels` here is genuinely 4 and genuinely used:
    early fusion feeds the whole RGB-D tensor to one encoder, so the stem's
    input width is the pipeline's output width. It is checked rather than
    assumed for the same reason the other two check theirs, and raises rather
    than asserts so the invariant survives `python -O`.

    `extra_channel_init` is emitted unconditionally and never left to the
    class default. All four EF classes default it to 'zero', every paper
    config passes 'mean', and the difference is invisible to everything except
    the stem weights themselves: same config keys otherwise, same parameter
    count, no error, no warning -- the depth channel would simply start dead
    instead of at the RGB channel mean. See EF_EXTRA_CHANNEL_INIT.
    """
    if stem_channels != 4:
        raise ValueError(
            f'EF backbones take the full RGB-D tensor; got {stem_channels} '
            'channels, expected 4')
    stem = copy.deepcopy(bl_stem)
    for key in EF_STEM_DELTA.get(backbone, {}).get('drop', ()):
        stem.pop(key, None)
    if 'in_channels' in stem:
        stem['in_channels'] = stem_channels
    stem['type'] = EF_TYPE[backbone]
    stem['extra_channel_init'] = EF_EXTRA_CHANNEL_INIT
    if pretrained:
        stem['init_cfg'] = dict(type='Pretrained', checkpoint=pretrained)
    return stem


def _hd_stem(backbone: str, bl_stem: dict, pretrained, stem_channels: int,
             depth_pretrained: bool, ablation: str | None = None):
    """Build the HD (Dual+, heavy depth-branch) backbone stem for `backbone`.

    Same shape of derivation as `_sd_stem`: start from the BL stem for the
    same backbone, since the RGB half of an HD backbone *is* the BL backbone,
    then apply the per-backbone delta recorded in `HD_STEM_DELTA` (read off
    tests/fixtures/paper/hd_*.merged.py, which is the truth if this ever
    disagrees).

    `stem_channels` is checked rather than used, for the same reason as in
    `_sd_stem`: every HD class hardcodes its RGB stream to 3 channels
    internally (DualResNetV1c18LateFusion sets `kwargs['in_channels'] = 3`,
    DualMiTB0LateFusion pops the key outright) and consumes the 4th channel
    through a separate depth encoder, so there is no config key to set it on.
    Raising rather than asserting keeps the invariant enforced under
    `python -O`.

    `ablation` selects among HD's control arms, and the whole point of the
    comparison is that they differ from HD in exactly one thing each -- so
    each is expressed as the smallest possible edit to the same dict rather
    than as its own branch:

    ``'bigate'`` / ``'rgb'``  swap `type` for the class in HD_BIGATE_TYPE /
        HD_RGB_TYPE and change nothing else. Notably `rgb` keeps
        `depth_pretrained` at HD's value: the arm controls for capacity *and*
        for initialisation, so its depth-slot encoder has to start where HD's
        did.
    ``'nogate'``  adds `fusion_use_gate=False`, which makes CrossModalGating
        skip building a gate at all and inject `rgb + d_proj` unweighted.
    ``'shuffled'``  changes no backbone key whatsoever -- it is a pipeline
        step (see `_pipelines`), and reaching this function it is
        indistinguishable from plain HD, which is correct.

    Verified against tests/fixtures/paper/hd-{nogate,bigate,rgb,shuffled}_*
    .merged.py: every one of those twenty backbone dicts is HD's for the same
    backbone with only the edit named above.
    """
    if stem_channels != 3:
        raise ValueError(
            f'HD backbones always take a 3-channel RGB stem; got {stem_channels}')
    hd_type = {'bigate': HD_BIGATE_TYPE, 'rgb': HD_RGB_TYPE}.get(
        ablation, HD_TYPE)[backbone]
    if backbone == 'convnext_atto':
        # DualConvNeXtAttoPlusSerial calls timm.create_model itself for both
        # streams, so none of TIMMBackbone's config keys (the BL stem) apply.
        stem = dict(type=hd_type, fusion_reduction=4,
                    depth_pretrained=depth_pretrained)
    else:
        delta = HD_STEM_DELTA[backbone]
        stem = copy.deepcopy(bl_stem)
        for key in delta['drop']:
            stem.pop(key, None)
        stem.update(copy.deepcopy(delta['add']))
        stem['type'] = hd_type
        stem['depth_pretrained'] = depth_pretrained
        if pretrained:
            stem['init_cfg'] = dict(type='Pretrained', checkpoint=pretrained)
    if ablation == 'nogate':
        stem['fusion_use_gate'] = False
    return stem


def _pipelines(r: Recipe, data_channels, shuffle):
    """Return (train_pipeline, val/test_pipeline) for this run.

    Two things here are easy to get wrong in opposite directions, so both are
    taken from the paper's own merged configs rather than from what sounds
    principled:

    `shuffle` (the depth-structure control) is applied to **every** split --
    train, val and test -- immediately before PackSegInputs. The arm's claim is
    that depth's contribution is its spatial arrangement, so it has to be
    measured on a model that never sees arranged depth at any point; validating
    on unshuffled depth would select checkpoints on a distribution the arm
    never trains or tests on. Reading the fixtures needs care on exactly this
    point: hd-shuffled_*.merged.py and ef-shuffled_*.merged.py do define a
    top-level `val_test_pipeline` without the shuffle step, but nothing uses
    it -- their `val_dataloader.dataset.pipeline` is the shuffled
    `test_pipeline`, verified in all eight. The dataloader is what runs.

    `data_channels == 3` (hd-rgb, and bl) emits **no depth loader at all**,
    rather than loading depth and letting the backbone drop it. hd-rgb feeds
    the RGB image to the depth-slot encoder, so there is no depth to read; the
    preprocessor is 3-channel to match.
    """
    load_depth = dict(type='LoadDepthAsChannel', depth_dir_name='depth',
                      depth_suffix='.npy', img_dir_name='images', max_depth=None)
    resize = dict(type='Resize', scale=tuple(r.data.size), keep_ratio=r.data.keep_ratio)
    aug = dict(type='ChamNetOnlineAugmentation',
               photometric='brightness_contrast', rotate_prob=0.0)
    shuf = [dict(type='ShuffleDepthChannel')] if shuffle else []

    train = [dict(type='LoadImageFromFile')]
    if data_channels == 4:
        train.append(load_depth)
    train += [dict(type='LoadAnnotations'), resize, aug] + shuf + [dict(type='PackSegInputs')]

    test = [dict(type='LoadImageFromFile')]
    if data_channels == 4:
        test.append(load_depth)
    test += [resize, dict(type='LoadAnnotations')] + shuf + [dict(type='PackSegInputs')]
    return train, test


def _dataloader(r: Recipe, split_dir, pipeline, root, batch_size, num_workers,
                shuffle_sampler):
    """Build one split's dataloader.

    `num_workers` is passed in rather than read from the recipe here because
    the paper's runs used a different number for training (8) than for
    evaluation (4), and that is not a performance-only knob. The shuffled
    control arms draw a fresh permutation *per sample* inside the worker
    processes, so the worker count decides which permutations the model is
    scored on: on hd/shuffled/resnet18, 8 workers gives 80.91/80.52 and 4
    gives 80.96/80.43 against a recorded 80.97/80.40, reproducibly. See
    verification/README.md. Emitting one value for every split would silently
    evaluate those arms on different inputs than the paper did.
    """
    return dict(
        batch_size=batch_size, num_workers=num_workers,
        persistent_workers=False,
        sampler=dict(type='InfiniteSampler', shuffle=True) if shuffle_sampler
        else dict(type='DefaultSampler', shuffle=False),
        dataset=dict(type='ChamNet', data_root=root,
                     data_prefix=dict(img_path=f'{split_dir}/images',
                                      seg_map_path=f'{split_dir}/masks'),
                     img_suffix='.jpg', seg_map_suffix='.png', pipeline=pipeline))


def build_config(method: str, backbone: str, ablation: str | None = None,
                 recipe: str | Path | Recipe = 'paper_v13',
                 data_root: str | None = None, seed: int = 31,
                 work_dir: str | None = None) -> Config:
    """(method, backbone, ablation) 을 실행 가능한 mmengine Config 로 만든다."""
    validate(method, ablation)

    r = recipe if isinstance(recipe, Recipe) else load_recipe(recipe)
    try:
        spec = BACKBONES[backbone]
    except KeyError:
        raise ValueError(f"unknown backbone {backbone!r}; choose one of "
                         f"{sorted(BACKBONES)}") from None
    root = data_root or r.data.root

    # data_channels: width of the tensor the pipeline loads and the
    # preprocessor normalises (RGB, or RGB+D once a run trains on depth).
    # stem_channels: the input width of the backbone actually swapped in.
    # ef is the one method where the two are equal at 4: it feeds the whole
    # RGB-D tensor to a single encoder, via dedicated 4-channel-native backbone
    # classes (ResNetV1c4Ch, MixVisionTransformer4Ch, MSCAN4Ch,
    # TIMMBackbone4Ch) rather than by reusing bl's 3-channel one -- see
    # _ef_stem. sd/hd's dual-branch backbones instead keep a fixed 3-channel
    # RGB stem regardless of data_channels (a separate depth branch consumes
    # the extra channel), and hd-rgb ablates that again -- which is why the two
    # stay separate names.
    #
    # hd-rgb is the one arm where a 4-channel *method* loads 3 channels: its
    # depth-slot encoder is fed the RGB image, so depth is never read at all.
    # Not read-and-discarded -- `_pipelines` emits no LoadDepthAsChannel step
    # and the preprocessor's mean/std carry three entries, exactly as in
    # tests/fixtures/paper/hd-rgb_*.merged.py.
    rgb_control = method == 'hd' and ablation == 'rgb'
    data_channels = 3 if method == 'bl' or rgb_control else 4
    stem_channels = 3 if method in ('sd', 'hd') else data_channels
    shuffle = ablation == 'shuffled'

    if method == 'ef':
        stem = _ef_stem(backbone, spec['stem'], spec['pretrained'], stem_channels)
    elif method == 'sd':
        stem = _sd_stem(backbone, spec['stem'], spec['pretrained'], stem_channels)
    elif method == 'hd':
        stem = _hd_stem(backbone, spec['stem'], spec['pretrained'], stem_channels,
                        r.hd.depth_pretrained, ablation)
    else:
        stem = copy.deepcopy(spec['stem'])
        if 'in_channels' in stem:
            stem['in_channels'] = stem_channels
        if spec['pretrained']:
            stem['init_cfg'] = dict(type='Pretrained', checkpoint=spec['pretrained'])

    train_pipe, test_pipe = _pipelines(r, data_channels, shuffle)
    preproc = _preprocessor(r.data.size, data_channels, spec['depth_mean_std'])

    model = dict(type='EncoderDecoder', data_preprocessor=preproc,
                 backbone=stem, decode_head=_decode_head(spec, r),
                 train_cfg=dict(), test_cfg=dict(mode='whole'))
    # The legacy top-level `pretrained` arg on EncoderDecoder is a no-op when it
    # is None (EncoderDecoder only forwards it to the backbone when it is set),
    # so where it appears is purely a fact about which hand-written config each
    # lineage descends from. It appears in every paper config except BL's and
    # SD's ConvNeXt ones — TIMMBackbone loads its own weights and those two
    # configs never bothered to write the key — while HD's ConvNeXt config,
    # written separately, does carry it. Confirmed key-by-key across all twelve
    # fixtures; emitted to match them rather than normalised away.
    if spec['pretrained'] or method == 'hd':
        model['pretrained'] = None
    aux = _aux_head(spec, r)
    if aux is not None:
        model['auxiliary_head'] = aux

    optim = r.optim_for(backbone)
    if method == 'hd':
        # See HD_OPTIM_EXTRA: HD MiT-B0's config states AdamW's default betas
        # explicitly where BL's and SD's leave them implicit.
        optim.update(copy.deepcopy(HD_OPTIM_EXTRA.get(backbone, {})))
    paramwise = optim.pop('paramwise_cfg')
    extra = spec.get('paramwise_extra')
    if extra:
        paramwise = dict(custom_keys={**paramwise['custom_keys'], **extra})
    warm, poly = r.schedule.warmup, r.schedule.poly

    cfg = Config(dict(
        default_scope='mmseg',
        model=model,
        train_dataloader=_dataloader(r, r.data.splits['train'], train_pipe, root,
                                     r.runtime.batch_size,
                                     r.runtime.num_workers, True),
        val_dataloader=_dataloader(r, r.data.splits['val'], test_pipe, root, 1,
                                   r.runtime.num_workers_eval, False),
        test_dataloader=_dataloader(r, r.data.splits['test'], test_pipe, root, 1,
                                    r.runtime.num_workers_eval, False),
        val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
        test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
        optim_wrapper=dict(type='OptimWrapper', optimizer=optim,
                           clip_grad=dict(r.optim.clip_grad),
                           paramwise_cfg=paramwise, accumulative_counts=1),
        param_scheduler=[
            dict(type='LinearLR', start_factor=warm['start_factor'],
                 by_epoch=False, begin=0, end=warm['iters']),
            dict(type='PolyLRRatio', eta_min_ratio=poly['eta_min_ratio'],
                 power=poly['power'], begin=warm['iters'],
                 end=r.schedule.max_iters, by_epoch=False),
        ],
        train_cfg=dict(type='IterBasedTrainLoop', max_iters=r.schedule.max_iters,
                       val_interval=r.runtime.val_interval),
        val_cfg=dict(type='ValLoop'), test_cfg=dict(type='TestLoop'),
        default_hooks=dict(
            checkpoint=dict(type='CheckpointHook', by_epoch=False,
                            save_best=r.runtime.save_best, rule='greater',
                            save_last=False, interval=-1, max_keep_ckpts=1),
            logger=dict(type='LoggerHook', interval=r.runtime.val_interval,
                        log_metric_by_epoch=False),
            timer=dict(type='IterTimerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            sampler_seed=dict(type='DistSamplerSeedHook'),
            visualization=dict(type='SegVisualizationHook', draw=False),
        ),
        custom_hooks=[
            dict(type='EarlyStoppingHook', check_finite=True, rule='greater',
                 **r.runtime.early_stopping),
            dict(type='IterLoggerHook', flush_secs=10, out_csv=None),
        ],
        randomness=dict(seed=seed, deterministic=False, diff_rank_seed=False),
        env_cfg=dict(cudnn_benchmark=True,
                     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                     dist_cfg=dict(backend='nccl')),
        log_processor=dict(by_epoch=False), log_level='INFO', resume=False,
        work_dir=work_dir or f'runs/{FLOW[method]}_{backbone}_{seed}',
    ))
    return cfg
