"""레시피(어떻게 학습하는가) + 백본표(이 백본이 요구하는 것) → mmengine Config."""
from __future__ import annotations

import copy
from pathlib import Path

from mmengine.config import Config

from chamnet.config.backbones import BACKBONES, NORM
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
    return dict(type='SegDataPreProcessor', mean=mean, std=std, bgr_to_rgb=True,
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


def _pipelines(r: Recipe, data_channels, shuffle):
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


def _dataloader(r: Recipe, split_dir, pipeline, root, batch_size, shuffle_sampler):
    return dict(
        batch_size=batch_size, num_workers=r.runtime.num_workers,
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
    # The two happen to be equal for every method this builder can construct
    # today (bl only; ef's early fusion feeds RGB+D straight into the same
    # backbone class once Task 8 adds it) but they diverge for sd/hd, whose
    # dual-branch backbone keeps a 3-channel RGB stem regardless of
    # data_channels, and again for the hd-rgb ablation. Keeping them as
    # separate names now means later tasks override stem_channels for those
    # backbones without touching how the pipeline/preprocessor are built.
    data_channels = 3 if method in ('bl',) else 4
    stem_channels = data_channels
    shuffle = ablation == 'shuffled'

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
    # The legacy top-level `pretrained` arg on EncoderDecoder is only set in the
    # source configs for backbones loaded via init_cfg; TIMMBackbone (convnext_atto)
    # loads its own weights and never carries this key — confirmed in its fixture.
    if spec['pretrained']:
        model['pretrained'] = None
    aux = _aux_head(spec, r)
    if aux is not None:
        model['auxiliary_head'] = aux

    optim = r.optim_for(backbone)
    paramwise = optim.pop('paramwise_cfg')
    extra = spec.get('paramwise_extra')
    if extra:
        paramwise = dict(custom_keys={**paramwise['custom_keys'], **extra})
    warm, poly = r.schedule.warmup, r.schedule.poly

    cfg = Config(dict(
        default_scope='mmseg',
        model=model,
        train_dataloader=_dataloader(r, r.data.splits['train'], train_pipe, root,
                                     r.runtime.batch_size, True),
        val_dataloader=_dataloader(r, r.data.splits['val'], test_pipe, root, 1, False),
        test_dataloader=_dataloader(r, r.data.splits['test'], test_pipe, root, 1, False),
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
