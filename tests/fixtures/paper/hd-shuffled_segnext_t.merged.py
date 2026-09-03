_depth_cfg_train = dict(
    depth_dir_name='depth',
    depth_suffix='.npy',
    img_dir_name='images',
    max_depth=None,
    type='LoadDepthAsChannel')
_depth_cfg_valtest = dict(
    depth_dir_name='depth',
    depth_suffix='_depth.npy',
    img_dir_name='images',
    max_depth=None,
    type='LoadDepthAsChannel')
base_scale = (
    896,
    512,
)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'
crop_size = (
    384,
    672,
)
custom_hooks = [
    dict(
        check_finite=True,
        min_delta=0.01,
        monitor='mIoU',
        patience=20,
        rule='greater',
        type='EarlyStoppingHook'),
    dict(flush_secs=10, out_csv=None, type='IterLoggerHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'hooks',
        'mmseg.datasets.chamoe',
        'mmseg.models.backbones.dual_mscan_late',
        'mmseg.datasets.transforms.load_depth',
        'mmseg.datasets.transforms.nyu_augmentation',
        'mmseg.datasets.transforms.online_augmentation',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        2.263805360189078,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
        2.5188649864301693,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = 'dataset_v2'
dataset_type = 'ChamNet'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=-1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='mIoU',
        save_last=False,
        type='CheckpointHook'),
    logger=dict(interval=20, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
ham_norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attention_kernel_paddings=[
            2,
            [
                0,
                3,
            ],
            [
                0,
                5,
            ],
            [
                0,
                10,
            ],
        ],
        attention_kernel_sizes=[
            5,
            [
                1,
                7,
            ],
            [
                1,
                11,
            ],
            [
                1,
                21,
            ],
        ],
        depth_pretrained=True,
        depths=[
            3,
            3,
            5,
            2,
        ],
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=[
            32,
            64,
            160,
            256,
        ],
        fusion_reduction=4,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth',
            type='Pretrained'),
        mlp_ratios=[
            8,
            8,
            4,
            4,
        ],
        norm_cfg=dict(requires_grad=True, type='BN'),
        type='DualMSCANLateFusion'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            2.263805360189078,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
            2.5188649864301693,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        ham_channels=256,
        ham_kwargs=dict(
            MD_R=16,
            MD_S=1,
            eval_steps=7,
            inv_t=100,
            rand_init=True,
            train_steps=6),
        in_channels=[
            32,
            64,
            160,
            256,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                class_weight=[
                    0.5,
                    3.0,
                    3.0,
                    1.0,
                    3.0,
                    0.5,
                    1.0,
                    1.0,
                ],
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                eps=0.001, loss_weight=1.0, type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=8,
        type='LightHamHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    accumulative_counts=1,
    clip_grad=dict(max_norm=5.0, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0002, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.0004, type='AdamW', weight_decay=0.01)
pad_mean = [
    103.53,
    116.28,
    123.675,
    2.263805360189078,
]
param_scheduler = [
    dict(begin=0, by_epoch=False, end=124, start_factor=0.01, type='LinearLR'),
    dict(
        begin=124,
        by_epoch=False,
        end=3760,
        eta_min_ratio=0.01,
        power=0.9,
        type='PolyLRRatio'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=37)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='test/images', seg_map_path='test/masks_gray'),
        data_root='real_dataset_v12_plus',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                depth_dir_name='depth',
                depth_suffix='_depth.npy',
                img_dir_name='images',
                max_depth=None,
                type='LoadDepthAsChannel'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='ShuffleDepthChannel'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='_mask_gray.png',
        type='ChamNet'),
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        depth_dir_name='depth',
        depth_suffix='_depth.npy',
        img_dir_name='images',
        max_depth=None,
        type='LoadDepthAsChannel'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='ShuffleDepthChannel'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=3760, type='IterBasedTrainLoop', val_interval=20)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root='real_dataset_v12_plus',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                depth_dir_name='depth',
                depth_suffix='.npy',
                img_dir_name='images',
                max_depth=None,
                type='LoadDepthAsChannel'),
            dict(type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                photometric='brightness_contrast',
                rotate_prob=0.0,
                type='ChamNetOnlineAugmentation'),
            dict(type='ShuffleDepthChannel'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='ChamNet'),
    num_workers=8,
    persistent_workers=False,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        depth_dir_name='depth',
        depth_suffix='.npy',
        img_dir_name='images',
        max_depth=None,
        type='LoadDepthAsChannel'),
    dict(type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        photometric='brightness_contrast',
        rotate_prob=0.0,
        type='ChamNetOnlineAugmentation'),
    dict(type='ShuffleDepthChannel'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1024,
                    512,
                ), type='Resize'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='valid/images', seg_map_path='valid/masks_gray'),
        data_root='real_dataset_v12_plus',
        img_suffix='.jpg',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                depth_dir_name='depth',
                depth_suffix='_depth.npy',
                img_dir_name='images',
                max_depth=None,
                type='LoadDepthAsChannel'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='ShuffleDepthChannel'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='_mask_gray.png',
        type='ChamNet'),
    num_workers=4,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
val_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        depth_dir_name='depth',
        depth_suffix='_depth.npy',
        img_dir_name='images',
        max_depth=None,
        type='LoadDepthAsChannel'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'work_dirs_v12+_512_controls_ext/37/chamnet_dual_plus_shuffled_segnext_t'
