"""백본별 아키텍처 사실. 레시피가 아니라 '이 백본이 요구하는 것'.

값은 논문(160-run 실험 grid)이 실제로 만든 병합 config
(tests/fixtures/paper/bl_*.merged.py) 에서 확인된 그대로다.
fixture 와 이 표가 어긋나면 fixture 가 옳다 — 표를 고친다.
"""

NORM = dict(type='SyncBN', requires_grad=True)

BACKBONES = {
    'resnet18': dict(
        stem=dict(type='ResNetV1c', depth=18, num_stages=4, out_indices=(0, 1, 2, 3),
                  dilations=(1, 1, 1, 1), strides=(1, 2, 2, 2), norm_cfg=NORM,
                  norm_eval=False, style='pytorch', contract_dilation=True),
        pretrained='open-mmlab://resnet18_v1c',
        in_channels=[64, 128, 256, 512],
        decode_head=dict(type='UPerHead', channels=128, pool_scales=(1, 2, 3, 6)),
        aux=dict(type='FCNHead', in_channels=256, in_index=2, channels=64, num_convs=1),
        depth_mean_std=(2.263805360189078, 2.5188649864301693),
    ),
    'mit_b0': dict(
        stem=dict(type='MixVisionTransformer', in_channels=3, embed_dims=32, num_stages=4,
                  num_layers=[2, 2, 2, 2], num_heads=[1, 2, 5, 8],
                  patch_sizes=[7, 3, 3, 3], sr_ratios=[8, 4, 2, 1],
                  out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True,
                  drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1),
        pretrained=('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
                    'segformer/mit_b0_20220624-7e0fe6dd.pth'),
        in_channels=[32, 64, 160, 256],
        decode_head=dict(type='SegformerHead', channels=256),
        aux=None,
        # MixVisionTransformer's overlap-patch-embed positional convs need their
        # own weight-decay treatment — confirmed via custom_keys in the fixture's
        # optim_wrapper (pos_block: decay_mult=0.0), also present verbatim in the
        # source config's optimizer paramwise_cfg.
        paramwise_extra=dict(pos_block=dict(decay_mult=0.0)),
        depth_mean_std=(2.263805360189078, 2.5188649864301693),
    ),
    'segnext_t': dict(
        stem=dict(type='MSCAN', embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2],
                  mlp_ratios=[8, 8, 4, 4], drop_rate=0.0, drop_path_rate=0.1,
                  attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                  attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                  act_cfg=dict(type='GELU'),
                  norm_cfg=dict(type='BN', requires_grad=True)),
        pretrained=('https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
                    'segnext/mscan_t_20230227-119e8c9f.pth'),
        in_channels=[32, 64, 160, 256],
        # LightHamHead needs its own ham_channels/ham_kwargs and uses GroupNorm,
        # not the SyncBN every other decode head uses — confirmed in the fixture.
        decode_head=dict(type='LightHamHead', channels=256, ham_channels=256,
                         ham_kwargs=dict(MD_R=16, MD_S=1, train_steps=6,
                                          eval_steps=7, inv_t=100, rand_init=True),
                         norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        aux=None,
        paramwise_extra=dict(pos_block=dict(decay_mult=0.0)),
        depth_mean_std=(2.263805360189078, 2.5188649864301693),
    ),
    'convnext_atto': dict(
        stem=dict(type='TIMMBackbone', model_name='convnext_atto', in_channels=3,
                  features_only=True, pretrained=True, out_indices=(0, 1, 2, 3)),
        pretrained=None,                       # timm 이 직접 적재
        in_channels=[40, 80, 160, 320],
        decode_head=dict(type='UPerHead', channels=256, pool_scales=(1, 2, 3, 6)),
        aux=dict(type='FCNHead', in_channels=160, in_index=2, channels=128, num_convs=1),
        depth_mean_std=(2.263805360189078, 2.5188649864301693),
    ),
}

# SD (Dual, shallow depth-branch) backbone registry type per backbone. The
# dual backbone classes hardcode their RGB stem to 3 input channels inside
# __init__ and don't expose it as a config key at all — confirmed against
# tests/fixtures/paper/sd_*.merged.py, none of which carry an 'in_channels'
# key anywhere in the backbone dict.
SD_TYPE = {
    'resnet18': 'DualResNetV1c18',
    'mit_b0': 'DualMiTB0',
    'segnext_t': 'DualMSCAN',
    'convnext_atto': 'DualConvNeXtAttoSerial',
}

# HD (Dual+, heavy depth-branch) backbone registry type per backbone. Where SD
# runs a ~0.2-0.4M depthwise-separable DepthBranch on the depth channel, HD runs
# a second copy of the full RGB architecture on it.
HD_TYPE = {
    'resnet18': 'DualResNetV1c18LateFusion',
    'mit_b0': 'DualMiTB0LateFusion',
    'segnext_t': 'DualMSCANLateFusion',
    # Note: the ConvNeXt HD class lives in the same source module as the SD one
    # (dual_convnext_serial.py upstream, chamnet/models/backbones/convnext.py
    # here) — not in a '*_late' module like the other three.
    'convnext_atto': 'DualConvNeXtAttoPlusSerial',
}

# How each HD backbone dict differs from the BL dict for the same backbone,
# read straight off tests/fixtures/paper/hd_*.merged.py. `drop` keys are
# removed from the BL stem, `add` keys are merged in; the builder then sets
# `type`, `depth_pretrained` and `init_cfg` itself.
#
# Three of the four are BL + fusion_reduction. mit_b0 is not, because the
# paper's HD MiT config was hand-written from the upstream SegFormer reference
# rather than derived from this project's own BL config: it spells out two
# MixVisionTransformer constructor defaults (act_cfg, norm_cfg) and leaves
# three others implicit (in_channels, num_stages, patch_sizes) — and it never
# passes fusion_reduction at all, taking DualMiTB0LateFusion's own default of
# 4. Every dropped key equals the class default, so the built model is the
# same architecture either way; the difference is only in which keys are
# written down, and it is reproduced literally because build_config's output
# is compared against that config key for key. The "these really are the
# defaults" claim is not left as a comment — tests/test_matches_paper.py
# checks it against MixVisionTransformer's actual signature.
HD_STEM_DELTA = {
    'resnet18': dict(drop=(), add=dict(fusion_reduction=4)),
    'mit_b0': dict(drop=('in_channels', 'num_stages', 'patch_sizes'),
                   add=dict(norm_cfg=dict(type='LN', eps=1e-6),
                            act_cfg=dict(type='GELU'))),
    'segnext_t': dict(drop=(), add=dict(fusion_reduction=4)),
    # convnext_atto is not derived from its BL stem at all: like the SD class,
    # DualConvNeXtAttoPlusSerial calls timm.create_model itself and takes only
    # fusion_reduction / depth_pretrained / init_cfg. See _hd_stem.
}

# Optimizer keys the paper's HD config for a backbone spells out where its BL
# and SD configs for the same backbone leave them implicit. Only one exists:
# HD MiT-B0's optimizer states betas explicitly. That value is AdamW's own
# default, so it changes nothing about the optimizer that gets built — it is
# emitted only so the released builder reproduces that config exactly rather
# than nearly. tests/test_matches_paper.py asserts it against torch's actual
# AdamW default, so if torch ever changed that default this stops being a
# cosmetic difference and the suite says so instead of silently drifting.
HD_OPTIM_EXTRA = {
    'mit_b0': dict(betas=(0.9, 0.999)),
}
