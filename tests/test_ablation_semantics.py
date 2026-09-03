"""Do the controlled variables in a config actually control anything?

A config key is a claim about what the model will do. These tests check the
claims that would otherwise be verified by reading a log line — which is
exactly how this project once shipped a control arm whose depth encoder was
hardcoded to random initialisation while its log said ``pretrained``. Every
assertion here observes model weights, never a flag read back off a config
and never a message.

Network: the checks below load the same ImageNet checkpoints training
would, so they need network access (or a warm ``~/.cache/torch/hub``) the
first time they run. That is deliberate — the only way to see whether
pretrained weights landed is to have the pretrained weights. These tests
fail rather than skip when the download fails, because a skip is how a check
like this quietly stops checking.

Two arms are covered here. HD's depth encoder (does ``depth_pretrained``
load anything, and is the transfer rule an average rather than a sum?) and
EF's widened stem (is the 4th input channel the RGB filter mean the paper's
configs ask for, rather than the zeros every EF class defaults to?). Both
are settings whose effect is invisible in the config, invisible in the
parameter count, and invisible in any log line.
"""
import copy

import pytest
import torch
from mmseg.registry import MODELS

import chamnet
from chamnet.config.builder import build_config

chamnet.register_all()

# Every test in this module builds a backbone that loads the same ImageNet
# checkpoints training would, so all of them need network access (or a warm
# ~/.cache/torch/hub) -- see the `network` marker's note in pyproject.toml
# for why they are not skipped when that is unavailable.
pytestmark = pytest.mark.network

BACKBONES = ['resnet18', 'mit_b0', 'segnext_t', 'convnext_atto']

# Where each HD backbone's depth encoder gets its pretrained weights, which
# decides how "did they actually load?" has to be observed:
#
#   'init_weights' — DualResNetV1c18LateFusion / DualMiTB0LateFusion /
#       DualMSCANLateFusion build a randomly-initialised depth encoder in
#       __init__ and fill it in later, from the RGB checkpoint, inside
#       init_weights(). So the load is visible as parameters changing across
#       that call.
#   'constructor' — DualConvNeXtAttoPlusSerial hands depth_pretrained straight
#       to timm.create_model, so the weights are already in place when __init__
#       returns and init_weights() changes nothing. Nothing to diff across;
#       the depth encoder is instead compared against an independently
#       constructed timm model, which is where those weights should have come
#       from.
LOADS_AT = {
    'resnet18': 'init_weights',
    'mit_b0': 'init_weights',
    'segnext_t': 'init_weights',
    'convnext_atto': 'constructor',
}


# The checkpoint key whose 3-channel filter each HD depth encoder adapts down
# to 1 channel. mit_b0 and segnext_t declare it on the class as
# FIRST_CONV_KEY and hand it to load_rgb_into_depth_encoder;
# DualResNetV1c18LateFusion predates that helper and hardcodes 'stem.0.weight'
# inside its own _load_pretrained_depth, so its key is written out here. The
# test below cross-checks this table against the class attribute wherever one
# exists, so the two cannot drift apart unnoticed.
HD_FIRST_CONV_KEY = {
    'resnet18': 'stem.0.weight',
    'mit_b0': 'layers.0.0.projection.weight',
    'segnext_t': 'patch_embed1.proj.0.weight',
}


def _timm_convnext_depth_encoder(pretrained):
    """An independent copy of the depth encoder DualConvNeXtAttoPlusSerial builds."""
    import timm
    return timm.create_model('convnext_atto', features_only=True,
                             pretrained=pretrained, in_chans=1,
                             out_indices=(0, 1, 2, 3))


def _hd_backbone(backbone, depth_pretrained=None, ablation=None):
    """Build just the HD backbone `build_config` emits, optionally overriding
    depth_pretrained so the same code path can be run as its own control."""
    cfg = build_config(method='hd', backbone=backbone, ablation=ablation,
                       recipe='paper', seed=37)
    stem = copy.deepcopy(cfg.model['backbone'])
    if depth_pretrained is not None:
        stem['depth_pretrained'] = depth_pretrained
    with chamnet.scoped(cfg):
        return MODELS.build(stem)


def _count_changed_by_init_weights(model):
    before = [p.detach().clone() for p in model.depth_backbone.parameters()]
    model.init_weights()
    after = list(model.depth_backbone.parameters())
    assert len(before) == len(after)
    return sum(1 for a, b in zip(after, before) if not torch.equal(a, b)), len(after)


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_depth_encoder_actually_loads_pretrained(backbone):
    """`depth_pretrained=True` must move real weights into the depth encoder.

    The paper's HD arm sets this on all four backbones (recipe `hd:
    depth_pretrained`, and all four hd_*.merged.py fixtures), and it is not a
    cosmetic setting: HD's depth encoder is a full second copy of the
    backbone, so pretrained-versus-random there is a bigger difference than
    the fusion design the paper is actually comparing. This project has
    already shipped one arm that claimed a pretrained init it never
    performed, and separately found HD's depth encoder to be pretrained on
    ResNet only when it was meant to be uniform across backbones — so the
    check is on the weights themselves, not on the flag or the log line.
    """
    assert build_config(method='hd', backbone=backbone, recipe='paper',
                        seed=37).model['backbone']['depth_pretrained'] is True, (
        'precondition: the recipe asks for a pretrained HD depth encoder')
    model = _hd_backbone(backbone)

    if LOADS_AT[backbone] == 'constructor':
        got = dict(model.depth_backbone.named_parameters())
        reference = dict(_timm_convnext_depth_encoder(True).named_parameters())
        assert set(got) == set(reference)
        differing = [k for k, v in reference.items() if not torch.equal(got[k], v)]
        assert not differing, (
            f'depth encoder disagrees with timm pretrained weights at {differing[:5]}')
        # ...and that agreement means something only because a random init
        # would not produce it.
        random_stem = _timm_convnext_depth_encoder(False).stem_0.weight
        assert not torch.equal(model.depth_backbone.stem_0.weight, random_stem)
        return

    changed, total = _count_changed_by_init_weights(model)
    assert changed == total, (
        f'{changed}/{total} depth-encoder parameters changed during '
        'init_weights(); a pretrained load should replace all of them')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_depth_encoder_stays_random_when_not_asked(backbone):
    """The other half: `depth_pretrained=False` must leave it random.

    Without this, a backbone that loaded the RGB checkpoint into its depth
    encoder unconditionally would pass the test above while making
    `depth_pretrained` a decoration — and the ablation it exists to support
    (random vs. ImageNet depth encoder, which on greenhouse data moves Pillar
    IoU by about 3 points) would silently be comparing a thing to itself.
    """
    model = _hd_backbone(backbone, depth_pretrained=False)

    if LOADS_AT[backbone] == 'constructor':
        reference = _timm_convnext_depth_encoder(True)
        assert not torch.equal(model.depth_backbone.stem_0.weight,
                               reference.stem_0.weight)
        return

    changed, total = _count_changed_by_init_weights(model)
    assert changed == 0, (
        f'{changed}/{total} depth-encoder parameters changed during '
        'init_weights() even though depth_pretrained=False')


@pytest.mark.parametrize('backbone', sorted(HD_FIRST_CONV_KEY))
def test_hd_depth_first_conv_is_the_channel_averaged_rgb_filter(backbone):
    """Check the transfer rule itself, not just that something was copied.

    The rule is: copy every layer verbatim, and adapt only the first
    convolution, *averaging* its three input channels into one so a
    single-channel input keeps the activation scale of the pretrained RGB
    filters. Averaging rather than summing is the entire content of that
    rule, and nothing else in this suite can see it. A sum still replaces
    every parameter, so `test_hd_depth_encoder_actually_loads_pretrained`
    still passes; it still moves the stem's norm, so
    `_load_pretrained_depth`'s own `norm_before == norm_after` guard still
    passes; it changes no config, so `test_hd_matches_paper` and the smoke
    test are untouched. The depth stem would simply start at 3x the intended
    activation scale with a fully green suite — which is the one thing that
    matters most on resnet18, the backbone whose pretrained-vs-random depth
    result this project actually quotes (-3.14 Pillar, see
    chamnet/models/depth_pretrain.py).

    So recompute the expected filter straight from the checkpoint and
    compare. All three checkpoint-loading backbones are covered: mit_b0 and
    segnext_t through the shared `load_rgb_into_depth_encoder`, resnet18
    through its own older `_load_pretrained_depth`, which implements the same
    rule separately and therefore needs its own check rather than inheriting
    one.

    convnext_atto is excluded on purpose: timm does its own adaptation there
    and *sums* rather than averages. Its stem is followed immediately by a
    LayerNorm, which normalises the constant scale difference away, so the
    two conventions agree in effect — see the note in
    DualConvNeXtAttoPlusSerial.
    """
    from mmengine.runner.checkpoint import _load_checkpoint

    key = HD_FIRST_CONV_KEY[backbone]
    model = _hd_backbone(backbone)
    declared = getattr(model, 'FIRST_CONV_KEY', None)
    if declared is not None:
        assert declared == key, (
            f'{backbone} declares FIRST_CONV_KEY={declared!r} but this test '
            f'expects {key!r}')
    model.init_weights()

    checkpoint = _load_checkpoint(model.init_cfg['checkpoint'], map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    rgb_filter = state_dict[key]
    assert rgb_filter.shape[1] == 3, 'the RGB checkpoint should have a 3ch first conv'

    depth_conv = model.depth_backbone
    for part in key[:-len('.weight')].split('.'):
        depth_conv = depth_conv[int(part)] if part.isdigit() else getattr(depth_conv, part)
    assert depth_conv.in_channels == 1
    expected = rgb_filter.mean(dim=1, keepdim=True)
    assert torch.equal(depth_conv.weight, expected), (
        f'{backbone} depth first conv is not the channel-averaged RGB filter; '
        f'ratio to expected ~'
        f'{(depth_conv.weight.norm() / expected.norm()).item():.3f}')


# ---------------------------------------------------------------------------
# EF (early fusion): one widened stem convolution, and what its 4th input
# channel starts as.
# ---------------------------------------------------------------------------

# Where each EF backbone's widened stem convolution lives on the built module.
# resnet18 and segnext_t declare the same path (plus '.weight') on the class as
# FIRST_CONV_KEY and use it to expand the checkpoint tensor;
# MixVisionTransformer4Ch hardcodes its key inline instead of declaring one,
# and TIMMBackbone4Ch reaches it through its `stem_conv_attr` argument on
# self.timm_model. The helper below cross-checks this table against the class
# attribute wherever one exists, and against the built module in every case, so
# the two cannot drift apart unnoticed.
EF_STEM_CONV_PATH = {
    'resnet18': 'stem.0',
    'mit_b0': 'layers.0.0.projection',
    'segnext_t': 'patch_embed1.proj.0',
    'convnext_atto': 'timm_model.stem_0',
}

# How each EF backbone's pretrained RGB stem filter is obtained independently
# of the model under test, to check what actually landed in it. Three load an
# mmseg checkpoint through init_cfg; ConvNeXt's weights come from timm inside
# __init__ and have no init_cfg, so an independently constructed timm model is
# the reference instead (the same shape of argument as LOADS_AT above).
EF_LOADS_AT = {
    'resnet18': 'init_cfg',
    'mit_b0': 'init_cfg',
    'segnext_t': 'init_cfg',
    'convnext_atto': 'timm',
}


def _get_by_path(module, dotted):
    obj = module
    for part in dotted.split('.'):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _backbone(method, backbone):
    """Build just the backbone `build_config` emits for (method, backbone)."""
    cfg = build_config(method=method, backbone=backbone, recipe='paper', seed=37)
    with chamnet.scoped(cfg):
        return MODELS.build(copy.deepcopy(cfg.model['backbone']))


def _ef_stem_conv(model, backbone):
    """Resolve the widened stem conv, refusing to be pointed at the wrong one.

    A table of module paths is exactly the kind of thing that silently rots:
    point it at some other convolution and every assertion below still runs,
    just against a layer nobody claimed anything about. Three cross-checks
    make that impossible — the path must resolve to a Conv2d, that conv must
    be the *only* 4-input-channel module anywhere in the backbone (which is
    what "early fusion widens exactly one convolution" means), and where the
    class declares FIRST_CONV_KEY it must name this same path.
    """
    conv = _get_by_path(model, EF_STEM_CONV_PATH[backbone])
    assert isinstance(conv, torch.nn.Conv2d), (
        f'{backbone}: {EF_STEM_CONV_PATH[backbone]!r} resolved to '
        f'{type(conv).__name__}, not a Conv2d')
    four_channel = [name for name, m in model.named_modules()
                    if getattr(m, 'in_channels', None) == 4]
    assert four_channel == [EF_STEM_CONV_PATH[backbone]], (
        f'{backbone}: expected exactly one 4-input-channel module, the stem at '
        f'{EF_STEM_CONV_PATH[backbone]!r}; found {four_channel}')
    declared = getattr(type(model), 'FIRST_CONV_KEY', None)
    if declared is not None:
        assert declared == EF_STEM_CONV_PATH[backbone] + '.weight', (
            f'{backbone} declares FIRST_CONV_KEY={declared!r}, which is not the '
            f'stem this test inspects ({EF_STEM_CONV_PATH[backbone]!r})')
    return conv


def _pretrained_rgb_stem_filter(model, backbone):
    """The 3-channel stem filter EF is supposed to have started from."""
    if EF_LOADS_AT[backbone] == 'timm':
        import timm
        reference = timm.create_model('convnext_atto', features_only=True,
                                      pretrained=True, in_chans=3,
                                      out_indices=(0, 1, 2, 3))
        return reference.stem_0.weight.detach(), reference.stem_0.bias.detach()

    from mmengine.runner.checkpoint import _load_checkpoint

    checkpoint = _load_checkpoint(model.init_cfg['checkpoint'], map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    return state_dict[EF_STEM_CONV_PATH[backbone] + '.weight'], None


@pytest.mark.parametrize('backbone', BACKBONES)
def test_ef_adds_only_the_stems_extra_input_channel(backbone):
    """EF's result cannot be explained by capacity — it is one input plane.

    The whole argument for the early-fusion arm is that it is the *cheapest*
    way to use depth: the architecture is the RGB baseline's, unchanged, with
    the first convolution widened from 3 input channels to 4. So the parameter
    difference against BL must be exactly that one plane of stem filter
    weights, `out_channels x kernel_h x kernel_w` — and nothing else.

    That number is not a constant across backbones (288 on ResNet-18's 3x3
    deep stem, 144 on SegNeXt-T's, 640 on ConvNeXt-Atto's 4x4 patch stem,
    1568 on MiT-B0's 7x7 patch embed), so a bound like "under a thousand"
    would both admit MiT-B0's real value as a failure and let a genuinely
    wrong stem pass on the other three. Compute the expected delta from the
    backbone's own stem conv shape instead: it is exact, it self-documents,
    and it is the claim the test is named for.
    """
    bl = _backbone('bl', backbone)
    ef = _backbone('ef', backbone)
    conv = _ef_stem_conv(ef, backbone)

    out_channels, in_channels, kh, kw = conv.weight.shape
    assert in_channels == 4
    expected = out_channels * kh * kw

    n_bl = sum(p.numel() for p in bl.parameters())
    n_ef = sum(p.numel() for p in ef.parameters())
    assert n_ef - n_bl == expected, (
        f'{backbone}: EF has {n_ef - n_bl} parameters more than BL, but '
        f'widening its stem conv by one input channel accounts for exactly '
        f'{expected} ({out_channels}x{kh}x{kw}). Something other than the '
        'stem changed.')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_ef_stem_takes_four_channels_initialised_from_the_rgb_mean(backbone):
    """The depth channel's filter must be the RGB filter mean, not zeros.

    All four EF classes default `extra_channel_init` to `'zero'`; all four of
    the paper's EF configs pass `'mean'` (see EF_EXTRA_CHANNEL_INIT). If
    `build_config` failed to emit it, the depth channel's stem weights would
    initialise to zeros — the depth input contributing literally nothing at
    step 0 — with no error, no warning, no config difference visible anywhere
    else, and an identical parameter count, so
    `test_ef_adds_only_the_stems_extra_input_channel` above cannot see it.
    This is the same family as HD's `depth_pretrained` trap, which this
    project has already shipped twice.

    Observing the flag would prove nothing, so this observes the weights the
    model would actually start training from, after `init_weights()` — and it
    checks the transformation, not merely that something moved: channel 3 must
    equal the *mean* of channels 0-2 (a sum, or a copy of one channel, is a
    different rule that still leaves it non-zero), and channels 0-2 must be
    the pretrained RGB filter itself, obtained independently of the model
    under test. That last part matters most for ConvNeXt, where the whole
    reason TIMMBackbone4Ch widens the stem by hand instead of asking timm for
    `in_chans=4` is that timm's `adapt_input_conv` would rescale the RGB
    filters to 0.75x — which this comparison would catch and a
    zero-versus-nonzero check would not.
    """
    model = _backbone('ef', backbone)
    model.init_weights()
    conv = _ef_stem_conv(model, backbone)
    weight = conv.weight.detach()
    assert weight.shape[1] == 4

    rgb_filter, rgb_bias = _pretrained_rgb_stem_filter(model, backbone)
    assert rgb_filter.shape[1] == 3, 'the RGB reference should be 3-channel'
    assert torch.equal(weight[:, :3], rgb_filter), (
        f'{backbone}: EF stem channels 0-2 are not the pretrained RGB filter '
        f'(ratio to expected ~'
        f'{(weight[:, :3].norm() / rgb_filter.norm()).item():.4f})')
    expected_depth = rgb_filter.mean(dim=1, keepdim=True)
    assert torch.equal(weight[:, 3:], expected_depth), (
        f'{backbone}: EF stem channel 3 is not the channel-averaged RGB '
        f'filter; ratio to expected ~'
        f'{(weight[:, 3:].norm() / expected_depth.norm()).item():.4f}')
    # ...and the whole point is that 'mean' is not 'zero', the class default a
    # config that forgot to say so would silently get.
    assert weight[:, 3:].abs().sum() > 0, (
        f'{backbone}: EF depth channel initialised to zeros — build_config '
        "did not pass extra_channel_init='mean'")
    if rgb_bias is not None:
        assert torch.equal(conv.bias.detach(), rgb_bias), (
            f'{backbone}: widening the stem lost its pretrained bias')


# ---------------------------------------------------------------------------
# HD's control arms: what each one actually changes about the built model.
#
# `tests/test_matches_paper.py` already checks that each arm's config is the
# paper's, key for key. That is a check on the *dict*, and every mechanism below
# is one a wrong dict cannot produce but a right dict can still fail to deliver
# — a backbone class that accepts `fusion_use_gate` and quietly ignores it, a
# depth-slot encoder that takes RGB but starts from the wrong weights. So these
# look at the built modules and their parameters.
# ---------------------------------------------------------------------------


def _fusion_parameters(model):
    """The parameters `build_config`'s fusion modules contribute, by name.

    Selected the way the arms differ — anything under a `fusion`/`gate` name —
    rather than by reaching into `.fusions`, so a variant that put its gate
    somewhere else would still be counted. Measured on all four backbones: the
    only matching root is `fusions`, i.e. nothing unrelated is swept in.
    """
    return {n: p for n, p in model.named_parameters()
            if 'fusion' in n or 'gate' in n}


@pytest.mark.parametrize('backbone', BACKBONES)
def test_gate_designs_differ_by_exactly_one_channel_gate(backbone):
    """NOGATE, CMG and BIGATE must be an arithmetic progression in fusion size.

    The three arms are meant to be the same fusion block with zero, one and two
    channel gates on it: NOGATE injects `rgb + d_proj`, CMG weights the depth
    term by a gate on depth, BIGATE adds a second gate on RGB. If that is what
    they are, then `CMG - NOGATE` and `BIGATE - CMG` are both exactly one gate
    MLP and therefore equal — a property, not a definition, so it was measured
    before being asserted. It holds on all four backbones:

        backbone       NOGATE     CMG     BIGATE    one gate
        resnet18       350,080  611,200   872,320    261,120
        mit_b0          97,280  169,472   241,664     72,192
        segnext_t       97,280  169,472   241,664     72,192
        convnext_atto  137,200  239,200   341,200    102,000

    Equality alone would not prove the difference is a *gate*, so the expected
    size is also recomputed from each stage's own width: a gate is
    `Linear(2C -> mid) + Linear(mid -> C)`, bias-free, with `mid = max(C//4,
    8)` at the fusion_reduction=4 every paper config uses — i.e. `3*C*mid` per
    stage. Both halves of the progression must equal that sum.

    What this catches that nothing else does: `fusion_use_gate` reaching a
    backbone class that accepts the argument and never passes it on. Two of the
    four HD classes (MSCAN, ConvNeXt) genuinely lacked that plumbing until this
    release; with the config key emitted and ignored, NOGATE would build a full
    CMG, `test_hd_nogate_matches_paper` would still pass, and the arm would be
    comparing HD against itself.
    """
    counts, dims = {}, None
    for ablation in ('nogate', None, 'bigate'):
        model = _hd_backbone(backbone, ablation=ablation)
        counts[ablation] = sum(p.numel() for p in _fusion_parameters(model).values())
        if ablation is None:
            # Stage widths read off the module every arm shares — the depth
            # projection — rather than restated from a table here.
            dims = [f.depth_proj[0].in_channels for f in model.fusions]

    one_gate = sum(2 * c * max(c // 4, 8) + max(c // 4, 8) * c for c in dims)
    no, cmg, bi = counts['nogate'], counts[None], counts['bigate']
    assert cmg - no == bi - cmg > 0, (
        f'{backbone}: fusion parameters {no}/{cmg}/{bi} (nogate/cmg/bigate) are '
        f'not an arithmetic progression: {cmg - no} != {bi - cmg}')
    assert cmg - no == one_gate, (
        f'{backbone}: the step between gate designs is {cmg - no} parameters, '
        f'but one channel-gate MLP over stage widths {dims} is {one_gate}. '
        'The arms differ by something other than a gate.')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_nogate_builds_no_gate_and_injects_depth_unweighted(backbone):
    """`fusion_use_gate=False` must remove the gate, not set it to 1.

    Observed on the built module: every fusion reports `use_gate=False` and has
    no gate submodule at all, and its output is exactly `rgb + depth_proj(d)`
    — recomputed here from the module's own projection, so a gate that had been
    pinned to some constant would show up as a mismatch rather than passing as
    "close enough".
    """
    model = _hd_backbone(backbone, ablation='nogate')
    for i, fusion in enumerate(model.fusions):
        assert fusion.use_gate is False, i
        assert not hasattr(fusion, 'gate'), f'stage {i} still built a gate MLP'

    fusion = model.fusions[0].eval()
    channels = fusion.depth_proj[0].in_channels
    rgb = torch.randn(2, channels, 8, 8)
    depth = torch.randn(2, channels, 8, 8)
    with torch.no_grad():
        assert torch.equal(fusion(rgb, depth), rgb + fusion.depth_proj(depth))


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_bigate_gates_the_rgb_stream_as_well_as_the_depth_stream(backbone):
    """BIGATE's whole content is that the RGB term is gated too.

    CMG is `rgb + d_proj*g(depth)`: the RGB path is untouched, which is the
    asymmetry the robustness argument rests on. BIGATE is
    `rgb*g(rgb) + d_proj*g(depth)`. Swapping the class without that second gate
    being wired into the forward pass would leave a module that has the extra
    parameters (so the arithmetic test above still passes) and never uses them.
    So this checks the gradient: after a backward through the fusion, the RGB
    gate must have received one, and plain HD must have no such module at all.
    """
    from chamnet.models.fusion import BiGateGating

    model = _hd_backbone(backbone, ablation='bigate')
    assert all(isinstance(f, BiGateGating) for f in model.fusions)
    assert not any('gate_rgb' in n for n in dict(
        _hd_backbone(backbone).named_parameters())), (
        'precondition: plain HD has no RGB-side gate to compare against')

    fusion = model.fusions[0]
    channels = fusion.depth_proj[0].in_channels
    fusion(torch.randn(2, channels, 8, 8),
           torch.randn(2, channels, 8, 8)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in fusion.gate_rgb.parameters()), (
        f'{backbone}: BiGate built an RGB gate that the forward pass ignores')


@pytest.mark.parametrize('backbone', BACKBONES)
def test_hd_rgb_depth_slot_takes_three_channels_from_the_unaveraged_filter(backbone):
    """The capacity control's depth-slot encoder is an RGB encoder.

    Two things have to be true, and neither is visible in the config. The
    encoder in the depth slot must take three input channels — it is fed the
    RGB image — and, because the arm controls for HD's initialisation as well
    as its capacity (`depth_pretrained` stays True), it must start from the
    pretrained stem *unmodified*. That is the opposite of HD's own rule, which
    averages the three RGB filters down to one for a 1-channel depth map; a
    3-channel encoder needs no adaptation, and averaging it would both change
    the weights and be the wrong shape.

    This is the check that makes the production `load_rgb_into_depth_encoder`
    behaviour load-bearing: the earlier version refused any depth encoder whose
    first conv was not 1-channel, so it could not initialise this arm at all.
    """
    model = _hd_backbone(backbone, ablation='rgb')

    if LOADS_AT[backbone] == 'constructor':
        depth_stem = model.depth_backbone.stem_0
        assert depth_stem.in_channels == 3
        reference = _timm_convnext_depth_encoder(True)  # 1ch, for contrast
        assert reference.stem_0.weight.shape[1] == 1
        import timm
        rgb_reference = timm.create_model('convnext_atto', features_only=True,
                                          pretrained=True, in_chans=3,
                                          out_indices=(0, 1, 2, 3))
        assert torch.equal(depth_stem.weight, rgb_reference.stem_0.weight), (
            'the depth-slot encoder is not timm\'s pretrained 3-channel stem')
        return

    from mmengine.runner.checkpoint import _load_checkpoint

    key = HD_FIRST_CONV_KEY[backbone]
    model.init_weights()
    checkpoint = _load_checkpoint(model.init_cfg['checkpoint'], map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    rgb_filter = state_dict[key]

    depth_conv = _get_by_path(model.depth_backbone, key[:-len('.weight')])
    assert depth_conv.in_channels == 3, (
        f'{backbone}: the depth-slot encoder still takes '
        f'{depth_conv.in_channels} channel(s); it is meant to take the RGB image')
    assert torch.equal(depth_conv.weight, rgb_filter), (
        f'{backbone}: the depth-slot stem is not the pretrained RGB filter '
        f'copied straight across (ratio to expected ~'
        f'{(depth_conv.weight.norm() / rgb_filter.norm()).item():.4f}); a '
        'channel-averaged copy would be HD\'s 1-channel rule applied to the '
        'wrong arm')
