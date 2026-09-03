from chamnet.config.schema import load_recipe


def test_paper_recipe_values_are_frozen():
    r = load_recipe('paper_v13')
    assert r.data.size == (512, 512)
    assert r.data.keep_ratio is False
    assert r.runtime.batch_size == 16
    # The paper's runs used a different worker count for training than for
    # evaluation, and that is load-bearing rather than a performance knob:
    # the shuffled control arms draw a permutation per sample inside the
    # worker processes, so the count decides the evaluated input. See
    # chamnet/config/builder.py::_dataloader.
    assert r.runtime.num_workers == 8
    assert r.runtime.num_workers_eval == 4
    assert r.optim.lr == 2.0e-4
    assert r.schedule.max_iters == 3760
    assert r.schedule.warmup['iters'] == 124
    assert r.loss.class_weight == [0.5, 3.0, 3.0, 1.0, 3.0, 0.5, 1.0, 1.0]
    assert r.runtime.seeds == list(range(31, 41))
    assert r.hd.depth_pretrained is True


def test_per_backbone_overrides_merge():
    r = load_recipe('paper_v13')
    base = r.optim_for('resnet18')
    assert base['weight_decay'] == 0.01
    assert base['paramwise_cfg']['custom_keys']['head']['lr_mult'] == 5.0
    assert 'betas' not in base

    seg = r.optim_for('segnext_t')
    assert seg['betas'] == (0.9, 0.999)
    assert seg['paramwise_cfg']['custom_keys']['head']['lr_mult'] == 10.0
    assert seg['weight_decay'] == 0.01          # 상속

    cvx = r.optim_for('convnext_atto')
    assert cvx['weight_decay'] == 0.05
    assert cvx['paramwise_cfg']['custom_keys']['bias']['decay_mult'] == 0.0
    assert cvx['paramwise_cfg']['custom_keys']['head']['lr_mult'] == 5.0   # 상속


def test_quick_recipe_is_shorter_but_same_shape():
    q = load_recipe('quick')
    assert q.schedule.max_iters == 200
    assert q.runtime.seeds == [31]
    assert q.loss.class_weight == load_recipe('paper_v13').loss.class_weight
