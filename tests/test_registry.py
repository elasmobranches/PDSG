def test_register_all_is_idempotent():
    from mmseg.registry import MODELS
    import chamnet
    chamnet.register_all()
    chamnet.register_all()          # calling it twice must not raise
    assert 'ChamNet' in __import__('mmseg.registry', fromlist=['DATASETS']).DATASETS.module_dict
