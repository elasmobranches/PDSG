def test_register_all_is_idempotent():
    from mmseg.registry import MODELS
    import chamnet
    chamnet.register_all()
    chamnet.register_all()          # 두 번 불러도 예외가 없어야 한다
    assert 'ChamNet' in __import__('mmseg.registry', fromlist=['DATASETS']).DATASETS.module_dict
