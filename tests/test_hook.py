import chamnet
from mmseg.registry import HOOKS

chamnet.register_all()


def test_iter_logger_hook_is_registered_and_constructible():
    hook = HOOKS.build(dict(type='IterLoggerHook', flush_secs=10, out_csv=None))
    assert hook is not None
