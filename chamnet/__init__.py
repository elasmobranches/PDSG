"""ChamNet — RGB-D fusion for greenhouse semantic segmentation."""
__version__ = '1.0.0'

_REGISTERED = False


def register_all() -> None:
    """Import every module that registers a type into the mmseg registries.

    Idempotent: mmengine raises on duplicate registration, so guard the imports.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from chamnet.datasets import dataset                       # noqa: F401
    from chamnet.datasets.transforms import load_depth         # noqa: F401
    from chamnet.datasets.transforms import augmentation       # noqa: F401
    from chamnet.hooks import iter_logger                      # noqa: F401
    _REGISTERED = True
