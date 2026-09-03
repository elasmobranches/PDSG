"""ChamNet — RGB-D fusion for greenhouse semantic segmentation."""
from contextlib import contextmanager

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
    from chamnet.models.backbones import resnet, mit, mscan, convnext  # noqa: F401
    from chamnet.models.backbones import early_fusion                  # noqa: F401
    # Registers ChamNetSegDataPreProcessor (see that module's docstring for
    # why it's a distinct name rather than an override of vanilla mmseg's
    # SegDataPreProcessor, which is buggy for >3-channel sd/hd/ef input).
    from chamnet.models import data_preprocessor                # noqa: F401
    _REGISTERED = True


@contextmanager
def scoped(cfg):
    """Activate ``cfg``'s mmseg default scope for the duration of the block.

    ``Runner.from_cfg`` reads ``cfg.default_scope`` itself, but any code that
    builds a model/dataset directly via ``MODELS.build``/``DATASETS.build``
    (tests, notebooks, ad-hoc scripts) needs an active scope first — without
    one, mmengine resolves nested registry lookups (e.g. a model's
    ``data_preprocessor``, or a pipeline step registered only on mmseg's
    child TRANSFORMS registry) against its own root registry instead of
    mmseg's, and they fail to resolve. Wraps
    ``DefaultScope.overwrite_default_scope`` so call sites don't repeat that
    incantation::

        with chamnet.scoped(cfg):
            model = MODELS.build(cfg.model)
    """
    from mmengine.registry import DefaultScope
    with DefaultScope.overwrite_default_scope(cfg.default_scope):
        yield
