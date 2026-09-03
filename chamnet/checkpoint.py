"""Reading checkpoints that mmengine wrote, under torch >= 2.6."""
from contextlib import contextmanager

import torch


@contextmanager
def mmengine_checkpoint_loading():
    """Load one mmengine checkpoint, restoring `torch.load` afterwards.

    torch 2.6 made ``weights_only=True`` the default for ``torch.load``, and
    an mmengine checkpoint holds more than tensors: alongside the state dict
    it carries the message hub, whose scalar histories are pickled
    ``mmengine.logging.HistoryBuffer`` objects wrapping numpy arrays. The safe
    unpickler refuses those, so on torch >= 2.6 *every* checkpoint this
    package writes is unreadable by ``Runner.load_checkpoint``::

        _pickle.UnpicklingError: Weights only load failed. ...
        Unsupported global: GLOBAL mmengine.logging.history_buffer.HistoryBuffer

    Which is worse than it sounds, because it happens at *load* time: a run
    trains for hours, and only then cannot be scored. mmengine 0.10.7 calls
    ``torch.load`` with no ``weights_only`` argument at all
    (``mmengine/runner/checkpoint.py``, ``load_from_local``), so there is no
    setting to pass through — hence patching the function for the duration of
    the call and putting it back.

    **The narrower fix was tried and does not hold.** torch's own error
    message recommends allowlisting the offending class with
    ``torch.serialization.safe_globals``, which would keep the safe unpickler
    on for the rest of the file. Measured on torch 2.9 / numpy 2.1, that
    allowlist has to name, in turn, ``HistoryBuffer``,
    ``numpy._core.multiarray._reconstruct``, ``numpy.ndarray``,
    ``numpy.dtype`` and ``numpy.dtypes.Float64DType`` — a chain that runs
    through a module path which does not exist before numpy 2, and whose end
    is not known in advance for a checkpoint whose buffers happen to hold a
    different dtype. An allowlist that turns out to be one entry short fails
    exactly where this one is meant to help: after the training. So the
    scope, not the strictness, is what is traded away here.

    What that scope is worth keeping small: leaving ``weights_only=False`` on
    process-wide from import time — as an earlier revision of
    ``tools/replay.py`` did — silently applies full unpickling to every other
    ``.pth`` the process opens afterwards, including ones the caller did not
    get from here. Inside this block the file being read is one the caller has
    already chosen to load as a model, which is the same trust either way.
    """
    original = torch.load

    def load(*args, **kwargs):
        return original(*args, **{**kwargs, 'weights_only': False})

    torch.load = load
    try:
        yield
    finally:
        torch.load = original
