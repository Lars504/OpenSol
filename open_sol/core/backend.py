"""Run backends (solver_plan S6): 'gpu' = cupy + cupyx.scipy.sparse, 'cpu' = numpy + scipy.sparse. The only module
importing cupy at module level: `cp` / `cupyx_sparse` / `cupyx_ndimage` are cupy's modules, or stand-ins when cupy
does not import (kernel objects build, every other use raises BackendUnavailable). No driver call at import."""
import functools
from dataclasses import dataclass

import numpy as np

BACKEND_NAMES = ('gpu', 'cpu')
_MISSING_TEXT = ("needs the GPU backend: install cupy or choose the CPU backend "
                 "(Solvers node backend 'cpu', api / CLI backend='cpu')")
# cupy constructors that run at module level (kernel definitions): without cupy they return a stand-in kernel
_KERNEL_FACTORIES = frozenset(('RawKernel', 'RawModule', 'ElementwiseKernel', 'ReductionKernel'))


class BackendUnavailable(RuntimeError):
    """A cupy call without an importable cupy (GPU work on a machine set up for the CPU backend)."""


class _MissingGPU:
    """Stand-in of a cupy attribute when cupy does not import: attribute chains stay stand-ins, isinstance is False,
    calls raise BackendUnavailable (kernel constructors return a stand-in kernel that raises when launched)."""
    __slots__ = ('_name',)

    def __init__(self, name):
        object.__setattr__(self, '_name', name)

    def __getattr__(self, attribute):
        if attribute.startswith('__'):
            raise AttributeError(attribute)
        return _MissingGPU(f'{self._name}.{attribute}')

    def __call__(self, *args, **kwargs):
        if self._name.rsplit('.', 1)[-1] in _KERNEL_FACTORIES:
            label = next((arg for arg in reversed(args) if isinstance(arg, str) and arg.isidentifier()), '')
            return _MissingGPU(f"{self._name} '{label}'" if label else self._name)
        raise BackendUnavailable(f'{self._name} {_MISSING_TEXT}{_import_error_text()}')

    def __instancecheck__(self, instance):
        return False

    def __subclasscheck__(self, subclass):
        return False

    def __or__(self, other):      # annotations such as `cp.ndarray | None`
        return self

    __ror__ = __or__

    def __repr__(self):
        return f'<{self._name}: cupy not available>'


def _numpy_dtype_stand_in(name):
    """cupy re-exports numpy's scalar types (cp.float32 is np.float32): keep them usable without cupy."""
    return getattr(np, name) if name in _NUMPY_ALIASES else None


_NUMPY_ALIASES = frozenset(('float16', 'float32', 'float64', 'complex64', 'complex128', 'int8', 'int16', 'int32',
                            'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool_', 'dtype', 'pi', 'inf', 'nan',
                            'newaxis', 'intp', 'generic', 'number', 'floating', 'complexfloating', 'integer'))


class _MissingCupy(_MissingGPU):
    """The `cp` stand-in: numpy scalar types and constants pass through, everything else is a _MissingGPU."""
    __slots__ = ()

    def __getattr__(self, attribute):
        value = _numpy_dtype_stand_in(attribute)
        if value is not None:
            return value
        return _MissingGPU.__getattr__(self, attribute)


_cupy_import_error = None
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupyx_ndimage
    import cupyx.scipy.sparse as cupyx_sparse
except Exception as exc:   # not installed, or a broken CUDA install: the CPU backend still works
    _cupy_import_error = f'{type(exc).__name__}: {exc}'
    cp = _MissingCupy('cupy')
    cupyx_sparse = _MissingGPU('cupyx.scipy.sparse')
    cupyx_ndimage = _MissingGPU('cupyx.scipy.ndimage')


def _import_error_text():
    return f' (cupy import failed: {_cupy_import_error})' if _cupy_import_error else ''


def cupy_installed() -> bool:
    """True when cupy imported (no driver query: GUI-thread safe)."""
    return _cupy_import_error is None


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """cupy imported and at least one CUDA device answers (one driver query, cached: call it on the worker thread)."""
    if not cupy_installed():
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def default_backend_name() -> str:
    """Backend of a script object built without one: 'gpu' when cupy imported, else 'cpu'."""
    return 'gpu' if cupy_installed() else 'cpu'


def _require_backend(backend):
    if backend not in BACKEND_NAMES:
        raise ValueError(f"backend must be one of {', '.join(BACKEND_NAMES)}, got {backend!r}")


def array_module(backend):
    """numpy ('cpu') or cupy ('gpu'; BackendUnavailable without cupy)."""
    _require_backend(backend)
    if backend == 'cpu':
        return np
    if not cupy_installed():
        raise BackendUnavailable(f"backend 'gpu' {_MISSING_TEXT}{_import_error_text()}")
    return cp


def sparse_module(backend):
    """scipy.sparse ('cpu') or cupyx.scipy.sparse ('gpu')."""
    _require_backend(backend)
    if backend == 'cpu':
        import scipy.sparse
        return scipy.sparse
    array_module(backend)
    return cupyx_sparse


def ndimage_module(backend):
    """scipy.ndimage ('cpu') or cupyx.scipy.ndimage ('gpu')."""
    _require_backend(backend)
    if backend == 'cpu':
        import scipy.ndimage
        return scipy.ndimage
    array_module(backend)
    return cupyx_ndimage


def array_module_or_default(xp):
    """xp, else the array module of the default backend (script objects built without one)."""
    return xp if xp is not None else array_module(default_backend_name())


def array_module_of(array):
    """numpy for numpy arrays / scalars, cupy for cupy arrays."""
    if cupy_installed() and isinstance(array, cp.ndarray):
        return cp
    return np


def backend_of(array) -> str:
    """'gpu' for a cupy array, else 'cpu'."""
    return 'gpu' if array_module_of(array) is not np else 'cpu'


def to_host(array):
    """numpy copy-or-view of a numpy or cupy array (one transfer for a cupy array)."""
    if isinstance(array, np.ndarray):
        return array
    if cupy_installed() and isinstance(array, cp.ndarray):
        return array.get()
    return np.asarray(array)


def synchronize(backend) -> None:
    """Wait for queued device work ('gpu'); nothing on the cpu."""
    if backend == 'gpu':
        cp.cuda.Device().synchronize()


@dataclass(frozen=True)
class Backend:
    """One run backend: name ('gpu' / 'cpu') and its array module xp (sparse_module / ndimage_module(name) for the
    sparse and ndimage modules)."""
    name: str
    xp: object

    def synchronize(self) -> None:
        synchronize(self.name)


@functools.lru_cache(maxsize=None)
def get_backend(name='gpu') -> Backend:
    """The Backend of name ('gpu' raises BackendUnavailable without cupy)."""
    return Backend(name, array_module(name))
