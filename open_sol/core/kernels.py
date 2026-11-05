"""Kernel twins (solver_plan S6): each fused cupy kernel paired with a numpy twin of the same call signature (and an
optional numba twin, built lazily when numba imports), so engine code is one path on both backends: a twin call runs
the cupy kernel for cupy arrays and the host twin for numpy arrays. Imports numpy and core.backend only."""
import importlib.util
import os
import warnings

import numpy as np

from .backend import cp

# name -> KernelTwin of every module-level twin (registered at import of its module)
REGISTRY = {}
# OPEN_SOL_CPU_KERNELS = 'numpy' keeps the numpy twins even with numba installed (default 'auto')
_CPU_KERNELS_VARIABLE = 'OPEN_SOL_CPU_KERNELS'
_numba_state = {'module': None, 'checked': False, 'enabled': None}


def numba_available() -> bool:
    """numba importable and not switched off (OPEN_SOL_CPU_KERNELS = 'numpy' or set_numba(False))."""
    if _numba_state['enabled'] is False:
        return False
    if (_numba_state['enabled'] is None
            and os.environ.get(_CPU_KERNELS_VARIABLE, 'auto').strip().lower() == 'numpy'):
        return False
    if not _numba_state['checked']:
        _numba_state['checked'] = True
        if importlib.util.find_spec('numba') is not None:
            try:
                import numba
                _numba_state['module'] = numba
            except Exception:   # a broken numba install: the numpy twins still work
                _numba_state['module'] = None
    return _numba_state['module'] is not None


def set_numba(enabled) -> None:
    """True / False forces the numba twins on (when importable) / off for every twin; None = the default rule."""
    _numba_state['enabled'] = None if enabled is None else bool(enabled)
    for kernel in REGISTRY.values():
        kernel._host = None


def _numba_compile_error(exc) -> bool:
    """True for numba's own errors (typing, lowering) and dispatch TypeErrors: raised before the twin runs."""
    errors = getattr(getattr(_numba_state['module'], 'core', None), 'errors', None)
    numba_error = getattr(errors, 'NumbaError', None)
    return isinstance(exc, TypeError) or (numba_error is not None and isinstance(exc, numba_error))


def _host_arrays(arguments) -> bool:
    """True when the first array among `arguments` is a numpy array (False for cupy, or when none is an array)."""
    for value in arguments:
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, cp.ndarray):
            return False
    return False


class KernelTwin:
    """One fused kernel on both backends: gpu = the cupy kernel object, cpu = numpy twin, numba = optional factory
    numba -> compiled twin. raw = RawKernel call convention kernel(grid, block, args) (the host twin gets *args)."""

    __slots__ = ('name', 'gpu', 'cpu', 'raw', '_numba_factory', '_host', '_host_kind')

    def __init__(self, name, gpu, cpu, numba=None, raw=False):
        self.name = name
        self.gpu = gpu
        self.cpu = cpu
        self.raw = bool(raw)
        self._numba_factory = numba
        self._host = None        # resolved host callable (numba twin or cpu), None = resolve at the next host call
        self._host_kind = None

    def __call__(self, *args, **kwargs):
        arguments = args[2] if self.raw else args
        if not _host_arrays(arguments):
            return self.gpu(*args, **kwargs)
        host = self._host if self._host is not None else self._resolve_host()
        if self._host_kind == 'numba':
            try:
                return host(*arguments)
            except Exception as exc:   # typing / dispatch failure (before any write): fall back to numpy for good
                if not _numba_compile_error(exc):
                    raise   # a failure mid-execution may have updated outputs in place: rerunning would repeat it
                warnings.warn(f"kernel twin '{self.name}': the numba twin failed ({type(exc).__name__}: {exc}); "
                              'using the numpy twin', RuntimeWarning, stacklevel=2)
                self._host, self._host_kind = self.cpu, 'numpy'
        return self.cpu(*arguments)

    def _resolve_host(self):
        """The numba twin when it has one and numba is available, else the numpy twin."""
        if self._numba_factory is not None and numba_available():
            try:
                self._host, self._host_kind = self._numba_factory(_numba_state['module']), 'numba'
                return self._host
            except Exception as exc:
                warnings.warn(f"kernel twin '{self.name}': building the numba twin failed ({type(exc).__name__}: "
                              f'{exc}); using the numpy twin', RuntimeWarning, stacklevel=3)
        self._host, self._host_kind = self.cpu, 'numpy'
        return self._host

    @property
    def host_kind(self) -> str:
        """'numba' or 'numpy': the host twin the next numpy call runs."""
        if self._host is None:
            self._resolve_host()
        return self._host_kind

    def __repr__(self):
        return f'<KernelTwin {self.name!r}>'


def _register(twin_kernel):
    if twin_kernel.name in REGISTRY:
        raise ValueError(f"kernel twin '{twin_kernel.name}' is registered twice")
    REGISTRY[twin_kernel.name] = twin_kernel
    return twin_kernel


def twin(name, gpu, cpu, numba=None):
    """Registered twin of an ElementwiseKernel / ReductionKernel (or any cupy callable): the numpy twin takes the same
    positional arguments (inputs, then outputs, written in place) and returns what the kernel returns."""
    return _register(KernelTwin(name, gpu, cpu, numba=numba))


def raw_twin(name, gpu, cpu, numba=None):
    """Registered twin of a RawKernel launched as kernel(grid, block, args): the numpy twin takes *args (grid and block
    are ignored on the host)."""
    return _register(KernelTwin(name, gpu, cpu, numba=numba, raw=True))


# A cp.ReductionKernel to one scalar runs on ONE thread block. Over the shortest axis first it spreads over many
# (RTX 3050, float64 window energy: 904 -> 124 us at 1e6 values, 3636 -> 319 us at 4e6); below this many values the
# single block is faster (13 vs 24 us at 1e4)
AXIS_FIRST_MIN_SIZE = 1 << 16


def axis_first_reduce(kernel, *args, combine=None):
    """kernel(*args) reduced to one value: on cupy arrays (args[0]'s shape) of >= AXIS_FIRST_MIN_SIZE values and >= 2
    dims first over the shortest axis, then combine(partials) (None: their sum); numpy / small / 1D: one call."""
    first = args[0]
    if not isinstance(first, cp.ndarray) or first.ndim < 2 or first.size < AXIS_FIRST_MIN_SIZE:
        return kernel(*args)
    partials = kernel(*args, axis=int(np.argmin(first.shape)))
    return partials.sum() if combine is None else combine(partials)


# real dtype -> C type of real_t in a RealRawKernel (the 2D FDTD stepper precisions single / double)
REAL_C_TYPES = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}


class RealRawKernel:
    """RawKernel compiled per real type: the CUDA source writes real_t (and may test REAL_IS_DOUBLE); a call
    kernel(grid, block, args) compiles / picks the variant of args[dtype_argument].dtype (float32 or float64)."""

    def __init__(self, source, name, dtype_argument=0):
        self.source = source
        self.name = name
        self.dtype_argument = int(dtype_argument)
        self._compiled = {}

    def __call__(self, grid, block, args, **kwargs):
        dtype = np.dtype(args[self.dtype_argument].dtype)
        kernel = self._compiled.get(dtype)
        if kernel is None:
            if dtype not in REAL_C_TYPES:
                raise TypeError(f"kernel '{self.name}' runs on float32 or float64 arrays, got {dtype.name}")
            c_type = REAL_C_TYPES[dtype]
            preamble = f'typedef {c_type} real_t;\n#define REAL_IS_DOUBLE {int(c_type == "double")}\n'
            kernel = self._compiled[dtype] = cp.RawKernel(preamble + self.source, self.name)
        return kernel(grid, block, args, **kwargs)


def real_raw_twin(name, source, entry, cpu, numba=None, dtype_argument=0):
    """raw_twin of a RealRawKernel (CUDA `source` with real_t, kernel `entry`): float32 and float64 arrays alike; the
    numpy / numba twins follow the dtype of their arrays."""
    return raw_twin(name, RealRawKernel(source, entry, dtype_argument), cpu, numba=numba)


def host_kinds() -> dict:
    """{name: 'numba' | 'numpy'} of every registered twin (resolves the host twins; no numba compile)."""
    return {name: kernel.host_kind for name, kernel in sorted(REGISTRY.items())}
