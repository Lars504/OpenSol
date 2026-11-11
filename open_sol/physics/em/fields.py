"""Fields2D_Ez (E_z, H_x, H_y) and Fields2D_Hz (H_z, E_x, E_y) field sets of 2D FDTD, plus collocate_to_centres."""
import numpy as np

from ...core.backend import array_module_of, array_module_or_default

from ...core.constants import DTYPE


def _field_dtype(dtype):
    """np.dtype of a field set: float32 (stepper 'single', default) or float64 (stepper 'double'); ValueError else."""
    dtype = np.dtype(DTYPE if dtype is None else dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f'2D FDTD fields are float32 or float64, got {dtype.name}')
    return dtype


class _FieldSet2D:
    """(nx, ny) zero arrays `components` of `xp` (cupy / numpy: the run backend, None = the default backend) in `dtype`
    (float32, float64 for the double stepper); `imag` = the imaginary field set of the same class under Bloch k != 0
    (set by the engine; DFT monitors add it as j F_im), else None."""

    components = ()

    def __init__(self, nx: int, ny: int, xp=None, dtype=None):
        self.nx = nx
        self.ny = ny
        self.xp = array_module_or_default(xp)
        self.dtype = _field_dtype(dtype)
        for name in self.components:
            setattr(self, name, self.xp.zeros((nx, ny), dtype=self.dtype))
        self.imag = None

    def reset(self):
        for name in self.components:
            getattr(self, name)[:] = 0
        if self.imag is not None:
            self.imag.reset()


class Fields2D_Ez(_FieldSet2D):
    """Ez-polarisation fields (Ez, Hx, Hy) as _FieldSet2D."""

    components = ('ez', 'hx', 'hy')


class Fields2D_Hz(_FieldSet2D):
    """Hz-polarisation fields as _FieldSet2D: hz at the cell centres, ex[i, j] on the hi-y face and ey[i, j] on the
    hi-x face of cell (i, j)."""

    components = ('hz', 'ex', 'ey')


def collocate_to_centres(face_field, axis: int, periodic: bool):
    """Hi-face samples (ex: hi-y faces, axis 1; ey: hi-x faces, axis 0) averaged onto the cell centres as a new float32
    (nx, ny) array of the input's module (a float64 field is summed in float64, then stored float32): the float32 fast
    path of the yee_grid.lower_face_partner rule (harmonic.face_mean_at_centres)."""
    if axis not in (0, 1):
        raise ValueError(f"collocate_to_centres axis must be 0 (x) or 1 (y), got {axis!r}")
    xp = array_module_of(face_field)
    centred = xp.empty(face_field.shape, dtype=DTYPE)
    faces = xp.moveaxis(face_field, axis, 0)       # views with the averaging axis first
    centres = xp.moveaxis(centred, axis, 0)
    xp.add(faces[1:], faces[:-1], out=centres[1:])
    xp.add(faces[0], faces[-1] if periodic else faces[0], out=centres[0])
    centred *= DTYPE(0.5)
    return centred
