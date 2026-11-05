"""Bloch / Floquet unit cells of the tetrahedra solid (piezo_3d_plan D9): the periodic identification of the periodic
axes on the full-mesh scalar map, its vector / coupled (block layout) masters, the prolongation x = P x_r over the free
masters, the rigid motions the periodicity leaves and the dofs on periodic faces. Host numpy (scipy inside functions)."""
# fem.dofs3d.bloch_masters (the em3d convention): each hi-side dof stands for its lo-side partner times e^{-j k L} per
# crossed period, u(x + L) = u(x) e^{-j k L} with exp(+j omega t); the three u components and phi share the scalar
# dof's master and phase. The test space is the conjugate one, so K_r = P^H K P and M_r = P^H M P: complex Hermitian,
# real symmetric when every phase is +-1 (k L a multiple of pi). A master is fixed with any dof it stands for (a support
# on either side). Rigid motions t + w x x: an axis with a phase != 1 admits none, a phase-1 axis keeps the
# translations and the rotation about itself (w x e_axis = 0); the same phase-1 rule keeps a constant potential, so a
# piezoelectric cell needs a ground electrode only then.
from dataclasses import dataclass

import numpy as np

from ...fem.dofs3d import bloch_masters

AXES = ('x', 'y', 'z')


@dataclass(frozen=True, eq=False)
class BlochCell3D:
    """Periodic identification of one assembly's scalar map: master (n,) int64, phase (n,) (float64 when every phase
    is real, else complex128), wavenumbers (kx, ky, kz) [rad/m] (0 off the periodic axes), periodic (3,) bool, trivial
    = every periodic phase is 1 (rigid translations and a constant potential are Bloch states)."""
    master: np.ndarray
    phase: np.ndarray
    wavenumbers: tuple
    periodic: tuple
    trivial: bool

    @property
    def complex_phase(self) -> bool:
        return bool(np.iscomplexobj(self.phase))

    @property
    def on_periodic_faces(self) -> np.ndarray:
        """Bool per scalar dof standing for another or stood for (a dof on a periodic face)."""
        n_dofs = self.master.size
        moved = self.master != np.arange(n_dofs)
        marked = moved.copy()
        marked[self.master[moved]] = True
        return marked

    def vector_masters(self, components=3) -> tuple:
        """(master, phase) over the interleaved vector dofs (dof = components d + component)."""
        offsets = np.arange(components, dtype=np.int64)
        master = (components * self.master[:, None] + offsets).reshape(-1)
        return master, np.repeat(self.phase, components)

    def coupled_masters(self, components=3) -> tuple:
        """(master, phase) over the coupled block layout [u (components n, interleaved) | phi (n)]."""
        vector_master, vector_phase = self.vector_masters(components)
        return (np.concatenate([vector_master, components * self.master.size + self.master]),
                np.concatenate([vector_phase, self.phase]))

    def rigid_rows(self) -> np.ndarray:
        """(k, 6) rows of the rigid motions (t, w) the phase-1 axes exclude: the w components normal to each axis."""
        rows = [np.eye(6)[3 + other] for axis, periodic in enumerate(self.periodic) if periodic
                for other in range(3) if other != axis]
        return np.array(rows).reshape(-1, 6)

    def text(self) -> str:
        """'k_x = 1.2e+06 rad/m, ...' of the periodic axes."""
        return ', '.join(f'k_{axis} = {self.wavenumbers[index]:.6g} rad/m' for index, axis in enumerate(AXES)
                         if self.periodic[index])


def bloch_cell_3d(dofs, periodic, wavenumbers) -> BlochCell3D:
    """BlochCell3D of a Lagrange DofMap3D for periodic (3,) bool axes and wavenumbers (kx, ky, kz) [rad/m]; ValueError
    when the two sides of a periodic axis are meshed differently."""
    periodic = tuple(bool(value) for value in periodic)
    wavenumbers = tuple(float(value) if periodic[index] else 0.0 for index, value in enumerate(wavenumbers))
    master, phase = bloch_masters(dofs, {axis: wavenumbers[axis] for axis in range(3) if periodic[axis]})
    if not np.any(phase.imag):
        phase = np.ascontiguousarray(phase.real)
    trivial = bool(np.all(phase == 1.0))
    return BlochCell3D(master=master, phase=phase, wavenumbers=wavenumbers, periodic=periodic, trivial=trivial)


def bloch_prolongation(master, phase, free) -> tuple:
    """(columns (n_r,) int64: the free masters, host CSR P (n, n_r)) with x = P x_r: each dof standing for a free
    master carries its phase (float64 P when the phases are real), the rest 0; a master is free when every dof it
    stands for is."""
    import scipy.sparse
    free = np.asarray(free, dtype=bool)
    columns = np.setdiff1d(master[free], master[~free])
    column_of = np.full(master.size, -1, dtype=np.int64)
    column_of[columns] = np.arange(columns.size)
    rows = np.flatnonzero(column_of[master] >= 0)
    prolongation = scipy.sparse.csr_matrix((phase[rows], (rows, column_of[master[rows]])),
                                           shape=(master.size, columns.size))
    return columns, prolongation
