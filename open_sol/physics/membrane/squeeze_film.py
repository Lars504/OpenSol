"""Runtime of the membrane feature squeeze_film_damping (MembraneSqueezeFilm): a fem.squeeze_film.SqueezeFilm on the
membrane's own TriangleMesh and Lagrange order, restricted to the footprint elements, vented per the vents choice, and
the gas damping rows per resonance of an eigenfrequency solve."""
# Footprint = the feature's domain selection AND the solved window elements without holes (decided per evaluation: a
# coupled tension can change the holes). The mode vector is the membrane's own full Lagrange field (peak 1 m), so it
# goes to the film directly (film mass matrix, exact on isoparametric curved elements); m_eff = Re(psi^H M psi) with
# the pencil mass rho_s, omega = Re omega. Vents 'open_edges': every boundary edge of the footprint except the sealed
# ones the solver reports (window edges of clamped / simply supported / symmetry faces, symmetry outlines, and held
# edges with membrane on both sides: clamped / simply supported lines or outlines inside the film); free edges, hole
# outlines and antisymmetry faces vent (p = 0; odd pressure on an odd mirror). 'all' vents the whole boundary, 'none'
# none. The SqueezeFilm is kept while the footprint and vents stay the same (an incompressible film: one factor for
# every mode).
import numpy as np

from ...core.backend import to_host
from ...fem.mesh import side_members
from ...fem.squeeze_film import SqueezeFilm, active_boundary_edges, film_table, mode_gas_damping

MEMBRANE_VENT_CHOICES = ('open_edges', 'all', 'none')


class MembraneSqueezeFilm:
    """One squeeze_film_damping feature of a membrane: footprint (bool per element), gas inputs (ambient None =
    incompressible), vents; the SqueezeFilm of the current footprint and the rows of the last eigen solve."""

    def __init__(self, name, footprint, gap, viscosity, ambient_pressure=None, *, mean_free_path=0.0,
                 slip='veijola', vents='open_edges'):
        if vents not in MEMBRANE_VENT_CHOICES:
            raise ValueError(f"squeeze film '{name}': vents must be one of {', '.join(MEMBRANE_VENT_CHOICES)}, got "
                             f'{vents!r}')
        self.name = str(name)
        self.footprint = np.asarray(to_host(footprint), dtype=bool).reshape(-1).copy()
        self.inputs = dict(gap=float(gap), viscosity=float(viscosity),
                           ambient_pressure=None if ambient_pressure is None else float(ambient_pressure),
                           mean_free_path=float(mean_free_path), slip=slip)
        self.vents = vents
        self.film = None      # fem.squeeze_film.SqueezeFilm of the current footprint
        self.rows = None      # mode_gas_damping rows of the last evaluated eigen solve
        self._key = None      # (active footprint, vent edges) the film was built for

    def label(self):
        return f"squeeze film '{self.name}'"

    def active_footprint(self, membrane_active, window, pml_faces) -> np.ndarray:
        """Bool per element of the film: the footprint on the solved window elements without holes; ValueError for a
        footprint reaching a PML ring or holding no membrane element."""
        if self.footprint.shape != membrane_active.shape:
            raise ValueError(f'{self.label()}: footprint mask has {self.footprint.size} entries, the mesh '
                             f'{membrane_active.size} elements')
        if pml_faces and (self.footprint & ~window).any():
            raise ValueError(f"{self.label()}: the footprint reaches the PML ring ({', '.join(pml_faces)}); select "
                             'objects that stay inside the window')
        active = self.footprint & membrane_active & window
        if not active.any():
            raise ValueError(f'{self.label()}: the selected domains hold no membrane element (holes or outside the '
                             'window)')
        return active

    def vent_edges(self, mesh, active, sealed_edges, interior_holds, membrane_active):
        """Vented edges of the footprint boundary per the vents choice (sealed: always-sealed edges; interior_holds:
        held edges, sealed where membrane lies on both sides)."""
        boundary = active_boundary_edges(mesh, active)
        if self.vents == 'all':
            return boundary
        if self.vents == 'none':
            return boundary[:0]
        holds = np.unique(np.asarray(interior_holds, dtype=np.int64))
        if holds.size:
            first, second = side_members(mesh.edge_elements[holds], membrane_active)
            holds = holds[first & second]
        sealed = np.union1d(np.asarray(sealed_edges, dtype=np.int64), holds)
        return boundary[~np.isin(boundary, sealed)]

    def build(self, mesh, active, vents, order) -> SqueezeFilm:
        """The SqueezeFilm of this footprint and vents (kept while both stay the same)."""
        if self.film is not None and np.array_equal(active, self._key[0]) and np.array_equal(vents, self._key[1]):
            return self.film
        self.film = SqueezeFilm(mesh, self.inputs['gap'], self.inputs['viscosity'], self.inputs['ambient_pressure'],
                                mean_free_path=self.inputs['mean_free_path'], slip=self.inputs['slip'],
                                vent_edges=vents, active=active, order=order, label=self.label)
        self._key = (active.copy(), np.asarray(vents).copy())
        return self.film

    def evaluate(self, film, shapes, omegas, masses) -> list:
        """mode_gas_damping rows per resonance (full-numbering shapes, Re omega [rad/s], m_eff [kg]); kept as rows."""
        self.rows = None
        self.rows = mode_gas_damping(film, shapes, omegas, masses)
        return self.rows

    @property
    def area(self) -> float:
        """Footprint area [m^2] of the last built film (0 before one)."""
        if self.film is None:
            return 0.0
        return float(np.abs(self.film.mesh.areas()[self._key[0]]).sum())

    def table(self) -> dict:
        """'table' result (FILM_TABLE_COLUMNS) with extras area [m^2], gap [m], film_elements and vent_edges;
        ValueError before an evaluation."""
        if self.rows is None:
            raise ValueError(f'{self.label()}: not evaluated yet: the gas damping is computed per resonance by an '
                             'eigenfrequency study')
        return film_table(self.rows, area=self.area, gap=self.inputs['gap'], film_elements=int(self._key[0].sum()),
                          vent_edges=int(len(self.film.vent_edges)))
