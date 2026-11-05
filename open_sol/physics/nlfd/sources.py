"""Nonlinear polarisation phasors of the NLFD-chi channels (the FDFD add_to_rhs / face_currents source protocol), the
channel field store they read, the lagged Kerr operator term, steady-state free carriers, the Newton-Krylov tangents
and the C / C* collocation that keeps the discrete power exchange exact."""
# Signs (physics/em/harmonic.py header): polarisation current J_std = j omega P, FDTD sign J_F = -j omega P, so
#   ez: rhs += -j omega mu0 J_F = -omega^2 mu0 P_z (P per volume: no cell area, unlike a filament current);
#   hz: centre P_x / P_y -> faces by C* (centres_to_faces), J_F there through harmonic.face_current_rhs, and
#       face_currents() so FDFDPhysics.face_fields returns E = (curl H + J_F) / (j omega eps0 eps_face).
# C = harmonic.face_mean_at_centres (cell k = mean of its hi face k and lo face k - 1; cell 0: wrapped last face or
# its own hi face). C* = its adjoint in the area-weighted inner products (cells w_i w_j, faces the dual widths
# (w_k + w_(k+1)) / 2 of yee_inverse_spacings): sum_c A_c (C f)_c p_c = sum_f A_f f_f (C* p)_f, so the centre work below
# equals the face work of the distributed current exactly (uniform periodic grids: C* = C^T, the plain face mean).
# Bloch k != 0: C of a channel's field takes its wrap factor e^{+j k L} on cell 0's wrapped lo face, C* of a target
# density its conjugate on the last face (ChannelFieldStore.wrap_factors), so C* stays the adjoint of C.
# Power delivered by P_k to channel k: W_k = (omega_k / 2) sum_c A_c Im(conj(E_k) . P_k) [W/m] (2D: per unit length in
# z). For a real tensor with full permutation symmetry, sum_k W_k = 0 for any fields once every term of the table acts.
# Sources never read a channel's FDFDPhysics directly: they read the ChannelFieldStore snapshot (ez E_z, hz E_x / E_y
# on their faces), captured after each channel solve, so a channel's own solve (SPM, back-action) cannot change what
# its sources returned for that solve, and the pump-depleted mixing writes the store only.
# Arrays live on the run backend (ChannelFieldStore.xp = domain.xp: cupy or numpy); every math function takes xp.
import numpy as np

from ...core.backend import array_module_of
from ...core.constants import C0, DTYPE, EPS0, HBAR, MU0
from ..em.harmonic import face_current_rhs, face_mean_at_centres
from ..em.yee_grid import face_average, grid_cell_widths
from .media import union_box
from .mixing import POLARIZATION_COMPONENTS
from .sheets import sheet_grid_density, sheet_site_polarizations, sheet_site_work


def _axis_view(axis, part):
    """Index tuple selecting `part` (a slice) along `axis` of a 2D array."""
    index = [slice(None), slice(None)]
    index[axis] = part
    return tuple(index)


def centres_to_faces(centre_values, axis, periodic, cell_widths, xp=None, wrap_factor=1.0):
    """C* of face_mean_at_centres(wrap_factor=) along `axis`: hi-face values (n,) from centre values, area-weighted
    (cell widths w, dual widths (w_k + w_(k+1)) / 2); a Bloch wrap_factor e^{+j k L} gives the last face cell 0 times
    conj(wrap_factor) (the cell one period above); complex or float arrays (xp None = their module), new array."""
    xp = array_module_of(centre_values) if xp is None else xp
    widths = np.asarray(cell_widths, dtype=np.float64)
    shape = [1, 1]
    shape[axis] = widths.size
    half_widths = xp.asarray((0.5 * widths).reshape(shape))
    dual_widths = xp.asarray((0.5 * (widths + np.roll(widths, -1))).reshape(shape))
    weighted = centre_values * half_widths        # each cell gives half its weight to its hi and to its lo face
    faces = weighted.copy()
    faces[_axis_view(axis, slice(None, -1))] += weighted[_axis_view(axis, slice(1, None))]   # lo face of cell k + 1
    first = _axis_view(axis, slice(0, 1))
    if periodic:   # cell 0's lo face is the wrapped last face
        wrapped = weighted[first] if wrap_factor == 1.0 else weighted[first] * complex(wrap_factor).conjugate()
        faces[_axis_view(axis, slice(-1, None))] += wrapped
    else:
        faces[first] += weighted[first]                               # cell 0 pairs with its own hi face
    faces /= dual_widths
    return faces


def cell_areas(domain):
    """float64 (nx, ny) Yee cell areas w_x,i w_y,j [m^2] (yee_grid.grid_cell_widths) on the domain's backend."""
    cell_dx, cell_dy = grid_cell_widths(domain.grid)
    return domain.xp.asarray(np.outer(cell_dx, cell_dy))


class ChannelFieldStore:
    """Snapshot of every channel's field the nonlinear sources read, in one flat complex128 vector of the run backend
    (allocated at the first capture): ez E_z at the centres, hz E_x (hi-y faces), E_y (hi-x faces); version 0 = never
    captured. wrap_factors[index] = that channel's FDFDPhysics.lower_wrap_factors (Bloch k; None = all 1)."""

    def __init__(self, channels, domain, boundary_config, wrap_factors=None):
        self.channels = tuple(channels)
        self.wrap_factors = (tuple((1.0, 1.0) for _ in self.channels) if wrap_factors is None
                             else tuple(tuple(factors) for factors in wrap_factors))
        self.xp = domain.xp
        self.shape = (domain.grid.nx, domain.grid.ny)
        n_cells = self.shape[0] * self.shape[1]
        self.counts = tuple(len(POLARIZATION_COMPONENTS[channel.polarization]) for channel in self.channels)
        offsets = np.concatenate(([0], np.cumsum(self.counts) * n_cells)).astype(int)
        self.blocks = tuple(slice(int(start), int(stop)) for start, stop in zip(offsets[:-1], offsets[1:]))
        self.size = int(offsets[-1])
        self.cell_dx, self.cell_dy = grid_cell_widths(domain.grid)
        self.periodic_x, self.periodic_y = bool(boundary_config.periodic_x), bool(boundary_config.periodic_y)
        self.versions = [0] * len(self.channels)
        self.values = None

    def vector(self):
        """The flat complex128 store (zeros at the first call): channel `index` occupies blocks[index]."""
        if self.values is None:
            self.values = self.xp.zeros(self.size, dtype=self.xp.complex128)
        return self.values

    def arrays(self, index, vector=None):
        """(nx, ny) views of channel `index` in the store (or in a store-layout `vector`): ez (E_z,), hz (E_x faces,
        E_y faces)."""
        vector = self.vector() if vector is None else vector
        return tuple(vector[self.blocks[index]].reshape((self.counts[index],) + self.shape))

    def read(self, index, physics, vector):
        """Copy channel `index`'s last solve (FDFDPhysics) into its block of a store-layout `vector` (no version)."""
        views = self.arrays(index, vector)
        if physics.polarization == 'ez':
            self.xp.copyto(views[0], physics.solution)
        else:
            for view, faces in zip(views, physics.face_fields()):
                self.xp.copyto(view, faces)

    def capture(self, index, physics):
        """Copy channel `index`'s last solve (FDFDPhysics) into the store and bump its version."""
        self.read(index, physics, self.vector())
        self.versions[index] += 1

    def write(self, values):
        """Overwrite every channel's stored field (a mixed pump-depleted iterate) and bump every version."""
        self.xp.copyto(self.values, values)
        self.versions = [version + 1 for version in self.versions]

    def key(self, indices):
        """Versions of channels `indices`: the cache key of what the sources computed from them."""
        return tuple(self.versions[index] for index in indices)

    def centre_vector(self, index, vector=None):
        """(E_x, E_y, E_z) complex128 full-grid centre phasors of channel `index` (absent components None; hz E
        collocated by C, new arrays) of the store (None when never captured) or of a store-layout `vector`."""
        if vector is None and self.versions[index] == 0:
            return None
        views = self.arrays(index, vector)
        if self.channels[index].polarization == 'ez':
            return None, None, views[0]
        wrap_x, wrap_y = self.wrap_factors[index]
        return (face_mean_at_centres(views[0], 1, self.periodic_y, self.xp, wrap_factor=wrap_y),
                face_mean_at_centres(views[1], 0, self.periodic_x, self.xp, wrap_factor=wrap_x), None)

    def grid_fields(self, index, vector=None):
        """(E_x hi-y faces, E_y hi-x faces, E_z centres) views of channel `index` (absent components None), what the
        sheet sites sample, of the store (None when never captured) or of a store-layout `vector`."""
        if vector is None and self.versions[index] == 0:
            return None
        views = self.arrays(index, vector)
        if self.channels[index].polarization == 'ez':
            return None, None, views[0]
        return views[0], views[1], None


def _term_fields(term, centre_fields):
    """Per source slot the centre field vector of term.sources (None entries when a source channel is unsolved)."""
    return [centre_fields.get(index) for index, _ in term.sources]


def _local_box(inner, outer):
    """Slices of box `inner` relative to the enclosing box `outer` (None = the full grid)."""
    if outer is None:
        return inner
    return tuple(slice(part.start - whole.start, part.stop - whole.start) for part, whole in zip(inner, outer))


def _density_arrays(components, shape, box, xp):
    """{component: complex128 zeros} of the full grid (box None) or of `box`."""
    if box is not None:
        shape = tuple(part.stop - part.start for part in box)
    return {component: xp.zeros(shape, dtype=xp.complex128) for component in components}


def _complex_cell_factor(cache, name, medium, xp):
    """complex128 copy of medium.cell_factor, made once per call (cache: {medium name: array})."""
    if name not in cache:
        cache[name] = medium.cell_factor.astype(xp.complex128)
    return cache[name]


def polarization_density(terms, centre_fields, media, components, shape, xp, box=None):
    """{component: complex128 full-grid P [C/m^2]} (or on `box`, which must hold every term's medium box) of
    MixingTerms from centre_fields[channel index] = (E_x, E_y, E_z) (None: unsolved, its terms skipped) and media
    {name: BulkMedium}; components = the target channel's model components."""
    density = _density_arrays(components, shape, box, xp)
    cell_factors = {}   # medium name -> complex128 cell factor, shared by the medium's terms (per call)
    box_fields = {}     # (medium, channel index, conjugate, component) -> box view / conjugate, shared likewise
    for term in terms:
        fields = _term_fields(term, centre_fields)
        if any(field is None for field in fields):
            continue
        medium = media[term.medium]
        local = _local_box(medium.box, box)
        cell_factor = _complex_cell_factor(cell_factors, term.medium, medium, xp)
        for component, source_components, value in term.components:
            product = cell_factor * (EPS0 * term.factor * value)
            for (index, conjugate), field, source_component in zip(term.sources, fields, source_components):
                key = (term.medium, index, conjugate, source_component)
                if key not in box_fields:
                    box_field = field[source_component][medium.box]
                    box_fields[key] = xp.conj(box_field) if conjugate else box_field
                product *= box_fields[key]
            density[component][local] += product
    return density


def polarization_tangent(terms, centre_fields, tangent_fields, media, components, shape, xp, box=None):
    """{component: complex128 dP} = d/dt P(E + t V) at real t = 0 of polarization_density (product rule over the
    source slots; a conjugated slot takes conj V): the exact Jacobian-vector product of the polynomial terms;
    tangent_fields as centre_fields (centre vectors of V)."""
    density = _density_arrays(components, shape, box, xp)
    cell_factors = {}   # medium name -> complex128 cell factor, shared by the medium's terms (per call)
    box_values = {}     # (medium, tangent?, channel index, conjugate, component) -> box view / conjugate, likewise
    for term in terms:
        fields, tangents = _term_fields(term, centre_fields), _term_fields(term, tangent_fields)
        if any(field is None for field in fields) or any(tangent is None for tangent in tangents):
            continue
        medium = media[term.medium]
        local = _local_box(medium.box, box)
        cell_factor = _complex_cell_factor(cell_factors, term.medium, medium, xp)
        for component, source_components, value in term.components:
            scale = cell_factor * (EPS0 * term.factor * value)
            for slot in range(len(term.sources)):
                product = scale
                for position, ((index, conjugate), field, tangent, source_component) in enumerate(
                        zip(term.sources, fields, tangents, source_components)):
                    key = (term.medium, position == slot, index, conjugate, source_component)
                    if key not in box_values:
                        values = (tangent if position == slot else field)[source_component][medium.box]
                        box_values[key] = xp.conj(values) if conjugate else values
                    product = product * box_values[key]
                density[component][local] += product
    return density


def kerr_diagonal(terms, target, centre_fields, media, components, box, xp):
    """{component a: complex128 box array M_aa}: relative permittivity change of the diagonal of channel `target`'s
    Kerr terms written as P_i = eps0 sum_a M_ia E_target,a (one plain target slot factored out; terms with an
    unsolved other source skipped)."""
    shape = tuple(part.stop - part.start for part in box)
    delta = {component: xp.zeros(shape, dtype=xp.complex128) for component in components}
    for term in terms:
        slot = term.sources.index((target, False))
        fields = _term_fields(term, centre_fields)
        if any(field is None for position, field in enumerate(fields) if position != slot):
            continue
        medium = media[term.medium]
        local = _local_box(medium.box, box)
        cell_factor = medium.cell_factor.astype(xp.complex128)
        for component, source_components, value in term.components:
            if component not in delta or source_components[slot] != component:
                continue
            product = cell_factor * (term.factor * value)
            for position, ((_, conjugate), field, source_component) in enumerate(
                    zip(term.sources, fields, source_components)):
                if position != slot:
                    box_field = field[source_component][medium.box]
                    product *= xp.conj(box_field) if conjugate else box_field
            delta[component][local] += product
    return delta


def delivered_power(omega, centre_field, density, cell_area, xp=None):
    """W [W/m] = (omega / 2) sum A Im(conj(E) . P) of a density on its target channel's centre field (host sync; xp
    None = the module of the cell areas)."""
    xp = array_module_of(cell_area) if xp is None else xp
    total = 0.0
    for component, values in density.items():
        total += float(xp.sum(cell_area * (xp.conj(centre_field[component]) * values).imag))
    return 0.5 * float(omega) * total


def density_face_currents(density, store, omega, face_density=None, target=0):
    """hz (J_x on hi-y, J_y on hi-x faces) complex128 FDTD-sign currents -j omega (C* P + P_faces) [A/m^2] of a centre
    density of channel `target` (its Bloch wrap factors) plus an optional face density {0: hi-y faces, 1: hi-x faces}
    (the sheets' deposit)."""
    scale = -1j * omega
    wrap_x, wrap_y = store.wrap_factors[target]
    jx_faces = centres_to_faces(density[0], 1, store.periodic_y, store.cell_dy, store.xp, wrap_factor=wrap_y)
    jy_faces = centres_to_faces(density[1], 0, store.periodic_x, store.cell_dx, store.xp, wrap_factor=wrap_x)
    if face_density is not None:
        jx_faces += face_density[0]
        jy_faces += face_density[1]
    jx_faces *= scale
    jy_faces *= scale
    return jx_faces, jy_faces


def _check_polarization(owner, polarization, system):
    if system.polarization != polarization:
        raise ValueError(f"{owner} of an {polarization!r} channel, but the system is {system.polarization!r}")


class DensityPhasor:
    """A fixed polarisation density {component: full-grid P [C/m^2]} of channel `target` as an FDFD phasor source (the
    Newton-Krylov Jacobian solves): ez -omega^2 mu0 P_z, hz face currents -j omega (C* P + the sheet face density)."""

    def __init__(self, density, polarization, store, face_density=None, target=0):
        self.density = density
        self.polarization = polarization
        self.target = int(target)
        self._store = store
        self._face_density = face_density
        self._face_currents = None

    def face_currents(self, physics, system):
        _check_polarization('DensityPhasor', 'hz', system)
        if self._face_currents is None:
            self._face_currents = density_face_currents(self.density, self._store, system.omega, self._face_density,
                                                        self.target)
        return self._face_currents

    def add_to_rhs(self, rhs, physics, system):
        _check_polarization('DensityPhasor', self.polarization, system)
        if system.polarization == 'ez':
            rhs += (-(system.omega ** 2) * MU0) * self.density[2]
            return
        rhs += face_current_rhs(system, *self.face_currents(physics, system))


class NonlinearPolarizationPhasor:
    """Nonlinear source of one channel: P of its mixing terms (+ the free-carrier polarisation eps0 delta_eps(N) E
    of the target, pump-depleted) from the stored source-channel fields, as an FDFD phasor source (ez: -omega^2 mu0 P_z
    at the centres; hz: face currents -j omega C* P). Terms of `sheets` (sheets.SheetMedium) are evaluated on their
    sites and deposited apart (ez centres, hz faces). skip_unsolved: terms with a never-captured source give 0
    (pump-depleted first sweep); else such a source raises (cascade order)."""

    def __init__(self, target, terms, store, media, skip_unsolved=False, free_carriers=None, sheets=()):
        self.target = int(target)
        self.terms = tuple(terms)
        self._store = store
        self._media = {medium.name: medium for medium in media}
        self._sheets = {sheet.name: sheet for sheet in sheets}
        self.bulk_terms = tuple(term for term in self.terms if term.medium not in self._sheets)
        self.sheet_terms = tuple(term for term in self.terms if term.medium in self._sheets)
        self._sheet_sites = {}        # sheet name -> (n_sites, 3) site P_s of the cached density
        self._sheet_density = None    # {component: full grid}: the sheets' deposit (ez centres, hz faces)
        self._skip_unsolved = bool(skip_unsolved)
        self.free_carriers = free_carriers
        indices = {index for term in self.terms for index, _ in term.sources}
        if free_carriers is not None:
            indices |= set(free_carriers.source_indices) | {self.target}
        self.source_indices = tuple(sorted(indices))
        self.polarization = store.channels[self.target].polarization
        self.components = POLARIZATION_COMPONENTS[self.polarization]
        self._density_key = None
        self._density = None
        self._face_key = None
        self._face_currents = None

    def density(self):
        """{component: complex128 full-grid P [C/m^2]} of the target's components, cached per source version."""
        key = self._store.key(self.source_indices)
        if key != self._density_key:
            self._density = self._sheet_density = None   # free the previous ones first
            self._face_currents = None
            fields = {index: self._store.centre_vector(index) for index in self.source_indices}
            missing = [index for index, field in fields.items() if field is None]
            if missing and not self._skip_unsolved:
                names = ', '.join(self._store.channels[index].name for index in missing)
                raise ValueError(f'nonlinear source of channel {self._store.channels[self.target].name}: source '
                                 f'channel(s) {names} not solved yet (the cascade solves every source channel before '
                                 'its targets)')
            xp = self._store.xp
            self._density = polarization_density(self.bulk_terms, fields, self._media, self.components,
                                                 self._store.shape, xp)
            if self.free_carriers is not None:
                self.free_carriers.add_density(self.target, fields, self._density)
            self._sheet_sites = {}
            if self.sheet_terms:
                grids = {index: self._store.grid_fields(index) for index in self.source_indices}
                self._sheet_sites = sheet_site_polarizations(self.sheet_terms, grids, self._sheets, self.components,
                                                             self._store.channels[self.target].omega, xp)
                self._sheet_density = sheet_grid_density(self._sheet_sites, self._sheets, self.components,
                                                         self._store.shape, xp)
            self._density_key = key
        return self._density

    def face_currents(self, physics, system):
        """hz: (J_x on hi-y, J_y on hi-x faces) complex128 FDTD-sign currents -j omega (C* P + P_sheets) [A/m^2]."""
        _check_polarization('NonlinearPolarizationPhasor face currents', 'hz', system)
        density = self.density()
        key = (self._density_key, system.omega)
        if key != self._face_key:
            self._face_currents = None
            self._face_currents = density_face_currents(density, self._store, system.omega, self._sheet_density,
                                                        self.target)
            self._face_key = key
        return self._face_currents

    def add_to_rhs(self, rhs, physics, system):
        _check_polarization('NonlinearPolarizationPhasor', self.polarization, system)
        if system.polarization == 'ez':
            scale = -(system.omega ** 2) * MU0
            rhs += scale * self.density()[2]
            if self._sheet_density is not None:
                rhs += scale * self._sheet_density[2]
            return
        rhs += face_current_rhs(system, *self.face_currents(physics, system))

    def power(self, cell_area):
        """W [W/m] delivered to the target channel's stored field: bulk centre work + the sheets' site work (0 before
        the target is captured; host sync)."""
        field = self._store.centre_vector(self.target)
        if field is None:
            return 0.0
        omega = self._store.channels[self.target].omega
        xp = self._store.xp
        work = delivered_power(omega, field, self.density(), cell_area, xp)
        if self._sheet_sites:
            work += sheet_site_work(self._sheet_sites, self._sheets, self._store.grid_fields(self.target), omega, xp)
        return work

    def magnitude(self):
        """float32 full-grid |P| [C/m^2] of the current density (renderer 'nonlinear_polarization_abs'; hz sheet face
        densities collocated by the face mean C)."""
        xp = self._store.xp
        values = dict(self.density())
        if self._sheet_density is not None:
            if self.polarization == 'ez':
                values[2] = values[2] + self._sheet_density[2]
            else:
                wrap_x, wrap_y = self._store.wrap_factors[self.target]
                values[0] = values[0] + face_mean_at_centres(self._sheet_density[0], 1, self._store.periodic_y, xp,
                                                             wrap_factor=wrap_y)
                values[1] = values[1] + face_mean_at_centres(self._sheet_density[1], 0, self._store.periodic_x, xp,
                                                             wrap_factor=wrap_x)
        intensity = sum(xp.abs(component) ** 2 for component in values.values())
        return xp.sqrt(intensity).astype(DTYPE)


class KerrOperatorTerm:
    """Lagged Kerr permittivity of one channel (pump-depleted, plan decision 7): the diagonal M_aa of its chi(3)
    |E_m|^2 E terms goes into the channel operator (FDFDPhysics.set_extra_permittivity on the media's union box), and
    the same operator change applied to the stored previous field is subtracted again as a source, so a fixed point
    solves the rhs form of every term (the operator part only stabilises the iteration; ez: the two cancel exactly).
    free_carriers: their permittivity change of this channel joins M (it is diagonal and linear in E already)."""

    def __init__(self, target, terms, store, media, free_carriers=None):
        self.target = int(target)
        self.terms = tuple(terms)
        self._store = store
        self._media = {medium.name: medium for medium in media}
        self.free_carriers = free_carriers
        boxes = [self._media[term.medium].box for term in self.terms]
        indices = {index for term in self.terms for index, _ in term.sources}
        if free_carriers is not None:
            boxes += [medium.box for medium in free_carriers.media]
            indices |= set(free_carriers.source_indices)
        self.box = union_box(boxes)
        self.source_indices = tuple(sorted(indices))
        self.polarization = store.channels[self.target].polarization
        self.components = POLARIZATION_COMPONENTS[self.polarization]
        self._rhs = None              # ez: box correction k0^2 M E_old
        self._face_currents = None    # hz: + j omega eps0 face_average(M) E_old on the faces

    def update(self, physics):
        """Recompute M from the stored fields, set it on the channel `physics` and prepare the correction source."""
        store = self._store
        xp = store.xp
        fields = {index: store.centre_vector(index) for index in self.source_indices}
        delta = kerr_diagonal(self.terms, self.target, fields, self._media, self.components, self.box, xp)
        if self.free_carriers is not None:
            self.free_carriers.add_permittivity(self.target, fields, delta, self.box)
        physics.set_extra_permittivity(delta[2] if self.polarization == 'ez' else (delta[0], delta[1]), self.box)
        self._rhs = self._face_currents = None
        if store.versions[self.target] == 0:   # no previous field: nothing to subtract (its Kerr rhs is skipped too)
            return
        previous = store.arrays(self.target)
        omega = physics.omega
        if self.polarization == 'ez':
            self._rhs = ((omega / C0) ** 2) * delta[2] * previous[0][self.box]
            return
        currents = []
        for component, axis, periodic, widths, faces in ((0, 1, store.periodic_y, store.cell_dy, previous[0]),
                                                         (1, 0, store.periodic_x, store.cell_dx, previous[1])):
            full = xp.zeros(store.shape, dtype=xp.complex128)
            full[self.box] = delta[component]
            face_delta = face_average(full, axis, periodic, widths, xp=xp)
            del full
            face_delta *= (1j * omega * EPS0) * faces
            currents.append(face_delta)
        self._face_currents = tuple(currents)

    def face_currents(self, physics, system):
        return self._face_currents if self._face_currents is not None else (None, None)

    def add_to_rhs(self, rhs, physics, system):
        if self.polarization == 'ez':
            if self._rhs is not None:
                rhs[self.box] += self._rhs
        elif self._face_currents is not None:
            rhs += face_current_rhs(system, *self._face_currents)


def _work_density(field, density, box, xp):
    """sum_c conj(E_c) P_c on `box` (complex; density box-shaped, field a full-grid centre vector)."""
    total = None
    for component, values in density.items():
        product = xp.conj(field[component][box]) * values
        total = product if total is None else total + product
    return total


class FreeCarriers:
    """Steady-state carrier density N [1/m^3] of the chi(3) media with a CarrierModel (pump-depleted, no diffusion):
    N = lifetime max(G, 0), G = -(1 / (4 hbar)) sum_k Im(conj(E_k) . P_kerr,k) per cell (q_k / (2 hbar omega_k): one
    pair per two absorbed photons) from the medium's applied SPM / XPM terms on the stored fields; every channel then
    sees delta_eps_k = n0 N CarrierModel.permittivity_factor(omega_k) on the medium's box (xp None = store.xp)."""

    def __init__(self, kerr_terms, store, media, xp=None):
        self._store = store
        self._xp = store.xp if xp is None else xp
        self._media = {medium.name: medium for medium in media}
        self.media = tuple(medium for medium in media if medium.carriers is not None)
        self.terms = {medium.name: tuple(term for term in kerr_terms if term.medium == medium.name)
                      for medium in self.media}
        indices = set()
        for terms in self.terms.values():
            for term in terms:
                indices |= {index for index, _ in term.sources} | {term.target}
        self.source_indices = tuple(sorted(indices))
        self._key = None
        self._state = {}   # medium name -> (N box float64, G > 0 box bool, {target: its Kerr density on the box})

    def refresh(self, fields):
        """Recompute N of every medium when the stored source fields changed; fields {index: centre vector} must be the
        store's current ones (the cache is keyed on the store, not on `fields`: another vector gets a stale N)."""
        key = self._store.key(self.source_indices)
        if key == self._key:
            return
        xp = self._xp
        self._state = {}
        for medium in self.media:
            terms = self.terms[medium.name]
            generation = xp.zeros(tuple(part.stop - part.start for part in medium.box), dtype=xp.float64)
            densities = {}
            for target in sorted({term.target for term in terms}):
                field = fields.get(target)
                if field is None:
                    continue
                components = POLARIZATION_COMPONENTS[self._store.channels[target].polarization]
                density = polarization_density([term for term in terms if term.target == target], fields, self._media,
                                               components, None, xp=xp, box=medium.box)
                generation -= _work_density(field, density, medium.box, xp).imag / (4.0 * HBAR)
                densities[target] = density
            positive = generation > 0.0
            self._state[medium.name] = (medium.carriers.lifetime * xp.where(positive, generation, 0.0), positive,
                                        densities)
        self._key = key

    def refresh_from_store(self):
        """refresh() with the store's current fields (scalars, renderer)."""
        self.refresh({index: self._store.centre_vector(index) for index in self.source_indices})

    def medium_states(self, fields):
        """{medium name: (N box float64, G > 0 box bool, {Kerr target: its Kerr density on the box})} of fields
        {index: centre vector} (refreshed; adjoint.carrier_cotangents)."""
        self.refresh(fields)
        return self._state

    def add_density(self, target, fields, density):
        """density[c] (full grid) += eps0 delta_eps_target E_target,c on every carrier medium's box."""
        field = fields.get(target)
        if field is None:
            return
        self.refresh(fields)
        omega = self._store.channels[target].omega
        for medium in self.media:
            coefficient = self._state[medium.name][0] * medium.carriers.index * (
                EPS0 * medium.carriers.permittivity_factor(omega))
            for component in density:
                density[component][medium.box] += coefficient * field[component][medium.box]

    def add_permittivity(self, target, fields, delta, box):
        """delta[c] (arrays on `box`) += the carrier permittivity change of channel `target` (Kerr operator term)."""
        self.refresh(fields)
        omega = self._store.channels[target].omega
        for medium in self.media:
            change = self._state[medium.name][0] * medium.carriers.index * medium.carriers.permittivity_factor(omega)
            local = _local_box(medium.box, box)
            for component in delta:
                delta[component][local] += change

    def carrier_tangent(self, fields, tangents):
        """{medium name: dN box} along the tangent centre vectors (Newton-Krylov; 0 where G <= 0)."""
        self.refresh(fields)
        xp = self._xp
        result = {}
        for medium in self.media:
            carriers, positive, densities = self._state[medium.name]
            d_generation = xp.zeros(carriers.shape, dtype=xp.float64)
            for target, density in densities.items():
                tangent = tangents.get(target)
                if tangent is None:
                    continue
                target_terms = [term for term in self.terms[medium.name] if term.target == target]
                d_density = polarization_tangent(target_terms, fields, tangents, self._media, tuple(density), None,
                                                 xp=xp, box=medium.box)
                change = (_work_density(tangent, density, medium.box, xp)
                          + _work_density(fields[target], d_density, medium.box, xp))
                d_generation -= change.imag / (4.0 * HBAR)
            result[medium.name] = medium.carriers.lifetime * xp.where(positive, d_generation, 0.0)
        return result

    def add_tangent(self, target, fields, tangents, carrier_tangent, density):
        """density[c] += eps0 n0 factor_k (dN E_k + N V_k): the carrier part of the Jacobian-vector product."""
        field, tangent = fields.get(target), tangents.get(target)
        if field is None or tangent is None:
            return
        omega = self._store.channels[target].omega
        for medium in self.media:
            carriers, box = self._state[medium.name][0], medium.box
            d_carriers = carrier_tangent[medium.name]
            factor = EPS0 * medium.carriers.permittivity_factor(omega)
            for component in density:
                change = d_carriers * field[component][box] + carriers * tangent[component][box]
                density[component][box] += (medium.carriers.index * change) * factor

    def peak_density(self):
        """Largest N [1/m^3] of the stored fields (host sync)."""
        self.refresh_from_store()
        return max((float(state[0].max()) for state in self._state.values()), default=0.0)

    def density_map(self, shape):
        """float32 full-grid N [1/m^3] of the stored fields (renderer 'carrier_density'; media add)."""
        self.refresh_from_store()
        total = self._xp.zeros(shape, dtype=DTYPE)
        for medium in self.media:
            total[medium.box] += self._state[medium.name][0].astype(DTYPE)
        return total


class NewtonLinearization:
    """Jacobian pieces of the pump-depleted Newton-Krylov solve at the stored fields x: per target channel with an
    applied phasor the directional derivative of its polarisation (mixing terms + free carriers) along a store-layout
    tangent v, as a DensityPhasor (one channel solve each); sheet terms by the product rule on their sites."""

    def __init__(self, phasors, store, media, free_carriers=None, sheets=()):
        self.phasors = dict(phasors)
        self._store = store
        self._media = {medium.name: medium for medium in media}
        self._sheets = {sheet.name: sheet for sheet in sheets}
        self.free_carriers = free_carriers
        self._key = None
        self._fields = None   # centre vectors at x, cached per store versions

    def tangent_context(self, tangent):
        """(centre vectors at x, centre vectors of v, carrier tangent or None, grid fields at x, grid fields of v)
        shared by one product's sources (grid fields: store views for the sheet sites)."""
        store = self._store
        indices = range(len(store.channels))
        key = tuple(store.versions)
        if key != self._key:
            self._fields = None
            self._fields = {index: store.centre_vector(index) for index in indices}
            self._key = key
        tangents = {index: store.centre_vector(index, tangent) for index in indices}
        carrier_tangent = (None if self.free_carriers is None
                           else self.free_carriers.carrier_tangent(self._fields, tangents))
        return (self._fields, tangents, carrier_tangent, {index: store.grid_fields(index) for index in indices},
                {index: store.grid_fields(index, tangent) for index in indices})

    def source(self, target, context):
        """DensityPhasor of dP of channel `target` along the context's tangent, None without an applied phasor."""
        phasor = self.phasors.get(target)
        if phasor is None:
            return None
        fields, tangents, carrier_tangent, grid_fields, grid_tangents = context
        xp = self._store.xp
        density = polarization_tangent(phasor.bulk_terms, fields, tangents, self._media, phasor.components,
                                       self._store.shape, xp)
        if self.free_carriers is not None:
            self.free_carriers.add_tangent(target, fields, tangents, carrier_tangent, density)
        face_density = None
        if phasor.sheet_terms:
            sites = sheet_site_polarizations(phasor.sheet_terms, grid_fields, self._sheets, phasor.components,
                                             self._store.channels[target].omega, xp, grid_tangents)
            sheet_density = sheet_grid_density(sites, self._sheets, phasor.components, self._store.shape, xp)
            if phasor.polarization == 'ez':
                density[2] += sheet_density[2]
            else:
                face_density = sheet_density
        return DensityPhasor(density, phasor.polarization, self._store, face_density, target)
