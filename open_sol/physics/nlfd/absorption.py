"""Absorbed power density of NonlinearHarmonicPhysics for the EM loss protocol (em_heat_joule): the channels' linear
q_abs without their lagged Kerr / carrier permittivity part, plus the exact nonlinear (TPA) and free-carrier absorption
of the applied bulk and sheet terms on the stored fields (arrays of the run backend)."""
# Picard puts the bulk Kerr / carrier diagonal into each channel operator lagged (FDFDPhysics.extra_permittivity), so
# a channel q_abs holds -Im(delta eps) |E|^2 of the face-collocated fields; newton keeps those terms in the rhs and
# q_abs holds none. Both are replaced by q_nl = -(omega_k / 2) Im(conj(E_k) . P_k) per cell of the applied SPM / XPM
# terms and the carrier polarisation (centre fields of the store); sheet SPM / XPM (rhs only in both methods) add their
# site work per tap in the tap's cell (sheets.sheet_site_absorption). The sum over the cells times the cell areas is
# exactly nonlinear_absorption + free_carrier_absorption (the scalars), whatever the iteration method.
from ...core.backend import array_module_of
from ...core.constants import DTYPE
from ..em.harmonic import absorbed_power_ez, absorbed_power_hz
from .mixing import POLARIZATION_COMPONENTS
from .sheets import sheet_site_absorption, sheet_site_polarizations
from .sources import polarization_density


def extra_permittivity_loss(physics):
    """float32 full-grid [W/m^3] share of a channel's q_abs from its extra_permittivity (the q_abs formula with that
    change only), None without one."""
    extra = physics.extra_permittivity
    if extra is None or physics.system is None or physics.solution is None:
        return None
    box, values = extra
    shape = physics.solution.shape
    xp = physics.xp
    full = []
    for part in values:
        array = xp.zeros(shape, dtype=xp.complex128)
        if part is not None:
            array[(...,) if box is None else box] = part
        full.append(array)
    system = physics.system
    if system.polarization == 'ez':
        return absorbed_power_ez(system.omega, full[0], physics.solution)
    ex_faces, ey_faces = physics.face_fields()
    return absorbed_power_hz(system.omega, full[0], full[1], ex_faces, ey_faces, system.periodic_x, system.periodic_y)


def _add_work_density(total, omega, field, density):
    """total -= (omega / 2) Im(conj(E) . P) per cell (float64 full grid)."""
    xp = array_module_of(total)
    for component, values in density.items():
        total -= (0.5 * float(omega)) * (xp.conj(field[component]) * values).imag


def kerr_terms(source, sheet=False) -> tuple:
    """The applied SPM / XPM (Kerr) terms of a channel phasor: its bulk terms, or (sheet) its sheet terms."""
    return tuple(term for term in (source.sheet_terms if sheet else source.bulk_terms) if term.kerr)


def work_fields(nl, target, kerr, carriers=True) -> dict:
    """{channel index: stored centre field} the bulk Kerr terms `kerr` read and (carriers, with free carriers) the
    carrier sources and the target."""
    indices = {index for term in kerr for index, _ in term.sources}
    if carriers and nl.free_carriers is not None:
        indices |= set(nl.free_carriers.source_indices) | {target}
    return {index: nl.store.centre_vector(index) for index in indices}


def kerr_density(nl, target, kerr, fields):
    """Bulk Kerr polarisation density {component: complex128 full grid} of channel `target`, None without terms."""
    if not kerr:
        return None
    components = POLARIZATION_COMPONENTS[nl.channels[target].polarization]
    return polarization_density(kerr, fields, {medium.name: medium for medium in nl.media}, components,
                                nl.store.shape, nl.store.xp)


def sheet_kerr_sites(nl, target, sheet_kerr):
    """(site polarisations, {name: sheet}, {channel index: grid fields}) of the sheet Kerr terms of channel `target`,
    None without terms."""
    if not sheet_kerr:
        return None
    channel = nl.channels[target]
    sheets = {sheet.name: sheet for sheet in nl.sheets}
    grids = {index: nl.store.grid_fields(index) for index in range(len(nl.channels))}
    sites = sheet_site_polarizations(sheet_kerr, grids, sheets, POLARIZATION_COMPONENTS[channel.polarization],
                                     channel.omega, nl.store.xp)
    return sites, sheets, grids


def carrier_density(nl, target, fields):
    """Free-carrier polarisation density {component: complex128 full grid} of channel `target`, None without
    carriers."""
    if nl.free_carriers is None:
        return None
    store = nl.store
    density = {component: store.xp.zeros(store.shape, dtype=store.xp.complex128)
               for component in POLARIZATION_COMPONENTS[nl.channels[target].polarization]}
    nl.free_carriers.add_density(target, fields, density)
    return density


def nonlinear_absorption_density(nonlinear_physics):
    """float64 full-grid [W/m^3] absorption of the applied Kerr (SPM / XPM) terms of bulk media and sheets and of the
    free-carrier polarisation on the stored fields (0 undepleted: no Kerr term acts there)."""
    nl = nonlinear_physics
    store = nl.store
    xp = store.xp
    total = xp.zeros(store.shape, dtype=xp.float64)
    for target, source in nl._sources.items():
        channel = nl.channels[target]
        field = store.centre_vector(target)
        if field is None:
            continue
        sheet_part = sheet_kerr_sites(nl, target, kerr_terms(source, sheet=True))
        if sheet_part is not None:
            sites, sheets, grids = sheet_part
            total += sheet_site_absorption(sites, sheets, grids[target], channel.omega, nl.cell_area, xp)
            del sites, sheet_part
        kerr = kerr_terms(source)   # none undepleted
        fields = work_fields(nl, target, kerr)
        density = kerr_density(nl, target, kerr, fields)
        if density is not None:
            _add_work_density(total, channel.omega, field, density)
        density = carrier_density(nl, target, fields)
        if density is not None:
            _add_work_density(total, channel.omega, field, density)
    return total


def absorbed_power_density(nonlinear_physics, out):
    """Fill float32 full-grid `out` [W/m^3] with sum over channels (q_abs - extra_permittivity_loss) +
    nonlinear_absorption_density (the take_absorbed_power value)."""
    out.fill(0)
    for physics in nonlinear_physics.channel_physics:
        q_abs = physics.take_absorbed_power()
        if q_abs is None:
            continue
        out += q_abs
        lagged = extra_permittivity_loss(physics)
        if lagged is not None:
            out -= lagged
    out += nonlinear_absorption_density(nonlinear_physics).astype(DTYPE)
    return out
