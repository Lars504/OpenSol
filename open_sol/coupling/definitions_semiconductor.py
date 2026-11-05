"""Coupling types between drift_diffusion and the EM types: em_semiconductor_generation (absorbed optical power ->
photogeneration) and semiconductor_em_plasma_dispersion (free carriers -> EM index, an E13 'index' contributor);
discovered and registered by definitions.register."""
from ..model.schema import CouplingType, Issue, Param
from ..physics.semiconductor.couplings import (PLASMA_ELECTRON_COEFFICIENT, PLASMA_HOLE_COEFFICIENT,
                                               PLASMA_HOLE_EXPONENT, make_generation_transform, make_plasma_transform)
from .coupling_term import CouplingTerm
from .index_terms import index_contributor, index_slots, target_type, writes_index_slots

DRIFT_DIFFUSION_TYPE_NAME = 'drift_diffusion'   # = physics.semiconductor.definition.DRIFT_DIFFUSION_TYPE_NAME
GENERATION_TYPE_NAME = 'em_semiconductor_generation'
PLASMA_TYPE_NAME = 'semiconductor_em_plasma_dispersion'
_GENERATION_SLOT = 'generation'
_ABSORBED_POWER_READERS = frozenset({'em_heat_joule', GENERATION_TYPE_NAME})   # call take_absorbed_power()


# ---- create callbacks (worker thread, the run backend) ----
def _create_generation(ctx, coupling_spec, values, every):
    """Overwriting CouplingTerm q_accum -> generation (q_accum only makes the runner enable FDTD loss accounting)."""
    transform = make_generation_transform(values['photon_wavelength'], values['quantum_efficiency'],
                                          ctx.domain.materials.shape, ctx.xp)   # (nx, ny), or (n_elements,)
    return CouplingTerm(coupling_spec.source, 'q_accum', coupling_spec.target, _GENERATION_SLOT, transform,
                        every=every, name=coupling_spec.name)


def _create_plasma(ctx, coupling_spec, values, every):
    """One 'index' contributor per index slot of the target (the same isotropic dn), a CouplingGroup when several."""
    transform = make_plasma_transform(values['electron_coefficient'], values['hole_coefficient'],
                                      values['hole_exponent'], ctx.domain.materials.shape, ctx.xp)
    return index_contributor(ctx, coupling_spec, every, 'N', lambda slot: transform)


# ---- validation (CPU only) ----
def _source_steps_in_time(vctx, physics_spec):
    """True when the physics' type steps in time with its settings (FDTD); False when unknown."""
    try:
        physics_type = vctx.registry.physics_type(physics_spec.type)
    except KeyError:
        return False
    if physics_type.steps_in_time is None:
        return False
    settings = vctx.values(physics_type.settings, physics_spec.settings)
    return bool(settings is not None and physics_type.steps_in_time(settings))


def _validate_generation(vctx, coupling_spec, values):
    """Errors: a time-stepped (FDTD) source without photon_wavelength, or whose absorbed power another coupling also
    takes (each take_absorbed_power() resets the FDTD window). The writer rule is model.validate's (`writes`)."""
    path = ('coupling', coupling_spec.name)
    try:
        source_spec = vctx.spec.physics_by_name(coupling_spec.source)
    except KeyError:
        return []
    if not _source_steps_in_time(vctx, source_spec):
        return []
    issues = []
    if values['photon_wavelength'] is None:
        issues.append(Issue('error', path, f"source '{source_spec.name}' steps in time and has no single solve "
                                           'frequency: set photon_wavelength (the vacuum wavelength of the light)'))
    others = [other.name for other in vctx.spec.couplings
              if other.enabled and other.name != coupling_spec.name and other.source == coupling_spec.source
              and other.type in _ABSORBED_POWER_READERS]
    if others:
        issues.append(Issue('error', path, f"coupling{'s' if len(others) > 1 else ''} {', '.join(others)} also take "
                                           f"the absorbed power of the time-stepped '{source_spec.name}': each "
                                           'transfer resets its averaging window, so each reader would get only part '
                                           'of it; keep one reader (or use an em_fdfd type)'))
    return issues


def _validate_plasma(vctx, coupling_spec, values):
    """Error when the target type has no index slot (the writer rule is model.validate's, through `writes`)."""
    physics_type = target_type(vctx, coupling_spec)
    if physics_type is None or index_slots(physics_type):
        return []
    return [Issue('error', ('coupling', coupling_spec.name),
                  f"target physics '{coupling_spec.target}' ({physics_type.name}) has no permittivity slot; its source "
                  f"slots: {', '.join(physics_type.source_slots) or 'none'}")]


def _writes_generation(vctx, coupling_spec):
    return {_GENERATION_SLOT: ''}


# ---- types ----
def _generation_type(absorbing_em_types):
    return CouplingType(
        GENERATION_TYPE_NAME, 'Photogeneration (electromagnetic waves -> semiconductor)', absorbing_em_types,
        DRIFT_DIFFUSION_TYPE_NAME,
        params=(
            Param('photon_wavelength', 'float', None, label='Photon wavelength', unit='m', min=0.0,
                  min_exclusive=True, optional=True,
                  help='Vacuum wavelength of the absorbed light, setting the photon energy hbar omega = 2 pi hbar c / '
                       'wavelength; null = the solve wavelength of an em_fdfd source (its omega). Required for a '
                       'time-stepped (em_fdtd) source (validation error).',
                  tip='Wavelength of the light in vacuum, which sets the energy of each photon. Leave empty to use '
                      "the wavelength of the frequency-domain wave physics."),
            Param('quantum_efficiency', 'float', 1.0, label='Quantum efficiency', unit='1', min=0.0, max=1.0,
                  help='Electron-hole pairs per absorbed photon (internal quantum efficiency); 1 for band-to-band '
                       'absorption. Free-carrier or metal absorption creates no pairs: such cells are either not '
                       'semiconductor cells (generation ignored there) or need a lower value.',
                  tip='Fraction of the absorbed photons that create an electron and a hole; 1 for normal '
                      'band-to-band absorption.')),
        summary='Turns the light absorbed by a wave physics into electron-hole pairs in the semiconductor, as in a '
                "photodetector. With a frequency-domain wave physics, use a 'Stationary' study.",
        description='Photogeneration: G = quantum_efficiency * q_abs / (hbar omega) [1/(m^3 s)], q_abs the absorbed '
                    'power density [W/m^3] taken through the EM loss protocol take_absorbed_power() (em_fdfd: the '
                    'time-averaged loss of the last solve; em_fdtd: the mean since the last transfer, which a second '
                    'reader of the same source would reset: validation error), omega from photon_wavelength or the '
                    "em_fdfd solve frequency. It overwrites the drift_diffusion slot 'generation' (one writer; the "
                    'generation features add separately) and acts only on the free semiconductor cells (n_intrinsic > '
                    '0, not held by an ohmic contact). Every absorbing cell counts: the optical sigma_e / eps '
                    'imaginary part of the EM material is the absorption, so give the semiconductor its band-to-band '
                    'loss at the wavelength (library silicon is lossless). Use a stationary study (em_fdfd solves, '
                    'then the carriers; with a plasma-dispersion coupling back to the EM physics the study iterates '
                    'to the fixed point); a frequency_domain study does not feed harmonic solutions to a background '
                    'physics (validation error).',
        create=_create_generation, validate=_validate_generation, writes=_writes_generation,
        mesh_transfer='value')


def _plasma_type(em_types):
    return CouplingType(
        PLASMA_TYPE_NAME, 'Free-carrier plasma dispersion (semiconductor -> electromagnetic waves)',
        DRIFT_DIFFUSION_TYPE_NAME, em_types,
        params=(
            Param('electron_coefficient', 'float', PLASMA_ELECTRON_COEFFICIENT, label='Electron index coefficient',
                  unit='m^3', min=0.0,
                  help='a in dn = -(a N + b P^c), N [1/m^3]; default 8.8e-28 = Soref-Bennett silicon at 1.55 um '
                       '(8.8e-22 cm^3); at 1.31 um 6.2e-28.',
                  tip='Index decrease per free electron density. The default is silicon at 1550 nm.'),
            Param('hole_coefficient', 'float', PLASMA_HOLE_COEFFICIENT, label='Hole index coefficient',
                  unit='m^2.4', min=0.0,
                  help='b in dn = -(a N + b P^c), P [1/m^3]; default 8.5e-18 * (1e-6)^0.8 = 1.347e-22 = '
                       'Soref-Bennett silicon at 1.55 um (8.5e-18 with P in cm^-3); at 1.31 um 6.0e-18 * (1e-6)^0.8 = '
                       '9.51e-23.',
                  tip='Index decrease per free hole density (raised to the hole exponent). The default is silicon '
                      'at 1550 nm.'),
            Param('hole_exponent', 'float', PLASMA_HOLE_EXPONENT, label='Hole exponent', unit='1', min=0.0,
                  help='c in dn = -(a N + b P^c); 0.8 for silicon (Soref-Bennett). Changing it changes the unit of '
                       'hole_coefficient.',
                  tip='Exponent of the hole density in the index change; 0.8 for silicon.')),
        summary='Lowers the refractive index of a wave physics where the semiconductor holds free electrons and '
                'holes, as in a silicon modulator. The defaults are silicon at 1550 nm.',
        description='Free-carrier plasma dispersion: each transfer gives the index change dn = -(electron_coefficient '
                    'N + hole_coefficient P^hole_exponent) from the electron / hole densities N, P [1/m^3] of the '
                    'last drift_diffusion solve (0 outside the semiconductor; edge-extended into the PML ring), the '
                    'same for every permittivity component the EM type has (E_z types: eps_r_update; H_z types: '
                    'eps_xx_update / eps_yy_update; em_modes: all three). It is an E13 index contributor: the slot '
                    'gets eps = (n0 + sum of the dn of every index coupling)^2 with n0 the unperturbed index, so it '
                    'adds to heat_em_thermo_optic / electrostatics_em_pockels (all need one every); an overwriting '
                    'field_transfer or a state-dependent permittivity property on the slot cannot be combined with '
                    'it. The densities are absolute (dn relative to carrier-free material, as the library silicon '
                    'index is undoped). Defaults: Soref-Bennett silicon at 1.55 um. The free-carrier absorption '
                    '(Soref-Bennett dalpha = 8.5e-18 N + 6.0e-18 P [1/cm], N, P in 1/cm^3) is not part of it (the EM '
                    'types have no loss slot): give the material a state-dependent sigma_e = n eps0 c0 dalpha, for '
                    'silicon at 1.55 um (n 3.48, N, P in 1/m^3) sigma_e = 7.85e-24 * semiconductor_N + 5.54e-24 * '
                    'semiconductor_P [S/m] (plus any background loss). Stationary study: one '
                    'fixed-point iteration behind within the loop; mode_analysis / frequency_domain: applied once '
                    'after the background solve (cross-section modulators with em_modes).',
        create=_create_plasma, validate=_validate_plasma, writes=writes_index_slots,
        mesh_transfer='field')   # N and the P it reads interpolated onto the EM mesh


def register(registry, physics):
    """When drift_diffusion is registered: the plasma-dispersion type (needs an EM type) and the generation type
    (needs an absorbing EM type); physics: definitions.RegisteredPhysics."""
    if DRIFT_DIFFUSION_TYPE_NAME not in physics.names:
        return
    if physics.em_types:
        registry.register_coupling(_plasma_type(physics.em_types))
    if physics.absorbing_em_types:
        registry.register_coupling(_generation_type(physics.absorbing_em_types))
