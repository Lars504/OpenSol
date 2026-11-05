"""Ready-made example models ('New from template'): the built-in ones plus those definition modules register;
every template validates without errors."""
from .schema import TemplateEntry, get_registry
from .spec import (OUTER_FACES, CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                   MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, StudySpec)


def _dielectric_cylinder():
    """Pulsed point source scattering off a glass cylinder (4 x 4 um, uniform 20 nm)."""
    return ModelSpec(
        title='Dielectric cylinder',
        parameters=[ParameterSpec('lambda0', '550*nm', 'source centre wavelength'),
                    ParameterSpec('r_cylinder', '0.6*um', 'cylinder radius')],
        domain=DomainSpec(size_x=4e-6, size_y=4e-6),
        geometry=[GeometrySpec('cylinder', 'circle', {'cx': 2e-6, 'cy': 2e-6, 'r': 'r_cylinder'}),
                  GeometrySpec('source_point', 'point', {'x': 0.8e-6, 'y': 2e-6}),
                  GeometrySpec('probe_point', 'point', {'x': 3.0e-6, 'y': 2e-6}),
                  GeometrySpec('flux_line', 'line', {'x0': 3.4e-6, 'y0': 0.5e-6, 'x1': 3.4e-6, 'y1': 3.5e-6})],
        materials=[MaterialSpec('glass', properties={'eps_r': 2.25}, selection=['cylinder'])],
        physics=[PhysicsSpec('em', 'em_fdtd_ez', features=[
            FeatureSpec('source', 'point_source', ['source_point'],
                        {'waveform': 'gaussian', 'wavelength': 'lambda0', 'bandwidth': 0.5, 'current': 1e-3}),
            FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': 'ez'}),
            FeatureSpec('transmitted_flux', 'flux_monitor', ['flux_line'],
                        {'wavelength_start': 400e-9, 'wavelength_stop': 800e-9, 'n_wavelengths': 41})])],
        meshes=[MeshSpec(mode='uniform', dx=20e-9)],
        study=StudySpec(type='time_dependent', settings={'duration': 150e-15, 'output_interval': 2.5e-15}))


def _gold_nanodisk():
    """Plane-wave (TFSF) scattering by a Drude gold disk with near-field DFT and far-field boxes (uniform 2 nm)."""
    spectral = {'wavelength_start': 400e-9, 'wavelength_stop': 900e-9, 'n_wavelengths': 11}
    return ModelSpec(
        title='Gold nanodisk scattering',
        parameters=[ParameterSpec('r_disk', '50*nm', 'disk radius')],
        domain=DomainSpec(size_x=1e-6, size_y=1e-6),
        geometry=[GeometrySpec('disk', 'circle', {'cx': 0.5e-6, 'cy': 0.5e-6, 'r': 'r_disk'}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': 0.35e-6, 'y0': 0.35e-6, 'x1': 0.65e-6, 'y1': 0.65e-6}),
                  GeometrySpec('near_field_box', 'rectangle',
                               {'x0': 0.4e-6, 'y0': 0.4e-6, 'x1': 0.6e-6, 'y1': 0.6e-6}),
                  GeometrySpec('far_field_box', 'rectangle',
                               {'x0': 0.25e-6, 'y0': 0.25e-6, 'x1': 0.75e-6, 'y1': 0.75e-6})],
        materials=[MaterialSpec('gold', library='gold', selection=['disk'])],
        physics=[PhysicsSpec('em', 'em_fdtd_ez', features=[
            FeatureSpec('plane_wave', 'plane_wave_tfsf', ['tfsf_box'],
                        {'waveform': 'gaussian', 'wavelength': 550e-9, 'bandwidth': 0.8, 'amplitude': 1.0}),
            FeatureSpec('near_field', 'dft_monitor', ['near_field_box'], dict(spectral)),
            FeatureSpec('far_field', 'far_field', ['far_field_box'], dict(spectral, n_angles=360))])],
        meshes=[MeshSpec(mode='uniform', dx=2e-9)],
        study=StudySpec(type='time_dependent', settings={'duration': 60e-15}))


def _joule_heating():
    """CW gaussian beam heating a gold disk in silica: EM -> heat (Joule) and heat -> EM (thermo-optic)."""
    return ModelSpec(
        title='Joule heating of a gold disk',
        parameters=[ParameterSpec('E0', 1e6, 'beam peak field at the waist [V/m]'),
                    ParameterSpec('r_disk', '100*nm', 'disk radius')],
        domain=DomainSpec(size_x=2e-6, size_y=2e-6),
        geometry=[GeometrySpec('disk', 'circle', {'cx': 1e-6, 'cy': 1e-6, 'r': 'r_disk'}),
                  GeometrySpec('beam_line', 'line', {'x0': 0.2e-6, 'y0': 0.4e-6, 'x1': 1.8e-6, 'y1': 0.4e-6}),
                  GeometrySpec('disk_center', 'point', {'x': 1e-6, 'y': 1e-6})],
        materials=[MaterialSpec('substrate', library='silica', selection=['all']),
                   MaterialSpec('gold', library='gold', selection=['disk'])],
        physics=[PhysicsSpec('em', 'em_fdtd_ez', settings={'track_absorption': True}, features=[
                     FeatureSpec('beam', 'gaussian_beam', ['beam_line'],
                                 {'waveform': 'cw', 'wavelength': 532e-9, 'amplitude': 'E0', 'waist': 0.4e-6})]),
                 PhysicsSpec('heat', 'heat', every=500, features=[
                     FeatureSpec('walls', 'temperature', list(OUTER_FACES), {'temperature': 293.15}),
                     FeatureSpec('disk_temperature', 'temperature_probe', ['disk_center'])])],
        couplings=[CouplingSpec('joule', 'em_heat_joule', 'em', 'heat'),
                   CouplingSpec('thermo_optic', 'heat_em_thermo_optic', 'heat', 'em')],
        meshes=[MeshSpec(mode='uniform', dx=10e-9)],
        study=StudySpec(type='time_dependent', settings={'duration': 300e-15}),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max'),
                        DerivedValueSpec('T_disk', 'T', ['disk'], 'mean'),
                        # heat Q = last window-averaged J.E power (em absorbed_power shows that window after a run)
                        DerivedValueSpec('P_absorbed', 'Q', ['disk'], 'integral')])


def _transient_heat_sink():
    """Transient heating of a silicon block: heater, convection on top, fixed temperature at the bottom."""
    return ModelSpec(
        title='Transient heat sink',
        parameters=[ParameterSpec('Q0', 5e10, 'heater power density [W/m^3]'),
                    ParameterSpec('h_top', 1e4, 'top-face heat transfer coefficient [W/(m^2 K)]')],
        domain=DomainSpec(size_x=2e-3, size_y=1e-3),
        geometry=[GeometrySpec('heater', 'rectangle', {'x0': 0.8e-3, 'y0': 0.84e-3, 'x1': 1.2e-3, 'y1': 0.94e-3}),
                  GeometrySpec('probe_point', 'point', {'x': 1.0e-3, 'y': 0.9e-3})],
        materials=[MaterialSpec('silicon', library='silicon', selection=['all'])],
        physics=[PhysicsSpec('heat', 'heat', settings={'time_mode': 'transient', 'time_step': 1e-4}, features=[
            FeatureSpec('heat_source', 'heat_source', ['heater'], {'Q': 'Q0'}),
            FeatureSpec('top_cooling', 'convection', ['y_hi'], {'h': 'h_top', 'T_ext': 293.15}),
            FeatureSpec('heat_sink', 'temperature', ['y_lo'], {'temperature': 293.15}),
            FeatureSpec('heater_temperature', 'temperature_probe', ['probe_point'])])],
        meshes=[MeshSpec(mode='uniform', dx=20e-6)],
        study=StudySpec(type='time_dependent', settings={'duration': 0.05, 'output_interval': 0.01}),
        derived_values=[DerivedValueSpec('T_heater_max', 'T', ['heater'], 'max')])


def _nonlinear_conduction():
    """Stationary conduction with temperature-dependent k = k_300*(300/T)**1.3 around a heated core."""
    return ModelSpec(
        title='Nonlinear heat conduction',
        parameters=[ParameterSpec('k_300', 130.0, 'thermal conductivity at 300 K [W/(m K)]'),
                    ParameterSpec('Q0', 1e12, 'core power density [W/m^3]')],
        domain=DomainSpec(size_x=1e-3, size_y=1e-3),
        geometry=[GeometrySpec('core', 'circle', {'cx': 0.5e-3, 'cy': 0.5e-3, 'r': 0.1e-3})],
        materials=[MaterialSpec('silicon', library='silicon', properties={'k': 'k_300*(300/T)**1.3'},
                                selection=['all'])],
        physics=[PhysicsSpec('heat', 'heat', settings={'T0': 300.0}, features=[
            FeatureSpec('heater', 'heat_source', ['core'], {'Q': 'Q0'}),
            FeatureSpec('walls', 'temperature', list(OUTER_FACES), {'temperature': 300.0})])],
        meshes=[MeshSpec(mode='uniform', dx=10e-6)],
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max'),
                        DerivedValueSpec('T_mean', 'T', [], 'mean')])


BUILTIN_TEMPLATES = {entry.key: entry for entry in (
    TemplateEntry(
        'dielectric_cylinder', 'Dielectric cylinder',
        'EM: a 550 nm gaussian point source scatters off a glass cylinder; field probe and transmitted flux spectrum.',
        _dielectric_cylinder, builtin=True,
        summary='Light pulse from a point source at 550 nm scattering off a glass cylinder. Field at a probe point '
                'and transmitted power spectrum.'),
    TemplateEntry(
        'gold_nanodisk', 'Gold nanodisk',
        'EM: TFSF plane wave on a Drude gold disk (uniform 2 nm mesh); near-field DFT and far-field pattern, '
        '400-900 nm.',
        _gold_nanodisk, builtin=True,
        summary='Plane-wave light pulse, electric field out of the plane, scattering off a gold nanodisk. Near-field '
                'intensity and far-field scattering pattern, 400-900 nm.'),
    TemplateEntry(
        'joule_heating', 'Joule heating',
        'EM + heat: a CW gaussian beam heats a gold disk in silica; Joule and thermo-optic couplings, '
        '293.15 K walls.',
        _joule_heating, builtin=True,
        summary='Continuous 532 nm laser beam heating a gold disk in silica, in time. The absorbed light heats the '
                'disk, and the temperature changes the refractive index in return.'),
    TemplateEntry(
        'transient_heat_sink', 'Transient heat sink',
        'Heat (transient): a heater in a silicon block, convection on top, fixed temperature at the bottom.',
        _transient_heat_sink, builtin=True,
        summary='Heat changing in time: a heater in a silicon block, cooled by convection on top and held at a fixed '
                'temperature at the bottom.'),
    TemplateEntry(
        'nonlinear_conduction', 'Nonlinear conduction',
        'Heat (stationary): temperature-dependent conductivity k_300*(300/T)**1.3 around a heated core.',
        _nonlinear_conduction, builtin=True,
        summary='Steady heat: a heated core in silicon whose thermal conductivity falls as the temperature rises.'),
)}


def template_entries(registry=None) -> list:
    """Built-in TemplateEntry list, then the registry's (definition modules); ValueError when a registered key
    repeats a built-in one. registry None = get_registry()."""
    registry = get_registry() if registry is None else registry
    registered = registry.registered_templates()
    for entry in registered:
        builtin = BUILTIN_TEMPLATES.get(entry.key)
        if builtin is not None:
            raise ValueError(f"registered template '{entry.key}' ({entry.label}) has the key of the built-in "
                             f"template '{builtin.key}' ({builtin.label}); register it under another key")
    return list(BUILTIN_TEMPLATES.values()) + registered


def make_template(key, registry=None) -> ModelSpec:
    """Fresh ModelSpec of template `key` (built-in or registered); KeyError listing every template key."""
    entries = {entry.key: entry for entry in template_entries(registry)}
    entry = entries.get(key)
    if entry is None:
        raise KeyError(f"unknown template {key!r}; available: {', '.join(entries)}")
    spec = entry.factory()
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"template '{key}': factory returned {type(spec).__name__}, not a ModelSpec")
    return spec
