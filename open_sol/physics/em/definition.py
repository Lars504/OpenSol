"""Model definitions of the EM physics types (registry-discovered): FDTD em_fdtd_ez / _hz (definition_fdtd), FDFD
em_fdfd_ez / _hz (definition_fdfd), cross-section em_modes (definition_modes), body-of-revolution em_fdfd_bor
(definition_bor) and em_fdtd_bor (definition_fdtd_bor), conical k_z em_fdfd_conical (definition_conical), on shared
helpers (definition_common), with their templates and self-tests."""
from .definition_bloch import register as register_bloch
from .definition_bor import EM_FDFD_BOR, register as register_bor
from .definition_common import DISPERSION_POLE_PARAMS
from .definition_conical import EM_FDFD_CONICAL, register as register_conical
from .definition_fdfd import EM_FDFD_EZ, EM_FDFD_HZ
from .definition_fdtd import EM_FDTD_EZ, EM_FDTD_HZ
from .definition_fdtd_bor import EM_FDTD_BOR, register as register_fdtd_bor
from .definition_modes import EM_MODES, register as register_modes
from .definition_nonlinear import register as register_nonlinear
from .definition_phase_change import register as register_phase_change
from .definition_resonance import register as register_resonance
from .selftests import fresnel_interface_spec
from .selftests_effective_index import RIB_TEMPLATE_DESCRIPTION, RIB_TEMPLATE_LABEL, rib_waveguide_spec
from .selftests_eigenfrequency import (DISK_EIGENMODES_DESCRIPTION, DISK_EIGENMODES_LABEL, DISK_EIGENMODES_SUMMARY,
                                       disk_resonator_eigenmodes_spec)
from .selftests_fdfd import FD_JOULE_HEATING_DESCRIPTION, FD_JOULE_HEATING_LABEL, fd_joule_heating_spec
from .selftests_harmonic_fdtd import (CYLINDER_HARMONIC_DESCRIPTION, CYLINDER_HARMONIC_LABEL,
                                      CYLINDER_HARMONIC_SUMMARY, cylinder_harmonic_fdtd_spec)
from .selftests_hz import GOLD_NANODISK_HZ_DESCRIPTION, GOLD_NANODISK_HZ_LABEL, gold_nanodisk_hz_spec
from .selftests_mode_ports import DIRECTIONAL_COUPLER_DESCRIPTION, DIRECTIONAL_COUPLER_LABEL, directional_coupler_spec
from .selftests_optimization import BEND_DESCRIPTION, BEND_LABEL, BEND_SUMMARY, waveguide_bend_optimization_spec

__all__ = ['DISPERSION_POLE_PARAMS', 'EM_FDFD_BOR', 'EM_FDFD_CONICAL', 'EM_FDFD_EZ', 'EM_FDFD_HZ', 'EM_FDTD_BOR',
           'EM_FDTD_EZ', 'EM_FDTD_HZ', 'EM_MODES', 'register']


def _selftest_cases():
    """The FDTD / FDFD SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_adjoint import ADJOINT_SELFTEST_CASES
    from .selftests_adjoint_backend import ADJOINT_BACKEND_SELFTEST_CASES
    from .selftests_anisotropy import ANISOTROPY_SELFTEST_CASES
    from .selftests_backend import FDTD_BACKEND_SELFTEST_CASES
    from .selftests_bend import BEND_SELFTEST_CASES
    from .selftests_conducting_sheet import CONDUCTING_SHEET_SELFTEST_CASES
    from .selftests_conformal_poles import CONFORMAL_POLE_SELFTEST_CASES
    from .selftests_effective_index import EFFECTIVE_INDEX_SELFTEST_CASES
    from .selftests_eigenfrequency import EIGENFREQUENCY_SELFTEST_CASES
    from .selftests_fdfd import FDFD_SELFTEST_CASES
    from .selftests_fdfd_backend import FDFD_BACKEND_SELFTEST_CASES
    from .selftests_fdfd_conformal import FDFD_CONFORMAL_SELFTEST_CASES
    from .selftests_fdfd_fem import FDFD_FEM_SELFTESTS
    from .selftests_harmonic_fdtd import HARMONIC_FDTD_SELFTEST_CASES
    from .selftests_hz import HZ_SELFTEST_CASES
    from .selftests_material_fit import MATERIAL_FIT_SELFTEST_CASES
    from .selftests_mode_ports import MODE_PORT_SELFTEST_CASES
    from .selftests_modes import MODE_SELFTEST_CASES
    from .selftests_operators import OPERATOR_SELFTEST_CASES
    from .selftests_optimization import OPTIMIZATION_SELFTEST_CASES
    from .selftests_polar import POLAR_SELFTEST_CASES
    from .selftests_precision import PRECISION_SELFTEST_CASES
    from .selftests_resonance_region import RESONANCE_REGION_SELFTEST_CASES
    from .selftests_symmetry import SYMMETRY_SELFTEST_CASES
    return [case for cases in (
        SELFTEST_CASES, OPERATOR_SELFTEST_CASES, MODE_SELFTEST_CASES, HZ_SELFTEST_CASES, MODE_PORT_SELFTEST_CASES,
        FDFD_SELFTEST_CASES, FDFD_BACKEND_SELFTEST_CASES, EIGENFREQUENCY_SELFTEST_CASES, FDFD_CONFORMAL_SELFTEST_CASES,
        CONFORMAL_POLE_SELFTEST_CASES, HARMONIC_FDTD_SELFTEST_CASES, ADJOINT_SELFTEST_CASES, OPTIMIZATION_SELFTEST_CASES,
        ADJOINT_BACKEND_SELFTEST_CASES, MATERIAL_FIT_SELFTEST_CASES, EFFECTIVE_INDEX_SELFTEST_CASES,
        CONDUCTING_SHEET_SELFTEST_CASES, BEND_SELFTEST_CASES, SYMMETRY_SELFTEST_CASES, ANISOTROPY_SELFTEST_CASES,
        FDTD_BACKEND_SELFTEST_CASES, PRECISION_SELFTEST_CASES, POLAR_SELFTEST_CASES, FDFD_FEM_SELFTESTS,
        RESONANCE_REGION_SELFTEST_CASES)
        for case in cases]


def register(registry):
    """Register the FDTD then the FDFD types (FDTD material Params stay first per name), the eight templates and the
    FDTD / FDFD selftest loader (_selftest_cases), then
    em_modes (definition_modes.register),
    em_fdfd_bor (definition_bor.register), em_fdtd_bor (definition_fdtd_bor.register) and em_fdfd_conical
    (definition_conical.register)."""
    # em_modes last, so the FDTD material Params stay the first per name (Registry.material_property_params)
    registry.register_physics(EM_FDTD_EZ)
    registry.register_physics(EM_FDTD_HZ)
    registry.register_physics(EM_FDFD_EZ)
    registry.register_physics(EM_FDFD_HZ)
    registry.register_template(
        'fresnel_interface', 'Fresnel interface',
        'EM: a 1 um gaussian pulse crosses a vacuum / glass (eps_r = 4) interface in a y-periodic strip; flux '
        'spectrum 0.9-1.1 um behind it (8/9 of a vacuum run).',
        fresnel_interface_spec,
        summary='Light pulse at 1000 nm crossing from vacuum into glass of refractive index 2. Transmitted power '
                'spectrum, 900-1100 nm: 8/9 of the incoming power.')
    registry.register_template(
        'gold_nanodisk_hz', GOLD_NANODISK_HZ_LABEL, GOLD_NANODISK_HZ_DESCRIPTION, gold_nanodisk_hz_spec,
        summary='Gold nanodisk lit by a plane-wave pulse with its electric field in the plane. Plasmon near-field '
                'intensity and far-field pattern, 400-900 nm.')
    registry.register_template(
        'directional_coupler', DIRECTIONAL_COUPLER_LABEL, DIRECTIONAL_COUPLER_DESCRIPTION, directional_coupler_spec,
        summary='Two silicon waveguides in silica with a 4 micrometre long coupling section, fed by a guided mode at '
                '1550 nm. Through, cross and reflected power and phase, 1450-1650 nm.')
    registry.register_template(
        'fd_joule_heating', FD_JOULE_HEATING_LABEL, FD_JOULE_HEATING_DESCRIPTION, fd_joule_heating_spec,
        summary='A 530 nm beam heats a 30 nm gold nanowire in water by about 9 K, and the temperature changes the '
                'refractive index in return. Frequency-domain light, steady heat.')
    registry.register_template(
        'effective_index_rib_waveguide', RIB_TEMPLATE_LABEL, RIB_TEMPLATE_DESCRIPTION, rib_waveguide_spec,
        summary='Top view of a silicon rib waveguide on a chip: each region gets the refractive index of its layer '
                'stack, then a guided mode at 1550 nm travels along the rib.')
    registry.register_template('disk_resonator_eigenmodes', DISK_EIGENMODES_LABEL, DISK_EIGENMODES_DESCRIPTION,
                               disk_resonator_eigenmodes_spec, summary=DISK_EIGENMODES_SUMMARY)
    registry.register_template('cylinder_harmonic_fdtd', CYLINDER_HARMONIC_LABEL, CYLINDER_HARMONIC_DESCRIPTION,
                               cylinder_harmonic_fdtd_spec, summary=CYLINDER_HARMONIC_SUMMARY)
    registry.register_template('waveguide_bend_optimization', BEND_LABEL, BEND_DESCRIPTION,
                               waveguide_bend_optimization_spec, summary=BEND_SUMMARY)
    registry.register_selftests(_selftest_cases)
    register_resonance(registry)   # resonance_monitor: 'disk_resonator' template + its self-tests
    register_bloch(registry)       # Bloch k (A6): 'bloch_band_diagram' template + oblique Fresnel self-tests
    register_nonlinear(registry)   # nonlinear_medium (C5): 'nonlinear_silicon_slab' template + Kerr / TPA self-tests
    register_phase_change(registry)   # phase_change_medium (C5): 'phase_change_gst_switch' template + self-tests
    register_modes(registry)
    register_bor(registry)         # em_fdfd_bor (symmetry 2.3): 'microdisk_bor_eigenmodes' template + BOR self-tests
    register_fdtd_bor(registry)    # em_fdtd_bor (symmetry 2.3): its self-tests
    register_conical(registry)     # em_fdfd_conical (symmetry 2.9): conical CPU and model-run self-tests
