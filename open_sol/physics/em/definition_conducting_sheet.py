"""Model feature 'conducting_sheet' of em_fdfd_ez / em_fdfd_hz (nonlinear_optics_plan stage 0): a zero-thickness 2D
material (graphene, a metal film thinner than a cell) of sheet conductivity sigma_s(omega) on an axis-aligned line."""
import math
from functools import partial

from ...core.constants import C0, ETA0
from ...core.memory_estimate import COMPLEX128_BYTES, FLOAT64_BYTES, MemoryEstimate
from ...model.schema import FeatureType, Issue, Param, feature_path
from .definition_common import mode_line, resolved_solved_wavelengths
from .definition_feature_validation import conducting_sheet_line_issues
from .sheet_conductivity import SHEET_MODELS, UNIVERSAL_SHEET_CONDUCTIVITY, SheetConductivity

FEATURE_NAME = 'conducting_sheet'
_CONSTANT = (('model', ('constant',)),)
_GRAPHENE = (('model', ('graphene_kubo',)),)
_INFO_WAVELENGTHS = 3   # solved wavelengths listed in the sigma info (first, middle, last)

_PARAMS = (
    Param('model', 'choice', 'constant', label='Conductivity model', choices=SHEET_MODELS,
          choice_labels=(('constant', 'Constant'), ('graphene_kubo', 'Graphene (Kubo formula)')),
          help="'constant': sigma_real + j sigma_imag at every solved wavelength; 'graphene_kubo': the Kubo formula of "
               'monolayer graphene per solved wavelength (intraband Drude e^2 tau (mu + 2 kT ln(1 + e^{-mu / kT})) / '
               '(pi hbar^2 (1 + j omega tau)) plus the temperature-smeared interband term, sigma_0 = e^2 / (4 hbar) '
               'above the 2 |E_F| edge; no spatial dispersion, no magnetic field).',
          tip='How the conductivity of the sheet is given.\n\n'
              '- Constant: one complex value at every wavelength.\n'
              '- Graphene: computed for a graphene monolayer from its Fermi energy, scattering time and temperature '
              'at each wavelength.'),
    Param('sigma_real', 'float', UNIVERSAL_SHEET_CONDUCTIVITY, label='Conductivity (real part)', unit='S',
          visible_if=_CONSTANT,
          help='Re sigma_s (> 0 absorbs). Default 6.085e-5 S = the universal optical conductivity of graphene (2.3 % '
               'absorption in vacuum); a metal film of thickness d << skin depth: sigma_s ~ sigma_bulk d.',
          tip='Real part of the sheet conductivity, in siemens: it sets the absorption. 6.1e-5 S is the optical '
              'conductivity of graphene (about 2.3 % absorption).'),
    Param('sigma_imag', 'float', 0.0, label='Conductivity (imaginary part)', unit='S', visible_if=_CONSTANT,
          help='Im sigma_s (exp(+j omega t)): < 0 inductive (free carriers, Drude 1 / (1 + j omega tau)), > 0 '
               'capacitive; Kubo literature in exp(-i omega t) has the opposite sign.',
          tip='Imaginary part of the sheet conductivity, in siemens. Negative for free carriers (inductive, as in '
              'metals and doped graphene), positive for a capacitive response.'),
    Param('fermi_energy', 'float', 0.2, label='Fermi energy', unit='eV', visible_if=_GRAPHENE,
          help='Chemical potential E_F relative to the Dirac point (|E_F| enters): gated or doped graphene 0.1-0.6 eV, '
               'n = (E_F / (hbar v_F))^2 / pi carriers per m^2 (v_F ~ 1e6 m/s). Interband absorption sets in at hbar '
               'omega = 2 |E_F| (wavelength 1.24e-6 / (2 |E_F| / eV) m).',
          tip='Fermi energy of the graphene, in electronvolts, set by doping or a gate: typically 0.1 to 0.6. Light '
              'with photon energy below twice this value is not absorbed by interband transitions.'),
    Param('scattering_time', 'float', 1e-13, label='Scattering time', unit='s', min=0.0, min_exclusive=True,
          visible_if=_GRAPHENE,
          help='Carrier relaxation time tau of the intraband (Drude) term: 1e-14 - 1e-12 s (CVD graphene ~1e-13 s; '
               'mobility e tau v_F^2 / |E_F|); the interband term is unbroadened.',
          tip='Average time between carrier collisions, in seconds: about 1e-13 for typical graphene. Longer times '
              'give sharper plasmon resonances and less absorption at long wavelengths.'),
    Param('temperature', 'float', 300.0, label='Temperature', unit='K', min=0.0, min_exclusive=True,
          visible_if=_GRAPHENE,
          help='Electron temperature of the Fermi functions (intraband weight, interband edge smeared over ~4 kT).',
          tip='Temperature of the electrons, in kelvin. It smooths the onset of interband absorption.'),
)

_DESCRIPTION = (
    'Zero-thickness conducting sheet (graphene, a TMD monolayer in linear response, a metal film much thinner than a '
    'cell) on an axis-aligned line: the tangential E drives K = sigma_s E_t [A/m], the normal E does not conduct. '
    'em_fdfd_ez: E_z of the line cells; the operator diagonal gains -j omega mu0 sigma_s / w (w the line-cell width '
    'across the line), exact for the 3-point stencil. em_fdfd_hz: E_x (horizontal line) / E_y (vertical) on the hi '
    'face of the line cells, whose face permittivity gains -j sigma_s / (omega eps0 w), w the face spacing; the line '
    'may not lie in the last grid cell before a non-periodic outer face. In vacuum at normal incidence it transmits '
    '|1 / (1 + eta0 sigma_s / 2)|^2 (on the grid exactly 1 / |1 + omega mu0 sigma_s w / (2 sin(k w))|^2, k the '
    'discrete wavenumber). Its loss 1/2 Re sigma_s |E_t|^2 per unit sheet area goes into absorbed_power (q_abs) of '
    'the line cells as a density over the line-cell width (em_heat_joule heats them). Only the operator sees it: '
    'line-source and TFSF calibrations, port mode solves and the material maps do not (keep source, port and TFSF '
    'lines off it; enclose it in the TFSF box as a scatterer). A span end on a cpml domain edge continues through the '
    'PML. Sheets on one line add. In an eigenfrequency study sigma_s stays at its target-wavelength value '
    '(first-order error away from it). The line covers cells [cell(start), cell(end) + 1). No result.')
_SUMMARY = ('A conducting sheet of zero thickness along a straight line, such as graphene or a very thin metal film: it '
            'reflects and absorbs light according to its sheet conductivity. Graphene can be computed from its Fermi '
            'energy.')


def sheet_conductivity_from_values(values):
    """SheetConductivity of resolved feature values."""
    if values['model'] == 'constant':
        return SheetConductivity('constant', value=complex(values['sigma_real'], values['sigma_imag']))
    return SheetConductivity('graphene_kubo', fermi_energy=float(values['fermi_energy']),
                             scattering_time=float(values['scattering_time']),
                             temperature=float(values['temperature']))


def _apply(ctx, solver, feature_spec, values, selection):
    axis, position, span = mode_line(ctx, selection, feature_spec)
    return solver.add_conducting_sheet(axis, position, span, conductivity=sheet_conductivity_from_values(values))


def _solved_wavelengths(vctx, physics_spec):
    """Solved wavelengths [m] of the physics (empty when its settings or the study do not resolve)."""
    wavelengths = resolved_solved_wavelengths(vctx, physics_spec)
    return sorted(wavelength for wavelength in wavelengths
                  if isinstance(wavelength, (int, float)) and math.isfinite(wavelength) and wavelength > 0.0)


def _validate(polarization, vctx, physics_spec, feature_spec, values):
    """Line placement (definition_feature_validation.conducting_sheet_line_issues) plus an info with sigma_s and the
    vacuum normal-incidence transmission at up to three solved wavelengths."""
    issues = conducting_sheet_line_issues(vctx, physics_spec, feature_spec, polarization)
    try:
        conductivity = sheet_conductivity_from_values(values)
    except (TypeError, KeyError):
        return issues   # reported by the param checks
    except ValueError as error:
        return issues + [Issue('error', feature_path(physics_spec, feature_spec), str(error))]
    wavelengths = _solved_wavelengths(vctx, physics_spec)
    if len(wavelengths) > _INFO_WAVELENGTHS:
        wavelengths = [wavelengths[0], wavelengths[len(wavelengths) // 2], wavelengths[-1]]
    parts = []
    for wavelength in wavelengths:
        sigma = conductivity(2.0 * math.pi * C0 / wavelength)
        transmission = 1.0 / abs(1.0 + float(ETA0) * sigma / 2.0) ** 2
        parts.append(f'{wavelength:.4g} m: sigma_s = {sigma.real:.4g} {"-" if sigma.imag < 0 else "+"} '
                     f'{abs(sigma.imag):.4g}j S (vacuum normal-incidence transmission {transmission:.4g})')
    if parts:
        issues.append(Issue('info', feature_path(physics_spec, feature_spec), '; '.join(parts)))
    return issues


def _memory_estimate(values, n_selected_cells, mesh):
    """Nothing kept on the device (host scalars and slices); per assembly / solve the line terms and the loss
    temporaries of the line cells (run items)."""
    n_cells = int(n_selected_cells)
    return MemoryEstimate.from_items(
        run_items=(('conducting sheet line terms', n_cells * (2 * COMPLEX128_BYTES + FLOAT64_BYTES)),))


def conducting_sheet_feature(polarization):
    """FeatureType 'conducting_sheet' of em_fdfd_<polarization> ('ez' | 'hz')."""
    return FeatureType(
        FEATURE_NAME, 'Conducting sheet', 'boundary_condition', 'line', params=_PARAMS, object_kinds=('line',),
        max_entities=1, acts_on_hi_face=polarization == 'hz', description=_DESCRIPTION, summary=_SUMMARY,
        apply=_apply, validate=partial(_validate, polarization), memory_estimate=_memory_estimate)
