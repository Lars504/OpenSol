"""Runtime of the brillouin_scattering feature (frequency_response): the bound optical fields of every scan position
sampled at the solid's quadrature points, the complex drive columns, the per-frequency FDT records and the result."""
# Status: the constructor below is the FROZEN contract of the feature's apply (definition_brillouin, phase P6), written
# with the math (brillouin.py) and the FEM assembly (fem.elastic_drives) in phase P2; bind / record / reset_records /
# result / summary are the solver integration of phase P3 and raise until it fills them in (no quiet stand-ins).
# Constructor values (all checked here, ValueError naming the value):
# - name: the feature name; optical_physics: the em_fdfd_ez / _hz / _bor physics name; collection_physics: '' = the
#   optical physics; incident_source / collection_source: focused_beam feature names ('' = the only enabled one);
# - detection: brillouin.DETECTIONS; temperature [K] > 0; photoelastic / moving_boundary: bools, at least one True;
# - axial_length [m] > 0 (planar solids only: scale mirror_factor / L; axisymmetric solids ignore it);
# - scan_offsets: (n, 2) focus offsets [m] in position order, brillouin.scan_positions(offsets_x, offsets_y) (x outer,
#   y inner; BOR: (0, dz)); position 1 = row 0 = the shown solution and the reduced-sweep load;
# - reference: brillouin.REFERENCES; fit_peak: bool;
# - element_mask: bool per solid mesh element of the drive selection (None = every solid element);
# - mirror_factor: 2 ** (model mirror planes) (core.mirror.mirror_faces(spec)), >= 1;
# - crystal_regions: ((bool mask per solid mesh element, (6, 6) Voigt p in model axes), ...) (planar only).
import math

import numpy as np

from .brillouin import DETECTIONS, REFERENCES

_P3 = 'is the frequency_response integration of phase P3 (not built yet)'


class BrillouinDrive:
    """One brillouin_scattering feature on a solid: its settings (module header) and, once bound by the study hook,
    the drive columns and per-frequency records."""

    def __init__(self, name, optical_physics, collection_physics, incident_source, collection_source, detection,
                 temperature, photoelastic, moving_boundary, axial_length, scan_offsets, reference, fit_peak,
                 element_mask, mirror_factor, crystal_regions):
        label = f"brillouin_scattering '{name}'"
        if not isinstance(optical_physics, str) or not optical_physics:
            raise ValueError(f'{label}: optical_physics must name an optical physics, got {optical_physics!r}')
        for what, value in (('collection_physics', collection_physics), ('incident_source', incident_source),
                            ('collection_source', collection_source)):
            if not isinstance(value, str):
                raise ValueError(f"{label}: {what} must be a name ('' = the default), got {value!r}")
        if detection not in DETECTIONS:
            raise ValueError(f"{label}: detection must be one of {', '.join(DETECTIONS)}, got {detection!r}")
        if reference not in REFERENCES:
            raise ValueError(f"{label}: reference must be one of {', '.join(REFERENCES)}, got {reference!r}")
        temperature, axial_length, mirror_factor = float(temperature), float(axial_length), float(mirror_factor)
        if not (temperature > 0.0 and math.isfinite(temperature)):
            raise ValueError(f'{label}: temperature must be > 0 K, got {temperature!r}')
        if not (axial_length > 0.0 and math.isfinite(axial_length)):
            raise ValueError(f'{label}: axial_length must be > 0 m, got {axial_length!r}')
        if not (mirror_factor >= 1.0 and math.isfinite(mirror_factor)):
            raise ValueError(f'{label}: mirror_factor must be >= 1 (2 per model mirror plane), got {mirror_factor!r}')
        if not (photoelastic or moving_boundary):
            raise ValueError(f'{label}: photoelastic and moving_boundary are both off: nothing drives the solid')
        offsets = np.array(scan_offsets, dtype=np.float64)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets) or not np.isfinite(offsets).all():
            raise ValueError(f'{label}: scan_offsets must be (n >= 1, 2) finite focus offsets [m] '
                             f'(brillouin.scan_positions), got shape {offsets.shape}')
        offsets.flags.writeable = False
        self.name = name
        self.optical_physics = optical_physics
        self.collection_physics = collection_physics or optical_physics
        self.incident_source = incident_source
        self.collection_source = collection_source
        self.detection = detection
        self.temperature = temperature
        self.photoelastic = bool(photoelastic)
        self.moving_boundary = bool(moving_boundary)
        self.axial_length = axial_length
        self.scan_offsets = offsets
        self.reference = reference
        self.fit_peak = bool(fit_peak)
        self.element_mask = None if element_mask is None else np.array(element_mask, dtype=bool)
        self.mirror_factor = mirror_factor
        self.crystal_regions = tuple((np.array(mask, dtype=bool), np.array(tensor, dtype=np.float64))
                                     for mask, tensor in crystal_regions)
        for mask, tensor in self.crystal_regions:
            if tensor.shape != (6, 6) or mask.ndim != 1:
                raise ValueError(f'{label}: a crystal region is (mask per solid element, (6, 6) p), got '
                                 f'{mask.shape} and {tensor.shape}')

    @property
    def n_positions(self) -> int:
        return len(self.scan_offsets)

    def bind(self, solid, physics_by_name, progress=None):
        raise NotImplementedError(f'BrillouinDrive.bind {_P3}')

    def record(self, frequency, readouts):
        raise NotImplementedError(f'BrillouinDrive.record {_P3}')

    def reset_records(self):
        raise NotImplementedError(f'BrillouinDrive.reset_records {_P3}')

    def result(self):
        raise NotImplementedError(f'BrillouinDrive.result {_P3}')

    def summary(self):
        raise NotImplementedError(f'BrillouinDrive.summary {_P3}')
