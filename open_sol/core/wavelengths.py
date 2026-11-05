"""Wavelength-range rule and the 'lambda = <nm> nm' label shared by studies, EM definitions and renderers."""
import numpy as np


def wavelength_range(start, stop, count):
    """numpy vacuum wavelengths [m]: count spaced linearly from start to stop, count = 1 gives their centre."""
    if count == 1:
        return np.array([0.5 * (start + stop)])
    return np.linspace(start, stop, count)


def resolved_wavelengths(values):
    """Wavelengths [m] of resolved frequency_domain study values in solve order: the explicit list when not empty
    (None or [] = the range), else wavelength_range of wavelength_start / wavelength_stop / n_wavelengths."""
    if values['wavelengths']:
        return [float(wavelength) for wavelength in values['wavelengths']]
    return wavelength_range(values['wavelength_start'], values['wavelength_stop'], values['n_wavelengths']).tolist()


def wavelength_label(wavelength):
    """Snapshot / progress / live-view label of a vacuum wavelength [m]: 'lambda = 1550 nm' ('.6g': 0.1 nm steps
    around 1550 nm keep distinct labels, which analytic.snapshot_index matches)."""
    return f'lambda = {wavelength * 1e9:.6g} nm'
