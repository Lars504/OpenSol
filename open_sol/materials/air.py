"""Library material AIR: room-temperature air with vacuum optics (eps_r = 1), heat conduction without convection and no
DC conductivity."""
from ._base import Material


AIR = Material(
    name='air',
    optical=dict(eps_r=1.0),
    thermal=dict(k=0.026, rho=1.225, cp_=1005.0, dndT=0.0, alpha=0.0),
    electrical=dict(eps_dc=1.0),
    description='Air at room temperature with vacuum optics; heat conduction only (no convection inside the gas: '
                'use a convection boundary feature).',
    summary='Air at room temperature: refractive index 1, an electrical insulator.\n\n'
            'It conducts heat but has no convection inside it: use a convection boundary for convective cooling.',
)
