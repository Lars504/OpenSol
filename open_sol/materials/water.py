"""Library material WATER: liquid water at 1.55 um and 20 C with its absorption as sigma_e, thermal,
DC (sigma_dc, eps_dc) and viscosity data."""
from ._base import Material


WATER = Material(
    name='water',
    # n = 1.3154, extinction 1.49e-4 at 1.55 um, 20 C: the absorption (12 /cm) enters as sigma_e
    optical=dict(eps_r=1.730, sigma_e=4.22),
    # alpha = 0: a liquid takes the shape of its container, so its region expands with the container
    thermal=dict(k=0.598, rho=998.2, cp_=4182.0, dndT=-9.0e-5, alpha=0.0),
    # pure water: sigma_dc 5.5e-6 S/m (NOT the optical loss proxy sigma_e), static eps 80.1 at 20 C
    electrical=dict(sigma_dc=5.5e-6, eps_dc=80.1),
    # dynamic viscosity at 20 C: a fluid in laminar_flow
    fluid=dict(viscosity=1.002e-3),
    description='Liquid water at 1.55 um and 20 C (n = 1.3154) with its absorption (extinction 1.49e-4, about '
                '12 /cm) as sigma_e. That sigma_e reproduces the absorption only near 1.55 um (sigma_e = 2 n kappa '
                'omega eps0) and is an optical loss proxy, not a conductivity (sigma_dc is pure water); the index is '
                "non-dispersive. alpha = 0 because a liquid region is its container's (every measured dndT, solid "
                'or liquid, includes the density change).',
    summary='Liquid water at 1550 nm and 20 degrees Celsius: refractive index 1.315, with its absorption of about 12 '
            'per centimetre.\n\n'
            'The optical values hold only near 1550 nm.',
)
