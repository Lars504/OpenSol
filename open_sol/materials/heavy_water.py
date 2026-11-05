"""Library material HEAVY_WATER: D2O at 1.55 um and 20 C with its absorption as sigma_e, thermal, DC (eps_dc) and
viscosity data."""
from ._base import Material


HEAVY_WATER = Material(
    name='heavy_water',
    # n = 1.3170, extinction 4.47e-6 at 1.55 um, 20 C: the absorption (0.36 /cm) enters as sigma_e
    optical=dict(eps_r=1.735, sigma_e=0.127),
    # dndT is an estimate (H2O value shifted by the 7 K higher density maximum); alpha = 0 as for water (liquid)
    thermal=dict(k=0.589, rho=1105.0, cp_=4219.0, dndT=-7.0e-5, alpha=0.0),
    electrical=dict(eps_dc=78.1),
    # dynamic viscosity at 20 C: a fluid in laminar_flow
    fluid=dict(viscosity=1.25e-3),
    description='Heavy water (D2O) at 1.55 um and 20 C (n = 1.3170) with its absorption (extinction 4.47e-6, about '
                '0.36 /cm) as sigma_e, valid only near 1.55 um (sigma_e = 2 n kappa omega eps0) and an optical loss '
                'proxy (no sigma_dc); the index is non-dispersive. dndT is an estimate from the water value; '
                "alpha = 0 because a liquid region is its container's (every measured dndT, solid or liquid, "
                'includes the density change).',
    summary='Heavy water at 1550 nm and 20 degrees Celsius: refractive index 1.317, with its weak absorption.\n\n'
            'The optical values hold only near 1550 nm.',
)
