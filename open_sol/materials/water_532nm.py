"""Library material WATER_532NM: liquid water at 532 nm and 20 C with the hypersonic (GHz) elastic, viscous and
photoelastic data of Brillouin scattering."""
from ._base import Material
from .water import WATER


WATER_532NM = Material(
    name='water_532nm',
    # n = 1.3350 at 532 nm, 20 C; the absorption (~4.5e-4 /cm) is left out: lossless
    optical=dict(eps_r=1.78223, sigma_e=0.0),
    # library water's (a copy, so the two entries cannot drift; alpha = 0: a liquid region follows its container)
    thermal=dict(WATER.thermal),
    # a compressible viscous liquid: no storage shear; bulk_modulus = rho v^2 (adiabatic, v = 1482.3 m/s);
    # bulk_viscosity from the ultrasonic absorption; one strain-optic scalar p11 = p12, no shear part
    mechanical=dict(shear_modulus=0.0, bulk_modulus=2.1933e9, bulk_viscosity=2.8e-3, photoelastic_p11=0.31,
                    photoelastic_p12=0.31, photoelastic_p44=0.0),
    # library water's DC data (a copy)
    electrical=dict(WATER.electrical),
    # dynamic viscosity at 20 C (IAPWS 2008); laminar_flow and the solid types read the same property
    fluid=dict(viscosity=1.0016e-3),
    description='Liquid water at 20 C and 0.1 MPa for spontaneous Brillouin scattering (GHz longitudinal phonons). '
                'Its optical data hold near 532 nm only (library water is the 1.55 um entry). '
                'Optics: n = 1.3350 +- 2e-4 at 20 C (IAPWS refractive-index formulation, Harvey et al., J. Phys. '
                'Chem. Ref. Data 27, 761 (1998); Daimon and Masumura, Appl. Opt. 46, 3811 (2007)); the absorption '
                '(about 4.5e-4 /cm, Pope and Fry, Appl. Opt. 36, 8710 (1997)) is left out, so the medium is lossless '
                '(probe heating is not modelled). '
                'Density as library water (IAPWS-95). bulk_modulus is the adiabatic K_S = rho v^2 of the ultrasonic '
                'sound speed v = 1482.3 m/s (Del Grosso and Mader, J. Acoust. Soc. Am. 52, 1442 (1972); Marczak, J. '
                'Acoust. Soc. Am. 102, 2776 (1997)); the isothermal static modulus is about 0.7 % lower. The '
                'hypersonic (GHz) sound speed is 0.3-0.8 % higher (reported Brillouin shifts of 7.46-7.50 GHz at 532 '
                'nm imply 1487-1494 m/s): UNVERIFIED, a model of the measured shift sets bulk_modulus itself. '
                'viscosity: IAPWS 2008 formulation (Huber et al., J. Phys. Chem. Ref. Data 38, 101 (2009)). '
                'bulk_viscosity from the classical ultrasonic absorption alpha / f^2 (about 25e-15 s^2/m at 20 C; '
                'Pinkerton, Proc. Phys. Soc. B 62, 129 (1949); Fisher and Simmons, J. Acoust. Soc. Am. 62, 558 '
                '(1977)) by 4/3 viscosity + bulk_viscosity = (alpha / f^2) rho v^3 / (2 pi^2); literature values '
                'span 2.4e-3 to 3.0e-3 Pa s (Dukhin and Goetz, J. Chem. Phys. 130, 124519 (2009); Holmes, Parker '
                'and Povey, J. Phys. Conf. Ser. 269, 012011 (2011)): UNVERIFIED digits. Structural relaxation lowers '
                'it at GHz: reported Brillouin linewidths of water at 532 nm (0.55-0.60 GHz FWHM) sit 10-15 % below '
                'the linewidth this ultrasonic value predicts. '
                'Photoelastic: a liquid has one strain-optic scalar (p11 = p12, no shear part p44); Dixon, J. Appl. '
                'Phys. 38, 5149 (1967) (633 nm, from the figure of merit M2), equal to the Lorentz-Lorenz value '
                '(eps - 1)(eps + 2) / (3 eps^2) within 1 %; +- 5 %, its dispersion from 633 to 532 nm neglected. '
                'Thermal and DC data as library water (dndT matches the visible 20 C value too). '
                'Solid mechanics (triangle / tetrahedron meshes): with shear_modulus 0 it is a compressible viscous '
                'liquid of formulation mixed in the frequency_response study only (G* = j omega viscosity, K* = '
                'bulk_modulus + j omega bulk_viscosity); the displacement formulation and every other study refuse '
                'a cell without shear stiffness. The model holds no entropy (thermal diffusion) mode: the Rayleigh '
                'central peak is absent, only the Brillouin doublet is modelled. Validity near 532 nm and 20 C.',
    summary='Liquid water at 532 nanometres and 20 degrees Celsius for Brillouin scattering: refractive index 1.335 '
            'without absorption, with the sound speed, viscosities and strain-optic coefficient that set its '
            'gigahertz Brillouin spectrum.\n\n'
            'The optical values hold only near 532 nm. Solid mechanics treats it as a liquid: use the Displacement '
            'and pressure formulation in a frequency response study.',
)
