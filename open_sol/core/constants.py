"""Physical constants (C0, EPS0, MU0, ETA0, PLANCK, HBAR, K_B, ELEMENTARY_CHARGE, N_A, STEFAN_BOLTZMANN), the
float32 DTYPE and the domain face names FACES / FACES_3D with their GUI labels."""
import numpy as np

DTYPE = np.float32
C0   = 299792458.0
EPS0 = 8.8541878128e-12
MU0  = 1.2566370614e-6
ETA0 = np.sqrt(MU0 / EPS0)
PLANCK = 6.62607015e-34  # Planck constant h [J s]
HBAR = 1.054571817e-34   # reduced Planck constant [J s]
K_B = 1.380649e-23       # Boltzmann constant [J/K]
ELEMENTARY_CHARGE = 1.602176634e-19   # elementary charge [C] (also 1 eV in J)
N_A = 6.02214076e23      # Avogadro constant [1/mol]
STEFAN_BOLTZMANN = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m^2 K^4)]

# Outer faces of the rectangular domain (PML padding, boundary conditions, selections), in this order.
FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi')
FACE_LABELS = dict(zip(FACES, ('Left edge', 'Right edge', 'Bottom edge', 'Top edge')))   # GUI / Param labels
# 3D domains (spec v2 with size_z) add the two z faces; FACES stays the 2D tuple every 2D module iterates.
Z_FACES = ('z_lo', 'z_hi')
FACES_3D = FACES + Z_FACES
FACE_LABELS.update(zip(Z_FACES, ('Lower z face', 'Upper z face')))
