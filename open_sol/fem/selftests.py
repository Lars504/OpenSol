"""Self-test case list of the fem package (registered by model.study.register); later FEM stages append theirs."""
from .selftests_builtin_tetrahedra import BUILTIN_TETRAHEDRA_SELFTEST_CASES
from .selftests_diffusion import DIFFUSION_SELFTEST_CASES
from .selftests_diffusion3d import DIFFUSION_3D_SELFTEST_CASES
from .selftests_display import DISPLAY_SELFTEST_CASES
from .selftests_elasticity3d import ELASTICITY_3D_SELFTEST_CASES
from .selftests_elements import ELEMENT_SELFTEST_CASES
from .selftests_harmonic import HARMONIC_SELFTEST_CASES
from .selftests_line_fem import LINE_FEM_SELFTEST_CASES
from .selftests_mesh_sizes3d import MESH_SIZES_3D_SELFTEST_CASES
from .selftests_mesh_symmetry import MESH_SYMMETRY_SELFTEST_CASES
from .selftests_mesher import MESHER_SELFTEST_CASES
from .selftests_periodic_mesh import PERIODIC_MESH_SELFTEST_CASES
from .selftests_prestress import PRESTRESS_SELFTEST_CASES
from .selftests_reduced_harmonic import REDUCED_HARMONIC_SELFTEST_CASES
from .selftests_refinement_places import REFINEMENT_PLACES_SELFTEST_CASES
from .selftests_resonance_search import RESONANCE_SEARCH_SELFTEST_CASES
from .selftests_rounded_rectangle import ROUNDED_RECTANGLE_SELFTEST_CASES
from .selftests_swept_prisms import SWEPT_PRISMS_SELFTEST_CASES
from .selftests_tangent_assembler import TANGENT_ASSEMBLER_SELFTEST_CASES
from .selftests_tetrahedra import TETRAHEDRA_SELFTEST_CASES

FEM_SELFTEST_CASES = (MESHER_SELFTEST_CASES + ELEMENT_SELFTEST_CASES + DIFFUSION_SELFTEST_CASES
                      + DISPLAY_SELFTEST_CASES + TETRAHEDRA_SELFTEST_CASES + DIFFUSION_3D_SELFTEST_CASES
                      + ELASTICITY_3D_SELFTEST_CASES + HARMONIC_SELFTEST_CASES + LINE_FEM_SELFTEST_CASES
                      + PRESTRESS_SELFTEST_CASES + MESH_SYMMETRY_SELFTEST_CASES + MESH_SIZES_3D_SELFTEST_CASES
                      + PERIODIC_MESH_SELFTEST_CASES + ROUNDED_RECTANGLE_SELFTEST_CASES
                      + BUILTIN_TETRAHEDRA_SELFTEST_CASES + REDUCED_HARMONIC_SELFTEST_CASES
                      + TANGENT_ASSEMBLER_SELFTEST_CASES + RESONANCE_SEARCH_SELFTEST_CASES
                      + REFINEMENT_PLACES_SELFTEST_CASES + SWEPT_PRISMS_SELFTEST_CASES)
