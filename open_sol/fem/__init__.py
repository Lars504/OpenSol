"""Finite elements on triangle meshes (solver_plan S7.5, mesh_implementation_plan section 2): mesh data, the own
conforming Delaunay mesher, quality, optional gmsh file IO, elements / quadrature / dofs / assembly, sampling; the
tetrahedron stack (mesh3d, assembly3d, diffusion3d, elasticity3d, ...) is imported by module. Host numpy at module
level (scipy inside functions); it imports nothing from model / physics (model.triangle_meshing feeds it the geometry).
domain.py (core MaterialGrid) and transfer.py (core.backend inside apply) are imported by module."""
from .assembly import (DirichletSystem, assemble_curl_curl, assemble_edge_load, assemble_edge_mass,
                       assemble_gradient_coupling, assemble_load, assemble_mass, assemble_stiffness,
                       assemble_vector_mass, eliminate_dirichlet, element_areas, integrate, interpolate, l2_error,
                       restrict)
from .dofs import DofMap, lagrange_dofs, nedelec_dofs
from .mesh import FACE_BITS, PSLG, Curve, TriangleMesh, build_triangle_mesh
from .mesher import RawMesh, SizeField, add_layer_ring, build_pslg, mesh_pslg
from .quality import MeshQuality, mesh_quality
from .sampling import (Location, PointLocator, QuadratureSamples, RasterSampler, display_triangulation, evaluate,
                       locator, quadrature_samples, raster_sampler)

__all__ = ['FACE_BITS', 'PSLG', 'Curve', 'TriangleMesh', 'build_triangle_mesh', 'RawMesh', 'SizeField',
           'add_layer_ring', 'build_pslg', 'mesh_pslg', 'MeshQuality', 'mesh_quality', 'DofMap', 'lagrange_dofs',
           'nedelec_dofs', 'DirichletSystem', 'assemble_curl_curl', 'assemble_edge_load', 'assemble_edge_mass',
           'assemble_gradient_coupling', 'assemble_load', 'assemble_mass', 'assemble_stiffness',
           'assemble_vector_mass', 'eliminate_dirichlet', 'element_areas', 'integrate', 'interpolate', 'l2_error',
           'restrict', 'Location', 'PointLocator', 'QuadratureSamples', 'RasterSampler', 'display_triangulation',
           'evaluate', 'locator', 'quadrature_samples', 'raster_sampler']
