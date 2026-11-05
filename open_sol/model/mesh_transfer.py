"""S7.2 grid transfers between the meshes of one build: the transfer factory (tensor <-> tensor: core.grid_transfer;
a triangle mesh: fem.transfer through TriangleTransfer, mesh plan stage T2) and the rule of where a cross-mesh
coupling's transform runs."""
import numpy as np

from ..core.backend import array_module_of
from ..core.constants import DTYPE
from ..core.grid_transfer import GridTransfer


def make_transfer(kind, source_domain, target_domain, axisymmetric=False):
    """GridTransfer ('interpolate' | 'conservative') from source_domain to target_domain (one model window, the
    target's backend); axisymmetric: r-weighted conservative measures. A triangle mesh on either side: a
    TriangleTransfer (same apply interface)."""
    kinds = (source_domain.grid_kind, target_domain.grid_kind)
    if kinds == ('tensor', 'tensor'):
        return GridTransfer.between(kind, source_domain, target_domain, axisymmetric)
    return TriangleTransfer(kind, source_domain, target_domain, axisymmetric)


class TriangleTransfer:
    """GridTransfer interface (apply(array, out=None) -> target-shaped float32) when a triangle mesh is involved. A
    triangle target is one value per element (P0: materials, source slots); a triangle source array is read by its
    length as per-element (P0) or full-mesh P1 / P2 coefficients (a solver's field_dofs), or by the dofs given to
    apply. One fem.transfer.FEMTransfer per source space, built on first use."""

    def __init__(self, kind, source_domain, target_domain, axisymmetric=False):
        for domain in (source_domain, target_domain):
            if domain.grid_kind not in ('tensor', 'triangles'):
                raise ValueError(f"no cross-mesh transfers with a {domain.grid_kind} mesh (triangle and tensor meshes "
                                 'only)')
            if domain.grid_kind == 'tensor' and (domain.dimensions != 2 or domain.coordinates.polar):
                raise ValueError('transfers with a triangle mesh need a 2D Cartesian tensor grid on the other side')
        self.kind = kind
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.axisymmetric = bool(axisymmetric) and kind == 'conservative'
        self.source_ndim = 1 if source_domain.grid_kind == 'triangles' else 2   # coupling_term: which state arrays move
        self._transfers = {}   # source space key -> (its DofMap or None, FEMTransfer)

    @property
    def nnz(self) -> int:
        return sum(transfer.nnz for _, transfer in self._transfers.values())

    def _source_dofs(self, length, dofs):
        """(key, DofMap or None for a tensor source) of a source array of `length` values."""
        from ..fem.dofs import lagrange_dofs

        if self.source_domain.grid_kind == 'tensor':
            return 'tensor', None
        if dofs is not None:
            return id(dofs), dofs
        mesh = self.source_domain.mesh
        sizes = {0: mesh.n_elements, 1: mesh.n_points, 2: mesh.n_points + mesh.n_edges}
        orders = [order for order, size in sizes.items() if size == length]
        if len(orders) != 1:
            raise ValueError(f"transfer from triangle mesh '{self.source_domain.mesh_name}': {length} values are "
                             f'{"ambiguous" if orders else "none"} of per element ({sizes[0]}), P1 ({sizes[1]}) or P2 '
                             f'({sizes[2]}) coefficients; pass the dofs')
        return orders[0], lagrange_dofs(mesh, orders[0])

    def _build(self, source_dofs):
        from ..fem import transfer as fem_transfer
        from ..fem.dofs import lagrange_dofs

        source, target = self.source_domain, self.target_domain
        target_dofs = lagrange_dofs(target.mesh, 0) if target.grid_kind == 'triangles' else None
        if source_dofs is None:
            return fem_transfer.tensor_to_triangles(self.kind, *source.cell_edges(), source.physical_window(),
                                                    target_dofs, self.axisymmetric)
        if target_dofs is None:
            return fem_transfer.triangles_to_tensor(self.kind, source_dofs, *target.cell_edges(),
                                                    target.physical_window(), self.axisymmetric)
        return fem_transfer.triangles_to_triangles(self.kind, source_dofs, target_dofs, self.axisymmetric)

    def apply(self, array, out=None, dofs=None):
        """Target-shaped float32 transfer of a source array (real; numpy or cupy), into out when given (returned);
        dofs: the fem DofMap of a triangle source array (None: by its length)."""
        xp = array_module_of(array)
        if xp.iscomplexobj(array):
            raise ValueError('grid transfer: complex arrays are not transferred (real fields and densities only)')
        key, source_dofs = self._source_dofs(int(np.prod(array.shape)), dofs)
        entry = self._transfers.get(key)
        if entry is None:   # the entry keeps its DofMap alive, so an id(dofs) key is never reused by another one
            entry = self._transfers[key] = (source_dofs, self._build(source_dofs))
        values = entry[1].apply(array if array.dtype == DTYPE else array.astype(DTYPE))
        if out is None:
            return values
        out[...] = values
        return out


def transform_side(coupling_type) -> str:
    """'target' when a cross-mesh coupling's transform runs on the target mesh (mesh_transfer 'field'), 'source' for
    'value'; ValueError for a one-mesh type (None)."""
    if coupling_type.mesh_transfer == 'field':
        return 'target'
    if coupling_type.mesh_transfer == 'value':
        return 'source'
    raise ValueError(f"coupling type '{coupling_type.name}' works within one mesh only (mesh_transfer None)")


def transfer_kind(coupling_type) -> str:
    """GridTransfer kind of a cross-mesh coupling: 'interpolate' for 'field', the type's value_kind for 'value'."""
    return 'interpolate' if transform_side(coupling_type) == 'target' else coupling_type.value_kind
