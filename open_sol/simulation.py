"""Simulation scripting API: a Domain with named physics solvers and couplings stepped by one MultiphysicsRunner
(Simulation(size_x, size_y, dx) = the single 2D EM-FDTD convenience path)."""
import math
import sys
import time as _time
import warnings
from types import MappingProxyType

import numpy as np

from .core import gpu_memory
from .core.domain import Domain
from .physics.em import EMPhysics, BoundaryConfig
from .coupling.runner import MultiphysicsRunner
from .viz import Visualizer


class Simulation:
    """Domain + named physics + couplings driven by one MultiphysicsRunner; single-physics attrs fall through."""

    def __init__(self, size_x: float = None, size_y: float = None, dx: float = None, npml: int = 15,
                 courant: float = None, boundaries=None, *, polarization: str = 'ez'):
        """courant = fraction of the 2D CFL limit in (0, 1] (default 0.99, as the model's courant_factor), never
        C0 dt / dx; run() finalizes every physics once (Courant check, mode-port warnings, thermo-optic n0)."""
        if size_x is None or size_y is None or dx is None:
            raise ValueError("Simulation() needs (size_x, size_y, dx, ...) for the single-physics EM-FDTD convenience "
                             "path; use Simulation.multiphysics(domain) for the multi-physics path.")

        bc = _resolve_em_boundary(boundaries, npml)
        domain = Domain.uniform(size_x, size_y, dx, npml=npml, courant=courant, boundary_config=bc)
        self._init_common(domain)
        self.add_physics('em', EMPhysics(domain, polarization=polarization, boundary_config=bc))

    @classmethod
    def from_mesh(cls, mesh_config, npml: int = 15, courant: float = None, boundaries=None, polarization: str = 'ez'):
        bc = _resolve_em_boundary(boundaries, npml)
        domain = Domain.from_mesh(mesh_config, npml=npml, boundary_config=bc, courant=courant)
        sim = cls.multiphysics(domain)
        sim.add_physics('em', EMPhysics(domain, polarization=polarization, boundary_config=bc))
        return sim

    @classmethod
    def multiphysics(cls, domains):
        """Empty Simulation on one Domain, or a list of them (per-physics meshes, S7.2: one window, its own cells
        each; sim.domain = the first, the default of add_physics)."""
        sim = cls.__new__(cls)
        sim._init_common(domains)
        return sim

    def _init_common(self, domains):
        self.domains = list(domains) if isinstance(domains, (list, tuple)) else [domains]
        if not self.domains:
            raise ValueError('Simulation needs at least one Domain')
        self.domain = self.domains[0]
        self._physics_domains = {}   # physics name -> its Domain
        self.runner = MultiphysicsRunner()
        self.callbacks = []
        self._physics_by_name = {}
        self._finalized_names = set()   # physics finalized by finalize() (run() or model.builder)
        self.viz = Visualizer(self)

    # ------------------------------------------------------------- accessors

    @property
    def grid(self):
        return self.domain.grid

    @property
    def materials(self):
        return self.domain.materials

    @property
    def physics(self):
        """Read-only name -> solver mapping."""
        return MappingProxyType(self._physics_by_name)

    # ---------------------------------------------------------- multi-physics

    def add_physics(self, name: str, solver, every: int = 1, domain=None):
        """Attach solver on `domain` (one of sim.domains; None: the solver's own `domain` when it is one, else
        sim.domain)."""
        if domain is None:
            own = getattr(solver, 'domain', None)
            domain = own if any(own is candidate for candidate in self.domains) else self.domain
        elif not any(domain is candidate for candidate in self.domains):
            raise ValueError(f"physics '{name}': its domain is not one of the simulation's domains")
        grid = domain.coordinates.grid
        if grid not in solver.coordinate_grids:
            raise ValueError(f"physics '{name}' ({type(solver).__name__}) does not solve on a {grid} grid (domain "
                             f"coordinates); it supports {', '.join(solver.coordinate_grids)}")
        self.runner.add(name, solver, every=every)
        self._physics_by_name[name] = solver
        self._physics_domains[name] = domain
        return solver

    def domain_of(self, name: str):
        """The Domain physics `name` runs on (S7.2: per-physics meshes)."""
        return self._physics_domains[name]

    def add_coupling(self, coupling):
        self.runner.add_coupling(coupling)
        return coupling

    def add_material(self, mask, material=None, dispersion=None, domain=None, **properties):
        """Paint library `material` (name, Material or None) + model-vocabulary `properties` (eps_r, k, sigma_dc, ...)
        on a full-grid cupy bool mask (domain.rect_mask / circle_mask) of `domain` (None: sim.domain; only its
        physics see it) with the builder's rules; later calls win cells."""
        from .materials import Material, effective_material_data
        from .model.builder import merged_property_params, owned_property_writes
        from .model.schema import get_registry
        from .physics.em.definition_common import background_permittivity, make_pole
        if self.domain is None:
            raise RuntimeError('simulation cleared')
        domain = self.domain if domain is None else domain
        materials = domain.materials
        on_domain = {name: solver for name, solver in self._physics_by_name.items()
                     if self._physics_domains[name] is domain}
        if not (isinstance(mask, domain.xp.ndarray) and mask.dtype == np.bool_
                and mask.shape == (materials.nx, materials.ny)):
            raise ValueError(f'add_material needs a bool backend mask of the full grid {(materials.nx, materials.ny)} '
                             f'(domain.rect_mask / circle_mask), got {type(mask).__name__} '
                             f'{getattr(mask, "dtype", "")} {getattr(mask, "shape", "")}')
        registry = get_registry()
        known_names = {param.name for physics_type in registry.physics_types()
                       for param in physics_type.material_properties}
        unknown = sorted(set(properties) - known_names)
        if unknown:
            raise ValueError(f'unknown material properties {unknown}; valid: {", ".join(sorted(known_names))}')
        overrides = {name: value for name, value in properties.items() if value is not None}
        if isinstance(material, Material):
            merged, poles = effective_material_data(
                None, {**material.model_properties(), **overrides},
                material.dispersion_models() if dispersion is None else dispersion)
        else:
            merged, poles = effective_material_data(material, overrides, dispersion)
        table_names = sorted({param.name for physics_type in registry.physics_types()
                              for param in physics_type.material_properties if param.kind == 'table'}
                             & set(merged))
        if table_names:
            raise ValueError(f'material properties {table_names} are tables, which add_material does not write: use '
                             "the physics' own method (magnetostatics: add_bh_curve)")
        for name, value in merged.items():
            if isinstance(value, str):
                raise ValueError(f'material property {name}={value!r}: scripts take numbers or full-grid arrays; '
                                 f'expressions need a model (open_sol.build)')

        # present physics types = those whose material arrays are all registered (the builder's merge and ownership)
        array_names = set(materials.property_names())
        property_params = merged_property_params(
            physics_type for physics_type in registry.physics_types()
            if physics_type.material_schema and set(physics_type.material_schema) <= array_names)
        dispersive_solvers = [solver for solver in on_domain.values() if hasattr(solver, 'add_dispersive')]
        for name, solver in on_domain.items():
            if any(_pole_cells_overlap(box, cropped_mask, mask)
                   for _, box, cropped_mask in getattr(solver, 'dispersive_regions', ())):
                raise ValueError(f"add_material: the mask overlaps dispersive cells of physics '{name}' (poles are "
                                 f"never removed): paint dispersive materials after the materials they replace")

        written_arrays = []
        for name, _, owned_arrays in owned_property_writes(merged, property_params, array_names):
            for array in owned_arrays:
                materials.assign(array, mask, merged[name])
                written_arrays.append(array)
        if poles and dispersive_solvers:
            dispersive_poles = [make_pole(pole, float) for pole in poles]
            params = tuple(property_params.values())
            for array in dict.fromkeys(array for solver in dispersive_solvers for array in solver.permittivity_arrays):
                materials.assign(array, mask, background_permittivity(merged, poles, params, array))
                written_arrays.append(array)
            for solver in dispersive_solvers:
                for pole in dispersive_poles:
                    solver.add_dispersive(pole, mask)
        if written_arrays:
            changed_names = tuple(dict.fromkeys(written_arrays))
            for solver in on_domain.values():
                solver.materials_changed(changed_names)

    def finalize(self, names=None) -> dict:
        """finalize() the named physics (default: every one not finalized yet) in add order; returns {name:
        finalize_warnings}. Call it again by name after changing materials of a finalized physics."""
        if names is None:
            names = [name for name in self._physics_by_name if name not in self._finalized_names]
        warnings_by_name = {}
        for name in names:
            solver = self._physics_by_name[name]
            solver.finalize()
            self._finalized_names.add(name)
            warnings_by_name[name] = list(getattr(solver, 'finalize_warnings', ()))
        return warnings_by_name

    def clear(self):
        """Drop the solvers, couplings and material arrays (breaks the mode-port physics cycles, which otherwise wait
        for the cyclic gc); a cleared Simulation is dead (Session.close calls this)."""
        self.runner.clear()
        self._physics_by_name.clear()
        self._physics_domains.clear()
        self._finalized_names.clear()
        self.callbacks = []
        self.viz = None
        self.domain = None
        self.domains = []

    # ------------------------------------------------------------- GPU memory

    def memory_report(self, file=None) -> dict:
        """Print (to file, default stdout) and return the memory status, the run-backend array bytes (cupy on 'gpu',
        numpy on 'cpu') owned by the materials (per domain: 'materials <mesh>' when several), each physics and the
        couplings, and host bytes of CPU factorisations."""
        if self.domain is None:
            raise RuntimeError('simulation cleared')
        backend = self.domain.backend.name
        status = gpu_memory.memory_status() if backend == 'gpu' else gpu_memory.cpu_memory_status()
        seen = set()   # allocations already counted (a shared array counts for its first owner)
        owners = {}
        for index, domain in enumerate(self.domains):
            label = 'materials' if len(self.domains) == 1 else f"materials {domain.mesh_name or index + 1}"
            owners[label] = gpu_memory.owned_bytes(domain.materials, seen, backend=backend)
        for name, solver in self._physics_by_name.items():
            owner_key = f'physics {name}' if name in owners or name == 'couplings' else name
            owners[owner_key] = gpu_memory.owned_bytes(solver, seen, backend=backend)
        owners['couplings'] = gpu_memory.owned_bytes(list(self.runner.couplings), seen, backend=backend)
        # Host memory outside the arrays: the factors of the last scipy_splu factorisation (None before it and for
        # back ends that keep nothing on the host).
        host_owners = {name: solver.host_factor_bytes for name, solver in self._physics_by_name.items()
                       if getattr(solver, 'host_factor_bytes', None) is not None}
        if backend == 'gpu':
            budget = 'not set' if status.budget is None else gpu_memory.format_bytes(status.budget)
            lines = ['GPU memory (pool counters cover the whole process)',
                     f'  pool used      {gpu_memory.format_bytes(status.pool_used)}',
                     f'  pool reserved  {gpu_memory.format_bytes(status.pool_reserved)}',
                     f'  other          {gpu_memory.format_bytes(status.other)}',
                     f'  device free    {gpu_memory.format_bytes(status.device_free)}',
                     f'  device total   {gpu_memory.format_bytes(status.device_total)}',
                     f'  budget         {budget}',
                     'Owned by this simulation (cupy arrays)']
        else:
            lines = ['Host memory (cpu backend)',
                     f'  available      {gpu_memory.format_bytes(status.device_free)}',
                     f'  total          {gpu_memory.format_bytes(status.device_total)}',
                     'Owned by this simulation (numpy arrays)']
        lines += [f'  {name:<14} {gpu_memory.format_bytes(size)}' for name, size in owners.items()]
        if host_owners:
            lines.append('Host memory of CPU factorisations')
            lines += [f'  {name:<14} {gpu_memory.format_bytes(size)}' for name, size in host_owners.items()]
        print('\n'.join(lines), file=file)
        return {'status': status.as_dict(), 'owners': owners, 'host_owners': host_owners}

    # --------------------------------------- physics access via __getattr__

    def __getattr__(self, name):
        # Only fires when normal lookup fails. Dunder lookups (copy/pickle probes) never fall through to a physics.
        if name.startswith('__'):
            raise AttributeError(name)
        by_name = self.__dict__.get('_physics_by_name')
        if by_name:
            if name in by_name:
                return by_name[name]
            if len(by_name) == 1:
                sole_name, sole = next(iter(by_name.items()))
                try:
                    return getattr(sole, name)
                except AttributeError:
                    raise AttributeError(f"'Simulation' has no attribute '{name}' and neither has its only physics "
                                         f"'{sole_name}'.") from None
            attached = list(by_name)
            raise AttributeError(f"'Simulation' has no attribute '{name}'. Multiple physics attached ({attached}); "
                                 f"call sim.<name>.{name}(...) explicitly (e.g. sim.{attached[0]}.{name}).")
        raise AttributeError(f"'Simulation' has no attribute '{name}' and no physics are attached.")

    # --------------------------------------------------------------- run loop

    def run(self, duration: float = None, n_steps: int = None, print_every: int = 500, flush: bool = True):
        """finalize() new physics (warns their finalize_warnings), runner.prepare() (warns its prepare_warnings),
        advance by `duration` [s] or `n_steps` ticks, then runner.flush() and warn new runtime_warnings; pass
        flush=False for all but the last chunk of a chunked run (each flush is an extra end-of-run transfer)."""
        if (duration is None) == (n_steps is None):
            raise ValueError("Simulation.run needs exactly one of duration [s] or n_steps")
        if n_steps is not None and int(n_steps) < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        if duration is not None and not float(duration) >= 0:
            raise ValueError(f"duration must be >= 0, got {duration}")

        runner = self.runner
        if not math.isfinite(runner.dt_sync):
            raise ValueError(f"Simulation.run: nothing steps in time (every physics in {list(self._physics_by_name)} "
                             f"is quasi-static, dt_limit inf); call physics.step() or solve_harmonic(omega) directly")
        # First run after add_physics: Courant check, port mode solves, heat n0 from the final materials.
        for name, messages in self.finalize().items():
            for message in messages:
                warnings.warn(f"physics '{name}': {message}", stacklevel=2)
        for message in runner.prepare():   # runner.prepare_warnings (stride cost: the model path validates it)
            warnings.warn(message, stacklevel=2)
        t_end = None if duration is None else runner.t + float(duration)
        tick_budget = sys.maxsize if n_steps is None else int(n_steps)
        span = f"n_steps={tick_budget}" if n_steps is not None else f"duration={float(duration):.4e} s"
        print(f"\nOpen_sol: physics={list(self._physics_by_name)}, "
              f"couplings={len(runner.couplings)}, dt_sync={runner.dt_sync:.4e} s, {span}")

        callbacks = list(self.callbacks)
        runtime_warnings_before = {name: len(getattr(solver, 'runtime_warnings', ()))
                                   for name, solver in self._physics_by_name.items()}
        wall_start = _time.perf_counter()
        ticks_done = 0
        while ticks_done < tick_budget:
            if callbacks:
                chunk = 1
            elif print_every > 0:
                chunk = print_every - ticks_done % print_every
            else:
                chunk = tick_budget - ticks_done
            n_advanced = runner.advance(min(chunk, tick_budget - ticks_done), t_end)
            if n_advanced == 0:
                break
            ticks_done += n_advanced
            for callback in callbacks:
                callback(runner)
            if print_every > 0 and ticks_done % print_every == 0:
                print(f"  tick {runner.tick}, t = {runner.t:.4e} s")
        if flush:
            runner.flush()
        for name, n_before in runtime_warnings_before.items():   # e.g. a mode port re-solved after its materials drifted
            for message in getattr(self._physics_by_name[name], 'runtime_warnings', ())[n_before:]:
                warnings.warn(f"physics '{name}': {message}", stacklevel=2)

        wall = _time.perf_counter() - wall_start
        print(f"Done: {ticks_done} ticks in {wall:.2f} s (t = {runner.t:.4e} s)")

    # -------------------------------------------------- viz convenience shims

    def get_field_snapshot(self, component=None):
        """Renderer field over the physical window (numpy); None = the primary field of the first physics."""
        return self.viz._get_phys_field(component)

    def get_eps_r_snapshot(self):
        """Permittivity over the physical window (numpy); eps_xx on a simulation with only H_z physics."""
        return self.viz._get_phys_eps()

    def plot_geometry(self, filename=None):
        return self.viz.plot_geometry(filename=filename)

    def plot_setup(self, filename=None):
        return self.viz.plot_setup(filename=filename)

    def plot_mesh(self, filename=None):
        return self.viz.plot_mesh(filename=filename)

    def plot_mesh_quality(self, filename=None):
        return self.viz.plot_mesh_quality(filename=filename)

    def plot_field(self, component=None, vmax=None, cmap=None, show_geometry=True, title=None, filename=None,
                   physics=None):
        """Plot a renderer field; component None = the primary field of `physics` (default the first physics)."""
        return self.viz.plot_field(component=component, vmax=vmax, cmap=cmap, show_geometry=show_geometry, title=title,
                                   filename=filename, physics=physics)

    def plot_dashboard(self, filename=None):
        return self.viz.plot_dashboard(filename=filename)

    def mesh_info(self):
        """Print grid size, cell-size ranges, PML padding and the grid's EM time step."""
        domain = self.domain
        grid = domain.grid
        dx_cells, dy_cells = domain.cell_widths()
        window_x, window_y = domain.physical_window()
        _, x_max, _, y_max = domain.physical_extent()
        n_window_x = window_x.stop - window_x.start
        n_window_y = window_y.stop - window_y.start
        kind = 'non-uniform' if domain.non_uniform else 'uniform'
        print(f"\n--- Mesh info ({kind}) ---")
        print(f"  Physical domain: {x_max*1e6:.3f} x {y_max*1e6:.3f} µm, "
              f"{n_window_x} x {n_window_y} = {n_window_x * n_window_y:,} cells")
        print(f"  Total grid: {grid.nx} x {grid.ny} = {grid.nx * grid.ny:,} cells, "
              f"PML cells per side {domain.npml_sides}")
        print(f"  dx: {dx_cells.min()*1e9:.2f}-{dx_cells.max()*1e9:.2f} nm, "
              f"dy: {dy_cells.min()*1e9:.2f}-{dy_cells.max()*1e9:.2f} nm")
        if domain.non_uniform:
            print(f"  Max adjacent ratio: x={grid.max_cell_ratio_x:.3f}, y={grid.max_cell_ratio_y:.3f}")
        print(f"  Grid dt = {float(grid.dt):.4e} s")


def _pole_cells_overlap(box, cropped_mask, mask) -> bool:
    """True when `mask` covers a cell of one dispersive region (box / cropped_mask None = everywhere); one host sync."""
    if box is None:
        return bool(mask.any())
    return bool((mask[box] & cropped_mask).any())


def _resolve_em_boundary(boundaries, npml):
    if boundaries is None:
        return BoundaryConfig.all_pml(npml)
    if isinstance(boundaries, str):
        presets = {'pml': BoundaryConfig.all_pml(npml), 'pec': BoundaryConfig.all_pec(),
                   'pmc': BoundaryConfig.all_pmc(), 'periodic_x': BoundaryConfig.with_periodic_x(npml),
                   'periodic_y': BoundaryConfig.with_periodic_y(npml), 'periodic': BoundaryConfig.periodic_xy()}
        bc = presets.get(boundaries)
        if bc is None:
            raise ValueError(f"Unknown boundary preset: '{boundaries}'. Options: {list(presets.keys())}")
        return bc
    return boundaries
