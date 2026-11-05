"""Solvers (solver_plan Part I): roles, SolverRole / SolverType and the role runtimes (pcg, line_block,
sparse_direct). Imports only the pure roles module: runtimes and definition.py are imported by their users."""
from .roles import (BACKENDS, GRID_KINDS, OPERATOR_CLASSES, PRECISIONS, SOLVER_ROLES, ResolvedSolver, SolverRequest,
                    SolverRole, SolverType, resolved_precision, solver_misfit)
