"""Developer tools of the repository (benchmarks, profilers): never shipped with open_sol, never imported by it."""
import os

# no on-disk gmsh cache in any profiler or its child processes (open_sol's default folder is on for api.run): a profile
# times the mesher, never a disk read (the script-run case profilers set it in profiler_harness too)
os.environ['OPEN_SOL_MESH_CACHE'] = ''
