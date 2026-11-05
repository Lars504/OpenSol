# Installing open_sol

The steps below assume conda (Anaconda, Miniconda or Miniforge) is installed. They make a separate environment
called `open_sol`, install the required packages, install open_sol from the repository, and check that it works.
GPU support (cupy) is an optional extra step at the end. Without it everything runs on the CPU backend.

Run the commands in an **Anaconda Prompt** (Windows) or a terminal with conda on the PATH (Linux / macOS).

## 1. Create the environment

Make an environment with Python 3.11 (open_sol needs 3.11 or newer; it is developed on 3.11.7) and activate it.

```bash
conda create -n open_sol -c conda-forge python=3.11
```

```bash
conda activate open_sol
```

Activate it again (`conda activate open_sol`) in every new prompt before you use open_sol.

## 2. Install the required packages

numpy, scipy and matplotlib are required (scipy runs the CPU backend, the host direct solvers for FDFD, modes,
eigenfrequencies, drift-diffusion and magnetic AC, and many self-tests; there is no scipy-less fallback). PySide6 and
pyqtgraph run the GUI; leave them out only if you will use just the CLI / Python API.

```bash
conda install -c conda-forge numpy scipy matplotlib pyside6 pyqtgraph
```

Optional extras, all from conda-forge:

- `numba`: faster CPU-backend kernels (open_sol uses it when it imports; set `OPEN_SOL_CPU_KERNELS=numpy` to switch it off).
- `psutil`: more accurate host-memory (RAM) readings for the memory estimates.
- `gmsh`: the external mesher for triangle meshes (open_sol calls the `gmsh` executable; set `OPEN_SOL_GMSH` to its
  path if it is not on the PATH). The built-in mesher needs nothing extra. Set `OPEN_SOL_MESH_CACHE` to a folder to keep
  gmsh tetrahedron meshes on disk across runs (a re-run with the same geometry and sizes skips gmsh; at most 2 GB, the
  least recently used files go first).

```bash
conda install -c conda-forge numba psutil gmsh
```

## 3. Install open_sol

Go to the repository root (the folder holding `pyproject.toml`) and install it in editable mode. Editable (`-e`)
means changes to the code take effect without reinstalling, and the GUI's Examples menu can find `examples/` beside
the package. `--no-deps` stops pip from replacing the conda packages from step 2.

```bash
cd path/to/open_sol
```

```bash
pip install -e . --no-deps
```

Installing is optional: you can also run everything from the repository root with `python -m open_sol ...`.

## 4. Check the installation

A quick CPU check: the package imports, and the built-in checks run (every template and example model validates,
the API reference is current):

```bash
python -m open_sol selftest templates_validate
```

Start the GUI:

```bash
python -m open_sol.gui
```

Without cupy the GUI's Solvers 'Run on' Automatic picks the CPU, and physics types that have no CPU backend are
greyed out / refused by validation. Add `--backend cpu` to `check` / `run` to force the CPU on the CLI.

## 5. Optional: GPU support with cupy

cupy runs open_sol on an NVIDIA GPU. It needs an NVIDIA GPU, a recent NVIDIA driver, and the CUDA libraries
(runtime, NVRTC for the kernels open_sol compiles at run time, cuBLAS / cuSPARSE / cuSOLVER). open_sol is developed
with cupy 13.3.

### 5a. Check the driver

```bash
nvidia-smi
```

The top-right "CUDA Version" is the newest CUDA the driver supports. Pick the cupy CUDA line at or below it:
CUDA 12 for driver 525 or newer (most current setups), CUDA 11 for older drivers. If `nvidia-smi` is not found,
install or update the NVIDIA driver first (nvidia.com/drivers).

### 5b. Install cupy (pick ONE route)

**Route 1 (recommended): conda-forge.** conda installs the CUDA libraries into the environment, so only the driver is
needed, not a separate CUDA Toolkit. Use `cuda-version=11` instead for a CUDA 11 driver.

```bash
conda install -c conda-forge cupy=13.3 cuda-version=12
```

**Route 2: pip wheel with a system CUDA Toolkit.** Install the NVIDIA CUDA Toolkit of the matching major version
first (developer.nvidia.com/cuda-downloads) and make sure `CUDA_PATH` (Windows) or `CUDA_HOME` / `LD_LIBRARY_PATH`
(Linux) point at it. Then install the matching wheel (`cupy-cuda11x` for CUDA 11):

```bash
pip install cupy-cuda12x==13.3.0
```

Do not mix the routes: having both a conda `cupy` and a pip `cupy-cudaXXx` in one environment breaks the import.

### 5c. Check the GPU backend

cupy sees the GPU (prints the number of devices, 1 or more):

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

open_sol sees the GPU (prints True):

```bash
python -c "from open_sol.core.backend import gpu_available; print(gpu_available())"
```

The quick analytic self-tests on the GPU (a few minutes):

```bash
python -m open_sol selftest
```

If cupy fails to import, open_sol still starts on the CPU backend; the import error is shown in the message when a GPU
run is refused. Common causes: a driver older than the cupy CUDA line, both install routes in one environment, or
(route 2) the toolkit not on the path.

### 5d. Optional: cuDSS (nvmath-python)

NVIDIA's cuDSS sparse direct solver factorises on the GPU (solver back ends / factorizations `cudss` and
`cudss_mixed`: heat, electric, magnetic and solid models, the FEM eigensolvers). It comes with nvmath-python; without
it those settings are validation errors and their self-tests pass as skipped. Some examples need it (e.g. the 3D
eigenfrequency examples `em3d/fdfd_mie_sphere_resonance_3d`, `nanobeam_cavity_eighth_3d`,
`fem_mie_sphere_resonance_3d`). Pick the CUDA line of your driver (`[cu13]` for CUDA 13):

```bash
pip install "nvmath-python[cu12]"
```

Check it (prints True):

```bash
python -c "from open_sol.solvers.cudss_factor import cudss_available; print(cudss_available())"
```

## Updating and removing

After `git pull` nothing needs reinstalling (editable install) unless `pyproject.toml` changed; then repeat step 3.

Remove the environment completely:

```bash
conda env remove -n open_sol
```

## Next steps

`README.md` has the quickstarts (template, check, run; the Python API; Simulation scripts). `python -m open_sol
describe templates` lists the template models.
