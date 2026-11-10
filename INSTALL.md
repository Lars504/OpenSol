# Installing open_sol

The steps below assume conda (Anaconda, Miniconda or Miniforge) is installed. They make a separate environment
called `open_sol`, install the packages, install open_sol from the repository, and check that it works.

Run the commands in an **Anaconda Prompt** (Windows) or a terminal with conda on the PATH (Linux / macOS).

## Which packages

| Level | Packages | What it means |
|---|---|---|
| Essential | numpy, scipy, matplotlib (+ PySide6, pyqtgraph for the GUI) | open_sol does not run without them; scipy holds the CPU backend and the host direct solvers (SuperLU, ARPACK), there is no fallback |
| Preferred | numba (CPU); cupy and nvmath-python (NVIDIA GPU) | used on their own once installed (Automatic solvers pick them, nothing to set); without them the same models run, slower or on the CPU only |
| Optional | python-mumps, pydiso (+ threadpoolctl), gmsh, psutil | used only where a solver node or mesh setting selects them (MUMPS / PARDISO, gmsh); psutil only refines a RAM reading |

Install the essential packages (step 2), then the preferred ones your machine can use (step 3, step 6), and an
optional one only when a model asks for it (step 7).

## 1. Create the environment

Make an environment with Python 3.11 (open_sol needs 3.11 or newer; it is developed on 3.11.7) and activate it.

```bash
conda create -n open_sol -c conda-forge python=3.11
```

```bash
conda activate open_sol
```

Activate it again (`conda activate open_sol`) in every new prompt before you use open_sol.

## 2. Essential packages

numpy, scipy and matplotlib are required (scipy runs the CPU backend, the host direct solvers for FDFD, modes,
eigenfrequencies, drift-diffusion and magnetic AC, and many self-tests; there is no scipy-less fallback). PySide6 and
pyqtgraph run the GUI; leave them out only if you will use just the CLI / Python API.

```bash
conda install -c conda-forge numpy scipy matplotlib pyside6 pyqtgraph
```

## 3. Preferred on every machine: numba

numba compiles the CPU-backend kernels (the time steppers' field updates, CPML, losses); open_sol uses it when it
imports, and without it the same kernels run as numpy twins, which give the same results more slowly. Set
`OPEN_SOL_CPU_KERNELS=numpy` to switch it off.

```bash
conda install -c conda-forge numba
```

## 4. Install open_sol

Go to the repository root (the folder holding `pyproject.toml`) and install it in editable mode. Editable (`-e`)
means changes to the code take effect without reinstalling, and the GUI's Examples menu can find `examples/` beside
the package. `--no-deps` stops pip from replacing the conda packages installed above.

```bash
cd path/to/open_sol
```

```bash
pip install -e . --no-deps
```

Installing is optional: you can also run everything from the repository root with `python -m open_sol ...`.

## 5. Check the installation

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

## 6. Preferred with an NVIDIA GPU: cupy and cuDSS

cupy runs open_sol on an NVIDIA GPU. It needs an NVIDIA GPU, a recent NVIDIA driver, and the CUDA libraries
(runtime, NVRTC for the kernels open_sol compiles at run time, cuBLAS / cuSPARSE / cuSOLVER). open_sol is developed
with cupy 13.3. Skip this step on a machine without an NVIDIA GPU.

### 6a. Check the driver

```bash
nvidia-smi
```

The top-right "CUDA Version" is the newest CUDA the driver supports. Pick the cupy CUDA line at or below it:
CUDA 12 for driver 525 or newer (most current setups), CUDA 11 for older drivers. If `nvidia-smi` is not found,
install or update the NVIDIA driver first (nvidia.com/drivers).

### 6b. Install cupy (pick ONE route)

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

### 6c. Check the GPU backend

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

### 6d. cuDSS (nvmath-python)

NVIDIA's cuDSS sparse direct solver factorises on the GPU (solver back ends / factorizations `cudss` and
`cudss_mixed`: heat, electric, magnetic and solid models, the FEM eigensolvers). It comes with nvmath-python. With it
installed, Automatic direct solvers on the GPU pick cuDSS where the role allows it; without it they fall back to
SuperLU on the host, explicit cuDSS settings are validation errors and their self-tests pass as skipped. Some examples
need it (e.g. the 3D eigenfrequency examples `em3d/fdfd_mie_sphere_resonance_3d`, `nanobeam_cavity_eighth_3d`,
`fem_mie_sphere_resonance_3d`). Pick the CUDA line of your driver (`[cu13]` for CUDA 13):

```bash
pip install "nvmath-python[cu12]"
```

Check it (prints True):

```bash
python -c "from open_sol.solvers.cudss_factor import cudss_available; print(cudss_available())"
```

## 7. Optional packages

None of these is ever chosen on its own: install one when a model or solver node asks for it.

### 7a. gmsh and psutil

- `gmsh`: the external mesher for triangle / tetrahedron meshes, used when a mesh selects it (open_sol calls the
  `gmsh` executable; set `OPEN_SOL_GMSH` to its path if it is not on the PATH). The built-in mesher needs nothing
  extra. Set `OPEN_SOL_MESH_CACHE` to a folder to keep gmsh tetrahedron meshes on disk across runs (a re-run with the
  same geometry and sizes skips gmsh; at most 2 GB, the least recently used files go first).
- `psutil`: the available RAM for the memory estimates and host factor checks. Without it open_sol asks Windows /
  Linux directly, so it matters only on other systems.

```bash
conda install -c conda-forge gmsh psutil
```

### 7b. Multicore direct solvers on the CPU (MUMPS, PARDISO)

Two host sparse direct solvers use every processor core: MUMPS and Intel MKL PARDISO. A solver node selects them
(`sparse_direct` back_end / `arpack_shift_invert` factorization `mumps` or `pardiso`, and their `_mixed`
single-precision variants) where the physics offers them; Automatic never picks them (it uses SuperLU), and they run
on both the CPU and the GPU backend (the factor stays in host memory). Without the package such a node is a
validation error and their self-tests pass as skipped. Install either, both or neither.

**MUMPS (python-mumps).** python-mumps is a BSD-licensed wrapper of MUMPS (CeCILL-C licence, LGPL-like; neither is
bundled with open_sol). conda-forge ships both prebuilt; `pip install python-mumps` builds the wrapper against a MUMPS
installation of your own.

```bash
conda install -c conda-forge python-mumps
```

Check it (prints True):

```bash
python -c "from open_sol.solvers.host_direct import host_direct_available; print(host_direct_available('mumps'))"
```

MUMPS runs OpenMP / BLAS threads: set `OMP_NUM_THREADS` to limit them.

**Intel MKL PARDISO (pydiso).** pydiso is an MIT-licensed wrapper of the PARDISO solver in Intel oneMKL (Intel
Simplified Software License). It runs on x86-64 processors only (Intel or AMD; not on ARM such as Apple silicon).
Pick ONE route:

```bash
pip install pydiso
```

```bash
conda install -c conda-forge pydiso
```

The pip wheels (Linux, Windows) pull MKL in, a large download. Check it (prints True):

```bash
python -c "from open_sol.solvers.host_direct import host_direct_available; print(host_direct_available('pardiso'))"
```

PARDISO runs MKL threads: set `MKL_NUM_THREADS` to limit them. With numba kernels or a threaded BLAS in the same
process, limit the threads so the libraries do not oversubscribe the cores.

**threadpoolctl (with PARDISO).** numpy and scipy from pip or conda-forge use OpenBLAS, which keeps a thread pool of
its own beside MKL's. The two get in each other's way: a PARDISO solve right after a threaded numpy call runs 4 to 25
times slower (measured on 4 cores; eigenfrequency studies suffer most). Install threadpoolctl alongside pydiso and
open_sol holds the OpenBLAS threads back while PARDISO works:

```bash
conda install -c conda-forge threadpoolctl
```

Without it, or to cover the numpy calls of a physics between two PARDISO solves as well, set the environment
variable `OPENBLAS_NUM_THREADS=1` before starting Python (dense numpy linear algebra then runs on one core). A numpy
built on MKL (`conda install -c conda-forge "libblas=*=*mkl"`) shares MKL's threads and needs neither.

## Updating and removing

After `git pull` nothing needs reinstalling (editable install) unless `pyproject.toml` changed; then repeat step 4.

Remove the environment completely:

```bash
conda env remove -n open_sol
```

## Next steps

`README.md` has the quickstarts (template, check, run; the Python API; Simulation scripts). `python -m open_sol
describe templates` lists the template models.
