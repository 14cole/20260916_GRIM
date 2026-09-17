# Running 2D solves

Every 2D run, in the GHOST solver tab or a batch driver, uses one automatic
setup. You choose the physical problem; the solver chooses how to compute it.

## What you choose

| Setting | Values |
| --- | --- |
| Geometry and units | A `.geo` file or the Geometry tab; inches or meters. |
| Frequencies | A list or a start/stop/step sweep in GHz. |
| Azimuths | A list or a start/stop/step sweep in degrees. |
| Scattering mode | Monostatic, or bistatic with observation angles. |
| Mesh certification | On: solve base and fine meshes and publish the fine result only if both channels converge. Off: solve one mesh, marked `mesh_convergence_certified=false`. |
| Accuracy target | Standard, or tight (1% maximum complex change) for the base/fine comparison. |

## What is automatic

Monostatic runs:

- **Backend.** Dense or compressed Galerkin, ranked by predicted work for the
  geometry, materials and angle count, and admitted against available RAM
  (90% of currently available memory). Compressed solves may keep up to 60%
  of that limit (at least 2 GiB) for both polarizations' operators and the
  preconditioner.
- **Mesh.** An adaptive polynomial mesh: a quadratic candidate with a cubic
  accuracy check, local refinement when the comparison fails, and a global
  linear mesh as the final fallback.
- **Numerics.** Double-precision LU, automatic incident-field basis reuse, and
  up to 256 angles per batch.
- **Threads.** Up to four assembly threads. Factorization and angle solves use
  every host core for a desktop solve, or the worker's CPU reservation in a
  batch run.
- **Checkpoints.** Completed frequencies are saved to the application cache
  and reused when an identical run is repeated or resumed. They require
  matching geometry, angles, material file contents, certification settings
  and solver source. An interrupted frequency restarts from its beginning.
  Checkpoints do not replace exporting the final result.

Bistatic runs use dense LU on the global linear mesh.

## Batch drivers

Edit the CONFIG block of `ghost_backend/run_local_monostatic.py` or
`ghost_backend/run_hpc_monostatic.py`:

| Setting | Meaning |
| --- | --- |
| `FRD_DIR`, `OPN_DIR` | Input geometry folders, searched recursively. |
| `FREQUENCIES_GHZ`, `AZIMUTHS_DEG` | The sweep. |
| `OUTPUT_DIR` | Output root; each run gets a new timestamped folder. |
| `GEOMETRY_UNITS`, `MESH_CERTIFICATION`, `ACCURACY_TARGET` | As in the GUI. |
| `WORKERS` (local) | Optional ceiling on concurrent solves. |
| `MAX_SOLVE_GB` | Optional per-solve RAM ceiling in GiB. |
| SLURM settings (HPC) | `N_NODES`, `N_JOBS`, `ARRAY_THROTTLE`, partition, account, QoS, walltime, cores and memory per node, mail, extra `#SBATCH` lines, job prologue, `PYTHON_EXE`, `SUBMIT`. |

Both drivers cost every geometry/frequency unit from the mesh the solver will
build, run the most expensive units first, and admit concurrent solves against
the machine's or node's memory. 2D drivers do not read JSON configuration
files, and portable HPC bundles are for BoR requests.

## Status reporting

The status text reports assembly, factorization, angle solving and mesh
certification work, with elapsed solve time and sampled process RAM. Base and
fine mesh phases are identified separately. The percentage can remain fixed
while a long stage runs. RAM is sampled every 50 ms and includes other work in
the same process; it is not an exact allocation peak. Stage timings in exported
metadata are inclusive and may overlap.

## Python API

Public 2D solve functions still accept `execution_options` for tests,
benchmarks and diagnostics. Production runs do not need it; they use
`ghost_backend.execution.options.automatic_run(scattering)`.
