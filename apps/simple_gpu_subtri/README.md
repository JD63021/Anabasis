# apps/simple_gpu

Steady incompressible SIMPLE solver app for OpenFOAM `polyMesh` input.

Build:

```bash
./build_simple_gpu_hypre31_double.sh
```

Run reference case:

```bash
mpirun -n 1 ./simple_gpu_dp \
  -case-config cases/simple/cylinder_re20_steady_simple.case
```
