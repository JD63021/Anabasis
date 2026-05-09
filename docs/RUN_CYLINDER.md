# Run the cylinder case

This command assumes the OpenFOAM `polyMesh` for the cylinder case is available at:

```text
/tmp/meshCase/constant/polyMesh
```

Build first from the repository root:

```bash
cd ~/anabasis_v1_1

export PETSC_DIR=$HOME/src/petsc
export PETSC_ARCH=arch-linux-cuda-opt
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:${LD_LIBRARY_PATH:-}"

SM_ARCH=sm_86 ./build_simple_gpu.sh
```

Use `SM_ARCH=sm_80` on A100.

Run the cylinder case:

```bash
cd ~/anabasis_v1_1

mkdir -p runs/cylinder

mpirun -n 1 ./simple_gpu \
  -case-config cases/cylinder.case \
  -out-prefix runs/cylinder/case \
  2>&1 | tee runs/cylinder/run.log
```

Check force output:

```bash
grep -E "Ubar, D, H|CD_vector|CL_y_vector|CL_z_vector|Wrote VTU" runs/cylinder/run.log
```
