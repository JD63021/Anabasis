# Validated cases in v1.1d-final

## 1. Steady SIMPLE cylinder, Re≈20

Case:

```text
cases/simple/cylinder_re20_steady_simple.case
```

Executable:

```text
./simple_gpu_dp
```

Purpose:

- Checks the steady SIMPLE branch on the cylinder mesh.
- Uses the stable absolute-pressure path: `pSolveMode ofAbsolute`, `rcMode oflike`.

## 2. Transient PIMPLE/BDF2 Schaefer-Turek / FEATFLOW 3D cylinder

Case:

```text
cases/pimple/cylinder_3d3z_sine_re100_bdf2_800k.case
```

Executable:

```text
./pimple_gpu_bdf2_dp
```

Purpose:

- Validates the transient PIMPLE/BDF2-momentum branch against FEATFLOW force data.
- Uses a time-dependent parabolic-box sine inlet.
- Writes force coefficients to CSV every 0.01 s.

Validated robust pressure-gradient settings:

```text
pGradScheme gauss
pDeltaMinCos 0.1
```

The optional LSQ comparison case is provided separately:

```text
cases/pimple/cylinder_3d3z_sine_re100_bdf2_lsq_p005.case
```
