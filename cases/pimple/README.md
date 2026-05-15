# PIMPLE reference cases

Use these with the transient PIMPLE app/executable:

```bash
./pimple_gpu_bdf2_dp -case-config cases/pimple/cylinder_3d3z_sine_re100_bdf2_800k.case
```

`nsteps` means physical time steps in these case files.  The validated path is BDF2 momentum with `ddtCorr` off.
