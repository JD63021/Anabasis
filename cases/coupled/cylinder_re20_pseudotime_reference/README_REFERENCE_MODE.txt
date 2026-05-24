REFERENCE MODE FOR COUPLED PSEUDO-TIME CYLINDER Re=20
========================================================

Use reference.case in this folder.

This is the validated M2 mode:

  picardConvergenceMode oflike
  coupledPressureNonOrthCorr 0
  coupledPressureCorrectVelocity 0
  pNonOrthScale 1
  momNonOrthScale 1

Important runtime rule:

  ANABASIS_COUPLED_RC_FULL_NONORTH=0

Do NOT enable ANABASIS_COUPLED_RC_FULL_NONORTH=1 for this reference case.
That experimental full-face Rhie-Chow nonorth path was found to shift the
drag upward.

Known reference behavior on current cylinder mesh:

  SIMPLE C_drag ≈ 6.09183149
  Coupled M2 reference mode should converge close to the SIMPLE drag.

Recommended run command:

  cd /home/jd/anabasis_v0_coupled_ethier

  ANABASIS_COUPLED_FACE_SKEW_CORR=1 \
  ANABASIS_COUPLED_RC_FULL_NONORTH=0 \
  ANABASIS_COUPLED_SUBTRI_CONTINUITY=1 \
  ANABASIS_COUPLED_SUBTRI_FLUX_RECON=1 \
  mpirun -n 1 \
    /home/jd/anabasis_v0_coupled_ethier/build_coupled_ethier_dp/apps/coupled_transient_nonorth_gpu/coupled_transient_nonorth_gpu \
    -case-config /home/jd/anabasis_v0_coupled_ethier/runs/coupled_cylinder_re20_pseudotime/reference.case \
    2>&1 | tee /home/jd/anabasis_v0_coupled_ethier/runs/coupled_cylinder_re20_pseudotime/reference_run.log
