#pragma once

#include <string>
#include <vector>

#include "pressure_lib_adapter.h"

struct VelocityLibOptions {
  double rho = 1.0;
  double mu = 1.0;
  double uRelax = 1.0;
  std::string convectionScheme = "central";   // central | upwind1
  std::string diffusionScheme  = "nonorth";   // orth | nonorth
};

struct VelocityLibAssembly {
  std::vector<double> values;   // matrix coefficients in fixed CSR order
  std::vector<double> rhs;      // assembled rhs
  std::vector<double> rAU;      // vol / diag(unrelaxed)
  double wallTime = 0.0;        // assembly time only
};

struct VelocityLibAssemblyHandle {
  void* impl = nullptr;
  int nRows = 0;
  int nnz = 0;
};

void init_velocity_lib_assembly_handle(
    VelocityLibAssemblyHandle& handle,
    const PressureLibMeshData& meshData);

VelocityLibAssembly assemble_velocity_with_libscalar(
    VelocityLibAssemblyHandle& handle,
    const std::vector<double>& qLag,
    const std::vector<double>& uConv,
    const std::vector<double>& vConv,
    const std::vector<double>& wConv,
    const std::vector<double>& gradPcomp,
    const std::vector<std::string>& bcQType,
    const std::vector<double>& bcQFaceVal,
    const std::vector<std::string>& bcUType,
    const std::vector<double>& bcUFaceVal,
    const std::vector<std::string>& bcVType,
    const std::vector<double>& bcVFaceVal,
    const std::vector<std::string>& bcWType,
    const std::vector<double>& bcWFaceVal,
    const VelocityLibOptions& opt);

void destroy_velocity_lib_assembly_handle(VelocityLibAssemblyHandle& handle);
