#pragma once

#include "common.h"

inline double phi_exact_xyz(double x, double y, double z) {
  return std::sin(kPi*x) * std::sin(kPi*y) * std::sin(kPi*z);
}

inline double rhs_exact_xyz(double x, double y, double z) {
  return 3.0 * kPi * kPi * phi_exact_xyz(x, y, z);
}

inline double phi_exact(const std::array<double,3>& x) {
  return phi_exact_xyz(x[0], x[1], x[2]);
}

inline double rhs_exact(const std::array<double,3>& x) {
  return rhs_exact_xyz(x[0], x[1], x[2]);
}
