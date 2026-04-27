#pragma once

#include "common.h"

inline double phi_exact_xyz(double x, double y, double z) {
  return std::sin(kPi*x) * std::sin(kPi*y) * std::sin(kPi*z);
}

inline std::array<double,3> grad_phi_exact_xyz(double x, double y, double z) {
  return {
    kPi * std::cos(kPi*x) * std::sin(kPi*y) * std::sin(kPi*z),
    kPi * std::sin(kPi*x) * std::cos(kPi*y) * std::sin(kPi*z),
    kPi * std::sin(kPi*x) * std::sin(kPi*y) * std::cos(kPi*z)
  };
}

inline double rhs_exact_xyz(double x, double y, double z) {
  return 3.0 * kPi * kPi * phi_exact_xyz(x, y, z);
}

inline double phi_exact(const std::array<double,3>& x) {
  return phi_exact_xyz(x[0], x[1], x[2]);
}

inline std::array<double,3> grad_phi_exact(const std::array<double,3>& x) {
  return grad_phi_exact_xyz(x[0], x[1], x[2]);
}

inline double rhs_exact(const std::array<double,3>& x) {
  return rhs_exact_xyz(x[0], x[1], x[2]);
}

inline double normal_grad_exact(const std::array<double,3>& x, const std::array<double,3>& outwardNormal) {
  return dot3(grad_phi_exact(x), outwardNormal);
}
