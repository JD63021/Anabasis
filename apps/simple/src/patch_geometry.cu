#include "patch_geometry.h"

#include <cmath>
#include <stdexcept>

namespace pipebc {
namespace {

static inline std::array<double,3> add3(const std::array<double,3>& a,
                                        const std::array<double,3>& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

static inline std::array<double,3> mul3(double s, const std::array<double,3>& a) {
  return {s * a[0], s * a[1], s * a[2]};
}

static inline double dot3(const std::array<double,3>& a,
                          const std::array<double,3>& b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static inline std::array<double,3> cross3(const std::array<double,3>& a,
                                          const std::array<double,3>& b) {
  return {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
  };
}

static inline double norm3(const std::array<double,3>& a) {
  return std::sqrt(dot3(a, a));
}

static inline std::array<double,3> normalize3(const std::array<double,3>& a) {
  const double n = norm3(a);
  if (n <= 1.0e-30) return {0.0, 0.0, 0.0};
  return {a[0]/n, a[1]/n, a[2]/n};
}

static inline std::array<double,3> choose_tangent_seed(const std::array<double,3>& n) {
  const std::array<double,3> ex{{1.0, 0.0, 0.0}};
  const std::array<double,3> ey{{0.0, 1.0, 0.0}};
  const std::array<double,3> ez{{0.0, 0.0, 1.0}};

  const double ax = std::fabs(dot3(n, ex));
  const double ay = std::fabs(dot3(n, ey));
  const double az = std::fabs(dot3(n, ez));

  if (ax <= ay && ax <= az) return ex;
  if (ay <= ax && ay <= az) return ey;
  return ez;
}

static void validate_input(const PatchGeometryInput& in) {
  if (!in.xf) throw std::runtime_error("PatchGeometryInput.xf is null");
  if (!in.nf) throw std::runtime_error("PatchGeometryInput.nf is null");
  if (!in.Sf) throw std::runtime_error("PatchGeometryInput.Sf is null");
  if (!in.Af) throw std::runtime_error("PatchGeometryInput.Af is null");
  if (!in.patchNames) throw std::runtime_error("PatchGeometryInput.patchNames is null");
  if (!in.patchStartFace) throw std::runtime_error("PatchGeometryInput.patchStartFace is null");
  if (!in.patchNFaces) throw std::runtime_error("PatchGeometryInput.patchNFaces is null");

  const std::size_t nFaces = in.Af->size();
  if (in.xf->size() != nFaces) throw std::runtime_error("xf size mismatch");
  if (in.nf->size() != nFaces) throw std::runtime_error("nf size mismatch");
  if (in.Sf->size() != nFaces) throw std::runtime_error("Sf size mismatch");
  if (in.nFaces != static_cast<int>(nFaces)) throw std::runtime_error("nFaces mismatch");
  if (in.nInternalFaces < 0 || in.nInternalFaces > in.nFaces) {
    throw std::runtime_error("nInternalFaces out of range");
  }
  if (in.patchStartFace->size() != in.patchNames->size()) {
    throw std::runtime_error("patchStartFace size mismatch");
  }
  if (in.patchNFaces->size() != in.patchNames->size()) {
    throw std::runtime_error("patchNFaces size mismatch");
  }
}

} // namespace

std::vector<PatchGeometrySummary>
build_patch_geometry_table(const PatchGeometryInput& in) {
  validate_input(in);

  const int nPatches = static_cast<int>(in.patchNames->size());
  std::vector<PatchGeometrySummary> out(nPatches);

  for (int pidx = 0; pidx < nPatches; ++pidx) {
    out[pidx].patchIndex = pidx;
    out[pidx].patchName = (*in.patchNames)[pidx];

    const int start = (*in.patchStartFace)[pidx];
    const int count = (*in.patchNFaces)[pidx];
    const int end = start + count;

    if (count < 0 || start < in.nInternalFaces || end > in.nFaces) {
      throw std::runtime_error(
          "Invalid patch face range for patch '" + out[pidx].patchName + "'");
    }

    std::array<double,3> centroidAccum{{0.0, 0.0, 0.0}};
    std::array<double,3> sumSf{{0.0, 0.0, 0.0}};
    double area = 0.0;

    for (int f = start; f < end; ++f) {
      out[pidx].faces.push_back(f);

      const double Af = (*in.Af)[f];
      area += Af;
      centroidAccum = add3(centroidAccum, mul3(Af, (*in.xf)[f]));
      sumSf = add3(sumSf, (*in.Sf)[f]);
    }

    out[pidx].totalArea = area;
    out[pidx].sumSf = sumSf;

    if (area > 1.0e-30) {
      out[pidx].centroid = mul3(1.0 / area, centroidAccum);
    } else {
      out[pidx].centroid = {0.0, 0.0, 0.0};
    }

    const double sumSfNorm = norm3(sumSf);
    out[pidx].averageNormal = normalize3(sumSf);
    out[pidx].planarity = (area > 1.0e-30) ? (sumSfNorm / area) : 0.0;

    if (norm3(out[pidx].averageNormal) > 0.0) {
      const auto seed = choose_tangent_seed(out[pidx].averageNormal);
      out[pidx].tangent1 = normalize3(cross3(out[pidx].averageNormal, seed));
      out[pidx].tangent2 = normalize3(cross3(out[pidx].averageNormal, out[pidx].tangent1));
    } else {
      out[pidx].tangent1 = {1.0, 0.0, 0.0};
      out[pidx].tangent2 = {0.0, 1.0, 0.0};
    }
  }

  return out;
}

int
find_patch_index_by_name(const std::vector<PatchGeometrySummary>& patches,
                         const std::string& patchName) {
  for (int i = 0; i < static_cast<int>(patches.size()); ++i) {
    if (patches[i].patchName == patchName) return i;
  }
  return -1;
}

const PatchGeometrySummary&
get_patch_geometry_or_throw(const std::vector<PatchGeometrySummary>& patches,
                            const std::string& patchName) {
  const int idx = find_patch_index_by_name(patches, patchName);
  if (idx < 0) {
    throw std::runtime_error("Patch '" + patchName + "' not found in patch geometry table");
  }
  return patches[idx];
}

bool
is_patch_nearly_planar(const PatchGeometrySummary& patch,
                       double threshold) {
  return patch.planarity >= threshold;
}

} // namespace pipebc
