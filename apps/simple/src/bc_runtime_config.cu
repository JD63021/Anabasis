#include "bc_runtime_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pipebc {
namespace {

static std::string trim_copy(const std::string& s) {
  std::size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  std::size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

[[noreturn]] static void parse_fail(
    const std::string& path,
    int lineNo,
    const std::string& msg) {
  throw std::runtime_error(
      "BC config parse error in '" + path + "' at line " +
      std::to_string(lineNo) + ": " + msg);
}

static double parse_double_token(
    const std::string& tok,
    const std::string& path,
    int lineNo,
    const std::string& what) {
  try {
    std::size_t pos = 0;
    double v = std::stod(tok, &pos);
    if (pos != tok.size()) {
      parse_fail(path, lineNo, "invalid numeric token for " + what + ": '" + tok + "'");
    }
    return v;
  } catch (const std::exception&) {
    parse_fail(path, lineNo, "invalid numeric token for " + what + ": '" + tok + "'");
  }
}

static NormalVectorMode parse_normal_mode(
    const std::string& tok,
    const std::string& path,
    int lineNo) {
  if (tok == "average_patch_normal") return NormalVectorMode::AveragePatchNormal;
  if (tok == "local_face_normal")   return NormalVectorMode::LocalFaceNormal;
  parse_fail(path, lineNo,
             "normal mode must be 'average_patch_normal' or 'local_face_normal', got '" + tok + "'");
}

static std::vector<std::string> tokenize_line(const std::string& line) {
  std::istringstream iss(line);
  std::vector<std::string> out;
  for (std::string tok; iss >> tok; ) out.push_back(tok);
  return out;
}

} // namespace

RuntimeBCConfig load_runtime_bc_config(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open BC config file: " + path);
  }

  RuntimeBCConfig cfg;
  std::string raw;
  int lineNo = 0;

  while (std::getline(in, raw)) {
    ++lineNo;

    const std::size_t hashPos = raw.find('#');
    if (hashPos != std::string::npos) raw = raw.substr(0, hashPos);

    const std::string line = trim_copy(raw);
    if (line.empty()) continue;

    const auto tok = tokenize_line(line);
    if (tok.empty()) continue;

    if (tok[0] == "velocity") {
      if (tok.size() < 3) {
        parse_fail(path, lineNo, "velocity line must be: velocity <patch> <type> ...");
      }

      const std::string& patch = tok[1];
      const std::string& type  = tok[2];

      if (type == "wall_noslip") {
        if (tok.size() != 3) parse_fail(path, lineNo, "wall_noslip takes no extra arguments");
        cfg.velocityPatchSpecs.push_back(make_wall_noslip_bc(patch));
      } else if (type == "zero_gradient") {
        if (tok.size() != 3) parse_fail(path, lineNo, "zero_gradient takes no extra arguments");
        cfg.velocityPatchSpecs.push_back(make_zero_gradient_velocity_bc(patch));
      } else if (type == "fixed_uniform_vector") {
        if (tok.size() != 6) {
          parse_fail(path, lineNo,
                     "fixed_uniform_vector syntax: velocity <patch> fixed_uniform_vector <ux> <uy> <uz>");
        }
        const double ux = parse_double_token(tok[3], path, lineNo, "ux");
        const double uy = parse_double_token(tok[4], path, lineNo, "uy");
        const double uz = parse_double_token(tok[5], path, lineNo, "uz");
        cfg.velocityPatchSpecs.push_back(
            make_fixed_uniform_vector_bc(patch, std::array<double,3>{{ux, uy, uz}}));
      } else if (type == "fixed_normal_speed") {
        if (tok.size() != 4 && tok.size() != 5) {
          parse_fail(path, lineNo,
                     "fixed_normal_speed syntax: velocity <patch> fixed_normal_speed <value> [average_patch_normal|local_face_normal]");
        }
        const double value = parse_double_token(tok[3], path, lineNo, "normal speed");
        const NormalVectorMode mode =
            (tok.size() == 5)
                ? parse_normal_mode(tok[4], path, lineNo)
                : NormalVectorMode::AveragePatchNormal;
        cfg.velocityPatchSpecs.push_back(
            make_fixed_normal_speed_bc(patch, value, mode));
      } else if (type == "fixed_flow_rate") {
        if (tok.size() != 4 && tok.size() != 5) {
          parse_fail(path, lineNo,
                     "fixed_flow_rate syntax: velocity <patch> fixed_flow_rate <value> [average_patch_normal|local_face_normal]");
        }
        const double value = parse_double_token(tok[3], path, lineNo, "flow rate");
        const NormalVectorMode mode =
            (tok.size() == 5)
                ? parse_normal_mode(tok[4], path, lineNo)
                : NormalVectorMode::AveragePatchNormal;
        cfg.velocityPatchSpecs.push_back(
            make_fixed_flow_rate_bc(patch, value, mode));
      } else {
        parse_fail(path, lineNo, "unknown velocity BC type: '" + type + "'");
      }

    } else if (tok[0] == "pressure") {
      if (tok.size() < 3) {
        parse_fail(path, lineNo, "pressure line must be: pressure <patch> <type> ...");
      }

      const std::string& patch = tok[1];
      const std::string& type  = tok[2];

      if (type == "zero_gradient") {
        if (tok.size() != 3) parse_fail(path, lineNo, "zero_gradient takes no extra arguments");
        cfg.pressurePatchSpecs.push_back(make_pressure_zero_gradient_bc(patch));
      } else if (type == "open") {
        if (tok.size() != 3) parse_fail(path, lineNo, "open takes no extra arguments");
        cfg.pressurePatchSpecs.push_back(make_pressure_open_bc(patch));
      } else if (type == "fixed_value") {
        if (tok.size() != 4) {
          parse_fail(path, lineNo,
                     "fixed_value syntax: pressure <patch> fixed_value <value>");
        }
        const double value = parse_double_token(tok[3], path, lineNo, "pressure value");
        cfg.pressurePatchSpecs.push_back(make_pressure_fixed_value_bc(patch, value));
      } else {
        parse_fail(path, lineNo, "unknown pressure BC type: '" + type + "'");
      }

    } else {
      parse_fail(path, lineNo, "line must start with 'velocity' or 'pressure'");
    }
  }

  return cfg;
}

void validate_runtime_bc_config_against_patches(
    const RuntimeBCConfig& cfg,
    const std::vector<std::string>& patchNames) {
  std::unordered_set<std::string> known(patchNames.begin(), patchNames.end());

  for (const auto& spec : cfg.velocityPatchSpecs) {
    if (!known.count(spec.patchName)) {
      throw std::runtime_error(
          "Velocity BC refers to unknown patch '" + spec.patchName + "'");
    }
  }

  for (const auto& spec : cfg.pressurePatchSpecs) {
    if (!known.count(spec.patchName)) {
      throw std::runtime_error(
          "Pressure BC refers to unknown patch '" + spec.patchName + "'");
    }
  }
}

} // namespace pipebc
