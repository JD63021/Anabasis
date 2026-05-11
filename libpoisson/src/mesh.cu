#include "mesh.h"

static std::string read_file_to_string(const std::string& filename) {
  std::ifstream in(filename.c_str());
  if (!in) throw std::runtime_error("Could not open " + filename);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string strip_comments(const std::string& txt) {
  std::string s = txt;
  for (;;) {
    std::size_t a = s.find("/*");
    if (a == std::string::npos) break;
    std::size_t b = s.find("*/", a + 2);
    if (b == std::string::npos) break;
    s.erase(a, b - a + 2);
  }
  std::stringstream in(s);
  std::string line, out;
  while (std::getline(in, line)) {
    std::size_t p = line.find("//");
    if (p != std::string::npos) line = line.substr(0, p);
    out += line;
    out.push_back('\n');
  }
  return out;
}

static std::string extract_main_list(const std::string& txt) {
  std::size_t startIdx = std::string::npos, endIdx = std::string::npos;
  for (std::size_t i = 0; i < txt.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(txt[i]))) {
      std::size_t j = i;
      while (j < txt.size() && std::isdigit(static_cast<unsigned char>(txt[j]))) ++j;
      while (j < txt.size() && std::isspace(static_cast<unsigned char>(txt[j]))) ++j;
      if (j < txt.size() && txt[j] == '(') {
        startIdx = i;
        endIdx = j;
        break;
      }
    }
  }
  if (startIdx == std::string::npos) throw std::runtime_error("Could not locate top-level OpenFOAM list");
  std::size_t startPos = endIdx;
  int depth = 0;
  std::size_t endPos = std::string::npos;
  for (std::size_t i = startPos; i < txt.size(); ++i) {
    if (txt[i] == '(') ++depth;
    else if (txt[i] == ')') {
      --depth;
      if (depth == 0) { endPos = i; break; }
    }
  }
  if (endPos == std::string::npos) throw std::runtime_error("Failed to match parentheses");
  return txt.substr(startPos + 1, endPos - startPos - 1);
}

static std::vector<std::array<double,3>> read_foam_points(const std::string& filename) {
  std::string inside = extract_main_list(strip_comments(read_file_to_string(filename)));
  std::vector<std::array<double,3>> P;
  std::size_t pos = 0;
  while (true) {
    std::size_t a = inside.find('(', pos);
    if (a == std::string::npos) break;
    std::size_t b = inside.find(')', a + 1);
    if (b == std::string::npos) break;
    std::stringstream ss(inside.substr(a + 1, b - a - 1));
    std::array<double,3> p{};
    ss >> p[0] >> p[1] >> p[2];
    P.push_back(p);
    pos = b + 1;
  }
  return P;
}

static std::vector<std::vector<int>> read_foam_faces(const std::string& filename) {
  std::string inside = extract_main_list(strip_comments(read_file_to_string(filename)));
  std::vector<std::vector<int>> faces;
  std::size_t pos = 0;
  while (pos < inside.size()) {
    while (pos < inside.size() && std::isspace(static_cast<unsigned char>(inside[pos]))) ++pos;
    if (pos >= inside.size()) break;
    if (!std::isdigit(static_cast<unsigned char>(inside[pos]))) { ++pos; continue; }
    std::size_t q = pos;
    while (q < inside.size() && std::isdigit(static_cast<unsigned char>(inside[q]))) ++q;
    int k = std::atoi(inside.substr(pos, q - pos).c_str());
    while (q < inside.size() && std::isspace(static_cast<unsigned char>(inside[q]))) ++q;
    if (q >= inside.size() || inside[q] != '(') throw std::runtime_error("Malformed faces list");
    std::size_t r = inside.find(')', q + 1);
    if (r == std::string::npos) throw std::runtime_error("Malformed faces list");
    std::stringstream ss(inside.substr(q + 1, r - q - 1));
    std::vector<int> fv(k);
    for (int i = 0; i < k; ++i) ss >> fv[i];
    faces.push_back(fv);
    pos = r + 1;
  }
  return faces;
}

static std::vector<int> read_foam_labels(const std::string& filename) {
  std::string inside = extract_main_list(strip_comments(read_file_to_string(filename)));
  std::stringstream ss(inside);
  std::vector<int> vals;
  int v;
  while (ss >> v) vals.push_back(v);
  return vals;
}

static std::vector<PatchInfo> read_foam_boundary(const std::string& filename) {
  std::string inside = extract_main_list(strip_comments(read_file_to_string(filename)));
  std::vector<PatchInfo> patches;
  std::size_t pos = 0;
  while (pos < inside.size()) {
    while (pos < inside.size() && std::isspace(static_cast<unsigned char>(inside[pos]))) ++pos;
    if (pos >= inside.size()) break;
    if (!(std::isalpha(static_cast<unsigned char>(inside[pos])) || inside[pos] == '_')) { ++pos; continue; }
    std::size_t a = pos;
    while (pos < inside.size() && (std::isalnum(static_cast<unsigned char>(inside[pos])) || inside[pos] == '_')) ++pos;
    std::string name = inside.substr(a, pos - a);
    while (pos < inside.size() && std::isspace(static_cast<unsigned char>(inside[pos]))) ++pos;
    if (pos >= inside.size() || inside[pos] != '{') continue;
    int depth = 1;
    std::size_t bodyStart = ++pos;
    while (pos < inside.size() && depth > 0) {
      if (inside[pos] == '{') ++depth;
      else if (inside[pos] == '}') --depth;
      ++pos;
    }
    std::string body = inside.substr(bodyStart, pos - bodyStart - 1);
    PatchInfo p;
    p.name = name;
    auto find_int = [&](const std::string& key) -> int {
      std::size_t k = body.find(key);
      if (k == std::string::npos) return 0;
      k += key.size();
      while (k < body.size() && !std::isdigit(static_cast<unsigned char>(body[k])) && body[k] != '-') ++k;
      std::size_t e = k;
      while (e < body.size() && (std::isdigit(static_cast<unsigned char>(body[e])) || body[e] == '-')) ++e;
      return std::atoi(body.substr(k, e - k).c_str());
    };
    auto find_word = [&](const std::string& key) -> std::string {
      std::size_t k = body.find(key);
      if (k == std::string::npos) return "";
      k += key.size();
      while (k < body.size() && std::isspace(static_cast<unsigned char>(body[k]))) ++k;
      std::size_t e = k;
      while (e < body.size() && (std::isalnum(static_cast<unsigned char>(body[e])) || body[e] == '_')) ++e;
      return body.substr(k, e - k);
    };
    p.nFaces = find_int("nFaces");
    p.startFace = find_int("startFace");
    p.type = find_word("type");
    patches.push_back(p);
  }
  return patches;
}

Mesh read_openfoam_polymesh(const std::string& polyMeshDir) {
  Mesh mesh;
  auto patches = read_foam_boundary(polyMeshDir + "/boundary");
  mesh.P = read_foam_points(polyMeshDir + "/points");
  mesh.faces = read_foam_faces(polyMeshDir + "/faces");
  std::vector<int> owner0 = read_foam_labels(polyMeshDir + "/owner");
  std::vector<int> neigh0 = read_foam_labels(polyMeshDir + "/neighbour");

  mesh.nFaces = static_cast<int>(mesh.faces.size());
  mesh.nInternalFaces = static_cast<int>(neigh0.size());
  mesh.nCells = 0;
  for (int v : owner0) mesh.nCells = std::max(mesh.nCells, v + 1);
  for (int v : neigh0) mesh.nCells = std::max(mesh.nCells, v + 1);

  mesh.owner.resize(mesh.nFaces);
  mesh.neigh.assign(mesh.nInternalFaces, 0);
  for (int i = 0; i < mesh.nFaces; ++i) mesh.owner[i] = owner0[i];
  for (int i = 0; i < mesh.nInternalFaces; ++i) mesh.neigh[i] = neigh0[i];

  mesh.bPatch.assign(mesh.nFaces, 0);
  mesh.patchNames.resize(patches.size());
  for (std::size_t k = 0; k < patches.size(); ++k) {
    mesh.patchNames[k] = patches[k].name;
    for (int f = patches[k].startFace; f < patches[k].startFace + patches[k].nFaces; ++f) {
      mesh.bPatch[f] = static_cast<int>(k) + 1;
    }
  }

  mesh.cellFaces.assign(mesh.nCells, {});
  mesh.cellOrient.assign(mesh.nCells, {});
  for (int f = 0; f < mesh.nFaces; ++f) {
    int P = mesh.owner[f];
    mesh.cellFaces[P].push_back(f);
    mesh.cellOrient[P].push_back(+1);
  }
  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    int N = mesh.neigh[f];
    mesh.cellFaces[N].push_back(f);
    mesh.cellOrient[N].push_back(-1);
  }

  mesh.cc.assign(mesh.nCells, {0,0,0});
  mesh.vol.assign(mesh.nCells, 0.0);
  for (int c = 0; c < mesh.nCells; ++c) {
    std::set<int> vertsSet;
    for (int f : mesh.cellFaces[c]) for (int v : mesh.faces[f]) vertsSet.insert(v);
    std::array<double,3> c0{0,0,0};
    for (int v : vertsSet) c0 = add3(c0, mesh.P[v]);
    c0 = mul3(1.0 / std::max(static_cast<int>(vertsSet.size()), 1), c0);

    double V = 0.0;
    std::array<double,3> M{0,0,0};
    for (std::size_t j = 0; j < mesh.cellFaces[c].size(); ++j) {
      int f = mesh.cellFaces[c][j];
      int ori = mesh.cellOrient[c][j];
      std::vector<int> fv = mesh.faces[f];
      if (ori < 0) std::reverse(fv.begin(), fv.end());
      auto a = mesh.P[fv[0]];
      for (std::size_t i = 1; i + 1 < fv.size(); ++i) {
        auto b = mesh.P[fv[i]];
        auto d = mesh.P[fv[i+1]];
        double vTet = dot3(sub3(a, c0), cross3(sub3(b, c0), sub3(d, c0))) / 6.0;
        auto cTet = mul3(0.25, add3(add3(c0, a), add3(b, d)));
        V += vTet;
        M = add3(M, mul3(vTet, cTet));
      }
    }
    if (V <= 0.0) throw std::runtime_error("Non-positive cell volume at cell " + std::to_string(c));
    mesh.vol[c] = V;
    mesh.cc[c] = mul3(1.0 / V, M);
  }

  mesh.xf.assign(mesh.nFaces, {0,0,0});
  mesh.Af.assign(mesh.nFaces, 0.0);
  mesh.nf.assign(mesh.nFaces, {0,0,0});
  mesh.Sf.assign(mesh.nFaces, {0,0,0});
  for (int f = 0; f < mesh.nFaces; ++f) {
    std::array<double,3> xfc{0,0,0};
    for (int v : mesh.faces[f]) xfc = add3(xfc, mesh.P[v]);
    xfc = mul3(1.0 / std::max(static_cast<int>(mesh.faces[f].size()), 1), xfc);
    mesh.xf[f] = xfc;

    auto a = mesh.P[mesh.faces[f][0]];
    std::array<double,3> areaVec{0,0,0};
    for (std::size_t i = 1; i + 1 < mesh.faces[f].size(); ++i) {
      auto b = mesh.P[mesh.faces[f][i]];
      auto d = mesh.P[mesh.faces[f][i+1]];
      areaVec = add3(areaVec, mul3(0.5, cross3(sub3(b,a), sub3(d,a))));
    }
    double areaMag = norm3(areaVec);
    if (areaMag <= 1e-30) throw std::runtime_error("Degenerate face area at face " + std::to_string(f));
    auto nloc = mul3(1.0 / areaMag, areaVec);
    int P = mesh.owner[f];
    std::array<double,3> dtest = (f < mesh.nInternalFaces)
      ? sub3(mesh.cc[mesh.neigh[f]], mesh.cc[P])
      : sub3(mesh.xf[f], mesh.cc[P]);
    if (dot3(nloc, dtest) < 0.0) nloc = mul3(-1.0, nloc);
    mesh.Af[f] = areaMag;
    mesh.nf[f] = nloc;
    mesh.Sf[f] = mul3(areaMag, nloc);
  }

  mesh.cellNbrs.assign(mesh.nCells, {});
  mesh.cellBFace.assign(mesh.nCells, {});
  for (int f = 0; f < mesh.nFaces; ++f) {
    int P = mesh.owner[f];
    if (f < mesh.nInternalFaces) {
      int N = mesh.neigh[f];
      mesh.cellNbrs[P].push_back(N);
      mesh.cellNbrs[N].push_back(P);
    } else {
      mesh.cellBFace[P].push_back(f);
    }
  }
  for (int c = 0; c < mesh.nCells; ++c) {
    std::sort(mesh.cellNbrs[c].begin(), mesh.cellNbrs[c].end());
    mesh.cellNbrs[c].erase(std::unique(mesh.cellNbrs[c].begin(), mesh.cellNbrs[c].end()), mesh.cellNbrs[c].end());
  }

  mesh.maxNonOrthDeg = 0.0;
  for (int f = 0; f < mesh.nInternalFaces; ++f) {
    auto d = sub3(mesh.cc[mesh.neigh[f]], mesh.cc[mesh.owner[f]]);
    double cosang = std::fabs(dot3(d, mesh.nf[f])) / std::max(norm3(d), 1e-30);
    cosang = std::min(1.0, std::max(0.0, cosang));
    mesh.maxNonOrthDeg = std::max(mesh.maxNonOrthDeg, std::acos(cosang) * 180.0 / kPi);
  }

  return mesh;
}
