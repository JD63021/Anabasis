
#include "mesh.h"
#include "precision.h"
#include "hypre_backend.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// DG2/DG1 experimental app
// -----------------------------------------------------------------------------
// This file was forked from apps/dg2_cg1_pmass_simple_diag/src/main.cu.
// Pressure is intentionally DG1-discontinuous: 4 pressure rows per tetrahedron.
// The channel/cylinder continuity operator below follows the MATLAB reference
// NSE2D_CHANNEL_DG2DG1_SINGLEFLUX_READY:
//   r_K(q) = -int_K grad(q).u + int_{dK} q uhat_n,
//   uhat_n = 0.5*(u^-+u^+).n^- on interior faces,
// inlet/wall normal flux is prescribed in bD, and outlet remains open/interior.
// No pressure jump/SIPG penalty is added in this DG2/DG1 variant.
static constexpr bool kDG2DG1Pressure = true;


struct TetP2Geom {
    std::array<int,4> v{};
    std::array<std::array<double,3>,4> x{};
    std::array<std::array<double,3>,10> xP2{};
    std::array<std::array<double,3>,4> gradLam{};
    double vol = 0.0;
};

struct QuadTetPoint {
    std::array<double,4> lam{};
    double w = 0.0;
};

struct QuadTriPoint {
    std::array<double,3> mu{};
    double w = 0.0;
};

struct RectCSR {
    int nRows = 0;
    int nCols = 0;
    int nnz = 0;
    std::vector<int> rowOffsets;
    std::vector<int> cols;
    std::vector<double> values;
};

static double wall_seconds()
{
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

static double tet_signed_volume(
    const std::array<double,3>& a,
    const std::array<double,3>& b,
    const std::array<double,3>& c,
    const std::array<double,3>& d)
{
    return dot3(sub3(b,a), cross3(sub3(c,a), sub3(d,a))) / 6.0;
}

static std::array<double,3> lincomb4(
    const std::array<std::array<double,3>,4>& x,
    const std::array<double,4>& lam)
{
    std::array<double,3> q{0.0,0.0,0.0};
    for (int a=0; a<4; ++a) q = add3(q, mul3(lam[a], x[a]));
    return q;
}

static double p_test_exact(const std::array<double,3>& x)
{
    return std::sin(kPi*x[0]) * std::sin(kPi*x[1]) * std::sin(kPi*x[2]);
}

static double compute_weighted_mean(const std::vector<double>& x, const std::vector<double>& w)
{
    double num = 0.0, den = 0.0;
    for (std::size_t i=0; i<x.size(); ++i) { num += w[i]*x[i]; den += w[i]; }
    return num / std::max(den, 1e-300);
}

static double weighted_l2_norm(const std::vector<double>& x, const std::vector<double>& w)
{
    double s = 0.0;
    for (std::size_t i=0; i<x.size(); ++i) s += w[i] * x[i] * x[i];
    return std::sqrt(std::max(0.0, s));
}

static double dot_vec(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (std::size_t i=0; i<a.size(); ++i) s += a[i]*b[i];
    return s;
}

static double norm_vec(const std::vector<double>& a)
{
    return std::sqrt(std::max(0.0, dot_vec(a,a)));
}

static double norm_vec_free_rows(
    const std::vector<double>& a,
    const std::vector<unsigned char>* lockedRows)
{
    if (!lockedRows || lockedRows->empty()) {
        return norm_vec(a);
    }

    double s = 0.0;
    const int n = (int)a.size();

    for (int i=0; i<n; ++i) {
        if (i < (int)lockedRows->size() && (*lockedRows)[i]) {
            continue;
        }
        s += a[i] * a[i];
    }

    return std::sqrt(std::max(0.0, s));
}

static double max_abs_vec(const std::vector<double>& a)
{
    double m = 0.0;
    for (double v : a) m = std::max(m, std::abs(v));
    return m;
}

static void compute_tet_gradients(TetP2Geom& tg)
{
    const auto e1 = sub3(tg.x[1], tg.x[0]);
    const auto e2 = sub3(tg.x[2], tg.x[0]);
    const auto e3 = sub3(tg.x[3], tg.x[0]);
    const double det = dot3(e1, cross3(e2, e3));
    if (det <= 0.0) throw std::runtime_error("compute_tet_gradients: non-positive oriented tet");
    const auto row0 = mul3(1.0/det, cross3(e2, e3));
    const auto row1 = mul3(1.0/det, cross3(e3, e1));
    const auto row2 = mul3(1.0/det, cross3(e1, e2));
    tg.gradLam[1] = row0;
    tg.gradLam[2] = row1;
    tg.gradLam[3] = row2;
    tg.gradLam[0] = mul3(-1.0, add3(add3(row0,row1),row2));
    tg.vol = det/6.0;
}

static void fill_p2_nodes(TetP2Geom& tg)
{
    for (int i=0; i<4; ++i) tg.xP2[i] = tg.x[i];
    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (int e=0; e<6; ++e) tg.xP2[4+e] = mul3(0.5, add3(tg.x[edge[e][0]], tg.x[edge[e][1]]));
}

static std::vector<TetP2Geom> reconstruct_tets(const Mesh& mesh)
{
    std::vector<TetP2Geom> tets(mesh.nCells);
    int nonTet=0, neg=0, deg=0;
    double sumVol=0.0, maxRelVolDiff=0.0;
    for (int c=0; c<mesh.nCells; ++c) {
        std::set<int> vertsSet;
        for (int f : mesh.cellFaces[c]) for (int v : mesh.faces[f]) vertsSet.insert(v);
        if ((int)vertsSet.size()!=4 || (int)mesh.cellFaces[c].size()!=4) { nonTet++; continue; }
        TetP2Geom tg;
        int k=0;
        for (int v : vertsSet) tg.v[k++] = v;
        double sv = tet_signed_volume(mesh.P[tg.v[0]], mesh.P[tg.v[1]], mesh.P[tg.v[2]], mesh.P[tg.v[3]]);
        if (sv < 0.0) { std::swap(tg.v[2], tg.v[3]); sv = -sv; neg++; }
        for (int i=0; i<4; ++i) tg.x[i] = mesh.P[tg.v[i]];
        compute_tet_gradients(tg);
        fill_p2_nodes(tg);
        if (tg.vol <= 1e-300) deg++;
        sumVol += tg.vol;
        if (c < (int)mesh.vol.size() && mesh.vol[c] > 0.0) maxRelVolDiff = std::max(maxRelVolDiff, std::abs(tg.vol - mesh.vol[c]) / mesh.vol[c]);
        tets[c] = tg;
    }
    std::printf("\n--- Tet/P2 reconstruction ---\n");
    std::printf("cells                         = %d\n", mesh.nCells);
    std::printf("nonTetCells                   = %d\n", nonTet);
    std::printf("flippedOrientationTets         = %d\n", neg);
    std::printf("degenerateTets                = %d\n", deg);
    std::printf("sum reconstructed volume       = %.17e\n", sumVol);
    std::printf("max relative vol diff vs Mesh  = %.3e\n", maxRelVolDiff);
    if (nonTet || deg) throw std::runtime_error("This diagnostic currently expects a pure non-degenerate tet polyMesh.");
    return tets;
}

static void p2_tet_basis(const std::array<double,4>& lam, double N[10])
{
    N[0] = lam[0]*(2.0*lam[0]-1.0);
    N[1] = lam[1]*(2.0*lam[1]-1.0);
    N[2] = lam[2]*(2.0*lam[2]-1.0);
    N[3] = lam[3]*(2.0*lam[3]-1.0);
    N[4] = 4.0*lam[0]*lam[1];
    N[5] = 4.0*lam[0]*lam[2];
    N[6] = 4.0*lam[0]*lam[3];
    N[7] = 4.0*lam[1]*lam[2];
    N[8] = 4.0*lam[1]*lam[3];
    N[9] = 4.0*lam[2]*lam[3];
}

static void p2_tet_grad(const TetP2Geom& K, const std::array<double,4>& lam, std::array<double,3> G[10])
{
    for (int a=0; a<4; ++a) G[a] = mul3(4.0*lam[a]-1.0, K.gradLam[a]);
    G[4] = mul3(4.0, add3(mul3(lam[0], K.gradLam[1]), mul3(lam[1], K.gradLam[0])));
    G[5] = mul3(4.0, add3(mul3(lam[0], K.gradLam[2]), mul3(lam[2], K.gradLam[0])));
    G[6] = mul3(4.0, add3(mul3(lam[0], K.gradLam[3]), mul3(lam[3], K.gradLam[0])));
    G[7] = mul3(4.0, add3(mul3(lam[1], K.gradLam[2]), mul3(lam[2], K.gradLam[1])));
    G[8] = mul3(4.0, add3(mul3(lam[1], K.gradLam[3]), mul3(lam[3], K.gradLam[1])));
    G[9] = mul3(4.0, add3(mul3(lam[2], K.gradLam[3]), mul3(lam[3], K.gradLam[2])));
}

static std::vector<std::pair<double,double>> gauss_legendre_1d(int n)
{
    if (n==4) return {{0.06943184420297371,0.17392742256872692},{0.33000947820757187,0.32607257743127305},{0.6699905217924281,0.32607257743127305},{0.9305681557970262,0.17392742256872692}};
    if (n==5) return {{0.04691007703066802,0.11846344252809454},{0.23076534494715845,0.23931433524968324},{0.5,0.28444444444444444},{0.7692346550528415,0.23931433524968324},{0.9530899229693319,0.11846344252809454}};
    throw std::runtime_error("gauss_legendre_1d: only n=4 or n=5 implemented");
}

static std::vector<QuadTetPoint> make_tet_quad(int n1d)
{
    const auto gl = gauss_legendre_1d(n1d);
    std::vector<QuadTetPoint> q;
    double sum = 0.0;
    for (const auto& aw: gl) {
        const double a=aw.first, wa=aw.second;
        for (const auto& bw: gl) {
            const double b=bw.first, wb=bw.second;
            for (const auto& cw: gl) {
                const double c=cw.first, wc=cw.second;
                QuadTetPoint qp;
                qp.lam[0] = 1.0-a;
                qp.lam[1] = a*(1.0-b);
                qp.lam[2] = a*b*(1.0-c);
                qp.lam[3] = a*b*c;
                qp.w = 6.0*wa*wb*wc*a*a*b;
                sum += qp.w;
                q.push_back(qp);
            }
        }
    }
    for (auto& qp: q) qp.w /= sum;
    return q;
}

static std::vector<QuadTriPoint> make_tri_quad(int n1d)
{
    const auto gl = gauss_legendre_1d(n1d);
    std::vector<QuadTriPoint> q;
    double sum = 0.0;
    for (const auto& aw: gl) {
        const double a=aw.first, wa=aw.second;
        for (const auto& bw: gl) {
            const double b=bw.first, wb=bw.second;
            QuadTriPoint qp;
            qp.mu[0] = 1.0-a;
            qp.mu[1] = a*(1.0-b);
            qp.mu[2] = a*b;
            qp.w = 2.0*wa*wb*a;
            sum += qp.w;
            q.push_back(qp);
        }
    }
    for (auto& qp: q) qp.w /= sum;
    return q;
}

static bool face_lam_on_tet(const std::vector<int>& faceVerts, const TetP2Geom& K, const std::array<double,3>& mu, std::array<double,4>& lam)
{
    lam = {0.0,0.0,0.0,0.0};
    if (faceVerts.size() != 3) return false;
    for (int j=0; j<3; ++j) {
        bool found = false;
        for (int a=0; a<4; ++a) {
            if (K.v[a] == faceVerts[j]) { lam[a] = mu[j]; found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

static void check_p2_basis(const std::vector<TetP2Geom>& tets, const std::vector<QuadTetPoint>& qt)
{
    double maxSumN = 0.0, maxSumG = 0.0;
    const int nCheck = std::min<int>(tets.size(), 97);
    for (int c=0; c<nCheck; ++c) {
        for (const auto& qp: qt) {
            double N[10]; std::array<double,3> G[10];
            p2_tet_basis(qp.lam, N);
            p2_tet_grad(tets[c], qp.lam, G);
            double sN = 0.0; std::array<double,3> sG{0.0,0.0,0.0};
            for (int i=0; i<10; ++i) { sN += N[i]; sG = add3(sG,G[i]); }
            maxSumN = std::max(maxSumN, std::abs(sN-1.0));
            maxSumG = std::max(maxSumG, norm3(sG));
        }
    }
    std::printf("P2 basis max |sum N - 1|      = %.3e\n", maxSumN);
    std::printf("P2 basis max |sum grad N|     = %.3e\n", maxSumG);
}

static RectCSR rows_to_rectcsr(int nRows, int nCols, const std::vector<std::map<int,double>>& rows)
{
    RectCSR A;
    A.nRows = nRows; A.nCols = nCols;
    A.rowOffsets.assign(nRows+1, 0);
    for (int r=0; r<nRows; ++r) A.rowOffsets[r+1] = A.rowOffsets[r] + (int)rows[r].size();
    A.nnz = A.rowOffsets[nRows];
    A.cols.resize(A.nnz);
    A.values.resize(A.nnz, 0.0);
    for (int r=0; r<nRows; ++r) {
        int p = A.rowOffsets[r];
        for (const auto& kv: rows[r]) { A.cols[p] = kv.first; A.values[p] = kv.second; ++p; }
    }
    return A;
}

static CSRPattern rows_to_csrpattern(int nRows, const std::vector<std::map<int,double>>& rows, std::vector<double>& values)
{
    CSRPattern A;
    A.nRows = nRows;
    A.rowOffsets.assign(nRows+1, 0);
    A.ncols.assign(nRows, 0);
    A.rows.resize(nRows);
    A.diagPos.assign(nRows, -1);
    for (int r=0; r<nRows; ++r) {
        A.rows[r] = static_cast<HYPRE_BigInt>(r);
        A.ncols[r] = static_cast<HYPRE_Int>(rows[r].size());
        A.rowOffsets[r+1] = A.rowOffsets[r] + (int)rows[r].size();
    }
    A.nnz = A.rowOffsets[nRows];
    A.cols.resize(A.nnz);
    values.resize(A.nnz);
    for (int r=0; r<nRows; ++r) {
        int p = A.rowOffsets[r];
        for (const auto& kv: rows[r]) {
            A.cols[p] = static_cast<HYPRE_BigInt>(kv.first);
            values[p] = kv.second;
            if (kv.first == r) A.diagPos[r] = p;
            ++p;
        }
    }
    return A;
}

static void assemble_Ap_cg1_direction(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& qt,
    const std::vector<QuadTriPoint>& qf,
    int dir,
    RectCSR& Ap,
    int& faceMappingFailures,
    const std::vector<char>* boundaryPressureFaceMask = nullptr)
{
    const int nU = 10 * mesh.nCells;
    const int nP = (int)mesh.P.size();
    std::vector<std::map<int,double>> rows(nU);

    // Volume contribution: Ap_d(v_i, p_a) = - int_K lambda_a d(v_i)/dx_d dV.
    for (int c=0; c<mesh.nCells; ++c) {
        const TetP2Geom& K = tets[c];
        for (const auto& qp: qt) {
            std::array<double,3> G[10];
            p2_tet_grad(K, qp.lam, G);
            const double w = K.vol * qp.w;
            for (int i=0; i<10; ++i) {
                const int row = 10*c + i;
                const double gradTest = G[i][dir];
                for (int a=0; a<4; ++a) rows[row][K.v[a]] += -qp.lam[a] * gradTest * w;
            }
        }
    }

    faceMappingFailures = 0;
    // Face flux contribution: Ap_d += int_F p_hat n_d v.
    for (int f=0; f<mesh.nFaces; ++f) {
        const int P = mesh.owner[f];
        const bool interior = (f < mesh.nInternalFaces);
        const int N = interior ? mesh.neigh[f] : -1;
        const TetP2Geom& KP = tets[P];
        const TetP2Geom* KN = interior ? &tets[N] : nullptr;
        const double nd = mesh.nf[f][dir];
        const double area = mesh.Af[f];

        for (const auto& fq: qf) {
            std::array<double,4> lamP{};
            if (!face_lam_on_tet(mesh.faces[f], KP, fq.mu, lamP)) { faceMappingFailures++; continue; }
            double NP2[10]; p2_tet_basis(lamP, NP2);
            const double w = area * fq.w;
            if (interior) {
                std::array<double,4> lamN{};
                if (!face_lam_on_tet(mesh.faces[f], *KN, fq.mu, lamN)) { faceMappingFailures++; continue; }
                double NN2[10]; p2_tet_basis(lamN, NN2);
                for (int i=0; i<10; ++i) {
                    const int rowP = 10*P + i;
                    for (int a=0; a<4; ++a) {
                        rows[rowP][KP.v[a]] += 0.5 * w * nd * NP2[i] * lamP[a];
                        rows[rowP][KN->v[a]] += 0.5 * w * nd * NP2[i] * lamN[a];
                    }
                }
                for (int i=0; i<10; ++i) {
                    const int rowN = 10*N + i;
                    for (int a=0; a<4; ++a) {
                        rows[rowN][KP.v[a]] += -0.5 * w * nd * NN2[i] * lamP[a];
                        rows[rowN][KN->v[a]] += -0.5 * w * nd * NN2[i] * lamN[a];
                    }
                }
            } else {
                if (boundaryPressureFaceMask && !(*boundaryPressureFaceMask)[f]) {
                    continue;
                }
                for (int i=0; i<10; ++i) {
                    const int rowP = 10*P + i;
                    for (int a=0; a<4; ++a) rows[rowP][KP.v[a]] += w * nd * NP2[i] * lamP[a];
                }
            }
        }
    }

    Ap = rows_to_rectcsr(nU, nP, rows);
}

static void apply_rect(const RectCSR& A, const std::vector<double>& x, std::vector<double>& y)
{
    y.assign(A.nRows, 0.0);
    for (int r=0; r<A.nRows; ++r) {
        double s = 0.0;
        for (int p=A.rowOffsets[r]; p<A.rowOffsets[r+1]; ++p) s += A.values[p] * x[A.cols[p]];
        y[r] = s;
    }
}


static RectCSR negated_rectcsr(const RectCSR& A)
{
    RectCSR B = A;
    for (double& v : B.values) v = -v;
    return B;
}

static void apply_neg_transpose_rect(const RectCSR& A, const std::vector<double>& x, std::vector<double>& y)
{
    y.assign(A.nCols, 0.0);
    for (int r=0; r<A.nRows; ++r) {
        for (int p=A.rowOffsets[r]; p<A.rowOffsets[r+1]; ++p) y[A.cols[p]] -= A.values[p] * x[r];
    }
}

static void apply_pos_transpose_rect(const RectCSR& A, const std::vector<double>& x, std::vector<double>& y)
{
    y.assign(A.nCols, 0.0);
    for (int r=0; r<A.nRows; ++r) {
        for (int p=A.rowOffsets[r]; p<A.rowOffsets[r+1]; ++p) y[A.cols[p]] += A.values[p] * x[r];
    }
}

static std::vector<std::array<std::array<double,10>,10>> compute_p2_mass_inverses(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& qt)
{
    std::vector<std::array<std::array<double,10>,10>> invM(mesh.nCells);
    for (int c=0; c<mesh.nCells; ++c) {
        double aug[10][20]{};
        for (const auto& qp: qt) {
            double N[10]; p2_tet_basis(qp.lam, N);
            const double w = tets[c].vol * qp.w;
            for (int i=0; i<10; ++i) for (int j=0; j<10; ++j) aug[i][j] += w * N[i] * N[j];
        }
        for (int i=0; i<10; ++i) aug[i][10+i] = 1.0;
        for (int k=0; k<10; ++k) {
            int piv = k;
            double best = std::abs(aug[k][k]);
            for (int r=k+1; r<10; ++r) if (std::abs(aug[r][k]) > best) { best = std::abs(aug[r][k]); piv = r; }
            if (best < 1e-300) throw std::runtime_error("P2 mass inverse failed: singular local mass matrix");
            if (piv != k) for (int j=0; j<20; ++j) std::swap(aug[k][j], aug[piv][j]);
            const double invPivot = 1.0 / aug[k][k];
            for (int j=0; j<20; ++j) aug[k][j] *= invPivot;
            for (int r=0; r<10; ++r) if (r != k) {
                const double f = aug[r][k];
                if (std::abs(f) == 0.0) continue;
                for (int j=0; j<20; ++j) aug[r][j] -= f * aug[k][j];
            }
        }
        for (int i=0; i<10; ++i) for (int j=0; j<10; ++j) invM[c][i][j] = aug[i][10+j];
    }
    return invM;
}

static void assemble_lpmass_schur(
    int nP,
    const Mesh& mesh,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    CSRPattern& lpPat,
    std::vector<double>& LpValues)
{
    std::vector<std::map<int,double>> rows(nP);
    auto add_direction = [&](const RectCSR& Ap) {
        for (int c=0; c<mesh.nCells; ++c) {
            for (int a=0; a<10; ++a) {
                const int rowA = 10*c + a;
                for (int b=0; b<10; ++b) {
                    const int rowB = 10*c + b;
                    const double mij = invM[c][a][b];
                    if (std::abs(mij) <= 1e-300) continue;
                    for (int pa=Ap.rowOffsets[rowA]; pa<Ap.rowOffsets[rowA+1]; ++pa) {
                        const int colA = Ap.cols[pa];
                        const double va = Ap.values[pa];
                        if (std::abs(va) <= 1e-300) continue;
                        for (int pb=Ap.rowOffsets[rowB]; pb<Ap.rowOffsets[rowB+1]; ++pb) {
                            const int colB = Ap.cols[pb];
                            const double vb = Ap.values[pb];
                            if (std::abs(vb) <= 1e-300) continue;
                            rows[colA][colB] += va * mij * vb;
                        }
                    }
                }
            }
        }
    };
    add_direction(Apx);
    add_direction(Apy);
    add_direction(Apz);
    lpPat = rows_to_csrpattern(nP, rows, LpValues);
}


static double schur_safe_recip(double d, double scale)
{
    const double tiny = 1e-14 * std::max(scale, 1.0);
    if (std::abs(d) >= tiny) return 1.0 / d;
    return ((d < 0.0) ? -1.0 : 1.0) / tiny;
}

static std::vector<double> make_scalar_schur_inverse_from_Arel(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::string& mode)
{
    std::vector<double> d(pat.nRows, 0.0);
    const bool rowMode =
        (mode == "rowsumschur" || mode == "rowsschur" ||
         mode == "rowsum" || mode == "row");

    for (int r=0; r<pat.nRows; ++r) {
        double v = 0.0;
        if (rowMode) {
            for (int q=pat.rowOffsets[r]; q<pat.rowOffsets[r+1]; ++q) v += A[q];
        } else {
            for (int q=pat.rowOffsets[r]; q<pat.rowOffsets[r+1]; ++q) {
                if ((int)pat.cols[q] == r) {
                    v = A[q];
                    break;
                }
            }
        }
        d[r] = v;
    }

    double scale = 1.0;
    for (double v: d) scale = std::max(scale, std::abs(v));

    std::vector<double> h(pat.nRows, 0.0);
    for (int i=0; i<pat.nRows; ++i) h[i] = schur_safe_recip(d[i], scale);
    return h;
}

static void assemble_lp_scalar_schur(
    int nP,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<double>& hInv,
    CSRPattern& lpPat,
    std::vector<double>& LpValues)
{
    std::vector<std::map<int,double>> rows(nP);

    auto add_direction = [&](const RectCSR& Ap) {
        for (int r=0; r<Ap.nRows; ++r) {
            const double h = hInv[r];
            if (!std::isfinite(h) || std::abs(h) <= 1e-300) continue;

            for (int pa=Ap.rowOffsets[r]; pa<Ap.rowOffsets[r+1]; ++pa) {
                const int colA = Ap.cols[pa];
                const double va = Ap.values[pa];
                if (std::abs(va) <= 1e-300) continue;

                for (int pb=Ap.rowOffsets[r]; pb<Ap.rowOffsets[r+1]; ++pb) {
                    const int colB = Ap.cols[pb];
                    const double vb = Ap.values[pb];
                    if (std::abs(vb) <= 1e-300) continue;
                    rows[colA][colB] += va * h * vb;
                }
            }
        }
    };

    add_direction(Apx);
    add_direction(Apy);
    add_direction(Apz);

    lpPat = rows_to_csrpattern(nP, rows, LpValues);
}

static void apply_csr(const CSRPattern& pat, const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y)
{
    y.assign(pat.nRows, 0.0);
    for (int r=0; r<pat.nRows; ++r) {
        double s = 0.0;
        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) s += A[p] * x[(int)pat.cols[p]];
        y[r] = s;
    }
}


static void print_velocity_solve_audit(
    const char* comp,
    int simpleIt,
    const std::string& solverName,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::vector<double>& xIn,
    const std::vector<double>& xOut,
    const std::vector<unsigned char>* lockedRows,
    const HypreSolveInfo& info,
    int maxit,
    double relTol,
    double absTol)
{
    if ((int)rhs.size() != pat.nRows || (int)xOut.size() != pat.nRows ||
        (int)xIn.size() != pat.nRows || (int)A.size() != pat.nnz) {
        std::printf("    velocitySolveAudit: it=%d comp=%s solver=%s SIZE_MISMATCH rows=%d nnz=%d rhs=%zu xIn=%zu xOut=%zu A=%zu\n",
            simpleIt, comp, solverName.c_str(), pat.nRows, pat.nnz,
            rhs.size(), xIn.size(), xOut.size(), A.size());
        return;
    }

    std::vector<double> Ax;
    apply_csr(pat, A, xOut, Ax);

    double r2All = 0.0;
    double r2Free = 0.0;
    double r2Locked = 0.0;
    double rhs2All = 0.0;
    double rhs2Free = 0.0;
    double x2All = 0.0;
    double x2Free = 0.0;
    double xIn2All = 0.0;
    double xIn2Free = 0.0;
    double dx2All = 0.0;
    double dx2Free = 0.0;
    double dx2Locked = 0.0;
    double maxAbsAll = 0.0;
    double maxAbsFree = 0.0;
    double maxAbsLocked = 0.0;
    double lockedXMax = 0.0;
    int nLocked = 0;
    int nFree = 0;

    for (int i=0; i<pat.nRows; ++i) {
        const bool locked = (lockedRows && i < (int)lockedRows->size() && (*lockedRows)[i]);
        const double ri = rhs[i] - Ax[i];
        const double dxi = xOut[i] - xIn[i];
        const double ari = std::abs(ri);

        r2All += ri * ri;
        rhs2All += rhs[i] * rhs[i];
        x2All += xOut[i] * xOut[i];
        xIn2All += xIn[i] * xIn[i];
        dx2All += dxi * dxi;
        maxAbsAll = std::max(maxAbsAll, ari);

        if (locked) {
            ++nLocked;
            r2Locked += ri * ri;
            dx2Locked += dxi * dxi;
            maxAbsLocked = std::max(maxAbsLocked, ari);
            lockedXMax = std::max(lockedXMax, std::abs(xOut[i]));
        } else {
            ++nFree;
            r2Free += ri * ri;
            rhs2Free += rhs[i] * rhs[i];
            x2Free += xOut[i] * xOut[i];
            xIn2Free += xIn[i] * xIn[i];
            dx2Free += dxi * dxi;
            maxAbsFree = std::max(maxAbsFree, ari);
        }
    }

    const double trueAbsAll = std::sqrt(std::max(0.0, r2All));
    const double trueAbsFree = std::sqrt(std::max(0.0, r2Free));
    const double lockedResNorm = std::sqrt(std::max(0.0, r2Locked));
    const double rhsNormAll = std::sqrt(std::max(0.0, rhs2All));
    const double rhsNormFree = std::sqrt(std::max(0.0, rhs2Free));
    const double xNormAll = std::sqrt(std::max(0.0, x2All));
    const double xNormFree = std::sqrt(std::max(0.0, x2Free));
    const double xInNormAll = std::sqrt(std::max(0.0, xIn2All));
    const double xInNormFree = std::sqrt(std::max(0.0, xIn2Free));
    const double dxNormAll = std::sqrt(std::max(0.0, dx2All));
    const double dxNormFree = std::sqrt(std::max(0.0, dx2Free));
    const double dxNormLocked = std::sqrt(std::max(0.0, dx2Locked));
    const double trueRelAll = trueAbsAll / std::max(rhsNormAll, 1e-300);
    const double trueRelFree = trueAbsFree / std::max(rhsNormFree, 1e-300);
    const double dxRelAll = dxNormAll / std::max(xInNormAll, 1e-300);
    const double dxRelFree = dxNormFree / std::max(xInNormFree, 1e-300);

    const bool relStillHigh = (relTol > 0.0 && info.finalRelResNorm > relTol && trueRelAll > relTol);
    const bool absStillHigh = (absTol > 0.0 && trueAbsAll > absTol);
    const int hitMaxit = (info.iterations >= maxit && (relStillHigh || absStillHigh || (relTol <= 0.0 && absTol <= 0.0))) ? 1 : 0;

    std::printf(
        "    velocitySolveAudit: it=%d comp=%s solver=%s reportedIts=%d reportedFinal=%.17e hitMaxit=%d "
        "trueRelAll=%.17e trueRelFree=%.17e trueAbsAll=%.17e trueAbsFree=%.17e "
        "maxAbsAll=%.17e maxAbsFree=%.17e rhsNormAll=%.17e rhsNormFree=%.17e "
        "xNormAll=%.17e xNormFree=%.17e dxNormAll=%.17e dxNormFree=%.17e "
        "dxRelAll=%.17e dxRelFree=%.17e lockedRows=%d freeRows=%d "
        "lockedResNorm=%.17e lockedResMax=%.17e lockedXMax=%.17e lockedDxNorm=%.17e\n",
        simpleIt, comp, solverName.c_str(), info.iterations, info.finalRelResNorm, hitMaxit,
        trueRelAll, trueRelFree, trueAbsAll, trueAbsFree,
        maxAbsAll, maxAbsFree, rhsNormAll, rhsNormFree,
        xNormAll, xNormFree, dxNormAll, dxNormFree,
        dxRelAll, dxRelFree, nLocked, nFree,
        lockedResNorm, maxAbsLocked, lockedXMax, dxNormLocked);
}

static double max_csr_row_sum_abs(const CSRPattern& pat, const std::vector<double>& A)
{
    double m = 0.0;
    for (int r=0; r<pat.nRows; ++r) {
        double s = 0.0;
        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) s += A[p];
        m = std::max(m, std::abs(s));
    }
    return m;
}



static int append_pressure_zero_rows_to_pin_set(
    const char* name,
    const Mesh& mesh,
    const CSRPattern& pat,
    const std::vector<double>& vals,
    std::vector<int>& pins,
    double diagTol,
    double offTol,
    int maxPrint)
{
    const int n = pat.nRows;
    std::vector<char> already(n, 0);

    for (int p : pins) {
        if (p >= 0 && p < n) already[p] = 1;
    }

    int added = 0;

    for (int i=0; i<n; ++i) {
        double diag = 0.0;
        double absOff = 0.0;
        bool finite = true;
        const int rowNnz = pat.rowOffsets[i+1] - pat.rowOffsets[i];

        for (int kk=pat.rowOffsets[i]; kk<pat.rowOffsets[i+1]; ++kk) {
            const int j = pat.cols[kk];
            const double a = vals[kk];

            if (!std::isfinite(a)) finite = false;

            if (j == i) diag += a;
            else absOff += std::abs(a);
        }

        const bool bad =
            (!finite) ||
            (rowNnz <= 0) ||
            (std::abs(diag) <= diagTol && absOff <= offTol);

        if (bad && !already[i]) {
            pins.push_back(i);
            already[i] = 1;
            added++;

            if (added <= maxPrint) {
                double x = 0.0, y = 0.0, z = 0.0;
                if (i >= 0 && i < (int)mesh.P.size()) {
                    x = mesh.P[i][0];
                    y = mesh.P[i][1];
                    z = mesh.P[i][2];
                }

                std::printf(
                    "pressureRepairZeroRows: %s adding pressure pin row=%d nnz=%d diag=%.17e absOff=%.17e xyz=(%.17e,%.17e,%.17e)\n",
                    name, i, rowNnz, diag, absOff, x, y, z);
            }
        }
    }

    if (added > 0) {
        std::printf("pressureRepairZeroRows: %s addedPins=%d totalPinsNow=%zu diagTol=%.3e offTol=%.3e\n",
                    name, added, pins.size(), diagTol, offTol);
    }

    return added;
}


static void audit_pressure_csr_rows_for_amg(
    const char* name,
    const Mesh& mesh,
    const CSRPattern& pat,
    const std::vector<double>& vals,
    int maxPrint)
{
    const int n = pat.nRows;

    int badDiag = 0;
    int tinyDiag = 0;
    int nnz1 = 0;
    int noOffdiag = 0;
    int weakOffdiag = 0;
    int nonfinite = 0;

    double minDiag = std::numeric_limits<double>::infinity();
    double maxDiag = 0.0;
    double minAbsOff = std::numeric_limits<double>::infinity();
    double maxAbsOff = 0.0;
    double minOffRatio = std::numeric_limits<double>::infinity();

    struct RowInfo {
        int row;
        int nnz;
        double diag;
        double absOff;
        double ratio;
        double x, y, z;
    };

    std::vector<RowInfo> suspects;

    for (int i=0; i<n; ++i) {
        double diag = 0.0;
        double absOff = 0.0;
        int rowNnz = pat.rowOffsets[i+1] - pat.rowOffsets[i];

        for (int kk=pat.rowOffsets[i]; kk<pat.rowOffsets[i+1]; ++kk) {
            const int j = pat.cols[kk];
            const double a = vals[kk];

            if (!std::isfinite(a)) nonfinite++;

            if (j == i) diag += a;
            else absOff += std::abs(a);
        }

        const double absDiag = std::abs(diag);
        minDiag = std::min(minDiag, absDiag);
        maxDiag = std::max(maxDiag, absDiag);
        minAbsOff = std::min(minAbsOff, absOff);
        maxAbsOff = std::max(maxAbsOff, absOff);

        if (!std::isfinite(diag) || !std::isfinite(absOff)) nonfinite++;
        if (diag <= 0.0) badDiag++;
        if (absDiag < 1e-30) tinyDiag++;
        if (rowNnz <= 1) nnz1++;
        if (absOff <= 1e-30) noOffdiag++;

        const double ratio = absOff / std::max(absDiag, 1e-300);
        minOffRatio = std::min(minOffRatio, ratio);

        if (ratio < 1e-10 && rowNnz > 1) weakOffdiag++;

        if ((rowNnz <= 1 || absOff <= 1e-30 || ratio < 1e-10 || diag <= 0.0) &&
            (int)suspects.size() < maxPrint) {
            double x = 0.0, y = 0.0, z = 0.0;
            if (i >= 0 && i < (int)mesh.P.size()) {
                x = mesh.P[i][0];
                y = mesh.P[i][1];
                z = mesh.P[i][2];
            }
            suspects.push_back({i, rowNnz, diag, absOff, ratio, x, y, z});
        }
    }

    std::printf("\n--- Pressure CSR AMG row audit: %s ---\n", name);
    std::printf("rows=%d nnz=%d\n", pat.nRows, pat.nnz);
    std::printf("diag abs min/max       = %.17e %.17e\n", minDiag, maxDiag);
    std::printf("offdiag abs min/max    = %.17e %.17e\n", minAbsOff, maxAbsOff);
    std::printf("min offdiag/diag ratio = %.17e\n", minOffRatio);
    std::printf("badDiag=%d tinyDiag=%d nnz1=%d noOffdiag=%d weakOffdiag=%d nonfinite=%d\n",
                badDiag, tinyDiag, nnz1, noOffdiag, weakOffdiag, nonfinite);

    for (const RowInfo& r : suspects) {
        std::printf("  suspect row=%d nnz=%d diag=%.17e absOff=%.17e ratio=%.17e xyz=(%.17e,%.17e,%.17e)\n",
                    r.row, r.nnz, r.diag, r.absOff, r.ratio, r.x, r.y, r.z);
    }
}


static double max_csr_symmetry_error(const CSRPattern& pat, const std::vector<double>& A)
{
    std::vector<std::map<int,double>> rows(pat.nRows);
    for (int r=0; r<pat.nRows; ++r) for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) rows[r][(int)pat.cols[p]] = A[p];
    double m = 0.0;
    for (int r=0; r<pat.nRows; ++r) {
        for (const auto& kv: rows[r]) {
            const int c = kv.first;
            double aji = 0.0;
            auto it = rows[c].find(r);
            if (it != rows[c].end()) aji = it->second;
            m = std::max(m, std::abs(kv.second - aji));
        }
    }
    return m;
}

static void add_diag_if_missing(CSRPattern& pat, std::vector<double>& A)
{
    bool missing = false;
    for (int r=0; r<pat.nRows; ++r) if (pat.diagPos[r] < 0) missing = true;
    if (!missing) return;
    std::vector<std::map<int,double>> rows(pat.nRows);
    for (int r=0; r<pat.nRows; ++r) {
        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) rows[r][(int)pat.cols[p]] = A[p];
        rows[r][r] += 0.0;
    }
    pat = rows_to_csrpattern(pat.nRows, rows, A);
}

static void pin_csr_symmetric_unit(CSRPattern& pat, std::vector<double>& A, std::vector<double>& rhs, int pin)
{
    add_diag_if_missing(pat, A);
    for (int r=0; r<pat.nRows; ++r) {
        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) {
            const int c = (int)pat.cols[p];
            if (r == pin || c == pin) A[p] = 0.0;
        }
    }
    if (pat.diagPos[pin] < 0) throw std::runtime_error("pin_csr_symmetric_unit: missing diagonal after insertion");
    A[pat.diagPos[pin]] = 1.0;
    rhs[pin] = 0.0;
}

static void zero_values_at_indices(std::vector<double>& x, const std::vector<int>& idx)
{
    for (int i : idx) {
        if (i >= 0 && i < (int)x.size()) x[i] = 0.0;
    }
}


static void force_csr_structural_identity_pins(
    CSRPattern& pat,
    std::vector<double>& A,
    const std::vector<int>& pins)
{
    if (pins.empty()) return;

    std::vector<char> isPinned(pat.nRows, 0);
    for (int pin : pins) {
        if (pin >= 0 && pin < pat.nRows) isPinned[pin] = 1;
    }

    std::vector<std::map<int,double>> rows(pat.nRows);

    for (int r=0; r<pat.nRows; ++r) {
        if (isPinned[r]) {
            rows[r][r] = 1.0;
            continue;
        }

        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) {
            const int c = pat.cols[p];
            if (c >= 0 && c < pat.nRows && isPinned[c]) continue;
            rows[r][c] += A[p];
        }
    }

    for (int pin : pins) {
        if (pin >= 0 && pin < pat.nRows) {
            rows[pin].clear();
            rows[pin][pin] = 1.0;
        }
    }

    pat = rows_to_csrpattern(pat.nRows, rows, A);
}


static void pin_csr_symmetric_zero_set(
    CSRPattern& pat,
    std::vector<double>& A,
    const std::vector<int>& pins)
{
    if (pins.empty()) return;

    add_diag_if_missing(pat, A);

    std::vector<char> isPinned(pat.nRows, 0);
    for (int pin : pins) {
        if (pin >= 0 && pin < pat.nRows) isPinned[pin] = 1;
    }

    for (int r=0; r<pat.nRows; ++r) {
        for (int p=pat.rowOffsets[r]; p<pat.rowOffsets[r+1]; ++p) {
            const int c = (int)pat.cols[p];
            if (isPinned[r] || (c >= 0 && c < pat.nRows && isPinned[c])) {
                A[p] = 0.0;
            }
        }
    }

    for (int pin : pins) {
        if (pin < 0 || pin >= pat.nRows) continue;
        if (pat.diagPos[pin] < 0) {
            throw std::runtime_error("pin_csr_symmetric_zero_set: missing diagonal after insertion");
        }
        A[pat.diagPos[pin]] = 1.0;
    }
}


static std::vector<HYPRE_Complex> to_hypre_complex_vec(const std::vector<double>& x)
{
    std::vector<HYPRE_Complex> y(x.size());
    for (std::size_t i=0; i<x.size(); ++i) y[i] = static_cast<HYPRE_Complex>(x[i]);
    return y;
}

static std::vector<double> from_hypre_complex_vec(const std::vector<HYPRE_Complex>& x)
{
    std::vector<double> y(x.size());
    for (std::size_t i=0; i<x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

static CSRPattern build_dg2_scalar_pattern(const Mesh& mesh)
{
    CSRPattern pat;
    pat.nRows = 10 * mesh.nCells;

    std::vector<std::set<int>> rowSets(pat.nRows);
    for (int c = 0; c < mesh.nCells; ++c) {
        std::set<int> coupledCells;
        coupledCells.insert(c);
        for (int nb : mesh.cellNbrs[c]) coupledCells.insert(nb);

        for (int li = 0; li < 10; ++li) {
            const int row = 10*c + li;
            for (int cc : coupledCells) {
                for (int lj = 0; lj < 10; ++lj) rowSets[row].insert(10*cc + lj);
            }
        }
    }

    pat.rows.resize(pat.nRows);
    pat.ncols.resize(pat.nRows);
    pat.rowOffsets.resize(pat.nRows + 1, 0);
    for (int r = 0; r < pat.nRows; ++r) {
        pat.rows[r] = static_cast<HYPRE_BigInt>(r);
        pat.ncols[r] = static_cast<HYPRE_Int>(rowSets[r].size());
        pat.rowOffsets[r+1] = pat.rowOffsets[r] + (int)rowSets[r].size();
    }

    pat.nnz = pat.rowOffsets.back();
    pat.cols.resize(pat.nnz);
    pat.diagPos.assign(pat.nRows, -1);
    for (int r = 0; r < pat.nRows; ++r) {
        int k = 0;
        const int off = pat.rowOffsets[r];
        for (int col : rowSets[r]) {
            pat.cols[off+k] = static_cast<HYPRE_BigInt>(col);
            if (col == r) pat.diagPos[r] = off+k;
            k++;
        }
    }
    return pat;
}

static int find_col_pos(const CSRPattern& pat, int row, int col)
{
    const int b = pat.rowOffsets[row];
    const int e = pat.rowOffsets[row+1];
    auto it = std::lower_bound(pat.cols.begin()+b, pat.cols.begin()+e, static_cast<HYPRE_BigInt>(col));
    if (it == pat.cols.begin()+e || *it != static_cast<HYPRE_BigInt>(col)) {
        throw std::runtime_error("CSR pattern missing entry row=" + std::to_string(row) + " col=" + std::to_string(col));
    }
    return (int)(it - pat.cols.begin());
}

static void add_A(const CSRPattern& pat, std::vector<double>& A, int row, int col, double val)
{
    A[find_col_pos(pat, row, col)] += val;
}


static double gMmsAmp = 1.0;
static double gVelocityBlockJacobiShift = 0.0;
static double gVelocityBlockJacobiPivotFloor = 1e-30;
static int gFlowIsNse = 0;

// Momentum DG convection flux.  Mass/continuity flux stays central always.
// Options: lf/upwind/rusanov (default) or central/centered.
static std::string gMomentumFluxMode = "lf";

static bool momentum_flux_is_central()
{
    return gMomentumFluxMode == "central" ||
           gMomentumFluxMode == "centered" ||
           gMomentumFluxMode == "centre" ||
           gMomentumFluxMode == "central_differencing";
}


static std::string gCaseMode = "mms";
static int gCaseIsChannel = 0;
static double gChannelInletUx = 1.0;
static int gChannelInletNormalVelocity = 1;

// Inlet profile mode:
//   0 = uniform speed magnitude gChannelInletUx
//   1 = parabolic box profile in y-z plane,
//       speed = Um*(1-yhat^2)*(1-zhat^2).
// This matches the old FV cylinder case:
//   parabolic_box_inlet 0.41 0.45 y z average_patch_normal
// whose area-average speed is (4/9)*Um = 0.2 for Um=0.45.
static int gChannelInletProfileMode = 0;
static double gChannelParabolicBoxWidth = 0.41;
static double gChannelParabolicBoxCy = 0.0;
static double gChannelParabolicBoxCz = 0.0;
static double gChannelParabolicBoxHy = 0.205;
static double gChannelParabolicBoxHz = 0.205;
static int gChannelParabolicBoxReady = 0;
static int gChannelPressureOutlet = 0;
static int gChannelPressureOutletSinglePin = 0;
static int gChannelPressureOutletDirichletAll = 1;
static int gChannelAdjustOutletFlux = 0;
static int gChannelOutletCompatibility = 1;



// Patch-name/range mode for real OpenFOAM polyMesh boundaries.
// Reuses the existing channel machinery:
//   0 = wall, 1 = inlet, 2 = outlet.
static int gChannelUsePatchBC = 0;
static std::string gChannelInletPatchName;
static std::string gChannelOutletPatchName;
static std::string gChannelWallPatchName;

// Fixed-vector inlet direction used when -channelNormalInlet 0.
// Cube-channel default is (1,0,0); pipe mode default is (0,0,1).
static double gChannelInletDirX = 1.0;
static double gChannelInletDirY = 0.0;
static double gChannelInletDirZ = 0.0;

// Optional explicit OpenFOAM boundary ranges. These avoid relying on
// mesh.patchStartFace/mesh.patchNFaces if the mesh reader does not fill them.
static int gChannelInletStartFace  = -1;
static int gChannelInletNFaces     = 0;
static int gChannelOutletStartFace = -1;
static int gChannelOutletNFaces    = 0;
static int gChannelWallStartFace   = -1;
static int gChannelWallNFaces      = 0;


static double gChannelXMin = 0.0;
static double gChannelXMax = 1.0;
static double gChannelFaceTol = 1e-8;

static double channel_x_tol()
{
    return gChannelFaceTol * std::max(1.0, std::abs(gChannelXMax - gChannelXMin));
}

static double boundary_face_centroid_x(const Mesh& mesh, int f)
{
    double x = 0.0;
    for (int a=0; a<3; ++a) {
        x += mesh.P[mesh.faces[f][a]][0];
    }
    return x / 3.0;
}

static bool channel_is_inlet_face(const Mesh& mesh, int f)
{
    if (!gCaseIsChannel || f < mesh.nInternalFaces) return false;

    // Patch/range mode: make even old direct calls patch-aware.
    if (gChannelUsePatchBC) {
        if (gChannelInletStartFace >= 0 && gChannelInletNFaces > 0 &&
            f >= gChannelInletStartFace && f < gChannelInletStartFace + gChannelInletNFaces) {
            return true;
        }

        if (!gChannelInletPatchName.empty()) {
            std::size_t n = mesh.patchNames.size();
            if (mesh.patchStartFace.size() < n) n = mesh.patchStartFace.size();
            if (mesh.patchNFaces.size() < n) n = mesh.patchNFaces.size();

            for (std::size_t pp=0; pp<n; ++pp) {
                if (mesh.patchNames[pp] != gChannelInletPatchName) continue;
                const int start = mesh.patchStartFace[pp];
                const int end   = start + mesh.patchNFaces[pp];
                return (f >= start && f < end);
            }
        }

        return false;
    }

    const double xc = boundary_face_centroid_x(mesh, f);
    return std::abs(xc - gChannelXMin) <= channel_x_tol();
}



static bool channel_is_outlet_face(const Mesh& mesh, int f)
{
    if (!gCaseIsChannel || f < mesh.nInternalFaces) return false;

    // Patch/range mode: make even old direct calls patch-aware.
    if (gChannelUsePatchBC) {
        if (gChannelOutletStartFace >= 0 && gChannelOutletNFaces > 0 &&
            f >= gChannelOutletStartFace && f < gChannelOutletStartFace + gChannelOutletNFaces) {
            return true;
        }

        if (!gChannelOutletPatchName.empty()) {
            std::size_t n = mesh.patchNames.size();
            if (mesh.patchStartFace.size() < n) n = mesh.patchStartFace.size();
            if (mesh.patchNFaces.size() < n) n = mesh.patchNFaces.size();

            for (std::size_t pp=0; pp<n; ++pp) {
                if (mesh.patchNames[pp] != gChannelOutletPatchName) continue;
                const int start = mesh.patchStartFace[pp];
                const int end   = start + mesh.patchNFaces[pp];
                return (f >= start && f < end);
            }
        }

        return false;
    }

    const double xc = boundary_face_centroid_x(mesh, f);
    return std::abs(xc - gChannelXMax) <= channel_x_tol();
}



// 0=wall, 1=inlet, 2=outlet.  This is intentionally coordinate fallback based;
// later it can be replaced by OpenFOAM patch-name classification.


static bool channel_face_in_explicit_range(int f, int startFace, int nFaces)
{
    return (startFace >= 0 && nFaces > 0 && f >= startFace && f < startFace + nFaces);
}

static bool channel_face_in_named_patch(const Mesh& mesh, int f, const std::string& patchName)
{
    if (patchName.empty()) return false;

    // Some Mesh reader paths may fill patchNames but not these parallel
    // start/nFaces arrays. Bounds-check before indexing to avoid segfaults.
    const std::size_t n =
        std::min(mesh.patchNames.size(),
        std::min(mesh.patchStartFace.size(), mesh.patchNFaces.size()));

    for (std::size_t p=0; p<n; ++p) {
        if (mesh.patchNames[p] != patchName) continue;

        const int start = mesh.patchStartFace[p];
        const int end   = start + mesh.patchNFaces[p];
        return (f >= start && f < end);
    }

    return false;
}



static bool find_channel_patch_range_from_mesh(
    const Mesh& mesh,
    const std::string& patchName,
    int& startFace,
    int& nFaces)
{
    startFace = -1;
    nFaces = 0;

    if (patchName.empty()) return false;

    for (std::size_t p = 0; p < mesh.patchNames.size(); ++p) {
        if (mesh.patchNames[p] != patchName) continue;

        // Preferred path: mesh reader filled OpenFOAM boundary ranges.
        if (p < mesh.patchStartFace.size() &&
            p < mesh.patchNFaces.size() &&
            mesh.patchStartFace[p] >= 0 &&
            mesh.patchNFaces[p] > 0) {
            startFace = mesh.patchStartFace[p];
            nFaces    = mesh.patchNFaces[p];
            return true;
        }

        // Fallback path: infer contiguous range from bPatch labels.
        // OpenFOAM boundary patches should be contiguous, but this also warns
        // if a strange reader/path creates a non-contiguous label set.
        const int patchId = static_cast<int>(p) + 1;
        int first = -1;
        int last  = -1;
        int count = 0;

        const int bPatchSize = static_cast<int>(mesh.bPatch.size());
        for (int f = mesh.nInternalFaces; f < mesh.nFaces && f < bPatchSize; ++f) {
            if (mesh.bPatch[f] != patchId) continue;
            if (first < 0) first = f;
            last = f;
            ++count;
        }

        if (count > 0) {
            startFace = first;
            nFaces = last - first + 1;
            if (nFaces != count) {
                std::printf("WARNING: patch %s has non-contiguous bPatch labels: first=%d last=%d count=%d impliedNFaces=%d\n",
                    patchName.c_str(), first, last, count, nFaces);
            }
            return true;
        }

        return false;
    }

    return false;
}

static void resolve_channel_patch_ranges_from_mesh(const Mesh& mesh)
{
    if (!gCaseIsChannel || !gChannelUsePatchBC) return;

    auto resolve_one =
        [&](const char* role,
            const std::string& patchName,
            int& startFace,
            int& nFaces,
            bool mandatory)
        {
            if (patchName.empty()) return;

            if (startFace >= 0 && nFaces > 0) {
                std::printf("channelPatchRange: role=%s patch=%s startFace=%d nFaces=%d source=manual-cli\n",
                    role, patchName.c_str(), startFace, nFaces);
                return;
            }

            int s = -1;
            int n = 0;
            if (!find_channel_patch_range_from_mesh(mesh, patchName, s, n)) {
                std::string msg = std::string("Could not auto-resolve OpenFOAM boundary patch '") +
                                  patchName + "' for role=" + role +
                                  ". Check constant/polyMesh/boundary or patch spelling.";
                if (mandatory) throw std::runtime_error(msg);
                std::printf("WARNING: %s\n", msg.c_str());
                return;
            }

            startFace = s;
            nFaces = n;

            std::printf("channelPatchRange: role=%s patch=%s startFace=%d nFaces=%d source=auto-boundary\n",
                role, patchName.c_str(), startFace, nFaces);
        };

    resolve_one("inlet",  gChannelInletPatchName,  gChannelInletStartFace,  gChannelInletNFaces,  true);
    resolve_one("outlet", gChannelOutletPatchName, gChannelOutletStartFace, gChannelOutletNFaces, true);
    resolve_one("wall",   gChannelWallPatchName,   gChannelWallStartFace,   gChannelWallNFaces,   !gChannelWallPatchName.empty());

    auto ranges_overlap = [](int a0, int an, int b0, int bn) -> bool {
        if (a0 < 0 || b0 < 0 || an <= 0 || bn <= 0) return false;
        const int a1 = a0 + an;
        const int b1 = b0 + bn;
        return std::max(a0, b0) < std::min(a1, b1);
    };

    if (ranges_overlap(gChannelInletStartFace, gChannelInletNFaces,
                       gChannelOutletStartFace, gChannelOutletNFaces)) {
        throw std::runtime_error("Auto-resolved inlet/outlet patch ranges overlap. Check patch names.");
    }
    if (ranges_overlap(gChannelInletStartFace, gChannelInletNFaces,
                       gChannelWallStartFace, gChannelWallNFaces)) {
        std::printf("WARNING: inlet and wall patch ranges overlap; wall trace rows may win at rims.\n");
    }
    if (ranges_overlap(gChannelOutletStartFace, gChannelOutletNFaces,
                       gChannelWallStartFace, gChannelWallNFaces)) {
        std::printf("WARNING: outlet and wall patch ranges overlap; wall trace rows may win at rims.\n");
    }
}



static int channel_boundary_kind(const Mesh& mesh, int f)
{
    if (!gCaseIsChannel || f < mesh.nInternalFaces) return -1;

    if (gChannelUsePatchBC) {
        if (channel_face_in_explicit_range(f, gChannelInletStartFace,  gChannelInletNFaces) ||
            channel_face_in_named_patch(mesh, f, gChannelInletPatchName)) {
            return 1;
        }

        if (channel_face_in_explicit_range(f, gChannelOutletStartFace, gChannelOutletNFaces) ||
            channel_face_in_named_patch(mesh, f, gChannelOutletPatchName)) {
            return 2;
        }

        if (channel_face_in_explicit_range(f, gChannelWallStartFace,   gChannelWallNFaces) ||
            (!gChannelWallPatchName.empty() && channel_face_in_named_patch(mesh, f, gChannelWallPatchName))) {
            return 0;
        }

        // Unknown boundary patches are safest as no-slip walls for now.
        return 0;
    }

    if (channel_is_inlet_face(mesh, f)) return 1;
    if (channel_is_outlet_face(mesh, f)) return 2;
    return 0;
}






static double channel_inlet_profile_speed(const std::array<double,3>& x)
{
    if (gChannelInletProfileMode != 1) return gChannelInletUx;

    const double hy = std::max(std::abs(gChannelParabolicBoxHy), 1e-300);
    const double hz = std::max(std::abs(gChannelParabolicBoxHz), 1e-300);

    const double yhat = (x[1] - gChannelParabolicBoxCy) / hy;
    const double zhat = (x[2] - gChannelParabolicBoxCz) / hz;

    double fy = 1.0 - yhat*yhat;
    double fz = 1.0 - zhat*zhat;

    // Clamp tiny/mesh-rim overshoots to zero.
    if (fy < 0.0) fy = 0.0;
    if (fz < 0.0) fz = 0.0;

    return gChannelInletUx * fy * fz;
}

static void setup_channel_parabolic_box_profile(const Mesh& mesh)
{
    if (!gCaseIsChannel || gChannelInletProfileMode != 1) return;

    double ymin =  std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmin =  std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();
    int nVerts = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        if (channel_boundary_kind(mesh, f) != 1) continue;
        for (int v : mesh.faces[f]) {
            if (v < 0 || v >= (int)mesh.P.size()) continue;
            const auto& xp = mesh.P[v];
            ymin = std::min(ymin, xp[1]);
            ymax = std::max(ymax, xp[1]);
            zmin = std::min(zmin, xp[2]);
            zmax = std::max(zmax, xp[2]);
            ++nVerts;
        }
    }

    if (nVerts <= 0 || !std::isfinite(ymin) || !std::isfinite(zmin)) {
        std::printf("WARNING: parabolic-box inlet profile could not find inlet patch vertices; using center=(0,0), width=%.17g\n",
            gChannelParabolicBoxWidth);
        gChannelParabolicBoxCy = 0.0;
        gChannelParabolicBoxCz = 0.0;
        gChannelParabolicBoxHy = 0.5 * std::max(gChannelParabolicBoxWidth, 1e-12);
        gChannelParabolicBoxHz = 0.5 * std::max(gChannelParabolicBoxWidth, 1e-12);
        gChannelParabolicBoxReady = 0;
        return;
    }

    gChannelParabolicBoxCy = 0.5 * (ymin + ymax);
    gChannelParabolicBoxCz = 0.5 * (zmin + zmax);

    if (gChannelParabolicBoxWidth > 0.0) {
        gChannelParabolicBoxHy = 0.5 * gChannelParabolicBoxWidth;
        gChannelParabolicBoxHz = 0.5 * gChannelParabolicBoxWidth;
    } else {
        gChannelParabolicBoxHy = 0.5 * std::max(ymax - ymin, 1e-12);
        gChannelParabolicBoxHz = 0.5 * std::max(zmax - zmin, 1e-12);
    }

    gChannelParabolicBoxReady = 1;

    const double ubarExpected = (4.0/9.0) * gChannelInletUx;
    std::printf("Channel inlet profile    parabolic_box y-z: Um=%.17g expectedUbar=%.17g center=(%.17g,%.17g) halfWidth=(%.17g,%.17g) inletBBoxY=(%.17g,%.17g) inletBBoxZ=(%.17g,%.17g) vertices=%d\n",
        gChannelInletUx, ubarExpected,
        gChannelParabolicBoxCy, gChannelParabolicBoxCz,
        gChannelParabolicBoxHy, gChannelParabolicBoxHz,
        ymin, ymax, zmin, zmax, nVerts);
}

static std::array<double,3> channel_velocity_bc(
    const std::array<double,3>& x,
    const std::array<double,3>& outwardNormal)
{
    // Coordinate-mode cube channel: only xmin is inlet.
    // Patch-mode pipe/channel: caller already knows this is the inlet patch.
    if (!gChannelUsePatchBC && std::abs(x[0] - gChannelXMin) > channel_x_tol()) {
        return {0.0, 0.0, 0.0};
    }

    const double inletSpeed = channel_inlet_profile_speed(x);

    // Normal inlet: prescribe inflow speed along -outwardNormal.
    if (gChannelInletNormalVelocity) {
        return {
            -inletSpeed * outwardNormal[0],
            -inletSpeed * outwardNormal[1],
            -inletSpeed * outwardNormal[2]
        };
    }

    // Fixed-vector inlet. Default cube channel is (1,0,0);
    // pipe mode can use (0,0,1) with -channelInletDirZ 1.
    double dx = gChannelInletDirX;
    double dy = gChannelInletDirY;
    double dz = gChannelInletDirZ;

    const double mag = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (mag <= 1e-300) {
        dx = 1.0; dy = 0.0; dz = 0.0;
    } else {
        dx /= mag; dy /= mag; dz /= mag;
    }

    return {inletSpeed * dx, inletSpeed * dy, inletSpeed * dz};
}



static std::vector<int> collect_channel_outlet_pressure_nodes(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets)
{
    std::vector<int> rows;
    if (!gCaseIsChannel || !gChannelPressureOutlet) return rows;

    // DG1 pressure rows are element-local: pRow = 4*cell + localVertex.
    // Use the actual TetP2Geom local ordering K.v[a].  Do NOT reconstruct a
    // sorted vertex set here: reconstruct_tets() may swap local vertices to
    // fix orientation, and using a different order pins the wrong DG1 rows.
    std::vector<char> mark((std::size_t)4 * (std::size_t)mesh.nCells, 0);

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        if (channel_boundary_kind(mesh, f) != 2) continue;
        const int c = mesh.owner[f];
        if (c < 0 || c >= mesh.nCells || c >= (int)tets.size()) continue;

        const TetP2Geom& K = tets[c];
        std::set<int> faceVerts(mesh.faces[f].begin(), mesh.faces[f].end());

        for (int a=0; a<4; ++a) {
            if (faceVerts.count(K.v[a])) {
                const int row = 4*c + a;
                if (row >= 0 && row < (int)mark.size()) mark[row] = 1;
            }
        }
    }

    for (int i=0; i<(int)mark.size(); ++i) {
        if (mark[i]) rows.push_back(i);
    }

    if ((!gChannelPressureOutletDirichletAll) && gChannelPressureOutletSinglePin && rows.size() > 1) {
        rows = { rows[rows.size()/2] };
    }

    return rows;
}



static void print_channel_pressure_pin_geometry_diagnostic(
    const Mesh& mesh,
    const std::vector<int>& pins)
{
    if (!gCaseIsChannel) return;

    if (pins.empty()) {
        std::printf("Channel pressure pin diag: count=0 -- no outlet pressure nodes found.\n");
        return;
    }

    double xmin= std::numeric_limits<double>::infinity();
    double ymin= std::numeric_limits<double>::infinity();
    double zmin= std::numeric_limits<double>::infinity();
    double xmax=-std::numeric_limits<double>::infinity();
    double ymax=-std::numeric_limits<double>::infinity();
    double zmax=-std::numeric_limits<double>::infinity();
    double xs=0.0, ys=0.0, zs=0.0;
    int good=0;

    for (int id : pins) {
        if (id < 0 || id >= (int)mesh.P.size()) continue;
        const auto& x = mesh.P[id];
        xmin = std::min(xmin, x[0]); xmax = std::max(xmax, x[0]);
        ymin = std::min(ymin, x[1]); ymax = std::max(ymax, x[1]);
        zmin = std::min(zmin, x[2]); zmax = std::max(zmax, x[2]);
        xs += x[0]; ys += x[1]; zs += x[2];
        ++good;
    }

    if (good <= 0) {
        int rmin = pins[0], rmax = pins[0];
        for (int id : pins) { rmin = std::min(rmin, id); rmax = std::max(rmax, id); }
        std::printf("Channel pressure pin diag: count=%zu DG1-local rows rowRange=(%d,%d); geometry print skipped because rows are not CG mesh vertices.\n",
                    pins.size(), rmin, rmax);
        return;
    }

    const double inv = 1.0 / (double)good;
    std::printf(
        "Channel pressure pin diag: count=%zu valid=%d centroid=(%.17e,%.17e,%.17e) "
        "bbox x=(%.17e,%.17e) y=(%.17e,%.17e) z=(%.17e,%.17e)\n",
        pins.size(), good, xs*inv, ys*inv, zs*inv,
        xmin, xmax, ymin, ymax, zmin, zmax);
}



static int add_channel_outlet_pressure_penalty_to_csr(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    double beta,
    CSRPattern& pat,
    std::vector<double>& vals)
{
    if (!gCaseIsChannel || beta <= 0.0) return 0;

    const int nP = pat.nRows;
    std::vector<std::map<int,double>> rows(nP);

    for (int r=0; r<pat.nRows; ++r) {
        for (int k=pat.rowOffsets[r]; k<pat.rowOffsets[r+1]; ++k) {
            rows[r][(int)pat.cols[k]] += vals[k];
        }
    }

    int outletFaces = 0;
    int mapFail = 0;
    double areaSum = 0.0;
    double penaltyTrace = 0.0;
    double penaltyAbs = 0.0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        if (channel_boundary_kind(mesh, f) != 2) continue;

        const int c = mesh.owner[f];
        if (c < 0 || c >= (int)tets.size()) continue;

        const TetP2Geom& K = tets[c];
        const double area = mesh.Af[f];
        const double hF = std::max(std::sqrt(std::max(area, 0.0)), 1e-14);

        ++outletFaces;
        areaSum += area;

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            const double w = beta * area * q.w / hF;

            for (int a=0; a<4; ++a) {
                const int ia = 4*c + a;
                if (ia < 0 || ia >= nP) continue;
                for (int b=0; b<4; ++b) {
                    const int ib = 4*c + b;
                    if (ib < 0 || ib >= nP) continue;

                    const double v = w * lam[a] * lam[b];
                    if (std::abs(v) <= 1e-300) continue;

                    rows[ia][ib] += v;
                    penaltyAbs += std::abs(v);
                    if (ia == ib) penaltyTrace += v;
                }
            }
        }
    }

    pat = rows_to_csrpattern(nP, rows, vals);

    std::printf(
        "Channel outlet pressure penalty bridge: beta=%.6e faces=%d area=%.17e traceAdd=%.6e absAdd=%.6e mapFail=%d; outlet pressure rows kept alive unless separately pinned\n",
        beta, outletFaces, areaSum, penaltyTrace, penaltyAbs, mapFail);

    return outletFaces;
}

static void print_channel_pressure_patch_node_stats(
    int it,
    const char* stage,
    const Mesh& mesh,
    const std::vector<double>& p,
    const std::vector<double>& pcorr)
{
    if (!gCaseIsChannel) return;

    const int nNode = (int)mesh.P.size();
    if ((int)p.size() != nNode) {
        std::printf("    channelPressurePatchStats: it=%d stage=%s DG1-local pressure rows=%zu; CG1 node patch stats skipped.\n",
            it, stage, p.size());
        return;
    }

    std::vector<unsigned char> markIn(nNode,0), markOut(nNode,0), markWall(nNode,0);

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;
        std::vector<unsigned char>* mark = nullptr;
        if (bkind == 1) mark = &markIn;
        else if (bkind == 2) mark = &markOut;
        else mark = &markWall;

        for (int v : mesh.faces[f]) {
            if (v >= 0 && v < nNode) (*mark)[v] = 1;
        }
    }

    auto print_one = [&](const char* name, const std::vector<unsigned char>& mark) {
        int cnt = 0;
        double pmin= std::numeric_limits<double>::infinity();
        double pmax=-std::numeric_limits<double>::infinity();
        double psum=0.0;
        double pcmin= std::numeric_limits<double>::infinity();
        double pcmax=-std::numeric_limits<double>::infinity();
        double pcsum=0.0;
        double pcabsmax=0.0;

        for (int i=0; i<nNode; ++i) {
            if (!mark[i]) continue;
            ++cnt;
            const double pv = (i < (int)p.size()) ? p[i] : 0.0;
            const double cv = (i < (int)pcorr.size()) ? pcorr[i] : 0.0;
            pmin = std::min(pmin, pv); pmax = std::max(pmax, pv); psum += pv;
            pcmin = std::min(pcmin, cv); pcmax = std::max(pcmax, cv); pcsum += cv;
            pcabsmax = std::max(pcabsmax, std::abs(cv));
        }

        if (cnt == 0) {
            std::printf("    channelPressurePatchStats: it=%d stage=%s patch=%s nodes=0\n",
                it, stage, name);
            return;
        }

        const double inv = 1.0 / (double)cnt;
        std::printf(
            "    channelPressurePatchStats: it=%d stage=%s patch=%s nodes=%d "
            "p(avg,min,max)=(% .6e,% .6e,% .6e) "
            "pcorr(avg,min,max,maxAbs)=(% .6e,% .6e,% .6e,%.6e)\n",
            it, stage, name, cnt,
            psum*inv, pmin, pmax,
            pcsum*inv, pcmin, pcmax, pcabsmax);
    };

    print_one("inlet", markIn);
    print_one("outlet", markOut);
    print_one("wall", markWall);
}

static void print_channel_stage_flux_diagnostic(
    const char* stage,
    int it,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz)
{
    if (!gCaseIsChannel) return;

    double areaIn = 0.0, areaOut = 0.0, areaWall = 0.0;
    double fluxIn = 0.0, fluxOut = 0.0, fluxWall = 0.0;
    double uzIn = 0.0, uzOut = 0.0;
    double unIn = 0.0, unOut = 0.0;
    double magIn = 0.0, magOut = 0.0;
    int facesIn = 0, facesOut = 0, facesWall = 0, mapFail = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;

        const int M = mesh.owner[f];
        const auto& K = tets[M];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        if (bkind == 1) { areaIn += area; ++facesIn; }
        else if (bkind == 2) { areaOut += area; ++facesOut; }
        else { areaWall += area; ++facesWall; }

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            double u0 = 0.0, u1 = 0.0, u2 = 0.0;
            for (int j=0; j<10; ++j) {
                u0 += ux[10*M+j] * N[j];
                u1 += uy[10*M+j] * N[j];
                u2 += uz[10*M+j] * N[j];
            }

            const double ww = area * q.w;
            const double un = u0*n[0] + u1*n[1] + u2*n[2];
            const double umag = std::sqrt(u0*u0 + u1*u1 + u2*u2);

            if (bkind == 1) {
                fluxIn += ww*un;
                uzIn += ww*u2;
                unIn += ww*un;
                magIn += ww*umag;
            } else if (bkind == 2) {
                fluxOut += ww*un;
                uzOut += ww*u2;
                unOut += ww*un;
                magOut += ww*umag;
            } else {
                fluxWall += ww*un;
            }
        }
    }

    const double invAIn = areaIn > 1e-300 ? 1.0/areaIn : 0.0;
    const double invAOut = areaOut > 1e-300 ? 1.0/areaOut : 0.0;

    std::printf(
        "    channelStageFlux: it=%d stage=%s "
        "flux(in,out,wall,net)=(% .6e,% .6e,% .6e,% .6e) "
        "avgUz(in,out)=(% .6e,% .6e) avgUn(in,out)=(% .6e,% .6e) "
        "avgMag(in,out)=(%.6e,%.6e) faces(in,out,wall)=(%d,%d,%d) mapFail=%d\n",
        it, stage,
        fluxIn, fluxOut, fluxWall, fluxIn + fluxOut + fluxWall,
        uzIn*invAIn, uzOut*invAOut,
        unIn*invAIn, unOut*invAOut,
        magIn*invAIn, magOut*invAOut,
        facesIn, facesOut, facesWall, mapFail);
}

static void print_channel_wall_flux_split_diagnostic(
    const char* stage,
    int it,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz)
{
    if (!gCaseIsChannel) return;

    const int nU = 10 * mesh.nCells;
    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    std::vector<unsigned char> inletTrace(nU, 0);
    std::vector<unsigned char> outletTrace(nU, 0);

    auto mark_face_rows = [&](int f, std::vector<unsigned char>& mark) {
        const int c = mesh.owner[f];
        const auto& K = tets[c];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;
            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] &&
                         faceHasLocalVertex[edge[e][1]];
            }

            if (onFace) {
                const int row = 10*c + i;
                if (row >= 0 && row < nU) mark[row] = 1;
            }
        }
    };

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind == 1) mark_face_rows(f, inletTrace);
        if (bkind == 2) mark_face_rows(f, outletTrace);
    }

    // categories:
    // 0 pure wall
    // 1 wall touching inlet trace rows
    // 2 wall touching outlet trace rows
    // 3 wall touching both inlet and outlet traces
    double area[4] = {0,0,0,0};
    double flux[4] = {0,0,0,0};
    double absFlux[4] = {0,0,0,0};
    int faces[4] = {0,0,0,0};
    int mapFail = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != 0) continue;

        const int M = mesh.owner[f];
        const auto& K = tets[M];
        const auto n = mesh.nf[f];
        const double Af = mesh.Af[f];

        bool touchInlet = false;
        bool touchOutlet = false;

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;
            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] &&
                         faceHasLocalVertex[edge[e][1]];
            }

            if (!onFace) continue;

            const int row = 10*M + i;
            if (row < 0 || row >= nU) continue;

            if (inletTrace[row]) touchInlet = true;
            if (outletTrace[row]) touchOutlet = true;
        }

        const int cat = touchInlet && touchOutlet ? 3 :
                        touchOutlet ? 2 :
                        touchInlet ? 1 : 0;

        area[cat] += Af;
        faces[cat]++;

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            double u0 = 0.0, u1 = 0.0, u2 = 0.0;
            for (int j=0; j<10; ++j) {
                u0 += ux[10*M+j] * N[j];
                u1 += uy[10*M+j] * N[j];
                u2 += uz[10*M+j] * N[j];
            }

            const double ww = Af * q.w;
            const double un = u0*n[0] + u1*n[1] + u2*n[2];

            flux[cat] += ww * un;
            absFlux[cat] += ww * std::abs(un);
        }
    }

    auto avg_abs = [&](int c) {
        return area[c] > 1e-300 ? absFlux[c] / area[c] : 0.0;
    };

    std::printf(
        "    channelWallFluxSplit: it=%d stage=%s "
        "pure(flux,absAvg,A,F)=(% .6e,%.6e,%.6e,%d) "
        "inletRim=(% .6e,%.6e,%.6e,%d) "
        "outletRim=(% .6e,%.6e,%.6e,%d) "
        "mixedRim=(% .6e,%.6e,%.6e,%d) mapFail=%d\n",
        it, stage,
        flux[0], avg_abs(0), area[0], faces[0],
        flux[1], avg_abs(1), area[1], faces[1],
        flux[2], avg_abs(2), area[2], faces[2],
        flux[3], avg_abs(3), area[3], faces[3],
        mapFail);
}

static double exact_p(const std::array<double,3>& x)
{
    return std::sin(kPi*x[0]) * std::sin(kPi*x[1]) * std::sin(kPi*x[2]);
}

static std::array<double,3> grad_exact_p(const std::array<double,3>& x)
{
    return {
        kPi*std::cos(kPi*x[0])*std::sin(kPi*x[1])*std::sin(kPi*x[2]),
        kPi*std::sin(kPi*x[0])*std::cos(kPi*x[1])*std::sin(kPi*x[2]),
        kPi*std::sin(kPi*x[0])*std::sin(kPi*x[1])*std::cos(kPi*x[2])
    };
}

static double exact_u_comp(int d, const std::array<double,3>& x)
{
    const double sx = std::sin(kPi*x[0]);
    const double sy = std::sin(kPi*x[1]);
    const double sz = std::sin(kPi*x[2]);

    if (d == 0) return gMmsAmp * sx*sx * std::sin(2.0*kPi*x[1]) * sz;
    if (d == 1) return -gMmsAmp * sy*sy * std::sin(2.0*kPi*x[0]) * sz;
    return 0.0;
}

static double lap_exact_u_comp(int d, const std::array<double,3>& x)
{
    const double sx = std::sin(kPi*x[0]);
    const double sy = std::sin(kPi*x[1]);
    const double sz = std::sin(kPi*x[2]);

    if (d == 0) {
        const double A = sx*sx;
        const double B = std::sin(2.0*kPi*x[1]);
        const double C = sz;
        return gMmsAmp * (2.0*kPi*kPi*std::cos(2.0*kPi*x[0])*B*C - 5.0*kPi*kPi*A*B*C);
    }

    if (d == 1) {
        const double D = sy*sy;
        const double E = std::sin(2.0*kPi*x[0]);
        const double C = sz;
        return gMmsAmp * (-2.0*kPi*kPi*std::cos(2.0*kPi*x[1])*E*C + 5.0*kPi*kPi*D*E*C);
    }

    return 0.0;
}

static std::array<double,3> exact_convective_term(const std::array<double,3>& x)
{
    const double A = gMmsAmp;

    const double sx = std::sin(kPi*x[0]);
    const double sy = std::sin(kPi*x[1]);
    const double sz = std::sin(kPi*x[2]);

    const double s2x = std::sin(2.0*kPi*x[0]);
    const double s2y = std::sin(2.0*kPi*x[1]);
    const double c2x = std::cos(2.0*kPi*x[0]);
    const double c2y = std::cos(2.0*kPi*x[1]);

    const double ux = A * sx*sx * s2y * sz;
    const double uy = -A * sy*sy * s2x * sz;

    const double dux_dx = A * kPi * s2x * s2y * sz;
    const double dux_dy = A * 2.0*kPi * sx*sx * c2y * sz;

    const double duy_dx = -A * 2.0*kPi * sy*sy * c2x * sz;
    const double duy_dy = -A * kPi * s2y * s2x * sz;

    return {
        ux*dux_dx + uy*dux_dy,
        ux*duy_dx + uy*duy_dy,
        0.0
    };
}

static double stokes_force_comp(int d, const std::array<double,3>& x, double nu)
{
    if (gCaseIsChannel) {
        (void)d;
        (void)x;
        (void)nu;
        return 0.0;
    }

    const auto gp = grad_exact_p(x);

    double f = -nu * lap_exact_u_comp(d, x) + gp[d];

    if (gFlowIsNse) {
        const auto c = exact_convective_term(x);
        f += c[d];
    }

    return f;
}

static void add_scaled_values(std::vector<double>& y, const std::vector<double>& x, double a)
{
    if (y.size() != x.size()) throw std::runtime_error("add_scaled_values: size mismatch");
    for (std::size_t i=0; i<y.size(); ++i) y[i] += a*x[i];
}

static void axpy_vec(std::vector<double>& y, double a, const std::vector<double>& x)
{
    if (y.size() != x.size()) throw std::runtime_error("axpy_vec: size mismatch");
    for (std::size_t i=0; i<y.size(); ++i) y[i] += a*x[i];
}

struct VelocityFreeRowsTolInfo {
    double rhsAll = 0.0;
    double rhsFree = 0.0;
    double axFree = 0.0;
    double scaleFree = 0.0;
    double requestedRelTol = 0.0;
    double requestedAbsTol = 0.0;
    double effectiveRelTol = 0.0;
};

static VelocityFreeRowsTolInfo velocity_make_free_rows_tol_info(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::vector<double>& xForScale,
    const std::vector<unsigned char>* lockedRows,
    double requestedRelTol,
    double requestedAbsTol,
    double scaleFloor)
{
    VelocityFreeRowsTolInfo info;
    info.requestedRelTol = requestedRelTol;
    info.requestedAbsTol = requestedAbsTol;
    info.effectiveRelTol = requestedRelTol;
    info.rhsAll = norm_vec(rhs);
    info.rhsFree = norm_vec_free_rows(rhs, lockedRows);

    if (!lockedRows || lockedRows->empty() || requestedRelTol <= 0.0 || info.rhsAll <= 1e-300) {
        info.scaleFree = std::max(info.rhsFree, scaleFloor);
        return info;
    }

    std::vector<double> Ax;
    apply_csr(pat, A, xForScale, Ax);
    info.axFree = norm_vec_free_rows(Ax, lockedRows);

    // Robust free-row scale: useful for high-viscosity, low-RHS, moving-wall,
    // inlet, and arbitrary nonzero Dirichlet cases.
    info.scaleFree = std::max(std::max(info.rhsFree, info.axFree), scaleFloor);

    // Existing solvers internally stop on ||r_all|| / ||rhs_all||.
    // Locked rows are identity rows and should have zero residual after enforcement,
    // so ||r_all|| ~= ||r_free||. Ask the old solver for an equivalent all-row tolerance.
    const double targetAbs = std::max(0.0, requestedAbsTol) + requestedRelTol * info.scaleFree;
    double eff = targetAbs / std::max(info.rhsAll, 1e-300);

    // Never loosen the user's requested tolerance.
    eff = std::min(eff, requestedRelTol);

    // Avoid absurd underflow-style requests unless the user explicitly changes the solver.
    if (requestedRelTol > 0.0) {
        eff = std::max(eff, 1e-14);
    }

    info.effectiveRelTol = eff;
    return info;
}

static double velocity_free_rows_relative_residual(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::vector<double>& x,
    const std::vector<unsigned char>* lockedRows,
    double scaleFloor,
    double* absFreeOut = nullptr,
    double* rhsFreeOut = nullptr,
    double* axFreeOut = nullptr,
    double* scaleFreeOut = nullptr)
{
    std::vector<double> Ax;
    apply_csr(pat, A, x, Ax);

    double r2Free = 0.0;
    const int n = (int)rhs.size();

    for (int i=0; i<n; ++i) {
        if (lockedRows && i < (int)lockedRows->size() && (*lockedRows)[i]) {
            continue;
        }
        const double ri = rhs[i] - Ax[i];
        r2Free += ri * ri;
    }

    const double absFree = std::sqrt(std::max(0.0, r2Free));
    const double rhsFree = norm_vec_free_rows(rhs, lockedRows);
    const double axFree = norm_vec_free_rows(Ax, lockedRows);
    const double scaleFree = std::max(std::max(rhsFree, axFree), scaleFloor);

    if (absFreeOut) *absFreeOut = absFree;
    if (rhsFreeOut) *rhsFreeOut = rhsFree;
    if (axFreeOut) *axFreeOut = axFree;
    if (scaleFreeOut) *scaleFreeOut = scaleFree;

    return absFree / std::max(scaleFree, 1e-300);
}



static void subtract_weighted_mean(std::vector<double>& x, const std::vector<double>& w)
{
    const double m = compute_weighted_mean(x, w);
    for (double& v : x) v -= m;
}

static void assemble_dg2_sipg_stokes_matrix_rhs(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const CSRPattern& pat,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    double nu,
    double sigma,
    std::vector<double>& Aphys,
    std::vector<double>& M,
    std::array<std::vector<double>,3>& F)
{
    Aphys.assign(pat.nnz, 0.0);
    M.assign(pat.nnz, 0.0);
    for (int d=0; d<3; ++d) F[d].assign(pat.nRows, 0.0);

    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];

        for (const auto& q : tq) {
            double N[10];
            std::array<double,3> G[10];

            p2_tet_basis(q.lam, N);
            p2_tet_grad(K, q.lam, G);

            const double w = K.vol * q.w;
            const auto xq = lincomb4(K.x, q.lam);

            double f[3] = {
                stokes_force_comp(0,xq,nu),
                stokes_force_comp(1,xq,nu),
                stokes_force_comp(2,xq,nu)
            };

            for (int i=0; i<10; ++i) {
                const int row = 10*c + i;

                for (int d=0; d<3; ++d) {
                    F[d][row] += w * f[d] * N[i];
                }

                for (int j=0; j<10; ++j) {
                    add_A(pat, Aphys, row, 10*c + j, w * nu * dot3(G[j], G[i]));
                    add_A(pat, M,     row, 10*c + j, w * N[j] * N[i]);
                }
            }
        }
    }

    int faceMapFailures = 0;

    for (int f=0; f<mesh.nFaces; ++f) {
        const int P = mesh.owner[f];
        const bool interior = (f < mesh.nInternalFaces);
        const int Ncell = interior ? mesh.neigh[f] : -1;

        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        const auto& KP = tets[P];
        const TetP2Geom* KN = interior ? &tets[Ncell] : nullptr;

        const double hP = std::max(3.0*KP.vol/std::max(area,1e-300), 1e-300);
        double invh = 1.0/hP;

        if (interior) {
            const double hN = std::max(3.0*KN->vol/std::max(area,1e-300), 1e-300);
            invh = std::max(invh, 1.0/hN);
        }

        const double penalty = sigma * 9.0 * nu * invh;

        for (const auto& qf : fq) {
            std::array<double,4> lamP{}, lamN{};

            if (!face_lam_on_tet(mesh.faces[f], KP, qf.mu, lamP)) {
                faceMapFailures++;
                continue;
            }

            double phiP[10];
            std::array<double,3> gradP[10];

            p2_tet_basis(lamP, phiP);
            p2_tet_grad(KP, lamP, gradP);

            const double w = area * qf.w;

            if (interior) {
                if (!face_lam_on_tet(mesh.faces[f], *KN, qf.mu, lamN)) {
                    faceMapFailures++;
                    continue;
                }

                double phiN[10];
                std::array<double,3> gradN[10];

                p2_tet_basis(lamN, phiN);
                p2_tet_grad(*KN, lamN, gradN);

                for (int ti=0; ti<20; ++ti) {
                    const bool testOwner = (ti < 10);
                    const int i = testOwner ? ti : ti-10;
                    const int rowCell = testOwner ? P : Ncell;
                    const int row = 10*rowCell + i;

                    const double phiTest = testOwner ? phiP[i] : phiN[i];
                    const auto& gradTest = testOwner ? gradP[i] : gradN[i];
                    const double sTest = testOwner ? 1.0 : -1.0;
                    const double jumpTest = sTest * phiTest;
                    const double avgFluxTest = 0.5 * nu * dot3(gradTest, n);

                    for (int tj=0; tj<20; ++tj) {
                        const bool trialOwner = (tj < 10);
                        const int j = trialOwner ? tj : tj-10;
                        const int colCell = trialOwner ? P : Ncell;
                        const int col = 10*colCell + j;

                        const double phiTrial = trialOwner ? phiP[j] : phiN[j];
                        const auto& gradTrial = trialOwner ? gradP[j] : gradN[j];
                        const double sTrial = trialOwner ? 1.0 : -1.0;
                        const double jumpTrial = sTrial * phiTrial;
                        const double avgFluxTrial = 0.5 * nu * dot3(gradTrial, n);

                        const double aij = w * (
                            -avgFluxTrial*jumpTest
                            -avgFluxTest*jumpTrial
                            +penalty*jumpTrial*jumpTest
                        );

                        add_A(pat, Aphys, row, col, aij);
                    }
                }
            } else {
                if (channel_is_outlet_face(mesh, f)) {
                    continue;
                }

                const auto xq = lincomb4(KP.x, lamP);

                double g[3];

                if (gCaseIsChannel) {
                    const int bkind = channel_boundary_kind(mesh, f);
                    const auto gbc = (bkind == 1)
                        ? channel_velocity_bc(xq, n)
                        : std::array<double,3>{0.0, 0.0, 0.0};
                    g[0] = gbc[0];
                    g[1] = gbc[1];
                    g[2] = gbc[2];
                } else {
                    g[0] = exact_u_comp(0,xq);
                    g[1] = exact_u_comp(1,xq);
                    g[2] = exact_u_comp(2,xq);
                }

                for (int i=0; i<10; ++i) {
                    const int row = 10*P + i;
                    const double gradTestN = nu * dot3(gradP[i], n);

                    for (int d=0; d<3; ++d) {
                        F[d][row] += w * (-gradTestN*g[d] + penalty*g[d]*phiP[i]);
                    }

                    for (int j=0; j<10; ++j) {
                        const double gradTrialN = nu * dot3(gradP[j], n);

                        const double aij = w * (
                            -gradTrialN*phiP[i]
                            -gradTestN*phiP[j]
                            +penalty*phiP[j]*phiP[i]
                        );

                        add_A(pat, Aphys, row, 10*P + j, aij);
                    }
                }
            }
        }
    }

    std::printf("face mapping failures          = %d\n", faceMapFailures);
    if (faceMapFailures) throw std::runtime_error("face mapping failed in SIPG Stokes assembly");
}

static void apply_divergence_from_Ap(
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    std::vector<double>& r)
{
    std::vector<double> tmp;

    apply_neg_transpose_rect(Apx, ux, r);

    apply_neg_transpose_rect(Apy, uy, tmp);
    axpy_vec(r, 1.0, tmp);

    apply_neg_transpose_rect(Apz, uz, tmp);
    axpy_vec(r, 1.0, tmp);
}


static void apply_divergence_from_Ap_selected_velocity_rows(
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    const std::vector<unsigned char>* rowMask,
    std::vector<double>& r)
{
    const int nP = Apx.nCols;
    r.assign(nP, 0.0);

    auto add_direction = [&](const RectCSR& Ap, const std::vector<double>& u) {
        const int nRows = Ap.nRows;
        for (int row=0; row<nRows; ++row) {
            if (rowMask) {
                if (row >= (int)rowMask->size() || !(*rowMask)[row]) continue;
            }

            const double ur = u[row];
            if (std::abs(ur) <= 1e-300) continue;

            for (int k=Ap.rowOffsets[row]; k<Ap.rowOffsets[row+1]; ++k) {
                const int col = Ap.cols[k];
                r[col] -= Ap.values[k] * ur;
            }
        }
    };

    add_direction(Apx, ux);
    add_direction(Apy, uy);
    add_direction(Apz, uz);
}

static void apply_gradient_rhs(const RectCSR& Ap, const std::vector<double>& p, std::vector<double>& g)
{
    apply_rect(Ap, p, g);
}

static HypreSolveInfo solve_hypre_csr_vec(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt)
{
    std::vector<HYPRE_Complex> hA = to_hypre_complex_vec(A);
    std::vector<HYPRE_Complex> hb = to_hypre_complex_vec(rhs);
    std::vector<HYPRE_Complex> hx = to_hypre_complex_vec(x);

    HypreSolveInfo info = solve_system_hypre_gpu(pat, hA, hb, hx, opt);

    x = from_hypre_complex_vec(hx);
    return info;
}

static void init_reusable_hypre_csr_vec(
    HypreReusableSystem& sys,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const HypreOptions& opt)
{
    std::vector<HYPRE_Complex> hA = to_hypre_complex_vec(A);
    std::vector<HYPRE_Complex> hzero(pat.nRows, static_cast<HYPRE_Complex>(0));

    init_reusable_hypre_system_gpu(sys, pat.rows, pat.ncols, pat.cols, opt);
    update_reusable_hypre_system_gpu(sys, hA, hzero, opt, true);
}

static HypreSolveInfo solve_reusable_hypre_rhs_vec(
    HypreReusableSystem& sys,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt)
{
    const double t0 = wall_seconds();
    std::vector<HYPRE_Complex> hb = to_hypre_complex_vec(rhs);
    std::vector<HYPRE_Complex> hx = to_hypre_complex_vec(x);

    const double tUpdate0 = wall_seconds();
    update_reusable_hypre_rhs_gpu(sys, hb, opt);
    const double tUpdate = wall_seconds() - tUpdate0;

    const double tSolve0 = wall_seconds();
    HypreSolveInfo info = solve_reusable_hypre_system_gpu(sys, hx);
    const double tSolve = wall_seconds() - tSolve0;

    x = from_hypre_complex_vec(hx);

    if (opt.profile) {
        const char* label = opt.profileLabel.empty() ? "p-reuse" : opt.profileLabel.c_str();
        std::printf("[HYPRE_PROFILE] kind=reusable_rhs label=%s rows=%d rhsUpdate=%.6e solveAndGet=%.6e total=%.6e\n",
            label, sys.nRows, tUpdate, tSolve, wall_seconds() - t0);
    }

    return info;
}

static HypreSolveInfo solve_reusable_hypre_matrix_rhs_vec(
    HypreReusableSystem& sys,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt,
    bool recomputeFinalResidual = true)
{
    const double t0 = wall_seconds();
    std::vector<HYPRE_Complex> hA = to_hypre_complex_vec(A);
    std::vector<HYPRE_Complex> hb = to_hypre_complex_vec(rhs);
    std::vector<HYPRE_Complex> hx = to_hypre_complex_vec(x);

    const double tUpdate0 = wall_seconds();
    update_reusable_hypre_system_gpu(sys, hA, hb, opt, true);
    const double tUpdate = wall_seconds() - tUpdate0;

    const double tSolve0 = wall_seconds();
    HypreSolveInfo info = solve_reusable_hypre_system_gpu(sys, hx);
    const double tSolve = wall_seconds() - tSolve0;

    x = from_hypre_complex_vec(hx);

    // Preserve the historical velocity reporting convention only when requested.
    // This host-side SpMV is diagnostic only; it does not affect the algorithm.
    if (recomputeFinalResidual) {
        std::vector<double> Ax;
        apply_csr(pat, A, x, Ax);
        axpy_vec(Ax, -1.0, rhs);
        const double r = norm_vec(Ax);
        const double b = std::max(norm_vec(rhs), 1e-300);
        info.finalRelResNorm = r / b;
    }

    if (opt.profile) {
        const char* label = opt.profileLabel.empty() ? "u-reuse" : opt.profileLabel.c_str();
        std::printf("[HYPRE_PROFILE] kind=reusable_matrix_rhs label=%s rows=%d nnz=%d matrixRhsUpdateSetup=%.6e solveAndGet=%.6e total=%.6e\n",
            label, sys.nRows, sys.nnz, tUpdate, tSolve, wall_seconds() - t0);
    }

    return info;
}

static double dg2_vector_l2_error(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    double* exactNormOut=nullptr)
{
    double e2=0.0;
    double n2=0.0;

    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];

        for (const auto& q : tq) {
            double N[10];
            p2_tet_basis(q.lam, N);

            double uh[3] = {0.0,0.0,0.0};

            for (int i=0; i<10; ++i) {
                uh[0] += ux[10*c+i]*N[i];
                uh[1] += uy[10*c+i]*N[i];
                uh[2] += uz[10*c+i]*N[i];
            }

            const auto xq = lincomb4(K.x, q.lam);

            const double ue[3] = {
                exact_u_comp(0,xq),
                exact_u_comp(1,xq),
                exact_u_comp(2,xq)
            };

            const double w = K.vol*q.w;

            for (int d=0; d<3; ++d) {
                const double de=uh[d]-ue[d];
                e2 += w*de*de;
                n2 += w*ue[d]*ue[d];
            }
        }
    }

    if (exactNormOut) *exactNormOut = std::sqrt(std::max(0.0,n2));
    return std::sqrt(std::max(0.0,e2));
}

static double cg1_pressure_l2_error(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<double>& p,
    const std::vector<double>& pWeights,
    double* exactNormOut=nullptr)
{
    std::vector<double> diffV(p.size(), 0.0);

    for (std::size_t i=0; i<p.size(); ++i) {
        diffV[i] = p[i] - exact_p(mesh.P[i]);
    }

    const double off = compute_weighted_mean(diffV, pWeights);

    double e2=0.0;
    double n2=0.0;

    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];

        for (const auto& q : tq) {
            double ph=0.0;

            for (int a=0; a<4; ++a) {
                ph += q.lam[a]*p[K.v[a]];
            }

            const auto xq = lincomb4(K.x, q.lam);
            const double pe = exact_p(xq);
            const double er = ph - pe - off;
            const double w = K.vol*q.w;

            e2 += w*er*er;
            n2 += w*pe*pe;
        }
    }

    if (exactNormOut) *exactNormOut = std::sqrt(std::max(0.0,n2));
    return std::sqrt(std::max(0.0,e2));
}




static double dg1_pressure_at_lam_cell(
    int cell,
    const std::array<double,4>& lam,
    const std::vector<double>& p)
{
    double ph = 0.0;
    const int base = 4*cell;
    for (int a=0; a<4; ++a) {
        const int row = base + a;
        if (row >= 0 && row < (int)p.size()) ph += lam[a] * p[row];
    }
    return ph;
}

static double dg1_pressure_at_p2_local_node(
    int cell,
    int localP2,
    const std::vector<double>& p)
{
    const int base = 4*cell;
    auto P = [&](int a) -> double {
        const int row = base + a;
        return (row >= 0 && row < (int)p.size()) ? p[row] : 0.0;
    };
    switch (localP2) {
        case 0: return P(0);
        case 1: return P(1);
        case 2: return P(2);
        case 3: return P(3);
        case 4: return 0.5*(P(0) + P(1));
        case 5: return 0.5*(P(0) + P(2));
        case 6: return 0.5*(P(0) + P(3));
        case 7: return 0.5*(P(1) + P(2));
        case 8: return 0.5*(P(1) + P(3));
        case 9: return 0.5*(P(2) + P(3));
        default: return 0.0;
    }
}

static double cg1_pressure_at_p2_local_node(
    const TetP2Geom& K,
    int localP2,
    const std::vector<double>& p)
{
    switch (localP2) {
        case 0: return p[K.v[0]];
        case 1: return p[K.v[1]];
        case 2: return p[K.v[2]];
        case 3: return p[K.v[3]];
        case 4: return 0.5*(p[K.v[0]] + p[K.v[1]]);
        case 5: return 0.5*(p[K.v[0]] + p[K.v[2]]);
        case 6: return 0.5*(p[K.v[0]] + p[K.v[3]]);
        case 7: return 0.5*(p[K.v[1]] + p[K.v[2]]);
        case 8: return 0.5*(p[K.v[1]] + p[K.v[3]]);
        case 9: return 0.5*(p[K.v[2]] + p[K.v[3]]);
        default: return 0.0;
    }
}

static void write_dg2_dg1_quadratic_tet_vtu(
    const std::string& path,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    const std::vector<double>& p,
    const std::vector<double>& pWeights)
{
    if ((int)ux.size() != 10*mesh.nCells ||
        (int)uy.size() != 10*mesh.nCells ||
        (int)uz.size() != 10*mesh.nCells ||
        (int)p.size()  != 4*mesh.nCells) {
        throw std::runtime_error("write_dg2_dg1_quadratic_tet_vtu: inconsistent field sizes");
    }

    std::ofstream os(path);
    if (!os) {
        throw std::runtime_error("write_dg2_dg1_quadratic_tet_vtu: could not open output file: " + path);
    }

    std::vector<double> pDiffDg1(p.size(), 0.0);
    for (int c=0; c<mesh.nCells; ++c) {
        for (int a=0; a<4; ++a) {
            pDiffDg1[4*c+a] = p[4*c+a] - exact_p(tets[c].x[a]);
        }
    }
    const double pMeanOffset = compute_weighted_mean(pDiffDg1, pWeights);

    const int nCells = mesh.nCells;
    const int nPoints = 10*nCells;

    // Average duplicate DG2 values onto shared geometric P2 locations
    // for visualization. This is not used by the solver.
    // Key (v,v) = vertex node; key (min(v0,v1),max(v0,v1)) = edge midpoint.
    std::map<std::pair<int,int>, std::array<double,4>> uCgAverageAcc;

    const int localP2EdgeVerts[10][2] = {
        {0,0}, {1,1}, {2,2}, {3,3},
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    };

    auto p2_key = [&](const TetP2Geom& K, int localP2) -> std::pair<int,int> {
        const int a = localP2EdgeVerts[localP2][0];
        const int b = localP2EdgeVerts[localP2][1];
        int va = K.v[a];
        int vb = K.v[b];
        if (va > vb) std::swap(va, vb);
        return {va, vb};
    };

    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];

        for (int i=0; i<10; ++i) {
            auto key = p2_key(K, i);
            auto& acc = uCgAverageAcc[key];

            acc[0] += ux[10*c+i];
            acc[1] += uy[10*c+i];
            acc[2] += uz[10*c+i];
            acc[3] += 1.0;
        }
    }

    auto p2_cg_average_value = [&](const TetP2Geom& K, int localP2) -> std::array<double,3> {
        auto key = p2_key(K, localP2);
        const auto it = uCgAverageAcc.find(key);

        if (it == uCgAverageAcc.end() || it->second[3] <= 0.0) {
            return {0.0, 0.0, 0.0};
        }

        const auto& acc = it->second;
        return {acc[0]/acc[3], acc[1]/acc[3], acc[2]/acc[3]};
    };

    auto cell_average_velocity = [&](int c) -> std::array<double,3> {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;

        for (int i=0; i<10; ++i) {
            sx += ux[10*c+i];
            sy += uy[10*c+i];
            sz += uz[10*c+i];
        }

        return {sx/10.0, sy/10.0, sz/10.0};
    };

    // VTK_QUADRATIC_TETRA ordering:
    // vertices 0,1,2,3 then edges (0-1),(1-2),(2-0),(0-3),(1-3),(2-3).
    // Local code P2 order:
    // vertices 0,1,2,3 then edges (0-1),(0-2),(0-3),(1-2),(1-3),(2-3).
    const int vtkToLocalP2[10] = {0,1,2,3,4,7,5,6,8,9};


    os << std::setprecision(17);
    os << "<?xml version=\"1.0\"?>\n";
    os << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    os << "  <UnstructuredGrid>\n";
    os << "    <Piece NumberOfPoints=\"" << nPoints << "\" NumberOfCells=\"" << nCells << "\">\n";

    os << "      <PointData Vectors=\"U\" Scalars=\"p\">\n";

    os << "        <DataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          " << ux[10*c+i] << " " << uy[10*c+i] << " " << uz[10*c+i] << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"U_cg_average\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            const auto ua = p2_cg_average_value(K, i);
            os << "          " << ua[0] << " " << ua[1] << " " << ua[2] << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"U_cell_average_pointrep\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto uc = cell_average_velocity(c);
        for (int kk=0; kk<10; ++kk) {
            os << "          " << uc[0] << " " << uc[1] << " " << uc[2] << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"U_exact\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          "
               << exact_u_comp(0,K.xP2[i]) << " "
               << exact_u_comp(1,K.xP2[i]) << " "
               << exact_u_comp(2,K.xP2[i]) << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"U_error\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          "
               << ux[10*c+i] - exact_u_comp(0,K.xP2[i]) << " "
               << uy[10*c+i] - exact_u_comp(1,K.xP2[i]) << " "
               << uz[10*c+i] - exact_u_comp(2,K.xP2[i]) << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"p\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          " << dg1_pressure_at_p2_local_node(c, i, p) << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"p_exact_shifted\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          " << exact_p(K.xP2[i]) + pMeanOffset << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Float64\" Name=\"p_error_mean_free\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            const double ph = dg1_pressure_at_p2_local_node(c, i, p);
            os << "          " << ph - exact_p(K.xP2[i]) - pMeanOffset << "\n";
        }
    }
    os << "        </DataArray>\n";

    os << "      </PointData>\n";

    os << "      <CellData Scalars=\"cell_id\">\n";
    os << "        <DataArray type=\"Int32\" Name=\"cell_id\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) os << "          " << c << "\n";
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"Float64\" Name=\"cell_volume\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) os << "          " << tets[c].vol << "\n";
    os << "        </DataArray>\n";
    os << "        <DataArray type=\"Float64\" Name=\"U_cell_average\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto uc = cell_average_velocity(c);
        os << "          " << uc[0] << " " << uc[1] << " " << uc[2] << "\n";
    }
    os << "        </DataArray>\n";
    os << "      </CellData>\n";

    os << "      <Points>\n";
    os << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        const auto& K = tets[c];
        for (int kk=0; kk<10; ++kk) {
            const int i = vtkToLocalP2[kk];
            os << "          " << K.xP2[i][0] << " " << K.xP2[i][1] << " " << K.xP2[i][2] << "\n";
        }
    }
    os << "        </DataArray>\n";
    os << "      </Points>\n";

    os << "      <Cells>\n";
    os << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) {
        os << "          ";
        for (int kk=0; kk<10; ++kk) os << 10*c + kk << (kk == 9 ? "" : " ");
        os << "\n";
    }
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) os << "          " << 10*(c+1) << "\n";
    os << "        </DataArray>\n";

    os << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (int c=0; c<nCells; ++c) os << "          24\n";
    os << "        </DataArray>\n";
    os << "      </Cells>\n";

    os << "    </Piece>\n";
    os << "  </UnstructuredGrid>\n";
    os << "</VTKFile>\n";
}

static void apply_local_invM_cellwise(
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    const std::vector<double>& g,
    std::vector<double>& z)
{
    z.assign(g.size(), 0.0);

    const int nCells = (int)invM.size();

    if ((int)g.size() != 10*nCells) {
        throw std::runtime_error("apply_local_invM_cellwise: size mismatch");
    }

    for (int c=0; c<nCells; ++c) {
        for (int i=0; i<10; ++i) {
            double acc = 0.0;

            for (int j=0; j<10; ++j) {
                acc += invM[c][i][j] * g[10*c + j];
            }

            z[10*c + i] = acc;
        }
    }
}

static double correct_velocity_mass_schur_direction(
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    const RectCSR& Ap,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale)
{
    std::vector<double> g;
    std::vector<double> z;

    apply_rect(Ap, pcorr, g);
    apply_local_invM_cellwise(invM, g, z);

    double z2 = 0.0;

    for (std::size_t i=0; i<u.size(); ++i) {
        const double du = -scale * z[i];
        u[i] += du;
        z2 += du*du;
    }

    return std::sqrt(std::max(0.0, z2));
}



static double correct_velocity_mass_schur_direction_masked(
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    const RectCSR& Ap,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale,
    const std::vector<unsigned char>* lockedRows)
{
    std::vector<double> g;
    std::vector<double> z;

    apply_rect(Ap, pcorr, g);
    apply_local_invM_cellwise(invM, g, z);

    double z2 = 0.0;

    for (std::size_t i=0; i<u.size(); ++i) {
        if (lockedRows && i < lockedRows->size() && (*lockedRows)[i]) {
            continue;
        }

        const double du = -scale * z[i];
        u[i] += du;
        z2 += du*du;
    }

    return std::sqrt(std::max(0.0, z2));
}

// MATLAB-open / constrained-subspace projection correction:
//   r = Bopen*u + bD,
//   Lp = Bopen * Pfree * H * Pfree * Bopen^T,
//   Lp*pcorr = -r,
//   u <- u + Pfree * H * Pfree * Bopen^T * pcorr.
// Here Bopen^T is stored as a velocity-row x pressure-column RectCSR.
static double correct_velocity_mass_schur_open_transpose_direction_masked(
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    const RectCSR& BopenT,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale,
    const std::vector<unsigned char>* lockedRows)
{
    std::vector<double> g;
    std::vector<double> z;

    apply_rect(BopenT, pcorr, g);

    // Symmetric constrained-subspace projection:
    //   du = Pfree * H * Pfree * Bopen^T pcorr.
    // The pre-H mask is essential when H is a full local block inverse.
    if (lockedRows) {
        const std::size_t n = std::min(g.size(), lockedRows->size());
        for (std::size_t i=0; i<n; ++i) {
            if ((*lockedRows)[i]) g[i] = 0.0;
        }
    }

    apply_local_invM_cellwise(invM, g, z);

    double z2 = 0.0;

    for (std::size_t i=0; i<u.size(); ++i) {
        if (lockedRows && i < lockedRows->size() && (*lockedRows)[i]) {
            continue;
        }

        const double du = scale * z[i];
        u[i] += du;
        z2 += du*du;
    }

    return std::sqrt(std::max(0.0, z2));
}

static double correct_velocity_diag_schur_direction(
    const std::vector<double>& hInv,
    const RectCSR& Ap,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale)
{
    std::vector<double> g;
    apply_rect(Ap, pcorr, g);

    double z2 = 0.0;
    for (std::size_t i=0; i<u.size(); ++i) {
        const double du = -scale * hInv[i] * g[i];
        u[i] += du;
        z2 += du*du;
    }

    return std::sqrt(std::max(0.0, z2));
}

static void make_pressure_rhs_compatible(std::vector<double>& rhs)
{
    double s = 0.0;

    for (double v : rhs) {
        s += v;
    }

    const double mean = s / std::max<std::size_t>(rhs.size(), 1);

    for (double& v : rhs) {
        v -= mean;
    }
}



static int count_effective_nnz(const std::vector<double>& a)
{
    int n = 0;

    for (double v : a) {
        if (std::abs(v) > 0.0) ++n;
    }

    return n;
}


struct ConvLFPatternCache {
    bool ready = false;
    std::vector<int> volPos;       // nCells * 10 * 10
    std::vector<int> faceOffsets;  // nFaces+1
    std::vector<int> facePos;      // interior faces: 4*10*10, boundary faces: 10*10
};

static ConvLFPatternCache build_conv_lf_pattern_cache(const Mesh& mesh, const CSRPattern& pat)
{
    ConvLFPatternCache cache;

    cache.volPos.assign((std::size_t)mesh.nCells * 100u, -1);
    for (int c=0; c<mesh.nCells; ++c) {
        for (int i=0; i<10; ++i) {
            const int row = 10*c + i;
            for (int j=0; j<10; ++j) {
                const int col = 10*c + j;
                cache.volPos[(std::size_t)c*100u + (std::size_t)i*10u + (std::size_t)j] =
                    find_col_pos(pat, row, col);
            }
        }
    }

    cache.faceOffsets.assign(mesh.nFaces + 1, 0);
    int totalFacePos = 0;
    for (int f=0; f<mesh.nFaces; ++f) {
        cache.faceOffsets[f] = totalFacePos;
        totalFacePos += (f < mesh.nInternalFaces) ? 400 : 100;
    }

    cache.faceOffsets[mesh.nFaces] = totalFacePos;
    cache.facePos.assign(totalFacePos, -1);

    for (int f=0; f<mesh.nFaces; ++f) {
        const int M = mesh.owner[f];
        const bool interior = (f < mesh.nInternalFaces);
        const int P = interior ? mesh.neigh[f] : -1;
        int out = cache.faceOffsets[f];

        if (interior) {
            for (int i=0; i<10; ++i) {
                const int rowM = 10*M + i;
                const int rowP = 10*P + i;
                for (int j=0; j<10; ++j) {
                    const int colM = 10*M + j;
                    const int colP = 10*P + j;
                    cache.facePos[out++] = find_col_pos(pat, rowM, colM);
                    cache.facePos[out++] = find_col_pos(pat, rowM, colP);
                    cache.facePos[out++] = find_col_pos(pat, rowP, colM);
                    cache.facePos[out++] = find_col_pos(pat, rowP, colP);
                }
            }
        } else {
            for (int i=0; i<10; ++i) {
                const int row = 10*M + i;
                for (int j=0; j<10; ++j) {
                    const int col = 10*M + j;
                    cache.facePos[out++] = find_col_pos(pat, row, col);
                }
            }
        }
    }

    cache.ready = true;
    return cache;
}



static void dg2_cuda_check(cudaError_t err, const char* expr, const char* file, int line)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            " for " + expr + " : " + cudaGetErrorString(err));
    }
}

#define DG2_CUDA_CHECK(expr) dg2_cuda_check((expr), #expr, __FILE__, __LINE__)

template <typename T>
static T* dg2_cuda_malloc_copy_vec(const std::vector<T>& h, const char* label)
{
    T* d = nullptr;
    if (h.empty()) return nullptr;
    DG2_CUDA_CHECK(cudaMalloc((void**)&d, h.size() * sizeof(T)));
    DG2_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    (void)label;
    return d;
}

template <typename T>
static T* dg2_cuda_malloc_count(std::size_t n, const char* label)
{
    T* d = nullptr;
    if (n == 0) return nullptr;
    DG2_CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(T)));
    (void)label;
    return d;
}

static void dg2_cuda_free_ptr(void*& p)
{
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

struct GpuConvLFState {
    bool ready = false;

    int nCells = 0;
    int nFaces = 0;
    int nInternalFaces = 0;
    int nTq = 0;
    int nFq = 0;
    int nU = 0;
    int nnz = 0;

    double* d_vol = nullptr;
    double* d_gradLam = nullptr;

    double* d_tqLam = nullptr;
    double* d_tqW = nullptr;
    double* d_fqW = nullptr;

    int* d_owner = nullptr;
    int* d_neigh = nullptr;
    double* d_nf = nullptr;
    double* d_Af = nullptr;

    double* d_faceLamM = nullptr;
    double* d_faceLamP = nullptr;

    int* d_volPos = nullptr;
    int* d_faceOffsets = nullptr;
    int* d_facePos = nullptr;

    double* d_Kdiff = nullptr;
    double* d_betaX = nullptr;
    double* d_betaY = nullptr;
    double* d_betaZ = nullptr;
    double* d_C = nullptr;
    double* d_Aphys = nullptr;
};

__device__ static void dg2_p2_basis_dev(const double lam[4], double N[10])
{
    N[0] = lam[0]*(2.0*lam[0]-1.0);
    N[1] = lam[1]*(2.0*lam[1]-1.0);
    N[2] = lam[2]*(2.0*lam[2]-1.0);
    N[3] = lam[3]*(2.0*lam[3]-1.0);
    N[4] = 4.0*lam[0]*lam[1];
    N[5] = 4.0*lam[0]*lam[2];
    N[6] = 4.0*lam[0]*lam[3];
    N[7] = 4.0*lam[1]*lam[2];
    N[8] = 4.0*lam[1]*lam[3];
    N[9] = 4.0*lam[2]*lam[3];
}

__device__ static void dg2_p2_grad_dev(const double* gl, const double lam[4], double G[30])
{
    for (int a=0; a<4; ++a) {
        const double s = 4.0*lam[a] - 1.0;
        G[3*a+0] = s * gl[3*a+0];
        G[3*a+1] = s * gl[3*a+1];
        G[3*a+2] = s * gl[3*a+2];
    }

    for (int k=0; k<3; ++k) {
        G[3*4+k] = 4.0*(lam[0]*gl[3*1+k] + lam[1]*gl[3*0+k]);
        G[3*5+k] = 4.0*(lam[0]*gl[3*2+k] + lam[2]*gl[3*0+k]);
        G[3*6+k] = 4.0*(lam[0]*gl[3*3+k] + lam[3]*gl[3*0+k]);
        G[3*7+k] = 4.0*(lam[1]*gl[3*2+k] + lam[2]*gl[3*1+k]);
        G[3*8+k] = 4.0*(lam[1]*gl[3*3+k] + lam[3]*gl[3*1+k]);
        G[3*9+k] = 4.0*(lam[2]*gl[3*3+k] + lam[3]*gl[3*2+k]);
    }
}

__global__ static void dg2_lf_conv_volume_kernel(
    int nCells,
    int nTq,
    const double* __restrict__ vol,
    const double* __restrict__ gradLam,
    const double* __restrict__ tqLam,
    const double* __restrict__ tqW,
    const double* __restrict__ betaX,
    const double* __restrict__ betaY,
    const double* __restrict__ betaZ,
    const int* __restrict__ volPos,
    double* __restrict__ C)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    const int ubase = 10*c;
    const double* gl = gradLam + 12*c;

    for (int q=0; q<nTq; ++q) {
        const double* lam = tqLam + 4*q;

        double N[10];
        double G[30];

        dg2_p2_basis_dev(lam, N);
        dg2_p2_grad_dev(gl, lam, G);

        double bx = 0.0;
        double by = 0.0;
        double bz = 0.0;

        for (int j=0; j<10; ++j) {
            bx += betaX[ubase+j] * N[j];
            by += betaY[ubase+j] * N[j];
            bz += betaZ[ubase+j] * N[j];
        }

        const double w = vol[c] * tqW[q];

        for (int i=0; i<10; ++i) {
            const double bdotGradTest =
                bx*G[3*i+0] + by*G[3*i+1] + bz*G[3*i+2];

            for (int j=0; j<10; ++j) {
                const int pos = volPos[100*c + 10*i + j];
                if (pos >= 0) {
                    atomicAdd(C + pos, -w * bdotGradTest * N[j]);
                }
            }
        }
    }
}

__global__ static void dg2_lf_conv_face_kernel(
    int nFaces,
    int nInternalFaces,
    int nFq,
    const int* __restrict__ owner,
    const int* __restrict__ neigh,
    const double* __restrict__ nf,
    const double* __restrict__ Af,
    const double* __restrict__ fqW,
    const double* __restrict__ faceLamM,
    const double* __restrict__ faceLamP,
    const double* __restrict__ betaX,
    const double* __restrict__ betaY,
    const double* __restrict__ betaZ,
    const int* __restrict__ faceOffsets,
    const int* __restrict__ facePos,
    double* __restrict__ C)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFaces) return;

    const int M = owner[f];
    const int P = (f < nInternalFaces) ? neigh[f] : -1;
    const bool interior = (f < nInternalFaces);

    const double nx = nf[3*f+0];
    const double ny = nf[3*f+1];
    const double nz = nf[3*f+2];
    const double area = Af[f];

    for (int q=0; q<nFq; ++q) {
        const double* lamM = faceLamM + 4*((size_t)f*nFq + q);

        double NM[10];
        dg2_p2_basis_dev(lamM, NM);

        double bxM = 0.0;
        double byM = 0.0;
        double bzM = 0.0;

        const int baseM = 10*M;
        for (int j=0; j<10; ++j) {
            bxM += betaX[baseM+j] * NM[j];
            byM += betaY[baseM+j] * NM[j];
            bzM += betaZ[baseM+j] * NM[j];
        }

        const double bnM = bxM*nx + byM*ny + bzM*nz;
        const double w = area * fqW[q];

        if (interior) {
            const double* lamP = faceLamP + 4*((size_t)f*nFq + q);

            double NP[10];
            dg2_p2_basis_dev(lamP, NP);

            double bxP = 0.0;
            double byP = 0.0;
            double bzP = 0.0;

            const int baseP = 10*P;
            for (int j=0; j<10; ++j) {
                bxP += betaX[baseP+j] * NP[j];
                byP += betaY[baseP+j] * NP[j];
                bzP += betaZ[baseP+j] * NP[j];
            }

            const double bnP = bxP*nx + byP*ny + bzP*nz;
            const double absM = fabs(bnM);
            const double absP = fabs(bnP);
            const double lambda = absM > absP ? absM : absP;

            const int foff = faceOffsets[f];

            for (int i=0; i<10; ++i) {
                for (int j=0; j<10; ++j) {
                    const double aMM = w *  0.5*(bnM + lambda) * NM[i]*NM[j];
                    const double aMP = w *  0.5*(bnP - lambda) * NM[i]*NP[j];
                    const double aPM = w * -0.5*(bnM + lambda) * NP[i]*NM[j];
                    const double aPP = w * -0.5*(bnP - lambda) * NP[i]*NP[j];

                    const int base = foff + 4*(10*i + j);

                    const int p0 = facePos[base + 0];
                    const int p1 = facePos[base + 1];
                    const int p2 = facePos[base + 2];
                    const int p3 = facePos[base + 3];

                    if (p0 >= 0) atomicAdd(C + p0, aMM);
                    if (p1 >= 0) atomicAdd(C + p1, aMP);
                    if (p2 >= 0) atomicAdd(C + p2, aPM);
                    if (p3 >= 0) atomicAdd(C + p3, aPP);
                }
            }
        } else {
            if (bnM >= 0.0) {
                const int foff = faceOffsets[f];

                for (int i=0; i<10; ++i) {
                    for (int j=0; j<10; ++j) {
                        const int pos = facePos[foff + 10*i + j];
                        if (pos >= 0) {
                            atomicAdd(C + pos, w * bnM * NM[i]*NM[j]);
                        }
                    }
                }
            }
        }
    }
}

__global__ static void dg2_lf_conv_combine_kernel(
    int nnz,
    const double* __restrict__ Kdiff,
    const double* __restrict__ C,
    double* __restrict__ Aphys)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        Aphys[i] = Kdiff[i] + C[i];
    }
}

static void destroy_gpu_conv_lf_state(GpuConvLFState& st)
{
    dg2_cuda_free_ptr((void*&)st.d_vol);
    dg2_cuda_free_ptr((void*&)st.d_gradLam);
    dg2_cuda_free_ptr((void*&)st.d_tqLam);
    dg2_cuda_free_ptr((void*&)st.d_tqW);
    dg2_cuda_free_ptr((void*&)st.d_fqW);
    dg2_cuda_free_ptr((void*&)st.d_owner);
    dg2_cuda_free_ptr((void*&)st.d_neigh);
    dg2_cuda_free_ptr((void*&)st.d_nf);
    dg2_cuda_free_ptr((void*&)st.d_Af);
    dg2_cuda_free_ptr((void*&)st.d_faceLamM);
    dg2_cuda_free_ptr((void*&)st.d_faceLamP);
    dg2_cuda_free_ptr((void*&)st.d_volPos);
    dg2_cuda_free_ptr((void*&)st.d_faceOffsets);
    dg2_cuda_free_ptr((void*&)st.d_facePos);
    dg2_cuda_free_ptr((void*&)st.d_Kdiff);
    dg2_cuda_free_ptr((void*&)st.d_betaX);
    dg2_cuda_free_ptr((void*&)st.d_betaY);
    dg2_cuda_free_ptr((void*&)st.d_betaZ);
    dg2_cuda_free_ptr((void*&)st.d_C);
    dg2_cuda_free_ptr((void*&)st.d_Aphys);
    st.ready = false;
}

static void init_gpu_conv_lf_state(
    GpuConvLFState& st,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    const CSRPattern& pat,
    const std::vector<double>& Kdiff,
    const ConvLFPatternCache& convCache)
{
    if (!convCache.ready) {
        throw std::runtime_error("init_gpu_conv_lf_state: ConvLFPatternCache is not ready");
    }

    destroy_gpu_conv_lf_state(st);

    st.nCells = mesh.nCells;
    st.nFaces = mesh.nFaces;
    st.nInternalFaces = mesh.nInternalFaces;
    st.nTq = (int)tq.size();
    st.nFq = (int)fq.size();
    st.nU = pat.nRows;
    st.nnz = pat.nnz;

    std::vector<double> h_vol(mesh.nCells, 0.0);
    std::vector<double> h_gradLam((size_t)mesh.nCells * 12u, 0.0);

    for (int c=0; c<mesh.nCells; ++c) {
        h_vol[c] = tets[c].vol;
        for (int a=0; a<4; ++a) {
            for (int k=0; k<3; ++k) {
                h_gradLam[(size_t)c*12u + (size_t)a*3u + k] = tets[c].gradLam[a][k];
            }
        }
    }

    std::vector<double> h_tqLam((size_t)st.nTq * 4u, 0.0);
    std::vector<double> h_tqW(st.nTq, 0.0);

    for (int q=0; q<st.nTq; ++q) {
        for (int a=0; a<4; ++a) h_tqLam[(size_t)q*4u + a] = tq[q].lam[a];
        h_tqW[q] = tq[q].w;
    }

    std::vector<double> h_fqW(st.nFq, 0.0);
    for (int q=0; q<st.nFq; ++q) h_fqW[q] = fq[q].w;

    std::vector<int> h_owner(mesh.nFaces, -1);
    std::vector<int> h_neigh(mesh.nFaces, -1);
    std::vector<double> h_nf((size_t)mesh.nFaces * 3u, 0.0);
    std::vector<double> h_Af(mesh.nFaces, 0.0);

    for (int f=0; f<mesh.nFaces; ++f) {
        h_owner[f] = mesh.owner[f];
        h_neigh[f] = (f < mesh.nInternalFaces) ? mesh.neigh[f] : -1;
        h_Af[f] = mesh.Af[f];
        h_nf[(size_t)f*3u + 0] = mesh.nf[f][0];
        h_nf[(size_t)f*3u + 1] = mesh.nf[f][1];
        h_nf[(size_t)f*3u + 2] = mesh.nf[f][2];
    }

    std::vector<double> h_faceLamM((size_t)mesh.nFaces * (size_t)st.nFq * 4u, 0.0);
    std::vector<double> h_faceLamP((size_t)mesh.nFaces * (size_t)st.nFq * 4u, 0.0);

    int faceMapFailures = 0;

    for (int f=0; f<mesh.nFaces; ++f) {
        const int M = mesh.owner[f];
        const bool interior = (f < mesh.nInternalFaces);
        const int P = interior ? mesh.neigh[f] : -1;

        for (int q=0; q<st.nFq; ++q) {
            std::array<double,4> lamM{}, lamP{};

            if (!face_lam_on_tet(mesh.faces[f], tets[M], fq[q].mu, lamM)) {
                faceMapFailures++;
                continue;
            }

            for (int a=0; a<4; ++a) {
                h_faceLamM[4*((size_t)f*st.nFq + q) + a] = lamM[a];
            }

            if (interior) {
                if (!face_lam_on_tet(mesh.faces[f], tets[P], fq[q].mu, lamP)) {
                    faceMapFailures++;
                    continue;
                }

                for (int a=0; a<4; ++a) {
                    h_faceLamP[4*((size_t)f*st.nFq + q) + a] = lamP[a];
                }
            }
        }
    }

    if (faceMapFailures) {
        throw std::runtime_error("init_gpu_conv_lf_state: face mapping failed");
    }

    st.d_vol = dg2_cuda_malloc_copy_vec(h_vol, "vol");
    st.d_gradLam = dg2_cuda_malloc_copy_vec(h_gradLam, "gradLam");

    st.d_tqLam = dg2_cuda_malloc_copy_vec(h_tqLam, "tqLam");
    st.d_tqW = dg2_cuda_malloc_copy_vec(h_tqW, "tqW");
    st.d_fqW = dg2_cuda_malloc_copy_vec(h_fqW, "fqW");

    st.d_owner = dg2_cuda_malloc_copy_vec(h_owner, "owner");
    st.d_neigh = dg2_cuda_malloc_copy_vec(h_neigh, "neigh");
    st.d_nf = dg2_cuda_malloc_copy_vec(h_nf, "nf");
    st.d_Af = dg2_cuda_malloc_copy_vec(h_Af, "Af");

    st.d_faceLamM = dg2_cuda_malloc_copy_vec(h_faceLamM, "faceLamM");
    st.d_faceLamP = dg2_cuda_malloc_copy_vec(h_faceLamP, "faceLamP");

    st.d_volPos = dg2_cuda_malloc_copy_vec(convCache.volPos, "volPos");
    st.d_faceOffsets = dg2_cuda_malloc_copy_vec(convCache.faceOffsets, "faceOffsets");
    st.d_facePos = dg2_cuda_malloc_copy_vec(convCache.facePos, "facePos");

    st.d_Kdiff = dg2_cuda_malloc_copy_vec(Kdiff, "Kdiff");

    st.d_betaX = dg2_cuda_malloc_count<double>(st.nU, "betaX");
    st.d_betaY = dg2_cuda_malloc_count<double>(st.nU, "betaY");
    st.d_betaZ = dg2_cuda_malloc_count<double>(st.nU, "betaZ");
    st.d_C = dg2_cuda_malloc_count<double>(st.nnz, "Cbeta");
    st.d_Aphys = dg2_cuda_malloc_count<double>(st.nnz, "Aphys");

    st.ready = true;

    size_t freeB = 0, totalB = 0;
    DG2_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    std::printf("GPU LF convection state       = ready=1 nCells=%d nFaces=%d nTq=%d nFq=%d nnz=%d free=%.1f MiB total=%.1f MiB\n",
        st.nCells, st.nFaces, st.nTq, st.nFq, st.nnz,
        (double)freeB / (1024.0*1024.0),
        (double)totalB / (1024.0*1024.0));
}

static bool rebuild_physical_operator_from_beta_gpu(
    GpuConvLFState& st,
    const std::vector<double>& Kdiff,
    const std::vector<double>& betaX,
    const std::vector<double>& betaY,
    const std::vector<double>& betaZ,
    std::vector<double>& Aphys,
    int* convNnzOut)
{
    if (!st.ready) return false;

    if ((int)betaX.size() != st.nU || (int)betaY.size() != st.nU || (int)betaZ.size() != st.nU) {
        throw std::runtime_error("rebuild_physical_operator_from_beta_gpu: beta size mismatch");
    }

    if ((int)Kdiff.size() != st.nnz) {
        throw std::runtime_error("rebuild_physical_operator_from_beta_gpu: Kdiff size mismatch");
    }

    Aphys.resize(st.nnz);

    DG2_CUDA_CHECK(cudaMemcpy(st.d_betaX, betaX.data(), betaX.size()*sizeof(double), cudaMemcpyHostToDevice));
    DG2_CUDA_CHECK(cudaMemcpy(st.d_betaY, betaY.data(), betaY.size()*sizeof(double), cudaMemcpyHostToDevice));
    DG2_CUDA_CHECK(cudaMemcpy(st.d_betaZ, betaZ.data(), betaZ.size()*sizeof(double), cudaMemcpyHostToDevice));

    DG2_CUDA_CHECK(cudaMemset(st.d_C, 0, (size_t)st.nnz * sizeof(double)));

    const int block = 128;
    const int gridCells = (st.nCells + block - 1) / block;
    const int gridFaces = (st.nFaces + block - 1) / block;
    const int gridNnz = (st.nnz + block - 1) / block;

    dg2_lf_conv_volume_kernel<<<gridCells, block>>>(
        st.nCells,
        st.nTq,
        st.d_vol,
        st.d_gradLam,
        st.d_tqLam,
        st.d_tqW,
        st.d_betaX,
        st.d_betaY,
        st.d_betaZ,
        st.d_volPos,
        st.d_C);

    DG2_CUDA_CHECK(cudaGetLastError());

    dg2_lf_conv_face_kernel<<<gridFaces, block>>>(
        st.nFaces,
        st.nInternalFaces,
        st.nFq,
        st.d_owner,
        st.d_neigh,
        st.d_nf,
        st.d_Af,
        st.d_fqW,
        st.d_faceLamM,
        st.d_faceLamP,
        st.d_betaX,
        st.d_betaY,
        st.d_betaZ,
        st.d_faceOffsets,
        st.d_facePos,
        st.d_C);

    DG2_CUDA_CHECK(cudaGetLastError());

    dg2_lf_conv_combine_kernel<<<gridNnz, block>>>(
        st.nnz,
        st.d_Kdiff,
        st.d_C,
        st.d_Aphys);

    DG2_CUDA_CHECK(cudaGetLastError());
    DG2_CUDA_CHECK(cudaDeviceSynchronize());

    DG2_CUDA_CHECK(cudaMemcpy(Aphys.data(), st.d_Aphys, (size_t)st.nnz * sizeof(double), cudaMemcpyDeviceToHost));

    if (convNnzOut) {
        int n = 0;
        for (int i=0; i<st.nnz; ++i) {
            if (std::abs(Aphys[i] - Kdiff[i]) > 0.0) ++n;
        }
        *convNnzOut = n;
    }

    return true;
}

static void assemble_dg2_convection_lf_operator_from_dg2_velocity(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const CSRPattern& pat,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& betaX,
    const std::vector<double>& betaY,
    const std::vector<double>& betaZ,
    std::vector<double>& C,
    const ConvLFPatternCache* convCache = nullptr)
{
    const int nU = 10*mesh.nCells;

    if ((int)betaX.size() != nU || (int)betaY.size() != nU || (int)betaZ.size() != nU) {
        throw std::runtime_error("assemble_dg2_convection_lf_operator_from_dg2_velocity: beta size mismatch");
    }

    C.assign(pat.nnz, 0.0);

    const bool haveConvCache =
        convCache && convCache->ready &&
        (int)convCache->volPos.size() == 100*mesh.nCells &&
        (int)convCache->faceOffsets.size() == mesh.nFaces + 1;

    // Conservative DG volume form:
    //   c(u,v) = - int_K u beta.grad(v)
    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];

        for (const auto& q : tq) {
            double N[10];
            std::array<double,3> G[10];

            p2_tet_basis(q.lam, N);
            p2_tet_grad(K, q.lam, G);

            std::array<double,3> beta{0.0,0.0,0.0};

            for (int j=0; j<10; ++j) {
                beta[0] += betaX[10*c+j] * N[j];
                beta[1] += betaY[10*c+j] * N[j];
                beta[2] += betaZ[10*c+j] * N[j];
            }

            const double w = K.vol * q.w;

            for (int i=0; i<10; ++i) {
                const int row = 10*c + i;
                const double bdotGradTest = dot3(beta, G[i]);

                for (int j=0; j<10; ++j) {
                    const int col = 10*c + j;
                    const double aij = -w * bdotGradTest * N[j];
                    if (haveConvCache) {
                        const int pos =
                            convCache->volPos[(std::size_t)c*100u + (std::size_t)i*10u + (std::size_t)j];
                        C[pos] += aij;
                    } else {
                        add_A(pat, C, row, col, aij);
                    }
                }
            }
        }
    }

    int faceMapFailures = 0;

    // LF/upwind face fluxes.  Normal is owner -> neighbour/boundary.
    for (int f=0; f<mesh.nFaces; ++f) {
        const int M = mesh.owner[f];
        const bool interior = (f < mesh.nInternalFaces);
        const int P = interior ? mesh.neigh[f] : -1;

        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        const auto& KM = tets[M];
        const TetP2Geom* KP = interior ? &tets[P] : nullptr;

        for (const auto& qf : fq) {
            std::array<double,4> lamM{}, lamP{};

            if (!face_lam_on_tet(mesh.faces[f], KM, qf.mu, lamM)) {
                faceMapFailures++;
                continue;
            }

            double NM[10];
            p2_tet_basis(lamM, NM);

            std::array<double,3> betaM{0.0,0.0,0.0};

            for (int j=0; j<10; ++j) {
                betaM[0] += betaX[10*M+j] * NM[j];
                betaM[1] += betaY[10*M+j] * NM[j];
                betaM[2] += betaZ[10*M+j] * NM[j];
            }

            const double bnM = dot3(betaM, n);
            const double w = area * qf.w;

            if (interior) {
                if (!face_lam_on_tet(mesh.faces[f], *KP, qf.mu, lamP)) {
                    faceMapFailures++;
                    continue;
                }

                double NP[10];
                p2_tet_basis(lamP, NP);

                std::array<double,3> betaP{0.0,0.0,0.0};

                for (int j=0; j<10; ++j) {
                    betaP[0] += betaX[10*P+j] * NP[j];
                    betaP[1] += betaY[10*P+j] * NP[j];
                    betaP[2] += betaZ[10*P+j] * NP[j];
                }

                const double bnP = dot3(betaP, n);
                // LF/upwind uses lambda=max(|beta^- . n|, |beta^+ . n|).
                // Central differencing is obtained by setting lambda=0 while
                // keeping the same conservative owner/neighbour face pair.
                const double lambda = momentum_flux_is_central()
                    ? 0.0
                    : std::max(std::abs(bnM), std::abs(bnP));

                for (int i=0; i<10; ++i) {
                    const int rowM = 10*M + i;
                    const int rowP = 10*P + i;

                    for (int j=0; j<10; ++j) {
                        const int colM = 10*M + j;
                        const int colP = 10*P + j;

                        const double aMM = w *  0.5*(bnM + lambda) * NM[i]*NM[j];
                        const double aMP = w *  0.5*(bnP - lambda) * NM[i]*NP[j];
                        const double aPM = w * -0.5*(bnM + lambda) * NP[i]*NM[j];
                        const double aPP = w * -0.5*(bnP - lambda) * NP[i]*NP[j];

                        if (haveConvCache) {
                            const int base = convCache->faceOffsets[f] + 4*(10*i + j);
                            C[convCache->facePos[base + 0]] += aMM;
                            C[convCache->facePos[base + 1]] += aMP;
                            C[convCache->facePos[base + 2]] += aPM;
                            C[convCache->facePos[base + 3]] += aPP;
                        } else {
                            add_A(pat, C, rowM, colM, aMM);
                            add_A(pat, C, rowM, colP, aMP);
                            add_A(pat, C, rowP, colM, aPM);
                            add_A(pat, C, rowP, colP, aPP);
                        }
                    }
                }
            } else {
                double ownerCoeff = 0.0;

                if (gCaseIsChannel) {
                    const int bkind = channel_boundary_kind(mesh, f);

                    if (bkind == 2) {
                        // Open outlet: exterior transported state is the interior state.
                        // This makes the boundary convective flux beta.n * u_M rather than
                        // an MMS zero-exterior inflow clamp.
                        ownerCoeff = bnM;
                    } else {
                        // Inlet/wall: LF flux with prescribed exterior state.  The matrix
                        // part multiplies u_M; the exterior prescribed part is assembled
                        // into a boundary RHS by assemble_channel_convection_lf_boundary_rhs.
                        const auto xq = lincomb4(KM.x, lamM);
                        const auto ubc = (bkind == 1)
                            ? channel_velocity_bc(xq, n)
                            : std::array<double,3>{0.0, 0.0, 0.0};
                        const double bnExt = dot3(ubc, n);
                        if (momentum_flux_is_central()) {
                            // Central boundary flux: 0.5*(beta_M.n*u_M + beta_ext.n*u_ext).
                            ownerCoeff = 0.5 * bnM;
                        } else {
                            const double lambda = std::max(std::abs(bnM), std::abs(bnExt));
                            ownerCoeff = 0.5 * (bnM + lambda);
                        }
                    }
                } else {
                    // Historical MMS boundary: upwind exterior transported value is zero.
                    // This is consistent with the manufactured velocity vanishing on the
                    // boundary and preserves the validated MMS path exactly.
                    ownerCoeff = momentum_flux_is_central()
                        ? 0.5 * bnM
                        : ((bnM >= 0.0) ? bnM : 0.0);
                }

                if (std::abs(ownerCoeff) > 0.0) {
                    for (int i=0; i<10; ++i) {
                        const int row = 10*M + i;

                        for (int j=0; j<10; ++j) {
                            const int col = 10*M + j;
                            const double aij = w * ownerCoeff * NM[i]*NM[j];
                            if (haveConvCache) {
                                const int base = convCache->faceOffsets[f] + 10*i + j;
                                C[convCache->facePos[base]] += aij;
                            } else {
                                add_A(pat, C, row, col, aij);
                            }
                        }
                    }
                }
            }
        }
    }

    if (faceMapFailures) {
        std::fprintf(stderr, "LF convection face mapping failures = %d\n", faceMapFailures);
        throw std::runtime_error("face mapping failed in LF convection assembly");
    }
}

static void assemble_channel_convection_lf_boundary_rhs(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& betaX,
    const std::vector<double>& betaY,
    const std::vector<double>& betaZ,
    std::vector<double>& Fx,
    std::vector<double>& Fy,
    std::vector<double>& Fz)
{
    const int nU = 10*mesh.nCells;
    Fx.assign(nU, 0.0);
    Fy.assign(nU, 0.0);
    Fz.assign(nU, 0.0);

    if (!gCaseIsChannel) return;

    int faceMapFailures = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);

        // Outlet uses exterior=interior, hence no prescribed-state RHS.
        // Walls have zero prescribed velocity, hence no RHS either.  Keeping
        // the general inlet/wall formula below is harmless, but skip outlet.
        if (bkind == 2) continue;

        const int M = mesh.owner[f];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];
        const auto& KM = tets[M];

        for (const auto& qf : fq) {
            std::array<double,4> lamM{};
            if (!face_lam_on_tet(mesh.faces[f], KM, qf.mu, lamM)) {
                faceMapFailures++;
                continue;
            }

            double NM[10];
            p2_tet_basis(lamM, NM);

            std::array<double,3> betaM{0.0,0.0,0.0};
            for (int j=0; j<10; ++j) {
                betaM[0] += betaX[10*M+j] * NM[j];
                betaM[1] += betaY[10*M+j] * NM[j];
                betaM[2] += betaZ[10*M+j] * NM[j];
            }

            const double bnM = dot3(betaM, n);
            const auto xq = lincomb4(KM.x, lamM);
            const auto ubc = (bkind == 1)
                ? channel_velocity_bc(xq, n)
                : std::array<double,3>{0.0, 0.0, 0.0};
            const double bnExt = dot3(ubc, n);

            double rhsCoeff = 0.0;
            if (momentum_flux_is_central()) {
                // Boundary central flux contribution from prescribed exterior state:
                // F*_ext = 0.5*bnExt*u_ext. Since C u is on the left-hand
                // side, move this known term to RHS.
                rhsCoeff = -0.5 * bnExt;
            } else {
                const double lambda = std::max(std::abs(bnM), std::abs(bnExt));

                // Boundary LF flux contribution from prescribed exterior state:
                // F*_ext = 0.5*(bnExt-lambda) * u_ext.
                // Since C u is on the left-hand side, move this known term to RHS.
                rhsCoeff = -0.5 * (bnExt - lambda);
            }
            if (std::abs(rhsCoeff) == 0.0) continue;

            const double w = area * qf.w;
            for (int i=0; i<10; ++i) {
                const int row = 10*M + i;
                const double wi = w * rhsCoeff * NM[i];
                Fx[row] += wi * ubc[0];
                Fy[row] += wi * ubc[1];
                Fz[row] += wi * ubc[2];
            }
        }
    }

    if (faceMapFailures) {
        std::fprintf(stderr, "channel LF convection RHS face mapping failures = %d\n", faceMapFailures);
        throw std::runtime_error("face mapping failed in channel LF convection RHS assembly");
    }
}




static int adjust_channel_outlet_flux_trace(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    std::vector<double>& ux,
    std::vector<double>& uy,
    std::vector<double>& uz,
    double* netBeforeOut,
    double* deltaUnOut,
    double* areaOutletOut)
{
    if (!gCaseIsChannel) return 0;

    double fluxIn = 0.0;
    double fluxOut = 0.0;
    double fluxWall = 0.0;
    double areaOut = 0.0;
    int mapFail = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;

        const int M = mesh.owner[f];
        const auto& K = tets[M];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        if (bkind == 2) areaOut += area;

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            double u0 = 0.0, u1 = 0.0, u2 = 0.0;
            for (int j=0; j<10; ++j) {
                u0 += ux[10*M+j] * N[j];
                u1 += uy[10*M+j] * N[j];
                u2 += uz[10*M+j] * N[j];
            }

            const double ww = area * q.w;
            const double un = u0*n[0] + u1*n[1] + u2*n[2];

            if (bkind == 1) fluxIn += ww*un;
            else if (bkind == 2) fluxOut += ww*un;
            else fluxWall += ww*un;
        }
    }

    const double netBefore = fluxIn + fluxOut + fluxWall;
    const double deltaUn = (areaOut > 1e-300) ? (-netBefore / areaOut) : 0.0;

    if (netBeforeOut) *netBeforeOut = netBefore;
    if (deltaUnOut) *deltaUnOut = deltaUn;
    if (areaOutletOut) *areaOutletOut = areaOut;

    if (areaOut <= 1e-300) return 0;

    const int nU = 10 * mesh.nCells;
    std::vector<unsigned char> touched(nU, 0);

    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    int nAdjusted = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        if (channel_boundary_kind(mesh, f) != 2) continue;

        const int c = mesh.owner[f];
        const auto& K = tets[c];
        const auto n = mesh.nf[f];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<3; ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (!onFace) continue;

            const int row = 10*c + i;
            if (row < 0 || row >= nU || touched[row]) continue;

            ux[row] += deltaUn * n[0];
            uy[row] += deltaUn * n[1];
            uz[row] += deltaUn * n[2];

            touched[row] = 1;
            ++nAdjusted;
        }
    }

    if (mapFail) {
        std::printf("WARNING: adjust_channel_outlet_flux_trace mapFail=%d\n", mapFail);
    }

    return nAdjusted;
}

static void print_channel_patch_component_flux_diagnostic(
    int it,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz)
{
    if (!gCaseIsChannel) return;

    double areaIn = 0.0, areaOut = 0.0, areaWall = 0.0;
    double fluxIn = 0.0, fluxOut = 0.0, fluxWall = 0.0;
    double absFluxWall = 0.0;

    double uzIn = 0.0, uzOut = 0.0;
    double unIn = 0.0, unOut = 0.0;
    double magIn = 0.0, magOut = 0.0;

    double nzIn = 0.0, nzOut = 0.0;

    int facesIn = 0, facesOut = 0, facesWall = 0;
    int mapFail = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;

        const int M = mesh.owner[f];
        const auto& K = tets[M];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        if (bkind == 1) { areaIn += area; ++facesIn; nzIn += area*n[2]; }
        else if (bkind == 2) { areaOut += area; ++facesOut; nzOut += area*n[2]; }
        else { areaWall += area; ++facesWall; }

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            double u0 = 0.0, u1 = 0.0, u2 = 0.0;
            for (int j=0; j<10; ++j) {
                u0 += ux[10*M+j] * N[j];
                u1 += uy[10*M+j] * N[j];
                u2 += uz[10*M+j] * N[j];
            }

            const double ww = area * q.w;
            const double un = u0*n[0] + u1*n[1] + u2*n[2];
            const double umag = std::sqrt(u0*u0 + u1*u1 + u2*u2);

            if (bkind == 1) {
                fluxIn += ww*un;
                uzIn += ww*u2;
                unIn += ww*un;
                magIn += ww*umag;
            } else if (bkind == 2) {
                fluxOut += ww*un;
                uzOut += ww*u2;
                unOut += ww*un;
                magOut += ww*umag;
            } else {
                fluxWall += ww*un;
                absFluxWall += ww*std::abs(un);
            }
        }
    }

    const double invAIn = areaIn > 1e-300 ? 1.0/areaIn : 0.0;
    const double invAOut = areaOut > 1e-300 ? 1.0/areaOut : 0.0;
    const double invAWall = areaWall > 1e-300 ? 1.0/areaWall : 0.0;

    std::printf("    pipePatchFlux: it=%d "
                "A(in,out,wall)=(%.6e,%.6e,%.6e) "
                "nbarZ(in,out)=(%.6e,%.6e) "
                "flux(in,out,wall)=(% .6e,% .6e,% .6e) net=% .6e "
                "avgUz(in,out)=(% .6e,% .6e) avgUn(in,out)=(% .6e,% .6e) "
                "avg|U|(in,out)=(%.6e,%.6e) wallAbsUnAvg=%.6e "
                "faces(in,out,wall)=(%d,%d,%d) mapFail=%d\n",
                it,
                areaIn, areaOut, areaWall,
                nzIn*invAIn, nzOut*invAOut,
                fluxIn, fluxOut, fluxWall, fluxIn + fluxOut + fluxWall,
                uzIn*invAIn, uzOut*invAOut,
                unIn*invAIn, unOut*invAOut,
                magIn*invAIn, magOut*invAOut,
                absFluxWall*invAWall,
                facesIn, facesOut, facesWall, mapFail);
}

static void print_channel_boundary_flux_diagnostic(
    int it,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz)
{
    if (!gCaseIsChannel) return;

    double inArea = 0.0, outArea = 0.0;
    double inFlux = 0.0, outFlux = 0.0;
    double inUx = 0.0, outUx = 0.0;
    double inMag = 0.0, outMag = 0.0;
    int inFaces = 0, outFaces = 0;
    int faceMapFailures = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != 1 && bkind != 2) continue;

        if (bkind == 1) ++inFaces;
        if (bkind == 2) ++outFaces;

        const int M = mesh.owner[f];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];
        const auto& KM = tets[M];

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], KM, q.mu, lam)) {
                ++faceMapFailures;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            std::array<double,3> u{0.0, 0.0, 0.0};
            for (int j=0; j<10; ++j) {
                u[0] += ux[10*M+j] * N[j];
                u[1] += uy[10*M+j] * N[j];
                u[2] += uz[10*M+j] * N[j];
            }

            const double w = area * q.w;
            const double un = dot3(u, n);
            const double mag = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);

            if (bkind == 1) {
                inArea += w;
                inFlux += w * un;
                inUx += w * u[0];
                inMag += w * mag;
            } else {
                outArea += w;
                outFlux += w * un;
                outUx += w * u[0];
                outMag += w * mag;
            }
        }
    }

    const double inAvgUx  = inUx  / std::max(inArea,  1e-300);
    const double outAvgUx = outUx / std::max(outArea, 1e-300);
    const double inAvgMag  = inMag  / std::max(inArea,  1e-300);
    const double outAvgMag = outMag / std::max(outArea, 1e-300);

    std::printf("    channelFlux: it=%d inletOutward=% .6e outletOutward=% .6e net=% .6e avgUx(in,out)=(% .6e,% .6e) avg|U|(in,out)=(% .6e,% .6e) faces(in,out)=(%d,%d) mapFail=%d\n",
        it, inFlux, outFlux, inFlux + outFlux,
        inAvgUx, outAvgUx, inAvgMag, outAvgMag,
        inFaces, outFaces, faceMapFailures);
}


static std::vector<double> compute_channel_pressure_face_weights(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    int wantKind)
{
    std::vector<double> w(mesh.P.size(), 0.0);
    if (!gCaseIsChannel) return w;

    int mapFail = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != wantKind) continue;

        const int M = mesh.owner[f];
        const auto& K = tets[M];
        const double area = mesh.Af[f];

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            const double ww = area * q.w;
            for (int a=0; a<4; ++a) {
                w[K.v[a]] += ww * lam[a];
            }
        }
    }

    if (mapFail) {
        std::printf("WARNING: channel outlet pressure-face weight map failures = %d\n", mapFail);
    }

    return w;
}

static void make_rhs_compatible_by_outlet_weights(
    std::vector<double>& rhs,
    const std::vector<double>& outletWeights)
{
    double sum = 0.0;
    double wsum = 0.0;

    for (double v : rhs) sum += v;
    for (double w : outletWeights) wsum += w;

    if (std::abs(wsum) <= 1e-300) {
        const double mean = rhs.empty() ? 0.0 : sum / (double)rhs.size();
        for (double& v : rhs) v -= mean;
        return;
    }

    for (std::size_t i=0; i<rhs.size(); ++i) {
        rhs[i] -= sum * outletWeights[i] / wsum;
    }
}


static void assemble_channel_direct_continuity_velocity_to_pressure(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    RectCSR& Bdx,
    RectCSR& Bdy,
    RectCSR& Bdz,
    std::vector<double>& fixedSource)
{
    const int nU = 10 * mesh.nCells;
    const int nP = 4 * mesh.nCells;

    std::vector<std::map<int,double>> rowsX(nU), rowsY(nU), rowsZ(nU);
    fixedSource.assign(nP, 0.0);

    // Volume contribution for DG1 pressure rows:
    //   r_K(q_a) += - int_K grad(lambda_a) . u dV.
    // Bdx/Bdy/Bdz are stored transposed: velocity-row x pressure-column = B^T.
    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];
        for (const auto& qp : tq) {
            double N[10];
            p2_tet_basis(qp.lam, N);
            const double ww = K.vol * qp.w;
            for (int j=0; j<10; ++j) {
                const int rowU = 10*c + j;
                for (int a=0; a<4; ++a) {
                    const int rowP = 4*c + a;
                    rowsX[rowU][rowP] += -ww * K.gradLam[a][0] * N[j];
                    rowsY[rowU][rowP] += -ww * K.gradLam[a][1] * N[j];
                    rowsZ[rowU][rowP] += -ww * K.gradLam[a][2] * N[j];
                }
            }
        }
    }

    int mapFail = 0;
    int inletFaces = 0;
    int outletFaces = 0;
    int wallFaces = 0;
    int interiorFaces = 0;

    // Interior faces: one central single-valued mass flux with opposite sign
    // on the neighbor residual.  n is owner(M) outward normal.
    for (int f=0; f<mesh.nInternalFaces; ++f) {
        const int M = mesh.owner[f];
        const int P = mesh.neigh[f];
        if (M < 0 || M >= mesh.nCells || P < 0 || P >= mesh.nCells) continue;
        ++interiorFaces;

        const auto& KM = tets[M];
        const auto& KP = tets[P];
        const auto n = mesh.nf[f];
        const double Af = mesh.Af[f];

        for (const auto& q : fq) {
            std::array<double,4> lamM{};
            std::array<double,4> lamP{};
            if (!face_lam_on_tet(mesh.faces[f], KM, q.mu, lamM)) { ++mapFail; continue; }
            if (!face_lam_on_tet(mesh.faces[f], KP, q.mu, lamP)) { ++mapFail; continue; }

            double NM[10], NP[10];
            p2_tet_basis(lamM, NM);
            p2_tet_basis(lamP, NP);

            const double ww = Af * q.w;

            for (int a=0; a<4; ++a) {
                const int pM = 4*M + a;
                const int pP = 4*P + a;

                for (int j=0; j<10; ++j) {
                    const int uM = 10*M + j;
                    const int uP = 10*P + j;

                    const double cMM =  0.5 * ww * lamM[a] * NM[j];
                    const double cMP =  0.5 * ww * lamM[a] * NP[j];
                    const double cPM = -0.5 * ww * lamP[a] * NM[j];
                    const double cPP = -0.5 * ww * lamP[a] * NP[j];

                    rowsX[uM][pM] += cMM * n[0]; rowsY[uM][pM] += cMM * n[1]; rowsZ[uM][pM] += cMM * n[2];
                    rowsX[uP][pM] += cMP * n[0]; rowsY[uP][pM] += cMP * n[1]; rowsZ[uP][pM] += cMP * n[2];
                    rowsX[uM][pP] += cPM * n[0]; rowsY[uM][pP] += cPM * n[1]; rowsZ[uM][pP] += cPM * n[2];
                    rowsX[uP][pP] += cPP * n[0]; rowsY[uP][pP] += cPP * n[1]; rowsZ[uP][pP] += cPP * n[2];
                }
            }
        }
    }

    // Boundary faces:
    //   inlet: prescribed normal flux in fixedSource bD,
    //   walls/cylinder: prescribed zero normal flux,
    //   outlet: open/natural, interior velocity contributes to B*u.
    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;

        if (bkind == 1) ++inletFaces;
        else if (bkind == 2) ++outletFaces;
        else ++wallFaces;

        const int c = mesh.owner[f];
        if (c < 0 || c >= mesh.nCells) continue;
        const auto& K = tets[c];
        const auto n = mesh.nf[f];
        const double Af = mesh.Af[f];

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) { ++mapFail; continue; }

            double N[10];
            p2_tet_basis(lam, N);
            const double ww = Af * q.w;

            if (bkind == 2) {
                for (int a=0; a<4; ++a) {
                    const int rowP = 4*c + a;
                    for (int j=0; j<10; ++j) {
                        const int rowU = 10*c + j;
                        const double coeff = ww * lam[a] * N[j];
                        rowsX[rowU][rowP] += coeff * n[0];
                        rowsY[rowU][rowP] += coeff * n[1];
                        rowsZ[rowU][rowP] += coeff * n[2];
                    }
                }
            } else if (bkind == 1) {
                const auto xq = lincomb4(K.x, lam);
                const auto ubc = channel_velocity_bc(xq, n);
                const double un = dot3(ubc, n);
                for (int a=0; a<4; ++a) fixedSource[4*c + a] += ww * lam[a] * un;
            } else {
                // no-penetration wall/cylinder: zero source
            }
        }
    }

    Bdx = rows_to_rectcsr(nU, nP, rowsX);
    Bdy = rows_to_rectcsr(nU, nP, rowsY);
    Bdz = rows_to_rectcsr(nU, nP, rowsZ);

    std::printf("DG2/DG1 single-flux continuity B assembly: Bdx/Bdy/Bdz nnz=%d/%d/%d fixedSourceNorm=%.6e faces(int,in,out,wall)=(%d,%d,%d,%d) mapFail=%d\n",
        Bdx.nnz, Bdy.nnz, Bdz.nnz, norm_vec(fixedSource),
        interiorFaces, inletFaces, outletFaces, wallFaces, mapFail);
}


static void apply_channel_direct_continuity_linear(
    const RectCSR& Bdx,
    const RectCSR& Bdy,
    const RectCSR& Bdz,
    const std::vector<double>& fixedSource,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    std::vector<double>& r)
{
    std::vector<double> tmp;

    apply_pos_transpose_rect(Bdx, ux, r);

    apply_pos_transpose_rect(Bdy, uy, tmp);
    axpy_vec(r, 1.0, tmp);

    apply_pos_transpose_rect(Bdz, uz, tmp);
    axpy_vec(r, 1.0, tmp);

    if (!fixedSource.empty()) {
        axpy_vec(r, 1.0, fixedSource);
    }
}

static void assemble_lpmass_direct_continuity_schur(
    int nP,
    const Mesh& mesh,
    const RectCSR& Bdx,
    const RectCSR& Bdy,
    const RectCSR& Bdz,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    CSRPattern& lpPat,
    std::vector<double>& LpValues)
{
    std::vector<std::map<int,double>> rows(nP);

    auto add_direction = [&](const RectCSR& Bd, const RectCSR& Ap) {
        for (int c=0; c<mesh.nCells; ++c) {
            for (int a=0; a<10; ++a) {
                const int rowB = 10*c + a;

                for (int b=0; b<10; ++b) {
                    const int rowA = 10*c + b;
                    const double mij = invM[c][a][b];
                    if (std::abs(mij) <= 1e-300) continue;

                    for (int pb=Bd.rowOffsets[rowB]; pb<Bd.rowOffsets[rowB+1]; ++pb) {
                        const int pTest = Bd.cols[pb];
                        const double vb = Bd.values[pb];
                        if (std::abs(vb) <= 1e-300) continue;

                        for (int pa=Ap.rowOffsets[rowA]; pa<Ap.rowOffsets[rowA+1]; ++pa) {
                            const int pTrial = Ap.cols[pa];
                            const double va = Ap.values[pa];
                            if (std::abs(va) <= 1e-300) continue;

                            // The existing mass-Schur RHS path later applies rhs = -rhs.
                            // The velocity correction then changes direct continuity by
                            //   delta(Bdirect u) = Bdirect_linear M^{-1} Ap pcorr.
                            // Therefore the operator assembled here must be
                            //   S = + Bdirect_linear M^{-1} Ap
                            // so that S pcorr = -Bdirect(u*) drives Bdirect(u*+du) -> 0.
                            rows[pTest][pTrial] += -vb * mij * va;
                        }
                    }
                }
            }
        }
    };

    add_direction(Bdx, Apx);
    add_direction(Bdy, Apy);
    add_direction(Bdz, Apz);

    lpPat = rows_to_csrpattern(nP, rows, LpValues);
}

static void assemble_lpmass_open_transpose_schur(
    int nP,
    const Mesh& mesh,
    const RectCSR& Bdx,
    const RectCSR& Bdy,
    const RectCSR& Bdz,
    const std::vector<std::array<std::array<double,10>,10>>& invM,
    const std::vector<unsigned char>* lockedCorrectionRows,
    CSRPattern& lpPat,
    std::vector<double>& LpValues)
{
    std::vector<std::map<int,double>> rows(nP);

    auto add_direction = [&](const RectCSR& Bd) {
        for (int c=0; c<mesh.nCells; ++c) {
            for (int a=0; a<10; ++a) {
                const int rowOut = 10*c + a;

                // MATLAB uses Pfree*H*B^T: mask the corrected output row after H.
                if (lockedCorrectionRows &&
                    rowOut < (int)lockedCorrectionRows->size() &&
                    (*lockedCorrectionRows)[rowOut]) {
                    continue;
                }

                for (int b=0; b<10; ++b) {
                    const int rowIn = 10*c + b;

                    // Right-side Pfree in Bopen * Pfree * H * Pfree * Bopen^T.
                    if (lockedCorrectionRows &&
                        rowIn < (int)lockedCorrectionRows->size() &&
                        (*lockedCorrectionRows)[rowIn]) {
                        continue;
                    }

                    const double hij = invM[c][a][b];
                    if (std::abs(hij) <= 1e-300) continue;

                    for (int pt=Bd.rowOffsets[rowOut]; pt<Bd.rowOffsets[rowOut+1]; ++pt) {
                        const int pTest = Bd.cols[pt];
                        const double vTest = Bd.values[pt];
                        if (std::abs(vTest) <= 1e-300) continue;

                        for (int pr=Bd.rowOffsets[rowIn]; pr<Bd.rowOffsets[rowIn+1]; ++pr) {
                            const int pTrial = Bd.cols[pr];
                            const double vTrial = Bd.values[pr];
                            if (std::abs(vTrial) <= 1e-300) continue;

                            // Bopen * Pfree * H * Pfree * Bopen^T.
                            // With rhs=-r and u <- Pfree*H*Pfree*Bopen^T*pcorr,
                            // r_new = r + Lp*pcorr.
                            rows[pTest][pTrial] += vTest * hij * vTrial;
                        }
                    }
                }
            }
        }
    };

    add_direction(Bdx);
    add_direction(Bdy);
    add_direction(Bdz);

    lpPat = rows_to_csrpattern(nP, rows, LpValues);
}

static void compute_channel_cg1_continuity_residual_direct(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    std::vector<double>& r,
    double* inletFluxOut,
    double* outletFluxOut,
    double* netFluxOut,
    double* residualSumOut)
{
    const int nP = 4 * mesh.nCells;
    r.assign(nP, 0.0);

    double inletFlux = 0.0;
    double outletFlux = 0.0;
    int mapFail = 0;

    // Volume part: -int_K grad(q).u.
    for (int c=0; c<mesh.nCells; ++c) {
        const auto& K = tets[c];
        for (const auto& qp : tq) {
            double N[10];
            p2_tet_basis(qp.lam, N);

            std::array<double,3> u{0.0,0.0,0.0};
            for (int j=0; j<10; ++j) {
                u[0] += ux[10*c+j] * N[j];
                u[1] += uy[10*c+j] * N[j];
                u[2] += uz[10*c+j] * N[j];
            }

            const double ww = K.vol * qp.w;
            for (int a=0; a<4; ++a) {
                r[4*c + a] += -ww * dot3(K.gradLam[a], u);
            }
        }
    }

    // Interior central single-valued flux.
    for (int f=0; f<mesh.nInternalFaces; ++f) {
        const int M = mesh.owner[f];
        const int P = mesh.neigh[f];
        if (M < 0 || M >= mesh.nCells || P < 0 || P >= mesh.nCells) continue;

        const auto& KM = tets[M];
        const auto& KP = tets[P];
        const auto n = mesh.nf[f];
        const double Af = mesh.Af[f];

        for (const auto& q : fq) {
            std::array<double,4> lamM{};
            std::array<double,4> lamP{};
            if (!face_lam_on_tet(mesh.faces[f], KM, q.mu, lamM)) { ++mapFail; continue; }
            if (!face_lam_on_tet(mesh.faces[f], KP, q.mu, lamP)) { ++mapFail; continue; }

            double NM[10], NP[10];
            p2_tet_basis(lamM, NM);
            p2_tet_basis(lamP, NP);

            std::array<double,3> uM{0.0,0.0,0.0};
            std::array<double,3> uP{0.0,0.0,0.0};
            for (int j=0; j<10; ++j) {
                uM[0] += ux[10*M+j] * NM[j]; uM[1] += uy[10*M+j] * NM[j]; uM[2] += uz[10*M+j] * NM[j];
                uP[0] += ux[10*P+j] * NP[j]; uP[1] += uy[10*P+j] * NP[j]; uP[2] += uz[10*P+j] * NP[j];
            }
            const double uhatn = 0.5 * dot3(add3(uM, uP), n);
            const double ww = Af * q.w;

            for (int a=0; a<4; ++a) {
                r[4*M + a] += ww * lamM[a] * uhatn;
                r[4*P + a] -= ww * lamP[a] * uhatn;
            }
        }
    }

    // Boundary fluxes.
    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind < 0) continue;
        const int M = mesh.owner[f];
        if (M < 0 || M >= mesh.nCells) continue;
        const auto& K = tets[M];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) { ++mapFail; continue; }

            double N[10];
            p2_tet_basis(lam, N);

            std::array<double,3> u{0.0,0.0,0.0};
            if (bkind == 2) {
                for (int j=0; j<10; ++j) {
                    u[0] += ux[10*M+j] * N[j];
                    u[1] += uy[10*M+j] * N[j];
                    u[2] += uz[10*M+j] * N[j];
                }
            } else if (bkind == 1) {
                const auto xq = lincomb4(K.x, lam);
                u = channel_velocity_bc(xq, n);
            } else {
                u = {0.0, 0.0, 0.0};
            }

            const double ww = area * q.w;
            const double un = dot3(u, n);
            if (bkind == 1) inletFlux += ww * un;
            if (bkind == 2) outletFlux += ww * un;
            for (int a=0; a<4; ++a) r[4*M + a] += ww * lam[a] * un;
        }
    }

    double rsum = 0.0;
    for (double v : r) rsum += v;

    if (inletFluxOut) *inletFluxOut = inletFlux;
    if (outletFluxOut) *outletFluxOut = outletFlux;
    if (netFluxOut) *netFluxOut = inletFlux + outletFlux;
    if (residualSumOut) *residualSumOut = rsum;

    if (mapFail) {
        std::printf("WARNING: DG2/DG1 single-flux continuity face map failures = %d\n", mapFail);
    }
}

static void print_dg1_cell_constant_residual_audit(
    const char* label,
    int it,
    const std::vector<double>& r)
{
    if (r.empty()) return;
    const int nCells = (int)r.size() / 4;
    if (4*nCells != (int)r.size()) return;
    double l2 = 0.0;
    double maxAbs = 0.0;
    double sum = 0.0;
    for (int c=0; c<nCells; ++c) {
        const double rc = r[4*c+0] + r[4*c+1] + r[4*c+2] + r[4*c+3];
        l2 += rc*rc;
        maxAbs = std::max(maxAbs, std::abs(rc));
        sum += rc;
    }
    std::printf("    dg1CellFluxAudit: it=%d stage=%s q1L2=%.6e q1Max=%.6e q1Sum=% .6e cells=%d\n",
        it, label, std::sqrt(std::max(0.0,l2)), maxAbs, sum, nCells);
}




static int enforce_channel_velocity_dirichlet_trace(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    std::vector<double>& ux,
    std::vector<double>& uy,
    std::vector<double>& uz)
{
    if (!gCaseIsChannel) return 0;

    const int nU = 10 * mesh.nCells;
    std::vector<unsigned char> kind(nU, 0);        // 0 none, 1 wall, 2 inlet
    std::vector<unsigned char> inletTrace(nU, 0);  // rows belonging to inlet trace
    std::vector<unsigned char> outletTrace(nU, 0); // rows belonging to outlet trace
    std::vector<double> tx(nU, 0.0), ty(nU, 0.0), tz(nU, 0.0);

    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    auto mark_face_rows = [&](int f, std::vector<unsigned char>& mark) {
        const int c = mesh.owner[f];
        const auto& K = tets[c];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<3; ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (onFace) {
                mark[10*c + i] = 1;
            }
        }
    };

    // First pass: identify cap/inlet/outlet trace rows globally.
    // This is needed because P2 trace rows on cap rims are also seen by wall faces.
    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind == 1) {
            mark_face_rows(f, inletTrace);
        } else if (bkind == 2) {
            mark_face_rows(f, outletTrace);
        }
    }

    int skippedWallOutletRows = 0;
    int skippedWallInletRows = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != 0 && bkind != 1) continue; // outlet is intentionally untouched

        const int c = mesh.owner[f];
        const auto& K = tets[c];
        const auto n = mesh.nf[f];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<3; ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (!onFace) continue;

            const int row = 10*c + i;

            if (bkind == 1) {
                // Inlet has priority over wall at geometric edges/corners.
                // If an exotic mesh has inlet/outlet overlap, inlet is still a prescribed velocity patch.
                const auto ubc = channel_velocity_bc(K.xP2[i], n);
                kind[row] = 2;
                tx[row] = ubc[0];
                ty[row] = ubc[1];
                tz[row] = ubc[2];
            } else {
                // Wall wins at geometric rims.
                // A boundary row touching the wall is no-slip even if it also lies
                // on an inlet/outlet cap edge.
                if (outletTrace[row]) ++skippedWallOutletRows;
                if (inletTrace[row])  ++skippedWallInletRows;

                kind[row] = 1;
                tx[row] = 0.0;
                ty[row] = 0.0;
                tz[row] = 0.0;
            }
        }
    }

    static int printedPriorityDiagnostic = 0;
    if (!printedPriorityDiagnostic && (skippedWallOutletRows > 0 || skippedWallInletRows > 0)) {
        std::printf("channelClampVelocityBC priority: wall wins at rim trace assignments "
                    "for outletRows=%d inletRows=%d; cap-rim rows are clamped as wall.\n",
                    skippedWallOutletRows, skippedWallInletRows);
        printedPriorityDiagnostic = 1;
    }

    int count = 0;
    for (int i=0; i<nU; ++i) {
        if (!kind[i]) continue;
        ux[i] = tx[i];
        uy[i] = ty[i];
        uz[i] = tz[i];
        ++count;
    }

    return count;
}




static int build_channel_velocity_dirichlet_trace_values(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    std::vector<unsigned char>& isDirichlet,
    std::vector<double>& tx,
    std::vector<double>& ty,
    std::vector<double>& tz,
    int* inletRowsOut,
    int* wallRowsOut,
    int* skippedWallOutletRowsOut,
    int* skippedWallInletRowsOut)
{
    const int nU = 10 * mesh.nCells;
    isDirichlet.assign(nU, 0);
    tx.assign(nU, 0.0);
    ty.assign(nU, 0.0);
    tz.assign(nU, 0.0);

    std::vector<unsigned char> inletTrace(nU, 0);
    std::vector<unsigned char> outletTrace(nU, 0);
    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    auto mark_face_rows = [&](int f, std::vector<unsigned char>& mark) {
        const int c = mesh.owner[f];
        const auto& K = tets[c];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;
            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (onFace) {
                mark[10*c + i] = 1;
            }
        }
    };

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind == 1) {
            mark_face_rows(f, inletTrace);
        } else if (bkind == 2) {
            mark_face_rows(f, outletTrace);
        }
    }

    int skippedWallOutletRows = 0;
    int skippedWallInletRows = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != 0 && bkind != 1) continue;

        const int c = mesh.owner[f];
        const auto& K = tets[c];
        const auto n = mesh.nf[f];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;
            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }
            if (!onFace) continue;

            const int row = 10*c + i;

            if (bkind == 1) {
                const auto ubc = channel_velocity_bc(K.xP2[i], n);
                isDirichlet[row] = 1;
                tx[row] = ubc[0];
                ty[row] = ubc[1];
                tz[row] = ubc[2];
            } else {
                // Wall wins at geometric rims.
                // A DG2 trace row on both wall and inlet/outlet is physically a wall row:
                // no-slip/no-penetration must override both inlet profile and outlet freedom.
                if (outletTrace[row]) ++skippedWallOutletRows;
                if (inletTrace[row])  ++skippedWallInletRows;

                isDirichlet[row] = 1;
                tx[row] = 0.0;
                ty[row] = 0.0;
                tz[row] = 0.0;
            }
        }
    }

    int inletRows = 0;
    int wallRows = 0;
    int totalRows = 0;
    for (int i=0; i<nU; ++i) {
        if (!isDirichlet[i]) continue;
        ++totalRows;
        if (std::abs(tx[i]) + std::abs(ty[i]) + std::abs(tz[i]) > 0.0) ++inletRows;
        else ++wallRows;
    }

    if (inletRowsOut) *inletRowsOut = inletRows;
    if (wallRowsOut) *wallRowsOut = wallRows;
    if (skippedWallOutletRowsOut) *skippedWallOutletRowsOut = skippedWallOutletRows;
    if (skippedWallInletRowsOut) *skippedWallInletRowsOut = skippedWallInletRows;

    return totalRows;
}

static int impose_channel_velocity_mask_values(
    const std::vector<unsigned char>& isDirichlet,
    const std::vector<double>& target,
    std::vector<double>& u)
{
    int n = 0;
    const int N = std::min<int>((int)u.size(), (int)isDirichlet.size());
    for (int i=0; i<N; ++i) {
        if (!isDirichlet[i]) continue;
        u[i] = target[i];
        ++n;
    }
    return n;
}

static int apply_channel_strong_velocity_matrix(
    const CSRPattern& pat,
    std::vector<double>& A,
    const std::vector<unsigned char>& isDirichlet)
{
    if ((int)isDirichlet.size() != pat.nRows) {
        throw std::runtime_error("apply_channel_strong_velocity_matrix: mask size mismatch");
    }
    if ((int)A.size() != pat.nnz) {
        throw std::runtime_error("apply_channel_strong_velocity_matrix: matrix size mismatch");
    }

    int nRows = 0;

    for (int r=0; r<pat.nRows; ++r) {
        const bool rowLocked = isDirichlet[r] != 0;
        if (rowLocked) ++nRows;

        for (int k=pat.rowOffsets[r]; k<pat.rowOffsets[r+1]; ++k) {
            const int c = (int)pat.cols[k];
            const bool colLocked = (c >= 0 && c < pat.nRows && isDirichlet[c] != 0);
            if (rowLocked || colLocked) {
                A[k] = 0.0;
            }
        }
    }

    for (int r=0; r<pat.nRows; ++r) {
        if (!isDirichlet[r]) continue;
        if (pat.diagPos[r] < 0) {
            throw std::runtime_error("apply_channel_strong_velocity_matrix: missing diagonal");
        }
        A[pat.diagPos[r]] = 1.0;
    }

    return nRows;
}

static int apply_channel_strong_velocity_rhs(
    const CSRPattern& pat,
    const std::vector<double>& Aoriginal,
    std::vector<double>& rhs,
    const std::vector<unsigned char>& isDirichlet,
    const std::vector<double>& target)
{
    if ((int)isDirichlet.size() != pat.nRows ||
        (int)target.size() != pat.nRows ||
        (int)rhs.size() != pat.nRows) {
        throw std::runtime_error("apply_channel_strong_velocity_rhs: size mismatch");
    }
    if ((int)Aoriginal.size() != pat.nnz) {
        throw std::runtime_error("apply_channel_strong_velocity_rhs: matrix size mismatch");
    }

    for (int r=0; r<pat.nRows; ++r) {
        if (isDirichlet[r]) continue;

        for (int k=pat.rowOffsets[r]; k<pat.rowOffsets[r+1]; ++k) {
            const int c = (int)pat.cols[k];
            if (c >= 0 && c < pat.nRows && isDirichlet[c]) {
                rhs[r] -= Aoriginal[k] * target[c];
            }
        }
    }

    int nRows = 0;
    for (int r=0; r<pat.nRows; ++r) {
        if (!isDirichlet[r]) continue;
        rhs[r] = target[r];
        ++nRows;
    }

    return nRows;
}

static int project_channel_dirichlet_normal_correction_trace(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<double>& uxBefore,
    const std::vector<double>& uyBefore,
    const std::vector<double>& uzBefore,
    std::vector<double>& ux,
    std::vector<double>& uy,
    std::vector<double>& uz,
    double* sumAbsDnOut,
    double* maxAbsDnOut)
{
    if (!gCaseIsChannel) {
        if (sumAbsDnOut) *sumAbsDnOut = 0.0;
        if (maxAbsDnOut) *maxAbsDnOut = 0.0;
        return 0;
    }

    const int nU = 10 * mesh.nCells;
    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    int nProjected = 0;
    double sumAbsDn = 0.0;
    double maxAbsDn = 0.0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);

        // Only velocity-Dirichlet boundaries:
        //   bkind=1 inlet
        //   bkind=0 wall
        // Outlet bkind=2 is intentionally untouched.
        if (bkind != 0 && bkind != 1) continue;

        const int c = mesh.owner[f];
        const auto& K = tets[c];
        const auto n = mesh.nf[f];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<3; ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (!onFace) continue;

            const int row = 10*c + i;
            if (row < 0 || row >= nU) continue;

            // Pressure-correction increment only:
            //   deltaU = U_after_pressure_correction - U_predictor
            const double dux = ux[row] - uxBefore[row];
            const double duy = uy[row] - uyBefore[row];
            const double duz = uz[row] - uzBefore[row];

            const double dn = dux*n[0] + duy*n[1] + duz*n[2];

            // Remove only normal part of correction increment.
            // Tangential correction, if any, is left untouched.
            ux[row] -= dn * n[0];
            uy[row] -= dn * n[1];
            uz[row] -= dn * n[2];

            const double adn = std::abs(dn);
            sumAbsDn += adn;
            maxAbsDn = std::max(maxAbsDn, adn);
            ++nProjected;
        }
    }

    if (sumAbsDnOut) *sumAbsDnOut = sumAbsDn;
    if (maxAbsDnOut) *maxAbsDnOut = maxAbsDn;
    return nProjected;
}



static void print_channel_outlet_pressure_correction_diagnostic(
    int it,
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& pcorr,
    const std::vector<double>& uxBefore,
    const std::vector<double>& uyBefore,
    const std::vector<double>& uzBefore,
    const std::vector<double>& uxAfter,
    const std::vector<double>& uyAfter,
    const std::vector<double>& uzAfter)
{
    if (!gCaseIsChannel) return;

    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    int nOutletFaces = 0;
    int nPrinted = 0;
    int mapFail = 0;

    double areaTot = 0.0;
    double gradNAreaSum = 0.0;
    double gradNMin =  std::numeric_limits<double>::infinity();
    double gradNMax = -std::numeric_limits<double>::infinity();

    double fluxBeforeTot = 0.0;
    double fluxAfterTot  = 0.0;

    double traceDunSum = 0.0;
    double traceDunAbsSum = 0.0;
    double traceDunMin =  std::numeric_limits<double>::infinity();
    double traceDunMax = -std::numeric_limits<double>::infinity();
    int traceDunCount = 0;

    double pcVertMin =  std::numeric_limits<double>::infinity();
    double pcVertMax = -std::numeric_limits<double>::infinity();
    double pcVertSum = 0.0;
    int pcVertCount = 0;

    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);
        if (bkind != 2) continue;

        ++nOutletFaces;

        const int c = mesh.owner[f];
        if (c < 0 || c >= (int)tets.size()) continue;

        const auto& K = tets[c];
        const auto n = mesh.nf[f];
        const double area = mesh.Af[f];

        double pv[4]{};
        std::array<double,3> gradPc{0.0,0.0,0.0};

        for (int a=0; a<4; ++a) {
            const int gid = K.v[a];
            pv[a] = (gid >= 0 && gid < (int)pcorr.size()) ? pcorr[gid] : 0.0;
            gradPc[0] += pv[a] * K.gradLam[a][0];
            gradPc[1] += pv[a] * K.gradLam[a][1];
            gradPc[2] += pv[a] * K.gradLam[a][2];

            pcVertMin = std::min(pcVertMin, pv[a]);
            pcVertMax = std::max(pcVertMax, pv[a]);
            pcVertSum += pv[a];
            ++pcVertCount;
        }

        const double gradN = gradPc[0]*n[0] + gradPc[1]*n[1] + gradPc[2]*n[2];

        areaTot += area;
        gradNAreaSum += area * gradN;
        gradNMin = std::min(gradNMin, gradN);
        gradNMax = std::max(gradNMax, gradN);

        double faceFluxBefore = 0.0;
        double faceFluxAfter = 0.0;

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                ++mapFail;
                continue;
            }

            double N[10];
            p2_tet_basis(lam, N);

            double ub0=0.0, ub1=0.0, ub2=0.0;
            double ua0=0.0, ua1=0.0, ua2=0.0;

            for (int j=0; j<10; ++j) {
                const int row = 10*c + j;
                ub0 += uxBefore[row] * N[j];
                ub1 += uyBefore[row] * N[j];
                ub2 += uzBefore[row] * N[j];

                ua0 += uxAfter[row] * N[j];
                ua1 += uyAfter[row] * N[j];
                ua2 += uzAfter[row] * N[j];
            }

            const double ww = area * q.w;
            faceFluxBefore += ww * (ub0*n[0] + ub1*n[1] + ub2*n[2]);
            faceFluxAfter  += ww * (ua0*n[0] + ua1*n[1] + ua2*n[2]);
        }

        fluxBeforeTot += faceFluxBefore;
        fluxAfterTot  += faceFluxAfter;

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) faceHasLocalVertex[a] = true;
            }
        }

        double faceDun[10]{};
        int faceDunN = 0;
        double faceDunSum = 0.0;
        double faceDunMin =  std::numeric_limits<double>::infinity();
        double faceDunMax = -std::numeric_limits<double>::infinity();

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (!onFace) continue;

            const int row = 10*c + i;

            const double dux = uxAfter[row] - uxBefore[row];
            const double duy = uyAfter[row] - uyBefore[row];
            const double duz = uzAfter[row] - uzBefore[row];

            const double dun = dux*n[0] + duy*n[1] + duz*n[2];

            faceDun[faceDunN++] = dun;
            faceDunSum += dun;
            faceDunMin = std::min(faceDunMin, dun);
            faceDunMax = std::max(faceDunMax, dun);

            traceDunSum += dun;
            traceDunAbsSum += std::abs(dun);
            traceDunMin = std::min(traceDunMin, dun);
            traceDunMax = std::max(traceDunMax, dun);
            ++traceDunCount;
        }

        if (nPrinted < 8) {
            std::printf(
                "    outletCorrFaceDiag: it=%d face=%d owner=%d area=%.6e "
                "n=(% .6e,% .6e,% .6e) "
                "pcorrVerts=(% .6e,% .6e,% .6e,% .6e) "
                "gradPc=(% .6e,% .6e,% .6e) gradPcDotN=% .6e "
                "fluxBefore=% .6e fluxAfter=% .6e dFlux=% .6e "
                "traceDun(avg,min,max)=",
                it, f, c, area,
                n[0], n[1], n[2],
                pv[0], pv[1], pv[2], pv[3],
                gradPc[0], gradPc[1], gradPc[2], gradN,
                faceFluxBefore, faceFluxAfter, faceFluxAfter - faceFluxBefore);

            if (faceDunN > 0) {
                std::printf("(% .6e,% .6e,% .6e) vals=(",
                    faceDunSum / (double)faceDunN, faceDunMin, faceDunMax);
                for (int k=0; k<faceDunN; ++k) {
                    std::printf("%s% .6e", k ? "," : "", faceDun[k]);
                }
                std::printf(")\n");
            } else {
                std::printf("(nan,nan,nan) vals=()\n");
            }

            ++nPrinted;
        }
    }

    if (nOutletFaces == 0) {
        std::printf("    outletCorrSummary: it=%d outletFaces=0 -- no outlet faces found.\n", it);
        return;
    }

    const double invArea = areaTot > 1e-300 ? 1.0 / areaTot : 0.0;
    const double invTrace = traceDunCount > 0 ? 1.0 / (double)traceDunCount : 0.0;
    const double invPc = pcVertCount > 0 ? 1.0 / (double)pcVertCount : 0.0;

    std::printf(
        "    outletCorrSummary: it=%d faces=%d area=%.6e "
        "pcorrVert(avg,min,max)=(% .6e,% .6e,% .6e) "
        "gradPcDotN(areaAvg,min,max)=(% .6e,% .6e,% .6e) "
        "outletFlux(before,after,diff)=(% .6e,% .6e,% .6e) "
        "traceDun(avg,min,max,absAvg)=(% .6e,% .6e,% .6e,% .6e) mapFail=%d\n",
        it, nOutletFaces, areaTot,
        pcVertSum*invPc, pcVertMin, pcVertMax,
        gradNAreaSum*invArea, gradNMin, gradNMax,
        fluxBeforeTot, fluxAfterTot, fluxAfterTot - fluxBeforeTot,
        traceDunSum*invTrace, traceDunMin, traceDunMax, traceDunAbsSum*invTrace,
        mapFail);
}




static void zero_channel_dirichlet_trace_rows_in_rectcsr(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    RectCSR& Ap,
    const char* name)
{
    if (!gCaseIsChannel) return;
    if (Ap.nRows <= 0) return;

    std::vector<unsigned char> inletTrace(Ap.nRows, 0);
    std::vector<unsigned char> outletTrace(Ap.nRows, 0);
    std::vector<unsigned char> wallTrace(Ap.nRows, 0);
    std::vector<unsigned char> rowMask(Ap.nRows, 0);

    const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    auto mark_face_rows = [&](int f, std::vector<unsigned char>& mark) {
        const int c = mesh.owner[f];
        const auto& K = tets[c];

        bool faceHasLocalVertex[4] = {false,false,false,false};

        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<(int)mesh.faces[f].size(); ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) {
                    faceHasLocalVertex[a] = true;
                }
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;

            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }

            if (onFace) {
                const int row = 10*c + i;
                if (row >= 0 && row < Ap.nRows) {
                    mark[row] = 1;
                }
            }
        }
    };

    int inletFaces = 0;
    int outletFaces = 0;
    int wallFaces = 0;

    // First pass: collect patch trace sets globally.
    // This is essential for cap rims, where the same local P2 row may be seen
    // both by an outlet cap face and a wall face.
    for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
        const int bkind = channel_boundary_kind(mesh, f);

        if (bkind == 1) {
            ++inletFaces;
            mark_face_rows(f, inletTrace);
        } else if (bkind == 2) {
            ++outletFaces;
            mark_face_rows(f, outletTrace);
        } else if (bkind == 0) {
            ++wallFaces;
            mark_face_rows(f, wallTrace);
        }
    }

    int inletRowsMasked = 0;
    int wallRowsMasked = 0;
    int protectedOutletRows = 0;
    int protectedInletWallRows = 0;

    // Wall wins at patch rims.
    // If a DG2 trace row touches a wall face, it is a no-slip correction row,
    // even if it also touches inlet/outlet cap faces.
    for (int r=0; r<Ap.nRows; ++r) {
        if (!wallTrace[r]) continue;

        if (outletTrace[r]) ++protectedOutletRows;
        if (inletTrace[r])  ++protectedInletWallRows;

        if (!rowMask[r]) {
            rowMask[r] = 1;
            ++wallRowsMasked;
        }
    }

    // Inlet rows are Dirichlet velocity rows, so they are masked from the Schur correction.
    // Outlet-only rows remain free.
    for (int r=0; r<Ap.nRows; ++r) {
        if (!inletTrace[r]) continue;
        if (rowMask[r]) continue;

        if (outletTrace[r]) {
            ++protectedOutletRows;
            continue;
        }

        rowMask[r] = 1;
        ++inletRowsMasked;
    }

    int maskedRows = 0;
    int zeroedValues = 0;

    for (int r=0; r<Ap.nRows; ++r) {
        if (!rowMask[r]) continue;

        ++maskedRows;

        const int b0 = Ap.rowOffsets[r];
        const int b1 = Ap.rowOffsets[r+1];

        for (int jj=b0; jj<b1; ++jj) {
            if (Ap.values[jj] != 0.0) {
                Ap.values[jj] = 0.0;
                ++zeroedValues;
            }
        }
    }

    std::printf("channelMaskDirichletSchurRows priority: Ap=%s "
                "faces(in,out,wall)=(%d,%d,%d) "
                "maskedRows=%d inletRowsMasked=%d wallRowsMasked=%d "
                "protectedOutletRows=%d protectedInletWallRows=%d zeroedValues=%d; "
                "wall trace rows win at inlet/outlet rims; outlet-only trace rows stay free.\n",
                name,
                inletFaces, outletFaces, wallFaces,
                maskedRows, inletRowsMasked, wallRowsMasked,
                protectedOutletRows, protectedInletWallRows, zeroedValues);
}


static void rebuild_physical_operator_from_beta(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const CSRPattern& pat,
    const std::vector<QuadTetPoint>& tq,
    const std::vector<QuadTriPoint>& fq,
    const std::vector<double>& Kdiff,
    const std::vector<double>& betaX,
    const std::vector<double>& betaY,
    const std::vector<double>& betaZ,
    std::vector<double>& Aphys,
    int* convNnzOut,
    const ConvLFPatternCache* convCache = nullptr)
{
    Aphys = Kdiff;

    if (convNnzOut) {
        *convNnzOut = 0;
    }

    if (!gFlowIsNse) {
        return;
    }

    std::vector<double> Cbeta;
    assemble_dg2_convection_lf_operator_from_dg2_velocity(mesh, tets, pat, tq, fq, betaX, betaY, betaZ, Cbeta, convCache);

    if (convNnzOut) {
        *convNnzOut = count_effective_nnz(Cbeta);
    }

    add_scaled_values(Aphys, Cbeta, 1.0);
}



static void left_scale_csr_rows(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::string& mode,
    double eps,
    std::vector<double>& As,
    std::vector<double>& bs)
{
    As = A;
    bs = rhs;

    if (mode == "none") {
        return;
    }

    if (mode != "l1row" && mode != "lumpeddiag" && mode != "rowsuml1" && mode != "diagabs") {
        throw std::runtime_error("Unknown -uLeftScale mode. Use none, l1row/lumpeddiag/rowsuml1, or diagabs.");
    }

    for (int i=0; i<pat.nRows; ++i) {
        double d = 0.0;

        if (mode == "diagabs") {
            for (int k=pat.rowOffsets[i]; k<pat.rowOffsets[i+1]; ++k) {
                if (pat.cols[k] == i) {
                    d = std::abs(A[k]);
                    break;
                }
            }
        } else {
            for (int k=pat.rowOffsets[i]; k<pat.rowOffsets[i+1]; ++k) {
                d += std::abs(A[k]);
            }
        }

        const double invD = 1.0 / std::max(d, eps);

        for (int k=pat.rowOffsets[i]; k<pat.rowOffsets[i+1]; ++k) {
            As[k] *= invD;
        }

        bs[i] *= invD;
    }
}


static bool invert_dense10_block_jacobi(
    double a[10][10],
    double inv[10][10],
    double pivotFloor)
{
    double aug[10][20]{};

    for (int i=0; i<10; ++i) {
        for (int j=0; j<10; ++j) {
            aug[i][j] = a[i][j];
        }
        aug[i][10+i] = 1.0;
    }

    for (int k=0; k<10; ++k) {
        int piv = k;
        double best = std::abs(aug[k][k]);

        for (int r=k+1; r<10; ++r) {
            const double v = std::abs(aug[r][k]);
            if (v > best) {
                best = v;
                piv = r;
            }
        }

        if (best < pivotFloor) {
            return false;
        }

        if (piv != k) {
            for (int j=0; j<20; ++j) {
                std::swap(aug[k][j], aug[piv][j]);
            }
        }

        const double invPivot = 1.0 / aug[k][k];

        for (int j=0; j<20; ++j) {
            aug[k][j] *= invPivot;
        }

        for (int r=0; r<10; ++r) {
            if (r == k) continue;

            const double f = aug[r][k];

            if (std::abs(f) == 0.0) continue;

            for (int j=0; j<20; ++j) {
                aug[r][j] -= f * aug[k][j];
            }
        }
    }

    for (int i=0; i<10; ++i) {
        for (int j=0; j<10; ++j) {
            inv[i][j] = aug[i][10+j];
        }
    }

    return true;
}

static double get_csr_value_or_zero(
    const CSRPattern& pat,
    const std::vector<double>& A,
    int row,
    int col)
{
    const int b = pat.rowOffsets[row];
    const int e = pat.rowOffsets[row+1];

    auto it = std::lower_bound(
        pat.cols.begin()+b,
        pat.cols.begin()+e,
        static_cast<HYPRE_BigInt>(col));

    if (it == pat.cols.begin()+e || *it != static_cast<HYPRE_BigInt>(col)) {
        return 0.0;
    }

    return A[(int)(it - pat.cols.begin())];
}

static void build_cellblock_jacobi_inverses_from_csr(
    const CSRPattern& pat,
    const std::vector<double>& A,
    double diagShift,
    double pivotFloor,
    std::vector<std::array<std::array<double,10>,10>>& invBlocks)
{
    if (pat.nRows % 10 != 0) {
        throw std::runtime_error("cellblock Jacobi requires rows divisible by 10");
    }

    const int nCells = pat.nRows / 10;
    invBlocks.resize(nCells);

    int failed = 0;

    for (int c=0; c<nCells; ++c) {
        double blk[10][10]{};
        double inv[10][10]{};

        for (int i=0; i<10; ++i) {
            const int row = 10*c + i;

            for (int j=0; j<10; ++j) {
                const int col = 10*c + j;
                blk[i][j] = get_csr_value_or_zero(pat, A, row, col);
            }
        }

        if (diagShift != 0.0) {
            double avgDiag = 0.0;

            for (int i=0; i<10; ++i) {
                avgDiag += std::abs(blk[i][i]);
            }

            avgDiag /= 10.0;

            const double shift = diagShift * std::max(avgDiag, 1e-300);

            for (int i=0; i<10; ++i) {
                blk[i][i] += shift;
            }
        }

        bool ok = invert_dense10_block_jacobi(blk, inv, pivotFloor);

        if (!ok) {
            failed++;

            // Robust fallback: scalar diagonal inverse for this one bad block.
            for (int i=0; i<10; ++i) {
                for (int j=0; j<10; ++j) {
                    inv[i][j] = 0.0;
                }

                const double d = std::abs(blk[i][i]) > pivotFloor ? blk[i][i] : (blk[i][i] >= 0.0 ? pivotFloor : -pivotFloor);
                inv[i][i] = 1.0 / d;
            }
        }

        for (int i=0; i<10; ++i) {
            for (int j=0; j<10; ++j) {
                invBlocks[c][i][j] = inv[i][j];
            }
        }
    }

    if (failed) {
        std::printf("WARNING: cellblock Jacobi fallback diagonal blocks = %d / %d\n", failed, nCells);
    }
}

static void left_precondition_csr_cellblock_same_pattern(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    double diagShift,
    double pivotFloor,
    std::vector<double>& As,
    std::vector<double>& bs)
{
    const int nCells = pat.nRows / 10;

    std::vector<std::array<std::array<double,10>,10>> invBlocks;
    build_cellblock_jacobi_inverses_from_csr(pat, A, diagShift, pivotFloor, invBlocks);

    As.assign(A.size(), 0.0);
    bs.assign(rhs.size(), 0.0);

    for (int c=0; c<nCells; ++c) {
        for (int i=0; i<10; ++i) {
            const int targetRow = 10*c + i;

            for (int r=0; r<10; ++r) {
                const int sourceRow = 10*c + r;
                const double coeff = invBlocks[c][i][r];

                if (std::abs(coeff) <= 0.0) continue;

                bs[targetRow] += coeff * rhs[sourceRow];

                for (int k=pat.rowOffsets[sourceRow]; k<pat.rowOffsets[sourceRow+1]; ++k) {
                    const int col = (int)pat.cols[k];

                    // This should exist because all 10 rows of a DG2 cell share the same coupled-cell stencil.
                    const int q = find_col_pos(pat, targetRow, col);

                    As[q] += coeff * A[k];
                }
            }
        }
    }
}



struct CpuMcgsColoring {
    int nRows = 0;
    int nColors = 0;
    std::vector<int> colorOffsets;
    std::vector<int> colorRows;
};

static double mcgs_dot(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) throw std::runtime_error("mcgs_dot size mismatch");

    double s = 0.0;

    for (std::size_t i=0; i<a.size(); ++i) {
        s += a[i]*b[i];
    }

    return s;
}

static CpuMcgsColoring build_cpu_mcgs_coloring(const CSRPattern& pat)
{
    const int n = pat.nRows;

    std::vector<int> color(n, -1);

    int nColors = 0;
    int maxDegree = 0;

    for (int i=0; i<n; ++i) {
        maxDegree = std::max(maxDegree, pat.rowOffsets[i+1] - pat.rowOffsets[i]);

        std::vector<unsigned char> used(std::max(1, nColors), 0);

        for (int pp=pat.rowOffsets[i]; pp<pat.rowOffsets[i+1]; ++pp) {
            const int j = (int)pat.cols[pp];

            if (j >= 0 && j < n && color[j] >= 0 && color[j] < nColors) {
                used[color[j]] = 1;
            }
        }

        int c = 0;

        while (c < nColors && used[c]) {
            ++c;
        }

        if (c == nColors) {
            ++nColors;
        }

        color[i] = c;
    }

    long long conflicts = 0;

    for (int i=0; i<n; ++i) {
        for (int pp=pat.rowOffsets[i]; pp<pat.rowOffsets[i+1]; ++pp) {
            const int j = (int)pat.cols[pp];

            if (j != i && j >= 0 && j < n && color[i] == color[j]) {
                ++conflicts;
            }
        }
    }

    if (conflicts != 0) {
        std::fprintf(stderr, "ERROR: CPU MCGS coloring has %lld same-color adjacency conflicts.\n", conflicts);
        throw std::runtime_error("CPU MCGS coloring conflict");
    }

    std::vector<int> counts(nColors, 0);

    for (int i=0; i<n; ++i) {
        counts[color[i]]++;
    }

    CpuMcgsColoring cg;
    cg.nRows = n;
    cg.nColors = nColors;
    cg.colorOffsets.assign(nColors + 1, 0);

    for (int c=0; c<nColors; ++c) {
        cg.colorOffsets[c+1] = cg.colorOffsets[c] + counts[c];
    }

    cg.colorRows.assign(n, 0);

    std::vector<int> cursor = cg.colorOffsets;

    for (int i=0; i<n; ++i) {
        const int c = color[i];
        cg.colorRows[cursor[c]++] = i;
    }

    int minRows = n;
    int maxRows = 0;

    for (int c=0; c<nColors; ++c) {
        const int cnt = cg.colorOffsets[c+1] - cg.colorOffsets[c];
        minRows = std::min(minRows, cnt);
        maxRows = std::max(maxRows, cnt);
    }

    std::printf("CPU_MCGS_COLORING nRows=%d nColors=%d minRows/color=%d maxRows/color=%d maxDegree=%d\n",
                n, nColors, minRows, maxRows, maxDegree);

    return cg;
}


static CpuMcgsColoring build_cellblock_coloring_from_csr(const CSRPattern& pat)
{
    if (pat.nRows % 10 != 0) {
        throw std::runtime_error("build_cellblock_coloring_from_csr: DG2 scalar row count is not divisible by 10");
    }

    const int nCells = pat.nRows / 10;

    std::vector<std::vector<int>> adj(nCells);
    int maxOneWayStencil = 0;

    for (int c=0; c<nCells; ++c) {
        std::vector<int> nb;
        nb.reserve(64);

        for (int ii=0; ii<10; ++ii) {
            const int row = 10*c + ii;
            for (int p=pat.rowOffsets[row]; p<pat.rowOffsets[row+1]; ++p) {
                const int col = (int)pat.cols[p];
                if (col < 0 || col >= pat.nRows) continue;

                const int cc = col / 10;
                if (cc != c) nb.push_back(cc);
            }
        }

        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());

        maxOneWayStencil = std::max(maxOneWayStencil, (int)nb.size());
        adj[c].swap(nb);
    }

    // Make graph explicitly undirected.  The momentum operator can be
    // nonsymmetric, but coloring must protect both A_ij and A_ji couplings.
    for (int c=0; c<nCells; ++c) {
        for (int j : adj[c]) {
            if (j >= 0 && j < nCells && j != c) adj[j].push_back(c);
        }
    }

    int maxDegree = 0;
    for (int c=0; c<nCells; ++c) {
        auto& v = adj[c];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        maxDegree = std::max(maxDegree, (int)v.size());
    }

    std::vector<int> color(nCells, -1);
    int nColors = 0;

    for (int c=0; c<nCells; ++c) {
        std::vector<unsigned char> used(std::max(1, nColors), 0);

        for (int j : adj[c]) {
            if (j >= 0 && j < nCells && color[j] >= 0 && color[j] < nColors) {
                used[color[j]] = 1;
            }
        }

        int k = 0;
        while (k < nColors && used[k]) ++k;

        if (k == nColors) ++nColors;
        color[c] = k;
    }

    long long conflicts = 0;
    for (int c=0; c<nCells; ++c) {
        for (int j : adj[c]) {
            if (j != c && j >= 0 && j < nCells && color[j] == color[c]) {
                ++conflicts;
            }
        }
    }

    if (conflicts != 0) {
        std::fprintf(stderr, "ERROR: GPU colored cell-BGS coloring has %lld same-color cell conflicts.\n", conflicts);
        throw std::runtime_error("GPU colored cell-BGS coloring conflict");
    }

    std::vector<int> counts(nColors, 0);
    for (int c=0; c<nCells; ++c) counts[color[c]]++;

    CpuMcgsColoring cg;
    cg.nRows = nCells;      // here nRows means cells
    cg.nColors = nColors;
    cg.colorOffsets.assign(nColors + 1, 0);

    for (int k=0; k<nColors; ++k) {
        cg.colorOffsets[k+1] = cg.colorOffsets[k] + counts[k];
    }

    cg.colorRows.assign(nCells, 0); // here colorRows stores cell ids

    std::vector<int> cursor = cg.colorOffsets;
    for (int c=0; c<nCells; ++c) {
        cg.colorRows[cursor[color[c]]++] = c;
    }

    int minCells = nCells;
    int maxCells = 0;

    for (int k=0; k<nColors; ++k) {
        const int cnt = cg.colorOffsets[k+1] - cg.colorOffsets[k];
        minCells = std::min(minCells, cnt);
        maxCells = std::max(maxCells, cnt);
    }

    std::printf("GPU_CELLBLOCK_COLORING nCells=%d nColors=%d minCells/color=%d maxCells/color=%d maxDegree=%d maxOneWayStencil=%d\n",
                nCells, nColors, minCells, maxCells, maxDegree, maxOneWayStencil);

    return cg;
}

static std::vector<double> extract_diag_mcgs(const CSRPattern& pat, const std::vector<double>& A)
{
    std::vector<double> diag(pat.nRows, 0.0);

    for (int i=0; i<pat.nRows; ++i) {
        for (int pp=pat.rowOffsets[i]; pp<pat.rowOffsets[i+1]; ++pp) {
            if ((int)pat.cols[pp] == i) {
                diag[i] = A[pp];
                break;
            }
        }
    }

    return diag;
}

static void mcgs_sweeps_in_place(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const CpuMcgsColoring& coloring,
    const std::vector<double>& diag,
    int sweeps,
    double omega,
    int symmetric)
{
    auto one_color_pass = [&](int c) {
        const int b = coloring.colorOffsets[c];
        const int e = coloring.colorOffsets[c+1];

        for (int kk=b; kk<e; ++kk) {
            const int i = coloring.colorRows[kk];

            double off = 0.0;
            double aii = diag[i];

            for (int pp=pat.rowOffsets[i]; pp<pat.rowOffsets[i+1]; ++pp) {
                const int j = (int)pat.cols[pp];
                const double aij = A[pp];

                if (j == i) {
                    aii = aij;
                } else {
                    off += aij * x[j];
                }
            }

            if (std::abs(aii) > 1e-300) {
                const double xgs = (rhs[i] - off) / aii;
                x[i] = (1.0 - omega)*x[i] + omega*xgs;
            }
        }
    };

    for (int sw=0; sw<sweeps; ++sw) {
        for (int c=0; c<coloring.nColors; ++c) {
            one_color_pass(c);
        }

        if (symmetric) {
            for (int c=coloring.nColors-1; c>=0; --c) {
                one_color_pass(c);
            }
        }
    }
}

static void apply_mcgs_preconditioner_zero_start(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& z,
    const CpuMcgsColoring& coloring,
    const std::vector<double>& diag,
    int sweeps,
    double omega,
    int symmetric)
{
    z.assign(rhs.size(), 0.0);
    mcgs_sweeps_in_place(pat, A, rhs, z, coloring, diag, sweeps, omega, symmetric);
}

static HypreSolveInfo solve_momentum_component_mcgs_cpu(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const CpuMcgsColoring& coloring,
    int sweeps,
    double omega,
    int symmetric,
    double relTol,
    double absTol,
    int monitor)
{
    const int n = pat.nRows;

    if ((int)x.size() != n) {
        x.assign(n, 0.0);
    }

    const std::vector<double> diag = extract_diag_mcgs(pat, A);
    const double rhsNorm = std::max(norm_vec(rhs), 1e-300);

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = 1.0;

    std::vector<double> Ax;
    std::vector<double> r(n, 0.0);

    if (monitor) {
        std::printf("CPU MCGS momentum: sweeps=%d omega=%.6g symmetric=%d relTol=%.3e absTol=%.3e\n",
                    sweeps, omega, symmetric, relTol, absTol);
    }

    for (int sw=0; sw<sweeps; ++sw) {
        mcgs_sweeps_in_place(pat, A, rhs, x, coloring, diag, 1, omega, symmetric);

        info.iterations = sw + 1;

        if (monitor || relTol > 0.0 || absTol > 0.0) {
            apply_csr(pat, A, x, Ax);

            for (int i=0; i<n; ++i) {
                r[i] = rhs[i] - Ax[i];
            }

            const double absRes = norm_vec(r);
            const double relRes = absRes / rhsNorm;

            info.finalRelResNorm = relRes;

            if (monitor && ((sw + 1) <= 10 || ((sw + 1) % 10) == 0)) {
                std::printf("CPU MCGS sweep %6d absRes %.17e relRes %.17e\n",
                            sw + 1, absRes, relRes);
            }

            if ((relTol > 0.0 && relRes <= relTol) || (absTol > 0.0 && absRes <= absTol)) {
                break;
            }
        }
    }

    if (!(monitor || relTol > 0.0 || absTol > 0.0)) {
        apply_csr(pat, A, x, Ax);

        for (int i=0; i<n; ++i) {
            r[i] = rhs[i] - Ax[i];
        }

        info.finalRelResNorm = norm_vec(r) / rhsNorm;
    }

    return info;
}

static HypreSolveInfo solve_momentum_component_bicgstab_mcgs_left_cpu(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const CpuMcgsColoring& coloring,
    int maxit,
    double relTol,
    double absTol,
    int preSweeps,
    double omega,
    int symmetric,
    int monitor)
{
    const int n = pat.nRows;

    if ((int)x.size() != n) {
        x.assign(n, 0.0);
    }

    const std::vector<double> diag = extract_diag_mcgs(pat, A);

    auto apply_Ahat = [&](const std::vector<double>& v, std::vector<double>& y) {
        std::vector<double> Av;
        apply_csr(pat, A, v, Av);
        apply_mcgs_preconditioner_zero_start(pat, A, Av, y, coloring, diag, preSweeps, omega, symmetric);
    };

    std::vector<double> bhat;
    apply_mcgs_preconditioner_zero_start(pat, A, rhs, bhat, coloring, diag, preSweeps, omega, symmetric);

    const double trueRhsNorm = std::max(norm_vec(rhs), 1e-300);
    const double bhatNorm = std::max(norm_vec(bhat), 1e-300);

    std::vector<double> Ax;
    apply_csr(pat, A, x, Ax);

    std::vector<double> rTrue(n, 0.0);

    for (int i=0; i<n; ++i) {
        rTrue[i] = rhs[i] - Ax[i];
    }

    std::vector<double> r;
    apply_mcgs_preconditioner_zero_start(pat, A, rTrue, r, coloring, diag, preSweeps, omega, symmetric);

    std::vector<double> rhat = r;
    std::vector<double> p(n, 0.0), v(n, 0.0), s(n, 0.0), t(n, 0.0);

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omegaB = 1.0;

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = norm_vec(rTrue) / trueRhsNorm;

    if (monitor) {
        std::printf("CPU BiCGSTAB+MCGS-left: maxit=%d relTol=%.3e absTol=%.3e preSweeps=%d omega=%.6g symmetric=%d trueRel0=%.17e\n",
                    maxit, relTol, absTol, preSweeps, omega, symmetric, info.finalRelResNorm);
    }

    for (int it=1; it<=maxit; ++it) {
        const double rho = mcgs_dot(rhat, r);

        if (!std::isfinite(rho) || std::abs(rho) < 1e-300) {
            if (monitor) std::printf("CPU BiCGSTAB+MCGS breakdown: rho %.17e\n", rho);
            break;
        }

        if (it == 1) {
            p = r;
        } else {
            if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
                if (monitor) std::printf("CPU BiCGSTAB+MCGS breakdown before beta: omega %.17e\n", omegaB);
                break;
            }

            const double beta = (rho/rhoOld) * (alpha/omegaB);

            for (int i=0; i<n; ++i) {
                p[i] = r[i] + beta*(p[i] - omegaB*v[i]);
            }
        }

        apply_Ahat(p, v);

        const double denom = mcgs_dot(rhat, v);

        if (!std::isfinite(denom) || std::abs(denom) < 1e-300) {
            if (monitor) std::printf("CPU BiCGSTAB+MCGS breakdown: denom %.17e\n", denom);
            break;
        }

        alpha = rho / denom;

        for (int i=0; i<n; ++i) {
            s[i] = r[i] - alpha*v[i];
        }

        const double sNorm = norm_vec(s) / bhatNorm;

        if (sNorm <= relTol) {
            for (int i=0; i<n; ++i) {
                x[i] += alpha*p[i];
            }

            info.iterations = it;
            break;
        }

        apply_Ahat(s, t);

        const double tt = mcgs_dot(t, t);

        if (!std::isfinite(tt) || std::abs(tt) < 1e-300) {
            if (monitor) std::printf("CPU BiCGSTAB+MCGS breakdown: tt %.17e\n", tt);
            break;
        }

        omegaB = mcgs_dot(t, s) / tt;

        for (int i=0; i<n; ++i) {
            x[i] += alpha*p[i] + omegaB*s[i];
            r[i] = s[i] - omegaB*t[i];
        }

        apply_csr(pat, A, x, Ax);

        for (int i=0; i<n; ++i) {
            rTrue[i] = rhs[i] - Ax[i];
        }

        const double absTrue = norm_vec(rTrue);
        const double relTrue = absTrue / trueRhsNorm;

        info.iterations = it;
        info.finalRelResNorm = relTrue;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("CPU BiCGSTAB+MCGS iter %6d trueAbs %.17e trueRel %.17e preRel %.17e\n",
                        it, absTrue, relTrue, norm_vec(r)/bhatNorm);
        }

        if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
            break;
        }

        if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
            if (monitor) std::printf("CPU BiCGSTAB+MCGS breakdown: omega %.17e\n", omegaB);
            break;
        }

        rhoOld = rho;
    }

    return info;
}



static void assemble_lp_cellblock_schur_from_invblocks(
    int nP,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<std::array<std::array<double,10>,10>>& invBlocks,
    CSRPattern& lpPat,
    std::vector<double>& LpValues)
{
    std::vector<std::map<int,double>> rows(nP);

    auto add_direction = [&](const RectCSR& Ap) {
        const int nCells = (int)invBlocks.size();
        for (int c=0; c<nCells; ++c) {
            for (int a=0; a<10; ++a) {
                const int rowA = 10*c + a;
                for (int b=0; b<10; ++b) {
                    const int rowB = 10*c + b;
                    const double mij = invBlocks[c][a][b];
                    if (std::abs(mij) <= 1e-300) continue;
                    for (int pa=Ap.rowOffsets[rowA]; pa<Ap.rowOffsets[rowA+1]; ++pa) {
                        const int colA = Ap.cols[pa];
                        const double va = Ap.values[pa];
                        if (std::abs(va) <= 1e-300) continue;
                        for (int pb=Ap.rowOffsets[rowB]; pb<Ap.rowOffsets[rowB+1]; ++pb) {
                            const int colB = Ap.cols[pb];
                            const double vb = Ap.values[pb];
                            if (std::abs(vb) <= 1e-300) continue;
                            rows[colA][colB] += va * mij * vb;
                        }
                    }
                }
            }
        }
    };

    add_direction(Apx);
    add_direction(Apy);
    add_direction(Apz);
    lpPat = rows_to_csrpattern(nP, rows, LpValues);
}

static double correct_velocity_cellblock_schur_direction(
    const std::vector<std::array<std::array<double,10>,10>>& invBlocks,
    const RectCSR& Ap,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale)
{
    std::vector<double> g;
    apply_rect(Ap, pcorr, g);

    double z2 = 0.0;
    const int nCells = (int)invBlocks.size();

    for (int c=0; c<nCells; ++c) {
        double du[10]{};

        for (int a=0; a<10; ++a) {
            double s = 0.0;
            for (int b=0; b<10; ++b) {
                s += invBlocks[c][a][b] * g[10*c + b];
            }
            du[a] = -scale * s;
        }

        for (int a=0; a<10; ++a) {
            u[10*c + a] += du[a];
            z2 += du[a] * du[a];
        }
    }

    return std::sqrt(std::max(0.0, z2));
}

// =============================================================================
// GPU-resident velocity Krylov + multi-color Gauss-Seidel (MCGS)
// =============================================================================
// This path intentionally uses the user CSR ordering directly and does not write
// into HYPRE's internal ParCSR value array.  That keeps it independent from the
// unsafe raw HYPRE direct-matrix experiment described in the KT.

struct GpuMcgsSystem {
    bool ready = false;
    int nRows = 0;
    int nnz = 0;
    int nColors = 0;
    int nPartials = 0;

    int *d_rowOffsets = nullptr;
    int *d_cols = nullptr;
    int *d_colorOffsets = nullptr;
    int *d_colorRows = nullptr;

    HYPRE_Complex *d_A = nullptr;
    HYPRE_Complex *d_rhs = nullptr;
    HYPRE_Complex *d_x = nullptr;
    HYPRE_Complex *d_Ax = nullptr;
    HYPRE_Complex *d_rTrue = nullptr;
    HYPRE_Complex *d_r = nullptr;
    HYPRE_Complex *d_rhat = nullptr;
    HYPRE_Complex *d_p = nullptr;
    HYPRE_Complex *d_v = nullptr;
    HYPRE_Complex *d_s = nullptr;
    HYPRE_Complex *d_t = nullptr;
    HYPRE_Complex *d_tmp = nullptr;
    HYPRE_Complex *d_bhat = nullptr;
    HYPRE_Complex *d_work1 = nullptr;
    HYPRE_Complex *d_work2 = nullptr;
    HYPRE_Complex *d_cellInv = nullptr;
    int *d_cellBad = nullptr;
    double *d_partials = nullptr;

    bool cellBlockMode = false;
    bool cellMatrixValid = false;
    int nCells = 0;
    int cellBlockSize = 10;
    long long cumulativeCellBlockBuilds = 0;
    long long cumulativeCellBlockApplies = 0;
    double cumulativeCellBlockBuildTime = 0.0;
    double lastCellBlockBuildTime = 0.0;
    int lastCellBlockBad = 0;

    std::vector<int> h_colorOffsets;

    long long cumulativeSolves = 0;
    long long cumulativeKrylovIts = 0;
    long long cumulativePrecondSweeps = 0;
    double cumulativeKrylovTime = 0.0;
    double cumulativePrecondTime = 0.0;
    double cumulativeMatvecTime = 0.0;
    double cumulativeUploadDownloadTime = 0.0;

    int lastIterations = 0;
    double lastRelRes = 0.0;
    double lastKrylovTime = 0.0;
    double lastPrecondTime = 0.0;
    double lastMatvecTime = 0.0;
};

static bool is_gpu_mcgs_solver_name(const std::string& s)
{
    return s == "mcgs_gpu" || s == "mcgs-gpu" ||
           s == "bicgstab_mcgs_gpu" || s == "bicgstab-mcgs-gpu" ||
           s == "bicgstab+mcgs_gpu" || s == "bicgstab+mcgs-gpu";
}

static bool is_gpu_cellblock_solver_name(const std::string& s)
{
    return s == "celljacobi_gpu" || s == "cellblock_gpu" || s == "blockjacobi_gpu" ||
           s == "bicgstab_celljacobi_gpu" || s == "bicgstab-celljacobi-gpu" ||
           s == "bicgstab_cellblock_gpu" || s == "bicgstab-cellblock-gpu" ||
           s == "bicgstab_cellasm_gpu" || s == "bicgstab-cellasm-gpu" ||
           s == "bicgstab_celljacobi_right_gpu" || s == "bicgstab-celljacobi-right-gpu" ||
           s == "bicgstab_cellblock_right_gpu" || s == "bicgstab-cellblock-right-gpu" ||
           s == "bicgstab_colored_cellgs_gpu" || s == "bicgstab-colored-cellgs-gpu" ||
           s == "bicgstab_cellmcgs_gpu" || s == "bicgstab-cellmcgs-gpu" ||
           s == "bicgstab_blockgs_gpu" || s == "bicgstab-blockgs-gpu";
}

static bool is_any_mcgs_solver_name(const std::string& s)
{
    return s == "mcgs" || s == "bicgstab_mcgs" || is_gpu_mcgs_solver_name(s);
}

__global__ static void dg2_gpu_csr_spmv_kernel(
    int n,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ cols,
    const HYPRE_Complex* __restrict__ A,
    const HYPRE_Complex* __restrict__ x,
    HYPRE_Complex* __restrict__ y)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;

    double sum = 0.0;
    for (int p = rowOffsets[r]; p < rowOffsets[r+1]; ++p) {
        sum += (double)A[p] * (double)x[cols[p]];
    }
    y[r] = (HYPRE_Complex)sum;
}

__global__ static void dg2_gpu_zero_kernel(int n, HYPRE_Complex* x)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (HYPRE_Complex)0;
}

__global__ static void dg2_gpu_copy_kernel(int n, const HYPRE_Complex* src, HYPRE_Complex* dst)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ static void dg2_gpu_residual_kernel(
    int n,
    const HYPRE_Complex* __restrict__ b,
    const HYPRE_Complex* __restrict__ Ax,
    HYPRE_Complex* __restrict__ r)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = (HYPRE_Complex)((double)b[i] - (double)Ax[i]);
}

__global__ static void dg2_gpu_p_update_kernel(
    int n,
    const HYPRE_Complex* __restrict__ r,
    HYPRE_Complex* __restrict__ p,
    const HYPRE_Complex* __restrict__ v,
    double beta,
    double omegaB,
    int firstIter)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (firstIter) {
        p[i] = r[i];
    } else {
        p[i] = (HYPRE_Complex)((double)r[i] + beta*((double)p[i] - omegaB*(double)v[i]));
    }
}

__global__ static void dg2_gpu_s_update_kernel(
    int n,
    const HYPRE_Complex* __restrict__ r,
    const HYPRE_Complex* __restrict__ v,
    HYPRE_Complex* __restrict__ s,
    double alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) s[i] = (HYPRE_Complex)((double)r[i] - alpha*(double)v[i]);
}

__global__ static void dg2_gpu_x_add_alpha_p_kernel(
    int n,
    HYPRE_Complex* __restrict__ x,
    const HYPRE_Complex* __restrict__ p,
    double alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (HYPRE_Complex)((double)x[i] + alpha*(double)p[i]);
}

__global__ static void dg2_gpu_x_r_update_kernel(
    int n,
    HYPRE_Complex* __restrict__ x,
    HYPRE_Complex* __restrict__ r,
    const HYPRE_Complex* __restrict__ p,
    const HYPRE_Complex* __restrict__ s,
    const HYPRE_Complex* __restrict__ t,
    double alpha,
    double omegaB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = (HYPRE_Complex)((double)x[i] + alpha*(double)p[i] + omegaB*(double)s[i]);
    r[i] = (HYPRE_Complex)((double)s[i] - omegaB*(double)t[i]);
}



__global__ static void dg2_gpu_x_r_update_right_kernel(
    int n,
    HYPRE_Complex* __restrict__ x,
    HYPRE_Complex* __restrict__ r,
    const HYPRE_Complex* __restrict__ phat,
    const HYPRE_Complex* __restrict__ s,
    const HYPRE_Complex* __restrict__ shat,
    const HYPRE_Complex* __restrict__ t,
    double alpha,
    double omegaB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = (HYPRE_Complex)((double)x[i] + alpha*(double)phat[i] + omegaB*(double)shat[i]);
    r[i] = (HYPRE_Complex)((double)s[i] - omegaB*(double)t[i]);
}

__global__ static void dg2_gpu_axpy_kernel(
    int n,
    HYPRE_Complex* __restrict__ y,
    const HYPRE_Complex* __restrict__ x,
    double alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (HYPRE_Complex)((double)y[i] + alpha*(double)x[i]);
}

__global__ static void dg2_gpu_diff_kernel(
    int n,
    const HYPRE_Complex* __restrict__ a,
    const HYPRE_Complex* __restrict__ b,
    HYPRE_Complex* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (HYPRE_Complex)((double)a[i] - (double)b[i]);
}

__global__ static void dg2_gpu_cellblock_build_inv10_kernel(
    int nCells,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ cols,
    const HYPRE_Complex* __restrict__ A,
    HYPRE_Complex* __restrict__ invBlocks,
    int* __restrict__ badCount)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;

    double aug[10][20];
    for (int i=0; i<10; ++i) {
        const int row = 10*c + i;
        for (int j=0; j<10; ++j) aug[i][j] = 0.0;
        for (int p=rowOffsets[row]; p<rowOffsets[row+1]; ++p) {
            const int col = cols[p] - 10*c;
            if (col >= 0 && col < 10) aug[i][col] = (double)A[p];
        }
        for (int j=0; j<10; ++j) aug[i][10+j] = (i == j) ? 1.0 : 0.0;
    }

    int bad = 0;
    for (int k=0; k<10; ++k) {
        int piv = k;
        double best = fabs(aug[k][k]);
        for (int r=k+1; r<10; ++r) {
            const double v = fabs(aug[r][k]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-30 || !isfinite(best)) { bad = 1; break; }
        if (piv != k) {
            for (int j=0; j<20; ++j) {
                const double tmp = aug[k][j];
                aug[k][j] = aug[piv][j];
                aug[piv][j] = tmp;
            }
        }
        const double invp = 1.0 / aug[k][k];
        for (int j=0; j<20; ++j) aug[k][j] *= invp;
        for (int r=0; r<10; ++r) {
            if (r == k) continue;
            const double f = aug[r][k];
            if (f == 0.0) continue;
            for (int j=0; j<20; ++j) aug[r][j] -= f * aug[k][j];
        }
    }

    const int base = 100*c;
    if (bad) {
        atomicAdd(badCount, 1);
        for (int i=0; i<10; ++i) {
            double aii = 0.0;
            const int row = 10*c + i;
            for (int p=rowOffsets[row]; p<rowOffsets[row+1]; ++p) {
                if (cols[p] == row) { aii = (double)A[p]; break; }
            }
            for (int j=0; j<10; ++j) invBlocks[base + 10*i + j] = (HYPRE_Complex)0;
            invBlocks[base + 10*i + i] = (HYPRE_Complex)((fabs(aii) > 1e-30) ? (1.0/aii) : 1.0);
        }
    } else {
        for (int i=0; i<10; ++i) {
            for (int j=0; j<10; ++j) invBlocks[base + 10*i + j] = (HYPRE_Complex)aug[i][10+j];
        }
    }
}

__global__ static void dg2_gpu_cellblock_apply_inv10_kernel(
    int nCells,
    const HYPRE_Complex* __restrict__ invBlocks,
    const HYPRE_Complex* __restrict__ rhs,
    HYPRE_Complex* __restrict__ z)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;
    const int rb = 10*c;
    const int ib = 100*c;
    double r[10];
    for (int j=0; j<10; ++j) r[j] = (double)rhs[rb+j];
    for (int i=0; i<10; ++i) {
        double s = 0.0;
        for (int j=0; j<10; ++j) s += (double)invBlocks[ib + 10*i + j] * r[j];
        z[rb+i] = (HYPRE_Complex)s;
    }
}


__global__ static void dg2_gpu_colored_cellblock_gs_color_kernel(
    int colorBegin,
    int colorEnd,
    const int* __restrict__ colorCells,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ cols,
    const HYPRE_Complex* __restrict__ A,
    const HYPRE_Complex* __restrict__ rhs,
    HYPRE_Complex* __restrict__ z,
    const HYPRE_Complex* __restrict__ invBlocks,
    double omega)
{
    const int kk = colorBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (kk >= colorEnd) return;

    const int c = colorCells[kk];
    const int rb = 10*c;
    const int ib = 100*c;

    double res[10];

    for (int i=0; i<10; ++i) {
        const int row = rb + i;

        double Az = 0.0;
        for (int p=rowOffsets[row]; p<rowOffsets[row+1]; ++p) {
            Az += (double)A[p] * (double)z[cols[p]];
        }

        res[i] = (double)rhs[row] - Az;
    }

    for (int i=0; i<10; ++i) {
        double dz = 0.0;

        for (int j=0; j<10; ++j) {
            dz += (double)invBlocks[ib + 10*i + j] * res[j];
        }

        z[rb+i] = (HYPRE_Complex)((double)z[rb+i] + omega*dz);
    }
}

__global__ static void dg2_gpu_mcgs_color_kernel(
    int colorBegin,
    int colorEnd,
    const int* __restrict__ colorRows,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ cols,
    const HYPRE_Complex* __restrict__ A,
    const HYPRE_Complex* __restrict__ rhs,
    HYPRE_Complex* __restrict__ x,
    double omega)
{
    const int kk = colorBegin + blockIdx.x * blockDim.x + threadIdx.x;
    if (kk >= colorEnd) return;

    const int row = colorRows[kk];
    double off = 0.0;
    double aii = 0.0;

    for (int p = rowOffsets[row]; p < rowOffsets[row+1]; ++p) {
        const int col = cols[p];
        const double a = (double)A[p];
        if (col == row) aii = a;
        else off += a * (double)x[col];
    }

    if (fabs(aii) > 1e-300) {
        const double xgs = ((double)rhs[row] - off) / aii;
        x[row] = (HYPRE_Complex)((1.0 - omega)*(double)x[row] + omega*xgs);
    }
}

__global__ static void dg2_gpu_dot_kernel(
    int n,
    const HYPRE_Complex* __restrict__ a,
    const HYPRE_Complex* __restrict__ b,
    double* __restrict__ partial)
{
    __shared__ double sh[256];
    const int tid = threadIdx.x;
    double sum = 0.0;

    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        sum += (double)a[i] * (double)b[i];
    }

    sh[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sh[0];
}

static void init_gpu_mcgs_system(GpuMcgsSystem& g, const CSRPattern& pat, const CpuMcgsColoring& coloring)
{
    if (g.ready) return;
    if (coloring.nRows != pat.nRows || coloring.nColors <= 0) {
        throw std::runtime_error("init_gpu_mcgs_system: valid CPU coloring is required before GPU MCGS setup");
    }

    g.nRows = pat.nRows;
    g.nnz = pat.nnz;
    g.nColors = coloring.nColors;
    g.nPartials = std::max(1, std::min(1024, (pat.nRows + 255) / 256));

    std::vector<int> h_cols(pat.nnz, 0);
    for (int i = 0; i < pat.nnz; ++i) {
        const HYPRE_BigInt c = pat.cols[i];
        if (c < 0 || c > std::numeric_limits<int>::max()) {
            throw std::runtime_error("init_gpu_mcgs_system: CSR column index does not fit in int");
        }
        h_cols[i] = (int)c;
    }

    g.d_rowOffsets = dg2_cuda_malloc_copy_vec(pat.rowOffsets, "mcgs-rowOffsets");
    g.d_cols = dg2_cuda_malloc_copy_vec(h_cols, "mcgs-cols");
    g.h_colorOffsets = coloring.colorOffsets;
    g.d_colorOffsets = dg2_cuda_malloc_copy_vec(coloring.colorOffsets, "mcgs-colorOffsets");
    g.d_colorRows = dg2_cuda_malloc_copy_vec(coloring.colorRows, "mcgs-colorRows");

    g.d_A = dg2_cuda_malloc_count<HYPRE_Complex>(g.nnz, "mcgs-A");
    g.d_rhs = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-rhs");
    g.d_x = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-x");
    g.d_Ax = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-Ax");
    g.d_rTrue = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-rTrue");
    g.d_r = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-r");
    g.d_rhat = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-rhat");
    g.d_p = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-p");
    g.d_v = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-v");
    g.d_s = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-s");
    g.d_t = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-t");
    g.d_tmp = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-tmp");
    g.d_bhat = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "mcgs-bhat");
    g.d_partials = dg2_cuda_malloc_count<double>(g.nPartials, "mcgs-partials");

    g.ready = true;

    size_t freeB = 0, totalB = 0;
    DG2_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    std::printf("GPU_MCGS_SETUP nRows=%d nnz=%d nColors=%d partials=%d precisionBytes=%zu free=%.1f MiB total=%.1f MiB\n",
                g.nRows, g.nnz, g.nColors, g.nPartials, sizeof(HYPRE_Complex),
                (double)freeB / (1024.0*1024.0), (double)totalB / (1024.0*1024.0));
}

static void destroy_gpu_mcgs_system(GpuMcgsSystem& g)
{
    void* p = nullptr;
    p = g.d_rowOffsets; dg2_cuda_free_ptr(p); g.d_rowOffsets = nullptr;
    p = g.d_cols; dg2_cuda_free_ptr(p); g.d_cols = nullptr;
    p = g.d_colorOffsets; dg2_cuda_free_ptr(p); g.d_colorOffsets = nullptr;
    p = g.d_colorRows; dg2_cuda_free_ptr(p); g.d_colorRows = nullptr;
    p = g.d_A; dg2_cuda_free_ptr(p); g.d_A = nullptr;
    p = g.d_rhs; dg2_cuda_free_ptr(p); g.d_rhs = nullptr;
    p = g.d_x; dg2_cuda_free_ptr(p); g.d_x = nullptr;
    p = g.d_Ax; dg2_cuda_free_ptr(p); g.d_Ax = nullptr;
    p = g.d_rTrue; dg2_cuda_free_ptr(p); g.d_rTrue = nullptr;
    p = g.d_r; dg2_cuda_free_ptr(p); g.d_r = nullptr;
    p = g.d_rhat; dg2_cuda_free_ptr(p); g.d_rhat = nullptr;
    p = g.d_p; dg2_cuda_free_ptr(p); g.d_p = nullptr;
    p = g.d_v; dg2_cuda_free_ptr(p); g.d_v = nullptr;
    p = g.d_s; dg2_cuda_free_ptr(p); g.d_s = nullptr;
    p = g.d_t; dg2_cuda_free_ptr(p); g.d_t = nullptr;
    p = g.d_tmp; dg2_cuda_free_ptr(p); g.d_tmp = nullptr;
    p = g.d_bhat; dg2_cuda_free_ptr(p); g.d_bhat = nullptr;
    p = g.d_work1; dg2_cuda_free_ptr(p); g.d_work1 = nullptr;
    p = g.d_work2; dg2_cuda_free_ptr(p); g.d_work2 = nullptr;
    p = g.d_cellInv; dg2_cuda_free_ptr(p); g.d_cellInv = nullptr;
    p = g.d_cellBad; dg2_cuda_free_ptr(p); g.d_cellBad = nullptr;
    p = g.d_partials; dg2_cuda_free_ptr(p); g.d_partials = nullptr;
    g.h_colorOffsets.clear();
    g.ready = false;
}

static void gpu_mcgs_upload_problem(
    GpuMcgsSystem& g,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::vector<double>& x)
{
    if (!g.ready) throw std::runtime_error("gpu_mcgs_upload_problem: GPU MCGS system is not initialized");
    if ((int)A.size() != g.nnz || (int)rhs.size() != g.nRows || (int)x.size() != g.nRows) {
        throw std::runtime_error("gpu_mcgs_upload_problem: size mismatch");
    }

    const double t0 = wall_seconds();
    std::vector<HYPRE_Complex> hA = to_hypre_complex_vec(A);
    std::vector<HYPRE_Complex> hb = to_hypre_complex_vec(rhs);
    std::vector<HYPRE_Complex> hx = to_hypre_complex_vec(x);
    DG2_CUDA_CHECK(cudaMemcpy(g.d_A, hA.data(), (size_t)g.nnz*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
    DG2_CUDA_CHECK(cudaMemcpy(g.d_rhs, hb.data(), (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
    DG2_CUDA_CHECK(cudaMemcpy(g.d_x, hx.data(), (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
    g.cumulativeUploadDownloadTime += wall_seconds() - t0;
}

static void gpu_mcgs_download_x(GpuMcgsSystem& g, std::vector<double>& x)
{
    const double t0 = wall_seconds();
    std::vector<HYPRE_Complex> hx(g.nRows);
    DG2_CUDA_CHECK(cudaMemcpy(hx.data(), g.d_x, (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyDeviceToHost));
    x = from_hypre_complex_vec(hx);
    g.cumulativeUploadDownloadTime += wall_seconds() - t0;
}

static void gpu_csr_spmv(GpuMcgsSystem& g, const HYPRE_Complex* d_x, HYPRE_Complex* d_y, double* timeAccum)
{
    const double t0 = wall_seconds();
    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;
    dg2_gpu_csr_spmv_kernel<<<grid, block>>>(g.nRows, g.d_rowOffsets, g.d_cols, g.d_A, d_x, d_y);
    DG2_CUDA_CHECK(cudaGetLastError());
    if (timeAccum) {
        DG2_CUDA_CHECK(cudaDeviceSynchronize());
        *timeAccum += wall_seconds() - t0;
    }
}

static double gpu_dot(GpuMcgsSystem& g, const HYPRE_Complex* d_a, const HYPRE_Complex* d_b)
{
    dg2_gpu_dot_kernel<<<g.nPartials, 256>>>(g.nRows, d_a, d_b, g.d_partials);
    DG2_CUDA_CHECK(cudaGetLastError());
    std::vector<double> partial(g.nPartials, 0.0);
    DG2_CUDA_CHECK(cudaMemcpy(partial.data(), g.d_partials, (size_t)g.nPartials*sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : partial) s += v;
    return s;
}

static double gpu_norm(GpuMcgsSystem& g, const HYPRE_Complex* d_a)
{
    return std::sqrt(std::max(0.0, gpu_dot(g, d_a, d_a)));
}

static void gpu_residual(GpuMcgsSystem& g, const HYPRE_Complex* d_x, HYPRE_Complex* d_r, double* matvecTime)
{
    gpu_csr_spmv(g, d_x, g.d_Ax, matvecTime);
    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;
    dg2_gpu_residual_kernel<<<grid, block>>>(g.nRows, g.d_rhs, g.d_Ax, d_r);
    DG2_CUDA_CHECK(cudaGetLastError());
}

static void gpu_mcgs_sweeps(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_rhs,
    HYPRE_Complex* d_x,
    int sweeps,
    double omega,
    int symmetric,
    double* precondTime)
{
    const double t0 = wall_seconds();
    const int block = 256;

    if ((int)g.h_colorOffsets.size() != g.nColors + 1) {
        throw std::runtime_error("gpu_mcgs_sweeps: missing host color offsets cache");
    }

    for (int sw = 0; sw < sweeps; ++sw) {
        for (int c = 0; c < g.nColors; ++c) {
            const int cb = g.h_colorOffsets[c];
            const int ce = g.h_colorOffsets[c+1];
            const int n = ce - cb;
            if (n <= 0) continue;
            const int grid = (n + block - 1) / block;
            dg2_gpu_mcgs_color_kernel<<<grid, block>>>(cb, ce, g.d_colorRows, g.d_rowOffsets, g.d_cols, g.d_A, d_rhs, d_x, omega);
            DG2_CUDA_CHECK(cudaGetLastError());
        }
        if (symmetric) {
            for (int c = g.nColors - 1; c >= 0; --c) {
                const int cb = g.h_colorOffsets[c];
                const int ce = g.h_colorOffsets[c+1];
                const int n = ce - cb;
                if (n <= 0) continue;
                const int grid = (n + block - 1) / block;
                dg2_gpu_mcgs_color_kernel<<<grid, block>>>(cb, ce, g.d_colorRows, g.d_rowOffsets, g.d_cols, g.d_A, d_rhs, d_x, omega);
                DG2_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    if (precondTime) {
        DG2_CUDA_CHECK(cudaDeviceSynchronize());
        *precondTime += wall_seconds() - t0;
    }
    g.cumulativePrecondSweeps += (long long)sweeps * (symmetric ? 2 : 1);
}

static void gpu_mcgs_preconditioner_zero_start(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_rhs,
    HYPRE_Complex* d_z,
    int sweeps,
    double omega,
    int symmetric,
    double* precondTime)
{
    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;
    dg2_gpu_zero_kernel<<<grid, block>>>(g.nRows, d_z);
    DG2_CUDA_CHECK(cudaGetLastError());
    gpu_mcgs_sweeps(g, d_rhs, d_z, sweeps, omega, symmetric, precondTime);
}

static void gpu_apply_Ahat_mcgs_left(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_vin,
    HYPRE_Complex* d_yout,
    int preSweeps,
    double omega,
    int symmetric,
    double* matvecTime,
    double* precondTime)
{
    gpu_csr_spmv(g, d_vin, g.d_tmp, matvecTime);
    gpu_mcgs_preconditioner_zero_start(g, g.d_tmp, d_yout, preSweeps, omega, symmetric, precondTime);
}

static HypreSolveInfo solve_momentum_component_mcgs_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int sweeps,
    double omega,
    int symmetric,
    double relTol,
    double absTol,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    gpu_mcgs_upload_problem(g, A, rhs, x);
    const double rhsNorm = std::max(gpu_norm(g, g.d_rhs), 1e-300);

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = 1.0;

    if (monitor) {
        std::printf("GPU MCGS momentum: sweeps=%d omega=%.6g symmetric=%d relTol=%.3e absTol=%.3e\n",
                    sweeps, omega, symmetric, relTol, absTol);
    }

    for (int sw = 1; sw <= std::max(1, sweeps); ++sw) {
        gpu_mcgs_sweeps(g, g.d_rhs, g.d_x, 1, omega, symmetric, &g.lastPrecondTime);
        info.iterations = sw;

        if (monitor || relTol > 0.0 || absTol > 0.0 || sw == sweeps) {
            gpu_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
            const double absRes = gpu_norm(g, g.d_rTrue);
            const double relRes = absRes / rhsNorm;
            info.finalRelResNorm = relRes;

            if (monitor && (sw <= 10 || (sw % 10) == 0)) {
                std::printf("GPU MCGS sweep %6d absRes %.17e relRes %.17e\n", sw, absRes, relRes);
            }

            if ((relTol > 0.0 && relRes <= relTol) || (absTol > 0.0 && absRes <= absTol)) {
                break;
            }
        }
    }

    gpu_mcgs_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    if (monitor) {
        std::printf("GPU MCGS done: iterations=%d relRes=%.17e time=%.6f precond=%.6f matvec=%.6f\n",
                    info.iterations, info.finalRelResNorm, g.lastKrylovTime, g.lastPrecondTime, g.lastMatvecTime);
    }

    return info;
}

static HypreSolveInfo solve_momentum_component_bicgstab_mcgs_left_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    double relTol,
    double absTol,
    int preSweeps,
    double omega,
    int symmetric,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);
    preSweeps = std::max(1, preSweeps);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    gpu_mcgs_upload_problem(g, A, rhs, x);

    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;

    gpu_mcgs_preconditioner_zero_start(g, g.d_rhs, g.d_bhat, preSweeps, omega, symmetric, &g.lastPrecondTime);
    const double trueRhsNorm = std::max(gpu_norm(g, g.d_rhs), 1e-300);
    const double bhatNorm = std::max(gpu_norm(g, g.d_bhat), 1e-300);

    gpu_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    gpu_mcgs_preconditioner_zero_start(g, g.d_rTrue, g.d_r, preSweeps, omega, symmetric, &g.lastPrecondTime);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_rhat);
    DG2_CUDA_CHECK(cudaGetLastError());

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omegaB = 1.0;

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = gpu_norm(g, g.d_rTrue) / trueRhsNorm;

    if (monitor) {
        std::printf("GPU BiCGSTAB+MCGS-left: maxit=%d relTol=%.3e absTol=%.3e preSweeps=%d omega=%.6g symmetric=%d trueRel0=%.17e\n",
                    maxit, relTol, absTol, preSweeps, omega, symmetric, info.finalRelResNorm);
    }

    if ((relTol > 0.0 && info.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && info.finalRelResNorm * trueRhsNorm <= absTol)) {
        gpu_mcgs_download_x(g, x);
        g.lastKrylovTime = wall_seconds() - tSolve0;
        g.lastIterations = info.iterations;
        g.lastRelRes = info.finalRelResNorm;
        g.cumulativeSolves++;
        g.cumulativeKrylovTime += g.lastKrylovTime;
        g.cumulativePrecondTime += g.lastPrecondTime;
        g.cumulativeMatvecTime += g.lastMatvecTime;
        return info;
    }

    for (int it = 1; it <= maxit; ++it) {
        const double rho = gpu_dot(g, g.d_rhat, g.d_r);
        if (!std::isfinite(rho) || std::abs(rho) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+MCGS breakdown: rho %.17e\n", rho);
            break;
        }

        double beta = 0.0;
        if (it > 1) {
            if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
                if (monitor) std::printf("GPU BiCGSTAB+MCGS breakdown before beta: omega %.17e\n", omegaB);
                break;
            }
            beta = (rho/rhoOld) * (alpha/omegaB);
        }
        dg2_gpu_p_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_p, g.d_v, beta, omegaB, it == 1);
        DG2_CUDA_CHECK(cudaGetLastError());

        gpu_apply_Ahat_mcgs_left(g, g.d_p, g.d_v, preSweeps, omega, symmetric, &g.lastMatvecTime, &g.lastPrecondTime);

        const double denom = gpu_dot(g, g.d_rhat, g.d_v);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+MCGS breakdown: denom %.17e\n", denom);
            break;
        }

        alpha = rho / denom;
        dg2_gpu_s_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_v, g.d_s, alpha);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double sNorm = gpu_norm(g, g.d_s) / bhatNorm;
        if (sNorm <= relTol && relTol > 0.0) {
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_p, alpha);
            DG2_CUDA_CHECK(cudaGetLastError());
            gpu_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
            const double absTrue = gpu_norm(g, g.d_rTrue);
            info.iterations = it;
            info.finalRelResNorm = absTrue / trueRhsNorm;
            break;
        }

        gpu_apply_Ahat_mcgs_left(g, g.d_s, g.d_t, preSweeps, omega, symmetric, &g.lastMatvecTime, &g.lastPrecondTime);

        const double tt = gpu_dot(g, g.d_t, g.d_t);
        if (!std::isfinite(tt) || std::abs(tt) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+MCGS breakdown: tt %.17e\n", tt);
            break;
        }

        omegaB = gpu_dot(g, g.d_t, g.d_s) / tt;
        dg2_gpu_x_r_update_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_r, g.d_p, g.d_s, g.d_t, alpha, omegaB);
        DG2_CUDA_CHECK(cudaGetLastError());

        gpu_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
        const double absTrue = gpu_norm(g, g.d_rTrue);
        const double relTrue = absTrue / trueRhsNorm;

        info.iterations = it;
        info.finalRelResNorm = relTrue;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("GPU BiCGSTAB+MCGS iter %6d trueAbs %.17e trueRel %.17e preRel %.17e\n",
                        it, absTrue, relTrue, gpu_norm(g, g.d_r)/bhatNorm);
        }

        if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
            break;
        }

        if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+MCGS breakdown: omega %.17e\n", omegaB);
            break;
        }

        rhoOld = rho;
    }

    gpu_mcgs_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    if (monitor) {
        std::printf("GPU BiCGSTAB+MCGS done: iterations=%d relRes=%.17e time=%.6f precond=%.6f matvec=%.6f\n",
                    info.iterations, info.finalRelResNorm, g.lastKrylovTime, g.lastPrecondTime, g.lastMatvecTime);
    }

    return info;
}



// =============================================================================
// GPU cell-block Jacobi / ASM-style velocity preconditioner
// =============================================================================
// Experimental DG-native path: one 10x10 scalar DG2 block per tet/cell.
// This deliberately lives beside the MCGS paths and does not disturb them.

static void init_gpu_cellblock_system(GpuMcgsSystem& g, const CSRPattern& pat)
{
    if (g.ready) return;
    if (pat.nRows % 10 != 0) throw std::runtime_error("init_gpu_cellblock_system: DG2 scalar row count is not divisible by 10");

    g.nRows = pat.nRows;
    g.nnz = pat.nnz;
    g.nCells = pat.nRows / 10;
    g.cellBlockSize = 10;

    CpuMcgsColoring cellColoring = build_cellblock_coloring_from_csr(pat);
    g.nColors = cellColoring.nColors;
    g.h_colorOffsets = cellColoring.colorOffsets;

    g.nPartials = std::max(1, std::min(1024, (pat.nRows + 255) / 256));
    g.cellBlockMode = true;
    g.cellMatrixValid = false;

    std::vector<int> h_cols(pat.nnz, 0);
    for (int i = 0; i < pat.nnz; ++i) {
        const HYPRE_BigInt c = pat.cols[i];
        if (c < 0 || c > std::numeric_limits<int>::max()) {
            throw std::runtime_error("init_gpu_cellblock_system: CSR column index does not fit in int");
        }
        h_cols[i] = (int)c;
    }

    g.d_rowOffsets = dg2_cuda_malloc_copy_vec(pat.rowOffsets, "cellblock-rowOffsets");
    g.d_cols = dg2_cuda_malloc_copy_vec(h_cols, "cellblock-cols");
    g.d_colorOffsets = dg2_cuda_malloc_copy_vec(cellColoring.colorOffsets, "cellblock-colorOffsets");
    g.d_colorRows = dg2_cuda_malloc_copy_vec(cellColoring.colorRows, "cellblock-colorCells");

    g.d_A = dg2_cuda_malloc_count<HYPRE_Complex>(g.nnz, "cellblock-A");
    g.d_rhs = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-rhs");
    g.d_x = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-x");
    g.d_Ax = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-Ax");
    g.d_rTrue = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-rTrue");
    g.d_r = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-r");
    g.d_rhat = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-rhat");
    g.d_p = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-p");
    g.d_v = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-v");
    g.d_s = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-s");
    g.d_t = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-t");
    g.d_tmp = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-tmp");
    g.d_bhat = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-bhat");
    g.d_work1 = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-work1");
    g.d_work2 = dg2_cuda_malloc_count<HYPRE_Complex>(g.nRows, "cellblock-work2");
    g.d_cellInv = dg2_cuda_malloc_count<HYPRE_Complex>((size_t)g.nCells * 100, "cellblock-inv10");
    g.d_cellBad = dg2_cuda_malloc_count<int>(1, "cellblock-bad");
    g.d_partials = dg2_cuda_malloc_count<double>(g.nPartials, "cellblock-partials");

    g.ready = true;

    size_t freeB = 0, totalB = 0;
    DG2_CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    std::printf("GPU_CELLBLOCK_SETUP nRows=%d nCells=%d nnz=%d block=10 nColors=%d invBytes=%.1f MiB precisionBytes=%zu free=%.1f MiB total=%.1f MiB\n",
                g.nRows, g.nCells, g.nnz, g.nColors,
                ((double)g.nCells * 100.0 * (double)sizeof(HYPRE_Complex)) / (1024.0*1024.0),
                sizeof(HYPRE_Complex),
                (double)freeB / (1024.0*1024.0), (double)totalB / (1024.0*1024.0));
}

static void gpu_cellblock_upload_problem(
    GpuMcgsSystem& g,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    const std::vector<double>& x)
{
    if (!g.ready || !g.cellBlockMode) throw std::runtime_error("gpu_cellblock_upload_problem: GPU cell-block system is not initialized");
    if ((int)A.size() != g.nnz || (int)rhs.size() != g.nRows || (int)x.size() != g.nRows) {
        throw std::runtime_error("gpu_cellblock_upload_problem: size mismatch");
    }

    const double t0 = wall_seconds();
    if (!g.cellMatrixValid) {
        std::vector<HYPRE_Complex> hA = to_hypre_complex_vec(A);
        DG2_CUDA_CHECK(cudaMemcpy(g.d_A, hA.data(), (size_t)g.nnz*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
        g.cellMatrixValid = true;
    }
    std::vector<HYPRE_Complex> hb = to_hypre_complex_vec(rhs);
    std::vector<HYPRE_Complex> hx = to_hypre_complex_vec(x);
    DG2_CUDA_CHECK(cudaMemcpy(g.d_rhs, hb.data(), (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
    DG2_CUDA_CHECK(cudaMemcpy(g.d_x, hx.data(), (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyHostToDevice));
    g.cumulativeUploadDownloadTime += wall_seconds() - t0;
}

static void gpu_cellblock_download_x(GpuMcgsSystem& g, std::vector<double>& x)
{
    const double t0 = wall_seconds();
    std::vector<HYPRE_Complex> hx(g.nRows);
    DG2_CUDA_CHECK(cudaMemcpy(hx.data(), g.d_x, (size_t)g.nRows*sizeof(HYPRE_Complex), cudaMemcpyDeviceToHost));
    x = from_hypre_complex_vec(hx);
    g.cumulativeUploadDownloadTime += wall_seconds() - t0;
}

static double gpu_cell_dot(GpuMcgsSystem& g, const HYPRE_Complex* d_a, const HYPRE_Complex* d_b)
{
    dg2_gpu_dot_kernel<<<g.nPartials, 256>>>(g.nRows, d_a, d_b, g.d_partials);
    DG2_CUDA_CHECK(cudaGetLastError());
    std::vector<double> partial(g.nPartials, 0.0);
    DG2_CUDA_CHECK(cudaMemcpy(partial.data(), g.d_partials, (size_t)g.nPartials*sizeof(double), cudaMemcpyDeviceToHost));
    double s = 0.0;
    for (double v : partial) s += v;
    return s;
}

static double gpu_cell_norm(GpuMcgsSystem& g, const HYPRE_Complex* d_a)
{
    return std::sqrt(std::max(0.0, gpu_cell_dot(g, d_a, d_a)));
}

static void gpu_cell_csr_spmv(GpuMcgsSystem& g, const HYPRE_Complex* d_x, HYPRE_Complex* d_y, double* timeAccum)
{
    const double t0 = wall_seconds();
    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;
    dg2_gpu_csr_spmv_kernel<<<grid, block>>>(g.nRows, g.d_rowOffsets, g.d_cols, g.d_A, d_x, d_y);
    DG2_CUDA_CHECK(cudaGetLastError());
    if (timeAccum) {
        DG2_CUDA_CHECK(cudaDeviceSynchronize());
        *timeAccum += wall_seconds() - t0;
    }
}

static void gpu_cell_residual(GpuMcgsSystem& g, const HYPRE_Complex* d_x, HYPRE_Complex* d_r, double* matvecTime)
{
    gpu_cell_csr_spmv(g, d_x, g.d_Ax, matvecTime);
    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;
    dg2_gpu_residual_kernel<<<grid, block>>>(g.nRows, g.d_rhs, g.d_Ax, d_r);
    DG2_CUDA_CHECK(cudaGetLastError());
}

static void gpu_cellblock_build_inverse(GpuMcgsSystem& g)
{
    if (!g.ready || !g.cellBlockMode) throw std::runtime_error("gpu_cellblock_build_inverse: not initialized");
    const double t0 = wall_seconds();
    DG2_CUDA_CHECK(cudaMemset(g.d_cellBad, 0, sizeof(int)));
    const int block = 64;
    const int grid = (g.nCells + block - 1) / block;
    dg2_gpu_cellblock_build_inv10_kernel<<<grid, block>>>(
        g.nCells, g.d_rowOffsets, g.d_cols, g.d_A, g.d_cellInv, g.d_cellBad);
    DG2_CUDA_CHECK(cudaGetLastError());
    DG2_CUDA_CHECK(cudaDeviceSynchronize());
    int hBad = 0;
    DG2_CUDA_CHECK(cudaMemcpy(&hBad, g.d_cellBad, sizeof(int), cudaMemcpyDeviceToHost));
    g.lastCellBlockBad = hBad;
    g.lastCellBlockBuildTime = wall_seconds() - t0;
    g.cumulativeCellBlockBuildTime += g.lastCellBlockBuildTime;
    g.cumulativeCellBlockBuilds++;
}

static void gpu_cellblock_preconditioner(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_rhs,
    HYPRE_Complex* d_z,
    int correctionIters,
    double omega,
    double* precondTime,
    double* matvecTime)
{
    const double t0 = wall_seconds();
    correctionIters = std::max(0, correctionIters);
    const int block = 128;
    const int gridCells = (g.nCells + block - 1) / block;
    const int gridRows = (g.nRows + 255) / 256;

    // Base block-Jacobi apply:
    //     z = D_cell^{-1} rhs
    dg2_gpu_cellblock_apply_inv10_kernel<<<gridCells, block>>>(g.nCells, g.d_cellInv, d_rhs, d_z);
    DG2_CUDA_CHECK(cudaGetLastError());

    // Correction sweeps:
    //     z <- z + omega D_cell^{-1}(rhs - A z)
    //
    // IMPORTANT:
    // The caller is allowed to pass d_z as g.d_work1, g.d_work2, or g.d_tmp.
    // Therefore this routine must NOT blindly use g.d_work1/g.d_work2 as
    // scratch, otherwise we can accidentally do in-place SpMV:
    //
    //     SpMV(d_z, d_z)
    //
    // That corrupts the Krylov vector and causes huge velocities/NaNs when
    // -uCellBlockIters > 0.
    auto pick_scratch = [&](HYPRE_Complex* avoid0,
                            const HYPRE_Complex* avoid1,
                            HYPRE_Complex* avoid2) -> HYPRE_Complex*
    {
        HYPRE_Complex* cand[4] = {g.d_work1, g.d_work2, g.d_tmp, g.d_Ax};
        for (HYPRE_Complex* q : cand) {
            if (q == nullptr) continue;
            if (q == avoid0) continue;
            if (q == avoid1) continue;
            if (q == avoid2) continue;
            return q;
        }
        return nullptr;
    };

    for (int k=0; k<correctionIters; ++k) {
        HYPRE_Complex* d_aux = pick_scratch(d_z, d_rhs, nullptr);
        HYPRE_Complex* d_res = pick_scratch(d_z, d_rhs, d_aux);

        if (d_aux == nullptr || d_res == nullptr) {
            throw std::runtime_error("gpu_cellblock_preconditioner: could not find non-aliased scratch buffers");
        }

        // d_aux = A z
        if (matvecTime) {
            const double tSpmv0 = wall_seconds();
            dg2_gpu_csr_spmv_kernel<<<gridRows, 256>>>(g.nRows, g.d_rowOffsets, g.d_cols, g.d_A, d_z, d_aux);
            DG2_CUDA_CHECK(cudaGetLastError());
            DG2_CUDA_CHECK(cudaDeviceSynchronize());
            *matvecTime += wall_seconds() - tSpmv0;
        } else {
            dg2_gpu_csr_spmv_kernel<<<gridRows, 256>>>(g.nRows, g.d_rowOffsets, g.d_cols, g.d_A, d_z, d_aux);
            DG2_CUDA_CHECK(cudaGetLastError());
        }

        // d_res = rhs - A z
        dg2_gpu_diff_kernel<<<gridRows, 256>>>(g.nRows, d_rhs, d_aux, d_res);
        DG2_CUDA_CHECK(cudaGetLastError());

        // d_aux = D_cell^{-1} d_res
        dg2_gpu_cellblock_apply_inv10_kernel<<<gridCells, block>>>(g.nCells, g.d_cellInv, d_res, d_aux);
        DG2_CUDA_CHECK(cudaGetLastError());

        // z += omega d_aux
        dg2_gpu_axpy_kernel<<<gridRows, 256>>>(g.nRows, d_z, d_aux, omega);
        DG2_CUDA_CHECK(cudaGetLastError());
    }

    DG2_CUDA_CHECK(cudaDeviceSynchronize());
    const double dt = wall_seconds() - t0;
    if (precondTime) *precondTime += dt;
    g.cumulativeCellBlockApplies++;
}


static void gpu_colored_cellgs_preconditioner(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_rhs,
    HYPRE_Complex* d_z,
    int sweeps,
    double omega,
    int symmetric,
    double* precondTime)
{
    const double t0 = wall_seconds();

    sweeps = std::max(1, sweeps);

    if (!g.ready || !g.cellBlockMode || g.nColors <= 0 || g.d_colorRows == nullptr) {
        throw std::runtime_error("gpu_colored_cellgs_preconditioner: GPU colored cell-block system is not initialized");
    }

    if ((int)g.h_colorOffsets.size() != g.nColors + 1) {
        throw std::runtime_error("gpu_colored_cellgs_preconditioner: missing host color offsets cache");
    }

    const int block = 128;
    const int gridRows = (g.nRows + 255) / 256;

    // Zero-start approximate block-GS solve:
    //     z ~= A^{-1} rhs
    dg2_gpu_zero_kernel<<<gridRows, 256>>>(g.nRows, d_z);
    DG2_CUDA_CHECK(cudaGetLastError());

    for (int sw=0; sw<sweeps; ++sw) {
        for (int color=0; color<g.nColors; ++color) {
            const int cb = g.h_colorOffsets[color];
            const int ce = g.h_colorOffsets[color+1];
            const int n = ce - cb;

            if (n <= 0) continue;

            const int grid = (n + block - 1) / block;

            dg2_gpu_colored_cellblock_gs_color_kernel<<<grid, block>>>(
                cb, ce,
                g.d_colorRows,
                g.d_rowOffsets,
                g.d_cols,
                g.d_A,
                d_rhs,
                d_z,
                g.d_cellInv,
                omega);

            DG2_CUDA_CHECK(cudaGetLastError());
        }

        if (symmetric) {
            for (int color=g.nColors-1; color>=0; --color) {
                const int cb = g.h_colorOffsets[color];
                const int ce = g.h_colorOffsets[color+1];
                const int n = ce - cb;

                if (n <= 0) continue;

                const int grid = (n + block - 1) / block;

                dg2_gpu_colored_cellblock_gs_color_kernel<<<grid, block>>>(
                    cb, ce,
                    g.d_colorRows,
                    g.d_rowOffsets,
                    g.d_cols,
                    g.d_A,
                    d_rhs,
                    d_z,
                    g.d_cellInv,
                    omega);

                DG2_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    DG2_CUDA_CHECK(cudaDeviceSynchronize());

    if (precondTime) {
        *precondTime += wall_seconds() - t0;
    }

    g.cumulativeCellBlockApplies++;
}

static void gpu_apply_Ahat_cellblock_left(
    GpuMcgsSystem& g,
    const HYPRE_Complex* d_x,
    HYPRE_Complex* d_y,
    int correctionIters,
    double omega,
    double* matvecTime,
    double* precondTime)
{
    gpu_cell_csr_spmv(g, d_x, g.d_work2, matvecTime);
    gpu_cellblock_preconditioner(g, g.d_work2, d_y, correctionIters, omega, precondTime, matvecTime);
}

static HypreSolveInfo solve_momentum_component_celljacobi_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int sweeps,
    double omega,
    int correctionIters,
    double relTol,
    double absTol,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);
    sweeps = std::max(1, sweeps);
    correctionIters = std::max(0, correctionIters);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    const bool needBuild = !g.cellMatrixValid;
    gpu_cellblock_upload_problem(g, A, rhs, x);
    if (needBuild) gpu_cellblock_build_inverse(g);

    const double rhsNorm = std::max(norm_vec(rhs), 1e-300);
    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / rhsNorm;

    if (monitor) {
        std::printf("GPU cell-Jacobi solver: maxSweeps=%d omega=%.6g corrIters=%d relTol=%.3e absTol=%.3e trueRel0=%.17e\n",
                    sweeps, omega, correctionIters, relTol, absTol, info.finalRelResNorm);
    }

    for (int sw=1; sw<=sweeps; ++sw) {
        gpu_cellblock_preconditioner(g, g.d_rTrue, g.d_tmp, correctionIters, 1.0, &g.lastPrecondTime, &g.lastMatvecTime);
        const int grid = (g.nRows + 255) / 256;
        dg2_gpu_axpy_kernel<<<grid, 256>>>(g.nRows, g.d_x, g.d_tmp, omega);
        DG2_CUDA_CHECK(cudaGetLastError());

        gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
        const double absRes = gpu_cell_norm(g, g.d_rTrue);
        const double relRes = absRes / rhsNorm;
        info.iterations = sw;
        info.finalRelResNorm = relRes;

        if (monitor && (sw <= 10 || (sw % 10) == 0)) {
            std::printf("GPU cell-Jacobi sweep %6d absRes %.17e relRes %.17e\n", sw, absRes, relRes);
        }
        if ((relTol > 0.0 && relRes <= relTol) || (absTol > 0.0 && absRes <= absTol)) break;
    }

    gpu_cellblock_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    return info;
}

static HypreSolveInfo solve_momentum_component_bicgstab_cellblock_left_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    double relTol,
    double absTol,
    int correctionIters,
    double omega,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);
    correctionIters = std::max(0, correctionIters);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    const bool needBuild = !g.cellMatrixValid;
    gpu_cellblock_upload_problem(g, A, rhs, x);
    if (needBuild) gpu_cellblock_build_inverse(g);

    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;

    gpu_cellblock_preconditioner(g, g.d_rhs, g.d_bhat, correctionIters, omega, &g.lastPrecondTime, &g.lastMatvecTime);
    const double trueRhsNorm = std::max(gpu_cell_norm(g, g.d_rhs), 1e-300);
    const double bhatNorm = std::max(gpu_cell_norm(g, g.d_bhat), 1e-300);

    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    gpu_cellblock_preconditioner(g, g.d_rTrue, g.d_r, correctionIters, omega, &g.lastPrecondTime, &g.lastMatvecTime);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_rhat);
    DG2_CUDA_CHECK(cudaGetLastError());

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omegaB = 1.0;

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    if (monitor) {
        std::printf("GPU BiCGSTAB+cellblock-left: maxit=%d relTol=%.3e absTol=%.3e corrIters=%d omega=%.6g trueRel0=%.17e\n",
                    maxit, relTol, absTol, correctionIters, omega, info.finalRelResNorm);
    }

    if ((relTol > 0.0 && info.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && info.finalRelResNorm * trueRhsNorm <= absTol)) {
        gpu_cellblock_download_x(g, x);
        g.lastKrylovTime = wall_seconds() - tSolve0;
        g.lastIterations = info.iterations;
        g.lastRelRes = info.finalRelResNorm;
        g.cumulativeSolves++;
        g.cumulativeKrylovTime += g.lastKrylovTime;
        g.cumulativePrecondTime += g.lastPrecondTime;
        g.cumulativeMatvecTime += g.lastMatvecTime;
        return info;
    }

    for (int it = 1; it <= maxit; ++it) {
        const double rho = gpu_cell_dot(g, g.d_rhat, g.d_r);
        if (!std::isfinite(rho) || std::abs(rho) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock breakdown: rho %.17e\n", rho);
            break;
        }

        double beta = 0.0;
        if (it > 1) {
            if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
                if (monitor) std::printf("GPU BiCGSTAB+cellblock breakdown before beta: omega %.17e\n", omegaB);
                break;
            }
            beta = (rho/rhoOld) * (alpha/omegaB);
        }
        dg2_gpu_p_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_p, g.d_v, beta, omegaB, it == 1);
        DG2_CUDA_CHECK(cudaGetLastError());

        gpu_apply_Ahat_cellblock_left(g, g.d_p, g.d_v, correctionIters, omega, &g.lastMatvecTime, &g.lastPrecondTime);

        const double denom = gpu_cell_dot(g, g.d_rhat, g.d_v);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock breakdown: denom %.17e\n", denom);
            break;
        }

        alpha = rho / denom;
        dg2_gpu_s_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_v, g.d_s, alpha);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double sNorm = gpu_cell_norm(g, g.d_s) / bhatNorm;
        if (sNorm <= relTol && relTol > 0.0) {
            // Do not accept convergence using only the left-preconditioned
            // intermediate BiCGSTAB residual. Cell-block Jacobi can make
            // M^{-1}s tiny while the true residual b-Ax is still large.
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_p, alpha);
            DG2_CUDA_CHECK(cudaGetLastError());
            gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
            const double absTrue = gpu_cell_norm(g, g.d_rTrue);
            const double relTrueS = absTrue / trueRhsNorm;
            info.iterations = it;
            info.finalRelResNorm = relTrueS;
            if ((relTol > 0.0 && relTrueS <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
                break;
            }
            // Reject the early-s shortcut and continue the normal omega step.
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_p, -alpha);
            DG2_CUDA_CHECK(cudaGetLastError());
            if (monitor) {
                std::printf("GPU BiCGSTAB+cellblock early-s rejected: preRel %.17e trueAbs %.17e trueRel %.17e\n",
                            sNorm, absTrue, relTrueS);
            }
        }

        gpu_apply_Ahat_cellblock_left(g, g.d_s, g.d_t, correctionIters, omega, &g.lastMatvecTime, &g.lastPrecondTime);

        const double tt = gpu_cell_dot(g, g.d_t, g.d_t);
        if (!std::isfinite(tt) || std::abs(tt) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock breakdown: tt %.17e\n", tt);
            break;
        }

        omegaB = gpu_cell_dot(g, g.d_t, g.d_s) / tt;
        dg2_gpu_x_r_update_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_r, g.d_p, g.d_s, g.d_t, alpha, omegaB);
        DG2_CUDA_CHECK(cudaGetLastError());

        gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
        const double absTrue = gpu_cell_norm(g, g.d_rTrue);
        const double relTrue = absTrue / trueRhsNorm;

        info.iterations = it;
        info.finalRelResNorm = relTrue;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("GPU BiCGSTAB+cellblock iter %6d trueAbs %.17e trueRel %.17e preRel %.17e\n",
                        it, absTrue, relTrue, gpu_cell_norm(g, g.d_r)/bhatNorm);
        }

        if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) break;
        if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock breakdown: omega %.17e\n", omegaB);
            break;
        }
        rhoOld = rho;
    }

    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    gpu_cellblock_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    if (monitor) {
        std::printf("GPU BiCGSTAB+cellblock done: iterations=%d relRes=%.17e time=%.6f precond=%.6f matvec=%.6f invBuild=%.6f badBlocks=%d\n",
                    info.iterations, info.finalRelResNorm, g.lastKrylovTime, g.lastPrecondTime,
                    g.lastMatvecTime, g.lastCellBlockBuildTime, g.lastCellBlockBad);
    }

    return info;
}



static HypreSolveInfo solve_momentum_component_bicgstab_cellblock_right_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    double relTol,
    double absTol,
    int correctionIters,
    double omega,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);
    correctionIters = std::max(0, correctionIters);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    const bool needBuild = !g.cellMatrixValid;
    gpu_cellblock_upload_problem(g, A, rhs, x);
    if (needBuild) gpu_cellblock_build_inverse(g);

    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;

    const double trueRhsNorm = std::max(gpu_cell_norm(g, g.d_rhs), 1e-300);

    // Right-preconditioned BiCGSTAB uses the true residual as the Krylov residual:
    //   r = b - A x
    //   phat = M^{-1} p
    //   v = A phat
    //   shat = M^{-1} s
    //   t = A shat
    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_r);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_rhat);
    DG2_CUDA_CHECK(cudaGetLastError());

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omegaB = 1.0;

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    if (monitor) {
        std::printf("GPU BiCGSTAB+cellblock-right: maxit=%d relTol=%.3e absTol=%.3e corrIters=%d omega=%.6g trueRel0=%.17e\n",
                    maxit, relTol, absTol, correctionIters, omega, info.finalRelResNorm);
    }

    if ((relTol > 0.0 && info.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && info.finalRelResNorm * trueRhsNorm <= absTol)) {
        gpu_cellblock_download_x(g, x);
        g.lastKrylovTime = wall_seconds() - tSolve0;
        g.lastIterations = info.iterations;
        g.lastRelRes = info.finalRelResNorm;
        g.cumulativeSolves++;
        g.cumulativeKrylovTime += g.lastKrylovTime;
        g.cumulativePrecondTime += g.lastPrecondTime;
        g.cumulativeMatvecTime += g.lastMatvecTime;
        return info;
    }

    for (int it = 1; it <= maxit; ++it) {
        const double rho = gpu_cell_dot(g, g.d_rhat, g.d_r);
        if (!std::isfinite(rho) || std::abs(rho) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock-right breakdown: rho %.17e\n", rho);
            break;
        }

        double beta = 0.0;
        if (it > 1) {
            if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
                if (monitor) std::printf("GPU BiCGSTAB+cellblock-right breakdown before beta: omega %.17e\n", omegaB);
                break;
            }
            beta = (rho / rhoOld) * (alpha / omegaB);
        }

        // p is in true residual space.
        dg2_gpu_p_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_p, g.d_v, beta, omegaB, it == 1);
        DG2_CUDA_CHECK(cudaGetLastError());

        // phat = M^{-1} p
        gpu_cellblock_preconditioner(g, g.d_p, g.d_bhat, correctionIters, omega,
                                      &g.lastPrecondTime, &g.lastMatvecTime);

        // v = A phat
        gpu_cell_csr_spmv(g, g.d_bhat, g.d_v, &g.lastMatvecTime);

        const double denom = gpu_cell_dot(g, g.d_rhat, g.d_v);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock-right breakdown: denom %.17e\n", denom);
            break;
        }

        alpha = rho / denom;

        // s = r - alpha v  in true residual space
        dg2_gpu_s_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_v, g.d_s, alpha);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double sAbs = gpu_cell_norm(g, g.d_s);
        const double sRel = sAbs / trueRhsNorm;

        if ((relTol > 0.0 && sRel <= relTol) || (absTol > 0.0 && sAbs <= absTol)) {
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_bhat, alpha);
            DG2_CUDA_CHECK(cudaGetLastError());

            gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
            const double absTrue = gpu_cell_norm(g, g.d_rTrue);
            const double relTrue = absTrue / trueRhsNorm;

            info.iterations = it;
            info.finalRelResNorm = relTrue;

            if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
                if (monitor) {
                    std::printf("GPU BiCGSTAB+cellblock-right early-s accepted: sRel %.17e trueRel %.17e\n",
                                sRel, relTrue);
                }
                break;
            }

            // Do not accept recurrence-only s convergence. Undo and continue omega step.
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_bhat, -alpha);
            DG2_CUDA_CHECK(cudaGetLastError());

            if (monitor) {
                std::printf("GPU BiCGSTAB+cellblock-right early-s rejected: sRel %.17e trueAbs %.17e trueRel %.17e\n",
                            sRel, absTrue, relTrue);
            }
        }

        // shat = M^{-1} s
        gpu_cellblock_preconditioner(g, g.d_s, g.d_work1, correctionIters, omega,
                                      &g.lastPrecondTime, &g.lastMatvecTime);

        // t = A shat
        gpu_cell_csr_spmv(g, g.d_work1, g.d_t, &g.lastMatvecTime);

        const double tt = gpu_cell_dot(g, g.d_t, g.d_t);
        if (!std::isfinite(tt) || std::abs(tt) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock-right breakdown: tt %.17e\n", tt);
            break;
        }

        omegaB = gpu_cell_dot(g, g.d_t, g.d_s) / tt;
        if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+cellblock-right breakdown: omega %.17e\n", omegaB);
            break;
        }

        // x += alpha phat + omega shat
        // r = s - omega t
        dg2_gpu_x_r_update_right_kernel<<<grid, block>>>(
            g.nRows, g.d_x, g.d_r, g.d_bhat, g.d_s, g.d_work1, g.d_t, alpha, omegaB);
        DG2_CUDA_CHECK(cudaGetLastError());

        // Reliable update: replace recurrence residual with true b-Ax every iteration.
        gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
        dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_r);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double absTrue = gpu_cell_norm(g, g.d_rTrue);
        const double relTrue = absTrue / trueRhsNorm;

        info.iterations = it;
        info.finalRelResNorm = relTrue;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("GPU BiCGSTAB+cellblock-right iter %6d trueAbs %.17e trueRel %.17e sRel %.17e omega %.17e\n",
                        it, absTrue, relTrue, sRel, omegaB);
        }

        if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
            break;
        }

        rhoOld = rho;
    }

    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    gpu_cellblock_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    if (monitor) {
        std::printf("GPU BiCGSTAB+cellblock-right done: iterations=%d relRes=%.17e time=%.6f precond=%.6f matvec=%.6f invBuild=%.6f badBlocks=%d\n",
                    info.iterations, info.finalRelResNorm, g.lastKrylovTime, g.lastPrecondTime,
                    g.lastMatvecTime, g.lastCellBlockBuildTime, g.lastCellBlockBad);
    }

    return info;
}

static HypreSolveInfo solve_momentum_component_bicgstab_colored_cellgs_right_gpu(
    GpuMcgsSystem& g,
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    double relTol,
    double absTol,
    int correctionIters,
    double omega,
    int symmetric,
    int monitor)
{
    (void)pat;
    if ((int)x.size() != g.nRows) x.assign(g.nRows, 0.0);
    correctionIters = std::max(0, correctionIters);

    g.lastKrylovTime = 0.0;
    g.lastPrecondTime = 0.0;
    g.lastMatvecTime = 0.0;
    const double tSolve0 = wall_seconds();

    const bool needBuild = !g.cellMatrixValid;
    gpu_cellblock_upload_problem(g, A, rhs, x);
    if (needBuild) gpu_cellblock_build_inverse(g);

    const int block = 256;
    const int grid = (g.nRows + block - 1) / block;

    const double trueRhsNorm = std::max(gpu_cell_norm(g, g.d_rhs), 1e-300);

    // Right-preconditioned BiCGSTAB uses the true residual as the Krylov residual:
    //   r = b - A x
    //   phat = M^{-1} p
    //   v = A phat
    //   shat = M^{-1} s
    //   t = A shat
    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_r);
    dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_rhat);
    DG2_CUDA_CHECK(cudaGetLastError());

    double rhoOld = 1.0;
    double alpha = 1.0;
    double omegaB = 1.0;

    HypreSolveInfo info;
    info.iterations = 0;
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    if (monitor) {
        std::printf("GPU BiCGSTAB+colored-cellBGS-right: maxit=%d relTol=%.3e absTol=%.3e sweeps=%d omega=%.6g symmetric=%d trueRel0=%.17e\n",
                    maxit, relTol, absTol, correctionIters, omega, symmetric, info.finalRelResNorm);
    }

    if ((relTol > 0.0 && info.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && info.finalRelResNorm * trueRhsNorm <= absTol)) {
        gpu_cellblock_download_x(g, x);
        g.lastKrylovTime = wall_seconds() - tSolve0;
        g.lastIterations = info.iterations;
        g.lastRelRes = info.finalRelResNorm;
        g.cumulativeSolves++;
        g.cumulativeKrylovTime += g.lastKrylovTime;
        g.cumulativePrecondTime += g.lastPrecondTime;
        g.cumulativeMatvecTime += g.lastMatvecTime;
        return info;
    }

    for (int it = 1; it <= maxit; ++it) {
        const double rho = gpu_cell_dot(g, g.d_rhat, g.d_r);
        if (!std::isfinite(rho) || std::abs(rho) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+colored-cellBGS-right breakdown: rho %.17e\n", rho);
            break;
        }

        double beta = 0.0;
        if (it > 1) {
            if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
                if (monitor) std::printf("GPU BiCGSTAB+colored-cellBGS-right breakdown before beta: omega %.17e\n", omegaB);
                break;
            }
            beta = (rho / rhoOld) * (alpha / omegaB);
        }

        // p is in true residual space.
        dg2_gpu_p_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_p, g.d_v, beta, omegaB, it == 1);
        DG2_CUDA_CHECK(cudaGetLastError());

        // phat = M^{-1} p
        gpu_colored_cellgs_preconditioner(g, g.d_p, g.d_bhat, correctionIters, omega,
                                            symmetric, &g.lastPrecondTime);

        // v = A phat
        gpu_cell_csr_spmv(g, g.d_bhat, g.d_v, &g.lastMatvecTime);

        const double denom = gpu_cell_dot(g, g.d_rhat, g.d_v);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+colored-cellBGS-right breakdown: denom %.17e\n", denom);
            break;
        }

        alpha = rho / denom;

        // s = r - alpha v  in true residual space
        dg2_gpu_s_update_kernel<<<grid, block>>>(g.nRows, g.d_r, g.d_v, g.d_s, alpha);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double sAbs = gpu_cell_norm(g, g.d_s);
        const double sRel = sAbs / trueRhsNorm;

        if ((relTol > 0.0 && sRel <= relTol) || (absTol > 0.0 && sAbs <= absTol)) {
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_bhat, alpha);
            DG2_CUDA_CHECK(cudaGetLastError());

            gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
            const double absTrue = gpu_cell_norm(g, g.d_rTrue);
            const double relTrue = absTrue / trueRhsNorm;

            info.iterations = it;
            info.finalRelResNorm = relTrue;

            if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
                if (monitor) {
                    std::printf("GPU BiCGSTAB+colored-cellBGS-right early-s accepted: sRel %.17e trueRel %.17e\n",
                                sRel, relTrue);
                }
                break;
            }

            // Do not accept recurrence-only s convergence. Undo and continue omega step.
            dg2_gpu_x_add_alpha_p_kernel<<<grid, block>>>(g.nRows, g.d_x, g.d_bhat, -alpha);
            DG2_CUDA_CHECK(cudaGetLastError());

            if (monitor) {
                std::printf("GPU BiCGSTAB+colored-cellBGS-right early-s rejected: sRel %.17e trueAbs %.17e trueRel %.17e\n",
                            sRel, absTrue, relTrue);
            }
        }

        // shat = M^{-1} s
        gpu_colored_cellgs_preconditioner(g, g.d_s, g.d_work1, correctionIters, omega,
                                            symmetric, &g.lastPrecondTime);

        // t = A shat
        gpu_cell_csr_spmv(g, g.d_work1, g.d_t, &g.lastMatvecTime);

        const double tt = gpu_cell_dot(g, g.d_t, g.d_t);
        if (!std::isfinite(tt) || std::abs(tt) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+colored-cellBGS-right breakdown: tt %.17e\n", tt);
            break;
        }

        omegaB = gpu_cell_dot(g, g.d_t, g.d_s) / tt;
        if (!std::isfinite(omegaB) || std::abs(omegaB) < 1e-300) {
            if (monitor) std::printf("GPU BiCGSTAB+colored-cellBGS-right breakdown: omega %.17e\n", omegaB);
            break;
        }

        // x += alpha phat + omega shat
        // r = s - omega t
        dg2_gpu_x_r_update_right_kernel<<<grid, block>>>(
            g.nRows, g.d_x, g.d_r, g.d_bhat, g.d_s, g.d_work1, g.d_t, alpha, omegaB);
        DG2_CUDA_CHECK(cudaGetLastError());

        // Reliable update: replace recurrence residual with true b-Ax every iteration.
        gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
        dg2_gpu_copy_kernel<<<grid, block>>>(g.nRows, g.d_rTrue, g.d_r);
        DG2_CUDA_CHECK(cudaGetLastError());

        const double absTrue = gpu_cell_norm(g, g.d_rTrue);
        const double relTrue = absTrue / trueRhsNorm;

        info.iterations = it;
        info.finalRelResNorm = relTrue;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("GPU BiCGSTAB+colored-cellBGS-right iter %6d trueAbs %.17e trueRel %.17e sRel %.17e omega %.17e\n",
                        it, absTrue, relTrue, sRel, omegaB);
        }

        if ((relTol > 0.0 && relTrue <= relTol) || (absTol > 0.0 && absTrue <= absTol)) {
            break;
        }

        rhoOld = rho;
    }

    gpu_cell_residual(g, g.d_x, g.d_rTrue, &g.lastMatvecTime);
    info.finalRelResNorm = gpu_cell_norm(g, g.d_rTrue) / trueRhsNorm;

    gpu_cellblock_download_x(g, x);

    g.lastKrylovTime = wall_seconds() - tSolve0;
    g.lastIterations = info.iterations;
    g.lastRelRes = info.finalRelResNorm;
    g.cumulativeSolves++;
    g.cumulativeKrylovIts += info.iterations;
    g.cumulativeKrylovTime += g.lastKrylovTime;
    g.cumulativePrecondTime += g.lastPrecondTime;
    g.cumulativeMatvecTime += g.lastMatvecTime;

    if (monitor) {
        std::printf("GPU BiCGSTAB+colored-cellBGS-right done: iterations=%d relRes=%.17e time=%.6f precond=%.6f matvec=%.6f invBuild=%.6f badBlocks=%d\n",
                    info.iterations, info.finalRelResNorm, g.lastKrylovTime, g.lastPrecondTime,
                    g.lastMatvecTime, g.lastCellBlockBuildTime, g.lastCellBlockBad);
    }

    return info;
}


static HypreSolveInfo solve_velocity_dispatch(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt,
    const std::string& uSolver,
    const std::string& leftScaleMode,
    double leftScaleEps,
    const CpuMcgsColoring& coloring,
    GpuMcgsSystem* gpuMcgs,
    int uSweeps,
    int uMcgsPreSweeps,
    double uOmega,
    int uSymmetric,
    int uCellBlockIters,
    double uCellBlockOmega,
    int monitor);

struct InexactSchurStats {
    long long pressureMatvecs = 0;
    long long innerSolves = 0;
    long long innerIters = 0;
    double innerWorstRel = 0.0;
    double applyTime = 0.0;
    double gmresTime = 0.0;
};

static void apply_inexact_schur_action(
    const CSRPattern& uPat,
    const std::vector<double>& Arel,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<double>& qIn,
    std::vector<double>& y,
    HypreOptions innerOpt,
    const std::string& innerSolver,
    const std::string& leftScaleMode,
    double leftScaleEps,
    const CpuMcgsColoring& coloring,
    GpuMcgsSystem* gpuMcgs,
    int uSweeps,
    int uMcgsPreSweeps,
    double uOmega,
    int uSymmetric,
    int uCellBlockIters,
    double uCellBlockOmega,
    const std::vector<double>& pWeights,
    int monitor,
    InexactSchurStats* stats)
{
    const double t0 = wall_seconds();

    std::vector<double> q = qIn;
    subtract_weighted_mean(q, pWeights);

    y.assign(Apx.nCols, 0.0);

    auto add_direction = [&](const RectCSR& Ap, const char* label) {
        std::vector<double> g;
        std::vector<double> z(Ap.nRows, 0.0);
        std::vector<double> tmp;

        apply_rect(Ap, q, g);

        HypreOptions optDir = innerOpt;
        optDir.profileLabel = std::string("inexactSchur-inner-") + label;

        HypreSolveInfo info = solve_velocity_dispatch(
            uPat, Arel, g, z,
            optDir,
            innerSolver,
            leftScaleMode,
            leftScaleEps,
            coloring,
            gpuMcgs,
            uSweeps,
            uMcgsPreSweeps,
            uOmega,
            uSymmetric,
            uCellBlockIters,
            uCellBlockOmega,
            monitor);

        apply_pos_transpose_rect(Ap, z, tmp);
        axpy_vec(y, 1.0, tmp);

        if (stats) {
            stats->innerSolves++;
            stats->innerIters += info.iterations;
            stats->innerWorstRel = std::max(stats->innerWorstRel, info.finalRelResNorm);
        }
    };

    add_direction(Apx, "x");
    add_direction(Apy, "y");
    add_direction(Apz, "z");

    subtract_weighted_mean(y, pWeights);

    if (stats) {
        stats->pressureMatvecs++;
        stats->applyTime += wall_seconds() - t0;
    }
}

static double apply_inexact_schur_velocity_correction(
    const CSRPattern& uPat,
    const std::vector<double>& Arel,
    const RectCSR& Ap,
    const std::vector<double>& pcorr,
    std::vector<double>& u,
    double scale,
    HypreOptions innerOpt,
    const std::string& innerSolver,
    const std::string& leftScaleMode,
    double leftScaleEps,
    const CpuMcgsColoring& coloring,
    GpuMcgsSystem* gpuMcgs,
    int uSweeps,
    int uMcgsPreSweeps,
    double uOmega,
    int uSymmetric,
    int uCellBlockIters,
    double uCellBlockOmega,
    int monitor,
    InexactSchurStats* stats)
{
    std::vector<double> g;
    std::vector<double> z(Ap.nRows, 0.0);

    apply_rect(Ap, pcorr, g);

    HypreSolveInfo info = solve_velocity_dispatch(
        uPat, Arel, g, z,
        innerOpt,
        innerSolver,
        leftScaleMode,
        leftScaleEps,
        coloring,
        gpuMcgs,
        uSweeps,
        uMcgsPreSweeps,
        uOmega,
        uSymmetric,
        uCellBlockIters,
        uCellBlockOmega,
        monitor);

    if (stats) {
        stats->innerSolves++;
        stats->innerIters += info.iterations;
        stats->innerWorstRel = std::max(stats->innerWorstRel, info.finalRelResNorm);
    }

    double n2 = 0.0;
    for (std::size_t i=0; i<u.size(); ++i) {
        const double du = -scale * z[i];
        u[i] += du;
        n2 += du * du;
    }

    return std::sqrt(std::max(0.0, n2));
}



static HypreSolveInfo solve_pressure_fgmres_split_hypre_prec(
    const CSRPattern& aPat,
    const std::vector<double>& aVals,
    const CSRPattern& precPat,
    const std::vector<double>& precVals,
    HypreReusableSystem* precSys,
    int useReusablePrec,
    const HypreOptions& precOptBase,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    int restart,
    double relTol,
    double absTol,
    int monitor)
{
    const int n = (int)rhs.size();
    if ((int)x.size() != n) x.assign(n, 0.0);

    restart = std::max(2, std::min(restart, maxit));

    HypreSolveInfo out;
    out.iterations = 0;
    out.finalRelResNorm = 1.0;

    const double bnorm = std::max(norm_vec(rhs), 1e-300);

    auto apply_prec = [&](const std::vector<double>& rr,
                          std::vector<double>& zz,
                          int outerIt) -> HypreSolveInfo {
        zz.assign(n, 0.0);

        HypreOptions opt = precOptBase;
        if (opt.profile) {
            opt.profileLabel = "splitFgmres-unmaskedAMG outer=" + std::to_string(outerIt);
        }

        if (useReusablePrec && precSys != nullptr) {
            return solve_reusable_hypre_rhs_vec(*precSys, rr, zz, opt);
        }

        return solve_hypre_csr_vec(precPat, precVals, rr, zz, opt);
    };

    auto update_solution = [&](int k,
                               const std::vector<std::vector<double>>& H,
                               const std::vector<double>& g,
                               const std::vector<std::vector<double>>& Z) {
        std::vector<double> y(k, 0.0);

        for (int i=k-1; i>=0; --i) {
            double s = g[i];
            for (int j=i+1; j<k; ++j) s -= H[i][j] * y[j];
            y[i] = s / H[i][i];
        }

        for (int j=0; j<k; ++j) {
            for (int i=0; i<n; ++i) {
                x[i] += Z[j][i] * y[j];
            }
        }
    };

    long long innerSolves = 0;
    long long innerIts = 0;

    while (out.iterations < maxit) {
        std::vector<double> Ax;
        apply_csr(aPat, aVals, x, Ax);

        std::vector<double> r = rhs;
        axpy_vec(r, -1.0, Ax);

        double beta = norm_vec(r);
        out.finalRelResNorm = beta / bnorm;

        if (monitor) {
            std::printf("splitFgmresMassSchur restart-begin outerIts=%d rel=%.6e abs=%.6e restart=%d reusableInner=%d\n",
                        out.iterations, out.finalRelResNorm, beta, restart, useReusablePrec);
        }

        if ((relTol > 0.0 && out.finalRelResNorm <= relTol) ||
            (absTol > 0.0 && beta <= absTol)) {
            return out;
        }

        std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> Z(restart,     std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));

        std::vector<double> cs(restart, 0.0);
        std::vector<double> sn(restart, 0.0);
        std::vector<double> g(restart + 1, 0.0);

        for (int i=0; i<n; ++i) V[0][i] = r[i] / beta;
        g[0] = beta;

        int used = 0;

        for (int j=0; j<restart && out.iterations < maxit; ++j) {
            HypreSolveInfo pi = apply_prec(V[j], Z[j], out.iterations);
            innerSolves++;
            innerIts += pi.iterations;

            std::vector<double> w;
            apply_csr(aPat, aVals, Z[j], w);

            for (int i=0; i<=j; ++i) {
                H[i][j] = dot_vec(w, V[i]);
                for (int k=0; k<n; ++k) {
                    w[k] -= H[i][j] * V[i][k];
                }
            }

            H[j+1][j] = norm_vec(w);

            if (H[j+1][j] > 1e-300) {
                for (int k=0; k<n; ++k) V[j+1][k] = w[k] / H[j+1][j];
            }

            for (int i=0; i<j; ++i) {
                const double hij  = H[i][j];
                const double hipj = H[i+1][j];
                H[i][j]   =  cs[i] * hij + sn[i] * hipj;
                H[i+1][j] = -sn[i] * hij + cs[i] * hipj;
            }

            const double h1 = H[j][j];
            const double h2 = H[j+1][j];
            const double denom = std::hypot(h1, h2);

            if (denom <= 1e-300 || !std::isfinite(denom)) {
                if (monitor) std::printf("splitFgmresMassSchur breakdown at outerIts=%d j=%d denom=%.17e\n",
                                         out.iterations, j, denom);
                used = j;
                break;
            }

            cs[j] = h1 / denom;
            sn[j] = h2 / denom;

            H[j][j]   =  cs[j] * h1 + sn[j] * h2;
            H[j+1][j] =  0.0;

            const double gj  = g[j];
            const double gj1 = g[j+1];
            g[j]   =  cs[j] * gj + sn[j] * gj1;
            g[j+1] = -sn[j] * gj + cs[j] * gj1;

            out.iterations++;
            used = j + 1;

            const double absRes = std::abs(g[j+1]);
            out.finalRelResNorm = absRes / bnorm;

            if (monitor && (out.iterations <= 10 || (out.iterations % 10) == 0)) {
                std::printf("splitFgmresMassSchur iter %6d rel %.6e abs %.6e innerSolves=%lld innerIts=%lld\n",
                            out.iterations, out.finalRelResNorm, absRes, innerSolves, innerIts);
            }

            if ((relTol > 0.0 && out.finalRelResNorm <= relTol) ||
                (absTol > 0.0 && absRes <= absTol)) {
                update_solution(used, H, g, Z);

                if (monitor) {
                    std::printf("splitFgmresMassSchur converged: outerIts=%d finalRel=%.6e innerSolves=%lld innerIts=%lld\n",
                                out.iterations, out.finalRelResNorm, innerSolves, innerIts);
                }

                return out;
            }
        }

        if (used <= 0) break;

        update_solution(used, H, g, Z);
    }

    if (monitor) {
        std::printf("splitFgmresMassSchur done: outerIts=%d finalRel=%.6e innerSolves=%lld innerIts=%lld\n",
                    out.iterations, out.finalRelResNorm, innerSolves, innerIts);
    }

    return out;
}


static HypreSolveInfo solve_pressure_pcg_split_hypre_prec(
    const CSRPattern& aPat,
    const std::vector<double>& aVals,
    const CSRPattern& precPat,
    const std::vector<double>& precVals,
    HypreReusableSystem* precSys,
    int useReusablePrec,
    const HypreOptions& precOptBase,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    int maxit,
    double relTol,
    double absTol,
    int monitor)
{
    const int n = (int)rhs.size();
    if ((int)x.size() != n) x.assign(n, 0.0);

    HypreSolveInfo out;
    out.iterations = 0;
    out.finalRelResNorm = 1.0;

    std::vector<double> Ax;
    apply_csr(aPat, aVals, x, Ax);

    std::vector<double> r = rhs;
    axpy_vec(r, -1.0, Ax);

    const double bnorm = std::max(norm_vec(rhs), 1e-300);
    double rnorm = norm_vec(r);
    out.finalRelResNorm = rnorm / bnorm;

    if (monitor) {
        std::printf("splitPcgMassSchur: maxit=%d relTol=%.3e absTol=%.3e rel0=%.6e reusableInner=%d\n",
                    maxit, relTol, absTol, out.finalRelResNorm, useReusablePrec);
    }

    if ((relTol > 0.0 && out.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && rnorm <= absTol)) {
        return out;
    }

    std::vector<double> z(n, 0.0), p(n, 0.0), Ap(n, 0.0);

    auto apply_prec = [&](const std::vector<double>& rr,
                          std::vector<double>& zz,
                          int outerIt) -> HypreSolveInfo {
        zz.assign(n, 0.0);

        HypreOptions opt = precOptBase;
        if (opt.profile) {
            opt.profileLabel = "splitPcg-unmaskedAMG outer=" + std::to_string(outerIt);
        }

        if (useReusablePrec && precSys != nullptr) {
            return solve_reusable_hypre_rhs_vec(*precSys, rr, zz, opt);
        }

        return solve_hypre_csr_vec(precPat, precVals, rr, zz, opt);
    };

    long long innerSolves = 0;
    long long innerIts = 0;

    HypreSolveInfo pi = apply_prec(r, z, 0);
    innerSolves++;
    innerIts += pi.iterations;

    double rz = dot_vec(r, z);
    if (!std::isfinite(rz) || std::abs(rz) <= 1e-300) {
        if (monitor) std::printf("splitPcgMassSchur: initial rz breakdown %.17e\n", rz);
        return out;
    }

    p = z;

    for (int it=1; it<=maxit; ++it) {
        apply_csr(aPat, aVals, p, Ap);

        const double pAp = dot_vec(p, Ap);
        if (!std::isfinite(pAp) || std::abs(pAp) <= 1e-300) {
            if (monitor) std::printf("splitPcgMassSchur: pAp breakdown at it=%d pAp=%.17e\n", it, pAp);
            break;
        }

        const double alpha = rz / pAp;

        for (int i=0; i<n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        rnorm = norm_vec(r);
        out.iterations = it;
        out.finalRelResNorm = rnorm / bnorm;

        if (monitor && (it <= 10 || (it % 10) == 0)) {
            std::printf("splitPcgMassSchur iter %6d rel %.6e abs %.6e innerSolves=%lld innerIts=%lld\n",
                        it, out.finalRelResNorm, rnorm, innerSolves, innerIts);
        }

        if ((relTol > 0.0 && out.finalRelResNorm <= relTol) ||
            (absTol > 0.0 && rnorm <= absTol)) {
            break;
        }

        pi = apply_prec(r, z, it);
        innerSolves++;
        innerIts += pi.iterations;

        const double rzNew = dot_vec(r, z);
        if (!std::isfinite(rzNew) || std::abs(rzNew) <= 1e-300) {
            if (monitor) std::printf("splitPcgMassSchur: rz breakdown at it=%d rz=%.17e\n", it, rzNew);
            break;
        }

        const double beta = rzNew / rz;
        for (int i=0; i<n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rzNew;
    }

    if (monitor) {
        std::printf("splitPcgMassSchur done: outerIts=%d finalRel=%.6e innerSolves=%lld innerIts=%lld\n",
                    out.iterations, out.finalRelResNorm, innerSolves, innerIts);
    }

    return out;
}


static HypreSolveInfo solve_pressure_inexact_schur_gmres(
    const CSRPattern& uPat,
    const std::vector<double>& Arel,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const std::vector<double>& pWeights,
    HypreOptions innerOpt,
    const std::string& innerSolver,
    const std::string& leftScaleMode,
    double leftScaleEps,
    const CpuMcgsColoring& coloring,
    GpuMcgsSystem* gpuMcgs,
    int uSweeps,
    int uMcgsPreSweeps,
    double uOmega,
    int uSymmetric,
    int uCellBlockIters,
    double uCellBlockOmega,
    int maxit,
    int restart,
    double relTol,
    double absTol,
    int monitor,
    InexactSchurStats* stats)
{
    const double t0 = wall_seconds();

    const int n = (int)rhs.size();
    restart = std::max(1, std::min(restart, maxit));
    if ((int)x.size() != n) x.assign(n, 0.0);
    subtract_weighted_mean(x, pWeights);

    std::vector<double> b = rhs;
    subtract_weighted_mean(b, pWeights);

    const double bnorm = std::max(norm_vec(b), 1e-300);

    HypreSolveInfo out;
    out.iterations = 0;
    out.finalRelResNorm = 1.0;

    std::vector<double> Ax;
    apply_inexact_schur_action(
        uPat, Arel, Apx, Apy, Apz, x, Ax,
        innerOpt, innerSolver, leftScaleMode, leftScaleEps,
        coloring, gpuMcgs,
        uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
        uCellBlockIters, uCellBlockOmega,
        pWeights, monitor, stats);

    std::vector<double> r = b;
    axpy_vec(r, -1.0, Ax);
    subtract_weighted_mean(r, pWeights);

    double beta = norm_vec(r);
    out.finalRelResNorm = beta / bnorm;

    if (monitor) {
        std::printf("inexactSchur GMRES: maxit=%d restart=%d relTol=%.3e absTol=%.3e rel0=%.6e innerSolver=%s\n",
            maxit, restart, relTol, absTol, out.finalRelResNorm, innerSolver.c_str());
    }

    if ((relTol > 0.0 && out.finalRelResNorm <= relTol) ||
        (absTol > 0.0 && beta <= absTol)) {
        if (stats) stats->gmresTime += wall_seconds() - t0;
        return out;
    }

    int totalIts = 0;

    while (totalIts < maxit) {
        beta = norm_vec(r);
        if (beta <= 1e-300) break;

        std::vector<std::vector<double>> V(restart + 1, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> H(restart + 1, std::vector<double>(restart, 0.0));
        std::vector<double> cs(restart, 0.0), sn(restart, 0.0), g(restart + 1, 0.0);

        for (int i=0; i<n; ++i) V[0][i] = r[i] / beta;
        g[0] = beta;

        int jDone = 0;
        bool converged = false;

        for (int j=0; j<restart && totalIts < maxit; ++j) {
            std::vector<double> w;

            apply_inexact_schur_action(
                uPat, Arel, Apx, Apy, Apz, V[j], w,
                innerOpt, innerSolver, leftScaleMode, leftScaleEps,
                coloring, gpuMcgs,
                uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
                uCellBlockIters, uCellBlockOmega,
                pWeights, monitor, stats);

            for (int i=0; i<=j; ++i) {
                H[i][j] = dot_vec(w, V[i]);
                for (int k=0; k<n; ++k) w[k] -= H[i][j] * V[i][k];
            }

            H[j+1][j] = norm_vec(w);
            if (H[j+1][j] > 1e-300) {
                for (int k=0; k<n; ++k) V[j+1][k] = w[k] / H[j+1][j];
            }

            for (int i=0; i<j; ++i) {
                const double hij = H[i][j];
                const double hip = H[i+1][j];
                H[i][j]   =  cs[i] * hij + sn[i] * hip;
                H[i+1][j] = -sn[i] * hij + cs[i] * hip;
            }

            const double h1 = H[j][j];
            const double h2 = H[j+1][j];
            const double rho = std::hypot(h1, h2);

            if (rho <= 1e-300) {
                cs[j] = 1.0;
                sn[j] = 0.0;
            } else {
                cs[j] = h1 / rho;
                sn[j] = h2 / rho;
            }

            H[j][j] = cs[j] * h1 + sn[j] * h2;
            H[j+1][j] = 0.0;

            const double gj = g[j];
            g[j]   =  cs[j] * gj;
            g[j+1] = -sn[j] * gj;

            totalIts++;
            jDone = j + 1;

            const double rel = std::abs(g[j+1]) / bnorm;

            if (monitor) {
                std::printf("inexactSchur GMRES iter %4d rel=%.6e innerSolves=%lld innerIts=%lld worstInner=%.3e\n",
                    totalIts, rel,
                    stats ? stats->innerSolves : 0,
                    stats ? stats->innerIters : 0,
                    stats ? stats->innerWorstRel : 0.0);
            }

            if ((relTol > 0.0 && rel <= relTol) ||
                (absTol > 0.0 && std::abs(g[j+1]) <= absTol)) {
                converged = true;
                break;
            }
        }

        std::vector<double> ycoef(jDone, 0.0);
        for (int i=jDone-1; i>=0; --i) {
            double s = g[i];
            for (int k=i+1; k<jDone; ++k) s -= H[i][k] * ycoef[k];
            ycoef[i] = s / std::max(std::abs(H[i][i]), 1e-300);
            if (H[i][i] < 0.0) ycoef[i] = -ycoef[i];
        }

        for (int i=0; i<jDone; ++i) {
            for (int k=0; k<n; ++k) x[k] += V[i][k] * ycoef[i];
        }

        subtract_weighted_mean(x, pWeights);

        apply_inexact_schur_action(
            uPat, Arel, Apx, Apy, Apz, x, Ax,
            innerOpt, innerSolver, leftScaleMode, leftScaleEps,
            coloring, gpuMcgs,
            uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
            uCellBlockIters, uCellBlockOmega,
            pWeights, monitor, stats);

        r = b;
        axpy_vec(r, -1.0, Ax);
        subtract_weighted_mean(r, pWeights);

        const double rnorm = norm_vec(r);
        out.iterations = totalIts;
        out.finalRelResNorm = rnorm / bnorm;

        if (converged ||
            (relTol > 0.0 && out.finalRelResNorm <= relTol) ||
            (absTol > 0.0 && rnorm <= absTol)) {
            break;
        }

        if (jDone == 0) break;
    }

    if (stats) stats->gmresTime += wall_seconds() - t0;
    return out;
}


static HypreSolveInfo solve_velocity_csr_vec(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    HypreOptions opt,
    const std::string& leftScaleMode,
    double leftScaleEps);

static HypreSolveInfo solve_velocity_dispatch(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    const HypreOptions& opt,
    const std::string& uSolver,
    const std::string& leftScaleMode,
    double leftScaleEps,
    const CpuMcgsColoring& coloring,
    GpuMcgsSystem* gpuMcgs,
    int uSweeps,
    int uMcgsPreSweeps,
    double uOmega,
    int uSymmetric,
    int uCellBlockIters,
    double uCellBlockOmega,
    int monitor)
{
    if (uSolver == "celljacobi_gpu" || uSolver == "cellblock_gpu" || uSolver == "blockjacobi_gpu") {
        if (!gpuMcgs || !gpuMcgs->ready || !gpuMcgs->cellBlockMode) throw std::runtime_error("uSolver=celljacobi_gpu requested but GPU cell-block system is not initialized");
        return solve_momentum_component_celljacobi_gpu(
            *gpuMcgs, pat, A, rhs, x,
            uSweeps,
            uCellBlockOmega,
            uCellBlockIters,
            opt.relTol,
            opt.absTol,
            monitor);
    }

    if (uSolver == "bicgstab_colored_cellgs_gpu" || uSolver == "bicgstab-colored-cellgs-gpu" ||
        uSolver == "bicgstab_cellmcgs_gpu" || uSolver == "bicgstab-cellmcgs-gpu" ||
        uSolver == "bicgstab_blockgs_gpu" || uSolver == "bicgstab-blockgs-gpu") {
        if (!gpuMcgs || !gpuMcgs->ready || !gpuMcgs->cellBlockMode) {
            throw std::runtime_error("uSolver=bicgstab_colored_cellgs_gpu requested but GPU colored cell-block system is not initialized");
        }

        return solve_momentum_component_bicgstab_colored_cellgs_right_gpu(
            *gpuMcgs, pat, A, rhs, x,
            opt.maxIter,
            opt.relTol,
            opt.absTol,
            std::max(1, uCellBlockIters),
            uCellBlockOmega,
            uSymmetric,
            monitor);
    }

    if (uSolver == "bicgstab_celljacobi_right_gpu" || uSolver == "bicgstab-celljacobi-right-gpu" ||
        uSolver == "bicgstab_cellblock_right_gpu" || uSolver == "bicgstab-cellblock-right-gpu") {
        if (!gpuMcgs || !gpuMcgs->ready || !gpuMcgs->cellBlockMode) throw std::runtime_error("uSolver=bicgstab_cellblock_right_gpu requested but GPU cell-block system is not initialized");
        return solve_momentum_component_bicgstab_cellblock_right_gpu(
            *gpuMcgs, pat, A, rhs, x,
            opt.maxIter,
            opt.relTol,
            opt.absTol,
            uCellBlockIters,
            uCellBlockOmega,
            monitor);
    }

    if (uSolver == "bicgstab_celljacobi_gpu" || uSolver == "bicgstab-celljacobi-gpu" ||
        uSolver == "bicgstab_cellblock_gpu" || uSolver == "bicgstab-cellblock-gpu" ||
        uSolver == "bicgstab_cellasm_gpu" || uSolver == "bicgstab-cellasm-gpu") {
        if (!gpuMcgs || !gpuMcgs->ready || !gpuMcgs->cellBlockMode) throw std::runtime_error("uSolver=bicgstab_cellblock_gpu requested but GPU cell-block system is not initialized");
        return solve_momentum_component_bicgstab_cellblock_left_gpu(
            *gpuMcgs, pat, A, rhs, x,
            opt.maxIter,
            opt.relTol,
            opt.absTol,
            uCellBlockIters,
            uCellBlockOmega,
            monitor);
    }

    if (uSolver == "mcgs_gpu" || uSolver == "mcgs-gpu") {
        if (!gpuMcgs || !gpuMcgs->ready) throw std::runtime_error("uSolver=mcgs_gpu requested but GPU MCGS system is not initialized");
        return solve_momentum_component_mcgs_gpu(
            *gpuMcgs, pat, A, rhs, x,
            uSweeps,
            uOmega,
            uSymmetric,
            opt.relTol,
            opt.absTol,
            monitor);
    }

    if (uSolver == "bicgstab_mcgs_gpu" || uSolver == "bicgstab-mcgs-gpu" ||
        uSolver == "bicgstab+mcgs_gpu" || uSolver == "bicgstab+mcgs-gpu") {
        if (!gpuMcgs || !gpuMcgs->ready) throw std::runtime_error("uSolver=bicgstab_mcgs_gpu requested but GPU MCGS system is not initialized");
        return solve_momentum_component_bicgstab_mcgs_left_gpu(
            *gpuMcgs, pat, A, rhs, x,
            opt.maxIter,
            opt.relTol,
            opt.absTol,
            uMcgsPreSweeps,
            uOmega,
            uSymmetric,
            monitor);
    }

    if (uSolver == "mcgs") {
        return solve_momentum_component_mcgs_cpu(
            pat, A, rhs, x,
            coloring,
            uSweeps,
            uOmega,
            uSymmetric,
            opt.relTol,
            opt.absTol,
            monitor);
    }

    if (uSolver == "bicgstab_mcgs") {
        return solve_momentum_component_bicgstab_mcgs_left_cpu(
            pat, A, rhs, x,
            coloring,
            opt.maxIter,
            opt.relTol,
            opt.absTol,
            uMcgsPreSweeps,
            uOmega,
            uSymmetric,
            monitor);
    }

    return solve_velocity_csr_vec(pat, A, rhs, x, opt, leftScaleMode, leftScaleEps);
}


static HypreSolveInfo solve_velocity_csr_vec(
    const CSRPattern& pat,
    const std::vector<double>& A,
    const std::vector<double>& rhs,
    std::vector<double>& x,
    HypreOptions opt,
    const std::string& leftScaleMode,
    double leftScaleEps)
{
    HypreSolveInfo info;

    if (leftScaleMode == "none") {
        info = solve_hypre_csr_vec(pat, A, rhs, x, opt);
    } else {
        std::vector<double> As;
        std::vector<double> bs;

        if (leftScaleMode == "cellblock" || leftScaleMode == "blockjacobi") {
            left_precondition_csr_cellblock_same_pattern(
                pat, A, rhs,
                gVelocityBlockJacobiShift,
                gVelocityBlockJacobiPivotFloor,
                As, bs);
        } else {
            left_scale_csr_rows(pat, A, rhs, leftScaleMode, leftScaleEps, As, bs);
        }

        // The custom left scaling itself is the preconditioner.
        // Avoid stacking HYPRE diagscale on top unless deliberately tested later.
        opt.precond = "none";

        info = solve_hypre_csr_vec(pat, As, bs, x, opt);
    }

    // Report original unscaled residual, not the residual of the scaled system.
    std::vector<double> Ax;
    apply_csr(pat, A, x, Ax);
    axpy_vec(Ax, -1.0, rhs);

    const double r = norm_vec(Ax);
    const double b = std::max(norm_vec(rhs), 1e-300);

    info.finalRelResNorm = r / b;

    return info;
}




struct DgPatchForceReport {
    bool requested = false;
    bool valid = false;
    int patchIndex = -1;
    std::string patchName;
    int nFaces = 0;
    double area = 0.0;
    double rho = 1.0;
    double mu = 0.0;
    double Uref = 1.0;
    double Aref = 1.0;
    double coeffDenom = 0.0;
    int normalSign = -1;

    std::array<double,3> dragDir{{1.0,0.0,0.0}};
    std::array<double,3> liftDir{{0.0,0.0,1.0}};
    std::array<double,3> spanDir{{0.0,1.0,0.0}};

    std::array<double,3> Fp{{0.0,0.0,0.0}};
    std::array<double,3> Fv{{0.0,0.0,0.0}};
    std::array<double,3> F{{0.0,0.0,0.0}};

    double FpDrag = 0.0, FvDrag = 0.0, FDrag = 0.0;
    double FpLift = 0.0, FvLift = 0.0, FLift = 0.0;
    double FpSpan = 0.0, FvSpan = 0.0, FSpan = 0.0;

    double CDrag = 0.0;
    double CLift = 0.0;
    double CSpan = 0.0;

    double minWallDistance = std::numeric_limits<double>::infinity();
    double maxWallDistance = 0.0;
    double maxAbsPressure = 0.0;
    double maxAbsShearTraction = 0.0;
    int mapFail = 0;
};

static std::array<double,3> dg_normalized_vec3(
    std::array<double,3> v,
    const std::array<double,3>& fallback)
{
    const double n = norm3(v);
    if (!std::isfinite(n) || n <= 1.0e-300) return fallback;
    return mul3(1.0/n, v);
}

static int dg_find_patch_index_by_name(const Mesh& mesh, const std::string& patchName)
{
    if (patchName.empty()) return -1;
    for (std::size_t i=0; i<mesh.patchNames.size(); ++i) {
        if (mesh.patchNames[i] == patchName) return (int)i;
    }
    return -1;
}

static double dg_dg1_pressure_at_lam(
    int cell,
    const std::array<double,4>& lam,
    const std::vector<double>& p)
{
    return dg1_pressure_at_lam_cell(cell, lam, p);
}

static double dg_cg1_pressure_at_lam(
    const TetP2Geom& K,
    const std::array<double,4>& lam,
    const std::vector<double>& p)
{
    double ph = 0.0;
    for (int a=0; a<4; ++a) {
        const int gv = K.v[a];
        if (gv >= 0 && gv < (int)p.size()) ph += lam[a] * p[gv];
    }
    return ph;
}

static DgPatchForceReport compute_dg2_dg1_patch_forces_stress(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::vector<QuadTriPoint>& fq,
    const std::string& patchName,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    const std::vector<double>& p,
    double rho,
    double mu,
    int normalSign,
    double Uref,
    double Aref,
    std::array<double,3> dragDir,
    std::array<double,3> liftDir,
    std::array<double,3> spanDir)
{
    DgPatchForceReport r;
    r.requested = true;
    r.patchName = patchName;
    r.rho = rho;
    r.mu = mu;
    r.Uref = Uref;
    r.Aref = Aref;
    r.normalSign = normalSign;
    r.dragDir = dg_normalized_vec3(dragDir, {{1.0,0.0,0.0}});
    r.liftDir = dg_normalized_vec3(liftDir, {{0.0,0.0,1.0}});
    r.spanDir = dg_normalized_vec3(spanDir, {{0.0,1.0,0.0}});
    r.coeffDenom = rho * Uref * Uref * Aref;

    const int patchIndex = dg_find_patch_index_by_name(mesh, patchName);
    r.patchIndex = patchIndex;

    if (patchIndex < 0) return r;
    if (patchIndex >= (int)mesh.patchStartFace.size()) return r;
    if (patchIndex >= (int)mesh.patchNFaces.size()) return r;
    if (Uref <= 0.0 || Aref <= 0.0 || r.coeffDenom <= 1.0e-300) return r;

    const int f0 = mesh.patchStartFace[patchIndex];
    const int f1 = f0 + mesh.patchNFaces[patchIndex];

    if (f0 < mesh.nInternalFaces || f1 > mesh.nFaces || f1 < f0) return r;

    r.valid = true;

    for (int f=f0; f<f1; ++f) {
        if (f < 0 || f >= mesh.nFaces || f >= (int)mesh.owner.size()) continue;

        const int c = mesh.owner[f];
        if (c < 0 || c >= (int)tets.size()) continue;

        const double A = mesh.Af[f];
        if (!(A > 1.0e-300)) continue;

        // mesh.nf is the fluid-domain outward normal.
        // For an internal cylinder/obstacle patch, use normalSign=-1 so that
        // n points from solid into fluid, matching the old FV force convention.
        std::array<double,3> n{{
            (double)normalSign * mesh.nf[f][0],
            (double)normalSign * mesh.nf[f][1],
            (double)normalSign * mesh.nf[f][2]}};

        n = dg_normalized_vec3(n, {{1.0,0.0,0.0}});

        const TetP2Geom& K = tets[c];

        for (const auto& q : fq) {
            std::array<double,4> lam{};
            if (!face_lam_on_tet(mesh.faces[f], K, q.mu, lam)) {
                r.mapFail++;
                continue;
            }

            double N[10];
            std::array<double,3> G[10];

            p2_tet_basis(lam, N);
            p2_tet_grad(K, lam, G);

            // gradU[velocity component][physical direction]
            double gradU[3][3] = {
                {0.0,0.0,0.0},
                {0.0,0.0,0.0},
                {0.0,0.0,0.0}
            };

            for (int i=0; i<10; ++i) {
                const int row = 10*c + i;

                const double uci[3] = {
                    (row < (int)ux.size() ? ux[row] : 0.0),
                    (row < (int)uy.size() ? uy[row] : 0.0),
                    (row < (int)uz.size() ? uz[row] : 0.0)
                };

                for (int d=0; d<3; ++d) {
                    gradU[0][d] += uci[0] * G[i][d];
                    gradU[1][d] += uci[1] * G[i][d];
                    gradU[2][d] += uci[2] * G[i][d];
                }
            }

            const double ph = dg_dg1_pressure_at_lam(c, lam, p);
            const double wA = q.w * A;

            double viscTraction[3] = {0.0,0.0,0.0};

            for (int d=0; d<3; ++d) {
                for (int e=0; e<3; ++e) {
                    viscTraction[d] += mu * (gradU[d][e] + gradU[e][d]) * n[e];
                }
            }

            const double shearMag = std::sqrt(
                viscTraction[0]*viscTraction[0] +
                viscTraction[1]*viscTraction[1] +
                viscTraction[2]*viscTraction[2]);

            for (int d=0; d<3; ++d) {
                const double fp = -ph * n[d] * wA;
                const double fv = viscTraction[d] * wA;

                r.Fp[d] += fp;
                r.Fv[d] += fv;
                r.F[d]  += fp + fv;
            }

            r.maxAbsPressure = std::max(r.maxAbsPressure, std::abs(ph));
            r.maxAbsShearTraction = std::max(r.maxAbsShearTraction, shearMag);
        }

        const std::array<double,3> dx = sub3(mesh.cc[c], mesh.xf[f]);
        double dn = std::abs(dot3(dx, n));

        if (dn <= 1.0e-14) dn = norm3(dx);

        if (dn > 1.0e-14) {
            r.minWallDistance = std::min(r.minWallDistance, dn);
            r.maxWallDistance = std::max(r.maxWallDistance, dn);
        }

        r.area += A;
        r.nFaces += 1;
    }

    r.FpDrag = dot3(r.Fp, r.dragDir);
    r.FvDrag = dot3(r.Fv, r.dragDir);
    r.FDrag  = dot3(r.F,  r.dragDir);

    r.FpLift = dot3(r.Fp, r.liftDir);
    r.FvLift = dot3(r.Fv, r.liftDir);
    r.FLift  = dot3(r.F,  r.liftDir);

    r.FpSpan = dot3(r.Fp, r.spanDir);
    r.FvSpan = dot3(r.Fv, r.spanDir);
    r.FSpan  = dot3(r.F,  r.spanDir);

    if (r.coeffDenom > 1.0e-300) {
        r.CDrag = 2.0 * r.FDrag / r.coeffDenom;
        r.CLift = 2.0 * r.FLift / r.coeffDenom;
        r.CSpan = 2.0 * r.FSpan / r.coeffDenom;
    }

    return r;
}

static void print_dg_patch_force_report(
    const DgPatchForceReport& r,
    int it,
    const char* stage)
{
    if (!r.requested) return;

    if (!r.valid) {
        std::printf("    dgPatchForce: it=%d stage=%s INVALID patch=%s patchIndex=%d Uref=%.6e Aref=%.6e\n",
            it, stage ? stage : "", r.patchName.c_str(), r.patchIndex, r.Uref, r.Aref);
        return;
    }

    std::printf("    dgPatchForce: it=%d stage=%s patch=%s faces=%d area=%.12e normalSign=%d mapFail=%d\n",
        it, stage ? stage : "", r.patchName.c_str(), r.nFaces, r.area, r.normalSign, r.mapFail);

    std::printf("    dgPatchForce: Fp=[%.12e %.12e %.12e] Fv=[%.12e %.12e %.12e] F=[%.12e %.12e %.12e]\n",
        r.Fp[0], r.Fp[1], r.Fp[2],
        r.Fv[0], r.Fv[1], r.Fv[2],
        r.F[0],  r.F[1],  r.F[2]);

    std::printf("    dgPatchForce: C_drag=%.12e C_lift=%.12e C_span=%.12e F_drag=%.12e F_lift=%.12e F_span=%.12e denom=%.12e\n",
        r.CDrag, r.CLift, r.CSpan,
        r.FDrag, r.FLift, r.FSpan,
        r.coeffDenom);

    std::printf("    dgPatchForce: drag(p,v,total)=[%.12e %.12e %.12e] lift(p,v,total)=[%.12e %.12e %.12e] span(p,v,total)=[%.12e %.12e %.12e] max|p|=%.12e max|tau|=%.12e dn[min,max]=[%.12e %.12e]\n",
        r.FpDrag, r.FvDrag, r.FDrag,
        r.FpLift, r.FvLift, r.FLift,
        r.FpSpan, r.FvSpan, r.FSpan,
        r.maxAbsPressure,
        r.maxAbsShearTraction,
        std::isfinite(r.minWallDistance) ? r.minWallDistance : 0.0,
        r.maxWallDistance);
}


struct DgPatchReactionForceReport {
    bool requested = false;
    bool valid = false;
    int patchIndex = -1;
    std::string patchName;
    int nFaces = 0;
    int markedRows = 0;
    double area = 0.0;
    double rho = 1.0;
    double Uref = 1.0;
    double Aref = 1.0;
    double coeffDenom = 0.0;
    bool usedConvBc = false;

    std::array<double,3> dragDir{{1.0,0.0,0.0}};
    std::array<double,3> liftDir{{0.0,0.0,1.0}};
    std::array<double,3> spanDir{{0.0,1.0,0.0}};

    // residual = Aphys*u + G(p) - F - convBC, evaluated on the original
    // unzeroed momentum stencil and summed on DG2 rows belonging to the patch.
    //
    // Important DG2/DG1 note:
    // The central single-flux continuity operator is excellent for mass balance,
    // but its transpose G=-B^T is not a full wall-pressure traction operator on
    // strongly imposed velocity rows.  In CG, row-reaction forces often inherit
    // the pressure boundary term naturally from the integrated-by-parts pressure
    // block.  Here that term must be accounted for explicitly.
    //
    // Therefore we print:
    //   raw reaction              = -raw residual
    //   pressureBoundaryForce     = pressure traction integrated on force patch
    //   pressureCompletedReaction = raw reaction + pressureBoundaryForce
    std::array<double,3> residual{{0.0,0.0,0.0}};
    std::array<double,3> reaction{{0.0,0.0,0.0}};
    std::array<double,3> pressureBoundaryForce{{0.0,0.0,0.0}};
    std::array<double,3> reactionPlusPressure{{0.0,0.0,0.0}};
    bool hasPressureBoundaryForce = false;

    double FDrag = 0.0, FLift = 0.0, FSpan = 0.0;
    double CDrag = 0.0, CLift = 0.0, CSpan = 0.0;

    double FpDrag = 0.0, FpLift = 0.0, FpSpan = 0.0;
    double CpDrag = 0.0, CpLift = 0.0, CpSpan = 0.0;

    double FcorrDrag = 0.0, FcorrLift = 0.0, FcorrSpan = 0.0;
    double CcorrDrag = 0.0, CcorrLift = 0.0, CcorrSpan = 0.0;

    double residualDrag = 0.0, residualLift = 0.0, residualSpan = 0.0;
    double residualCDrag = 0.0, residualCLift = 0.0, residualCSpan = 0.0;
    double maxAbsResidualRow = 0.0;
};

static std::vector<unsigned char> mark_dg2_patch_face_rows(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const std::string& patchName,
    int* patchIndexOut,
    int* nFacesOut,
    int* nRowsOut,
    double* areaOut)
{
    const int nU = 10 * mesh.nCells;
    std::vector<unsigned char> mask(nU, 0);

    int patchIndex = dg_find_patch_index_by_name(mesh, patchName);
    if (patchIndexOut) *patchIndexOut = patchIndex;
    if (nFacesOut) *nFacesOut = 0;
    if (nRowsOut) *nRowsOut = 0;
    if (areaOut) *areaOut = 0.0;

    if (patchIndex < 0) return mask;
    if (patchIndex >= (int)mesh.patchStartFace.size()) return mask;
    if (patchIndex >= (int)mesh.patchNFaces.size()) return mask;

    const int f0 = mesh.patchStartFace[patchIndex];
    const int f1 = f0 + mesh.patchNFaces[patchIndex];
    if (f0 < mesh.nInternalFaces || f1 > mesh.nFaces || f1 < f0) return mask;

    static const int edge[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    int nFaces = 0;
    double area = 0.0;

    for (int f=f0; f<f1; ++f) {
        if (f < 0 || f >= mesh.nFaces || f >= (int)mesh.owner.size()) continue;
        const int c = mesh.owner[f];
        if (c < 0 || c >= (int)tets.size()) continue;
        const TetP2Geom& K = tets[c];

        bool faceHasLocalVertex[4] = {false,false,false,false};
        for (int a=0; a<4; ++a) {
            for (int fv=0; fv<3; ++fv) {
                if (K.v[a] == mesh.faces[f][fv]) faceHasLocalVertex[a] = true;
            }
        }

        for (int i=0; i<10; ++i) {
            bool onFace = false;
            if (i < 4) {
                onFace = faceHasLocalVertex[i];
            } else {
                const int e = i - 4;
                onFace = faceHasLocalVertex[edge[e][0]] && faceHasLocalVertex[edge[e][1]];
            }
            if (!onFace) continue;
            const int row = 10*c + i;
            if (row >= 0 && row < nU) mask[row] = 1;
        }

        area += mesh.Af[f];
        ++nFaces;
    }

    int nRows = 0;
    for (unsigned char v : mask) if (v) ++nRows;

    if (nFacesOut) *nFacesOut = nFaces;
    if (nRowsOut) *nRowsOut = nRows;
    if (areaOut) *areaOut = area;
    return mask;
}

static DgPatchReactionForceReport compute_dg2_patch_reaction_forces_from_momentum_residual(
    const Mesh& mesh,
    const std::vector<TetP2Geom>& tets,
    const CSRPattern& uPat,
    const std::vector<double>& Aphys,
    const RectCSR& Apx,
    const RectCSR& Apy,
    const RectCSR& Apz,
    const std::array<std::vector<double>,3>& F,
    const std::string& patchName,
    const std::vector<double>& ux,
    const std::vector<double>& uy,
    const std::vector<double>& uz,
    const std::vector<double>& p,
    double rho,
    double Uref,
    double Aref,
    std::array<double,3> dragDir,
    std::array<double,3> liftDir,
    std::array<double,3> spanDir,
    const std::vector<double>* convBCx = nullptr,
    const std::vector<double>* convBCy = nullptr,
    const std::vector<double>* convBCz = nullptr,
    const std::array<double,3>* pressureBoundaryForce = nullptr)
{
    DgPatchReactionForceReport r;
    r.requested = true;
    r.patchName = patchName;
    r.rho = rho;
    r.Uref = Uref;
    r.Aref = Aref;
    r.dragDir = dg_normalized_vec3(dragDir, {{1.0,0.0,0.0}});
    r.liftDir = dg_normalized_vec3(liftDir, {{0.0,0.0,1.0}});
    r.spanDir = dg_normalized_vec3(spanDir, {{0.0,1.0,0.0}});
    r.coeffDenom = rho * Uref * Uref * Aref;

    if (Uref <= 0.0 || Aref <= 0.0 || r.coeffDenom <= 1.0e-300) return r;
    if ((int)Aphys.size() != uPat.nnz) return r;

    int patchIndex = -1;
    int nFaces = 0;
    int markedRows = 0;
    double area = 0.0;
    std::vector<unsigned char> rowMask = mark_dg2_patch_face_rows(
        mesh, tets, patchName, &patchIndex, &nFaces, &markedRows, &area);

    r.patchIndex = patchIndex;
    r.nFaces = nFaces;
    r.markedRows = markedRows;
    r.area = area;

    if (patchIndex < 0 || markedRows <= 0) return r;
    r.valid = true;

    std::vector<double> Rx, Ry, Rz;
    std::vector<double> gp;

    apply_csr(uPat, Aphys, ux, Rx);
    apply_gradient_rhs(Apx, p, gp);
    axpy_vec(Rx, 1.0, gp);
    axpy_vec(Rx, -1.0, F[0]);
    if (convBCx && convBCx->size() == Rx.size()) {
        axpy_vec(Rx, -1.0, *convBCx);
        r.usedConvBc = true;
    }

    apply_csr(uPat, Aphys, uy, Ry);
    apply_gradient_rhs(Apy, p, gp);
    axpy_vec(Ry, 1.0, gp);
    axpy_vec(Ry, -1.0, F[1]);
    if (convBCy && convBCy->size() == Ry.size()) {
        axpy_vec(Ry, -1.0, *convBCy);
        r.usedConvBc = true;
    }

    apply_csr(uPat, Aphys, uz, Rz);
    apply_gradient_rhs(Apz, p, gp);
    axpy_vec(Rz, 1.0, gp);
    axpy_vec(Rz, -1.0, F[2]);
    if (convBCz && convBCz->size() == Rz.size()) {
        axpy_vec(Rz, -1.0, *convBCz);
        r.usedConvBc = true;
    }

    const int n = std::min<int>((int)rowMask.size(), (int)Rx.size());
    for (int i=0; i<n; ++i) {
        if (!rowMask[i]) continue;
        r.residual[0] += Rx[i];
        r.residual[1] += Ry[i];
        r.residual[2] += Rz[i];
        r.maxAbsResidualRow = std::max(r.maxAbsResidualRow, std::abs(Rx[i]));
        r.maxAbsResidualRow = std::max(r.maxAbsResidualRow, std::abs(Ry[i]));
        r.maxAbsResidualRow = std::max(r.maxAbsResidualRow, std::abs(Rz[i]));
    }

    for (int d=0; d<3; ++d) {
        r.reaction[d] = -r.residual[d];

        if (pressureBoundaryForce) {
            r.pressureBoundaryForce[d] = (*pressureBoundaryForce)[d];
            r.hasPressureBoundaryForce = true;
        }

        r.reactionPlusPressure[d] = r.reaction[d] + r.pressureBoundaryForce[d];
    }

    r.FDrag = dot3(r.reaction, r.dragDir);
    r.FLift = dot3(r.reaction, r.liftDir);
    r.FSpan = dot3(r.reaction, r.spanDir);

    r.FpDrag = dot3(r.pressureBoundaryForce, r.dragDir);
    r.FpLift = dot3(r.pressureBoundaryForce, r.liftDir);
    r.FpSpan = dot3(r.pressureBoundaryForce, r.spanDir);

    r.FcorrDrag = dot3(r.reactionPlusPressure, r.dragDir);
    r.FcorrLift = dot3(r.reactionPlusPressure, r.liftDir);
    r.FcorrSpan = dot3(r.reactionPlusPressure, r.spanDir);

    r.residualDrag = dot3(r.residual, r.dragDir);
    r.residualLift = dot3(r.residual, r.liftDir);
    r.residualSpan = dot3(r.residual, r.spanDir);

    if (r.coeffDenom > 1.0e-300) {
        r.CDrag = 2.0 * r.FDrag / r.coeffDenom;
        r.CLift = 2.0 * r.FLift / r.coeffDenom;
        r.CSpan = 2.0 * r.FSpan / r.coeffDenom;

        r.CpDrag = 2.0 * r.FpDrag / r.coeffDenom;
        r.CpLift = 2.0 * r.FpLift / r.coeffDenom;
        r.CpSpan = 2.0 * r.FpSpan / r.coeffDenom;

        r.CcorrDrag = 2.0 * r.FcorrDrag / r.coeffDenom;
        r.CcorrLift = 2.0 * r.FcorrLift / r.coeffDenom;
        r.CcorrSpan = 2.0 * r.FcorrSpan / r.coeffDenom;

        r.residualCDrag = 2.0 * r.residualDrag / r.coeffDenom;
        r.residualCLift = 2.0 * r.residualLift / r.coeffDenom;
        r.residualCSpan = 2.0 * r.residualSpan / r.coeffDenom;
    }

    return r;
}

static void print_dg_patch_reaction_force_report(
    const DgPatchReactionForceReport& r,
    int it,
    const char* stage)
{
    if (!r.requested) return;

    if (!r.valid) {
        std::printf("    dgPatchReactionForce: it=%d stage=%s INVALID patch=%s patchIndex=%d rows=%d Uref=%.6e Aref=%.6e\n",
            it, stage ? stage : "", r.patchName.c_str(), r.patchIndex, r.markedRows, r.Uref, r.Aref);
        return;
    }

    std::printf("    dgPatchReactionForce: it=%d stage=%s patch=%s faces=%d markedRows=%d area=%.12e usedConvBC=%d\n",
        it, stage ? stage : "", r.patchName.c_str(), r.nFaces, r.markedRows, r.area, r.usedConvBc ? 1 : 0);

    std::printf("    dgPatchReactionForce: raw residual=Aphys*u+Gp-F-convBC summed on patch rows = [%.12e %.12e %.12e]\n",
        r.residual[0], r.residual[1], r.residual[2]);

    std::printf("    dgPatchReactionForce: raw reaction=-residual=[%.12e %.12e %.12e] C_drag=%.12e C_lift=%.12e C_span=%.12e F_drag=%.12e F_lift=%.12e F_span=%.12e denom=%.12e maxAbsRow=%.12e\n",
        r.reaction[0], r.reaction[1], r.reaction[2],
        r.CDrag, r.CLift, r.CSpan,
        r.FDrag, r.FLift, r.FSpan,
        r.coeffDenom,
        r.maxAbsResidualRow);

    if (r.hasPressureBoundaryForce) {
        std::printf("    dgPatchReactionForce: pressureBoundaryAdded(Fp from face traction)=[%.12e %.12e %.12e] C_drag=%.12e C_lift=%.12e C_span=%.12e\n",
            r.pressureBoundaryForce[0], r.pressureBoundaryForce[1], r.pressureBoundaryForce[2],
            r.CpDrag, r.CpLift, r.CpSpan);

        std::printf("    dgPatchReactionForce: pressureCompletedReaction=rawReaction+Fp=[%.12e %.12e %.12e] C_drag=%.12e C_lift=%.12e C_span=%.12e F_drag=%.12e F_lift=%.12e F_span=%.12e\n",
            r.reactionPlusPressure[0], r.reactionPlusPressure[1], r.reactionPlusPressure[2],
            r.CcorrDrag, r.CcorrLift, r.CcorrSpan,
            r.FcorrDrag, r.FcorrLift, r.FcorrSpan);
    }

    std::printf("    dgPatchReactionForce: oppositeSign rawResidualCoeffs C_drag=%.12e C_lift=%.12e C_span=%.12e F_drag=%.12e F_lift=%.12e F_span=%.12e\n",
        r.residualCDrag, r.residualCLift, r.residualCSpan,
        r.residualDrag, r.residualLift, r.residualSpan);
}

static std::array<double,3> parse_vec3_csv(
    const std::string& text,
    std::array<double,3> def)
{
    if (text.empty()) return def;

    std::string s = text;

    for (char& c : s) {
        if (c == ',' || c == ';' || c == ':') c = ' ';
    }

    const char* q = s.c_str();
    char* end = nullptr;

    double v[3];

    for (int i=0; i<3; ++i) {
        v[i] = std::strtod(q, &end);
        if (end == q) return def;
        q = end;
    }

    return {{v[0], v[1], v[2]}};
}

static const char* get_arg(int argc, char** argv, const char* key, const char* def)
{
    for (int i=1; i<argc-1; ++i) {
        if (std::string(argv[i]) == key) return argv[i+1];
    }

    return def;
}

static int get_int_arg(int argc, char** argv, const char* key, int def)
{
    return std::atoi(get_arg(argc, argv, key, std::to_string(def).c_str()));
}

static double get_double_arg(int argc, char** argv, const char* key, double def)
{
    return std::atof(get_arg(argc, argv, key, std::to_string(def).c_str()));
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    try {
        const std::string polyMeshDir = get_arg(argc, argv, "-polyMeshDir", "");

        const std::string flowMode = get_arg(argc, argv, "-flowMode", "nse");
        const std::string caseMode = get_arg(argc, argv, "-caseMode", "mms");
        const double channelInletUx = get_double_arg(argc, argv, "-channelInletUx",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? 0.45 : 1.0);
        const std::string channelInletProfileArg = get_arg(argc, argv, "-channelInletProfile",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? "parabolic_box" : "uniform");
        const double channelParabolicBoxWidth = get_double_arg(argc, argv, "-channelParabolicBoxWidth", 0.41);
        const int channelNormalInlet = get_int_arg(argc, argv, "-channelNormalInlet", 1);
        const int channelPressureOutlet = get_int_arg(argc, argv, "-channelPressureOutlet", 0);
        const int channelPressureOutletSinglePin = get_int_arg(argc, argv, "-channelPressureOutletSinglePin", 0);
        const int channelPressureOutletDirichletAll = get_int_arg(argc, argv, "-channelPressureOutletDirichletAll", 1);
        const double channelFaceTol = get_double_arg(argc, argv, "-channelFaceTol", 1e-8);
        const std::string channelInletPatch  = get_arg(argc, argv, "-channelInletPatch", "");
        const std::string channelOutletPatch = get_arg(argc, argv, "-channelOutletPatch", "");
        const std::string channelWallPatch   = get_arg(argc, argv, "-channelWallPatch", "");

        const double channelInletDirX = get_double_arg(argc, argv, "-channelInletDirX",
            (caseMode == "pipe" || caseMode == "Pipe") ? 0.0 : 1.0);
        const double channelInletDirY = get_double_arg(argc, argv, "-channelInletDirY", 0.0);
        const double channelInletDirZ = get_double_arg(argc, argv, "-channelInletDirZ",
            (caseMode == "pipe" || caseMode == "Pipe") ? 1.0 : 0.0);
        const int channelInletStartFace  = get_int_arg(argc, argv, "-channelInletStartFace", -1);
        const int channelInletNFaces     = get_int_arg(argc, argv, "-channelInletNFaces", 0);
        const int channelOutletStartFace = get_int_arg(argc, argv, "-channelOutletStartFace", -1);
        const int channelOutletNFaces    = get_int_arg(argc, argv, "-channelOutletNFaces", 0);
        const int channelWallStartFace   = get_int_arg(argc, argv, "-channelWallStartFace", -1);
        const int channelWallNFaces      = get_int_arg(argc, argv, "-channelWallNFaces", 0);
        const int channelDirectContinuity = get_int_arg(argc, argv, "-channelDirectContinuity", 1);
        const double channelDirectContinuitySign = get_double_arg(argc, argv, "-channelDirectContinuitySign", 1.0);
        const int channelClampVelocityBC = get_int_arg(argc, argv, "-channelClampVelocityBC", 1);
        const int channelProjectDirichletNormalCorr = get_int_arg(argc, argv, "-channelProjectDirichletNormalCorr", 0);
        const int channelOutletCorrDiag = get_int_arg(argc, argv, "-channelOutletCorrDiag", 0);
        const int channelMaskDirichletSchurRows = get_int_arg(argc, argv, "-channelMaskDirichletSchurRows", 0);
        const int channelStrongVelocityRows = get_int_arg(argc, argv, "-channelStrongVelocityRows", 0);
        const int channelConstrainVelocityCorrection = get_int_arg(argc, argv, "-channelConstrainVelocityCorrection", channelStrongVelocityRows);
        const int channelAdjustOutletFlux = get_int_arg(argc, argv, "-channelAdjustOutletFlux", 0);
        const int channelOutletCompatibility = get_int_arg(argc, argv, "-channelOutletCompatibility", 1);
        const double amp = get_double_arg(argc, argv, "-amp", 0.25);

        const double nu = get_double_arg(argc, argv, "-nu", 10.0);

        const int forceEnable = get_int_arg(argc, argv, "-forceEnable",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? 1 : 0);

        const std::string forcePatch = get_arg(argc, argv, "-forcePatch",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_3_0" : "");

        const int forceEvery = get_int_arg(argc, argv, "-forceEvery", 0);
        const int forceReactionEnable = get_int_arg(argc, argv, "-forceReactionEnable", forceEnable);
        const int forceNormalSign = get_int_arg(argc, argv, "-forceNormalSign", -1);

        const double forceRho = get_double_arg(argc, argv, "-forceRho", 1.0);
        const double forceMu = get_double_arg(argc, argv, "-forceMu", nu);

        const double forceUref = get_double_arg(argc, argv, "-forceUref",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? 0.2 : 1.0);

        const double forceAreaRef = get_double_arg(argc, argv, "-forceAreaRef",
            (caseMode == "cylinder" || caseMode == "Cylinder") ? 0.041 : 1.0);

        const std::array<double,3> forceDragDir =
            parse_vec3_csv(get_arg(argc, argv, "-forceDragDir", "1,0,0"), {{1.0,0.0,0.0}});

        const std::array<double,3> forceLiftDir =
            parse_vec3_csv(get_arg(argc, argv, "-forceLiftDir", "0,0,1"), {{0.0,0.0,1.0}});

        const std::array<double,3> forceSpanDir =
            parse_vec3_csv(get_arg(argc, argv, "-forceSpanDir", "0,1,0"), {{0.0,1.0,0.0}});
        const double sigma = get_double_arg(argc, argv, "-sigma", 20.0);
        const int quad1d = get_int_arg(argc, argv, "-quad1d", 4);
        const int nSimple = get_int_arg(argc, argv, "-nSimple", 200);
        const double alphaP = get_double_arg(argc, argv, "-alphaP", 0.30);
        const double alphaU = get_double_arg(argc, argv, "-alphaU", 1.0);
        const int printEvery = get_int_arg(argc, argv, "-printEvery", 5);
        const double tolCoupled = get_double_arg(argc, argv, "-tolCoupledRel", 1e-5);
        const double tolUpdate = get_double_arg(argc, argv, "-tolUpdateRel", 1e-5);

        const std::string uSolver = get_arg(argc, argv, "-uSolver", "bicgstab");
        const std::string uPrecond = get_arg(argc, argv, "-uPrecond", "diagscale");
        const int uMaxit = get_int_arg(argc, argv, "-uMaxit", 500);
        const double uTol = get_double_arg(argc, argv, "-uTol", 1e-10);
        const double uRelTol = get_double_arg(argc, argv, "-uRelTol", uTol);
        const double uAbsTol = get_double_arg(argc, argv, "-uAbsTol", uTol);
        const int uKDim = get_int_arg(argc, argv, "-uKDim", 80);
        const int uInnerMaxit = get_int_arg(argc, argv, "-uInnerMaxit", 30);
        const int uInnerKDim = get_int_arg(argc, argv, "-uInnerKDim", 30);
        const double uInnerRelTol = get_double_arg(argc, argv, "-uInnerRelTol", 0.05);
        const double uInnerAbsTol = get_double_arg(argc, argv, "-uInnerAbsTol", 0.0);
        const int uMonitor = get_int_arg(argc, argv, "-uMonitor", 0);

        const int uSweeps = get_int_arg(argc, argv, "-uSweeps", 80);
        const int uMcgsPreSweeps = get_int_arg(argc, argv, "-uMcgsPreSweeps", 2);
        const int uCellBlockIters = get_int_arg(argc, argv, "-uCellBlockIters", 0);
        const double uCellBlockOmega = get_double_arg(argc, argv, "-uCellBlockOmega", 1.0);
        const double uOmega = get_double_arg(argc, argv, "-uOmega", 0.8);
        const int uSymmetric = get_int_arg(argc, argv, "-uSymmetric", 0);
        
        const int velocitySolveUseFreeRows = get_int_arg(argc, argv, "-velocitySolveUseFreeRows", 0);
        const int simpleConvergenceUseFreeRows = get_int_arg(argc, argv, "-simpleConvergenceUseFreeRows", 0);
                const int simpleConvergenceRequireUpdateRel =
            get_int_arg(argc, argv, "-simpleConvergenceRequireUpdateRel", 1);
const double velocityFreeRowsScaleFloor = get_double_arg(argc, argv, "-velocityFreeRowsScaleFloor", 1e-30);
        const int velocityFreeRowsPrint = get_int_arg(argc, argv, "-velocityFreeRowsPrint", 1);
const int velocitySolveAudit = get_int_arg(argc, argv, "-velocitySolveAudit", 0);
        const int velocitySolveAuditEvery = get_int_arg(argc, argv, "-velocitySolveAuditEvery", 1);
        const int velocitySolveAuditDump = get_int_arg(argc, argv, "-velocitySolveAuditDump", 0);
        const int velocitySolveAuditDumpEvery = get_int_arg(argc, argv, "-velocitySolveAuditDumpEvery", 0);
        const std::string velocitySolveAuditPrefix = get_arg(argc, argv, "-velocitySolveAuditPrefix", "velocity_audit");

        std::string uLeftScaleMode = get_arg(argc, argv, "-uLeftScale", "none");
        const double uLeftScaleEps = get_double_arg(argc, argv, "-uLeftScaleEps", 1e-300);
        const double uBlockJacobiShift = get_double_arg(argc, argv, "-uBlockJacobiShift", 0.0);
        const double uBlockJacobiPivotFloor = get_double_arg(argc, argv, "-uBlockJacobiPivotFloor", 1e-30);

        const std::string pressureMode = get_arg(argc, argv, "-pressureMode", "pmass");
        const double massSchurScale = get_double_arg(argc, argv, "-massSchurScale", 1.0);
        const double channelPressureOutletPenaltyBeta = get_double_arg(argc, argv, "-channelPressureOutletPenaltyBeta", 0.0);
        const double schurBlockShift = get_double_arg(argc, argv, "-schurBlockShift", uBlockJacobiShift);
        const double schurBlockPivotFloor = get_double_arg(argc, argv, "-schurBlockPivotFloor", uBlockJacobiPivotFloor);
        const double schurVelocityCorrectionScale = get_double_arg(argc, argv, "-schurVelocityCorrectionScale", massSchurScale);
        const double hybridCellPmassWeight = get_double_arg(argc, argv, "-hybridCellPmassWeight", 1.0);
        const double hybridCellSchurWeight = get_double_arg(argc, argv, "-hybridCellSchurWeight", 0.001);
        const int pressureVelocityCorrection = get_int_arg(argc, argv, "-pressureVelocityCorrection", 1);
        const int massSchurProjectionAudit = get_int_arg(argc, argv, "-massSchurProjectionAudit", 1);
        const int massSchurUseModelContinuityRhs = get_int_arg(argc, argv, "-massSchurUseModelContinuityRhs", 0);
        const int massSchurUseFixedBcRhs = get_int_arg(argc, argv, "-massSchurUseFixedBcRhs", 0);
        const int massSchurUseDirectBcRhs = get_int_arg(argc, argv, "-massSchurUseDirectBcRhs", 0);
        const int massSchurUseDirectOperator = get_int_arg(argc, argv, "-massSchurUseDirectOperator", 0);
        const int massSchurUseOpenOutletB = get_int_arg(argc, argv, "-massSchurUseOpenOutletB", 0);
        const std::string schurInnerSolver = get_arg(argc, argv, "-schurInnerSolver", uSolver.c_str());
        const std::string schurInnerPrecond = get_arg(argc, argv, "-schurInnerPrecond", uPrecond.c_str());
        const int schurInnerMaxit = get_int_arg(argc, argv, "-schurInnerMaxit", 80);
        const double schurInnerRelTol = get_double_arg(argc, argv, "-schurInnerRelTol", 5e-2);
        const double schurInnerAbsTol = get_double_arg(argc, argv, "-schurInnerAbsTol", 0.0);
        const int schurInnerMonitor = get_int_arg(argc, argv, "-schurInnerMonitor", 0);
        const int schurPressureMaxit = get_int_arg(argc, argv, "-schurPressureMaxit", 12);
        const int schurPressureKdim = get_int_arg(argc, argv, "-schurPressureKdim", 12);
        const double schurPressureRelTol = get_double_arg(argc, argv, "-schurPressureRelTol", 1e-2);
        const double schurPressureAbsTol = get_double_arg(argc, argv, "-schurPressureAbsTol", 0.0);

        const double hybridNuCrossover = get_double_arg(argc, argv, "-hybridNuCrossover", 0.1);
        const double hybridNuPower = get_double_arg(argc, argv, "-hybridNuPower", 1.0);
        const double hybridPmassCoeff = get_double_arg(argc, argv, "-hybridPmassCoeff", 1.0);
        const double hybridSimpleCoeff = get_double_arg(argc, argv, "-hybridSimpleCoeff", 0.25);
        const int hybridVelocityCorrection = get_int_arg(argc, argv, "-hybridVelocityCorrection", 0);
        const double hybridVelocityCorrectionScale = get_double_arg(argc, argv, "-hybridVelocityCorrectionScale", 1.0);

        const std::string pSolver = get_arg(argc, argv, "-pSolver", "pcg");
        const std::string pPrecond = get_arg(argc, argv, "-pPrecond", "amg");
        const int pMaxit = get_int_arg(argc, argv, "-pMaxit", 8000);
        const double pTol = get_double_arg(argc, argv, "-pTol", 1e-12);
        const double pRelTol = get_double_arg(argc, argv, "-pRelTol", 0.0);
        const int pMonitor = get_int_arg(argc, argv, "-pMonitor", 0);

        const int pAmgCoarsenType = get_int_arg(argc, argv, "-pAmgCoarsenType", 8);
        const int pAmgInterpType = get_int_arg(argc, argv, "-pAmgInterpType", 6);
        const int pAmgRelaxType = get_int_arg(argc, argv, "-pAmgRelaxType", 18);
        const int pAmgAggLevels = get_int_arg(argc, argv, "-pAmgAggLevels", 0);
        const int pAmgKeepTranspose = get_int_arg(argc, argv, "-pAmgKeepTranspose", 1);
        const int pAmgPmax = get_int_arg(argc, argv, "-pAmgPmax", 4);
        const int pAmgNumSweeps = get_int_arg(argc, argv, "-pAmgNumSweeps", 1);
        const double pAmgStrongThreshold = get_double_arg(argc, argv, "-pAmgStrongThreshold", -1.0);
        const int pReuseAmg = get_int_arg(argc, argv, "-pReuseAmg", 1);
        const int pressureRepairZeroRows = get_int_arg(argc, argv, "-pressureRepairZeroRows", 0);
        const int pSplitPrecondUnmaskedMassSchur = get_int_arg(argc, argv, "-pSplitPrecondUnmaskedMassSchur", 0);
        const int pSplitPrecondInnerMaxit = get_int_arg(argc, argv, "-pSplitPrecondInnerMaxit", 8);
        const double pSplitPrecondInnerRelTol = get_double_arg(argc, argv, "-pSplitPrecondInnerRelTol", 1e-2);
        const double pSplitPrecondInnerAbsTol = get_double_arg(argc, argv, "-pSplitPrecondInnerAbsTol", 0.0);
        const int pSplitPrecondInnerMonitor = get_int_arg(argc, argv, "-pSplitPrecondInnerMonitor", 0);
        const int pSplitOuterFgmres = get_int_arg(argc, argv, "-pSplitOuterFgmres", 0);
        const int pSplitFgmresRestart = get_int_arg(argc, argv, "-pSplitFgmresRestart", 30);
        const int pSplitPrecondReusable = get_int_arg(argc, argv, "-pSplitPrecondReusable", 0);
        const int uReuseObjects = get_int_arg(argc, argv, "-uReuseObjects", 0);
        int convGpu = get_int_arg(argc, argv, "-convGpu", 0);
        const std::string momentumFluxArg = get_arg(argc, argv, "-momentumFlux", "lf");
        const int errEvery = get_int_arg(argc, argv, "-errEvery", 0);
        const int uCheckEvery = get_int_arg(argc, argv, "-uCheckEvery", 1);
        const int memEvery = get_int_arg(argc, argv, "-memEvery", 0);
        const int profileTimings = get_int_arg(argc, argv, "-profileTimings", 0);
        const int hypreProfile = get_int_arg(argc, argv, "-hypreProfile", 0);
        const int writeVtu = get_int_arg(argc, argv, "-writeVtu", 0);
        const std::string vtuFile = get_arg(argc, argv, "-vtuFile", "dg2cg1_solution.vtu");

        const int device = get_int_arg(argc, argv, "-device", 0);

        if (polyMeshDir.empty()) throw std::runtime_error("Missing -polyMeshDir");

        gMmsAmp = amp;
        gCaseMode = caseMode;
        gCaseIsChannel = (caseMode == "channel" || caseMode == "Channel" ||
                          caseMode == "inlet"   || caseMode == "pipe" || caseMode == "Pipe" ||
                          caseMode == "cylinder" || caseMode == "Cylinder");

        gChannelInletUx = channelInletUx;
        gChannelInletNormalVelocity = channelNormalInlet;
        gChannelInletProfileMode =
            (channelInletProfileArg == "parabolic_box" ||
             channelInletProfileArg == "parabolic" ||
             channelInletProfileArg == "cylinder_re20" ||
             channelInletProfileArg == "cylinder") ? 1 : 0;
        gChannelParabolicBoxWidth = channelParabolicBoxWidth;
        gChannelPressureOutlet = channelPressureOutlet;
        gChannelPressureOutletSinglePin = channelPressureOutletSinglePin;
        gChannelPressureOutletDirichletAll = channelPressureOutletDirichletAll;
        gChannelAdjustOutletFlux = channelAdjustOutletFlux;
        gChannelOutletCompatibility = channelOutletCompatibility;
        gChannelFaceTol = channelFaceTol;
        gChannelUsePatchBC =
            (!channelInletPatch.empty() || !channelOutletPatch.empty() || !channelWallPatch.empty() ||
             caseMode == "pipe" || caseMode == "Pipe" || caseMode == "cylinder" || caseMode == "Cylinder");

        gChannelInletPatchName =
            !channelInletPatch.empty() ? channelInletPatch :
            ((caseMode == "pipe" || caseMode == "Pipe") ? "patch_2_0" :
             ((caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_1_0" : ""));

        gChannelOutletPatchName =
            !channelOutletPatch.empty() ? channelOutletPatch :
            ((caseMode == "pipe" || caseMode == "Pipe") ? "patch_1_0" :
             ((caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_2_0" : ""));

        gChannelWallPatchName =
            !channelWallPatch.empty() ? channelWallPatch :
            ((caseMode == "pipe" || caseMode == "Pipe" || caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_0_0" : "");

        gChannelInletDirX = channelInletDirX;
        gChannelInletDirY = channelInletDirY;
        gChannelInletDirZ = channelInletDirZ;
        gChannelInletStartFace  = channelInletStartFace;
        gChannelInletNFaces     = channelInletNFaces;
        gChannelOutletStartFace = channelOutletStartFace;
        gChannelOutletNFaces    = channelOutletNFaces;
        gChannelWallStartFace   = channelWallStartFace;
        gChannelWallNFaces      = channelWallNFaces;

        std::printf("physicalBC runtime      caseMode=%s active=%d patchBC=%d inletPatch=%s outletPatch=%s wallPatch=%s inletDir=(%.6g,%.6g,%.6g) ranges in=(%d,%d) out=(%d,%d) wall=(%d,%d)\\n",
            caseMode.c_str(),
            gCaseIsChannel ? 1 : 0,
            gChannelUsePatchBC ? 1 : 0,
            gChannelInletPatchName.empty()  ? "<coord>" : gChannelInletPatchName.c_str(),
            gChannelOutletPatchName.empty() ? "<coord>" : gChannelOutletPatchName.c_str(),
            gChannelWallPatchName.empty()   ? "<none>"  : gChannelWallPatchName.c_str(),
            gChannelInletDirX, gChannelInletDirY, gChannelInletDirZ,
            gChannelInletStartFace, gChannelInletNFaces,
            gChannelOutletStartFace, gChannelOutletNFaces,
            gChannelWallStartFace, gChannelWallNFaces);
        std::printf("channelOutletCompatibility = %d; 1=outlet-weight pressure-RHS compatibility, 0=ordinary global compatibility\\n", gChannelOutletCompatibility);
        gChannelUsePatchBC =
            (!channelInletPatch.empty() || !channelOutletPatch.empty() || !channelWallPatch.empty() ||
             caseMode == "pipe" || caseMode == "Pipe" || caseMode == "cylinder" || caseMode == "Cylinder");

        gChannelInletPatchName =
            !channelInletPatch.empty() ? channelInletPatch :
            ((caseMode == "pipe" || caseMode == "Pipe") ? "patch_2_0" :
             ((caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_1_0" : ""));

        gChannelOutletPatchName =
            !channelOutletPatch.empty() ? channelOutletPatch :
            ((caseMode == "pipe" || caseMode == "Pipe") ? "patch_1_0" :
             ((caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_2_0" : ""));

        gChannelWallPatchName =
            !channelWallPatch.empty() ? channelWallPatch :
            ((caseMode == "pipe" || caseMode == "Pipe" || caseMode == "cylinder" || caseMode == "Cylinder") ? "patch_0_0" : "");

        gChannelInletDirX = channelInletDirX;
        gChannelInletDirY = channelInletDirY;
        gChannelInletDirZ = channelInletDirZ;

        if (gCaseIsChannel && convGpu) {
            if (gChannelUsePatchBC) {
                std::printf("channelPatchBC runtime   inletPatch=%s outletPatch=%s wallPatch=%s inletDir=(%.6g,%.6g,%.6g)\n",
                    gChannelInletPatchName.c_str(),
                    gChannelOutletPatchName.c_str(),
                    gChannelWallPatchName.c_str(),
                    gChannelInletDirX, gChannelInletDirY, gChannelInletDirZ);
            }
                        std::printf("Channel mode note        disabling -convGpu for now: physical inlet/outlet LF boundary RHS is assembled on CPU. MMS modes still use GPU convection when requested.\n");
            convGpu = 0;
        }
        gVelocityBlockJacobiShift = uBlockJacobiShift;
        gVelocityBlockJacobiPivotFloor = uBlockJacobiPivotFloor;
        gFlowIsNse = (flowMode == "nse" || flowMode == "NSE" || flowMode == "navierstokes");

        gMomentumFluxMode = momentumFluxArg;
        std::transform(gMomentumFluxMode.begin(), gMomentumFluxMode.end(),
                       gMomentumFluxMode.begin(),
                       [](unsigned char ch){ return (char)std::tolower(ch); });
        if (!(gMomentumFluxMode == "lf" || gMomentumFluxMode == "upwind" ||
              gMomentumFluxMode == "rusanov" || momentum_flux_is_central())) {
            throw std::runtime_error("Unknown -momentumFlux. Use lf/upwind/rusanov or central.");
        }

        if (!(gFlowIsNse || flowMode == "stokes" || flowMode == "Stokes")) {
            throw std::runtime_error("Unknown -flowMode. Use stokes or nse.");
        }

        if (convGpu && momentum_flux_is_central()) {
            std::printf("Momentum flux note       disabling -convGpu: -momentumFlux central is currently CPU-assembled.\n");
            convGpu = 0;
        }

        const bool inexactSchurMode0 =
            (pressureMode == "inexactschur" || pressureMode == "mfschur" ||
             pressureMode == "matrixfreeschur" || pressureMode == "arelinnerschur");

        const double gammaU = (1.0-alphaU) / std::max(alphaU, 1e-300);

        std::printf("\n=======================================================\n");
        std::printf("DG2/CG1 high-nu PMASS/Uzawa SIMPLE diagnostic\n");
        std::printf("precision              = %s sizeof(HYPRE_Complex)=%zu\n", anabasis_precision_name(), sizeof(HYPRE_Complex));
        std::printf("polyMeshDir            = %s\n", polyMeshDir.c_str());
        std::printf("flowMode               = %s\n", gFlowIsNse ? "nse" : "stokes");
        std::printf("momentumFlux           = %s%s\n",
                    gMomentumFluxMode.c_str(),
                    momentum_flux_is_central() ? " (central; no LF/Rusanov dissipation)" : " (LF/upwind/Rusanov)");
        std::printf("caseMode               = %s\n", caseMode.c_str());
        std::printf("channelInletUx          = %.17g channelNormalInlet=%d channelPressureOutlet=%d channelPressureOutletSinglePin=%d channelPressureOutletDirichletAll=%d channelFaceTol=%.3e\n",
                    channelInletUx, channelNormalInlet, channelPressureOutlet, channelPressureOutletSinglePin,
                    channelPressureOutletDirichletAll, channelFaceTol);
        std::printf("channelInletProfile       = %s parabolicBoxWidth=%.17g\n",
                    channelInletProfileArg.c_str(), channelParabolicBoxWidth);
        if (gCaseIsChannel) {
            std::printf("channelPatchBC          = %d inletPatch=%s outletPatch=%s wallPatch=%s inletDir=(%.6g,%.6g,%.6g)\n",
                gChannelUsePatchBC ? 1 : 0,
                gChannelInletPatchName.empty()  ? "<coord>"        : gChannelInletPatchName.c_str(),
                gChannelOutletPatchName.empty() ? "<coord>"        : gChannelOutletPatchName.c_str(),
                gChannelWallPatchName.empty()   ? "<default-wall>" : gChannelWallPatchName.c_str(),
                gChannelInletDirX, gChannelInletDirY, gChannelInletDirZ);
        }
        std::printf("amp                    = %.17g\n", amp);
        std::printf("nu                     = %.17g\n", nu);
        std::printf("sigma                  = %.17g\n", sigma);
        std::printf("alphaP                 = %.17g\n", alphaP);
        std::printf("alphaU                 = %.17g gammaU=%.17g\n", alphaU, gammaU);
        std::printf("pressureMode            = %s\n", pressureMode.c_str());
        std::printf("PMASS update            p'=-nu*Mp_lump^{-1}*B*u*, velocity correction=none\n");
        std::printf("massSchur update         Lp=Ap^T M^{-1} Ap, Lp p'=-B u*, optional u correction\n");
        std::printf("outlet pressure penalty  beta=%.6e; use with single outlet reference pin to keep outlet pressure rows alive\n",
                    channelPressureOutletPenaltyBeta);
        std::printf("inexactSchur update      matrix-free S(q)=Ap^T Arel^{-1} Ap q using inner velocity solves\n");
        std::printf("cellBlockSchur update    Lp=Ap^T blockdiag_10(Arel)^{-1} Ap, rebuilt each SIMPLE step\n");
        std::printf("hybridCellSchur update   pcorr=wP*PMASS + wC*cellBlockSchur, fixed blend\n");
        std::printf("diag/rowsum Schur        Lp=Ap^T diag(Arel or rowSum(Arel))^{-1} Ap, rebuilt each SIMPLE step\n");
        std::printf("hybridmass update        pcorr=wP*PMASS + wS*massSchur, default pressure-only\n");
        std::printf("hybrid settings          nu0=%.6g q=%.6g pmassCoeff=%.6g simpleCoeff=%.6g velCorr=%d velCorrScale=%.6g\n",
                    hybridNuCrossover, hybridNuPower, hybridPmassCoeff, hybridSimpleCoeff,
                    hybridVelocityCorrection, hybridVelocityCorrectionScale);
        std::printf("uSolver/uPrecond        = %s/%s\n", uSolver.c_str(), uPrecond.c_str());
        std::printf("u tolerances             rel=%.3e abs=%.3e kdim=%d innerMax=%d innerK=%d innerRel=%.3e innerAbs=%.3e\n",
                    uRelTol, uAbsTol, uKDim, uInnerMaxit, uInnerKDim, uInnerRelTol, uInnerAbsTol);
        std::printf("u custom left scale       = %s eps=%.3e blockShift=%.3e blockPivotFloor=%.3e\n",
                    uLeftScaleMode.c_str(), uLeftScaleEps, uBlockJacobiShift, uBlockJacobiPivotFloor);
        std::printf("Schur cell-block controls shift=%.3e pivotFloor=%.3e velCorrScale=%.3e\n",
                    schurBlockShift, schurBlockPivotFloor, schurVelocityCorrectionScale);
        std::printf("Mass-Schur projection controls scale=%.3e velCorr=%d audit=%d useModelRhs=%d fixedBcRhs=%d directBcRhs=%d directOperator=%d openOutletB=%d\n",
                    massSchurScale, pressureVelocityCorrection,
                    massSchurProjectionAudit, massSchurUseModelContinuityRhs,
                    massSchurUseFixedBcRhs, massSchurUseDirectBcRhs,
                    massSchurUseDirectOperator, massSchurUseOpenOutletB);
        if (massSchurUseOpenOutletB && massSchurUseDirectOperator) {
            std::printf("WARNING: both -massSchurUseOpenOutletB 1 and -massSchurUseDirectOperator 1 requested; MATLAB-open Bopen^T path takes precedence for Lp/correction.\n");
        }
        std::printf("Hybrid cell-Schur controls wP=%.3e wC=%.3e\n",
                    hybridCellPmassWeight, hybridCellSchurWeight);
        std::printf("u MCGS controls           sweeps=%d preSweeps=%d omega=%.6g symmetric=%d\n",
                    uSweeps, uMcgsPreSweeps, uOmega, uSymmetric);
        std::printf("u cell-block controls     correctionIters=%d omega=%.6g\n",
                    uCellBlockIters, uCellBlockOmega);
        std::printf("free-row convergence      velocitySolveUseFreeRows=%d simpleConvergenceUseFreeRows=%d scaleFloor=%.3e print=%d\n",
                    velocitySolveUseFreeRows, simpleConvergenceUseFreeRows,
                    velocityFreeRowsScaleFloor, velocityFreeRowsPrint);
        std::printf("outer convergence gate    requireUpdateRel=%d; if 0, stop is based on relCoupled only\n",
                    simpleConvergenceRequireUpdateRel);
        std::printf("velocity solve audit      enabled=%d every=%d dump=%d dumpEvery=%d prefix=%s\n",
                    velocitySolveAudit, velocitySolveAuditEvery,
                    velocitySolveAuditDump, velocitySolveAuditDumpEvery,
                    velocitySolveAuditPrefix.c_str());
        if (velocitySolveAuditDump || velocitySolveAuditDumpEvery > 0) {
            std::printf("velocity solve audit note = dump knobs are parsed for CLI compatibility; this patch prints text diagnostics only.\n");
        }
        std::printf("inexact Schur controls    inner=%s/%s innerMax=%d innerRel=%.3e pMax=%d pK=%d pRel=%.3e\n",
                    schurInnerSolver.c_str(), schurInnerPrecond.c_str(),
                    schurInnerMaxit, schurInnerRelTol,
                    schurPressureMaxit, schurPressureKdim, schurPressureRelTol);
        std::printf("pSolver/pPrecond        = %s/%s pTol=%.3e pRelTol=%.3e pMaxit=%d\n",
                    pSolver.c_str(), pPrecond.c_str(), pTol, pRelTol, pMaxit);
        std::printf("pAMG robust settings     coarsen=%d interp=%d relax=%d aggLevels=%d keepTranspose=%d pmax=%d sweeps=%d\n",
                    pAmgCoarsenType, pAmgInterpType, pAmgRelaxType, pAmgAggLevels, pAmgKeepTranspose, pAmgPmax, pAmgNumSweeps);
        std::printf("instrumentation          profileTimings=%d hypreProfile=%d pReuseAmg=%d uReuseObjects=%d convGpu=%d errEvery=%d uCheckEvery=%d memEvery=%d\n",
                    profileTimings, hypreProfile, pReuseAmg, uReuseObjects, convGpu, errEvery, uCheckEvery, memEvery);
        std::printf("force postprocess         enable=%d reaction=%d patch=%s normalSign=%d rho=%.6g mu=%.6g Uref=%.6g Aref=%.6g forceEvery=%d\n",
                    forceEnable, forceReactionEnable, forcePatch.c_str(), forceNormalSign, forceRho, forceMu, forceUref, forceAreaRef, forceEvery);
        std::printf("=======================================================\n");

        init_hypre_gpu_runtime(device);
        print_device_info(device);

        Mesh mesh = read_openfoam_polymesh(polyMeshDir);

        resolve_channel_patch_ranges_from_mesh(mesh);

        gChannelXMin =  std::numeric_limits<double>::infinity();
        gChannelXMax = -std::numeric_limits<double>::infinity();
        for (const auto& xp : mesh.P) {
            gChannelXMin = std::min(gChannelXMin, xp[0]);
            gChannelXMax = std::max(gChannelXMax, xp[0]);
        }

        setup_channel_parabolic_box_profile(mesh);

        std::vector<TetP2Geom> tets = reconstruct_tets(mesh);

        std::vector<int> channelPressurePins;
        bool channelPressureOutletActive = false;
        std::vector<char> channelPressureBoundaryFaceMask(mesh.nFaces, 1);

        std::vector<int> channelPressureOutletDirichletNodes;

        if (gCaseIsChannel) {
            std::fill(channelPressureBoundaryFaceMask.begin(), channelPressureBoundaryFaceMask.end(), 0);
            for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
                // Pressure-correction Poisson BC audit/formulation:
                //   inlet/wall: natural Neumann dp'/dn=0 -> no boundary pressure face term
                //   outlet:     p'=0 Dirichlet       -> outlet pressure nodes are pinned
                // Keeping the outlet pressure face term in Ap is harmless once all outlet
                // CG1 pressure nodes are pinned to zero; it also keeps diagnostics explicit.
                if (channel_boundary_kind(mesh, f) == 2) channelPressureBoundaryFaceMask[f] = 1;
            }

            int nOutletFaces = 0;
            int nInletFaces = 0;
            int nWallFaces = 0;
            int pMaskOutletFaces = 0;
            int pMaskInletFaces = 0;
            int pMaskWallFaces = 0;

            for (int f=mesh.nInternalFaces; f<mesh.nFaces; ++f) {
                const int bkind = channel_boundary_kind(mesh, f);
                if (bkind == 2) {
                    nOutletFaces++;
                    if (channelPressureBoundaryFaceMask[f]) pMaskOutletFaces++;
                } else if (bkind == 1) {
                    nInletFaces++;
                    if (channelPressureBoundaryFaceMask[f]) pMaskInletFaces++;
                } else {
                    nWallFaces++;
                    if (channelPressureBoundaryFaceMask[f]) pMaskWallFaces++;
                }
            }

            channelPressureOutletDirichletNodes = collect_channel_outlet_pressure_nodes(mesh, tets);
            channelPressurePins = channelPressureOutletDirichletNodes;
            channelPressureOutletActive = (gChannelPressureOutlet && !channelPressurePins.empty());

            std::printf("Channel mode BCs         xmin=%.17g xmax=%.17g tol=%.3e inletFaces=%d outletFaces=%d wallFaces=%d\n",
                gChannelXMin, gChannelXMax, channel_x_tol(), nInletFaces, nOutletFaces, nWallFaces);
            std::printf("Channel mode note        inlet velocity mode=%s speed=%.6g, walls no-slip, right outlet has no velocity Dirichlet/SIPG term, convective exterior=interior.\n",
                gChannelInletNormalVelocity ? "normal" : "cartesian-x", gChannelInletUx);
            std::printf("Channel strong BC mode   strongVelocityRows=%d constrainVelocityCorrection=%d; if enabled, inlet/wall DG2 rows are eliminated from the velocity solve and pressure correction is zeroed on those rows.\n",
                channelStrongVelocityRows, channelConstrainVelocityCorrection);
            std::printf("Channel pressure outlet  active=%d outletDirichletNodes=%zu mode=%s; singlePinRequest=%d; default active=0 means do-nothing outlet with global pressure mean removal.\n",
                channelPressureOutletActive ? 1 : 0,
                channelPressureOutletDirichletNodes.size(),
                gChannelPressureOutletDirichletAll ? "FULL_OUTLET_DIRICHLET" : (gChannelPressureOutletSinglePin ? "SINGLE_REFERENCE_PIN" : "OUTLET_NODE_SET"),
                gChannelPressureOutletSinglePin);

            std::printf("Channel pressure Poisson BC audit: inlet dpdn=0 natural faces=%d pressureFaceTerms=%d; wall dpdn=0 natural faces=%d pressureFaceTerms=%d; outlet p'=0 faces=%d pressureFaceTerms=%d outletDirichletNodes=%zu\n",
                nInletFaces, pMaskInletFaces,
                nWallFaces, pMaskWallFaces,
                nOutletFaces, pMaskOutletFaces,
                channelPressureOutletDirichletNodes.size());

            if (gCaseIsChannel && gChannelAdjustOutletFlux) {
                std::printf("WARNING: channelAdjustOutletFlux=1 is diagnostic-only and imposes outlet flux redistribution; keep it OFF for general pressure/physics-regulated outlet tests.\n");
            }

            if (gChannelPressureOutlet && gChannelPressureOutletSinglePin && gChannelPressureOutletDirichletAll) {
                std::printf("NOTE: -channelPressureOutletSinglePin 1 was requested, but -channelPressureOutletDirichletAll 1 overrides it and pins the full outlet CG1 pressure node set.\n");
            }

            print_channel_pressure_pin_geometry_diagnostic(mesh, channelPressureOutletDirichletNodes);
        }

        const auto tq = make_tet_quad(quad1d);
        const auto fq = make_tri_quad(quad1d);

        std::printf("tet quadrature points         = %zu\n", tq.size());
        std::printf("face quadrature points        = %zu\n", fq.size());

        check_p2_basis(tets, tq);

        const int nU = 10*mesh.nCells;
        const int nP = 4*mesh.nCells;

        std::vector<double> pWeights(nP, 0.0);

        for (int c=0; c<mesh.nCells; ++c) {
            for (int a=0; a<4; ++a) {
                pWeights[4*c + a] += tets[c].vol/4.0;
            }
        }

        std::vector<double> invPWeights(nP, 0.0);

        for (int i=0; i<nP; ++i) {
            invPWeights[i] = 1.0/std::max(pWeights[i], 1e-300);
        }

        std::vector<double> channelOutletPWeights(nP, 0.0);
        if (gCaseIsChannel) {
            channelOutletPWeights = compute_channel_pressure_face_weights(mesh, tets, fq, 2);
            double outletWsum = 0.0;
            for (double w : channelOutletPWeights) outletWsum += w;
            std::printf("Channel direct continuity outlet pressure weight sum = %.17e\n", outletWsum);
            std::printf("Channel direct continuity RHS active=%d sign=%.6g; compatibility defect is assigned to outlet weights.\n",
                channelDirectContinuity ? 1 : 0, channelDirectContinuitySign);
        }

        CSRPattern uPat = build_dg2_scalar_pattern(mesh);

        CpuMcgsColoring uColoring;
        if (is_any_mcgs_solver_name(uSolver) || (inexactSchurMode0 && is_any_mcgs_solver_name(schurInnerSolver))) {
            uColoring = build_cpu_mcgs_coloring(uPat);
        }

        GpuMcgsSystem uGpuMcgs;
        if (is_gpu_mcgs_solver_name(uSolver)) {
            init_gpu_mcgs_system(uGpuMcgs, uPat, uColoring);
        } else if (is_gpu_cellblock_solver_name(uSolver)) {
            init_gpu_cellblock_system(uGpuMcgs, uPat);
        }

        GpuMcgsSystem schurGpuMcgs;
        if (inexactSchurMode0) {
            if (is_gpu_mcgs_solver_name(schurInnerSolver)) {
                init_gpu_mcgs_system(schurGpuMcgs, uPat, uColoring);
            } else if (is_gpu_cellblock_solver_name(schurInnerSolver)) {
                init_gpu_cellblock_system(schurGpuMcgs, uPat);
            }
        }

        std::printf("\n--- DG2 velocity matrix pattern ---\n");
        std::printf("rows/component                = %d\n", uPat.nRows);
        std::printf("nnz/component                 = %d\n", uPat.nnz);
        std::printf("avg nnz/row                  = %.6f\n", (double)uPat.nnz/std::max(1,uPat.nRows));

        const double tConvCache0 = wall_seconds();
        ConvLFPatternCache convCache = build_conv_lf_pattern_cache(mesh, uPat);
        std::printf("Momentum convection cache     = mode=%s ready=%d volPos=%zu facePos=%zu buildTime=%.6f s\n",
            gMomentumFluxMode.c_str(),
            convCache.ready ? 1 : 0,
            convCache.volPos.size(),
            convCache.facePos.size(),
            wall_seconds() - tConvCache0);

        std::vector<double> Aphys;
        std::vector<double> M;
        std::vector<double> Arel;
        std::array<std::vector<double>,3> F;

        const double tAsm0 = wall_seconds();

        assemble_dg2_sipg_stokes_matrix_rhs(mesh, tets, uPat, tq, fq, nu, sigma, Aphys, M, F);

        std::vector<double> Kdiff = Aphys;

        GpuConvLFState convGpuState;
        bool convGpuActive = false;

        if (convGpu) {
            const double tGpuConv0 = wall_seconds();
            init_gpu_conv_lf_state(convGpuState, mesh, tets, tq, fq, uPat, Kdiff, convCache);
            convGpuActive = convGpuState.ready;
            std::printf("GPU LF convection enabled     = %d initTime=%.6f s\n",
                convGpuActive ? 1 : 0, wall_seconds() - tGpuConv0);
            std::printf("GPU LF convection note        = computes Cbeta/Aphys on GPU, then copies Aphys back to host for current HYPRE update path.\n");
        } else {
            std::printf("GPU LF convection disabled    = use -convGpu 1 to test CUDA convection rebuild.\n");
        }

        std::vector<double> beta0(uPat.nRows, 0.0);
        int initialConvNnz = 0;
        if (convGpuActive) {
            rebuild_physical_operator_from_beta_gpu(convGpuState, Kdiff, beta0, beta0, beta0, Aphys, &initialConvNnz);
        } else {
            rebuild_physical_operator_from_beta(mesh, tets, uPat, tq, fq, Kdiff, beta0, beta0, beta0, Aphys, &initialConvNnz, &convCache);
        }

        Arel = Aphys;

        if (gammaU != 0.0) {
            add_scaled_values(Arel, M, gammaU);
        }

        std::printf("assembly time                 = %.6f s\n", wall_seconds() - tAsm0);
        std::printf("initial nnz(Cbeta)            = %d momentumFlux=%s\n", initialConvNnz, gMomentumFluxMode.c_str());
        std::printf("max |Kdiff-Kdiff^T|           = %.3e\n", max_csr_symmetry_error(uPat, Kdiff));
        std::printf("max |Arel-Arel^T|             = %.3e\n", max_csr_symmetry_error(uPat, Arel));

        RectCSR Apx, Apy, Apz;
        int fx=0, fy=0, fz=0;

        const std::vector<char>* channelPressureBoundaryFaceMaskPtr = gCaseIsChannel ? &channelPressureBoundaryFaceMask : nullptr;
        assemble_Ap_cg1_direction(mesh, tets, tq, fq, 0, Apx, fx, channelPressureBoundaryFaceMaskPtr);
        assemble_Ap_cg1_direction(mesh, tets, tq, fq, 1, Apy, fy, channelPressureBoundaryFaceMaskPtr);
        assemble_Ap_cg1_direction(mesh, tets, tq, fq, 2, Apz, fz, channelPressureBoundaryFaceMaskPtr);

        RectCSR ApxUnmaskedForPressurePrecond = Apx;
        RectCSR ApyUnmaskedForPressurePrecond = Apy;
        RectCSR ApzUnmaskedForPressurePrecond = Apz;

        if (gCaseIsChannel && channelMaskDirichletSchurRows) {
            zero_channel_dirichlet_trace_rows_in_rectcsr(mesh, tets, Apx, "Apx");
            zero_channel_dirichlet_trace_rows_in_rectcsr(mesh, tets, Apy, "Apy");
            zero_channel_dirichlet_trace_rows_in_rectcsr(mesh, tets, Apz, "Apz");
        }

        std::printf("\n--- DG2/DG1 pressure coupling ---\n");
        std::printf("pressure rows                 = %d\n", nP);
        std::printf("Apx/Apy/Apz nnz               = %d %d %d\n", Apx.nnz, Apy.nnz, Apz.nnz);
        std::printf("face mapping failures          = %d\n", fx+fy+fz);

        std::vector<unsigned char> channelVelocityDirichletMask;
        std::vector<double> channelVelocityBcX;
        std::vector<double> channelVelocityBcY;
        std::vector<double> channelVelocityBcZ;

        if (gCaseIsChannel && (channelStrongVelocityRows || channelConstrainVelocityCorrection)) {
            int strongInletRows = 0;
            int strongWallRows = 0;
            int strongSkippedOutletRows = 0;
            int strongSkippedInletWallRows = 0;

            const int strongTotalRows = build_channel_velocity_dirichlet_trace_values(
                mesh, tets,
                channelVelocityDirichletMask,
                channelVelocityBcX,
                channelVelocityBcY,
                channelVelocityBcZ,
                &strongInletRows,
                &strongWallRows,
                &strongSkippedOutletRows,
                &strongSkippedInletWallRows);

            std::printf("Channel strong/correction velocity rows: total=%d inletLike=%d wallLike=%d protectedOutletOverlap=%d protectedInletWallOverlap=%d\n",
                strongTotalRows, strongInletRows, strongWallRows,
                strongSkippedOutletRows, strongSkippedInletWallRows);
        }

        const std::vector<unsigned char>* channelVelocityCorrectionMaskPtr =
            (gCaseIsChannel && channelConstrainVelocityCorrection && !channelVelocityDirichletMask.empty())
                ? &channelVelocityDirichletMask
                : nullptr;

        std::vector<std::array<std::array<double,10>,10>> invM;
        CSRPattern lpPat;
        std::vector<double> lpVals;
        RectCSR directBdx, directBdy, directBdz;
        std::vector<double> directBfixedSource;
        CSRPattern lpSplitPrePat;
        std::vector<double> lpSplitPreVals;
        const bool pressureSplitPrecondRequested =
            (pSplitPrecondUnmaskedMassSchur != 0 &&
             pressureMode == "massschur" &&
             pSolver == "pcg" &&
             pPrecond == "amg");
        const bool scalarArelSchurMode0 =
            (pressureMode == "diagschur" || pressureMode == "diag" ||
             pressureMode == "rowsumschur" || pressureMode == "rowsschur" ||
             pressureMode == "rowsum" || pressureMode == "row");
        const bool hybridCellSchurMode0 =
            (pressureMode == "hybridcellschur" || pressureMode == "hybridcell" ||
             pressureMode == "hybridblockschur" || pressureMode == "hybridarelcellschur");
        const bool cellBlockSchurMode0 =
            (pressureMode == "cellblockschur" || pressureMode == "cellschur" ||
             pressureMode == "blockschur" || pressureMode == "arelcellschur" ||
             hybridCellSchurMode0);

        if (pressureMode == "massschur" || pressureMode == "hybridmass" || scalarArelSchurMode0 || cellBlockSchurMode0) {
            const double tLp0 = wall_seconds();

            if (pressureMode == "massschur" || pressureMode == "hybridmass") {
                invM = compute_p2_mass_inverses(mesh, tets, tq);

                if ((massSchurUseDirectOperator || massSchurUseOpenOutletB) && gCaseIsChannel) {
                    assemble_channel_direct_continuity_velocity_to_pressure(
                        mesh, tets, tq, fq,
                        directBdx, directBdy, directBdz,
                        directBfixedSource);

                    // For DG2/DG1, directB*d is B^T.  The momentum pressure operator is
                    // G=-B^T because the predictor uses rhs = F - G*p.  This mirrors
                    // the MATLAB reference where G=-B'.
                    Apx = negated_rectcsr(directBdx);
                    Apy = negated_rectcsr(directBdy);
                    Apz = negated_rectcsr(directBdz);

                    if (massSchurUseOpenOutletB) {
                        assemble_lpmass_open_transpose_schur(
                            nP, mesh,
                            directBdx, directBdy, directBdz,
                            invM,
                            channelVelocityCorrectionMaskPtr,
                            lpPat, lpVals);

                        std::printf("Mass-Schur operator source     = MATLAB_OPEN Bopen * Pfree * M^{-1} * Pfree * Bopen^T\n");
                    } else {
                        assemble_lpmass_direct_continuity_schur(
                            nP, mesh,
                            directBdx, directBdy, directBdz,
                            Apx, Apy, Apz,
                            invM,
                            lpPat, lpVals);

                        std::printf("Mass-Schur operator source     = DIRECT_CONTINUITY * (-M^{-1} Ap)\n");
                    }
                } else {
                    assemble_lpmass_schur(nP, mesh, Apx, Apy, Apz, invM, lpPat, lpVals);
                    std::printf("Mass-Schur operator source     = Ap^T M^{-1} Ap\n");
                }
            } else if (cellBlockSchurMode0) {
                std::vector<std::array<std::array<double,10>,10>> invA0;
                build_cellblock_jacobi_inverses_from_csr(uPat, Arel, schurBlockShift, schurBlockPivotFloor, invA0);
                assemble_lp_cellblock_schur_from_invblocks(nP, Apx, Apy, Apz, invA0, lpPat, lpVals);
            } else {
                std::vector<double> hInv0 = make_scalar_schur_inverse_from_Arel(uPat, Arel, pressureMode);
                assemble_lp_scalar_schur(nP, Apx, Apy, Apz, hInv0, lpPat, lpVals);
            }

            if (massSchurScale != 1.0) {
                for (double& v : lpVals) v *= massSchurScale;
            }

            if (gCaseIsChannel && channelPressureOutletPenaltyBeta > 0.0) {
                add_channel_outlet_pressure_penalty_to_csr(
                    mesh, tets, fq,
                    channelPressureOutletPenaltyBeta,
                    lpPat, lpVals);
            }

            if (channelPressureOutletActive) {
                pin_csr_symmetric_zero_set(lpPat, lpVals, channelPressurePins);
                force_csr_structural_identity_pins(lpPat, lpVals, channelPressurePins);
            }

            if (pressureRepairZeroRows) {
                const int addedZeroPins = append_pressure_zero_rows_to_pin_set(
                    "masked Lp",
                    mesh,
                    lpPat,
                    lpVals,
                    channelPressurePins,
                    1e-30,
                    1e-30,
                    20);

                if (addedZeroPins > 0) {
                    pin_csr_symmetric_zero_set(lpPat, lpVals, channelPressurePins);
                    force_csr_structural_identity_pins(lpPat, lpVals, channelPressurePins);
                }
            }

            if (pressureSplitPrecondRequested) {
                assemble_lpmass_schur(
                    nP, mesh,
                    ApxUnmaskedForPressurePrecond,
                    ApyUnmaskedForPressurePrecond,
                    ApzUnmaskedForPressurePrecond,
                    invM,
                    lpSplitPrePat,
                    lpSplitPreVals);

                if (massSchurScale != 1.0) {
                    for (double& v : lpSplitPreVals) v *= massSchurScale;
                }

                if (channelPressureOutletActive) {
                    pin_csr_symmetric_zero_set(lpSplitPrePat, lpSplitPreVals, channelPressurePins);
                    force_csr_structural_identity_pins(lpSplitPrePat, lpSplitPreVals, channelPressurePins);
                }

                std::printf("\n--- Split-preconditioner pressure operator for AMG ---\n");
                std::printf("LpPre rows                    = %d\n", lpSplitPrePat.nRows);
                std::printf("LpPre nnz                     = %d\n", lpSplitPrePat.nnz);
                std::printf("LpPre source                  = unmasked massschur Ap^T M^{-1} Ap; Krylov operator remains masked Lp\n");
                std::printf("max |LpPre-LpPre^T|           = %.3e\n", max_csr_symmetry_error(lpSplitPrePat, lpSplitPreVals));
                std::printf("max |rowSum(LpPre)|           = %.3e\n", max_csr_row_sum_abs(lpSplitPrePat, lpSplitPreVals));
            }

            std::printf("\n--- Mass-Schur pressure operator for SIMPLE correction ---\n");
            std::printf("Lp rows                       = %d\n", lpPat.nRows);
            std::printf("Lp nnz                        = %d\n", lpPat.nnz);
            std::printf("avg nnz/row                   = %.6f\n", (double)lpPat.nnz/std::max(1,lpPat.nRows));
            std::printf("max |Lp-Lp^T|                 = %.3e\n", max_csr_symmetry_error(lpPat, lpVals));
            std::printf("max |rowSum(Lp)|              = %.3e\n", max_csr_row_sum_abs(lpPat, lpVals));
            audit_pressure_csr_rows_for_amg("masked Lp", mesh, lpPat, lpVals, 40);
            if (channelPressureOutletActive) {
                std::printf("Lp pressure outlet Dirichlet  = %zu outlet CG1 nodes with p'=0 and p=0; total pressure pins incl. repairs=%zu\n",
                    channelPressureOutletDirichletNodes.size(), channelPressurePins.size());
            }
            std::printf("Lp assembly time              = %.6f s\n", wall_seconds() - tLp0);
            std::printf("Pressure AMG reuse is controlled by -pReuseAmg; Lp itself remains numerically frozen.\n");
        } else if (pressureMode != "pmass" && !inexactSchurMode0) {
            throw std::runtime_error("Unknown -pressureMode. Use pmass, massschur, hybridmass, hybridcellschur, cellblockschur, inexactschur, diagschur, or rowsumschur.");
        }
        if ((scalarArelSchurMode0 || cellBlockSchurMode0) && pReuseAmg) {
            std::printf("Scalar Arel-Schur mode rebuilds Lp each SIMPLE step; reusable pressure AMG is not used for this experimental path.\n");
        }

        HypreOptions uopt;
        uopt.solver = uSolver;
        uopt.precond = uPrecond;
        uopt.maxIter = uMaxit;
        uopt.tol = uAbsTol;
        uopt.absTol = uAbsTol;
        uopt.relTol = uRelTol;
        uopt.monitor = uMonitor;
        uopt.gmresKDim = uKDim;
        uopt.fgmresInnerMaxIter = uInnerMaxit;
        uopt.fgmresInnerKDim = uInnerKDim;
        uopt.fgmresInnerRelTol = uInnerRelTol;
        uopt.fgmresInnerAbsTol = uInnerAbsTol;
        uopt.profile = hypreProfile;

        // If velocity AMG is deliberately tested, use the same GPU-safe choices.
        uopt.amgCoarsenType = pAmgCoarsenType;
        uopt.amgInterpType = pAmgInterpType;
        uopt.amgRelaxType = pAmgRelaxType;
        uopt.amgAggLevels = pAmgAggLevels;
        uopt.amgKeepTranspose = pAmgKeepTranspose;
        uopt.amgPmax = pAmgPmax;
        uopt.amgNumSweeps = pAmgNumSweeps;
        uopt.amgStrongThreshold = pAmgStrongThreshold;

        if (uPrecond == "lumpeddiag" || uPrecond == "l1row" || uPrecond == "rowsuml1") {
            uLeftScaleMode = "l1row";
            uopt.precond = "none";
        } else if (uPrecond == "diagabs") {
            uLeftScaleMode = "diagabs";
            uopt.precond = "none";
        } else if (uPrecond == "cellblock" || uPrecond == "blockjacobi") {
            uLeftScaleMode = "cellblock";
            uopt.precond = "none";
        }

        HypreOptions popt;
        popt.solver = pSolver;
        popt.precond = pPrecond;
        popt.maxIter = pMaxit;
        popt.tol = pTol;
        popt.absTol = pTol;
        popt.relTol = pRelTol;
        popt.monitor = pMonitor;
        popt.gmresKDim = 80;
        popt.amgCoarsenType = pAmgCoarsenType;
        popt.amgInterpType = pAmgInterpType;
        popt.amgRelaxType = pAmgRelaxType;
        popt.amgAggLevels = pAmgAggLevels;
        popt.amgKeepTranspose = pAmgKeepTranspose;
        popt.amgPmax = pAmgPmax;
        popt.amgNumSweeps = pAmgNumSweeps;
        popt.amgStrongThreshold = pAmgStrongThreshold;
        popt.profile = hypreProfile;

        HypreReusableSystem pressureReusable;
        bool pressureReusableActive = false;

        HypreReusableSystem pressureSplitPrecondReusable;
        bool pressureSplitPrecondActive = false;
        HypreOptions pSplitPrecondOpt = popt;

        if (pressureSplitPrecondRequested) {
            if (lpSplitPrePat.nRows <= 0 || lpSplitPreVals.empty()) {
                throw std::runtime_error("Split preconditioner requested but lpSplitPrePat/lpSplitPreVals were not built");
            }

            pSplitPrecondOpt.solver = "pcg";
            pSplitPrecondOpt.precond = "amg";
            pSplitPrecondOpt.maxIter = pSplitPrecondInnerMaxit;
            pSplitPrecondOpt.relTol = pSplitPrecondInnerRelTol;
            pSplitPrecondOpt.absTol = pSplitPrecondInnerAbsTol;
            pSplitPrecondOpt.tol = pSplitPrecondInnerAbsTol;
            pSplitPrecondOpt.monitor = pSplitPrecondInnerMonitor;
            pSplitPrecondOpt.profileLabel = "split-preconditioner-unmasked-Lp";

            const double tSplitSetup0 = wall_seconds();
            if (pSplitPrecondReusable) {
                init_reusable_hypre_csr_vec(
                    pressureSplitPrecondReusable,
                    lpSplitPrePat,
                    lpSplitPreVals,
                    pSplitPrecondOpt);
            }
            pressureSplitPrecondActive = true;

            std::printf("Pressure split PCG enabled: operator=masked massschur Lp, AMG-preconditioner=unmasked massschur Lp, setupTime=%.6f s rows=%d nnz=%d innerMaxit=%d innerRelTol=%.3e reusableInner=%d\n",
                wall_seconds() - tSplitSetup0,
                lpSplitPrePat.nRows,
                lpSplitPrePat.nnz,
                pSplitPrecondInnerMaxit,
                pSplitPrecondInnerRelTol,
                pSplitPrecondReusable);
        }

        if ((pressureMode == "massschur" || pressureMode == "hybridmass") && pReuseAmg && !pressureSplitPrecondActive) {
            if (pSolver == "pcg" && pPrecond == "amg") {
                HypreOptions poptInit = popt;
                poptInit.profileLabel = "pressure-Lp initial setup";
                const double tReuse0 = wall_seconds();
                init_reusable_hypre_csr_vec(pressureReusable, lpPat, lpVals, poptInit);
                pressureReusableActive = true;
                std::printf("Pressure reusable PCG/AMG setup enabled: setupTime=%.6f s rows=%d nnz=%d\n",
                    wall_seconds() - tReuse0, lpPat.nRows, lpPat.nnz);
            } else {
                std::printf("Pressure reusable PCG/AMG requested but disabled because pSolver/pPrecond=%s/%s; falling back to one-shot HYPRE solves.\n",
                    pSolver.c_str(), pPrecond.c_str());
            }
        } else if (pressureMode == "massschur" || pressureMode == "hybridmass") {
            std::printf("Pressure reusable PCG/AMG disabled by -pReuseAmg 0; one-shot pressure solves will rebuild setup.\n");
        }

        HypreReusableSystem velocityReusable;
        bool velocityReusableActive = false;

        if (uReuseObjects) {
            const bool solverOk = (uSolver == "bicgstab" || uSolver == "bicg" || uSolver == "pcg");
            const bool precondOk = (uopt.precond == "diagscale" || uopt.precond == "none" || uopt.precond == "amg");
            const bool leftScaleOk = (uLeftScaleMode == "none");

            if (solverOk && precondOk && leftScaleOk) {
                HypreOptions uoptInit = uopt;
                uoptInit.profileLabel = "velocity-Arel initial reusable setup";
                const double tUReuse0 = wall_seconds();
                init_reusable_hypre_csr_vec(velocityReusable, uPat, Arel, uoptInit);
                velocityReusableActive = true;
                std::printf("Velocity reusable HYPRE object enabled: setupTime=%.6f s rows=%d nnz=%d solver=%s precond=%s\n",
                    wall_seconds() - tUReuse0, uPat.nRows, uPat.nnz, uopt.solver.c_str(), uopt.precond.c_str());
                std::printf("Velocity reusable mode updates Arel once for Ux each SIMPLE step, then reuses the same matrix setup for Uy/Uz RHS-only solves.\n");
            } else {
                std::printf("Velocity reusable HYPRE requested but disabled: solverOk=%d precondOk=%d leftScaleOk=%d solver=%s precond=%s leftScale=%s. Falling back to one-shot velocity solves.\n",
                    (int)solverOk, (int)precondOk, (int)leftScaleOk, uSolver.c_str(), uopt.precond.c_str(), uLeftScaleMode.c_str());
            }
        } else {
            std::printf("Velocity reusable HYPRE object disabled by -uReuseObjects 0; one-shot velocity solves will rebuild IJ objects.\n");
        }
        if (gCaseIsChannel && channelStrongVelocityRows && velocityReusableActive) {
            velocityReusableActive = false;
            std::printf("Velocity reusable HYPRE object disabled because -channelStrongVelocityRows 1 builds a constrained velocity matrix for the predictor.\n");
        }

        std::vector<double> ux(nU,0.0);
        std::vector<double> uy(nU,0.0);
        std::vector<double> uz(nU,0.0);
        std::vector<double> p(nP,0.0);

        if (gCaseIsChannel && !channelVelocityDirichletMask.empty()) {
            impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcX, ux);
            impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcY, uy);
            impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcZ, uz);
        }

        auto finalize_pressure_rhs = [&](std::vector<double>& rhsP) {
            if (channelPressureOutletActive) {
                zero_values_at_indices(rhsP, channelPressurePins);
            } else if (gCaseIsChannel && channelDirectContinuity) {
                if (gChannelOutletCompatibility) {
                    make_rhs_compatible_by_outlet_weights(rhsP, channelOutletPWeights);
                } else {
                    make_pressure_rhs_compatible(rhsP);
                }
            } else {
                make_pressure_rhs_compatible(rhsP);
            }
        };

        auto finalize_pressure_correction = [&](std::vector<double>& pc) {
            if (channelPressureOutletActive) {
                zero_values_at_indices(pc, channelPressurePins);
            } else {
                subtract_weighted_mean(pc, pWeights);
            }
        };

        auto finalize_pressure_field = [&](std::vector<double>& pp) {
            if (channelPressureOutletActive) {
                zero_values_at_indices(pp, channelPressurePins);
            } else {
                subtract_weighted_mean(pp, pWeights);
            }
        };

        double coupled0 = 1.0;
        std::vector<double> massSchurAuditBoundaryContribution;

        auto print_massschur_projection_audit = [&](int it,
               const char* stage,
               const std::vector<double>& rhsPUsed,
               const std::vector<double>& pcorrUsed,
               const std::vector<double>& uxStar,
               const std::vector<double>& uyStar,
               const std::vector<double>& uzStar,
               const std::vector<double>& uxNow,
               const std::vector<double>& uyNow,
               const std::vector<double>& uzNow)
        {
            if (!massSchurProjectionAudit) return;
            if (!(it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) return;
            if (rhsPUsed.empty() || pcorrUsed.empty()) return;

            auto norm_excluding_pressure_pins = [&](const std::vector<double>& v) {
                if (!channelPressureOutletActive || channelPressurePins.empty()) return norm_vec(v);
                std::vector<double> tmp = v;
                zero_values_at_indices(tmp, channelPressurePins);
                return norm_vec(tmp);
            };

            std::vector<double> lpPcorr;
            apply_csr(lpPat, lpVals, pcorrUsed, lpPcorr);

            std::vector<double> rhsMinusLp = rhsPUsed;
            axpy_vec(rhsMinusLp, -1.0, lpPcorr);

            std::vector<double> rhsPlusLp = rhsPUsed;
            axpy_vec(rhsPlusLp, 1.0, lpPcorr);

            std::vector<double> bStarModel;
            std::vector<double> bNowModel;
            apply_divergence_from_Ap(Apx, Apy, Apz, uxStar, uyStar, uzStar, bStarModel);
            apply_divergence_from_Ap(Apx, Apy, Apz, uxNow, uyNow, uzNow, bNowModel);

            std::vector<double> rhsPlusBstar = rhsPUsed;
            axpy_vec(rhsPlusBstar, 1.0, bStarModel);

            const bool haveBcContribution =
                ((int)massSchurAuditBoundaryContribution.size() == (int)rhsPUsed.size());

            std::vector<double> rhsPlusBstarPlusBc = rhsPlusBstar;
            std::vector<double> bNowModelPlusBc = bNowModel;

            if (haveBcContribution) {
                axpy_vec(rhsPlusBstarPlusBc, 1.0, massSchurAuditBoundaryContribution);
                axpy_vec(bNowModelPlusBc, 1.0, massSchurAuditBoundaryContribution);
            }

            std::vector<double> modelPred = bStarModel;
            axpy_vec(modelPred, 1.0, lpPcorr);

            std::vector<double> modelIdentityDrift = bNowModel;
            axpy_vec(modelIdentityDrift, -1.0, modelPred);

            std::printf(
                "    massSchurProjectionAudit: it=%d stage=%s "
                "||rhs||=%.6e ||Lp pcorr||=%.6e "
                "||rhs-Lp pcorr||=%.6e ||rhs+Lp pcorr||=%.6e "
                "||rhs+Bmodel u*||=%.6e "
                "||Bmodel u*||=%.6e ||Bmodel u_now||=%.6e "
                "||Bmodel u_now-(Bmodel u*+Lp pcorr)||=%.6e "
                "exclPins: ||rhs-Lp pcorr||=%.6e ||Bmodel u_now||=%.6e\n",
                it, stage,
                norm_vec(rhsPUsed), norm_vec(lpPcorr),
                norm_vec(rhsMinusLp), norm_vec(rhsPlusLp),
                norm_vec(rhsPlusBstar),
                norm_vec(bStarModel), norm_vec(bNowModel),
                norm_vec(modelIdentityDrift),
                norm_excluding_pressure_pins(rhsMinusLp),
                norm_excluding_pressure_pins(bNowModel));

            if (haveBcContribution) {
                std::printf(
                    "    massSchurProjectionAuditBc: it=%d stage=%s "
                    "||gBC||=%.6e ||rhs+Bmodel u*+gBC||=%.6e "
                    "||Bmodel u_now+gBC||=%.6e "
                    "exclPins: ||rhs+Bmodel u*+gBC||=%.6e ||Bmodel u_now+gBC||=%.6e\n",
                    it, stage,
                    norm_vec(massSchurAuditBoundaryContribution),
                    norm_vec(rhsPlusBstarPlusBc),
                    norm_vec(bNowModelPlusBc),
                    norm_excluding_pressure_pins(rhsPlusBstarPlusBc),
                    norm_excluding_pressure_pins(bNowModelPlusBc));
            }

            if (gCaseIsChannel && massSchurUseOpenOutletB && directBdx.nRows > 0) {
                std::vector<double> rOpenStar;
                std::vector<double> rOpenNow;
                apply_channel_direct_continuity_linear(
                    directBdx, directBdy, directBdz, directBfixedSource,
                    uxStar, uyStar, uzStar,
                    rOpenStar);
                apply_channel_direct_continuity_linear(
                    directBdx, directBdy, directBdz, directBfixedSource,
                    uxNow, uyNow, uzNow,
                    rOpenNow);

                std::vector<double> rOpenPred = rOpenStar;
                axpy_vec(rOpenPred, 1.0, lpPcorr);

                std::vector<double> openIdentityDrift = rOpenNow;
                axpy_vec(openIdentityDrift, -1.0, rOpenPred);

                std::vector<double> rhsPlusOpenStar = rhsPUsed;
                axpy_vec(rhsPlusOpenStar, 1.0, rOpenStar);

                std::printf(
                    "    openBProjectionAudit: it=%d stage=%s "
                    "||rOpen*||=%.6e ||rOpen_now||=%.6e "
                    "||rOpen*+Lp pcorr||=%.6e "
                    "||rOpen_now-(rOpen*+Lp pcorr)||=%.6e "
                    "||rhs+rOpen*||=%.6e "
                    "exclPins: ||rOpen_now||=%.6e ||rOpen*+Lp pcorr||=%.6e ||rhs+rOpen*||=%.6e\n",
                    it, stage,
                    norm_vec(rOpenStar),
                    norm_vec(rOpenNow),
                    norm_vec(rOpenPred),
                    norm_vec(openIdentityDrift),
                    norm_vec(rhsPlusOpenStar),
                    norm_excluding_pressure_pins(rOpenNow),
                    norm_excluding_pressure_pins(rOpenPred),
                    norm_excluding_pressure_pins(rhsPlusOpenStar));
            }

            if (gCaseIsChannel) {
                std::vector<double> bDirectNow;
                double auditIn = 0.0;
                double auditOut = 0.0;
                double auditNet = 0.0;
                double auditSum = 0.0;
                compute_channel_cg1_continuity_residual_direct(
                    mesh, tets, tq, fq, uxNow, uyNow, uzNow, bDirectNow,
                    &auditIn, &auditOut, &auditNet, &auditSum);
                if (channelDirectContinuitySign != 1.0) {
                    for (double& v : bDirectNow) v *= channelDirectContinuitySign;
                    auditSum *= channelDirectContinuitySign;
                }

                std::vector<double> modelMinusDirect = bNowModel;
                axpy_vec(modelMinusDirect, -1.0, bDirectNow);

                std::printf(
                    "    massSchurProjectionAudit: it=%d stage=%s "
                    "||Bdirect u_now||=%.6e sumDirect=% .6e "
                    "flux(in,out,net)=(% .6e,% .6e,% .6e) "
                    "||Bmodel-Bdirect||=%.6e\n",
                    it, stage,
                    norm_vec(bDirectNow), auditSum,
                    auditIn, auditOut, auditNet,
                    norm_vec(modelMinusDirect));
            }
        };

        std::printf("\n--- PMASS/Uzawa SIMPLE iterations ---\n");
        std::printf("it  uItsX uItsY uItsZ  uFinX       uFinY       uFinZ       ||B u*||     ||B u||      momRes      momResFree  relCoupled  ||p'||      pUpdRel    uUpdRel    UrelL2     PrelL2\n");

        for (int it=1; it<=nSimple; ++it) {
            const double tIter0 = wall_seconds();
            double tmCopyOld = 0.0;
            double tmRhsBuild = 0.0;
            double tmUSolve = 0.0;
            double tmBuStar = 0.0;
            double tmPmass = 0.0;
            double tmSchurBuild = 0.0;
            double tmPSolve = 0.0;
            double tmVelCorr = 0.0;
            double tmPUpdate = 0.0;
            double tmConvRebuild = 0.0;
            double tmArelUpdate = 0.0;
            double tmBuNew = 0.0;
            double tmMomDiag = 0.0;
            double tmErrDiag = 0.0;
            const bool doUCheckThisIter =
                (uCheckEvery > 0) && (it == 1 || it % uCheckEvery == 0);

            std::vector<double> AvelStrong;
            const std::vector<double>* velocityAForSolvePtr = &Arel;

            if (gCaseIsChannel && channelStrongVelocityRows) {
                AvelStrong = Arel;

                const int nStrongMatrixRows =
                    apply_channel_strong_velocity_matrix(uPat, AvelStrong, channelVelocityDirichletMask);

                velocityAForSolvePtr = &AvelStrong;

                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    std::printf("    channelStrongVelocityRows: it=%d eliminatedRows=%d matrix=rows+columns zeroed, diag=1; RHS receives column-lifted Dirichlet values.\n",
                        it, nStrongMatrixRows);
                }
            }

            double tPhase = wall_seconds();

            const std::vector<double> uxOld = ux;
            const std::vector<double> uyOld = uy;
            const std::vector<double> uzOld = uz;
            const std::vector<double> pOld = p;
            tmCopyOld += wall_seconds() - tPhase;

            const bool doVelocitySolveAuditThisIter =
                (velocitySolveAudit != 0) &&
                (it == 1 || it == nSimple || velocitySolveAuditEvery <= 1 || (it % std::max(1, velocitySolveAuditEvery)) == 0);
            const std::vector<unsigned char>* velocitySolveAuditLockedRows =
                !channelVelocityDirichletMask.empty() ? &channelVelocityDirichletMask : nullptr;

            std::vector<double> gp;
            std::vector<double> rhs;
            std::vector<double> MuOld;
            std::vector<double> convBCx, convBCy, convBCz;

            if (gCaseIsChannel) {
                tPhase = wall_seconds();
                assemble_channel_convection_lf_boundary_rhs(
                    mesh, tets, fq, uxOld, uyOld, uzOld,
                    convBCx, convBCy, convBCz);
                tmRhsBuild += wall_seconds() - tPhase;
            }

            tPhase = wall_seconds();
            apply_gradient_rhs(Apx, pOld, gp);
            rhs = F[0];
            if (gCaseIsChannel) axpy_vec(rhs, 1.0, convBCx);
            axpy_vec(rhs, -1.0, gp);

            if (gammaU != 0.0) {
                apply_csr(uPat, M, uxOld, MuOld);
                axpy_vec(rhs, gammaU, MuOld);
            }
            if (gCaseIsChannel && channelStrongVelocityRows) {
                apply_channel_strong_velocity_rhs(uPat, Arel, rhs, channelVelocityDirichletMask, channelVelocityBcX);
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcX, ux);
            }
            tmRhsBuild += wall_seconds() - tPhase;

            HypreOptions uoptX = uopt;
            if (hypreProfile) uoptX.profileLabel = "Ux it=" + std::to_string(it);
            const std::vector<double> uxSolveInput = ux;
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                const auto vfInfo = velocity_make_free_rows_tol_info(
                    uPat, *velocityAForSolvePtr, rhs, ux,
                    channelVelocityCorrectionMaskPtr,
                    uoptX.relTol, uoptX.absTol, velocityFreeRowsScaleFloor);
                uoptX.relTol = vfInfo.effectiveRelTol;
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowTol: it=%d comp=X requestedRel=%.3e effectiveRel=%.3e rhsAll=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, vfInfo.requestedRelTol, vfInfo.effectiveRelTol,
                        vfInfo.rhsAll, vfInfo.rhsFree, vfInfo.axFree, vfInfo.scaleFree);
                }
            }
tPhase = wall_seconds();
            HypreSolveInfo ix = velocityReusableActive
                ? solve_reusable_hypre_matrix_rhs_vec(velocityReusable, uPat, *velocityAForSolvePtr, rhs, ux, uoptX, doUCheckThisIter)
                : solve_velocity_dispatch(uPat, *velocityAForSolvePtr, rhs, ux, uoptX, uSolver, uLeftScaleMode, uLeftScaleEps, uColoring, &uGpuMcgs, uSweeps, uMcgsPreSweeps, uOmega, uSymmetric, uCellBlockIters, uCellBlockOmega, uMonitor);
            if (gCaseIsChannel && channelStrongVelocityRows) {
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcX, ux);
            }
            if (doVelocitySolveAuditThisIter) {
                print_velocity_solve_audit("X", it, uSolver,
                    uPat, *velocityAForSolvePtr, rhs, uxSolveInput, ux,
                    velocitySolveAuditLockedRows, ix,
                    uoptX.maxIter, uoptX.relTol, uoptX.absTol);
            }
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                double vfAbs = 0.0, vfRhs = 0.0, vfAx = 0.0, vfScale = 0.0;
                const double vfRel = velocity_free_rows_relative_residual(
                    uPat, *velocityAForSolvePtr, rhs, ux,
                    channelVelocityCorrectionMaskPtr, velocityFreeRowsScaleFloor,
                    &vfAbs, &vfRhs, &vfAx, &vfScale);
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowFinal: it=%d comp=X oldReported=%.6e freeRel=%.6e absFree=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, ix.finalRelResNorm, vfRel, vfAbs, vfRhs, vfAx, vfScale);
                }
                ix.finalRelResNorm = vfRel;
            }
tmUSolve += wall_seconds() - tPhase;

            tPhase = wall_seconds();
            apply_gradient_rhs(Apy, pOld, gp);
            rhs = F[1];
            if (gCaseIsChannel) axpy_vec(rhs, 1.0, convBCy);
            axpy_vec(rhs, -1.0, gp);

            if (gammaU != 0.0) {
                apply_csr(uPat, M, uyOld, MuOld);
                axpy_vec(rhs, gammaU, MuOld);
            }
            if (gCaseIsChannel && channelStrongVelocityRows) {
                apply_channel_strong_velocity_rhs(uPat, Arel, rhs, channelVelocityDirichletMask, channelVelocityBcY);
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcY, uy);
            }
            tmRhsBuild += wall_seconds() - tPhase;

            HypreOptions uoptY = uopt;
            if (hypreProfile) uoptY.profileLabel = "Uy it=" + std::to_string(it);
            const std::vector<double> uySolveInput = uy;
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                const auto vfInfo = velocity_make_free_rows_tol_info(
                    uPat, *velocityAForSolvePtr, rhs, uy,
                    channelVelocityCorrectionMaskPtr,
                    uoptY.relTol, uoptY.absTol, velocityFreeRowsScaleFloor);
                uoptY.relTol = vfInfo.effectiveRelTol;
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowTol: it=%d comp=Y requestedRel=%.3e effectiveRel=%.3e rhsAll=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, vfInfo.requestedRelTol, vfInfo.effectiveRelTol,
                        vfInfo.rhsAll, vfInfo.rhsFree, vfInfo.axFree, vfInfo.scaleFree);
                }
            }
tPhase = wall_seconds();
            HypreSolveInfo iy = velocityReusableActive
                ? solve_reusable_hypre_rhs_vec(velocityReusable, rhs, uy, uoptY)
                : solve_velocity_dispatch(uPat, *velocityAForSolvePtr, rhs, uy, uoptY, uSolver, uLeftScaleMode, uLeftScaleEps, uColoring, &uGpuMcgs, uSweeps, uMcgsPreSweeps, uOmega, uSymmetric, uCellBlockIters, uCellBlockOmega, uMonitor);
            if (velocityReusableActive && doUCheckThisIter) {
                std::vector<double> AyCheck;
                apply_csr(uPat, *velocityAForSolvePtr, uy, AyCheck);
                axpy_vec(AyCheck, -1.0, rhs);
                iy.finalRelResNorm = norm_vec(AyCheck) / std::max(norm_vec(rhs), 1e-300);
            }
            if (gCaseIsChannel && channelStrongVelocityRows) {
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcY, uy);
            }
            if (doVelocitySolveAuditThisIter) {
                print_velocity_solve_audit("Y", it, uSolver,
                    uPat, *velocityAForSolvePtr, rhs, uySolveInput, uy,
                    velocitySolveAuditLockedRows, iy,
                    uoptY.maxIter, uoptY.relTol, uoptY.absTol);
            }
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                double vfAbs = 0.0, vfRhs = 0.0, vfAx = 0.0, vfScale = 0.0;
                const double vfRel = velocity_free_rows_relative_residual(
                    uPat, *velocityAForSolvePtr, rhs, uy,
                    channelVelocityCorrectionMaskPtr, velocityFreeRowsScaleFloor,
                    &vfAbs, &vfRhs, &vfAx, &vfScale);
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowFinal: it=%d comp=Y oldReported=%.6e freeRel=%.6e absFree=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, iy.finalRelResNorm, vfRel, vfAbs, vfRhs, vfAx, vfScale);
                }
                iy.finalRelResNorm = vfRel;
            }
tmUSolve += wall_seconds() - tPhase;

            tPhase = wall_seconds();
            apply_gradient_rhs(Apz, pOld, gp);
            rhs = F[2];
            if (gCaseIsChannel) axpy_vec(rhs, 1.0, convBCz);
            axpy_vec(rhs, -1.0, gp);

            if (gammaU != 0.0) {
                apply_csr(uPat, M, uzOld, MuOld);
                axpy_vec(rhs, gammaU, MuOld);
            }
            if (gCaseIsChannel && channelStrongVelocityRows) {
                apply_channel_strong_velocity_rhs(uPat, Arel, rhs, channelVelocityDirichletMask, channelVelocityBcZ);
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcZ, uz);
            }
            tmRhsBuild += wall_seconds() - tPhase;

            HypreOptions uoptZ = uopt;
            if (hypreProfile) uoptZ.profileLabel = "Uz it=" + std::to_string(it);
            const std::vector<double> uzSolveInput = uz;
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                const auto vfInfo = velocity_make_free_rows_tol_info(
                    uPat, *velocityAForSolvePtr, rhs, uz,
                    channelVelocityCorrectionMaskPtr,
                    uoptZ.relTol, uoptZ.absTol, velocityFreeRowsScaleFloor);
                uoptZ.relTol = vfInfo.effectiveRelTol;
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowTol: it=%d comp=Z requestedRel=%.3e effectiveRel=%.3e rhsAll=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, vfInfo.requestedRelTol, vfInfo.effectiveRelTol,
                        vfInfo.rhsAll, vfInfo.rhsFree, vfInfo.axFree, vfInfo.scaleFree);
                }
            }
tPhase = wall_seconds();
            HypreSolveInfo iz = velocityReusableActive
                ? solve_reusable_hypre_rhs_vec(velocityReusable, rhs, uz, uoptZ)
                : solve_velocity_dispatch(uPat, *velocityAForSolvePtr, rhs, uz, uoptZ, uSolver, uLeftScaleMode, uLeftScaleEps, uColoring, &uGpuMcgs, uSweeps, uMcgsPreSweeps, uOmega, uSymmetric, uCellBlockIters, uCellBlockOmega, uMonitor);
            if (velocityReusableActive && doUCheckThisIter) {
                std::vector<double> AzCheck;
                apply_csr(uPat, *velocityAForSolvePtr, uz, AzCheck);
                axpy_vec(AzCheck, -1.0, rhs);
                iz.finalRelResNorm = norm_vec(AzCheck) / std::max(norm_vec(rhs), 1e-300);
            }
            if (gCaseIsChannel && channelStrongVelocityRows) {
                impose_channel_velocity_mask_values(channelVelocityDirichletMask, channelVelocityBcZ, uz);
            }
            if (doVelocitySolveAuditThisIter) {
                print_velocity_solve_audit("Z", it, uSolver,
                    uPat, *velocityAForSolvePtr, rhs, uzSolveInput, uz,
                    velocitySolveAuditLockedRows, iz,
                    uoptZ.maxIter, uoptZ.relTol, uoptZ.absTol);
            }
                        if (velocitySolveUseFreeRows && channelVelocityCorrectionMaskPtr) {
                double vfAbs = 0.0, vfRhs = 0.0, vfAx = 0.0, vfScale = 0.0;
                const double vfRel = velocity_free_rows_relative_residual(
                    uPat, *velocityAForSolvePtr, rhs, uz,
                    channelVelocityCorrectionMaskPtr, velocityFreeRowsScaleFloor,
                    &vfAbs, &vfRhs, &vfAx, &vfScale);
                if (velocityFreeRowsPrint && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                    std::printf("    velocityFreeRowFinal: it=%d comp=Z oldReported=%.6e freeRel=%.6e absFree=%.6e rhsFree=%.6e axFree=%.6e scaleFree=%.6e\n",
                        it, iz.finalRelResNorm, vfRel, vfAbs, vfRhs, vfAx, vfScale);
                }
                iz.finalRelResNorm = vfRel;
            }
tmUSolve += wall_seconds() - tPhase;

            std::vector<double> rStar;
            double chInFluxStar = 0.0;
            double chOutFluxStar = 0.0;
            double chNetFluxStar = 0.0;
            double chResidualSumStar = 0.0;
            tPhase = wall_seconds();
            if (gCaseIsChannel && channelDirectContinuity) {
                compute_channel_cg1_continuity_residual_direct(
                    mesh, tets, tq, fq, ux, uy, uz, rStar,
                    &chInFluxStar, &chOutFluxStar, &chNetFluxStar, &chResidualSumStar);
                if (channelDirectContinuitySign != 1.0) {
                    for (double& v : rStar) v *= channelDirectContinuitySign;
                    chResidualSumStar *= channelDirectContinuitySign;
                }
                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    std::printf("    channelDirectContinuity: it=%d inlet=% .6e outlet=% .6e net=% .6e residualSum=% .6e sign=% .3e\n",
                        it, chInFluxStar, chOutFluxStar, chNetFluxStar,
                        chResidualSumStar, channelDirectContinuitySign);
                    print_dg1_cell_constant_residual_audit("predictor_afterUSolve", it, rStar);
                    print_channel_stage_flux_diagnostic("predictor_afterUSolve", it, mesh, tets, fq, ux, uy, uz);
                    print_channel_wall_flux_split_diagnostic("predictor_afterUSolve", it, mesh, tets, fq, ux, uy, uz);
                }
            } else {
                apply_divergence_from_Ap(Apx,Apy,Apz,ux,uy,uz,rStar);
                if (!channelPressureOutletActive) subtract_weighted_mean(rStar, pWeights);
            }
            tmBuStar += wall_seconds() - tPhase;

            const double massStar = norm_vec(rStar);

            std::vector<double> pcorr(nP,0.0);
            int pIts = 0;
            double pFinal = 0.0;
            double velCorrNorm = 0.0;
            bool haveMassSchurProjectionAuditState = false;
            massSchurAuditBoundaryContribution.clear();
            std::vector<double> massSchurAuditRhsP;
            std::vector<double> massSchurAuditPcorr;
            std::vector<double> uxAfterPressureCorrectionBeforeProjection;
            std::vector<double> uyAfterPressureCorrectionBeforeProjection;
            std::vector<double> uzAfterPressureCorrectionBeforeProjection;

            const std::vector<double> uxBeforePressureCorrection = ux;
            const std::vector<double> uyBeforePressureCorrection = uy;
            const std::vector<double> uzBeforePressureCorrection = uz;

            if (pressureMode == "pmass") {
                tPhase = wall_seconds();
                for (int i=0; i<nP; ++i) {
                    pcorr[i] = -nu * invPWeights[i] * rStar[i];
                }

                finalize_pressure_correction(pcorr);
                tmPmass += wall_seconds() - tPhase;
            } else if (pressureMode == "massschur") {
                std::vector<double> rhsP;

                if ((massSchurUseDirectOperator || massSchurUseOpenOutletB) && gCaseIsChannel) {
                    apply_channel_direct_continuity_linear(
                        directBdx, directBdy, directBdz, directBfixedSource,
                        ux, uy, uz,
                        rhsP);

                    if (channelDirectContinuitySign != 1.0) {
                        for (double& v : rhsP) v *= channelDirectContinuitySign;
                    }

                    std::vector<double> bDirectForCompare;
                    double cmpIn = 0.0, cmpOut = 0.0, cmpNet = 0.0, cmpSum = 0.0;
                    compute_channel_cg1_continuity_residual_direct(
                        mesh, tets, tq, fq, ux, uy, uz, bDirectForCompare,
                        &cmpIn, &cmpOut, &cmpNet, &cmpSum);

                    if (channelDirectContinuitySign != 1.0) {
                        for (double& v : bDirectForCompare) v *= channelDirectContinuitySign;
                        cmpSum *= channelDirectContinuitySign;
                    }

                    std::vector<double> diff = rhsP;
                    axpy_vec(diff, -1.0, bDirectForCompare);

                    std::printf("    %s: it=%d ||Bopen*u+g||=%.6e ||BdirectFunc||=%.6e ||diff||=%.6e directFlux(in,out,net)=(% .6e,% .6e,% .6e) sum=% .6e\n",
                        massSchurUseOpenOutletB ? "openBSchurRhs" : "directSchurRhs",
                        it,
                        norm_vec(rhsP),
                        norm_vec(bDirectForCompare),
                        norm_vec(diff),
                        cmpIn, cmpOut, cmpNet, cmpSum);

                    massSchurAuditBoundaryContribution.clear();
                } else if (massSchurUseFixedBcRhs) {
                    // Projection-like constrained Schur with an inhomogeneous fixed-BC term:
                    //
                    //   r_aug = B_free u_free + B_fixed u_fixed
                    //   Lp p' = -r_aug
                    //   u_free <- u_free + H_free B_free^T p'
                    //
                    // Apx/Apy/Apz are the masked/free operators used in Lp and correction.
                    // ApxUnmaskedForPressurePrecond/... are the pre-row-mask operators.
                    // Selecting only Dirichlet correction rows gives B_fixed u_fixed.
                    std::vector<double> bFree;
                    apply_divergence_from_Ap(Apx, Apy, Apz, ux, uy, uz, bFree);

                    std::vector<double> bFixed;
                    const std::vector<unsigned char>* fixedRowsMaskPtr =
                        !channelVelocityDirichletMask.empty()
                            ? &channelVelocityDirichletMask
                            : channelVelocityCorrectionMaskPtr;

                    if (fixedRowsMaskPtr) {
                        apply_divergence_from_Ap_selected_velocity_rows(
                            ApxUnmaskedForPressurePrecond,
                            ApyUnmaskedForPressurePrecond,
                            ApzUnmaskedForPressurePrecond,
                            ux, uy, uz,
                            fixedRowsMaskPtr,
                            bFixed);
                    } else {
                        bFixed.assign(nP, 0.0);
                    }

                    rhsP = bFree;
                    axpy_vec(rhsP, 1.0, bFixed);
                    massSchurAuditBoundaryContribution = bFixed;

                    if (gCaseIsChannel && channelDirectContinuity && massSchurUseDirectBcRhs) {
                        std::printf("WARNING: both -massSchurUseFixedBcRhs 1 and -massSchurUseDirectBcRhs 1 were requested; using fixed-BC Ap split.\n");
                    }

                    if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                        std::vector<double> bAug = bFree;
                        axpy_vec(bAug, 1.0, bFixed);

                        std::vector<double> bDirectForCompare;
                        double cmpIn = 0.0, cmpOut = 0.0, cmpNet = 0.0, cmpSum = 0.0;

                        if (gCaseIsChannel) {
                            compute_channel_cg1_continuity_residual_direct(
                                mesh, tets, tq, fq, ux, uy, uz, bDirectForCompare,
                                &cmpIn, &cmpOut, &cmpNet, &cmpSum);

                            if (channelDirectContinuitySign != 1.0) {
                                for (double& v : bDirectForCompare) v *= channelDirectContinuitySign;
                                cmpSum *= channelDirectContinuitySign;
                            }
                        }

                        std::printf(
                            "    massSchurBcRhs: it=%d mode=fixedApRows "
                            "||Bfree u*||=%.6e ||Bfixed uD||=%.6e ||Baug u*||=%.6e",
                            it,
                            norm_vec(bFree),
                            norm_vec(bFixed),
                            norm_vec(bAug));

                        if (!bDirectForCompare.empty()) {
                            std::vector<double> augMinusDirect = bAug;
                            axpy_vec(augMinusDirect, -1.0, bDirectForCompare);

                            std::printf(
                                " ||Bdirect u*||=%.6e ||Baug-Bdirect||=%.6e "
                                "directFlux(in,out,net)=(% .6e,% .6e,% .6e)",
                                norm_vec(bDirectForCompare),
                                norm_vec(augMinusDirect),
                                cmpIn, cmpOut, cmpNet);
                        }

                        std::printf("\n");
                    }
                } else if (massSchurUseDirectBcRhs && gCaseIsChannel && channelDirectContinuity) {
                    // Diagnostic bridge: same behavior as the old direct-continuity RHS,
                    // but audit the implied source term gBC = Bdirect - Bmodel.
                    std::vector<double> bModel;
                    apply_divergence_from_Ap(Apx, Apy, Apz, ux, uy, uz, bModel);

                    rhsP = rStar;
                    massSchurAuditBoundaryContribution = rStar;
                    axpy_vec(massSchurAuditBoundaryContribution, -1.0, bModel);
                } else if (massSchurUseModelContinuityRhs) {
                    apply_divergence_from_Ap(Apx, Apy, Apz, ux, uy, uz, rhsP);
                } else {
                    rhsP = rStar;
                }

                for (double& v : rhsP) {
                    v = -v;
                }

                finalize_pressure_rhs(rhsP);

                if (!massSchurAuditBoundaryContribution.empty() && channelPressureOutletActive) {
                    zero_values_at_indices(massSchurAuditBoundaryContribution, channelPressurePins);
                }

                HypreOptions poptIt = popt;
                if (hypreProfile) poptIt.profileLabel = "pMassSchur it=" + std::to_string(it);
                tPhase = wall_seconds();
                HypreSolveInfo pinfo;
                if (pressureSplitPrecondActive) {
                    if (pSplitOuterFgmres) {
                        pinfo = solve_pressure_fgmres_split_hypre_prec(
                            lpPat,
                            lpVals,
                            lpSplitPrePat,
                            lpSplitPreVals,
                            pSplitPrecondReusable ? &pressureSplitPrecondReusable : nullptr,
                            pSplitPrecondReusable,
                            pSplitPrecondOpt,
                            rhsP,
                            pcorr,
                            pMaxit,
                            pSplitFgmresRestart,
                            pRelTol,
                            pTol,
                            pMonitor);
                    } else {
                        pinfo = solve_pressure_pcg_split_hypre_prec(
                            lpPat,
                            lpVals,
                            lpSplitPrePat,
                            lpSplitPreVals,
                            pSplitPrecondReusable ? &pressureSplitPrecondReusable : nullptr,
                            pSplitPrecondReusable,
                            pSplitPrecondOpt,
                            rhsP,
                            pcorr,
                            pMaxit,
                            pRelTol,
                            pTol,
                            pMonitor);
                    }
                } else {
                    pinfo = pressureReusableActive
                        ? solve_reusable_hypre_rhs_vec(pressureReusable, rhsP, pcorr, poptIt)
                        : solve_hypre_csr_vec(lpPat, lpVals, rhsP, pcorr, poptIt);
                }
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorr);

                if (pressureVelocityCorrection) {
                    tPhase = wall_seconds();
                    double cx = 0.0, cy = 0.0, cz = 0.0;
                    if (massSchurUseOpenOutletB && gCaseIsChannel) {
                        cx = correct_velocity_mass_schur_open_transpose_direction_masked(invM, directBdx, pcorr, ux, massSchurScale, channelVelocityCorrectionMaskPtr);
                        cy = correct_velocity_mass_schur_open_transpose_direction_masked(invM, directBdy, pcorr, uy, massSchurScale, channelVelocityCorrectionMaskPtr);
                        cz = correct_velocity_mass_schur_open_transpose_direction_masked(invM, directBdz, pcorr, uz, massSchurScale, channelVelocityCorrectionMaskPtr);
                    } else {
                        cx = correct_velocity_mass_schur_direction_masked(invM, Apx, pcorr, ux, massSchurScale, channelVelocityCorrectionMaskPtr);
                        cy = correct_velocity_mass_schur_direction_masked(invM, Apy, pcorr, uy, massSchurScale, channelVelocityCorrectionMaskPtr);
                        cz = correct_velocity_mass_schur_direction_masked(invM, Apz, pcorr, uz, massSchurScale, channelVelocityCorrectionMaskPtr);
                    }
                    velCorrNorm = std::sqrt(cx*cx + cy*cy + cz*cz);
                    tmVelCorr += wall_seconds() - tPhase;
                }

                if (massSchurProjectionAudit) {
                    haveMassSchurProjectionAuditState = true;
                    massSchurAuditRhsP = rhsP;
                    massSchurAuditPcorr = pcorr;
                    uxAfterPressureCorrectionBeforeProjection = ux;
                    uyAfterPressureCorrectionBeforeProjection = uy;
                    uzAfterPressureCorrectionBeforeProjection = uz;
                    print_massschur_projection_audit(
                        it,
                        "afterCorrection_beforeProjection",
                        massSchurAuditRhsP,
                        massSchurAuditPcorr,
                        uxBeforePressureCorrection,
                        uyBeforePressureCorrection,
                        uzBeforePressureCorrection,
                        uxAfterPressureCorrectionBeforeProjection,
                        uyAfterPressureCorrectionBeforeProjection,
                        uzAfterPressureCorrectionBeforeProjection);
                }
            } else if (hybridCellSchurMode0) {
                std::vector<double> pcorrPmass(nP,0.0);
                std::vector<double> pcorrCell(nP,0.0);

                tPhase = wall_seconds();
                for (int i=0; i<nP; ++i) {
                    pcorrPmass[i] = -nu * invPWeights[i] * rStar[i];
                }
                finalize_pressure_correction(pcorrPmass);
                tmPmass += wall_seconds() - tPhase;

                tPhase = wall_seconds();
                std::vector<std::array<std::array<double,10>,10>> invAblocks;
                build_cellblock_jacobi_inverses_from_csr(
                    uPat, Arel, schurBlockShift, schurBlockPivotFloor, invAblocks);
                assemble_lp_cellblock_schur_from_invblocks(
                    nP, Apx, Apy, Apz, invAblocks, lpPat, lpVals);

                if (massSchurScale != 1.0) {
                    for (double& v : lpVals) v *= massSchurScale;
                }
                if (channelPressureOutletActive) {
                    pin_csr_symmetric_zero_set(lpPat, lpVals, channelPressurePins);
                }
                tmSchurBuild += wall_seconds() - tPhase;

                std::vector<double> rhsP = rStar;
                for (double& v : rhsP) v = -v;
                finalize_pressure_rhs(rhsP);

                HypreOptions poptIt = popt;
                if (hypreProfile) poptIt.profileLabel = "pHybridCellSchur it=" + std::to_string(it);

                tPhase = wall_seconds();
                HypreSolveInfo pinfo = solve_hypre_csr_vec(lpPat, lpVals, rhsP, pcorrCell, poptIt);
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorrCell);

                tPhase = wall_seconds();
                for (int i=0; i<nP; ++i) {
                    pcorr[i] = hybridCellPmassWeight * pcorrPmass[i]
                             + hybridCellSchurWeight * pcorrCell[i];
                }
                finalize_pressure_correction(pcorr);
                tmPmass += wall_seconds() - tPhase;

                if (it == 1) {
                    std::printf("Hybrid cell-Schur first norms: ||r||=%.3e, ||p_pmass||=%.3e, ||p_cell||=%.3e, wP=%.6e, wC=%.6e, ||pcorr||=%.3e\n",
                        massStar,
                        weighted_l2_norm(pcorrPmass, pWeights),
                        weighted_l2_norm(pcorrCell, pWeights),
                        hybridCellPmassWeight,
                        hybridCellSchurWeight,
                        weighted_l2_norm(pcorr, pWeights));
                }

                if (pressureVelocityCorrection && std::abs(hybridCellSchurWeight) > 0.0) {
                    tPhase = wall_seconds();
                    const double scale = schurVelocityCorrectionScale * hybridCellSchurWeight;

                    const double cx = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apx, pcorrCell, ux, scale);
                    const double cy = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apy, pcorrCell, uy, scale);
                    const double cz = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apz, pcorrCell, uz, scale);

                    velCorrNorm = std::sqrt(cx*cx + cy*cy + cz*cz);
                    tmVelCorr += wall_seconds() - tPhase;
                }
            } else if (cellBlockSchurMode0) {
                tPhase = wall_seconds();
                std::vector<std::array<std::array<double,10>,10>> invAblocks;
                build_cellblock_jacobi_inverses_from_csr(uPat, Arel, schurBlockShift, schurBlockPivotFloor, invAblocks);
                assemble_lp_cellblock_schur_from_invblocks(nP, Apx, Apy, Apz, invAblocks, lpPat, lpVals);

                if (massSchurScale != 1.0) {
                    for (double& v : lpVals) v *= massSchurScale;
                }
                if (channelPressureOutletActive) {
                    pin_csr_symmetric_zero_set(lpPat, lpVals, channelPressurePins);
                }

                tmSchurBuild += wall_seconds() - tPhase;

                std::vector<double> rhsP = rStar;
                for (double& v : rhsP) v = -v;
                finalize_pressure_rhs(rhsP);

                HypreOptions poptIt = popt;
                if (hypreProfile) poptIt.profileLabel = "pCellBlockSchur it=" + std::to_string(it);

                tPhase = wall_seconds();
                HypreSolveInfo pinfo = solve_hypre_csr_vec(lpPat, lpVals, rhsP, pcorr, poptIt);
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorr);

                if (pressureVelocityCorrection) {
                    tPhase = wall_seconds();

                    const double cx = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apx, pcorr, ux, schurVelocityCorrectionScale);
                    const double cy = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apy, pcorr, uy, schurVelocityCorrectionScale);
                    const double cz = correct_velocity_cellblock_schur_direction(
                        invAblocks, Apz, pcorr, uz, schurVelocityCorrectionScale);

                    velCorrNorm = std::sqrt(cx*cx + cy*cy + cz*cz);
                    tmVelCorr += wall_seconds() - tPhase;
                }
            } else if (scalarArelSchurMode0) {
                tPhase = wall_seconds();
                std::vector<double> hInv = make_scalar_schur_inverse_from_Arel(uPat, Arel, pressureMode);
                assemble_lp_scalar_schur(nP, Apx, Apy, Apz, hInv, lpPat, lpVals);
                if (massSchurScale != 1.0) {
                    for (double& v : lpVals) v *= massSchurScale;
                }
                if (channelPressureOutletActive) {
                    pin_csr_symmetric_zero_set(lpPat, lpVals, channelPressurePins);
                }
                tmSchurBuild += wall_seconds() - tPhase;

                std::vector<double> rhsP = rStar;
                for (double& v : rhsP) v = -v;
                finalize_pressure_rhs(rhsP);

                HypreOptions poptIt = popt;
                if (hypreProfile) poptIt.profileLabel = "pScalarArelSchur it=" + std::to_string(it);

                tPhase = wall_seconds();
                HypreSolveInfo pinfo = solve_hypre_csr_vec(lpPat, lpVals, rhsP, pcorr, poptIt);
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorr);

                if (pressureVelocityCorrection) {
                    tPhase = wall_seconds();
                    const double sx = correct_velocity_diag_schur_direction(hInv, Apx, pcorr, ux, massSchurScale);
                    const double sy = correct_velocity_diag_schur_direction(hInv, Apy, pcorr, uy, massSchurScale);
                    const double sz = correct_velocity_diag_schur_direction(hInv, Apz, pcorr, uz, massSchurScale);
                    velCorrNorm = std::sqrt(sx*sx + sy*sy + sz*sz);
                    tmVelCorr += wall_seconds() - tPhase;
                }
            } else if (inexactSchurMode0) {
                std::vector<double> rhsP = rStar;

                for (double& v : rhsP) {
                    v = -v;
                }

                finalize_pressure_rhs(rhsP);

                HypreOptions schurInnerOpt = uopt;
                schurInnerOpt.solver = schurInnerSolver;
                schurInnerOpt.precond = schurInnerPrecond;
                schurInnerOpt.maxIter = schurInnerMaxit;
                schurInnerOpt.relTol = schurInnerRelTol;
                schurInnerOpt.absTol = schurInnerAbsTol;
                schurInnerOpt.profileLabel = "inexactSchur-inner it=" + std::to_string(it);

                InexactSchurStats schurStats;

                std::fill(pcorr.begin(), pcorr.end(), 0.0);

                tPhase = wall_seconds();
                HypreSolveInfo pinfo = solve_pressure_inexact_schur_gmres(
                    uPat,
                    Arel,
                    Apx,
                    Apy,
                    Apz,
                    rhsP,
                    pcorr,
                    pWeights,
                    schurInnerOpt,
                    schurInnerSolver,
                    uLeftScaleMode,
                    uLeftScaleEps,
                    uColoring,
                    &schurGpuMcgs,
                    uSweeps,
                    uMcgsPreSweeps,
                    uOmega,
                    uSymmetric,
                    uCellBlockIters,
                    uCellBlockOmega,
                    schurPressureMaxit,
                    schurPressureKdim,
                    schurPressureRelTol,
                    schurPressureAbsTol,
                    schurInnerMonitor,
                    &schurStats);
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorr);

                if (it == 1) {
                    std::printf("Inexact Schur first norms: ||r||=%.3e ||pcorr||=%.3e pIts=%d pFinal=%.3e matvecs=%lld innerSolves=%lld innerIts=%lld worstInner=%.3e\n",
                        massStar,
                        weighted_l2_norm(pcorr, pWeights),
                        pIts,
                        pFinal,
                        schurStats.pressureMatvecs,
                        schurStats.innerSolves,
                        schurStats.innerIters,
                        schurStats.innerWorstRel);
                }

                if (pressureVelocityCorrection) {
                    tPhase = wall_seconds();

                    const double cx = apply_inexact_schur_velocity_correction(
                        uPat, Arel, Apx, pcorr, ux, massSchurScale,
                        schurInnerOpt, schurInnerSolver, uLeftScaleMode, uLeftScaleEps,
                        uColoring, &schurGpuMcgs,
                        uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
                        uCellBlockIters, uCellBlockOmega,
                        schurInnerMonitor, &schurStats);

                    const double cy = apply_inexact_schur_velocity_correction(
                        uPat, Arel, Apy, pcorr, uy, massSchurScale,
                        schurInnerOpt, schurInnerSolver, uLeftScaleMode, uLeftScaleEps,
                        uColoring, &schurGpuMcgs,
                        uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
                        uCellBlockIters, uCellBlockOmega,
                        schurInnerMonitor, &schurStats);

                    const double cz = apply_inexact_schur_velocity_correction(
                        uPat, Arel, Apz, pcorr, uz, massSchurScale,
                        schurInnerOpt, schurInnerSolver, uLeftScaleMode, uLeftScaleEps,
                        uColoring, &schurGpuMcgs,
                        uSweeps, uMcgsPreSweeps, uOmega, uSymmetric,
                        uCellBlockIters, uCellBlockOmega,
                        schurInnerMonitor, &schurStats);

                    velCorrNorm = std::sqrt(cx*cx + cy*cy + cz*cz);
                    tmVelCorr += wall_seconds() - tPhase;
                }

                if (printEvery == 1 || it == 1) {
                    std::printf("    inexactSchur stats: pIts=%d pFinal=%.3e matvecs=%lld innerSolves=%lld innerIts=%lld worstInner=%.3e applyTime=%.6f gmresTime=%.6f\n",
                        pIts,
                        pFinal,
                        schurStats.pressureMatvecs,
                        schurStats.innerSolves,
                        schurStats.innerIters,
                        schurStats.innerWorstRel,
                        schurStats.applyTime,
                        schurStats.gmresTime);
                }
            } else if (pressureMode == "hybridmass") {
                std::vector<double> pcorrPmass(nP,0.0);
                std::vector<double> pcorrSimple(nP,0.0);

                tPhase = wall_seconds();
                for (int i=0; i<nP; ++i) {
                    pcorrPmass[i] = -nu * invPWeights[i] * rStar[i];
                }

                finalize_pressure_correction(pcorrPmass);
                tmPmass += wall_seconds() - tPhase;

                std::vector<double> rhsP = rStar;

                for (double& v : rhsP) {
                    v = -v;
                }

                finalize_pressure_rhs(rhsP);

                HypreOptions poptIt = popt;
                if (hypreProfile) poptIt.profileLabel = "pHybridSimple it=" + std::to_string(it);
                tPhase = wall_seconds();
                HypreSolveInfo pinfo = pressureReusableActive
                    ? solve_reusable_hypre_rhs_vec(pressureReusable, rhsP, pcorrSimple, poptIt)
                    : solve_hypre_csr_vec(lpPat, lpVals, rhsP, pcorrSimple, poptIt);
                tmPSolve += wall_seconds() - tPhase;

                pIts = pinfo.iterations;
                pFinal = pinfo.finalRelResNorm;

                finalize_pressure_correction(pcorrSimple);

                const double nuAbs = std::abs(nu);
                const double nu0 = std::max(std::abs(hybridNuCrossover), 1e-300);
                const double q = std::max(std::abs(hybridNuPower), 1e-300);
                const double a = std::pow(nuAbs, q);
                const double b = std::pow(nu0, q);
                const double chi = a / std::max(a + b, 1e-300);
                const double wP = hybridPmassCoeff * chi;
                const double wS = hybridSimpleCoeff * (1.0 - chi);

                tPhase = wall_seconds();
                for (int i=0; i<nP; ++i) {
                    pcorr[i] = wP*pcorrPmass[i] + wS*pcorrSimple[i];
                }

                finalize_pressure_correction(pcorr);
                tmPmass += wall_seconds() - tPhase;

                if (it == 1) {
                    std::printf("Hybrid mass first norms: ||r||=%.3e, ||p_pmass||=%.3e, ||p_simple||=%.3e, chi=%.6e, wP=%.6e, wS=%.6e, ||pcorr||=%.3e\n",
                        massStar,
                        weighted_l2_norm(pcorrPmass, pWeights),
                        weighted_l2_norm(pcorrSimple, pWeights),
                        chi,
                        wP,
                        wS,
                        weighted_l2_norm(pcorr, pWeights));
                }

                if (hybridVelocityCorrection) {
                    tPhase = wall_seconds();
                    const double scale = hybridVelocityCorrectionScale * wS;
                    const double cx = correct_velocity_mass_schur_direction_masked(invM, Apx, pcorrSimple, ux, scale, channelVelocityCorrectionMaskPtr);
                    const double cy = correct_velocity_mass_schur_direction_masked(invM, Apy, pcorrSimple, uy, scale, channelVelocityCorrectionMaskPtr);
                    const double cz = correct_velocity_mass_schur_direction_masked(invM, Apz, pcorrSimple, uz, scale, channelVelocityCorrectionMaskPtr);
                    velCorrNorm = std::sqrt(cx*cx + cy*cy + cz*cz);
                    tmVelCorr += wall_seconds() - tPhase;
                }
            }

            if (gCaseIsChannel && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                print_channel_pressure_patch_node_stats(it, "afterPSolve_beforeClamp", mesh, pOld, pcorr);
                print_channel_stage_flux_diagnostic("afterPressureCorrection_beforeClamp", it, mesh, tets, fq, ux, uy, uz);
                print_channel_wall_flux_split_diagnostic("afterPressureCorrection_beforeClamp", it, mesh, tets, fq, ux, uy, uz);

                if (channelVelocityCorrectionMaskPtr) {
                    double lockedDu2 = 0.0, freeDu2 = 0.0;
                    double lockedMax = 0.0, freeMax = 0.0;
                    int lockedCnt = 0, freeCnt = 0;

                    const int N = std::min<int>((int)ux.size(), (int)channelVelocityCorrectionMaskPtr->size());
                    for (int ii=0; ii<N; ++ii) {
                        const double dx = ux[ii] - uxBeforePressureCorrection[ii];
                        const double dy = uy[ii] - uyBeforePressureCorrection[ii];
                        const double dz = uz[ii] - uzBeforePressureCorrection[ii];
                        const double d = std::sqrt(dx*dx + dy*dy + dz*dz);

                        if ((*channelVelocityCorrectionMaskPtr)[ii]) {
                            lockedDu2 += d*d;
                            lockedMax = std::max(lockedMax, d);
                            ++lockedCnt;
                        } else {
                            freeDu2 += d*d;
                            freeMax = std::max(freeMax, d);
                            ++freeCnt;
                        }
                    }

                    std::printf(
                        "    channelVelCorrMaskAudit: it=%d lockedRows=%d freeRows=%d "
                        "lockedDuL2=%.6e lockedDuMax=%.6e freeDuL2=%.6e freeDuMax=%.6e\n",
                        it, lockedCnt, freeCnt,
                        std::sqrt(std::max(0.0, lockedDu2)), lockedMax,
                        std::sqrt(std::max(0.0, freeDu2)), freeMax);
                }

                std::vector<double> rDirectAudit;
                double auditIn = 0.0, auditOut = 0.0, auditNet = 0.0, auditSum = 0.0;
                compute_channel_cg1_continuity_residual_direct(
                    mesh, tets, tq, fq, ux, uy, uz, rDirectAudit,
                    &auditIn, &auditOut, &auditNet, &auditSum);
                if (channelDirectContinuitySign != 1.0) {
                    for (double& v : rDirectAudit) v *= channelDirectContinuitySign;
                    auditSum *= channelDirectContinuitySign;
                }

                std::vector<double> rApAudit;
                apply_divergence_from_Ap(Apx, Apy, Apz, ux, uy, uz, rApAudit);
                std::vector<double> rDiffAudit = rApAudit;
                axpy_vec(rDiffAudit, -1.0, rDirectAudit);

                std::printf("    channelProjectionAudit: it=%d stage=afterPressureCorrection_beforeClamp "
                            "||Bdirect u||=%.6e sumDirect=% .6e flux(in,out,net)=(% .6e,% .6e,% .6e) "
                            "||Bap u||=%.6e ||Bap-Bdirect||=%.6e\n",
                    it,
                    norm_vec(rDirectAudit), auditSum, auditIn, auditOut, auditNet,
                    norm_vec(rApAudit), norm_vec(rDiffAudit));
            }

            if (gCaseIsChannel && channelOutletCorrDiag &&
                (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                print_channel_outlet_pressure_correction_diagnostic(
                    it, mesh, tets, fq, pcorr,
                    uxBeforePressureCorrection,
                    uyBeforePressureCorrection,
                    uzBeforePressureCorrection,
                    ux, uy, uz);
            }

            if (gCaseIsChannel && channelProjectDirichletNormalCorr) {
                tPhase = wall_seconds();
                double sumAbsDn = 0.0;
                double maxAbsDn = 0.0;

                const int nProjectedNormalCorr =
                    project_channel_dirichlet_normal_correction_trace(
                        mesh, tets,
                        uxBeforePressureCorrection,
                        uyBeforePressureCorrection,
                        uzBeforePressureCorrection,
                        ux, uy, uz,
                        &sumAbsDn, &maxAbsDn);

                tmVelCorr += wall_seconds() - tPhase;

                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    std::printf(
                        "    channelProjectDirichletNormalCorr: it=%d projectedFaceDofs=%d sumAbsDn=%.6e maxAbsDn=%.6e; inlet/wall normal correction removed, outlet untouched.\n",
                        it, nProjectedNormalCorr, sumAbsDn, maxAbsDn);

                    print_channel_stage_flux_diagnostic(
                        "afterDirichletNormalProjection_beforeClamp",
                        it, mesh, tets, fq, ux, uy, uz);
                    print_channel_wall_flux_split_diagnostic(
                        "afterDirichletNormalProjection_beforeClamp",
                        it, mesh, tets, fq, ux, uy, uz);
                }
            }

            if (gCaseIsChannel && channelClampVelocityBC) {
                tPhase = wall_seconds();
                const int nClamp = enforce_channel_velocity_dirichlet_trace(mesh, tets, ux, uy, uz);
                tmVelCorr += wall_seconds() - tPhase;
                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    std::printf("    channelClampVelocityBC: it=%d clampedDG2Dofs=%d inlet/walls reapplied after pressure correction; outlet untouched.\n",
                        it, nClamp);
                    print_channel_stage_flux_diagnostic("afterClamp", it, mesh, tets, fq, ux, uy, uz);
                print_channel_wall_flux_split_diagnostic("afterClamp", it, mesh, tets, fq, ux, uy, uz);
                }
            }

            if (gCaseIsChannel && gChannelAdjustOutletFlux) {
                tPhase = wall_seconds();
                double netBeforeAdjust = 0.0;
                double deltaUnAdjust = 0.0;
                double areaOutletAdjust = 0.0;
                const int nAdjustedOutlet = adjust_channel_outlet_flux_trace(
                    mesh, tets, fq, ux, uy, uz,
                    &netBeforeAdjust, &deltaUnAdjust, &areaOutletAdjust);
                tmVelCorr += wall_seconds() - tPhase;

                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    std::printf("    channelAdjustOutletFlux: it=%d adjustedDG2Dofs=%d netBefore=% .6e areaOut=%.6e deltaUn=% .6e expectedNetAfter≈0\n",
                        it, nAdjustedOutlet, netBeforeAdjust, areaOutletAdjust, deltaUnAdjust);
                }
            }

            if (haveMassSchurProjectionAuditState) {
                print_massschur_projection_audit(
                    it,
                    "afterProjectionClampAdjust_finalVelocity",
                    massSchurAuditRhsP,
                    massSchurAuditPcorr,
                    uxBeforePressureCorrection,
                    uyBeforePressureCorrection,
                    uzBeforePressureCorrection,
                    ux,
                    uy,
                    uz);
            }

            tPhase = wall_seconds();
            for (int i=0; i<nP; ++i) {
                p[i] = pOld[i] + alphaP*pcorr[i];
            }

            finalize_pressure_field(p);
            tmPUpdate += wall_seconds() - tPhase;

            if (gCaseIsChannel && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                print_channel_pressure_patch_node_stats(it, "afterPUpdate", mesh, p, pcorr);
            }

            int convNnzIter = 0;
            tPhase = wall_seconds();
            if (convGpuActive) {
                rebuild_physical_operator_from_beta_gpu(convGpuState, Kdiff, ux, uy, uz, Aphys, &convNnzIter);
            } else {
                rebuild_physical_operator_from_beta(mesh, tets, uPat, tq, fq, Kdiff, ux, uy, uz, Aphys, &convNnzIter, &convCache);
            }
            tmConvRebuild += wall_seconds() - tPhase;

            tPhase = wall_seconds();
            Arel = Aphys;

            if (gammaU != 0.0) {
                add_scaled_values(Arel, M, gammaU);
            }
            if (is_gpu_cellblock_solver_name(uSolver) && uGpuMcgs.ready) {
                uGpuMcgs.cellMatrixValid = false;
            }
            if (inexactSchurMode0 && is_gpu_cellblock_solver_name(schurInnerSolver) && schurGpuMcgs.ready) {
                schurGpuMcgs.cellMatrixValid = false;
            }
            tmArelUpdate += wall_seconds() - tPhase;

            std::vector<double> rNew;
            tPhase = wall_seconds();
            if (gCaseIsChannel && channelDirectContinuity) {
                double chInFluxNew = 0.0;
                double chOutFluxNew = 0.0;
                double chNetFluxNew = 0.0;
                double chResidualSumNew = 0.0;
                compute_channel_cg1_continuity_residual_direct(
                    mesh, tets, tq, fq, ux, uy, uz, rNew,
                    &chInFluxNew, &chOutFluxNew, &chNetFluxNew, &chResidualSumNew);
                if (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple) {
                    print_dg1_cell_constant_residual_audit("afterPressureCorrection_finalVelocity", it, rNew);
                }
                if (channelDirectContinuitySign != 1.0) {
                    for (double& v : rNew) v *= channelDirectContinuitySign;
                }
            } else {
                apply_divergence_from_Ap(Apx,Apy,Apz,ux,uy,uz,rNew);
                if (!channelPressureOutletActive) subtract_weighted_mean(rNew, pWeights);
            }
            tmBuNew += wall_seconds() - tPhase;

            const double massNew = norm_vec(rNew);

            if (gCaseIsChannel && (it == 1 || it % std::max(1, printEvery) == 0 || it == nSimple)) {
                print_channel_boundary_flux_diagnostic(it, mesh, tets, fq, ux, uy, uz);
                print_channel_patch_component_flux_diagnostic(it, mesh, tets, fq, ux, uy, uz);
            }

            tPhase = wall_seconds();
            std::vector<double> Ax, Ay, Az;
            std::vector<double> gpX, gpY, gpZ;

            apply_csr(uPat, Aphys, ux, Ax);
            apply_gradient_rhs(Apx, p, gpX);
            axpy_vec(Ax, 1.0, gpX);
            axpy_vec(Ax, -1.0, F[0]);

            apply_csr(uPat, Aphys, uy, Ay);
            apply_gradient_rhs(Apy, p, gpY);
            axpy_vec(Ay, 1.0, gpY);
            axpy_vec(Ay, -1.0, F[1]);

            apply_csr(uPat, Aphys, uz, Az);
            apply_gradient_rhs(Apz, p, gpZ);
            axpy_vec(Az, 1.0, gpZ);
            axpy_vec(Az, -1.0, F[2]);

            const double momRes = std::sqrt(
                norm_vec(Ax)*norm_vec(Ax) +
                norm_vec(Ay)*norm_vec(Ay) +
                norm_vec(Az)*norm_vec(Az)
            );

            const double momResFree = std::sqrt(
                norm_vec_free_rows(Ax, channelVelocityCorrectionMaskPtr) *
                norm_vec_free_rows(Ax, channelVelocityCorrectionMaskPtr) +
                norm_vec_free_rows(Ay, channelVelocityCorrectionMaskPtr) *
                norm_vec_free_rows(Ay, channelVelocityCorrectionMaskPtr) +
                norm_vec_free_rows(Az, channelVelocityCorrectionMaskPtr) *
                norm_vec_free_rows(Az, channelVelocityCorrectionMaskPtr)
            );

            const double momResForConvergence =
                (simpleConvergenceUseFreeRows && channelVelocityCorrectionMaskPtr) ? momResFree : momRes;
            const double coupled = std::sqrt(momResForConvergence*momResForConvergence + massNew*massNew);

            if (it == 1) {
                coupled0 = std::max(coupled, 1.0);
            }

            const double relCoupled = coupled / coupled0;

            const double pUpdRel =
                weighted_l2_norm(pcorr, pWeights) * std::abs(alphaP) /
                std::max(weighted_l2_norm(pOld, pWeights), 1.0);

            std::vector<double> dux = ux;
            std::vector<double> duy = uy;
            std::vector<double> duz = uz;

            axpy_vec(dux,-1.0,uxOld);
            axpy_vec(duy,-1.0,uyOld);
            axpy_vec(duz,-1.0,uzOld);

            const double uUpd =
                std::sqrt(norm_vec(dux)*norm_vec(dux) +
                          norm_vec(duy)*norm_vec(duy) +
                          norm_vec(duz)*norm_vec(duz));

            const double uOldN =
                std::max(std::sqrt(norm_vec(uxOld)*norm_vec(uxOld) +
                                   norm_vec(uyOld)*norm_vec(uyOld) +
                                   norm_vec(uzOld)*norm_vec(uzOld)), 1.0);

            const double uUpdRel = uUpd/uOldN;

            tmMomDiag += wall_seconds() - tPhase;

            const bool simpleCoupledConverged = (relCoupled < tolCoupled);
            const bool simpleUpdateConverged =
                (!simpleConvergenceRequireUpdateRel ||
                 (pUpdRel < tolUpdate && uUpdRel < tolUpdate));
            const bool convergedNow = simpleCoupledConverged && simpleUpdateConverged;

            const bool printNow =
                (it==1 || it % printEvery == 0 || convergedNow);

            const bool doErrDiag =
                (errEvery > 0) && (it==1 || it % errEvery == 0 || convergedNow);

            double uErrRel = std::numeric_limits<double>::quiet_NaN();
            double pErrRel = std::numeric_limits<double>::quiet_NaN();

            if (doErrDiag) {
                double uN=0.0;
                double pN=0.0;

                tPhase = wall_seconds();
                const double uErr = dg2_vector_l2_error(mesh,tets,tq,ux,uy,uz,&uN);
                const double pErr = cg1_pressure_l2_error(mesh,tets,tq,p,pWeights,&pN);
                tmErrDiag += wall_seconds() - tPhase;

                uErrRel = uErr / std::max(uN, 1e-300);
                pErrRel = pErr / std::max(pN, 1e-300);
            }

            if (printNow) {
                std::puts("    columns: it uItsX uItsY uItsZ  uFinX uFinY uFinZ  ||B u*|| ||B u||  momRes momResFree relCoupled  ||pCorr|| pUpdRel uUpdRel  UrelL2 PrelL2");
                std::printf("%3d %6d %6d %6d  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e  %.3e\n",
                    it,
                    ix.iterations, iy.iterations, iz.iterations,
                    ix.finalRelResNorm, iy.finalRelResNorm, iz.finalRelResNorm,
                    massStar,
                    massNew,
                    momRes,
                    momResFree,
                    relCoupled,
                    weighted_l2_norm(pcorr,pWeights),
                    pUpdRel,
                    uUpdRel,
                    uErrRel,
                    pErrRel);

                if (inexactSchurMode0) {
                    std::printf("    pSolve(inexactschur): pIts=%d pFinal=%.3e velCorrNorm=%.3e pressureVelocityCorrection=%d inner=%s innerMax=%d innerRel=%.3e\n",
                        pIts, pFinal, velCorrNorm, pressureVelocityCorrection,
                        schurInnerSolver.c_str(), schurInnerMaxit, schurInnerRelTol);
                } else if (pressureMode == "massschur") {
                    std::printf("    pSolve: pIts=%d pFinal=%.3e velCorrNorm=%.3e pressureVelocityCorrection=%d\n",
                        pIts, pFinal, velCorrNorm, pressureVelocityCorrection);
                } else if (hybridCellSchurMode0) {
                    std::printf("    pSolve(%s): pIts=%d pFinal=%.3e velCorrNorm=%.3e pressureVelocityCorrection=%d schurBuild=%.6f wP=%.3e wC=%.3e\n",
                        pressureMode.c_str(), pIts, pFinal, velCorrNorm, pressureVelocityCorrection,
                        tmSchurBuild, hybridCellPmassWeight, hybridCellSchurWeight);
                } else if (cellBlockSchurMode0) {
                    std::printf("    pSolve(%s): pIts=%d pFinal=%.3e velCorrNorm=%.3e pressureVelocityCorrection=%d schurBuild=%.6f velCorrScale=%.3e\n",
                        pressureMode.c_str(), pIts, pFinal, velCorrNorm, pressureVelocityCorrection, tmSchurBuild, schurVelocityCorrectionScale);
                } else if (scalarArelSchurMode0) {
                    std::printf("    pSolve(%s): pIts=%d pFinal=%.3e velCorrNorm=%.3e pressureVelocityCorrection=%d schurBuild=%.6f\n",
                        pressureMode.c_str(), pIts, pFinal, velCorrNorm, pressureVelocityCorrection, tmSchurBuild);
                } else if (pressureMode == "hybridmass") {
                    std::printf("    pSolve(simple branch): pIts=%d pFinal=%.3e velCorrNorm=%.3e hybridVelocityCorrection=%d nnz(Cbeta)=%d\n",
                        pIts, pFinal, velCorrNorm, hybridVelocityCorrection, convNnzIter);
                }

                if (forceEnable && forceEvery > 0 && (it == 1 || it % forceEvery == 0 || convergedNow)) {
                    const DgPatchForceReport fr = compute_dg2_dg1_patch_forces_stress(
                        mesh, tets, fq, forcePatch, ux, uy, uz, p,
                        forceRho, forceMu, forceNormalSign, forceUref, forceAreaRef,
                        forceDragDir, forceLiftDir, forceSpanDir);
                    print_dg_patch_force_report(fr, it, "iteration");
                    if (forceReactionEnable) {
                        const DgPatchReactionForceReport rr = compute_dg2_patch_reaction_forces_from_momentum_residual(
                            mesh, tets, uPat, Aphys, Apx, Apy, Apz, F, forcePatch,
                            ux, uy, uz, p, forceRho, forceUref, forceAreaRef,
                            forceDragDir, forceLiftDir, forceSpanDir,
                            &convBCx, &convBCy, &convBCz, &fr.Fp);
                        print_dg_patch_reaction_force_report(rr, it, "iteration");
                    }
                }

                if (profileTimings) {
                    const double tmIter = wall_seconds() - tIter0;
                    std::printf("    timing: total=%.6f copyOld=%.6f rhsBuild=%.6f uSolve=%.6f BuStar=%.6f pmass=%.6f schurBuild=%.6f pSolve=%.6f velCorr=%.6f pUpdate=%.6f convRebuild=%.6f ArelUpdate=%.6f BuNew=%.6f momDiag=%.6f errDiag=%.6f\n",
                        tmIter, tmCopyOld, tmRhsBuild, tmUSolve, tmBuStar, tmPmass, tmSchurBuild, tmPSolve,
                        tmVelCorr, tmPUpdate, tmConvRebuild, tmArelUpdate, tmBuNew, tmMomDiag, tmErrDiag);
                    if (uGpuMcgs.ready && uGpuMcgs.cellBlockMode) {
                        std::printf("    gpuCellBlock: solves=%lld krylovIts=%lld lastIts=%d lastRel=%.3e lastTotal=%.6f lastPrecond=%.6f lastMatvec=%.6f lastInvBuild=%.6f badBlocks=%d cumTotal=%.6f cumPrecond=%.6f cumMatvec=%.6f cumUploadDownload=%.6f cumInvBuild=%.6f invBuilds=%lld applies=%lld corrIters=%d\n",
                            uGpuMcgs.cumulativeSolves,
                            uGpuMcgs.cumulativeKrylovIts,
                            uGpuMcgs.lastIterations,
                            uGpuMcgs.lastRelRes,
                            uGpuMcgs.lastKrylovTime,
                            uGpuMcgs.lastPrecondTime,
                            uGpuMcgs.lastMatvecTime,
                            uGpuMcgs.lastCellBlockBuildTime,
                            uGpuMcgs.lastCellBlockBad,
                            uGpuMcgs.cumulativeKrylovTime,
                            uGpuMcgs.cumulativePrecondTime,
                            uGpuMcgs.cumulativeMatvecTime,
                            uGpuMcgs.cumulativeUploadDownloadTime,
                            uGpuMcgs.cumulativeCellBlockBuildTime,
                            uGpuMcgs.cumulativeCellBlockBuilds,
                            uGpuMcgs.cumulativeCellBlockApplies,
                            uCellBlockIters);
                    } else if (uGpuMcgs.ready) {
                        std::printf("    gpuMcgs: solves=%lld krylovIts=%lld lastIts=%d lastRel=%.3e lastTotal=%.6f lastPrecond=%.6f lastMatvec=%.6f cumTotal=%.6f cumPrecond=%.6f cumMatvec=%.6f cumUploadDownload=%.6f sweeps=%lld\n",
                            uGpuMcgs.cumulativeSolves,
                            uGpuMcgs.cumulativeKrylovIts,
                            uGpuMcgs.lastIterations,
                            uGpuMcgs.lastRelRes,
                            uGpuMcgs.lastKrylovTime,
                            uGpuMcgs.lastPrecondTime,
                            uGpuMcgs.lastMatvecTime,
                            uGpuMcgs.cumulativeKrylovTime,
                            uGpuMcgs.cumulativePrecondTime,
                            uGpuMcgs.cumulativeMatvecTime,
                            uGpuMcgs.cumulativeUploadDownloadTime,
                            uGpuMcgs.cumulativePrecondSweeps);
                    }
                }
            }

            if (memEvery > 0 && (it == 1 || it % memEvery == 0 || convergedNow)) {
                size_t freeB = 0;
                size_t totalB = 0;
                cudaError_t memErr = cudaMemGetInfo(&freeB, &totalB);
                if (memErr == cudaSuccess) {
                    const double freeMiB = (double)freeB / (1024.0*1024.0);
                    const double totalMiB = (double)totalB / (1024.0*1024.0);
                    const double usedMiB = totalMiB - freeMiB;
                    std::printf("    gpuMem: used=%.1f MiB free=%.1f MiB total=%.1f MiB\n",
                        usedMiB, freeMiB, totalMiB);
                } else {
                    std::printf("    gpuMem: cudaMemGetInfo failed: %s\n", cudaGetErrorString(memErr));
                }
            }

            if (convergedNow) {
                break;
            }
        }

        if (forceEnable) {
            const DgPatchForceReport fr = compute_dg2_dg1_patch_forces_stress(
                mesh, tets, fq, forcePatch, ux, uy, uz, p,
                forceRho, forceMu, forceNormalSign, forceUref, forceAreaRef,
                forceDragDir, forceLiftDir, forceSpanDir);
            print_dg_patch_force_report(fr, -1, "final");
            if (forceReactionEnable) {
                const DgPatchReactionForceReport rr = compute_dg2_patch_reaction_forces_from_momentum_residual(
                    mesh, tets, uPat, Aphys, Apx, Apy, Apz, F, forcePatch,
                    ux, uy, uz, p, forceRho, forceUref, forceAreaRef,
                    forceDragDir, forceLiftDir, forceSpanDir,
                    nullptr, nullptr, nullptr, &fr.Fp);
                print_dg_patch_reaction_force_report(rr, -1, "final");
            }
        }

        if (writeVtu) {
            const double tVtu0 = wall_seconds();
            write_dg2_dg1_quadratic_tet_vtu(vtuFile, mesh, tets, ux, uy, uz, p, pWeights);
            std::printf("VTU written                    = %s points=%d cells=%d writeTime=%.6f s\n",
                vtuFile.c_str(), 10*mesh.nCells, mesh.nCells, wall_seconds() - tVtu0);
        }

        if (convGpuState.ready) {
            destroy_gpu_conv_lf_state(convGpuState);
        }

        if (uGpuMcgs.ready) {
            destroy_gpu_mcgs_system(uGpuMcgs);
        }

        if (velocityReusable.initialized) {
            destroy_reusable_hypre_system_gpu(velocityReusable);
        }

        if (pressureReusable.initialized) {
            destroy_reusable_hypre_system_gpu(pressureReusable);
        }

        finalize_hypre_gpu_runtime();
        MPI_Finalize();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
