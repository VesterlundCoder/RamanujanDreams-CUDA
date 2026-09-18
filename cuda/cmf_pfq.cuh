// cmf_pfq.cuh
//
// Core pFq Conservative Matrix Field definitions for CUDA.
// Shared by all three arithmetic variants (f64, rns, df64).
//
// The pFq CMF axis matrix is I + C/a, where C is the theta companion matrix
// derived from the hypergeometric differential equation:
//   D(theta) = theta * prod_i(theta + y_i - 1) - z * prod_j(theta + x_j)
//
// Right-multiplying by I + C/a is O(dim^2) using companion sparsity.
//
// Reference: RamanujanMachine/ramanujantools CMF.pFq
//
// IMPORTANT: Projective normalization (dividing by max entry) is used to
// prevent overflow. The log of the scale factor is accumulated in log_scale
// so that the actual magnitude of p and q can be recovered:
//   actual_p = normalized_p * exp(log_scale)
//   actual_q = normalized_q * exp(log_scale)
// This is essential for correct delta (irrationality measure) computation,
// which requires the actual q value: delta = -(1 + log|L - p/q| / log(q))

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

// Write a double as valid JSON (NaN/Inf -> null)
__host__ __device__
inline void json_double(char *buf, int size, double x) {
    if (!isfinite(x)) snprintf(buf, size, "null");
    else snprintf(buf, size, "%.17g", x);
}

// Maximum supported rank (8F7 needs rank 8)
#define MAX_DIM 8
// Maximum supported axes (8F7 has p=8, q=7, so 15 axes)
#define MAX_AXES 16

// CMF type descriptor
struct CMFDesc {
    int p;          // number of numerator parameters
    int q;          // number of denominator parameters
    int rank;       // matrix dimension (max(p, q+1), minus 1 if p==q+1 and z==1)
    int naxes;      // p + q
    double z;       // z value (typically 1/2 or 1)
};

// Get CMF descriptor for a given pFq type
__host__ __device__
inline CMFDesc make_cmf_desc(int p, int q, double z) {
    CMFDesc d;
    d.p = p;
    d.q = q;
    d.z = z;
    d.naxes = p + q;
    // predict_rank: N = max(p, q+1); if z==1 and p==q+1, N -= 1
    d.rank = (p > q + 1) ? p : (q + 1);
    if (z == 1.0 && p == q + 1) d.rank -= 1;
    return d;
}

// Position in the CMF lattice
struct Position {
    double v[MAX_AXES];
};

// Generate the known zeta initial point for a pFq CMF.
// For zeta_n, the initial point is (1,...,1; 2,...,2) with z=1.
// x_i = 1 for all p numerator axes, y_i = 2 for all q denominator axes.
// This corresponds to the Riemann zeta function and its generalizations.
// Example: 6F5 zeta5 = (1,1,1,1,1,1; 2,2,2,2,2) with z=1
__host__ __device__
inline Position zeta_initial_point(int p, int q) {
    Position pos;
    for (int i = 0; i < p; ++i) pos.v[i] = 1.0;       // x_i = 1
    for (int i = 0; i < q; ++i) pos.v[p + i] = 2.0;   // y_i = 2
    return pos;
}

// Small dense matrix (row-major)
struct WalkMatrix {
    int dim;
    double a[MAX_DIM][MAX_DIM];
};

// Initialize identity matrix
__host__ __device__
inline WalkMatrix identity_matrix(int dim) {
    WalkMatrix M;
    M.dim = dim;
    for (int i = 0; i < MAX_DIM; ++i)
        for (int j = 0; j < MAX_DIM; ++j)
            M.a[i][j] = (i == j && i < dim) ? 1.0 : 0.0;
    return M;
}

// Projective normalization: divide by largest absolute entry
// Returns the log of the scale factor via log_scale (accumulated)
__host__ __device__
inline bool normalize_projective(WalkMatrix &M, double &log_scale) {
    double mx = 0.0;
    for (int i = 0; i < M.dim; ++i) {
        for (int j = 0; j < M.dim; ++j) {
            double x = fabs(M.a[i][j]);
            if (x > mx) mx = x;
        }
    }
    if (!(mx > 0.0) || !isfinite(mx)) return false;
    double inv = 1.0 / mx;
    for (int i = 0; i < M.dim; ++i)
        for (int j = 0; j < M.dim; ++j)
            M.a[i][j] *= inv;
    log_scale += log(mx);  // Accumulate log of scale factor
    return true;
}

// Overload without scale tracking (for compatibility)
__host__ __device__
inline bool normalize_projective(WalkMatrix &M) {
    double dummy_scale = 0.0;
    return normalize_projective(M, dummy_scale);
}

// Compute polynomial coefficients of prod_j(theta + vals[j])
// out[k] = coefficient of theta^k, k=0..count
__host__ __device__
inline void poly_theta_plus(const double *vals, int count, double *out) {
    for (int k = 0; k <= count; ++k) out[k] = 0.0;
    out[0] = 1.0;
    int deg = 0;
    for (int j = 0; j < count; ++j) {
        double next[MAX_DIM + 1] = {0};
        for (int k = 0; k <= deg; ++k) {
            next[k]     += out[k] * vals[j];   // * vals[j]
            next[k + 1] += out[k];             // * theta
        }
        ++deg;
        for (int k = 0; k <= deg; ++k) out[k] = next[k];
    }
}

// Build the last column of the monic theta companion matrix.
// D(theta) = theta * prod_i(theta + y_i - 1) - z * prod_j(theta + x_j)
// Companion: subdiagonal 1s, last column = [-d0/lead, ..., -d_{dim-1}/lead]
__host__ __device__
inline bool theta_companion_column(const Position &pos, double z,
                                    int dim, int p, int q, double col[MAX_DIM]) {
    double xvals[MAX_AXES];
    double yvals[MAX_AXES];
    for (int i = 0; i < p; ++i) xvals[i] = pos.v[i];
    for (int i = 0; i < q; ++i) yvals[i] = pos.v[p + i] - 1.0;

    double px[MAX_DIM + 1] = {0};
    double py[MAX_DIM + 1] = {0};
    poly_theta_plus(xvals, p, px);
    poly_theta_plus(yvals, q, py);

    // D coefficients, ascending in theta
    double d[MAX_DIM + 1] = {0};
    for (int k = 0; k <= p; ++k) d[k] = -z * px[k];
    for (int k = 0; k <= q; ++k) d[k + 1] += py[k];

    double lead = d[dim];
    double lead_scale = 1.0;
    if (dim == p && p == q + 1) {
        // LC(D) = 1 - z (for the generic case)
        lead_scale = fmax(1.0, fabs(1.0 - z));
    } else {
        for (int i = 0; i < p; ++i) lead_scale += fabs(pos.v[i]);
        for (int i = 0; i < q; ++i) lead_scale += fabs(pos.v[p + i] - 1.0);
    }
    if (!isfinite(lead) || fabs(lead) <= 64.0 * 2.220446049250313e-16 * lead_scale)
        return false;

    for (int i = 0; i < dim; ++i) {
        col[i] = -d[i] / lead;
        if (!isfinite(col[i])) return false;
    }
    return true;
}

// Right multiply M by A = I + C/a (companion form)
// Each row update is O(dim) using companion structure
// log_scale accumulates the normalization scale factor
__host__ __device__
inline bool right_mul_affine_companion(WalkMatrix &M,
                                        const double col[MAX_DIM],
                                        double denom, bool inverse,
                                        double &log_scale) {
    const int dim = M.dim;
    const double eps = 2.220446049250313e-16;
    if (!isfinite(denom) || fabs(denom) <= 64.0 * eps) return false;

    WalkMatrix out;
    out.dim = dim;

    if (!inverse) {
        double inva = 1.0 / denom;
        for (int r = 0; r < dim; ++r) {
            double dot = 0.0;
            for (int k = 0; k < dim; ++k) dot += M.a[r][k] * col[k];
            for (int j = 0; j < dim - 1; ++j)
                out.a[r][j] = M.a[r][j] + M.a[r][j + 1] * inva;
            out.a[r][dim - 1] = M.a[r][dim - 1] + dot * inva;
        }
    } else {
        double inva = 1.0 / denom;
        for (int r = 0; r < dim; ++r) {
            double alpha[MAX_DIM] = {0};
            double beta[MAX_DIM] = {0};
            alpha[dim - 1] = 0.0;
            beta[dim - 1] = 1.0;
            for (int j = dim - 2; j >= 0; --j) {
                alpha[j] = M.a[r][j] - alpha[j + 1] * inva;
                beta[j]  = -beta[j + 1] * inva;
            }
            double dot_alpha = 0.0, dot_beta = 0.0;
            for (int k = 0; k < dim; ++k) {
                dot_alpha += alpha[k] * col[k];
                dot_beta  += beta[k]  * col[k];
            }
            double solve_denom = 1.0 + dot_beta * inva;
            double solve_scale = fmax(1.0, fabs(dot_beta * inva));
            if (!isfinite(solve_denom) ||
                fabs(solve_denom) <= 128.0 * eps * solve_scale)
                return false;
            double t = (M.a[r][dim - 1] - dot_alpha * inva) / solve_denom;
            if (!isfinite(t)) return false;
            for (int j = 0; j < dim; ++j) {
                out.a[r][j] = alpha[j] + beta[j] * t;
                if (!isfinite(out.a[r][j])) return false;
            }
        }
    }

    M = out;
    return normalize_projective(M, log_scale);
}

// Overload without scale tracking (for compatibility)
__host__ __device__
inline bool right_mul_affine_companion(WalkMatrix &M,
                                        const double col[MAX_DIM],
                                        double denom, bool inverse) {
    double dummy_scale = 0.0;
    return right_mul_affine_companion(M, col, denom, inverse, dummy_scale);
}

// Apply one CMF lattice-axis step
// x_i +1:  I + C(pos) / x_i
// x_i -1:  [I + C(pos with x_i-1) / (x_i-1)]^{-1}
// y_i -1:  I + C(pos) / (y_i-1)
// y_i +1:  [I + C(pos with y_i+1) / y_i]^{-1}
__host__ __device__
inline bool apply_axis_step(WalkMatrix &walk, Position &pos,
                            int axis, int sign, const CMFDesc &desc,
                            double &log_scale) {
    const int dim = walk.dim;
    Position eval_pos = pos;
    double denom = 0.0;
    bool inverse = false;

    if (axis < desc.p) {
        // x axis
        if (sign > 0) {
            denom = pos.v[axis];
        } else {
            eval_pos.v[axis] -= 1.0;
            denom = pos.v[axis] - 1.0;
            inverse = true;
        }
    } else {
        // y axis
        if (sign < 0) {
            denom = pos.v[axis] - 1.0;
        } else {
            eval_pos.v[axis] += 1.0;
            denom = pos.v[axis];
            inverse = true;
        }
    }

    double col[MAX_DIM] = {0};
    if (!theta_companion_column(eval_pos, desc.z, dim, desc.p, desc.q, col))
        return false;
    if (!right_mul_affine_companion(walk, col, denom, inverse, log_scale))
        return false;
    pos.v[axis] += (double)sign;
    return true;
}

// Overload without scale tracking
__host__ __device__
inline bool apply_axis_step(WalkMatrix &walk, Position &pos,
                            int axis, int sign, const CMFDesc &desc) {
    double dummy_scale = 0.0;
    return apply_axis_step(walk, pos, axis, sign, desc, dummy_scale);
}

// Apply one full trajectory displacement
// Deterministic path: consume largest-magnitude layers first; within each
// simple diagonal use reverse axis order. CMF conservation makes any
// nonsingular path equivalent.
__host__ __device__
inline bool apply_trajectory_step(WalkMatrix &walk, Position &pos,
                                   const int dir[MAX_AXES], const CMFDesc &desc,
                                   double &log_scale) {
    int max_abs = 0;
    for (int a = 0; a < desc.naxes; ++a)
        max_abs = max_abs > abs(dir[a]) ? max_abs : abs(dir[a]);

    for (int level = max_abs; level >= 1; --level) {
        for (int axis = desc.naxes - 1; axis >= 0; --axis) {
            if (abs(dir[axis]) < level) continue;
            int sign = (dir[axis] > 0) ? 1 : -1;
            if (!apply_axis_step(walk, pos, axis, sign, desc, log_scale))
                return false;
        }
    }
    return true;
}

// Overload without scale tracking
__host__ __device__
inline bool apply_trajectory_step(WalkMatrix &walk, Position &pos,
                                   const int dir[MAX_AXES], const CMFDesc &desc) {
    double dummy_scale = 0.0;
    return apply_trajectory_step(walk, pos, dir, desc, dummy_scale);
}

// Snapshot structure for limit/delta computation
// log_scale accumulates the log of normalization scale factors,
// allowing recovery of actual p,q magnitudes:
//   actual_q = normalized_q * exp(log_scale)
struct WalkSnapshots {
    WalkMatrix m1;  // after N-2h
    WalkMatrix m2;  // after N-h
    WalkMatrix m3;  // after N
    double log_scale1;  // accumulated log scale at N-2h
    double log_scale2;  // accumulated log scale at N-h
    double log_scale3;  // accumulated log scale at N
    bool valid;
};

// Forward walk from start, applying trajectory dir repeatedly
// Tracks normalization scale for correct delta computation
__host__ __device__
inline bool forward_walk(const int dir[MAX_AXES], const Position &start,
                          const CMFDesc &desc, int N, int h,
                          WalkSnapshots &snaps) {
    if (N <= 0 || h <= 0 || N - 2 * h <= 0) return false;
    WalkMatrix walk = identity_matrix(desc.rank);
    Position pos = start;
    int cp1 = N - 2 * h;
    int cp2 = N - h;

    double log_scale = 0.0;
    snaps.valid = false;
    snaps.log_scale1 = snaps.log_scale2 = snaps.log_scale3 = 0.0;

    for (int step = 1; step <= N; ++step) {
        if (!apply_trajectory_step(walk, pos, dir, desc, log_scale))
            return false;
        if (step == cp1) { snaps.m1 = walk; snaps.log_scale1 = log_scale; }
        if (step == cp2) { snaps.m2 = walk; snaps.log_scale2 = log_scale; }
        if (step == N)   { snaps.m3 = walk; snaps.log_scale3 = log_scale; }
    }
    snaps.valid = true;
    return true;
}

// Limit result
struct LimitResult {
    double L;
    int row_i;
    int row_j;
    int column;
    int method;  // 0=raw, 1=aitken, 2=lyap
};

// Compute limits from walk snapshots
// Uses the ramanujantools approach: p = M[0, dim-1], q = M[1, dim-1]
// Also scans other column ratios for additional candidate limits
__host__ __device__
inline void compute_limits(const WalkSnapshots &snaps,
                           LimitResult *results, int &n_results, int max_results) {
    const int dim = snaps.m3.dim;
    const double DEN_MIN = 1e-280;
    n_results = 0;

    // Primary limit: ramanujantools default projection
    // p = M[0, dim-1], q = M[1, dim-1], limit = p/q
    {
        int col = dim - 1;
        double p3 = snaps.m3.a[0][col], q3 = snaps.m3.a[1][col];
        double p2 = snaps.m2.a[0][col], q2 = snaps.m2.a[1][col];
        double p1 = snaps.m1.a[0][col], q1 = snaps.m1.a[1][col];

        if (fabs(q3) > DEN_MIN && fabs(q2) > DEN_MIN && fabs(q1) > DEN_MIN) {
            double r1 = p1 / q1, r2 = p2 / q2, r3 = p3 / q3;
            if (isfinite(r1) && isfinite(r2) && isfinite(r3)) {
                double sc = fmax(1.0, fmax(fabs(r1), fmax(fabs(r2), fabs(r3))));

                // Aitken delta-squared on primary projection
                double denom = r1 - 2.0 * r2 + r3;
                if (fabs(denom) > 1e-14 * sc) {
                    double d = r2 - r3;
                    double L = r3 - d * d / denom;
                    if (isfinite(L) && n_results < max_results)
                        results[n_results++] = {L, 0, 1, col, 1};
                }

                // Raw ratio (ramanujantools default)
                if (fabs(r3) > 1e-15 && n_results < max_results)
                    results[n_results++] = {r3, 0, 1, col, 0};
            }
        }
    }

    // Secondary: scan other column ratios for additional candidate limits
    for (int col = 0; col < dim; ++col) {
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (i == j) continue;
                // Skip the primary projection (already done above)
                if (col == dim - 1 && i == 0 && j == 1) continue;
                if (col == dim - 1 && i == 0 && j == 1) continue;

                double d1 = snaps.m1.a[j][col];
                double d2 = snaps.m2.a[j][col];
                double d3 = snaps.m3.a[j][col];
                if (fabs(d1) < DEN_MIN || fabs(d2) < DEN_MIN || fabs(d3) < DEN_MIN)
                    continue;

                double r1 = snaps.m1.a[i][col] / d1;
                double r2 = snaps.m2.a[i][col] / d2;
                double r3 = snaps.m3.a[i][col] / d3;

                if (!isfinite(r1) || !isfinite(r2) || !isfinite(r3)) continue;
                double sc = fmax(1.0, fmax(fabs(r1), fmax(fabs(r2), fabs(r3))));

                // Aitken delta-squared
                double denom = r1 - 2.0 * r2 + r3;
                if (fabs(denom) > 1e-14 * sc) {
                    double d = r2 - r3;
                    double L = r3 - d * d / denom;
                    if (isfinite(L) && fabs(L) > 1e-15 && n_results < max_results) {
                        results[n_results++] = {L, i, j, col, 1};
                    }
                }

                // Raw
                if (fabs(r3) > 1e-15 && n_results < max_results) {
                    results[n_results++] = {r3, i, j, col, 0};
                }
            }
        }
    }
}

// Compute delta (irrationality measure) using ramanujantools formula:
//   delta = -(1 + log|L - p/q| / log(q))
// where q is the ACTUAL (un-normalized) denominator.
//
// Following ramanujantools CMF.delta():
//   - Use the deeper snapshot (N) as the "true" limit L = p_N/q_N
//   - Use the shallower snapshot (N-h) as the approximation p_{N-h}/q_{N-h}
//   - Use q_{N-h} (with recovered scale) as the denominator
// This gives a meaningful delta because the approximation at N-h differs
// from the limit at N.
//
// We keep everything in log space to avoid overflow (exp(log_scale) can be ~10^300).
//
// Reference: ramanujantools Limit.delta(L), CMF.delta()
__host__ __device__
inline double compute_delta(const WalkSnapshots &snaps, double L) {
    if (!snaps.valid) return NAN;

    const int dim = snaps.m3.dim;
    const int col = dim - 1;

    // Use N-h snapshot as the approximation, N as the "true" limit
    // (following ramanujantools CMF.delta which uses [depth, 2*depth])
    double p_approx = snaps.m2.a[0][col];  // p at N-h
    double q_approx = snaps.m2.a[1][col];  // q at N-h (normalized)

    if (fabs(q_approx) < 1e-280) return NAN;

    // Recover log(actual_q) in log space
    double log_q = log(fabs(q_approx)) + snaps.log_scale2;
    if (log_q <= 0.0) return NAN;

    // Limit estimate: use provided L, or Aitken, or p_N/q_N
    double p_N = snaps.m3.a[0][col];
    double q_N = snaps.m3.a[1][col];
    double L_N = (fabs(q_N) > 1e-280) ? p_N / q_N : NAN;

    double L_used = L;
    if (!isfinite(L_used)) {
        // Aitken acceleration from three depths
        double p_Nh = snaps.m2.a[0][col], q_Nh = snaps.m2.a[1][col];
        double p_N2h = snaps.m1.a[0][col], q_N2h = snaps.m1.a[1][col];
        if (fabs(q_Nh) > 1e-280 && fabs(q_N2h) > 1e-280) {
            double r1 = p_N2h / q_N2h, r2 = p_Nh / q_Nh, r3 = L_N;
            double denom = r1 - 2.0 * r2 + r3;
            if (isfinite(denom) && fabs(denom) > 1e-30) {
                double d = r2 - r3;
                L_used = r3 - d * d / denom;
            }
        }
        if (!isfinite(L_used)) L_used = L_N;
    }

    if (!isfinite(L_used)) return NAN;

    // delta = -(1 + log|L - p_approx/q_approx| / log(q_approx))
    double approx = p_approx / q_approx;
    if (!isfinite(approx)) return NAN;

    double diff = fabs(L_used - approx);
    if (diff <= 0.0) return INFINITY;
    if (!isfinite(diff)) return NAN;

    double log_diff = log(diff);
    return -(1.0 + log_diff / log_q);
}

// Compute precision (convergence quality) in digits
// ramanujantools: precision = -log10(|p_N/q_N - p_{N-1}/q_{N-1}|)
__host__ __device__
inline double compute_precision(const WalkSnapshots &snaps) {
    if (!snaps.valid) return NAN;

    const int dim = snaps.m3.dim;
    const int col = dim - 1;

    double p_N = snaps.m3.a[0][col], q_N = snaps.m3.a[1][col];
    double p_Nh = snaps.m2.a[0][col], q_Nh = snaps.m2.a[1][col];

    if (fabs(q_N) < 1e-280 || fabs(q_Nh) < 1e-280) return NAN;

    double L_N = p_N / q_N;
    double L_Nh = p_Nh / q_Nh;

    if (!isfinite(L_N) || !isfinite(L_Nh)) return NAN;

    double diff = fabs(L_N - L_Nh);
    if (diff <= 0.0) return INFINITY;

    return -log10(diff);
}

// Compute kamidelta using ramanujantools spectral approach
// ramanujantools Matrix.kamidelta:
//   errors = [log(|lambda_0| / |lambda_i|) for subleading eigenvalues]
//   slope = linear fit of log(q_reduced) vs depth
//   delta = -1 + error / slope
//
// For CUDA, we approximate this using the convergence rate and
// the recovered q scale. This is an approximation; for exact delta,
// use the RNS variant with full CRT reconstruction.
__host__ __device__
inline double compute_kamidelta(const WalkSnapshots &snaps, double L_target) {
    if (!snaps.valid) return NAN;

    const int dim = snaps.m3.dim;
    const int col = dim - 1;

    double p_N = snaps.m3.a[0][col], q_N = snaps.m3.a[1][col];
    double p_Nh = snaps.m2.a[0][col], q_Nh = snaps.m2.a[1][col];
    double p_N2h = snaps.m1.a[0][col], q_N2h = snaps.m1.a[1][col];

    if (fabs(q_N) < 1e-280 || fabs(q_Nh) < 1e-280 || fabs(q_N2h) < 1e-280)
        return NAN;

    // Use the proper delta formula with recovered q scale
    // First, estimate the limit using Aitken acceleration
    double L_N = p_N / q_N;
    double L_Nh = p_Nh / q_Nh;
    double L_N2h = p_N2h / q_N2h;

    if (!isfinite(L_N) || !isfinite(L_Nh) || !isfinite(L_N2h)) return NAN;

    double L_aitken = NAN;
    double denom_aitk = L_N2h - 2.0 * L_Nh + L_N;
    if (fabs(denom_aitk) > 1e-30) {
        double d = L_Nh - L_N;
        L_aitken = L_N - d * d / denom_aitk;
    }

    double L = isfinite(L_target) ? L_target : (isfinite(L_aitken) ? L_aitken : L_N);

    // Use the proper delta formula with recovered q
    return compute_delta(snaps, L);
}
