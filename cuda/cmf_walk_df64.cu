// cmf_walk_df64.cu
//
// Variant 3: CUDA df64 (double-single) pFq CMF walk
//
// Uses two float32 values (hi, lo) to represent ~48-bit precision,
// emulating double-single arithmetic with float32 operations.
// On H100, FP32 is full-rate while FP64 is 1/2-rate, so df64 can
// potentially achieve higher throughput than native float64 for
// similar precision.
//
// Based on the Knuth double-single algorithm used in the Apple Silicon
// Metal implementation (bench_kernels.metal).
//
// Build:
//   nvcc -O3 -arch=sm_90 -std=c++17 -o cmf_walk_df64 cmf_walk_df64.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cuda_runtime.h>

#include "cmf_pfq.cuh"

#define MAX_LIMITS 256

// ---- Double-single (df64) arithmetic ----
// Each number is represented as (hi, lo) where value = hi + lo
// and |lo| << |hi|. This gives ~48 bits of mantissa.

struct df64 {
    float hi, lo;
};

__host__ __device__
inline df64 make_df64(double x) {
    df64 r;
    r.hi = (float)x;
    r.lo = (float)(x - (double)r.hi);
    return r;
}

__host__ __device__
inline double to_double(df64 x) {
    return (double)x.hi + (double)x.lo;
}

// Two-sum: error-free addition of two floats
__host__ __device__
inline void two_sum(float a, float b, float &hi, float &lo) {
    hi = a + b;
    float bv = hi - a;
    lo = (a - (hi - bv)) + (b - bv);
}

// Two-product: error-free multiplication using fmaf
__host__ __device__
inline void two_prod(float a, float b, float &hi, float &lo) {
    hi = a * b;
#if defined(__CUDA_ARCH__)
    lo = __fmaf_rn(a, b, -hi);
#else
    lo = fmaf(a, b, -hi);
#endif
}

// df64 + df64
__host__ __device__
inline df64 df64_add(df64 a, df64 b) {
    float shi, slo;
    two_sum(a.hi, b.hi, shi, slo);
    slo += a.lo + b.lo;
    two_sum(shi, slo, shi, slo);
    return {shi, slo};
}

// df64 - df64
__host__ __device__
inline df64 df64_sub(df64 a, df64 b) {
    df64 nb = {-b.hi, -b.lo};
    return df64_add(a, nb);
}

// df64 * df64
__host__ __device__
inline df64 df64_mul(df64 a, df64 b) {
    float ph, pl;
    two_prod(a.hi, b.hi, ph, pl);
    pl += a.hi * b.lo + a.lo * b.hi;
    two_sum(ph, pl, ph, pl);
    return {ph, pl};
}

// df64 * float
__host__ __device__
inline df64 df64_mul_f(df64 a, float b) {
    float ph, pl;
    two_prod(a.hi, b, ph, pl);
    pl += a.lo * b;
    two_sum(ph, pl, ph, pl);
    return {ph, pl};
}

// df64 / float
__host__ __device__
inline df64 df64_div_f(df64 a, float b) {
    float inv_b = 1.0f / b;
    return df64_mul_f(a, inv_b);
}

// isfinite
__host__ __device__
inline bool df64_finite(df64 a) {
    return isfinite(a.hi) && isfinite(a.lo);
}

// ---- df64 matrix operations ----

struct df64Matrix {
    int dim;
    df64 a[MAX_DIM][MAX_DIM];
};

__host__ __device__
inline df64Matrix df64_identity(int dim) {
    df64Matrix M;
    M.dim = dim;
    for (int i = 0; i < MAX_DIM; ++i)
        for (int j = 0; j < MAX_DIM; ++j)
            M.a[i][j] = (i == j && i < dim) ? make_df64(1.0) : make_df64(0.0);
    return M;
}

__host__ __device__
inline bool df64_normalize(df64Matrix &M, double &log_scale) {
    float mx = 0.0f;
    for (int i = 0; i < M.dim; ++i)
        for (int j = 0; j < M.dim; ++j) {
            float x = fabsf(M.a[i][j].hi);
            if (x > mx) mx = x;
        }
    if (!(mx > 0.0f) || !isfinite(mx)) return false;
    float inv = 1.0f / mx;
    for (int i = 0; i < M.dim; ++i)
        for (int j = 0; j < M.dim; ++j)
            M.a[i][j] = df64_mul_f(M.a[i][j], inv);
    log_scale += log((double)mx);  // Accumulate log of scale factor
    return true;
}

// Overload without scale tracking (for compatibility)
__host__ __device__
inline bool df64_normalize(df64Matrix &M) {
    double dummy = 0.0;
    return df64_normalize(M, dummy);
}

// Theta companion column in df64
__host__ __device__
inline bool theta_companion_column_df64(const Position &pos, double z,
                                         int dim, int p, int q,
                                         df64 col[MAX_DIM]) {
    double cold[MAX_DIM] = {0};
    if (!theta_companion_column(pos, z, dim, p, q, cold)) return false;
    for (int i = 0; i < dim; ++i) col[i] = make_df64(cold[i]);
    return true;
}

// Right multiply by I + C/a in df64
__host__ __device__
inline bool right_mul_df64(df64Matrix &M, const df64 col[MAX_DIM],
                           df64 denom, bool inverse, double &log_scale) {
    const int dim = M.dim;
    if (!df64_finite(denom) || fabs(denom.hi) <= 64.0f * 1.19e-7f) return false;

    df64Matrix out;
    out.dim = dim;

    if (!inverse) {
        df64 inva = df64_div_f(make_df64(1.0), denom.hi);
        for (int r = 0; r < dim; ++r) {
            df64 dot = make_df64(0.0);
            for (int k = 0; k < dim; ++k)
                dot = df64_add(dot, df64_mul(M.a[r][k], col[k]));
            for (int j = 0; j < dim - 1; ++j)
                out.a[r][j] = df64_add(M.a[r][j], df64_mul(M.a[r][j+1], inva));
            out.a[r][dim-1] = df64_add(M.a[r][dim-1], df64_mul(dot, inva));
        }
    } else {
        df64 inva = df64_div_f(make_df64(1.0), denom.hi);
        for (int r = 0; r < dim; ++r) {
            df64 alpha[MAX_DIM], beta[MAX_DIM];
            alpha[dim-1] = make_df64(0.0);
            beta[dim-1] = make_df64(1.0);
            for (int j = dim-2; j >= 0; --j) {
                alpha[j] = df64_sub(M.a[r][j], df64_mul(alpha[j+1], inva));
                beta[j] = df64_sub(make_df64(0.0), df64_mul(beta[j+1], inva));
            }
            df64 da = make_df64(0.0), db = make_df64(0.0);
            for (int k = 0; k < dim; ++k) {
                da = df64_add(da, df64_mul(alpha[k], col[k]));
                db = df64_add(db, df64_mul(beta[k], col[k]));
            }
            df64 sd = df64_add(make_df64(1.0), df64_mul(db, inva));
            if (!df64_finite(sd) || fabs(sd.hi) <= 1e-10f) return false;
            df64 inv_sd = df64_div_f(make_df64(1.0), sd.hi);
            df64 t = df64_mul(df64_sub(M.a[r][dim-1], df64_mul(da, inva)), inv_sd);
            for (int j = 0; j < dim; ++j)
                out.a[r][j] = df64_add(alpha[j], df64_mul(beta[j], t));
        }
    }

    M = out;
    return df64_normalize(M, log_scale);
}

// Overload without scale tracking
__host__ __device__
inline bool right_mul_df64(df64Matrix &M, const df64 col[MAX_DIM],
                           df64 denom, bool inverse) {
    double dummy = 0.0;
    return right_mul_df64(M, col, denom, inverse, dummy);
}

__host__ __device__
inline bool apply_axis_step_df64(df64Matrix &walk, Position &pos,
                                   int axis, int sign, const CMFDesc &desc,
                                   double &log_scale) {
    Position eval_pos = pos;
    double denom = 0.0;
    bool inverse = false;

    if (axis < desc.p) {
        if (sign > 0) { denom = pos.v[axis]; }
        else { eval_pos.v[axis] -= 1.0; denom = pos.v[axis] - 1.0; inverse = true; }
    } else {
        if (sign < 0) { denom = pos.v[axis] - 1.0; }
        else { eval_pos.v[axis] += 1.0; denom = pos.v[axis]; inverse = true; }
    }

    df64 col[MAX_DIM] = {0};
    if (!theta_companion_column_df64(eval_pos, desc.z, walk.dim, desc.p, desc.q, col))
        return false;
    if (!right_mul_df64(walk, col, make_df64(denom), inverse, log_scale))
        return false;
    pos.v[axis] += (double)sign;
    return true;
}

// Overload without scale tracking
__host__ __device__
inline bool apply_axis_step_df64(df64Matrix &walk, Position &pos,
                                   int axis, int sign, const CMFDesc &desc) {
    double dummy = 0.0;
    return apply_axis_step_df64(walk, pos, axis, sign, desc, dummy);
}

__host__ __device__
inline bool apply_trajectory_step_df64(df64Matrix &walk, Position &pos,
                                        const int dir[MAX_AXES], const CMFDesc &desc,
                                        double &log_scale) {
    int max_abs = 0;
    for (int a = 0; a < desc.naxes; ++a)
        max_abs = max_abs > abs(dir[a]) ? max_abs : abs(dir[a]);
    for (int level = max_abs; level >= 1; --level) {
        for (int axis = desc.naxes - 1; axis >= 0; --axis) {
            if (abs(dir[axis]) < level) continue;
            int sign = (dir[axis] > 0) ? 1 : -1;
            if (!apply_axis_step_df64(walk, pos, axis, sign, desc, log_scale)) return false;
        }
    }
    return true;
}

// Convert df64 matrix to double for limit computation
__host__ __device__
inline void df64_to_walk(const df64Matrix &src, WalkMatrix &dst) {
    dst.dim = src.dim;
    for (int i = 0; i < src.dim; ++i)
        for (int j = 0; j < src.dim; ++j)
            dst.a[i][j] = to_double(src.a[i][j]);
}

struct DF64Input {
    int dir[MAX_AXES];
    Position start;
};

struct DF64Result {
    double limit;
    double delta;
    double p;
    double q;
    double precision;
    double log_scale;
    int valid;
    int n_limits;
};

__global__
void walk_kernel_df64(const DF64Input *inputs, DF64Result *results,
                       CMFDesc desc, int N, int h, int n_traj) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;

    const DF64Input &inp = inputs[idx];
    df64Matrix walk = df64_identity(desc.rank);
    Position pos = inp.start;

    WalkMatrix snap1, snap2, snap3;
    int cp1 = N - 2 * h, cp2 = N - h;
    bool ok = true;
    double log_scale = 0.0;
    double ls1 = 0.0, ls2 = 0.0, ls3 = 0.0;

    for (int step = 1; step <= N; ++step) {
        if (!apply_trajectory_step_df64(walk, pos, inp.dir, desc, log_scale)) {
            ok = false; break;
        }
        if (step == cp1) { df64_to_walk(walk, snap1); ls1 = log_scale; }
        if (step == cp2) { df64_to_walk(walk, snap2); ls2 = log_scale; }
        if (step == N)   { df64_to_walk(walk, snap3); ls3 = log_scale; }
    }

    DF64Result &res = results[idx];
    res.valid = ok ? 1 : 0;
    if (!ok) { res.limit = NAN; res.delta = NAN; res.precision = NAN; res.log_scale = NAN; res.n_limits = 0; return; }

    WalkSnapshots snaps = {snap1, snap2, snap3, ls1, ls2, ls3, true};
    LimitResult limits[MAX_LIMITS];
    int n_lim = 0;
    compute_limits(snaps, limits, n_lim, MAX_LIMITS);
    res.n_limits = n_lim;

    res.limit = NAN;
    for (int i = 0; i < n_lim; ++i) if (limits[i].method == 1) { res.limit = limits[i].L; break; }
    if (isnan(res.limit)) for (int i = 0; i < n_lim; ++i) if (limits[i].method == 2) { res.limit = limits[i].L; break; }
    if (isnan(res.limit) && n_lim > 0) res.limit = limits[0].L;

    int col = desc.rank - 1;
    res.p = snap3.a[0][col]; res.q = snap3.a[1][col];
    res.precision = compute_precision(snaps);
    res.log_scale = ls3;  // Track accumulated log scale at depth N
    res.delta = compute_kamidelta(snaps, res.limit);
}

bool parse_cmf_type(const char *s, int &p, int &q) { return sscanf(s, "%dF%d", &p, &q) == 2; }

void generate_initial_points(int p, int q, int count, Position *points, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i)
        for (int a = 0; a < p+q; ++a) points[i].v[a] = 2.0 + (double)(rng() % 19);
}

void generate_trajectories(int p, int q, int count, int *dirs, uint64_t seed, int dmax = 3) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        for (int a = 0; a < p+q; ++a) dirs[i*MAX_AXES+a] = (int)(rng() % (2*dmax+1)) - dmax;
        bool az = true;
        for (int a = 0; a < p+q; ++a) if (dirs[i*MAX_AXES+a]) { az = false; break; }
        if (az) dirs[i*MAX_AXES] = 1;
    }
}

int run_benchmark(int p, int q, double z, int N, int h, int n_traj,
                  int mode, const char *output_path, int block_size, bool use_zeta) {
    CMFDesc desc = make_cmf_desc(p, q, z);
    fprintf(stderr, "=== df64: %dF%d z=%.4f N=%d h=%d n_traj=%d rank=%d%s ===\n",
            p, q, z, N, h, n_traj, desc.rank, use_zeta ? " [ZETA]" : "");

    std::vector<DF64Input> h_inputs(n_traj);
    std::vector<DF64Result> h_results(n_traj);

    if (mode == 0) {
        if (use_zeta) {
            Position zeta_start = zeta_initial_point(p, q);
            for (int i = 0; i < n_traj; ++i) h_inputs[i].start = zeta_start;
        } else {
            generate_initial_points(p, q, n_traj, (Position*)h_inputs.data(), 42);
        }
        for (int i = 0; i < n_traj; ++i) { memset(h_inputs[i].dir, 0, sizeof(h_inputs[i].dir)); h_inputs[i].dir[0] = 1; }
    } else {
        Position fs;
        if (use_zeta) {
            fs = zeta_initial_point(p, q);
        } else {
            for (int a = 0; a < p+q; ++a) fs.v[a] = (a < p) ? 2.0 : 3.0;
        }
        for (int i = 0; i < n_traj; ++i) h_inputs[i].start = fs;
        generate_trajectories(p, q, n_traj, (int*)h_inputs.data(), 42);
    }

    DF64Input *d_inputs; DF64Result *d_results;
    cudaMalloc(&d_inputs, n_traj * sizeof(DF64Input));
    cudaMalloc(&d_results, n_traj * sizeof(DF64Result));
    cudaMemcpy(d_inputs, h_inputs.data(), n_traj * sizeof(DF64Input), cudaMemcpyHostToDevice);

    int grid = (n_traj + block_size - 1) / block_size;
    walk_kernel_df64<<<1, 1>>>(d_inputs, d_results, desc, N, h, 1);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    size_t mb, mt; cudaMemGetInfo(&mb, &mt);
    cudaEventRecord(start);
    walk_kernel_df64<<<grid, block_size>>>(d_inputs, d_results, desc, N, h, n_traj);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    size_t mf; cudaMemGetInfo(&mf, &mt); size_t mu = mt - mf;
    cudaMemcpy(h_results.data(), d_results, n_traj * sizeof(DF64Result), cudaMemcpyDeviceToHost);

    int nv = 0, nl = 0; double sd = 0;
    for (int i = 0; i < n_traj; ++i) {
        if (h_results[i].valid) { nv++; if (!isnan(h_results[i].limit)) nl++; if (!isnan(h_results[i].delta)) sd += h_results[i].delta; }
    }
    double ad = (nv > 0) ? sd / nv : NAN;
    double tps = n_traj / (ms / 1000.0);
    fprintf(stderr, "  Valid: %d/%d  Limits: %d  Avg delta: %.6f  Time: %.3f ms  %.0f traj/s  Mem: %.2f MB\n",
            nv, n_traj, nl, ad, ms, tps, mu / 1e6);

    std::ofstream out(output_path);
    out << "{\n  \"variant\": \"df64\",\n  \"cmf\": \"" << p << "F" << q << "\",\n";
    out << "  \"p\": " << p << ", \"q\": " << q << ", \"z\": " << z << ",\n";
    out << "  \"rank\": " << desc.rank << ", \"naxes\": " << desc.naxes << ",\n";
    out << "  \"N\": " << N << ", \"h\": " << h << ", \"n_traj\": " << n_traj << ",\n";
    out << "  \"mode\": \"" << (mode == 0 ? "limit" : "delta") << "\",\n";
    out << "  \"block_size\": " << block_size << ",\n";
    out << "  \"gpu_time_ms\": " << ms << ", \"throughput_traj_per_s\": " << tps << ",\n";
    out << "  \"mem_used_mb\": " << mu / 1e6 << ",\n";
    char dbuf[32];
    json_double(dbuf, sizeof(dbuf), ad);
    out << "  \"n_valid\": " << nv << ", \"n_with_limits\": " << nl << ", \"avg_delta\": " << dbuf << "\n}\n";

    cudaFree(d_inputs); cudaFree(d_results);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

int main(int argc, char **argv) {
    int p = 6, q = 5; double z = 0.5; int N = 1000, h = 50, n_traj = 10000;
    int mode = 0, block_size = 256;
    bool use_zeta = false;
    const char *output_path = "results_df64.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* { if (++i >= argc) exit(1); return argv[i]; };
        if (arg == "--cmf" && i+1 < argc) { if (!parse_cmf_type(argv[++i], p, q)) return 1; }
        else if (arg == "--z") z = atof(next());
        else if (arg == "--N") N = atoi(next());
        else if (arg == "--h") h = atoi(next());
        else if (arg == "--n-traj") n_traj = atoi(next());
        else if (arg == "--mode") { mode = (strcmp(next(), "delta") == 0) ? 1 : 0; }
        else if (arg == "--block-size") block_size = atoi(next());
        else if (arg == "--output") output_path = next();
        else if (arg == "--zeta") { use_zeta = true; z = 1.0; }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "GPU: %s (SM %d.%d, %d MPs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return run_benchmark(p, q, z, N, h, n_traj, mode, output_path, block_size, use_zeta);
}
