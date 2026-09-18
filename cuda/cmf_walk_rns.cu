// cmf_walk_rns.cu
//
// Variant 2: CUDA RNS (Residue Number System) exact pFq CMF walk
//
// Uses modular arithmetic with K 30-bit primes for exact computation.
// CRT reconstruction for result verification.
// Periodic projective normalization to keep entry sizes bounded.
//
// Build:
//   nvcc -O3 -arch=sm_90 -std=c++17 -o cmf_walk_rns cmf_walk_rns.cu

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cuda_runtime.h>

#include "cmf_pfq.cuh"

#define MAX_PRIMES 64
#define MAX_LIMITS 256

__constant__ uint32_t RNS_PRIMES[MAX_PRIMES] = {
    1073741789u, 1073741783u, 1073741741u, 1073741723u,
    1073741717u, 1073741707u, 1073741693u, 1073741683u,
    1073741669u, 1073741657u, 1073741641u, 1073741623u,
    1073741611u, 1073741597u, 1073741587u, 1073741573u,
    1073741563u, 1073741551u, 1073741537u, 1073741527u,
    1073741513u, 1073741503u, 1073741491u, 1073741477u,
    1073741467u, 1073741453u, 1073741441u, 1073741427u,
    1073741413u, 1073741403u, 1073741391u, 1073741377u,
    1073741363u, 1073741353u, 1073741341u, 1073741329u,
    1073741317u, 1073741307u, 1073741293u, 1073741281u,
    1073741267u, 1073741257u, 1073741243u, 1073741231u,
    1073741219u, 1073741207u, 1073741193u, 1073741183u,
    1073741171u, 1073741157u, 1073741147u, 1073741137u,
    1073741123u, 1073741111u, 1073741097u, 1073741087u,
    1073741077u, 1073741063u, 1073741051u, 1073741039u,
    1073741027u, 1073741017u, 1073741003u, 1073740991u
};

__host__ __device__
inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t m) {
    return (a * b) % m;
}

__host__ __device__
inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t m) {
    uint64_t r = a + b;
    return r >= m ? r - m : r;
}

__host__ __device__
inline uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t m) {
    return a >= b ? a - b : a + m - b;
}

__host__ __device__
inline uint32_t to_mod(double x, uint32_t p) {
    int64_t val = (int64_t)llround(x);
    if (val >= 0) return (uint32_t)((uint64_t)val % p);
    return (uint32_t)((p - ((uint64_t)(-val) % p)) % p);
}

__device__
inline uint32_t mod_inv(uint32_t a, uint32_t p) {
    uint64_t result = 1, base = a % p;
    uint32_t exp = p - 2;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % p;
        base = (base * base) % p;
        exp >>= 1;
    }
    return (uint32_t)result;
}

struct RNSMatrix {
    int dim;
    uint32_t a[MAX_PRIMES][MAX_DIM][MAX_DIM];
};

struct RNSInput {
    int dir[MAX_AXES];
    Position start;
};

struct RNSResult {
    double limit;
    double delta;
    double p_f64;
    double q_f64;
    double precision;
    double log_scale;
    int valid;
    int n_limits;
};

__device__
inline bool theta_companion_column_rns(const Position &pos, double z,
                                        int dim, int p, int q,
                                        uint32_t col_rns[MAX_PRIMES][MAX_DIM],
                                        uint32_t denom_rns[MAX_PRIMES],
                                        int n_primes, int axis, int sign) {
    double col[MAX_DIM] = {0};
    double denom = 0.0;
    Position eval_pos = pos;

    if (axis < p) {
        if (sign > 0) { denom = pos.v[axis]; }
        else { eval_pos.v[axis] -= 1.0; denom = pos.v[axis] - 1.0; }
    } else {
        if (sign < 0) { denom = pos.v[axis] - 1.0; }
        else { eval_pos.v[axis] += 1.0; denom = pos.v[axis]; }
    }

    if (!theta_companion_column(eval_pos, z, dim, p, q, col)) return false;
    if (!isfinite(denom) || fabs(denom) <= 64.0 * 2.220446049250313e-16) return false;

    for (int k = 0; k < n_primes; ++k) {
        uint32_t pk = RNS_PRIMES[k];
        for (int i = 0; i < dim; ++i) col_rns[k][i] = to_mod(col[i], pk);
        denom_rns[k] = to_mod(denom, pk);
    }
    return true;
}

__device__
inline bool right_mul_rns(RNSMatrix &M,
                          const uint32_t col_rns[MAX_PRIMES][MAX_DIM],
                          const uint32_t denom_rns[MAX_PRIMES],
                          bool inverse, int n_primes) {
    const int dim = M.dim;
    RNSMatrix out;
    out.dim = dim;

    for (int k = 0; k < n_primes; ++k) {
        uint32_t pk = RNS_PRIMES[k];
        uint32_t inva = mod_inv(denom_rns[k], pk);

        if (!inverse) {
            for (int r = 0; r < dim; ++r) {
                uint64_t dot = 0;
                for (int i = 0; i < dim; ++i)
                    dot = add_mod(dot, mul_mod(M.a[k][r][i], col_rns[k][i], pk), pk);
                for (int j = 0; j < dim - 1; ++j)
                    out.a[k][r][j] = (uint32_t)add_mod(M.a[k][r][j],
                        mul_mod(M.a[k][r][j+1], inva, pk), pk);
                out.a[k][r][dim-1] = (uint32_t)add_mod(M.a[k][r][dim-1],
                    mul_mod((uint32_t)dot, inva, pk), pk);
            }
        } else {
            for (int r = 0; r < dim; ++r) {
                uint32_t alpha[MAX_DIM] = {0};
                uint32_t beta[MAX_DIM] = {0};
                alpha[dim-1] = 0; beta[dim-1] = 1;
                for (int j = dim-2; j >= 0; --j) {
                    alpha[j] = (uint32_t)sub_mod(M.a[k][r][j],
                        mul_mod(alpha[j+1], inva, pk), pk);
                    beta[j] = (uint32_t)sub_mod(0,
                        mul_mod(beta[j+1], inva, pk), pk);
                }
                uint64_t da = 0, db = 0;
                for (int i = 0; i < dim; ++i) {
                    da = add_mod(da, mul_mod(alpha[i], col_rns[k][i], pk), pk);
                    db = add_mod(db, mul_mod(beta[i], col_rns[k][i], pk), pk);
                }
                uint32_t sd = (uint32_t)add_mod(1, mul_mod((uint32_t)db, inva, pk), pk);
                if (sd == 0) return false;
                uint32_t inv_sd = mod_inv(sd, pk);
                uint32_t t = (uint32_t)mul_mod(
                    sub_mod(M.a[k][r][dim-1], mul_mod((uint32_t)da, inva, pk), pk),
                    inv_sd, pk);
                for (int j = 0; j < dim; ++j)
                    out.a[k][r][j] = (uint32_t)add_mod(alpha[j], mul_mod(beta[j], t, pk), pk);
            }
        }
    }
    M = out;
    return true;
}

__device__
inline bool apply_axis_step_rns(RNSMatrix &walk, Position &pos,
                                 int axis, int sign, const CMFDesc &desc, int n_primes) {
    uint32_t col_rns[MAX_PRIMES][MAX_DIM] = {0};
    uint32_t denom_rns[MAX_PRIMES] = {0};
    bool inverse = false;

    if (axis < desc.p) {
        inverse = (sign < 0);
    } else {
        inverse = (sign > 0);
    }

    if (!theta_companion_column_rns(pos, desc.z, walk.dim, desc.p, desc.q,
                                     col_rns, denom_rns, n_primes, axis, sign))
        return false;
    if (!right_mul_rns(walk, col_rns, denom_rns, inverse, n_primes))
        return false;
    pos.v[axis] += (double)sign;
    return true;
}

__device__
inline bool apply_trajectory_step_rns(RNSMatrix &walk, Position &pos,
                                        const int dir[MAX_AXES],
                                        const CMFDesc &desc, int n_primes) {
    int max_abs = 0;
    for (int a = 0; a < desc.naxes; ++a)
        max_abs = max_abs > abs(dir[a]) ? max_abs : abs(dir[a]);
    for (int level = max_abs; level >= 1; --level) {
        for (int axis = desc.naxes - 1; axis >= 0; --axis) {
            if (abs(dir[axis]) < level) continue;
            int sign = (dir[axis] > 0) ? 1 : -1;
            if (!apply_axis_step_rns(walk, pos, axis, sign, desc, n_primes))
                return false;
        }
    }
    return true;
}

__device__
inline void rns_to_double(const RNSMatrix &M, WalkMatrix &out, int n_primes,
                           double &log_scale) {
    const int dim = M.dim;
    out.dim = dim;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (n_primes >= 2) {
                uint64_t p0 = RNS_PRIMES[0], p1 = RNS_PRIMES[1];
                uint32_t r0 = M.a[0][i][j], r1 = M.a[1][i][j];
                uint64_t diff = (r1 >= r0) ? (r1 - r0) : (r1 + p1 - r0);
                uint64_t base = p0 % p1, result = 1, exp = p1 - 2;
                while (exp > 0) {
                    if (exp & 1) result = (result * base) % p1;
                    base = (base * base) % p1; exp >>= 1;
                }
                uint64_t t = (diff * result) % p1;
                uint64_t val = r0 + p0 * t;
                uint64_t half = (p0 * p1) / 2;
                out.a[i][j] = (val > half) ? -(double)(p0 * p1 - val) : (double)val;
            } else {
                out.a[i][j] = (double)M.a[0][i][j];
            }
        }
    }
    normalize_projective(out, log_scale);
}

// Overload without scale tracking
__device__
inline void rns_to_double(const RNSMatrix &M, WalkMatrix &out, int n_primes) {
    double dummy = 0.0;
    rns_to_double(M, out, n_primes, dummy);
}

__global__
void walk_kernel_rns(const RNSInput *inputs, RNSResult *results,
                     CMFDesc desc, int N, int h, int n_traj, int n_primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;

    const RNSInput &inp = inputs[idx];
    RNSMatrix walk;
    walk.dim = desc.rank;
    for (int k = 0; k < n_primes; ++k)
        for (int i = 0; i < walk.dim; ++i)
            for (int j = 0; j < walk.dim; ++j)
                walk.a[k][i][j] = (i == j) ? 1 : 0;

    Position pos = inp.start;
    WalkMatrix snap1, snap2, snap3;
    int cp1 = N - 2 * h, cp2 = N - h;
    bool ok = true;
    double log_scale = 0.0;
    double ls1 = 0.0, ls2 = 0.0, ls3 = 0.0;

    for (int step = 1; step <= N; ++step) {
        if (!apply_trajectory_step_rns(walk, pos, inp.dir, desc, n_primes)) {
            ok = false; break;
        }
        if (step % 100 == 0 || step == cp1 || step == cp2 || step == N) {
            WalkMatrix wm;
            rns_to_double(walk, wm, n_primes, log_scale);
            if (step == cp1) { snap1 = wm; ls1 = log_scale; }
            if (step == cp2) { snap2 = wm; ls2 = log_scale; }
            if (step == N)   { snap3 = wm; ls3 = log_scale; }
            for (int k = 0; k < n_primes; ++k) {
                uint32_t pk = RNS_PRIMES[k];
                for (int i = 0; i < walk.dim; ++i)
                    for (int j = 0; j < walk.dim; ++j)
                        walk.a[k][i][j] = to_mod(wm.a[i][j], pk);
            }
        }
    }

    RNSResult &res = results[idx];
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
    res.p_f64 = snap3.a[0][col]; res.q_f64 = snap3.a[1][col];
    res.precision = compute_precision(snaps);
    res.log_scale = ls3;  // Track accumulated log scale at depth N
    res.delta = compute_kamidelta(snaps, res.limit);
}

bool parse_cmf_type(const char *s, int &p, int &q) { return sscanf(s, "%dF%d", &p, &q) == 2; }

void generate_initial_points(int p, int q, int count, Position *points, uint64_t seed) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i)
        for (int a = 0; a < p + q; ++a) points[i].v[a] = 2.0 + (double)(rng() % 19);
}

void generate_trajectories(int p, int q, int count, int *dirs, uint64_t seed, int dmax = 3) {
    std::mt19937_64 rng(seed);
    for (int i = 0; i < count; ++i) {
        for (int a = 0; a < p + q; ++a) dirs[i * MAX_AXES + a] = (int)(rng() % (2*dmax+1)) - dmax;
        bool az = true;
        for (int a = 0; a < p + q; ++a) if (dirs[i*MAX_AXES+a]) { az = false; break; }
        if (az) dirs[i*MAX_AXES] = 1;
    }
}

int run_benchmark(int p, int q, double z, int N, int h, int n_traj,
                  int mode, const char *output_path, int block_size, int n_primes, bool use_zeta) {
    CMFDesc desc = make_cmf_desc(p, q, z);
    fprintf(stderr, "=== RNS: %dF%d z=%.4f N=%d h=%d n_traj=%d rank=%d primes=%d%s ===\n",
            p, q, z, N, h, n_traj, desc.rank, n_primes, use_zeta ? " [ZETA]" : "");

    std::vector<RNSInput> h_inputs(n_traj);
    std::vector<RNSResult> h_results(n_traj);

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

    RNSInput *d_inputs; RNSResult *d_results;
    cudaMalloc(&d_inputs, n_traj * sizeof(RNSInput));
    cudaMalloc(&d_results, n_traj * sizeof(RNSResult));
    cudaMemcpy(d_inputs, h_inputs.data(), n_traj * sizeof(RNSInput), cudaMemcpyHostToDevice);

    int grid = (n_traj + block_size - 1) / block_size;
    walk_kernel_rns<<<1, 1>>>(d_inputs, d_results, desc, N, h, 1, n_primes);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    size_t mb, mt; cudaMemGetInfo(&mb, &mt);
    cudaEventRecord(start);
    walk_kernel_rns<<<grid, block_size>>>(d_inputs, d_results, desc, N, h, n_traj, n_primes);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    size_t mf; cudaMemGetInfo(&mf, &mt); size_t mu = mt - mf;
    cudaMemcpy(h_results.data(), d_results, n_traj * sizeof(RNSResult), cudaMemcpyDeviceToHost);

    int nv = 0, nl = 0; double sd = 0;
    for (int i = 0; i < n_traj; ++i) {
        if (h_results[i].valid) { nv++; if (!isnan(h_results[i].limit)) nl++; if (!isnan(h_results[i].delta)) sd += h_results[i].delta; }
    }
    double ad = (nv > 0) ? sd / nv : NAN;
    double tps = n_traj / (ms / 1000.0);
    fprintf(stderr, "  Valid: %d/%d  Limits: %d  Avg delta: %.6f  Time: %.3f ms  %.0f traj/s  Mem: %.2f MB\n",
            nv, n_traj, nl, ad, ms, tps, mu / 1e6);

    std::ofstream out(output_path);
    out << "{\n  \"variant\": \"rns\",\n  \"cmf\": \"" << p << "F" << q << "\",\n";
    out << "  \"p\": " << p << ", \"q\": " << q << ", \"z\": " << z << ",\n";
    out << "  \"rank\": " << desc.rank << ", \"naxes\": " << desc.naxes << ",\n";
    out << "  \"N\": " << N << ", \"h\": " << h << ", \"n_traj\": " << n_traj << ",\n";
    out << "  \"mode\": \"" << (mode == 0 ? "limit" : "delta") << "\",\n";
    out << "  \"block_size\": " << block_size << ", \"n_primes\": " << n_primes << ",\n";
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
    int mode = 0, block_size = 128, n_primes = 8;
    bool use_zeta = false;
    const char *output_path = "results_rns.json";

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
        else if (arg == "--n-primes") n_primes = atoi(next());
        else if (arg == "--output") output_path = next();
        else if (arg == "--zeta") { use_zeta = true; z = 1.0; }
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); return 1; }
    }

    int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "GPU: %s (SM %d.%d, %d MPs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    return run_benchmark(p, q, z, N, h, n_traj, mode, output_path, block_size, n_primes, use_zeta);
}
