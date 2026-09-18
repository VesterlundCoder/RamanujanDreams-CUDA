// cmf_walk_f64.cu
//
// Variant 1: CUDA native float64 pFq CMF walk
//
// Each thread processes one trajectory independently.
// Uses native double precision (excellent on H100: 1/2 rate FP64).
// Projective normalization after each step prevents overflow.
//
// Build:
//   nvcc -O3 -arch=sm_90 -std=c++17 -o cmf_walk_f64 cmf_walk_f64.cu
//
// Usage:
//   ./cmf_walk_f64 --cmf 6F5 --z 0.5 --N 1000 --h 50 --n-traj 10000 \
//       --mode limit --output results_f64.json
//   ./cmf_walk_f64 --cmf 3F2 --z 0.5 --N 1000 --h 50 --n-traj 10000 \
//       --mode delta --output results_f64_delta.json

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <cuda_runtime.h>

#include "cmf_pfq.cuh"

// Maximum results per trajectory
#define MAX_LIMITS 256

// Trajectory input/output structures
struct TrajectoryInput {
    int dir[MAX_AXES];
    Position start;
};

struct TrajectoryResult {
    double limit;       // best limit estimate
    double delta;        // kamidelta (irrationality measure)
    double p;            // normalized p = M[0, dim-1]
    double q;            // normalized q = M[1, dim-1]
    double precision;    // -log10(|L_N - L_{N-h}|) convergence digits
    double log_scale;    // accumulated log normalization scale at depth N
    int valid;
    int n_limits;
    double elapsed_ms;
};

// Parse CMF type string
bool parse_cmf_type(const char *s, int &p, int &q) {
    if (sscanf(s, "%dF%d", &p, &q) == 2) return true;
    return false;
}

// Generate deterministic unique initial points for limit search
// Uses a hash-based approach to generate sym-unique starting positions
void generate_initial_points(int p, int q, int count, Position *points, uint64_t seed) {
    std::mt19937_64 rng(seed);
    int naxes = p + q;
    for (int i = 0; i < count; ++i) {
        for (int a = 0; a < naxes; ++a) {
            // Generate values in range [2, 20] for x-axes, [3, 20] for y-axes
            int val = 2 + (int)(rng() % 19);
            points[i].v[a] = (double)val;
        }
    }
}

// Generate deterministic trajectories for delta search
// Same initial point, different directions
void generate_trajectories(int p, int q, int count, int *dirs, uint64_t seed,
                            int dmax = 3) {
    std::mt19937_64 rng(seed);
    int naxes = p + q;
    for (int i = 0; i < count; ++i) {
        for (int a = 0; a < naxes; ++a) {
            // Direction in range [-dmax, dmax], excluding 0 for at least one axis
            dirs[i * MAX_AXES + a] = (int)(rng() % (2 * dmax + 1)) - dmax;
        }
        // Ensure at least one nonzero direction
        bool all_zero = true;
        for (int a = 0; a < naxes; ++a) {
            if (dirs[i * MAX_AXES + a] != 0) { all_zero = false; break; }
        }
        if (all_zero) dirs[i * MAX_AXES] = 1;
    }
}

// CUDA kernel: walk all trajectories in parallel
__global__
void walk_kernel_f64(const TrajectoryInput *inputs, TrajectoryResult *results,
                     CMFDesc desc, int N, int h, int n_traj) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_traj) return;

    // Start timer
    // (We measure on host side for simplicity; kernel timing via events)

    WalkSnapshots snaps;
    const TrajectoryInput &inp = inputs[idx];

    bool ok = forward_walk(inp.dir, inp.start, desc, N, h, snaps);

    TrajectoryResult &res = results[idx];
    res.valid = ok ? 1 : 0;

    if (!ok) {
        res.limit = NAN;
        res.delta = NAN;
        res.p = NAN;
        res.q = NAN;
        res.precision = NAN;
        res.log_scale = NAN;
        res.n_limits = 0;
        return;
    }

    // Compute limits
    LimitResult limits[MAX_LIMITS];
    int n_lim = 0;
    compute_limits(snaps, limits, n_lim, MAX_LIMITS);
    res.n_limits = n_lim;

    // Use the first Aitken limit if available, else first Lyapunov, else raw
    res.limit = NAN;
    for (int i = 0; i < n_lim; ++i) {
        if (limits[i].method == 1) { res.limit = limits[i].L; break; }
    }
    if (isnan(res.limit)) {
        for (int i = 0; i < n_lim; ++i) {
            if (limits[i].method == 2) { res.limit = limits[i].L; break; }
        }
    }
    if (isnan(res.limit) && n_lim > 0) {
        res.limit = limits[0].L;
    }

    // Compute p/q from default projection (ramanujantools: M[0,dim-1], M[1,dim-1])
    const int dim = desc.rank;
    const int col = dim - 1;
    res.p = snaps.m3.a[0][col];
    res.q = snaps.m3.a[1][col];
    res.log_scale = snaps.log_scale3;

    // Compute precision (convergence digits)
    res.precision = compute_precision(snaps);

    // Compute kamidelta using recovered q scale
    res.delta = compute_kamidelta(snaps, res.limit);
}

// Benchmark runner
int run_benchmark(int p, int q, double z, int N, int h, int n_traj,
                  int mode, const char *output_path, int block_size, bool use_zeta) {
    CMFDesc desc = make_cmf_desc(p, q, z);
    int naxes = p + q;

    fprintf(stderr, "=== f64 Benchmark: %dF%d z=%.4f N=%d h=%d n_traj=%d rank=%d%s ===\n",
            p, q, z, N, h, n_traj, desc.rank, use_zeta ? " [ZETA]" : "");

    // Generate test data
    std::vector<TrajectoryInput> h_inputs(n_traj);
    std::vector<TrajectoryResult> h_results(n_traj);

    if (mode == 0) {
        // Limit search: unique initial points, same trajectory
        if (use_zeta) {
            // Zeta mode: use the known zeta initial point
            Position zeta_start = zeta_initial_point(p, q);
            for (int i = 0; i < n_traj; ++i) {
                h_inputs[i].start = zeta_start;
            }
        } else {
            generate_initial_points(p, q, n_traj, (Position*)h_inputs.data(), 42);
        }
        // Use a fixed canonical trajectory: (1,0,...,0)
        for (int i = 0; i < n_traj; ++i) {
            memset(h_inputs[i].dir, 0, sizeof(h_inputs[i].dir));
            h_inputs[i].dir[0] = 1;
        }
    } else {
        // Delta search: same initial point, different trajectories
        Position fixed_start;
        if (use_zeta) {
            fixed_start = zeta_initial_point(p, q);
        } else {
            for (int a = 0; a < naxes; ++a) {
                fixed_start.v[a] = (a < p) ? 2.0 : 3.0;
            }
        }
        for (int i = 0; i < n_traj; ++i) {
            h_inputs[i].start = fixed_start;
        }
        generate_trajectories(p, q, n_traj, (int*)h_inputs.data(), 42);
    }

    // Allocate device memory
    TrajectoryInput *d_inputs;
    TrajectoryResult *d_results;
    size_t input_bytes = n_traj * sizeof(TrajectoryInput);
    size_t result_bytes = n_traj * sizeof(TrajectoryResult);

    cudaMalloc(&d_inputs, input_bytes);
    cudaMalloc(&d_results, result_bytes);
    cudaMemcpy(d_inputs, h_inputs.data(), input_bytes, cudaMemcpyHostToDevice);

    // Run kernel with CUDA event timing
    int grid_size = (n_traj + block_size - 1) / block_size;

    // Warmup
    walk_kernel_f64<<<1, 1>>>(d_inputs, d_results, desc, N, h, 1);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Memory before
    size_t mem_before, mem_after;
    cudaMemGetInfo(&mem_before, &mem_after);

    cudaEventRecord(start);
    walk_kernel_f64<<<grid_size, block_size>>>(d_inputs, d_results, desc, N, h, n_traj);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Memory after
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    size_t mem_used = mem_total - mem_free;

    cudaMemcpy(h_results.data(), d_results, result_bytes, cudaMemcpyDeviceToHost);

    // Compute statistics
    int n_valid = 0, n_with_limits = 0;
    double sum_delta = 0;
    for (int i = 0; i < n_traj; ++i) {
        if (h_results[i].valid) {
            n_valid++;
            if (!isnan(h_results[i].limit)) n_with_limits++;
            if (!isnan(h_results[i].delta)) sum_delta += h_results[i].delta;
        }
    }
    double avg_delta = (n_valid > 0) ? sum_delta / n_valid : NAN;

    // Throughput
    double traj_per_s = n_traj / (ms / 1000.0);

    fprintf(stderr, "  Valid: %d/%d  With limits: %d  Avg delta: %.6f\n",
            n_valid, n_traj, n_with_limits, avg_delta);
    fprintf(stderr, "  GPU time: %.3f ms  Throughput: %.0f traj/s  Mem used: %.2f MB\n",
            ms, traj_per_s, mem_used / 1e6);

    // Write results
    std::ofstream out(output_path);
    out << "{\n";
    out << "  \"variant\": \"f64\",\n";
    out << "  \"cmf\": \"" << p << "F" << q << "\",\n";
    out << "  \"p\": " << p << ", \"q\": " << q << ",\n";
    out << "  \"z\": " << z << ",\n";
    out << "  \"rank\": " << desc.rank << ",\n";
    out << "  \"naxes\": " << desc.naxes << ",\n";
    out << "  \"N\": " << N << ",\n";
    out << "  \"h\": " << h << ",\n";
    out << "  \"n_traj\": " << n_traj << ",\n";
    out << "  \"mode\": \"" << (mode == 0 ? "limit" : "delta") << "\",\n";
    out << "  \"block_size\": " << block_size << ",\n";
    out << "  \"gpu_time_ms\": " << ms << ",\n";
    out << "  \"throughput_traj_per_s\": " << traj_per_s << ",\n";
    out << "  \"mem_used_mb\": " << mem_used / 1e6 << ",\n";
    out << "  \"n_valid\": " << n_valid << ",\n";
    out << "  \"n_with_limits\": " << n_with_limits << ",\n";
    out << "  \"avg_delta\": " << (isfinite(avg_delta) ? std::to_string(avg_delta) : "null") << ",\n";
    out << "  \"results\": [\n";
    for (int i = 0; i < std::min(n_traj, 100); ++i) {
        char lb[32], db[32], pb[32], qb[32], prb[32], lsb[32];
        json_double(lb, sizeof(lb), h_results[i].limit);
        json_double(db, sizeof(db), h_results[i].delta);
        json_double(pb, sizeof(pb), h_results[i].p);
        json_double(qb, sizeof(qb), h_results[i].q);
        json_double(prb, sizeof(prb), h_results[i].precision);
        json_double(lsb, sizeof(lsb), h_results[i].log_scale);
        out << "    {\"idx\": " << i
            << ", \"valid\": " << h_results[i].valid
            << ", \"limit\": " << lb
            << ", \"delta\": " << db
            << ", \"p\": " << pb
            << ", \"q\": " << qb
            << ", \"precision\": " << prb
            << ", \"log_scale\": " << lsb
            << ", \"n_limits\": " << h_results[i].n_limits << "}";
        if (i < std::min(n_traj, 100) - 1) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    out.close();

    fprintf(stderr, "  Results written to: %s\n", output_path);

    // Cleanup
    cudaFree(d_inputs);
    cudaFree(d_results);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

int main(int argc, char **argv) {
    int p = 6, q = 5;
    double z = 0.5;
    int N = 1000, h = 50;
    int n_traj = 10000;
    int mode = 0;  // 0=limit, 1=delta
    int block_size = 256;
    bool use_zeta = false;
    const char *output_path = "results_f64.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (++i >= argc) { fprintf(stderr, "Missing value after %s\n", argv[i-1]); exit(1); }
            return argv[i];
        };
        if (arg == "--cmf" && i + 1 < argc) {
            if (!parse_cmf_type(argv[++i], p, q)) {
                fprintf(stderr, "Invalid CMF type: %s (use e.g. 6F5)\n", argv[i]);
                return 1;
            }
        } else if (arg == "--z") z = atof(next());
        else if (arg == "--N") N = atoi(next());
        else if (arg == "--h") h = atoi(next());
        else if (arg == "--n-traj") n_traj = atoi(next());
        else if (arg == "--mode") {
            const char *m = next();
            mode = (strcmp(m, "delta") == 0) ? 1 : 0;
        }
        else if (arg == "--block-size") block_size = atoi(next());
        else if (arg == "--output") output_path = next();
        else if (arg == "--zeta") {
            use_zeta = true;
            z = 1.0;  // Zeta initial points use z=1
        }
        else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            return 1;
        }
    }

    // Print GPU info
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    fprintf(stderr, "GPU: %s (SM %d.%d, %d MPs, %d threads/MP, %.0f MHz)\n",
            prop.name, prop.major, prop.minor, prop.multiProcessorCount,
            prop.maxThreadsPerMultiProcessor, prop.clockRate / 1e3);
    fprintf(stderr, "FP64: %s, %.1f GB global mem\n",
            prop.computeMode == cudaComputeModeDefault ? "supported" : "exclusive",
            prop.totalGlobalMem / 1e9);

    return run_benchmark(p, q, z, N, h, n_traj, mode, output_path, block_size, use_zeta);
}
