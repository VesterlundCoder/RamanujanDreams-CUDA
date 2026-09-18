# RamanujanDreams-CUDA

High-performance CUDA implementation of the Ramanujan Machine Conservative Matrix Field (CMF) walk for pFq hypergeometric functions, benchmarked on NVIDIA H100 GPUs at MareNostrum5 (BSC).

Validated against [ramanujantools](https://github.com/RamanujanMachine/ramanujantools).

## Three Arithmetic Variants

| Variant | Precision | Speed | Memory | Use Case |
|---------|-----------|-------|--------|----------|
| **df64** | ~48-bit (double-single) | Fastest | 3.3 GB | GPU limit sweeping, approximate delta search |
| **f64** | 53-bit (native double) | Fast | 3.0 GB | Limit sweeping with full double precision |
| **RNS** | Exact (modular) | ~100x slower | 12.6 GB | Exact verification, deep walks, exact delta |

## When to Use What

### GPU Limit Sweeping (df64 or f64)

The standard Ramanujan Machine workflow sweeps many initial points to discover promising limit values. This is embarrassingly parallel and ideal for GPU.

```bash
# Sweep 10,000 initial points for 6F5, z=0.5
./cmf_walk_df64 --cmf 6F5 --z 0.5 --N 1000 --h 50 --n-traj 10000 --mode limit --output results.json
```

- **df64**: Up to 789,000 traj/s (3F2 zeta). Best throughput, ~48-bit precision.
- **f64**: Within 5-20% of df64. Full 53-bit precision if needed.

### GPU Delta Search (df64, f64, or RNS)

For a known good initial point, search over different trajectories to find the best irrationality measure (delta). Traditionally done on CPU, but GPU is now viable.

```bash
# Search 10,000 trajectories from the zeta initial point for 6F5, z=1
./cmf_walk_df64 --cmf 6F5 --zeta --N 1000 --h 50 --n-traj 10000 --mode delta --output delta.json
```

- **df64/f64**: 100,000-10,000,000 traj/s. Approximate delta values.
- **RNS**: 500-1,350,000 traj/s. Exact delta values where floating-point fails.

### Exact Verification (RNS)

Run the same configuration with RNS and compare limit values. Agreement confirms floating-point precision is sufficient.

```bash
# Verify with exact arithmetic
./cmf_walk_rns --cmf 6F5 --zeta --N 1000 --h 50 --n-traj 10000 --mode limit --output verify.json --n-primes 8
```

### CPU Delta Refinement (traditional)

For highest-precision delta computation, use [ramanujantools](https://github.com/RamanujanMachine/ramanujantools) on CPU with exact rational arithmetic. The GPU implementations provide approximate values for rapid exploration; CPU refinement can then be applied to the most promising candidates.

## Supported CMFs

| CMF | p | q | Rank (z=0.5) | Rank (z=1) | Axes | Zeta Initial Point |
|-----|---|---|------|------|------|-------------------|
| 3F2 | 3 | 2 | 3 | 2 | 5 | (1,1,1; 2,2) |
| 4F3 | 4 | 3 | 4 | 3 | 7 | (1,1,1,1; 2,2,2) |
| 5F4 | 5 | 4 | 5 | 4 | 9 | (1,1,1,1,1; 2,2,2,2) |
| 6F5 | 6 | 5 | 6 | 5 | 11 | (1,1,1,1,1,1; 2,2,2,2,2) |
| 8F7 | 8 | 7 | 8 | 7 | 15 | (1,1,1,1,1,1,1,1; 2,2,2,2,2,2,2) |

Note: When z=1 and p=q+1, the rank is reduced by 1 (the zeta case).

## Building

```bash
# Requires CUDA 12.x and SM 90 (H100)
cd cuda
make all

# Or specify a different architecture
make all CUDA_ARCH=sm_80
```

## Running

```bash
# Limit search: 10,000 unique initial points (random)
./cmf_walk_f64 --cmf 6F5 --z 0.5 --N 1000 --h 50 --n-traj 10000 --mode limit --output results.json

# Limit search: zeta initial point with z=1
./cmf_walk_f64 --cmf 6F5 --zeta --N 1000 --h 50 --n-traj 10000 --mode limit --output results.json

# Delta search: zeta initial point, 10,000 different trajectories
./cmf_walk_f64 --cmf 6F5 --zeta --N 1000 --h 50 --n-traj 10000 --mode delta --output results.json

# RNS with 8 primes (exact arithmetic)
./cmf_walk_rns --cmf 6F5 --zeta --N 1000 --h 50 --n-traj 10000 --mode limit --output results.json --n-primes 8
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--cmf` | CMF type (e.g. 6F5) | 6F5 |
| `--z` | z parameter | 0.5 |
| `--N` | Walk depth | 1000 |
| `--h` | Snapshot spacing | 50 |
| `--n-traj` | Number of trajectories | 10000 |
| `--mode` | `limit` or `delta` | limit |
| `--output` | Output JSON file | results_{variant}.json |
| `--zeta` | Use zeta initial point (sets z=1) | off |
| `--n-primes` | Number of RNS primes (RNS only) | 8 |
| `--block-size` | CUDA block size | 256 (f64/df64), 128 (rns) |

## Benchmark Suite

```bash
# Run all 30 configurations (5 CMFs x 3 variants x 2 modes) with random points
python3 benchmarks/run_benchmark.py --bin-dir cuda --output-dir results/random --N 1000 --h 50 --n-traj 10000 --z 0.5

# Run with zeta initial points
python3 benchmarks/run_benchmark.py --bin-dir cuda --output-dir results/zeta --N 1000 --h 50 --n-traj 10000 --zeta
```

## Benchmark Results (NVIDIA H100, MareNostrum5)

### Limit Search - Zeta Initial Points (z=1, 10,000 trajectories)

| CMF | Variant | Throughput (traj/s) | Memory (MB) | Valid |
|-----|---------|---------------------|-------------|-------|
| 3F2 | df64 | 789,454 | 3,299 | 10000/10000 |
| 3F2 | f64 | 502,006 | 3,062 | 10000/10000 |
| 3F2 | rns | 5,355 | 12,606 | 10000/10000 |
| 6F5 | df64 | 423,859 | 3,299 | 10000/10000 |
| 6F5 | f64 | 324,933 | 3,062 | 10000/10000 |
| 6F5 | rns | 5,006 | 12,606 | 10000/10000 |
| 8F7 | df64 | 286,630 | 3,299 | 10000/10000 |
| 8F7 | f64 | 241,066 | 3,062 | 10000/10000 |
| 8F7 | rns | 4,586 | 12,606 | 10000/10000 |

### Delta Search - Zeta Initial Points (z=1, 10,000 trajectories)

| CMF | Variant | Throughput (traj/s) | Valid | Avg Delta |
|-----|---------|---------------------|-------|-----------|
| 5F4 | df64 | 10,593,900 | 6666 | 0 |
| 5F4 | rns | 515,320 | 6666 | -1.002 |
| 8F7 | df64 | 7,845,650 | 6666 | 0 |
| 8F7 | rns | 1,354,250 | 6666 | -0.996 |

Full results in `results/` directory.

## Algorithm

The pFq CMF axis matrix is `I + C/a`, where `C` is the theta companion matrix derived from the hypergeometric differential equation:

```
D(θ) = θ · ∏ᵢ(θ + yᵢ - 1) - z · ∏ⱼ(θ + xⱼ)
```

Right-multiplying by `I + C/a` uses the companion structure for O(dim²) updates instead of O(dim³) dense multiplication.

The walk proceeds from N=0 to N=1000, with snapshots at N-2h, N-h, and N for limit/delta computation:
- **Limit**: Aitken Δ² acceleration from three snapshot depths
- **Delta**: `δ = -(1 + log|L - p/q| / log(q))` following ramanujantools `Limit.delta(L)`
- **Scale tracking**: Log-space accumulation of normalization factors recovers the true denominator magnitude

### Log-Space Scale Tracking

Projective normalization (dividing by the max entry) prevents overflow but destroys the absolute scale of q. We accumulate `log_scale += log(max_entry)` at each normalization step, then recover:

```
log(q_true) = log|q_normalized| + log_scale
```

This enables correct delta computation despite per-step rescaling, with log_scale values of 585-772 (q ~ 10^254-335) at N=1000.

## Validation

The implementation is validated against [ramanujantools](https://github.com/RamanujanMachine/ramanujantools):
- Matrix construction (differential equation, companion form, axis matrices) ✓
- Walk ordering (T(0)·T(1)·...·T(N-1)) ✓
- p/q extraction (M[0,dim-1], M[1,dim-1]) ✓
- Delta formula (Limit.delta(L) with log-space scale recovery) ✓

## Project Structure

```
RamanujanDreams-CUDA/
├── cuda/
│   ├── cmf_pfq.cuh          # Shared header: CMF construction, walk, limit/delta
│   ├── cmf_walk_f64.cu      # Variant 1: Native float64
│   ├── cmf_walk_rns.cu      # Variant 2: RNS exact arithmetic
│   ├── cmf_walk_df64.cu     # Variant 3: Double-single (df64)
│   └── Makefile
├── benchmarks/
│   └── run_benchmark.py     # Benchmark runner
├── slurm/
│   └── submit_benchmark.sbatch  # SLURM job script for HPC
├── paper/
│   └── benchmark_paper.tex  # Benchmark paper / preprint
├── results/
│   ├── random/              # Random initial point results
│   └── zeta/                # Zeta initial point results
└── README.md
```

## License

MIT

## References

- [RamanujanMachine/ramanujantools](https://github.com/RamanujanMachine/ramanujantools) - Reference Python implementation
- [VesterlundCoder/applesilicon_rns_benchmark](https://github.com/VesterlundCoder/applesilicon_rns_benchmark) - Apple Silicon RNS benchmark (CPU baseline)
- [EuroHPC JU](https://eurohpc-ju.europa.eu/) - Computing resources on MareNostrum5 (BSC)

## Acknowledgments

This work was supported by EuroHPC JU computing resources on MareNostrum5 at Barcelona Supercomputing Center (BSC), under project ehpc916.
