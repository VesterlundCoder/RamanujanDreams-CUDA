#!/bin/bash
#SBATCH --account=ehpc916
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=rd_cuda_bench

# SLURM job for RamanujanDreams-CUDA benchmark on MareNostrum5
# Runs all 5 CMFs × 3 variants × 2 modes = 30 benchmark configurations

set -e

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"

# Load CUDA
module load cuda/12.8
echo "CUDA: $(nvcc --version | tail -1)"

# Setup
WORKDIR=/gpfs/scratch/ehpc916/ramanujan_dreams_cuda
RESULTS_DIR=$WORKDIR/results

mkdir -p $WORKDIR/cuda $RESULTS_DIR/individual $RESULTS_DIR/summary

# Copy code to scratch if not already there
if [ ! -f "$WORKDIR/cuda/cmf_pfq.cuh" ]; then
    echo "Copying code to scratch..."
    cp -r /gpfs/home/inc407859/ramanujan_dreams_cuda/* $WORKDIR/
fi

cd $WORKDIR/cuda

# Build all variants
echo ""
echo "=== Building CUDA variants ==="
make clean
make all CUDA_ARCH=sm_90
echo "Build complete."

# Verify binaries
ls -la cmf_walk_f64 cmf_walk_rns cmf_walk_df64

# Quick smoke test
echo ""
echo "=== Smoke test: 3F2, N=100, 100 trajectories ==="
./cmf_walk_f64 --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/smoke_f64.json
./cmf_walk_rns --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/smoke_rns.json
./cmf_walk_df64 --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/smoke_df64.json
echo "Smoke test passed."

# Full benchmark suite - random initial points
echo ""
echo "=== Running full benchmark suite (random initial points) ==="
echo "Configurations: 5 CMFs × 3 variants × 2 modes = 30 runs"
echo "Each: 10,000 trajectories, N=1000, h=50"
echo ""

cd $WORKDIR
python3 benchmarks/run_benchmark.py \
    --bin-dir cuda \
    --output-dir results/random \
    --N 1000 \
    --h 50 \
    --n-traj 10000 \
    --z 0.5 \
    --n-primes 8

# Full benchmark suite - zeta initial points
echo ""
echo "=== Running full benchmark suite (zeta initial points) ==="
echo "Configurations: 5 CMFs × 3 variants × 2 modes = 30 runs"
echo "Each: 10,000 trajectories, N=1000, h=50, z=1, zeta initial points"
echo ""

cd $WORKDIR
python3 benchmarks/run_benchmark.py \
    --bin-dir cuda \
    --output-dir results/zeta \
    --N 1000 \
    --h 50 \
    --n-traj 10000 \
    --z 1.0 \
    --zeta \
    --n-primes 8

echo ""
echo "=== Benchmark complete: $(date) ==="
echo "Results in: $RESULTS_DIR"
ls -la $RESULTS_DIR/summary/
