# Makefile for RamanujanDreams-CUDA
# Build all three CUDA variants for pFq CMF walk benchmarks
#
# Usage:
#   make all          # Build all variants
#   make f64          # Build only float64 variant
#   make rns          # Build only RNS variant
#   make df64         # Build only df64 variant
#   make clean        # Remove binaries
#   make test         # Quick smoke test

CUDA_ARCH ?= sm_90
NVCC ?= nvcc
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -std=c++17 -Xcompiler -O3 -Xcompiler -fopenmp

BINS = cmf_walk_f64 cmf_walk_rns cmf_walk_df64

.PHONY: all clean test

all: $(BINS)

cmf_walk_f64: cmf_walk_f64.cu cmf_pfq.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ cmf_walk_f64.cu

cmf_walk_rns: cmf_walk_rns.cu cmf_pfq.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ cmf_walk_rns.cu

cmf_walk_df64: cmf_walk_df64.cu cmf_pfq.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ cmf_walk_df64.cu

test: $(BINS)
	@echo "=== Smoke test: 3F2, N=100, 100 trajectories ==="
	./cmf_walk_f64 --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/test_f64.json
	./cmf_walk_rns --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/test_rns.json
	./cmf_walk_df64 --cmf 3F2 --z 0.5 --N 100 --h 10 --n-traj 100 --mode limit --output /tmp/test_df64.json
	@echo "=== Smoke test passed ==="

clean:
	rm -f $(BINS)
