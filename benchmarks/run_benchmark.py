#!/usr/bin/env python3
"""
RamanujanDreams-CUDA Benchmark Runner

Runs all benchmark configurations:
  5 CMFs (3F2, 4F3, 5F4, 6F5, 8F7)
  × 3 variants (f64, rns, df64)
  × 2 modes (limit search, delta search)
  × 10,000 trajectories each
  × N=1000, h=50

Outputs:
  - Individual JSON results per configuration
  - Aggregated comparison table (CSV + Markdown)
  - Summary report
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Benchmark configuration
CMF_TYPES = ["3F2", "4F3", "5F4", "6F5", "8F7"]
VARIANTS = ["f64", "rns", "df64"]
MODES = ["limit", "delta"]

# Known initial points for delta search (zeta-like constants)
# These correspond to classical starting points where the limit
# is a known mathematical constant
KNOWN_INITIAL_POINTS = {
    "3F2": {"start": [1, 1, 1, 1, 1], "z": 0.5, "constant": "zeta(3)-like"},
    "4F3": {"start": [1, 1, 1, 1, 1, 1, 1], "z": 0.5, "constant": "zeta(4)-like"},
    "5F4": {"start": [1, 1, 1, 1, 1, 1, 1, 1, 1], "z": 0.5, "constant": "zeta(5)-like"},
    "6F5": {"start": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "z": 0.5, "constant": "zeta(6)-like"},
    "8F7": {"start": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "z": 0.5, "constant": "zeta(8)-like"},
}

def run_benchmark(binary, cmf, z, N, h, n_traj, mode, output_path, 
                   block_size=None, n_primes=None, use_zeta=False):
    """Run a single benchmark configuration."""
    cmd = [binary, "--cmf", cmf, "--z", str(z), "--N", str(N), "--h", str(h),
           "--n-traj", str(n_traj), "--mode", mode, "--output", output_path]
    if block_size:
        cmd.extend(["--block-size", str(block_size)])
    if n_primes and "rns" in binary:
        cmd.extend(["--n-primes", str(n_primes)])
    if use_zeta:
        cmd.append("--zeta")

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None

    # Parse output JSON
    try:
        with open(output_path) as f:
            data = json.load(f)
        data["wall_time_s"] = elapsed
        return data
    except Exception as e:
        print(f"  ERROR parsing output: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="RamanujanDreams-CUDA Benchmark Runner")
    parser.add_argument("--bin-dir", type=Path, default=Path("cuda"),
                        help="Directory containing compiled binaries")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Output directory for results")
    parser.add_argument("--N", type=int, default=1000, help="Walk depth")
    parser.add_argument("--h", type=int, default=50, help="Snapshot spacing")
    parser.add_argument("--n-traj", type=int, default=10000, help="Trajectories per config")
    parser.add_argument("--z", type=float, default=0.5, help="z parameter")
    parser.add_argument("--cmfs", nargs="+", default=CMF_TYPES, help="CMF types to test")
    parser.add_argument("--variants", nargs="+", default=VARIANTS, help="Variants to test")
    parser.add_argument("--modes", nargs="+", default=MODES, help="Modes to test")
    parser.add_argument("--n-primes", type=int, default=8, help="RNS primes")
    parser.add_argument("--zeta", action="store_true",
                        help="Use known zeta initial points (1,...,1; 2,...,2) with z=1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "individual").mkdir(exist_ok=True)
    (args.output_dir / "summary").mkdir(exist_ok=True)

    all_results = []

    total_configs = len(args.cmfs) * len(args.variants) * len(args.modes)
    config_idx = 0

    for cmf in args.cmfs:
        for variant in args.variants:
            binary = args.bin_dir / f"cmf_walk_{variant}"
            if not binary.exists():
                print(f"  SKIP: {binary} not found")
                continue

            for mode in args.modes:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] {cmf} / {variant} / {mode}")

                output_file = args.output_dir / "individual" / f"{cmf}_{variant}_{mode}.json"

                result = run_benchmark(
                    str(binary), cmf, args.z, args.N, args.h,
                    args.n_traj, mode, str(output_file),
                    n_primes=args.n_primes, use_zeta=args.zeta
                )

                if result:
                    result["cmf"] = cmf
                    result["variant"] = variant
                    result["mode"] = mode
                    all_results.append(result)
                    print(f"  -> {result.get('throughput_traj_per_s', 0):.0f} traj/s, "
                          f"{result.get('gpu_time_ms', 0):.1f} ms, "
                          f"{result.get('mem_used_mb', 0):.1f} MB")

    # Write aggregated results
    summary_path = args.output_dir / "summary" / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results: {summary_path}")

    # Generate comparison table
    generate_comparison_table(all_results, args.output_dir / "summary")

    print(f"\nBenchmark complete. {len(all_results)} configurations run.")

def generate_comparison_table(results, output_dir):
    """Generate CSV and Markdown comparison tables."""
    if not results:
        return

    # CSV
    csv_path = output_dir / "benchmark_table.csv"
    with open(csv_path, "w") as f:
        f.write("cmf,variant,mode,rank,N,h,n_traj,gpu_time_ms,throughput_traj_s,"
                "mem_mb,n_valid,n_with_limits,avg_delta,wall_time_s\n")
        for r in results:
            f.write(f"{r.get('cmf','')},{r.get('variant','')},{r.get('mode','')},"
                    f"{r.get('rank',0)},{r.get('N',0)},{r.get('h',0)},"
                    f"{r.get('n_traj',0)},{r.get('gpu_time_ms',0):.3f},"
                    f"{r.get('throughput_traj_per_s',0):.1f},"
                    f"{r.get('mem_used_mb',0):.2f},"
                    f"{r.get('n_valid',0)},{r.get('n_with_limits',0)},"
                    f"{r.get('avg_delta','nan')},{r.get('wall_time_s',0):.3f}\n")
    print(f"CSV table: {csv_path}")

    # Markdown
    md_path = output_dir / "benchmark_table.md"
    with open(md_path, "w") as f:
        f.write("# RamanujanDreams-CUDA Benchmark Results\n\n")
        f.write("| CMF | Variant | Mode | Rank | GPU Time (ms) | Throughput (traj/s) | Memory (MB) | Valid | Limits | Avg Delta |\n")
        f.write("|-----|---------|------|------|---------------|---------------------|-------------|-------|--------|-----------|\n")
        for r in results:
            f.write(f"| {r.get('cmf','')} | {r.get('variant','')} | {r.get('mode','')} "
                    f"| {r.get('rank',0)} | {r.get('gpu_time_ms',0):.1f} "
                    f"| {r.get('throughput_traj_per_s',0):.0f} "
                    f"| {r.get('mem_used_mb',0):.1f} "
                    f"| {r.get('n_valid',0)}/{r.get('n_traj',0)} "
                    f"| {r.get('n_with_limits',0)} "
                    f"| {r.get('avg_delta','nan')} |\n")
    print(f"Markdown table: {md_path}")

if __name__ == "__main__":
    main()
