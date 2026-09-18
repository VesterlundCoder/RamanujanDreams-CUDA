#!/usr/bin/env python3
"""Generate benchmark charts for the RamanujanDreams-CUDA paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Color scheme
COLORS = {'f64': '#2196F3', 'rns': '#FF5722', 'df64': '#4CAF50'}
CMFS = ['3F2', '4F3', '5F4', '6F5', '8F7']
RANKS_RANDOM = [3, 4, 5, 6, 8]
RANKS_ZETA = [2, 3, 4, 5, 7]

def load_results(path):
    with open(path) as f:
        return json.load(f)

def get(result, key, default=0):
    v = result.get(key, default)
    if v is None:
        return default
    return v

# Load data
random_results = load_results(os.path.join(os.path.dirname(__file__), '..', 'results', 'random', 'summary', 'all_results.json'))
zeta_results = load_results(os.path.join(os.path.dirname(__file__), '..', 'results', 'zeta', 'summary', 'all_results.json'))

# ===== Chart 1: Throughput comparison (zeta limit search) =====
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(CMFS))
width = 0.25
for i, variant in enumerate(['f64', 'df64', 'rns']):
    vals = []
    for cmf in CMFS:
        r = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r:
            vals.append(get(r[0], 'throughput_traj_per_s', 0))
        else:
            vals.append(0)
    bars = ax.bar(x + i*width, vals, width, label=variant.upper(), color=COLORS[variant], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            label = f'{val/1000:.0f}K' if val >= 1000 else f'{val:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5000, label,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Throughput (trajectories/sec)', fontsize=12)
ax.set_title('Limit Search Throughput (Zeta Initial Points, z=1)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CMFS)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'throughput_zeta_limit.png'))
plt.close()

# ===== Chart 2: Throughput comparison (random limit search) =====
fig, ax = plt.subplots(figsize=(8, 5))
for i, variant in enumerate(['f64', 'df64', 'rns']):
    vals = []
    for cmf in CMFS:
        r = [r for r in random_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r:
            vals.append(get(r[0], 'throughput_traj_per_s', 0))
        else:
            vals.append(0)
    bars = ax.bar(x + i*width, vals, width, label=variant.upper(), color=COLORS[variant], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            label = f'{val/1000:.0f}K' if val >= 1000 else f'{val:.0f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5000, label,
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Throughput (trajectories/sec)', fontsize=12)
ax.set_title('Limit Search Throughput (Random Initial Points, z=0.5)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CMFS)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'throughput_random_limit.png'))
plt.close()

# ===== Chart 3: Memory usage comparison =====
fig, ax = plt.subplots(figsize=(8, 5))
for i, variant in enumerate(['f64', 'df64', 'rns']):
    vals = []
    for cmf in CMFS:
        r = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r:
            vals.append(get(r[0], 'mem_used_mb', 0))
        else:
            vals.append(0)
    ax.bar(x + i*width, vals, width, label=variant.upper(), color=COLORS[variant], edgecolor='black', linewidth=0.5)

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Memory Usage (MB)', fontsize=12)
ax.set_title('GPU Memory Usage by Variant (Zeta, z=1)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CMFS)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'memory_usage.png'))
plt.close()

# ===== Chart 4: GPU time comparison (log scale) =====
fig, ax = plt.subplots(figsize=(8, 5))
for i, variant in enumerate(['f64', 'df64', 'rns']):
    vals = []
    for cmf in CMFS:
        r = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r:
            vals.append(get(r[0], 'gpu_time_ms', 0))
        else:
            vals.append(0)
    bars = ax.bar(x + i*width, vals, width, label=variant.upper(), color=COLORS[variant], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val * 1.1, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('GPU Kernel Time (ms, log scale)', fontsize=12)
ax.set_title('GPU Kernel Time per 10,000 Trajectories (Zeta, z=1)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CMFS)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'gpu_time_log.png'))
plt.close()

# ===== Chart 5: Validity rate comparison (random vs zeta) =====
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.35
random_vals = []
zeta_vals = []
for cmf in CMFS:
    r = [r for r in random_results if r.get('cmf') == cmf and r.get('variant') == 'f64' and r.get('mode') == 'limit']
    random_vals.append(get(r[0], 'n_valid', 0) / 10000 * 100 if r else 0)
    z = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == 'f64' and r.get('mode') == 'limit']
    zeta_vals.append(get(z[0], 'n_valid', 0) / 10000 * 100 if z else 0)

bars1 = ax.bar(x - width/2, random_vals, width, label='Random (z=0.5)', color='#FF9800', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, zeta_vals, width, label='Zeta (z=1)', color='#9C27B0', edgecolor='black', linewidth=0.5)
for bar, val in zip(bars1, random_vals):
    ax.text(bar.get_x() + bar.get_width()/2., val + 1, f'{val:.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, zeta_vals):
    ax.text(bar.get_x() + bar.get_width()/2., val + 1, f'{val:.0f}%',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Validity Rate (%)', fontsize=12)
ax.set_title('Walk Validity: Random vs Zeta Initial Points (f64)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CMFS)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
ax.set_ylim(0, 115)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'validity_comparison.png'))
plt.close()

# ===== Chart 6: Delta search throughput (zeta) =====
fig, ax = plt.subplots(figsize=(8, 5))
for i, variant in enumerate(['f64', 'df64', 'rns']):
    vals = []
    for cmf in CMFS:
        r = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'delta']
        if r:
            vals.append(get(r[0], 'throughput_traj_per_s', 0))
        else:
            vals.append(0)
    bars = ax.bar(x + i*width, vals, width, label=variant.upper(), color=COLORS[variant], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if val > 0:
            label = f'{val/1e6:.1f}M' if val >= 1e6 else (f'{val/1000:.0f}K' if val >= 1000 else f'{val:.0f}')
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05, label,
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Throughput (trajectories/sec, log scale)', fontsize=12)
ax.set_title('Delta Search Throughput (Zeta Initial Points, z=1)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(CMFS)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'throughput_delta.png'))
plt.close()

# ===== Chart 7: Speedup over RNS =====
fig, ax = plt.subplots(figsize=(8, 5))
for variant, color in [('f64', COLORS['f64']), ('df64', COLORS['df64'])]:
    speedups = []
    for cmf in CMFS:
        r_base = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == 'rns' and r.get('mode') == 'limit']
        r_var = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r_base and r_var:
            base = get(r_base[0], 'throughput_traj_per_s', 1)
            val = get(r_var[0], 'throughput_traj_per_s', 1)
            speedups.append(val / base)
        else:
            speedups.append(0)
    ax.plot(CMFS, speedups, 'o-', label=f'{variant.upper()} vs RNS', color=color, linewidth=2, markersize=8)

ax.set_xlabel('CMF Type', fontsize=12)
ax.set_ylabel('Speedup (x)', fontsize=12)
ax.set_title('Speedup of Floating-Point Variants over RNS (Zeta Limit)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'speedup_over_rns.png'))
plt.close()

# ===== Chart 8: Scaling with CMF rank =====
fig, ax = plt.subplots(figsize=(8, 5))
for variant, color in [('f64', COLORS['f64']), ('df64', COLORS['df64']), ('rns', COLORS['rns'])]:
    ranks = RANKS_ZETA
    times = []
    for cmf in CMFS:
        r = [r for r in zeta_results if r.get('cmf') == cmf and r.get('variant') == variant and r.get('mode') == 'limit']
        if r:
            times.append(get(r[0], 'gpu_time_ms', 0))
        else:
            times.append(0)
    ax.plot(ranks, times, 'o-', label=variant.upper(), color=color, linewidth=2, markersize=8)

ax.set_xlabel('CMF Rank', fontsize=12)
ax.set_ylabel('GPU Kernel Time (ms)', fontsize=12)
ax.set_title('Scaling with CMF Rank (Zeta Limit, 10K trajectories)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'scaling_rank.png'))
plt.close()

print("All charts generated successfully:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}")
