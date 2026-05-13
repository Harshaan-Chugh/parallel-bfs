#!/usr/bin/env python3
"""Generate final-report figures from committed benchmark CSVs."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"


GRAPH_LABELS = {
    "medium_chain.edgelist": "m_chain",
    "medium_dense.edgelist": "m_dense",
    "medium_sparse.edgelist": "m_sparse",
    "large_dense.edgelist": "l_dense",
    "large_powerlaw.edgelist": "l_power",
    "large_sparse.edgelist": "l_sparse",
    "large_dense": "large_dense",
    "large_powerlaw": "large_powerlaw",
    "large_sparse": "large_sparse",
    "roadNet-PA": "roadNet-PA",
    "roadNet-TX": "roadNet-TX",
    "roadNet-CA": "roadNet-CA",
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def save_gpu_head_to_head():
    rows = read_csv(ROOT / "results" / "graph-first-gpu" / "summary_20260428_042111.csv")
    by_graph = defaultdict(list)
    for row in rows:
        by_graph[row["graph"]].append(row)

    order = [
        "medium_chain.edgelist",
        "medium_dense.edgelist",
        "medium_sparse.edgelist",
        "large_dense.edgelist",
        "large_powerlaw.edgelist",
        "large_sparse.edgelist",
    ]

    labels, spmv, graphfirst, best_variants = [], [], [], []
    for graph in order:
        graph_rows = by_graph[graph]
        gf = next(r for r in graph_rows if r["implementation"] == "graph-first")
        la_rows = [r for r in graph_rows if r["implementation"] == "linear-algebraic"]
        best = min(la_rows, key=lambda r: float(r["median_ms"]))
        labels.append(GRAPH_LABELS[graph])
        spmv.append(float(best["median_ms"]))
        graphfirst.append(float(gf["median_ms"]))
        best_variants.append(best["variant"].replace("warpbitmap", "warp+bm"))

    y = range(len(labels))
    height = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh([i - height / 2 for i in y], spmv, height, label="Best SpMV", color="#1f77b4")
    ax.barh([i + height / 2 for i in y], graphfirst, height, label="Graph-first", color="#d95f02")
    for i, (x, variant) in enumerate(zip(spmv, best_variants)):
        ax.text(x * 1.05, i - height / 2, variant, va="center", fontsize=7, color="#1f77b4")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Median BFS runtime (ms, log scale)")
    ax.set_title("Single-GPU BFS: best SpMV vs graph-first")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gpu_head_to_head.pdf")
    plt.close(fig)


def load_median_table(path):
    rows = read_csv(path)
    out = defaultdict(dict)
    for r in rows:
        key = (r["graph"], r["strategy"])
        out[key][int(r["ranks"])] = float(r["median_time_s"])
    return out


def save_mpi_synthetic_speedup():
    data = load_median_table(ROOT / "results" / "mpi_raw" / "timing_large_median.csv")
    graphs = ["large_dense", "large_powerlaw", "large_sparse"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.65), sharey=True)
    for ax, graph in zip(axes, graphs):
        for strategy, marker, color in [("1D", "o", "#1f77b4"), ("2D", "s", "#d95f02")]:
            times = data[(graph, strategy)]
            ranks = sorted(times)
            base = times[1]
            speedup = [base / times[p] for p in ranks]
            ax.plot(ranks, speedup, marker=marker, color=color, label=strategy)
        ax.plot([1, 16], [1, 16], color="0.7", linestyle="--", linewidth=1, label="ideal")
        ax.set_title(GRAPH_LABELS[graph])
        ax.set_xlabel("MPI ranks")
        ax.set_xticks([1, 4, 9, 16])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Speedup vs 1 rank")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MPI BFS: Synthetic Graphs", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mpi_synthetic_speedup.pdf", bbox_inches="tight")
    plt.close(fig)


def load_snap_medians():
    rows = read_csv(ROOT / "results" / "mpi_raw" / "timing_snap.csv")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["graph"], r["strategy"], int(r["ranks"]))].append(float(r["time_s"]))

    out = defaultdict(dict)
    for (graph, strategy, ranks), times in grouped.items():
        out[(graph, strategy)][ranks] = statistics.median(times)
    return out


def save_mpi_snap_speedup():
    data = load_snap_medians()
    graphs = ["roadNet-PA", "roadNet-TX", "roadNet-CA"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.65), sharey=True)
    for ax, graph in zip(axes, graphs):
        for strategy, marker, color in [("1D", "o", "#1f77b4"), ("2D", "s", "#d95f02")]:
            times = data[(graph, strategy)]
            ranks = sorted(times)
            base = times[1]
            speedup = [base / times[p] for p in ranks]
            ax.plot(ranks, speedup, marker=marker, color=color, label=strategy)
        ax.plot([1, 64], [1, 64], color="0.7", linestyle="--", linewidth=1, label="ideal")
        ax.set_title(graph)
        ax.set_xlabel("MPI ranks")
        ax.set_xticks([1, 4, 16, 64])
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Speedup vs 1 rank")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("MPI BFS: SNAP Road Networks", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mpi_snap_speedup.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_gpu_head_to_head()
    save_mpi_synthetic_speedup()
    save_mpi_snap_speedup()
    print(f"Wrote report figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
