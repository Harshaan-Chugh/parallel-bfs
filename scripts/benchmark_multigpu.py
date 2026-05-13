#!/usr/bin/env python3
"""Run optional 1/2/4-GPU BFS scaling experiments and write CSV summaries."""

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_output(output: str) -> dict:
    row = {}

    graph_match = re.search(
        r"Graph:\s+(\d+)\s+vertices,\s+(\d+)\s+directed edges\s+\(avg degree:\s+([0-9.]+)\)",
        output,
    )
    if graph_match:
        row["vertices"] = int(graph_match.group(1))
        row["directed_edges"] = int(graph_match.group(2))
        row["avg_degree"] = float(graph_match.group(3))

    time_match = re.search(r"BFS time:\s+([0-9.]+)\s+ms", output)
    if time_match:
        row["time_ms"] = float(time_match.group(1))

    level_match = re.search(r"Multi-GPU levels:\s+(\d+)", output)
    if level_match:
        row["levels"] = int(level_match.group(1))

    summary_match = re.search(
        r"Source:\s+(\d+)\s+\|\s+Reachable:\s+(\d+)/(\d+)\s+\|\s+Max depth:\s+(\d+)",
        output,
    )
    if summary_match:
        row["reported_source"] = int(summary_match.group(1))
        row["reachable"] = int(summary_match.group(2))
        row["max_depth"] = int(summary_match.group(4))

    if "directed_edges" in row and "time_ms" in row and row["time_ms"] > 0:
        row["gteps"] = row["directed_edges"] / row["time_ms"] / 1_000_000.0

    return row


def graph_files(root: Path, groups: List[str], snap_names: List[str]) -> List[Path]:
    graphs = []
    test_dir = root / "test-graphs"
    for group in groups:
        graphs.extend(sorted(test_dir.glob(f"{group}_*.edgelist")))

    snap_dir = root / "snap-graphs"
    for name in snap_names:
        path = snap_dir / f"{name}.txt"
        if path.exists():
            graphs.append(path)
        else:
            print(f"WARNING: missing SNAP graph {path}", file=sys.stderr)
    return graphs


def run_one(cmd: List[str], root: Path, env: dict) -> str:
    print("$ " + " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd, completed.stdout)
    return completed.stdout


def write_summary(rows: List[dict], path: Path) -> None:
    grouped = {}
    for row in rows:
        key = (row["graph"], int(row["gpu_count"]))
        grouped.setdefault(key, []).append(float(row["time_ms"]))

    base_by_graph = {}
    for (graph, gpu_count), times in grouped.items():
        if gpu_count == 1:
            base_by_graph[graph] = statistics.median(times)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph", "gpu_count", "median_ms", "min_ms", "max_ms", "speedup_vs_1gpu"])
        for graph, gpu_count in sorted(grouped):
            times = grouped[(graph, gpu_count)]
            med = statistics.median(times)
            base = base_by_graph.get(graph)
            speedup = base / med if base and med > 0 else ""
            writer.writerow([
                graph,
                gpu_count,
                f"{med:.3f}",
                f"{min(times):.3f}",
                f"{max(times):.3f}",
                f"{speedup:.3f}" if speedup != "" else "",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark optional multi-GPU BFS.")
    parser.add_argument("--groups", nargs="+", default=["large"],
                        choices=["tiny", "medium", "large"],
                        help="Synthetic graph groups to run.")
    parser.add_argument("--snap", nargs="*", default=[],
                        help="SNAP graph basenames, e.g. roadNet-PA roadNet-TX roadNet-CA.")
    parser.add_argument("--gpu-counts", nargs="+", type=int, default=[1, 2, 4],
                        help="GPU counts to test.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeated runs per graph/GPU count.")
    parser.add_argument("--source", type=int, default=0,
                        help="BFS source vertex.")
    parser.add_argument("--srun", action="store_true",
                        help="Prefix each command with srun -n 1 -G <gpu_count>.")
    parser.add_argument("--out", default=None,
                        help="Raw CSV path. Defaults to results/multigpu/benchmark_TIMESTAMP.csv.")
    parser.add_argument("--summary-out", default=None,
                        help="Summary CSV path. Defaults next to raw CSV.")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if any(g < 1 for g in args.gpu_counts):
        parser.error("--gpu-counts must be positive")

    root = repo_root()
    binary = root / "build" / "bfs_linalg_multigpu"
    if not binary.exists():
        print(f"Missing executable: {binary}. Run make linalg-multigpu first.", file=sys.stderr)
        return 2

    graphs = graph_files(root, args.groups, args.snap)
    if not graphs:
        print("No graph files matched the request.", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else root / "results" / "multigpu" / f"benchmark_{stamp}.csv"
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_name(
        out_path.name.replace("benchmark_", "summary_")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_gpus = max(args.gpu_counts)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(max_gpus)),
    )

    rows = []
    for graph in graphs:
        for gpu_count in args.gpu_counts:
            for repeat in range(1, args.repeats + 1):
                prefix = ["srun", "-n", "1", "-G", str(gpu_count)] if args.srun else []
                cmd = prefix + [str(binary), str(graph), str(args.source), str(gpu_count)]
                output = run_one(cmd, root, env)
                row = parse_output(output)
                row.update({
                    "implementation": "linear-algebraic-multigpu",
                    "variant": "bitmap-pull-replicated-csr",
                    "graph": graph.name,
                    "source": args.source,
                    "gpu_count": gpu_count,
                    "repeat": repeat,
                    "command": " ".join(cmd),
                })
                rows.append(row)

    fieldnames = [
        "implementation", "variant", "graph", "source", "gpu_count", "repeat",
        "vertices", "directed_edges", "avg_degree", "time_ms", "gteps",
        "reachable", "max_depth", "levels", "command",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, summary_path)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
