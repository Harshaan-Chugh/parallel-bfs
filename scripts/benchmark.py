#!/usr/bin/env python3
"""
Run BFS benchmarks and write a CSV summary.

Examples:
    python3 scripts/benchmark.py --repeats 10 --groups medium large
    python3 scripts/benchmark.py --repeats 3 --only graphfirst --srun
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List


LINALG_VARIANTS = ["baseline", "warp", "bitmap", "pushpull", "warpbitmap"]


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

    summary_match = re.search(
        r"Source:\s+(\d+)\s+\|\s+Reachable:\s+(\d+)/(\d+)\s+\|\s+Max depth:\s+(\d+)",
        output,
    )
    if summary_match:
        row["reported_source"] = int(summary_match.group(1))
        row["reachable"] = int(summary_match.group(2))
        row["max_depth"] = int(summary_match.group(4))

    graphfirst_match = re.search(
        r"Graph-first max level:\s+(\d+)\s+\|\s+top-down launches:\s+(\d+)\s+\|\s+bottom-up launches:\s+(\d+)",
        output,
    )
    if graphfirst_match:
        row["graphfirst_max_level"] = int(graphfirst_match.group(1))
        row["top_down_launches"] = int(graphfirst_match.group(2))
        row["bottom_up_launches"] = int(graphfirst_match.group(3))

    if "directed_edges" in row and "time_ms" in row and row["time_ms"] > 0:
        row["gteps"] = row["directed_edges"] / row["time_ms"] / 1_000_000.0

    return row


def graph_files(root: Path, groups: List[str]) -> List[Path]:
    test_dir = root / "test-graphs"
    graphs = []
    for group in groups:
        graphs.extend(sorted(test_dir.glob(f"{group}_*.edgelist")))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BFS implementations.")
    parser.add_argument("--groups", nargs="+", default=["medium", "large"],
                        choices=["tiny", "medium", "large"],
                        help="Graph size groups to run.")
    parser.add_argument("--repeats", type=int, default=10,
                        help="Number of repeated runs per implementation/graph.")
    parser.add_argument("--source", type=int, default=0,
                        help="BFS source vertex.")
    parser.add_argument("--only", choices=["all", "linalg", "graphfirst"], default="all",
                        help="Restrict which implementation family to run.")
    parser.add_argument("--srun", action="store_true",
                        help="Prefix each benchmark command with srun -n 1 -G 1.")
    parser.add_argument("--out", default=None,
                        help="CSV output path. Defaults to results/benchmark_TIMESTAMP.csv.")
    args = parser.parse_args()

    root = repo_root()
    build_dir = root / "build"
    linalg = build_dir / "bfs_linalg"
    graphfirst = build_dir / "bfs_graphfirst"

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.only in ("all", "linalg") and not linalg.exists():
        print(f"Missing executable: {linalg}. Run make first.", file=sys.stderr)
        return 2
    if args.only in ("all", "graphfirst") and not graphfirst.exists():
        print(f"Missing executable: {graphfirst}. Run make first.", file=sys.stderr)
        return 2

    graphs = graph_files(root, args.groups)
    if not graphs:
        print("No graph files matched the requested groups.", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else root / "results" / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    prefix = ["srun", "-n", "1", "-G", "1"] if args.srun else []

    rows = []
    for graph in graphs:
        for repeat in range(1, args.repeats + 1):
            if args.only in ("all", "linalg"):
                for variant in LINALG_VARIANTS:
                    cmd = prefix + [str(linalg), str(graph), str(args.source), variant]
                    output = run_one(cmd, root, env)
                    row = parse_output(output)
                    row.update({
                        "implementation": "linear-algebraic",
                        "variant": variant,
                        "graph": graph.name,
                        "source": args.source,
                        "repeat": repeat,
                        "command": " ".join(cmd),
                    })
                    rows.append(row)

            if args.only in ("all", "graphfirst"):
                cmd = prefix + [str(graphfirst), str(graph), str(args.source)]
                output = run_one(cmd, root, env)
                row = parse_output(output)
                row.update({
                    "implementation": "graph-first",
                    "variant": "direction-optimizing",
                    "graph": graph.name,
                    "source": args.source,
                    "repeat": repeat,
                    "command": " ".join(cmd),
                })
                rows.append(row)

    fieldnames = [
        "implementation", "variant", "graph", "source", "repeat",
        "vertices", "directed_edges", "avg_degree", "time_ms", "gteps",
        "reachable", "max_depth", "graphfirst_max_level",
        "top_down_launches", "bottom_up_launches", "command",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
