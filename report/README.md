# Final Report

LaTeX source and generated figures for the CS 5220 final project report.

Build from this directory:

```bash
make
```

Regenerate result figures only:

```bash
make figures
```

The figures are generated from the committed benchmark CSVs:

- `results/graph-first-gpu/summary_20260428_042111.csv`
- `results/multigpu/summary_20260513_035802.csv`
- `results/mpi_raw/timing_large_median.csv`
- `results/mpi_raw/timing_snap.csv`
