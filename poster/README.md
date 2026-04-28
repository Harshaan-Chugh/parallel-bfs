# Parallel-BFS Poster

LaTeX source for the conference poster comparing **linear-algebraic SpMV BFS**
and **graph-first direction-optimizing BFS** on a single NVIDIA A100.

## Format

- **Stock:** Matte
- **Dimensions:** **42 in (W) × 36 in (H)** — landscape (the standard
  large-format research-poster size for "42-inch wide" boards)
- **Layout (column widths):**
  - **25%** Introduction (left)
  - **50%** Methods + Results (middle, with two sub-columns)
  - **25%** Further Study + References (right)
- **Body font:** 17pt (set in `\documentclass[..., 17pt]{tikzposter}`)
- **Engine:** `pdflatex` (also works under `lualatex` / `xelatex`)
- **Class:** [`tikzposter`](https://ctan.org/pkg/tikzposter) v2.x

If you'd prefer a different height (e.g. 30" or 48"), change
`paperheight` in the `\geometry{...}` line near the top of `poster.tex`.
Because tikzposter v2 only accepts `a0/a1/a2` as documentclass paper
options, custom sizes are set via `\geometry{paperwidth=..., paperheight=...}`
after the class is loaded.

## Build

```bash
make            # pdflatex twice -> poster.pdf
make view       # open the PDF
make watch      # live-recompile via latexmk
make clean      # remove aux + pdf
```

Or by hand:

```bash
pdflatex poster.tex
pdflatex poster.tex      # second pass for cross-refs
```

The PDF is rendered at the true physical dimensions (42 in × 30 in), so the
print shop should select **"actual size / no scaling"** when printing.

## What's Included

The poster is set up as a checkpoint poster for the completed single-GPU
implementation and the measured Perlmutter benchmark results.

| Section | Status |
|---|---|
| Introduction (motivation, paradigms, research questions, contributions) | done |
| Common experimental setup + test-graph table | done |
| Linear-algebraic methods (5 kernel variants + fused snippet) | done |
| Results table (best SpMV variant vs. graph-first, from `results/summary_20260428_042111.csv`) | done |
| Graph-first methods (top-down/bottom-up + alpha/beta heuristic + flow diagram) | done |
| Further-study column (multi-GPU, real-world graphs, GraphBLAS extensions, energy) | done |
| References | done |
| Repository URL | done |
| Author and affiliation strings | done |

## Editing checklist before printing

1. Re-run the benchmark only if code or inputs change:
   `python3 scripts/benchmark.py --repeats 10 --groups medium large`.
2. Update block **8. Results** from `results/summary_20260428_042111.csv`.
3. Re-run `make` and proof-read at 100% zoom in your PDF viewer.

## Dependencies

A reasonably modern TeX Live (2020+) installation is sufficient. Required
packages, all standard:

- `tikzposter`
- `tikz` (with `positioning`, `arrows.meta`, `shapes.geometric`,
  `fit`, `backgrounds` libraries)
- `amsmath`, `amssymb`
- `booktabs`, `array`
- `xcolor`
- `listings`
- `graphicx`
- `lmodern`

On NERSC, `module load texlive` (if available) or run via Overleaf for quick
iteration.
