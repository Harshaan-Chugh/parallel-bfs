PDF = final_report.pdf
SRC = final_report.tex

.PHONY: all clean distclean view figures

all: figures $(PDF)

figures:
	cd .. && python3 scripts/plot_report_results.py

$(PDF): $(SRC) figures/gpu_head_to_head.pdf figures/mpi_synthetic_speedup.pdf figures/mpi_snap_speedup.pdf
	pdflatex -interaction=nonstopmode -halt-on-error $(SRC)
	pdflatex -interaction=nonstopmode -halt-on-error $(SRC)

view: $(PDF)
	open $(PDF)

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls comment.cut

distclean: clean
	rm -f $(PDF)
