# =============================================================================
#  Makefile for the parallel-BFS poster
#  Output: poster.pdf  (42 in x 30 in landscape, matte stock)
# =============================================================================

LATEX     = pdflatex
LATEXOPTS = -interaction=nonstopmode -halt-on-error
SRC       = poster.tex
PDF       = poster.pdf
AUX_EXTS  = aux log out toc nav snm vrb fls fdb_latexmk synctex.gz

.PHONY: all clean view watch

all: $(PDF)

# Run pdflatex twice so cross-references settle.
$(PDF): $(SRC)
	$(LATEX) $(LATEXOPTS) $(SRC)
	$(LATEX) $(LATEXOPTS) $(SRC)

view: $(PDF)
	@command -v xdg-open >/dev/null 2>&1 && xdg-open $(PDF) || \
	 command -v open     >/dev/null 2>&1 && open $(PDF) || \
	 echo "Open $(PDF) manually."

# Live-recompile loop (requires latexmk).
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(SRC)

clean:
	@for ext in $(AUX_EXTS); do rm -f *.$$ext; done
	rm -f $(PDF)
