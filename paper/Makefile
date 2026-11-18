LATEXMK ?= latexmk
BUILD_DIR := build
OUT_PDF := ../docs/vpxc-slgf-specification.pdf

.PHONY: all pdf clean

all: pdf

pdf:
	@mkdir -p $(BUILD_DIR) ../docs
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD_DIR) main.tex
	cp $(BUILD_DIR)/main.pdf $(OUT_PDF)

clean:
	rm -rf $(BUILD_DIR) *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.synctex.gz
