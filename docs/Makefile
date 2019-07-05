# depends on pandoc (>= 2.0) with pandoc-citeproc and pdf2svg

all: model.html

model.html: model.md style.css references.bib
	pandoc --filter pandoc-citeproc --lua-filter tikz.lua --mathjax -s -o "$@" "$<"
