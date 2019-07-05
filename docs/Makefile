# depends on pandoc (>= 2.0) and pdf2svg

all: model.html

model.html: model.md style.css
	pandoc --lua-filter tikz.lua --mathjax -s -o "$@" "$<" -c "$(word 2,$^)"
