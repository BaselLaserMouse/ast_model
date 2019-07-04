all: model.html

model.html: model.md
	pandoc --lua-filter tikz.lua --mathjax -s -o "$@" "$<"
