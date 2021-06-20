#!/bin/sh

mkdocs build

# [ -f ./site/index.html ] && command -v firefox && firefox ./site/index.html & disown
