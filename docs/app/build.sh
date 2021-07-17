#!/bin/sh

if [ -z "$(pip3 list | grep "mkdocs")" ] ; then
    echo "install mkdocs"
    pip3 install mkdocs
fi

if command -v mkdocs ; then
    mkdocs build
else
    python3 -m mkdocs -- build
fi

# [ -f ./site/index.html ] && command -v firefox && firefox ./site/index.html & disown
