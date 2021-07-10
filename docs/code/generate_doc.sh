#!/bin/env bash

NAME="funscript_editor"

IGNORE=( )

if ! command -v sphinx-apidoc ; then
    if command -v pacman ; then
        # Arch based
        sudo pacman -Sy python3-sphinx python-sphinx_rtd_theme
    elif command -v apt ; then
        # Debian based
        sudo apt update
        sudo apt install python3-sphinx
        pip3 install sphinx-rtd-theme
    else
        pip3 install sphinx sphinx-rtd-theme
    fi
fi

if [ -z "$(pip3 list | grep "sphinx-rtd-theme")" ] ; then
    pip3 install sphinx-rtd-theme
fi

WDIR=$PWD

if [ -f index.rst ]; then
    cd ../..
fi

closeHook() {
    cd "$WDIR"
}
trap closeHook SIGHUP SIGINT SIGTERM EXIT

[ -d python_env ] && source python_env/bin/activate
echo -n 'python: ' && which python

# generate doc
rm -vf docs/code/modules.rst
rm -vf docs/code/$NAME*.rst
sphinx-apidoc --implicit-namespaces -M -f -d 3 -o docs/code $NAME ${IGNORE[@]}
cd docs/code && make clean html && cd -

# view doc
# [ -f docs/code/_build/html/index.html ] && \
#     command -v firefox && \
#     firefox docs/code/_build/html/index.html & disown
