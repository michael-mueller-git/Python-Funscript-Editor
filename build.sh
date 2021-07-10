#!/bin/bash

rm -rf build
rm -rf dist/funscript-editor
rm -f dist/funscript-editor*.tar.gz
rm -f funscript-editor.spec
make docs

pyinstaller --add-data="funscript_editor/config/*:funscript_editor/config/" --add-data="assets/*:./" --hidden-import=PyQt5.sip --hidden-import=sip --hidden-import "pynput.keyboard._xorg" --hidden-import "pynput.mouse._xorg" funscript-editor.py

python_major_version="$(python3 --version | sed 's/\.*[0-9]*$//g' | awk '{print $2}' | tr -d '\n')"
echo -e "\n >> Run post build for python ${python_major_version}"

if [ -d $HOME/.local/lib/python${python_major_version}/site-packages/PyQt5 ]; then
    cp -fv $HOME/.local/lib/python${python_major_version}/site-packages/PyQt5/sip.cpython-*.so ~/Repos/public/Python-Funscript-Editor/dist/funscript-editor/PyQt5
else
    echo "WARNING: PyQtt.sip not found"
fi
echo "done"

echo -n "Get Version ... "
tag="$(git tag | sed 's/* //g' | sort | tail -n 1)"
echo "$tag"

echo "Add Version to buld files"
echo "$tag" > dist/funscript-editor/funscript_editor/VERSION.txt

echo "Copy Application Documentation"
mkdir -p dist/funscript-editor/funscript_editor/docs/
cp -rf docs/app/site/ dist/funscript-editor/funscript_editor/docs/

echo "Create Archive"
tar -zcvf dist/funscript-editor-${tag}.tar.gz dist/funscript-editor
