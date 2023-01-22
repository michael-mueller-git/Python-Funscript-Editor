#!/bin/sh

pyuic5 funscript_editor_view.ui -o funscript_editor_view.py
pyuic5 settings_view.ui -o settings_view.py
echo "ok"
