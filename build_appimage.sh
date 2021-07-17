#!/bin/bash

if [ ! -f ./linuxdeploy-x86_64.AppImage ]; then
    wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
fi

rm -f ./funscript-editor-*.AppImage

chmod +x linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-plugin-conda.sh

ARCH=x86_64 \
	./linuxdeploy-x86_64.AppImage \
	--appdir AppDir \
	--plugin conda \
	--output appimage \
	--icon-file icon.png \
	--desktop-file funscript-editor.desktop
