#!/bin/bash

echo "WARNING: The Full application in the AppImage only work on the build system. Only the Generator work portable for now!"

if [ ! -f ./linuxdeploy-x86_64.AppImage ]; then
    wget -c "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage"
fi

if [ ! -f ffmpeg-git-amd64-static.tar.xz ]; then
    wget -c "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz"
fi

rm -f ./funscript-editor-*.AppImage

chmod +x linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-plugin-conda.sh

echo "extract ffmpeg..."
mkdir -p tmp
tar -xf ffmpeg-git-amd64-static.tar.xz -C tmp
mv -f tmp/ffmpeg-git*/ffmpeg .
rm -rf tmp

ARCH=x86_64 \
	./linuxdeploy-x86_64.AppImage \
	--appdir AppDir \
	--plugin conda \
	--output appimage \
	--icon-file icon.png \
	--desktop-file funscript-editor.desktop
