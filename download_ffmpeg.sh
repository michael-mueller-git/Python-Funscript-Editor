#!/bin/bash

echo "[INFO] Download ffmpeg..."
rm -f /tmp/ffmpeg-git-amd64-static.tar.xz
wget --show-progress -O /tmp/ffmpeg-git-amd64-static.tar.xz "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz"

echo "[INFO] Extract ffmpeg..."
tmpDir=$(mktemp -d)
tar -xf /tmp/ffmpeg-git-amd64-static.tar.xz -C ${tmpDir}
mv -fv ${tmpDir}/ffmpeg-git*/ffmpeg ./funscript_editor/data
if [ "$?" = "0" ]; then
    echo "[INFO] FFmpeg is now successfully setup"
fi
rm -rf ${tmpDir}
rm -f /tmp/ffmpeg-git-amd64-static.tar.xz
