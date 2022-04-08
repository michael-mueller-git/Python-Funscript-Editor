#!/bin/bash

if command -v apt; then
    sudo apt install -y cmake build-essential libmpv-dev libglvnd-dev libxext-dev make git gcc g++ cmake libmpv-dev libatlas-base-dev
fi

if [ -d ./OFS ]; then
    echo "OpenFunscripter Source already downloaded"
    pushd OFS
    git pull
    git submodule update
else
    git clone https://github.com/OpenFunscripter/OFS.git
    pushd OFS
    git submodule update --init
    pushd lib/EASTL
    git submodule update --init
    popd
    echo "OpenFunscripter Source downloaded to ./OFS"
fi

echo "build OFS"
rm -rf build
mkdir -p build
pushd build
cmake ..
make -j$(expr $(nproc) \+ 1)
popd
popd

echo "install ofs extension"
mkdir -p ~/.local/share/OFS/OFS_data/extensions/MTFG
pushd ~/.local/share/OFS/OFS_data/extensions/MTFG
if [ -d ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor ]; then
    git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
fi

pushd ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor
git pull
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda env create -f environment_ubuntu.yaml
bash download_ffmpeg.sh
popd

cp -fv ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor/contrib/Installer/assets/main.lua ~/.local/share/OFS/OFS_data/extensions/MTFG/main.lua

popd
