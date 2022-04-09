#!/bin/bash
# Description: Installer for OFS + MTFG
# Requirements: On Debian based systems e.g. Ubuntu you have to install Anaconda or Miniconda
# befor running this installer.

if command -v apt; then
    sudo apt install -y cmake build-essential libmpv-dev libglvnd-dev libxext-dev make \
        git gcc g++ cmake libmpv-dev libatlas-base-dev
fi

OFS_DIR="$HOME/.local/share/OFS/application"

if [ -d $OFS_DIR ]; then
    echo ">> OpenFunscripter Source already downloaded (Updating...)"
    pushd $OFS_DIR
    git pull
    git submodule update
else
    mkdir -p `dirname $OFS_DIR`
    echo ">> Clone OpenFunscripter Source"
    git clone https://github.com/OpenFunscripter/OFS.git $OFS_DIR
    pushd $OFS_DIR
    git submodule update --init
    pushd $OFS_DIR/lib/EASTL
    git submodule update --init
    popd
    echo ">> OpenFunscripter Source downloaded to $OFS_DIR"
fi

echo ">> Build OFS"
rm -rf build
mkdir -p build
pushd build
cmake ..
make -j$(expr $(nproc) \+ 1)
popd  # build
popd  # $OFS_DIR

echo ">> Install ofs extension"
mkdir -p ~/.local/share/OFS/OFS_data/extensions/MTFG
pushd ~/.local/share/OFS/OFS_data/extensions/MTFG

if [ ! -d ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor ]; then
    git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
fi

pushd ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor
git pull

if command -v apt; then
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda env create -f environment_ubuntu.yaml
fi

if [ -f ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor/assets/ffmpeg ]; then
    cp -fv ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor/assets/ffmpeg \
        ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor/funscript_editor/data/ffmpeg
else
    # TODO newest ffmpeg break MTFG!!
    bash download_ffmpeg.sh
fi
popd

cp -fv ~/.local/share/OFS/OFS_data/extensions/MTFG/Python-Funscript-Editor/contrib/Installer/assets/main.lua \
    ~/.local/share/OFS/OFS_data/extensions/MTFG/main.lua
popd

if [ ! -e ~/.local/bin/OpenFunscripter ]; then
    mkdir -p ~/.local/bin
    ln -s `realpath $OFS_DIR`/bin/OpenFunscripter ~/.local/bin/OpenFunscripter
fi


mkdir -p ~/.local/share/applications

cat >~/.local/share/applications/OpenFunscripter.desktop <<EOL
[Desktop Entry]
Type=Application
Name=OpenFunscripter
Exec=`realpath $OFS_DIR`/bin/OpenFunscripter
Comment=OpenFunscripter
StartupWMClass=OpenFunscripter
Icon=`realpath $OFS_DIR`/bin/data/logo64.png
EOL
