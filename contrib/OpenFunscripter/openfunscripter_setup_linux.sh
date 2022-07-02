#!/bin/bash
# Description: Installer for OFS + MTFG
# Requirements: On Debian based systems e.g. Ubuntu you have to install Anaconda or Miniconda
# befor running this installer!

arg1="$1"

if [ "$EUID" -eq 0 ]; then
    echo "ERROR: You can not run this script with sudo!!"
    exit 1
fi

echo "install required packages"
if command -v apt; then
    # debian based distro:

    if [ ! -d ~/anaconda3 ] && [ ! -d ~/miniconda3 ]; then
        echo "ERROR: miniconda is not properly installed. Please first install [miniconda](https://docs.conda.io/en/latest/miniconda.html)"
        exit 1
    fi

    sudo apt install -y cmake build-essential libmpv-dev libglvnd-dev libxext-dev make \
        git gcc g++ cmake libmpv-dev libatlas-base-dev
fi

if command -v pacman; then
    # arch based distro:
    sudo pacman -Syu --needed --noconfirm
    sudo pacman -Sy --needed --noconfirm python-opencv python-pyqt5 git base-devel python python-pip mpv cmake
fi

OFS_APP_DIR="$HOME/.local/share/OFS/application"
OFS_EXTENSION_DIR="$HOME/.local/share/OFS/OFS2_data/extensions"

if [ -d $OFS_APP_DIR ]; then
    echo ">> OpenFunscripter Source already downloaded (Updating...)"
    pushd $OFS_APP_DIR
    git pull
    git submodule update --init
    pushd $OFS_APP_DIR/lib/EASTL
    git submodule update --init
    popd
else
    mkdir -p `dirname $OFS_APP_DIR`
    echo ">> Clone OpenFunscripter Source"
    git clone https://github.com/OpenFunscripter/OFS.git $OFS_APP_DIR
    pushd $OFS_APP_DIR
    git submodule update --init
    pushd $OFS_APP_DIR/lib/EASTL
    git submodule update --init
    popd
    echo ">> OpenFunscripter Source downloaded to $OFS_APP_DIR"
fi

if [ "$arg1" != "--latest" ]; then
    echo "Checkout latest OpenFunscripter release"
    git checkout $(git describe --tags `git rev-list --tags --max-count=1`)
    # git checkout 1.4.4
    git submodule update --init
    pushd $OFS_APP_DIR/lib/EASTL
    git submodule update --init
    popd
else
    echo "Use latest git commit (only for developers!)"
fi

echo ">> Build OFS"
rm -rf build
mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(expr $(nproc) \+ 1)
popd  # build
popd  # $OFS_APP_DIR

echo ">> Install ofs extension"
mkdir -p "$OFS_EXTENSION_DIR/MTFG"
pushd "$OFS_EXTENSION_DIR/MTFG"

if [ ! -d "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/.git" ]; then
    git clone https://github.com/michael-mueller-git/Python-Funscript-Editor.git
fi

pushd $OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor

echo "Update MTFG"
git reset --hard HEAD
git clean -fd
git pull --all

if [ "$arg1" != "--latest" ]; then
    echo "Checkout latest MTFG release"
    git checkout $(git describe --tags `git rev-list --tags --max-count=1`)
else
    echo "Use latest git commit (only for developers!)"
    if git branch -a | grep -q "next" ; then
        echo "Switch to 'next' branch"
        git checkout next
        git pull
    fi
fi

if command -v apt; then
    # debian based distro:
    source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
    conda env create -f environment_ubuntu.yaml
fi

if command -v pacman; then
    # arch based distro:
    pip install -r requirements.txt
fi

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/assets/ffmpeg" \
    "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/funscript_editor/data/ffmpeg"
popd

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/contrib/Installer/assets/main.lua" \
    "$OFS_EXTENSION_DIR/MTFG/main.lua"

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/contrib/Installer/assets/json.lua" \
    "$OFS_EXTENSION_DIR/MTFG/json.lua"

popd

if [ ! -e ~/.local/bin/OpenFunscripter ]; then
    mkdir -p ~/.local/bin
    ln -s `realpath $OFS_APP_DIR`/bin/OpenFunscripter ~/.local/bin/OpenFunscripter
fi

mkdir -p ~/.local/share/applications

cat >~/.local/share/applications/OpenFunscripter.desktop <<EOL
[Desktop Entry]
Type=Application
Name=OpenFunscripter
Exec=`realpath $OFS_APP_DIR`/bin/OpenFunscripter
Comment=OpenFunscripter
StartupWMClass=OpenFunscripter
Icon=`realpath $OFS_APP_DIR`/bin/data/logo64.png
EOL


echo "Installation Completed"
