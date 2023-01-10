#!/bin/bash
# Description: Installer for OFS + MTFG

arg1="$1"

if [ "$EUID" -eq 0 ]; then
    echo "ERROR: You can not run this script with sudo!!"
    exit 1
fi

echo "install required packages"
if command -v apt; then
    # debian based distro:
    sudo mkdir -p /nix
    sudo chown $USER /nix
    sudo apt install curl
    sh <(curl -L https://nixos.org/nix/install)
    . /home/$USER/.nix-profile/etc/profile.d/nix.sh

    echo "Install OFS build dependencies"
    sudo apt install -y cmake build-essential libmpv-dev libglvnd-dev libxext-dev make \
        git gcc g++ cmake libmpv-dev libatlas-base-dev
fi

if [ ! -f ~/.config/nix/nix.conf ]; then
    mkdir -p ~/.config/nix
    echo "experimental-features = nix-command flakes" >  ~/.config/nix/nix.conf
fi

OFS_APP_DIR="$HOME/.local/share/OFS/application"
OFS_EXTENSION_DIR="$HOME/.local/share/OFS/OFS3_data/extensions"

if [ -d $OFS_APP_DIR ]; then
    echo ">> OpenFunscripter Source already downloaded (Updating...)"
    pushd $OFS_APP_DIR
    git pull
    git reset --hard HEAD
    git clean -fd
    git remote prune origin
    git checkout master
    git pull
    git submodule update --init
else
    mkdir -p `dirname $OFS_APP_DIR`
    echo ">> Clone OpenFunscripter Source"
    git clone https://github.com/OpenFunscripter/OFS.git $OFS_APP_DIR
    pushd $OFS_APP_DIR
    git submodule update --init
    echo ">> OpenFunscripter Source downloaded to $OFS_APP_DIR"
fi

if [ "$arg1" != "--latest" ]; then
    echo "Checkout latest OpenFunscripter release"
    git checkout $(git describe --tags `git rev-list --tags --max-count=1`)
    git submodule update --init
else
    echo "Use latest git commit (only for developers!)"
fi

echo ">> Build OFS in $PWD"
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
git remote prune origin
git checkout main
git pull

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

# build nix environment
nix develop --command sleep 1

popd

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/assets/ffmpeg" \
    "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/funscript_editor/data/ffmpeg"

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/contrib/Installer/assets/main.lua" \
    "$OFS_EXTENSION_DIR/MTFG/main.lua"

cp -fv "$OFS_EXTENSION_DIR/MTFG/Python-Funscript-Editor/contrib/Installer/assets/json.lua" \
    "$OFS_EXTENSION_DIR/MTFG/json.lua"

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

echo -e "\n"
echo "Installation Completed"

if [ "$arg1" = "--latest" ]; then
    echo "WARNING: you have install the latest application code"
fi
