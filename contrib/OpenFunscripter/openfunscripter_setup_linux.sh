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
    sudo apt install -y curl

    if ! command -v nix; then
        sh <(curl -L https://nixos.org/nix/install) --daemon --yes
    fi

    echo "Install OFS AppImage dependencies"
    sudo apt install -y fuse
fi

if [ -f /etc/profile.d/nix.sh ]; then
    . /etc/profile.d/nix.sh
fi

if [ -f /home/$USER/.nix-profile/etc/profile.d/nix.sh ]; then
    . /home/$USER/.nix-profile/etc/profile.d/nix.sh
fi

if ! command -v nix; then
    echo "This installer require the package manager nix"
    exit 1
fi

if [ ! -f ~/.config/nix/nix.conf ]; then
    mkdir -p ~/.config/nix
    echo "experimental-features = nix-command flakes" >  ~/.config/nix/nix.conf
    sudo systemctl restart nix-daemon.service
fi

OFS_APP_DIR="$HOME/.local/share/OFS/application"
OFS_EXTENSION_DIR="$HOME/.local/share/OFS/OFS3_data/extensions"

ofs_appimage_download_url=$(curl -s -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/OpenFunscripter/OFS/releases/latest | grep -Eo "https://.*64x.*AppImage")

echo "OFS AppImage Download URL: $ofs_appimage_download_url"
mkdir -p $OFS_APP_DIR/bin/data
rm -rf $OFS_APP_DIR/bin/OpenFunscripter
wget -c "$ofs_appimage_download_url" -O $OFS_APP_DIR/bin/OpenFunscripter
wget -c https://raw.githubusercontent.com/OpenFunscripter/OFS/master/data/logo64.png -O $OFS_APP_DIR/bin/data/logo64.png
chmod +x $OFS_APP_DIR/bin/OpenFunscripter

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

echo "build nix environment"
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

# When nix was installed with this scrip we need an reboot to work
echo "You may need to restart you computer to get MTFG working"
