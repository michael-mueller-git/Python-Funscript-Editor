#! /bin/bash

set -e

if [ "$DEBUG" != "" ]; then
    set -x
fi

script=$(readlink -f "$0")

show_usage() {
    echo "Usage: $script --appdir <path to AppDir>"
    echo
    echo "Bundles software available as conda packages into an AppDir"
}

_isterm() {
    tty -s && [[ "$TERM" != "" ]] && tput colors &>/dev/null
}

log() {
    _isterm && tput setaf 3
    _isterm && tput bold
    echo -*- "$@"
    _isterm && tput sgr0
    return 0
}

APPDIR=

while [ "$1" != "" ]; do
    case "$1" in
        --plugin-api-version)
            echo "0"
            exit 0
            ;;
        --appdir)
            APPDIR="$2"
            shift
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            log "Invalid argument: $1"
            log
            show_usage
            exit 1
            ;;
    esac
done

if [ "$APPDIR" == "" ]; then
    show_usage
    exit 1
fi

mkdir -p "$APPDIR"

CONDA_DOWNLOAD_DIR="$PWD"

# the user can specify a directory into which the conda installer is downloaded
# if they don't specify one, we use a temporary directory with a predictable name to preserve downloaded files across runs
# this should reduce the download overhead
# if one is specified, the installer will not be re-downloaded unless it has changed
if [ "$CONDA_DOWNLOAD_DIR" != "" ]; then
    # resolve path relative to cwd
    if [[ "$CONDA_DOWNLOAD_DIR" != /* ]]; then
        CONDA_DOWNLOAD_DIR="$(readlink -f "$CONDA_DOWNLOAD_DIR")"
    fi

    log "Using user-specified download directory: $CONDA_DOWNLOAD_DIR"
else
    # create temporary directory into which downloaded files are put
    CONDA_DOWNLOAD_DIR="/tmp/linuxdeploy-plugin-conda-$(id -u)"

    log "Using default temporary download directory: $CONDA_DOWNLOAD_DIR"
fi

# make sure the directory exists
mkdir -p "$CONDA_DOWNLOAD_DIR"

# install Miniconda, a self contained Python distribution, into AppDir
miniconda_installer_filename=Miniconda3-latest-Linux-x86_64.sh

pushd "$CONDA_DOWNLOAD_DIR"
    miniconda_url=https://repo.anaconda.com/miniconda/"$miniconda_installer_filename"
    # let's make sure the file exists before we then rudimentarily ensure mutual exclusive access to it with flock
    # we set the timestamp to epoch 0; this should likely trigger a redownload for the first time
    touch "$miniconda_installer_filename" -d '@0'
    flock "$miniconda_installer_filename" wget -N -c "$miniconda_url"
popd

export CONDA_ALWAYS_YES="true"

if [ -d "$APPDIR"/usr/conda ]; then
    log "WARNING: conda prefix directory exists: $APPDIR/usr/conda"
    log "Please make sure you perform a clean build before releases to make sure your process works properly."
    
    # activate environment
    . "$APPDIR"/usr/conda/bin/activate
else
    # install into usr/conda/ instead of usr/ to make sure that the libraries shipped with conda don't overwrite or
    # interfere with libraries bundled by other plugins or linuxdeploy itself
    bash "$CONDA_DOWNLOAD_DIR"/"$miniconda_installer_filename" -b -p "$APPDIR"/usr/conda -f
    . "$APPDIR"/usr/conda/bin/activate
    conda env update -f environment_appimage.yaml
fi

# build and install local python package
make docs package
if [ -d dist ]; then
    find dist -iname "*.whl" | sort | tail -n 1 | xargs -I {} pip3 install --force-reinstall {}
fi

if [ -f ffmpeg ]; then
    cp -fv ffmpeg $APPDIR/usr/conda/lib/python3*/site-packages/funscript_editor/data
    chmod +x $APPDIR/usr/conda/lib/python3*/site-packages/funscript_editor/data/ffmpeg
fi

# create missing symlinks
mkdir -p "$APPDIR"/usr/bin/
mkdir -p "$APPDIR"/usr/lib/
pushd "$APPDIR"
for i in usr/conda/bin/*; do
    rm -f usr/bin/"$(basename "$i")"
    ln -s ../../"$i" usr/bin/
done

for i in usr/conda/lib/*; do
    rm -f usr/lib/"$(basename "$i")"
    ln -s ../../"$i" usr/lib/
done

rm -f usr/conda/bin/platforms
ln -s ../plugins/platforms usr/conda/bin
popd

# disable history substitution, b/c we use ! in quoted strings
set +H
APPDIR_FULL="$(pwd)/$APPDIR"
pushd "$APPDIR_FULL"

# replace absolute paths in some specific files (regex could result in false replacements in other files)
[ -f usr/conda/etc/profile.d/conda.sh ] && sed -i --follow-symlinks "s|'$APPDIR_FULL|\"\${APPDIR}\"'|g" usr/conda/etc/profile.d/conda.sh
[ -f usr/conda/etc/profile.d/conda.sh ] && sed -i --follow-symlinks "s|$APPDIR_FULL|\${APPDIR}|g" usr/conda/etc/profile.d/conda.sh
[ -f usr/conda/etc/profile.d/conda.csh ] && sed -i --follow-symlinks "s|$APPDIR_FULL|\${APPDIR}|g" usr/conda/etc/profile.d/conda.csh
[ -f usr/conda/etc/fish/conf.d/conda.fish ] && sed -i --follow-symlinks "s|$APPDIR_FULL|\$APPDIR|g" usr/conda/etc/fish/conf.d/conda.fish

# generic files in usr/conda/bin/ and usr/conda/condabin/
for i in usr/conda/bin/* usr/conda/condabin/*; do
    if [ -f "$i" ]; then
        # shebangs
        sed -i --follow-symlinks "s|^#!$APPDIR_FULL/usr/conda/bin/|#!/usr/bin/env |" "$i"
        # perl assignments (must be before bash assignments)
        sed -ri --follow-symlinks "s|^(my.*=[[:space:]]*\")$APPDIR_FULL|\1\$ENV{APPDIR} . \"|g" "$i"
        # bash assignments
        sed -ri --follow-symlinks "s|(=[[:space:]]*\")$APPDIR_FULL|\1\${APPDIR}|g" "$i"
    fi
done

# specific files in usr/conda/bin/ (regex could result in false replacements in other files)
[ -f usr/conda/bin/python3-config ] && sed -i --follow-symlinks "s|$APPDIR_FULL|\${APPDIR}|g" usr/conda/bin/python3-config
[ -f usr/conda/bin/ncursesw6-config ] && sed -i --follow-symlinks "s|$APPDIR_FULL|\${APPDIR}|g" usr/conda/bin/ncursesw6-config
[ -f usr/conda/etc/fonts/fonts.conf ] && sed -i --follow-symlinks "s|<cachedir>$APPDIR_FULL|<cachedir>/tmp|g" usr/conda/etc/fonts/fonts.conf 
[ -f usr/conda/etc/fonts/fonts.conf ] && sed -i --follow-symlinks "s|<dir>$APPDIR_FULL|<dir>|g" usr/conda/etc/fonts/fonts.conf 

popd

# generate linuxdeploy-plugin-conda-hook
mkdir -p "$APPDIR"/apprun-hooks
cat > "$APPDIR"/apprun-hooks/linuxdeploy-plugin-conda-hook.sh <<\EOF
# generated by linuxdeploy-plugin-conda

# export APPDIR variable to allow for running from extracted AppDir as well
export APPDIR="${APPDIR:-$(readlink -f "$(dirname "$0")")}"
# export PATH to allow /usr/bin/env shebangs to use the supplied applications
export PATH="$APPDIR"/usr/bin:"$PATH"
# set font config path
export FONTCONFIG_PATH="$APPDIR"/usr/conda/etc/fonts
export FONTCONFIG_FILE="$APPDIR"/usr/conda/etc/fonts/fonts.conf
EOF


exit 0
