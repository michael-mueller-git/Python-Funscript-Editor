#!/bin/bash

if [ -d ./OFS ]; then
    echo "OpenFunscripter Source already downloaded"
    pushd OFS
    git pull
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

echo "copy lua plugin"
cp -vf funscript_generator_linux.lua OFS/bin/data/lua
