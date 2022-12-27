#!/usr/bin/bash

root_dir="$(dirname $0)"
cmd="python3 `dirname $0`/funscript-editor.py $@"
echo "cmd: $cmd" > /tmp/mtfg-nix.log
echo "dir: $root_dir" >> /tmp/mtfg-nix.log
# cd $root_dir && nix-shell --run "$cmd" >> /tmp/mtfg-nix.log 2>&1
cd $root_dir && nix develop --command $cmd >> /tmp/mtfg-nix.log 2>&1
