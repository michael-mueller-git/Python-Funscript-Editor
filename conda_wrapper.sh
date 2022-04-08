#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate funscript-editor
python3 `dirname $0`/funscript-editor.py $@
