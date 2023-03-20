#!/bin/bash
set -e
set -u

CONDA_RUN="/opt/anaconda3/bin/conda run --no-capture-output"

if [ $# -lt 2 ]
then
    echo 'Require at least two arguments:' >&2
    $CONDA_RUN python serve-backend.py --help
    exit 1
fi

LLAMA_DIR="$1"
MODEL_SIZE="$2"
nproc_per_node=`ls -1q $LLAMA_DIR/$MODEL_SIZE/*.pth | wc -l`

$CONDA_RUN \
    torchrun --nproc_per_node $nproc_per_node \
    serve-backend.py "$@"
