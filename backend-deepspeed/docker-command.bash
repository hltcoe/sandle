#!/bin/bash
set -e
set -u

CONDA_RUN="/opt/anaconda3/bin/conda run --no-capture-output"

if [ $# -lt 1 ]
then
    echo 'Require at least one argument:' >&2
    $CONDA_RUN python serve-backend.py --help
    exit 1
fi

$CONDA_RUN ds_report
$CONDA_RUN python serve-backend.py "$@"
