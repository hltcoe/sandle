#!/bin/bash
set -e
set -u

CONDA_RUN="/opt/anaconda3/bin/conda run --no-capture-output"

$CONDA_RUN ds_report
$CONDA_RUN uvicorn --host 0.0.0.0 main:app "$@"
