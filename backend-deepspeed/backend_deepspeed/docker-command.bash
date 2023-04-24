#!/bin/bash
# This script should be run under an init process to ensure the deepspeed child process is reaped.
# Examples: docker run --init (Docker 1.13+) or tini or dumb-init
set -e
set -u

CONDA_RUN="/opt/anaconda3/bin/conda run --no-capture-output"

$CONDA_RUN ds_report
$CONDA_RUN python &
$CONDA_RUN uvicorn --host 0.0.0.0 "$@" main:app
