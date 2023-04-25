#!/bin/bash
# This script should be run under an init process to ensure the deepspeed child process is reaped.
# Examples: docker run --init (Docker 1.13+) or tini or dumb-init
set -e
set -u

ds_report
python serve-mii.py &
uvicorn --host 0.0.0.0 "$@" main:app
