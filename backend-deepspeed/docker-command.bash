#!/bin/bash
set -e
set -u

ds_report
uvicorn --host 0.0.0.0 "$@" main:app
