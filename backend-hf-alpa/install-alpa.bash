#!/bin/bash

set -e

git clone -b v0.1.6 --recursive https://github.com/alpa-projects/alpa.git

cd alpa
pip install .

cd build_jaxlib
# The build script will automatically try a number of subdirectories
# of the specified cudnn_path for headers and libraries, so all we need
# to do is find the right prefix
LD_LIBRARY_PATH=/usr/local/cuda/compat python build/build.py \
    --cuda_path /usr/local/cuda \
    --cuda_version 11.0 \
    --cudnn_path /usr \
    --cudnn_version 8 \
    --enable_cuda \
    --tf_path=$(pwd)/../third_party/tensorflow-alpa
pip install dist/*.whl
