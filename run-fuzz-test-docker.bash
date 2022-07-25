#!/bin/bash

set -e

RESTLER_IMAGES=`docker images -q restler | wc -l`
if [ "$RESTLER_IMAGES" -eq 0 ]
then
    echo '"restler" image not found; cloning repository and building'
    REPO_DIR=`mktemp -d`
    git clone -b v8.5.0 https://github.com/microsoft/restler-fuzzer.git $REPO_DIR
    docker build -t restler $REPO_DIR
    rm -rf $REPO_DIR
fi

mkdir -p fuzz-test-output
docker run \
    --mount type=bind,src=$HOME/openaisle/docs/swagger.yaml,dst=/swagger.yaml,readonly \
    --mount type=bind,src=$PWD/run-fuzz-test.sh,dst=/run-fuzz-test.sh,readonly \
    --mount type=bind,src=$PWD/generate-fuzz-test-token.py,dst=/generate-fuzz-test-token.py,readonly \
    --mount type=bind,src=$PWD/check-fuzz-test-output.py,dst=/check-fuzz-test-output.py,readonly \
    --mount type=bind,src=$PWD/fuzz-test-output,dst=/output \
    --network openaisle_default \
    -it \
    restler \
    sh /run-fuzz-test.sh
