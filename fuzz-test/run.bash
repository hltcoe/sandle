#!/bin/bash

set -e
set -u

FUZZ_TEST_DIR=fuzz-test

if [[ $# -ne 1 || ! -d "$FUZZ_TEST_DIR" ]]
then
    echo "Usage: $0 FUZZ_TEST_TOKEN" >&2
    echo >&2
    echo "Must be run from the SANDLE repository root." >&2
    exit 1
fi
FUZZ_TEST_TOKEN="$1"

REPO_DIR=$FUZZ_TEST_DIR/restler-fuzzer
RESTLER_TAG=restler-fuzzer:8.5.0

restler_images=`docker images -q restler-fuzzer | wc -l`
if [ "$restler_images" -eq 0 ]
then
    echo "'$RESTLER_TAG' image not found; cloning repository and building"
    git clone -b v8.5.0 https://github.com/microsoft/restler-fuzzer.git $REPO_DIR
    docker build -t $RESTLER_TAG $REPO_DIR
fi

mkdir -p $FUZZ_TEST_DIR/output
docker run \
    --mount type=bind,src=$PWD/docs/swagger.yaml,dst=/swagger.yaml,readonly \
    --mount type=bind,src=$PWD/$FUZZ_TEST_DIR/run-helper.sh,dst=/run-helper.sh,readonly \
    --mount type=bind,src=$PWD/$FUZZ_TEST_DIR/generate-token.py,dst=/generate-token.py,readonly \
    --mount type=bind,src=$PWD/$FUZZ_TEST_DIR/check-output.py,dst=/check-output.py,readonly \
    --mount type=bind,src=$PWD/$FUZZ_TEST_DIR/output,dst=/output \
    --network sandle_default \
    -it \
    restler-fuzzer \
    sh /run-helper.sh $FUZZ_TEST_TOKEN
