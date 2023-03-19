#!/bin/sh

set -e
set -u

if [ $# -ne 1 ]
then
    echo "Usage: $0 FUZZ_TEST_TOKEN" >&2
    exit 1
fi
FUZZ_TEST_TOKEN="$1"

PYTHON=python3
RESTLER=/RESTler/restler/Restler
API_SPEC=/swagger.yaml
OUTPUT_DIR=/output

cd $OUTPUT_DIR
rm -rf *
$RESTLER compile --api_spec $API_SPEC
for subcmd in test fuzz-lean fuzz
do
    $RESTLER $subcmd \
        --grammar_file Compile/grammar.py \
        --dictionary_file Compile/dict.json \
        --settings Compile/engine_settings.json \
        --no_ssl \
        --host demo \
        --token_refresh_command "$PYTHON /generate-token.py $FUZZ_TEST_TOKEN" \
        --token_refresh_interval 60
    $PYTHON /check-output.py $subcmd
done
