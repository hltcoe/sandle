#!/bin/sh

set -e

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
        --token_refresh_command "$PYTHON /generate-fuzz-test-token.py" \
        --token_refresh_interval 60
    $PYTHON /check-fuzz-test-output.py $subcmd
done
