#!/bin/bash

import json
import sys
from pathlib import Path

subcommand = sys.argv[1]
output_directory = ''.join(t.title() for t in subcommand.split('-'))
with open(Path(output_directory) / 'ResponseBuckets/runSummary.json') as f:
    run_summary = json.load(f)

print(json.dumps(run_summary, indent=2))
sys.exit(run_summary['failedRequestsCount'])
