#!/usr/bin/env bash

reuben --config-file ./examples/xquad/config-csv-output.yaml variance-components ./examples/xquad/data.jsonl
reuben --config-file ./examples/xquad/config-csv-output.yaml compare-models-aggregate ./examples/xquad/data.jsonl
