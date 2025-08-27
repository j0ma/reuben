#!/usr/bin/env bash

reuben --config-file ./examples/flores/config-rich-output.yaml analyze --aggregate-analysis ./examples/flores/data.jsonl 2> /dev/null
reuben --config-file ./examples/flores/config-rich-output.yaml analyze --variance-components --task-resampling-method none ./examples/flores/data.jsonl 2> /dev/null
reuben --config-file ./examples/flores/config-rich-output.yaml analyze --pairwise-diffs --task-resampling-method none ./examples/flores/data.jsonl 2> /dev/null
