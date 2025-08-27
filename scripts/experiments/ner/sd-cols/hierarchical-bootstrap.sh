#!/usr/bin/env bash

folder_containing_this_script=$(dirname ${BASH_SOURCE})
parse_args_script=${folder_containing_this_script}/../parse-args.sh
input_jsonl=${folder_containing_this_script}/data.jsonl

source ${parse_args_script}

${python_cmd} reuben/cli.py \
    compare-models-aggregate \
	--replication-resampling-method ${replication_resampling_flag} \
	--task-resampling-method ${task_resampling_flag} \
	${task_resampling_num_tasks_flag} \
	${task_resampling_with_replacement_flag} \
	--num-bootstrap-resamples 10000 \
	--seed-sd-col 'SD (seed)' \
	--boot-sd-col 'SD (boot)' \
	--score-col Mean \
	--model-col Model \
	--task-col Task \
	${standardized_flag} \
	--rounding 2 \
	${pickle_flag} \
    ${input_jsonl}
