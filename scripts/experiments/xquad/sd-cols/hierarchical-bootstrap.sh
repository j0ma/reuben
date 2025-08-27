#!/usr/bin/env bash

export folder_containing_this_script=$(dirname ${BASH_SOURCE})
export parse_args_script=${folder_containing_this_script}/../parse-args.sh
export input_jsonl=${folder_containing_this_script}/data.jsonl

export sd_type=${sd_type:-"seed-and-boot"}
export boot_sd_col="SD_boot_avg"
export seed_sd_col="SD_seed"

source ${parse_args_script}

echo "SD type: ${sd_type}"

${python_cmd} reuben/cli.py \
    compare-models-aggregate \
	--replication-resampling-method ${replication_resampling_flag} \
	--task-resampling-method ${task_resampling_flag} \
	--num-bootstrap-resamples 10000 \
	${task_resampling_num_tasks_flag} \
	${task_resampling_with_replacement_flag} \
	${sd_flag} \
	--score-col F1 \
	--model-col Model \
	--task-col Language \
	${standardized_flag} \
	--rounding 2 \
	${pickle_flag} \
    ${input_jsonl}
