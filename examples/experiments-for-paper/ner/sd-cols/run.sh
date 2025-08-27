#!/usr/bin/env bash

export folder_containing_this_script=$(dirname ${BASH_SOURCE})
export input_jsonl=${folder_containing_this_script}/data.jsonl

export sd_type=${sd_type:-"seed-boot"}
export config_yaml=${folder_containing_this_script}/config-${sd_type}.yaml

reuben --config-file ${config_yaml} \
    variance-components ${input_jsonl} \
    --output-path ${folder_containing_this_script}/results/${sd_type}/variance-components \
    --pickle-output-folder ${folder_containing_this_script}/results/${sd_type}/variance-components/pkl

reuben --config-file ${config_yaml} \
    compare-models-aggregate ${input_jsonl} \
    --output-path ${folder_containing_this_script}/results/${sd_type}/aggregate-analysis \
    --pickle-output-folder ${folder_containing_this_script}/results/${sd_type}/aggregate-analysis/pkl
