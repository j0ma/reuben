#!/usr/bin/env bash

debug_mode=${debug_mode:-no}

if [ "${debug_mode}" = "yes" ]
then
    python_cmd="python -m pudb"
else
    python_cmd="python"
fi

if [ "${replication_resampling}" = "no" ]
then
    replication_resampling_flag="none"
else
    replication_resampling_flag=${replication_resampling}
fi

if [ -n "${task_resampling_num_tasks}" ]
then
    task_resampling_num_tasks_flag="--task-resampling-num-tasks ${task_resampling_num_tasks}"
else
    task_resampling_num_tasks_flag=""
fi

if [ "${task_resampling_with_replacement}" = "yes" ]
then
    task_resampling_with_replacement_flag="--task-resampling-with-replacement"
else
    task_resampling_with_replacement_flag=""
fi

if [ "${task_resampling}" = "no" ]
then
    task_resampling_flag="none"
    task_resampling_num_tasks_flag=""
elif [ "${task_resampling}" = "nonparametric" ] 
then
    task_resampling_flag="nonparametric"
else
    task_resampling_flag=${task_resampling}
fi

if [ "${standardized}" = "yes" ]
then
    standardized_flag="--standardized"
else
    standardized_flag=""
fi

dataset=${dataset:-./data/test-data/openner_f1_mean_sd_twosds_df.jsonl}
pickle_folder=${pickle_folder:-./misc/}
pickle_flag="--pickle-output-folder ${pickle_folder}"

case $sd_type in
    seed-only)
	sd_flag="--replication-sd-col ${seed_sd_col}"
	;;
    boot-only)
	sd_flag="--replication-sd-col ${boot_sd_col}"
	;;
    seed-and-boot)
	sd_flag="--seed-sd-col ${seed_sd_col} --boot-sd-col ${boot_sd_col}"
	;;
    *)
	echo "Unknown SD type: ${sd_type}" && exit 1
esac
