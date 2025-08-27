from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

import reuben.aggregators as agg
from reuben.analysis import pairwise_diffs_numpy, rank_crosstab, rank_rows
from reuben.resampling import (
    ReplicationResamplingMethod,
    TaskResamplingConfig,
    TaskResamplingMethod,
    resample_residual_given_sds,
)
from reuben.utils import (
    apply_no_styling,
    create_mean_plus_sd_array,
    estimate_correlation_ci_with_fisher_transform,
    get_suffix,
)


def variance_components_summary(data, config, rounding: int = 3) -> pd.DataFrame:
    task_col, model_col, score_col = config.task_col, config.model_col, config.score_col

    seed_sd_col, boot_sd_col, repl_sd_col = (
        config.seed_sd_col,
        config.boot_sd_col,
        config.replication_sd_col,
    )

    data_by_task = data.set_index([task_col, model_col])

    task_means = data_by_task[score_col]
    between_task_sds = task_means.groupby(model_col).std()

    repl = data_by_task[repl_sd_col].unstack(task_col)
    seed = data_by_task[seed_sd_col].unstack(task_col) if seed_sd_col else None
    boot = data_by_task[boot_sd_col].unstack(task_col) if boot_sd_col else None

    def _min_mean_max(df, prefix, axis=1):
        tbl = df.agg(["min", "mean", "max"], axis=axis).rename(
            columns={
                "min": f"{prefix}_min",
                "mean": f"{prefix}_mean",
                "max": f"{prefix}_max",
            }
        )
        return tbl

    out_parts = []
    if seed is not None:
        out_parts.append(_min_mean_max(seed, "seed_sd"))
    if boot is not None:
        out_parts.append(_min_mean_max(boot, "boot_sd"))
    out_parts.append(_min_mean_max(repl, "repl_sd"))

    out = pd.concat(out_parts, axis=1)
    if (seed is not None) and (boot is not None):
        boot_to_seed_ratio = (boot / seed).replace([np.inf, -np.inf], np.nan)
        boot_seed_corr = seed.corrwith(boot, axis=1).rename("boot_seed_corr")
        out = pd.concat(
            [out, _min_mean_max(boot_to_seed_ratio, "boot_to_seed_ratio")], axis=1
        )
        out = pd.concat([out, boot_seed_corr], axis=1)

    out["between_sd"] = between_task_sds
    out = out.reset_index().rename(columns={model_col: config.model_col})
    return out.round(rounding)


def compute_base_statistics(data, raw_data, config):
    data_by_task = data.set_index([config.task_col, config.model_col])

    uniq_models = list(pd.Series(data[config.model_col].unique()))
    num_models = data[config.model_col].nunique()
    num_uniq_tasks = data[config.task_col].nunique()

    task_means = data_by_task[config.score_col]
    leaderboard_task_means_df = task_means.unstack(level=config.task_col).fillna(
        task_means.mean()
    )
    leaderboard_task_means = leaderboard_task_means_df.to_numpy()
    orig_results = leaderboard_task_means
    between_task_sds = task_means.groupby(config.model_col).std()

    metadata = {
        "uniq_models": uniq_models,
        "num_models": num_models,
        "num_uniq_tasks": num_uniq_tasks,
        "between_task_sds": between_task_sds,
        "data_by_task": data_by_task,
    }
    return leaderboard_task_means, orig_results, metadata


def apply_resampling(leaderboard_task_means, raw_data, config, metadata):
    data_by_task = metadata["data_by_task"]
    num_models = metadata["num_models"]
    num_uniq_tasks = metadata["num_uniq_tasks"]
    between_task_sds = metadata["between_task_sds"]

    should_resample_tasks = TaskResamplingMethod.is_not_none(
        config.task_resampling_method
    )
    should_resample_replications = ReplicationResamplingMethod.is_not_none(
        config.replication_resampling_method
    )
    if not (should_resample_tasks or should_resample_replications):
        return leaderboard_task_means[..., None]

    if config.num_bootstrap_resamples <= 0:
        raise ValueError(
            f"Must provide --num-bootstrap-resamples > 0. Got: {config.num_bootstrap_resamples}."
        )

    num_tasks_requested = (
        config.task_resampling_num_tasks
        if config.task_resampling_num_tasks is not None
        else num_uniq_tasks
    )

    if should_resample_tasks:
        task_resampling_cfg = TaskResamplingConfig(
            num_tasks=int(num_tasks_requested),
            fraction=(num_tasks_requested / num_uniq_tasks),
            replace=config.task_resampling_with_replacement,
            subsample=bool(num_tasks_requested < num_uniq_tasks),
        )
        _num_tasks = int(task_resampling_cfg.num_tasks)

        if config.task_resampling_method == TaskResamplingMethod.nonparametric:
            choice_options = np.arange(num_uniq_tasks)
            resampled_task_ids = np.array(
                [
                    np.random.choice(
                        choice_options,
                        size=_num_tasks,
                        replace=config.task_resampling_with_replacement,
                    )
                    for _ in range(config.num_bootstrap_resamples)
                ]
            ).T
            leaderboard = leaderboard_task_means[
                :, resampled_task_ids
            ]

        elif config.task_resampling_method == TaskResamplingMethod.parametric:
            leaderboard_grand_means = leaderboard_task_means.mean(axis=1)
            repeated_between_lang_sds = np.tile(
                between_task_sds.to_numpy()[..., None], reps=(1, _num_tasks)
            )
            _task_deviations = resample_residual_given_sds(
                standard_deviations=repeated_between_lang_sds, num_samples=1
            )
            task_deviations = _task_deviations.reshape(
                num_models, _num_tasks, 1
            )
            leaderboard = (
                leaderboard_grand_means[..., None, None] + task_deviations
            )

        else:
            raise NotImplementedError("Semiparametric sampling not implemented")

    else:
        _num_tasks = num_uniq_tasks
        leaderboard = leaderboard_task_means[..., None]

    if should_resample_replications:
        orig_results = leaderboard_task_means

        if (
            config.replication_resampling_method
            == ReplicationResamplingMethod.parametric
        ):
            replication_deviation_sds = data_by_task[
                config.replication_sd_col
            ]
            
            sds_matrix = (
                replication_deviation_sds.unstack(config.task_col)
                .fillna(replication_deviation_sds.mean(axis=None))
                .to_numpy()
            )
            replication_devs = resample_residual_given_sds(
                standard_deviations=sds_matrix,
                num_samples=config.num_bootstrap_resamples,
            ).transpose(1, 2, 0) 

        elif (
            config.replication_resampling_method
            == ReplicationResamplingMethod.nonparametric
        ):
            if config.replication_idx_col is not None:
                replication_indices = raw_data[config.replication_idx_col]
            elif config.boot_idx_col is not None and config.seed_idx_col is not None:
                pairs = list(
                    product(
                        raw_data[config.boot_idx_col].unique(),
                        raw_data[config.seed_idx_col].unique(),
                    )
                )
                replication_idx_map = {(b, s): ix for ix, (b, s) in enumerate(pairs)}
                replication_indices = raw_data.apply(
                    lambda row: replication_idx_map[
                        (row[config.boot_idx_col], row[config.seed_idx_col])
                    ],
                    axis=1,
                )
            elif config.boot_idx_col is not None:
                replication_indices = raw_data[config.boot_idx_col]
            elif config.seed_idx_col is not None:
                replication_indices = raw_data[config.seed_idx_col]
            else:
                raise ValueError(
                    "Must provide either `replication_idx_col`, `boot_idx_col`, or `seed_idx_col`."
                )

            _replication_col = "Replication"
            raw_data = raw_data.copy()
            raw_data[_replication_col] = replication_indices

            resampled_scores = (
                raw_data.groupby([config.model_col, config.task_col])[config.score_col]
                .sample(n=config.num_bootstrap_resamples, replace=True)
                .to_numpy()
            ).reshape(
                num_models, metadata["num_uniq_tasks"], config.num_bootstrap_resamples
            )

            replication_devs = resampled_scores - orig_results[..., None]
        else:
            raise NotImplementedError(
                f"Replication resampling method {config.replication_resampling_method} not implemented."
            )

        if should_resample_tasks:
            if config.task_resampling_method == TaskResamplingMethod.nonparametric:
                _replication_devs = np.zeros(
                    shape=(num_models, _num_tasks, config.num_bootstrap_resamples)
                )
                for b in range(config.num_bootstrap_resamples):
                    taskids_thisboot = resampled_task_ids[:, b]
                    _replication_devs[:, :, b] = replication_devs[
                        :, taskids_thisboot, b
                    ]
                replication_devs = _replication_devs
            elif _num_tasks < metadata["num_uniq_tasks"]:
                replication_devs = replication_devs[:, :_num_tasks, :]

        leaderboard = leaderboard + replication_devs

    return leaderboard


def compute_all_statistics(leaderboard, orig_results, uniq_models, config):
    stats = {}
    stats["agg_over_task"] = defaultdict(lambda: defaultdict(dict))

    aggregators = {"arithmetic_mean", "geometric_mean", "median"}
    for agg_name in aggregators:
        if config.standardized and agg_name == "geometric_mean":
            continue

        agg_name_np = f"{agg_name}_numpy"
        agg_result = agg.apply_aggregator(name=agg_name_np, arr=leaderboard, axis=1)
        agg_result_origscore = agg.apply_aggregator(
            name=agg_name_np, arr=orig_results, axis=1
        )
        agg_result_mean = agg.apply_aggregator(
            arr=agg_result, name="arithmetic_mean_numpy", axis=-1
        )
        agg_result_sd = agg.apply_aggregator(arr=agg_result, name="sd_numpy", axis=-1)
        agg_result_prettyprint = create_mean_plus_sd_array(
            means=agg_result_origscore,
            sds=agg_result_sd,
            rounding=config.rounding,
            postprocess_fn=apply_no_styling,
            sd_multiplier=1,
        )

        stats["agg_over_task"]["orig"][agg_name] = agg_result_origscore
        stats["agg_over_task"]["raw"][agg_name] = agg_result
        stats["agg_over_task"]["mean"][agg_name] = agg_result_mean
        stats["agg_over_task"]["sd"][agg_name] = agg_result_sd
        stats["agg_over_task"]["pretty"][agg_name] = agg_result_prettyprint

    # Pairwise diffs
    stats["pairwise_diffs"] = defaultdict(lambda: defaultdict(dict))
    for agg_name, agg_results in stats["agg_over_task"]["raw"].items():
        if config.standardized and agg_name == "geometric_mean":
            continue
        diffs = pairwise_diffs_numpy(
            agg_results, keep_diag=True, remove_first_row_and_col=False
        )
        diffs_origscore = pairwise_diffs_numpy(
            stats["agg_over_task"]["orig"][agg_name],
            keep_diag=True,
            remove_first_row_and_col=False,
        )

        diffs_mean = agg.apply_aggregator(
            arr=diffs, name="arithmetic_mean_numpy", axis=-1
        )
        diffs_sd = agg.apply_aggregator(arr=diffs, name="sd_numpy", axis=-1)

        diffs_effect_size = (diffs_origscore / diffs_sd).round(config.rounding)
        diffs_effect_size[np.isnan(diffs_effect_size)] = 0
        diffs_effect_size_triu = np.triu(diffs_effect_size, k=1)[:-1, 1:]

        diffs_effect_size_df = pd.DataFrame(
            diffs_effect_size_triu,
            index=uniq_models[:-1],
            columns=uniq_models[1:],
        )

        diffs_effect_size_str = diffs_effect_size.astype(str) + np.vectorize(
            get_suffix
        )(np.abs(diffs_effect_size) >= 2)[1].astype(str)
        diffs_effect_size_str_triu = np.triu(diffs_effect_size_str, k=1)[:-1, 1:]

        diffs_effect_size_str_df = pd.DataFrame(
            diffs_effect_size_str_triu,
            index=uniq_models[:-1],
            columns=uniq_models[1:],
        )

        diffs_prettyprint = create_mean_plus_sd_array(
            means=diffs_origscore,
            sds=diffs_sd,
            rounding=config.rounding,
            sd_multiplier=1,
        )
        np.fill_diagonal(diffs_prettyprint, "")

        diffs_prettyprint = pd.DataFrame(
            np.triu(diffs_prettyprint)[:-1, 1:],
            index=uniq_models[:-1],
            columns=uniq_models[1:],
        )

        stats["pairwise_diffs"]["orig"][agg_name] = diffs_origscore
        stats["pairwise_diffs"]["raw"][agg_name] = diffs
        stats["pairwise_diffs"]["mean"][agg_name] = diffs_mean
        stats["pairwise_diffs"]["sd"][agg_name] = diffs_sd
        stats["pairwise_diffs"]["effect_size"][agg_name] = diffs_effect_size_str_df
        stats["pairwise_diffs"]["effect_size_numeric"][agg_name] = diffs_effect_size_df
        stats["pairwise_diffs"]["pretty"][agg_name] = diffs_prettyprint

    # Ranks
    stats["ranks"] = {"raw": {}, "crosstab": {}}
    for agg_name, agg_results in stats["agg_over_task"]["raw"].items():
        if config.standardized and agg_name == "geometric_mean":
            continue
        ranks_with_agg = rank_rows(agg_results)
        ranks_with_agg_xtab = rank_crosstab(ranks_with_agg, row_index=uniq_models)

        stats["ranks"]["raw"][agg_name] = ranks_with_agg
        stats["ranks"]["crosstab"][agg_name] = ranks_with_agg_xtab

    def crosstabs_to_concatenate():
        for agg_name, xtab in stats["ranks"]["crosstab"].items():
            yield (
                xtab.reset_index(names=[config.model_col])
                .assign(Aggregator=agg_name)
                .set_index(["Aggregator", config.model_col])
            )

    stats["ranks"]["crosstab"]["summary"] = pd.concat(crosstabs_to_concatenate())

    # Correlation matrices
    stats["correlation_matrices"] = {"task2task": {}, "model2model": {}}

    model2model_corr_obs = np.corrcoef(orig_results)
    model2model_corr_sims = np.dstack(
        [np.corrcoef(arr) for arr in leaderboard.transpose(-1, 0, 1)]
    )
    model2model_corr_std = model2model_corr_sims.std(axis=-1)

    model2model_corr_ci = estimate_correlation_ci_with_fisher_transform(
        model2model_corr_obs, rounding=config.rounding
    )
    model2model_corr_pretty = pd.DataFrame(
        model2model_corr_ci,
        index=uniq_models[:-1],
        columns=uniq_models[1:],
    )

    stats["correlation_matrices"]["model2model"]["observed"] = model2model_corr_obs
    stats["correlation_matrices"]["model2model"]["sd"] = model2model_corr_std
    stats["correlation_matrices"]["model2model"]["pretty"] = model2model_corr_pretty

    task2task_corr_obs = np.corrcoef(orig_results.T)
    stats["correlation_matrices"]["task2task"]["observed"] = task2task_corr_obs

    return stats
