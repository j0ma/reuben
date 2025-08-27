from collections import defaultdict
from itertools import product
import pickle
import math

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
    grab_data_and_preprocess,
)


def variance_components_summary(data, config, rounding: int = 3) -> pd.DataFrame:
    task_col, model_col, score_col = (
        config.task_col,
        config.model_col,
        config.score_col,
    )
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

    def _min_mean_max(df, prefix):
        tbl = (
            df.agg(["min", "mean", "max"], axis=1)
            .rename(
                columns={
                    "min": f"{prefix}_min",
                    "mean": f"{prefix}_mean",
                    "max": f"{prefix}_max",
                }
            )
            .astype(float)
        )
        return tbl

    pieces = [_min_mean_max(repl, "repl_sd")]
    if seed is not None:
        pieces.append(_min_mean_max(seed, "seed_sd"))
    if boot is not None:
        pieces.append(_min_mean_max(boot, "boot_sd"))

    out = pd.concat(pieces, axis=1)
    out["between_sd"] = between_task_sds
    return out.reset_index().round(rounding)


def variance_components_per_task(data, config, rounding: int = 3) -> pd.DataFrame:
    task_col, model_col, repl_sd_col = (
        config.task_col,
        config.model_col,
        config.replication_sd_col,
    )
    data_by_task = data.set_index([task_col, model_col])
    repl = data_by_task[repl_sd_col].unstack(task_col)

    records = []
    for m in repl.index:
        for t in repl.columns:
            records.append(
                {
                    model_col: m,
                    task_col: t,
                    repl_sd_col: float(repl.loc[m, t]),
                }
            )
    return pd.DataFrame(records).round(rounding)


def compute_base_statistics(data, raw_data, config):
    data_by_task = data.set_index([config.task_col, config.model_col])
    uniq_models = raw_data[config.model_col].unique()
    num_models = len(uniq_models)
    num_uniq_tasks = raw_data[config.task_col].nunique()

    task_means = data_by_task[config.score_col]
    leaderboard_task_means_df = task_means.unstack(config.task_col).fillna(
        task_means.mean()
    )
    leaderboard_task_means = leaderboard_task_means_df.to_numpy()
    between_task_sds = task_means.groupby(config.model_col).std()

    return leaderboard_task_means, {
        "uniq_models": uniq_models,
        "num_models": num_models,
        "num_uniq_tasks": num_uniq_tasks,
        "between_task_sds": between_task_sds,
        "data_by_task": data_by_task,
    }


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
            leaderboard = leaderboard_task_means[:, resampled_task_ids]

        elif config.task_resampling_method == TaskResamplingMethod.parametric:
            leaderboard_grand_means = leaderboard_task_means.mean(axis=1)
            repeated_between_lang_sds = np.tile(
                between_task_sds.to_numpy()[..., None], reps=(1, _num_tasks)
            )
            _task_deviations = resample_residual_given_sds(
                standard_deviations=repeated_between_lang_sds, num_samples=1
            )
            task_deviations = _task_deviations.reshape(num_models, _num_tasks, 1)
            leaderboard = leaderboard_grand_means[..., None, None] + task_deviations

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
            replication_deviation_sds = data_by_task[config.replication_sd_col]

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


def compute_pairwise_diff_statistics(
    leaderboard_task_means,
    leaderboard_replicates,
    uniq_models,
    tasks,
    threshold: float = 2.0,
    rounding: int = 2,
):
    num_models, num_tasks = leaderboard_task_means.shape
    summary_rows, details = [], {}

    for i in range(num_models - 1):
        for j in range(i + 1, num_models):
            a, b = uniq_models[i], uniq_models[j]

            # Task-wise diffs
            diff_obs = leaderboard_task_means[i] - leaderboard_task_means[j]
            diff_rep = leaderboard_replicates[i] - leaderboard_replicates[j]

            # Within-task SDs from the replicate dimension (axis-1)
            within_sds = diff_rep.std(axis=1, ddof=0)
            within_sds[within_sds == 0] = np.nan

            between_sd = float(diff_obs.std(ddof=0))
            within_sd_avg = float(np.nanmean(within_sds))
            sd_eta = float(np.nanstd(within_sds))

            avg_diff = float(diff_obs.mean())
            std_err_mean = float(between_sd / math.sqrt(num_tasks))

            sd_pred = math.sqrt(between_sd**2 + within_sd_avg**2)

            # Significance buckets
            ci_low = diff_obs - 1.96 * within_sds
            ci_high = diff_obs + 1.96 * within_sds
            z = diff_obs / within_sds

            outliers = []
            a_beats_b = []
            b_beats_a = []
            inconclusive = []

            for k in range(num_tasks):
                task = tasks[k]
                if np.abs(z[k]) >= threshold:
                    dest_arr = outliers
                if (diff_obs[k] > 0) and (ci_low[k] > 0):
                    dest_arr = a_beats_b
                elif (diff_obs[k] < 0) and (ci_high[k] < 0):
                    dest_arr = b_beats_a
                else:
                    dest_arr = inconclusive

                dest_arr.append({
                    "task": task,
                    "mean": round(float(diff_obs[k]), rounding),
                    "std_err_mean": round(float(within_sds[k]), rounding)
                    if not np.isnan(within_sds[k])
                    else None,
                })


            summary_rows.append(
                {
                    "ModelA": a,
                    "ModelB": b,
                    "n_tasks": num_tasks,
                    "between_sd": round(between_sd, rounding),
                    "within_sd_mean": round(within_sd_avg, rounding),
                    "within_sd_sd": round(sd_eta, rounding),
                    "mean": round(avg_diff, rounding),
                    "std_err_mean": round(std_err_mean, rounding),
                    "predictive_sd": round(sd_pred, rounding),
                }
            )
            details[(a, b)] = {
                "outliers": outliers,
                "A_beats_B": a_beats_b,
                "B_beats_A": b_beats_a,
                "inconclusive": inconclusive,
            }

    return summary_rows, details


def run_analysis(
    *,
    data_path: str,
    config,
    include_pairwise: bool = True,
    include_varcomp: bool = False,
    pairwise_diffs_task_level_details: bool = False,
    fix_outliers: bool = False,
    leaderboard_pkl: str | None = None,
):
    (
        data,
        seed_sd_col,
        boot_sd_col,
        repl_sd_col,
        raw_data,
    ) = grab_data_and_preprocess(
        data_path,
        config.score_col,
        config.model_col,
        config.task_col,
        config.replication_sd_col,
        config.seed_sd_col,
        config.boot_sd_col,
        config.replication_idx_col,
        config.seed_idx_col,
        config.boot_idx_col,
        standardized=config.standardized,
        return_raw_data=True,
        fix_outliers=fix_outliers,
    )
    config.seed_sd_col = seed_sd_col
    config.boot_sd_col = boot_sd_col
    config.replication_sd_col = repl_sd_col

    uniq_models = raw_data[config.model_col].unique()
    tasks = sorted(raw_data[config.task_col].unique())

    if leaderboard_pkl:
        with open(leaderboard_pkl, "rb") as f:
            leaderboard = pickle.load(f)
        leaderboard_task_means = leaderboard[..., 0]
    else:
        lb_task_means, metadata = compute_base_statistics(data, raw_data, config)
        leaderboard_task_means = lb_task_means
        leaderboard = apply_resampling(lb_task_means, raw_data, config, metadata)

    agg_stats = compute_all_statistics(
        leaderboard, leaderboard_task_means, uniq_models, config
    )

    pairwise_stats = None
    if include_pairwise:
        summary, details = compute_pairwise_diff_statistics(
            leaderboard_task_means,
            leaderboard,
            uniq_models,
            tasks,
            threshold=2,
        )
        pairwise_stats = {"summary": summary, "details": details}

    varcomp_stats = None
    if include_varcomp:
        vc_summary = variance_components_summary(data, config, rounding=config.rounding)
        vc_per_task = variance_components_per_task(
            data, config, rounding=config.rounding
        )
        varcomp_stats = {
            "summary": vc_summary.to_dict(orient="records"),
            "per_task": vc_per_task.to_dict(orient="records"),
            "model_col": config.model_col,
        }

    return {
        "aggregate": agg_stats,
        "pairwise": pairwise_stats,
        "varcomp": varcomp_stats,
        "metadata": {
            "uniq_models": uniq_models,
            "tasks": tasks,
            "leaderboard": leaderboard,
            "task_level_detail": pairwise_diffs_task_level_details,
        },
    }
