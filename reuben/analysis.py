from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as sps


def winsorize(a, *, thresh=3.5, method="mad", replace="median", axis=None):
    a = np.asarray(a)
    if method == "z":
        mu = np.mean(a, axis=axis, keepdims=True)
        sigma = np.std(a, axis=axis, ddof=0, keepdims=True)
        z = (a - mu) / sigma
        mask = np.abs(z) > thresh
    elif method == "mad":
        med = np.median(a, axis=axis, keepdims=True)
        mad = np.median(np.abs(a - med), axis=axis, keepdims=True)
        z = 0.6745 * (a - med) / mad
        mask = np.abs(z) > thresh
    elif method == "iqr":
        q1 = np.percentile(a, 25, axis=axis, keepdims=True)
        q3 = np.percentile(a, 75, axis=axis, keepdims=True)
        iqr = q3 - q1
        lower = q1 - thresh * iqr
        upper = q3 + thresh * iqr
        mask = (a < lower) | (a > upper)
    else:
        raise ValueError(method)

    if replace == "mean":
        stat = np.mean(a, axis=axis, keepdims=True)
    elif replace == "median":
        stat = np.median(a, axis=axis, keepdims=True)
    else:
        raise ValueError(replace)

    out = a.copy()
    out[mask] = stat
    return out, mask


def count_uniq_vals_over_cols(aggregated_repls, row_index=None):
    M, R = aggregated_repls.shape

    rows = np.repeat(np.arange(M), R)
    flat = pd.Series(aggregated_repls.ravel(), name="value")
    df = pd.DataFrame({"row": rows, "value": flat})

    counts = pd.crosstab(index=df["row"], columns=df["value"])

    counts = counts.reindex(np.arange(M), fill_value=0)

    counts.index.name = ""
    counts.columns.name = "rank"
    if row_index:
        counts.index = row_index

    return counts


def rank_rows(arr):
    out = sps.rankdata(arr, axis=0)
    maxval = np.max([x for x in out.ravel() if ~np.isnan(x)], axis=None)
    out = maxval + 1 - out
    return out


def rank_crosstab(replicated_ranks, row_index=None):
    histogram = count_uniq_vals_over_cols(replicated_ranks, row_index=row_index)
    return histogram


def pairwise_diffs_numpy(
    arr: np.ndarray, keep_diag: bool = True, remove_first_row_and_col: bool = True
) -> np.ndarray:
    k = 0 if keep_diag else 1
    if remove_first_row_and_col:
        diffs = np.triu(arr - np.expand_dims(arr, 1), k=k)[:-1, 1:]
    else:
        diffs = np.triu(arr - np.expand_dims(arr, 1), k=k)
    return diffs


def df_to_replications(
    data, score_col, model_col, task_col, seed_col, bootstrap_col, masked: bool = False
):
    # 1) build an xarray Dataset with a 4‐level index
    data = data.set_index([bootstrap_col, seed_col, task_col, model_col])
    ds = data.to_xarray()

    # 2) pull out the DataArray of interest
    da = ds[score_col]

    # 3) reorder dims so they come out (bootstrap, seed, task, model)
    dims = [bootstrap_col, seed_col, task_col, model_col]
    da = da.transpose(*dims)

    # 4) hand back either .values or a masked‐array
    return da.to_masked_array() if masked else da.values


def infer_parameters(replications):
    _, _, T, M = replications.shape

    overall_means_hat = replications.reshape(-1, M).mean(axis=0)
    task_means_hat = replications.reshape(-1, T, M).mean(axis=0)

    # First collapse seed and bootstrap to one "replication dimension".
    # Then take the mean over that to get task-level means.
    # Then take the SD over the tasks.
    between_task_sd_hat = replications.reshape(-1, T, M).mean(axis=0).std(axis=0)

    # For the seed SD, we first average all the bootstrap replications
    # to get something close to the original data and then take the SD
    # over seeds.
    seed_sd_hat = replications.mean(axis=0).std(axis=0)

    # For the bootstrap SD, we first take the SD
    # across the bootstrap replications for each seed,
    # and then average over seeds
    boot_sd_hat = replications.std(axis=0).mean(axis=0)

    inferred_comps = {
        "Mean": overall_means_hat,
        "Task mean": task_means_hat,
        "SD (between)": between_task_sd_hat,
        "SD (seed)": seed_sd_hat,
        "SD (boot)": boot_sd_hat,
    }

    return inferred_comps


def compute_replication_suff_stats(
    data: pd.DataFrame,
    score_col: str,
    model_col: str,
    task_col: str,
    seed_idx_col: str,
    boot_idx_col: str,
    replication_idx_col: Optional[str] = None,
) -> pd.DataFrame:

    average_perf = data.groupby([model_col, task_col])[score_col].mean()
    out = pd.DataFrame(
        {
            score_col: average_perf,
        }
    )

    replications = df_to_replications(
        data=data,
        score_col=score_col,
        model_col=model_col,
        task_col=task_col,
        seed_col=seed_idx_col,
        bootstrap_col=boot_idx_col,
    )
    inferred = infer_parameters(replications)

    out = out.copy()
    out["sd_seed"] = inferred["SD (seed)"].flatten()
    out["sd_boot"] = inferred["SD (boot)"].flatten()
    out["sd_repl"] = np.sqrt(out["sd_seed"] ** 2 + out["sd_boot"] ** 2)

    return out


# Basic descriptive stats
def compute_task_mean_and_std(
    data: pd.DataFrame,
    score_col: str = "Score",
    task_col: str = "Task",
) -> pd.DataFrame:
    score_stats = data.groupby(task_col)[score_col].agg(["mean", "std"])
    score_stats = score_stats.rename(columns={"mean": "score_mean", "std": "score_sd"})
    return score_stats


def standardize_data(
    data: pd.DataFrame,
    stats: pd.DataFrame,
    score_col: str = "Score",
    score_sd_cols: Optional[list[str]] = None,
    task_col: str = "Task",
):
    if score_sd_cols is None:
        score_sd_cols = []

    _data = data.copy()
    combo = pd.merge(
        left=_data,
        right=stats,
        left_index=False,
        left_on=task_col,
        right_index=True,
    )
    score_centered = combo[score_col] - combo["score_mean"]
    combo["score_standardized"] = score_centered / combo["score_sd"]
    _data[score_col] = combo["score_standardized"]

    for sdcol in score_sd_cols:
        if not sdcol:
            raise ValueError(f"Invalid SD column detected: {score_sd_cols}")
        combo[f"{sdcol}_standardized"] = combo[sdcol] / combo["score_sd"]
        _data[sdcol] = combo[f"{sdcol}_standardized"]

    return _data
