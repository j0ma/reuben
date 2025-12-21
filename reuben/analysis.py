from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as sps


def count_uniq_vals_over_cols(aggregated_repls, row_index=None):
    M, R = aggregated_repls.shape

    rows = np.repeat(np.arange(M), R)
    flat = pd.Series(aggregated_repls.ravel(), name="value")
    df = pd.DataFrame({"row": rows, "value": flat})

    counts = pd.crosstab(index=df["row"], columns=df["value"])

    counts = counts.reindex(np.arange(M), fill_value=0)

    counts.index.name = ""
    counts.columns.name = "rank"
    if row_index is not None:
        counts.index = row_index

    return counts


def rank_rows(arr):
    out = sps.rankdata(arr, axis=0)
    maxval = np.max([x for x in out.ravel() if ~np.isnan(x)], axis=None)
    out = maxval + 1 - out
    return out


def rank_crosstab(replicated_ranks, row_index=None, normalize=True, rounding=2):
    histogram = count_uniq_vals_over_cols(replicated_ranks, row_index=row_index)
    if normalize:
        histogram = (100 * histogram.div(histogram.sum(axis=1), axis=0)).round(rounding)
        histogram = histogram.replace(0, "-")
        histogram = (histogram.astype(str) + "%").replace("-%", "-")

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
    data = data.set_index([bootstrap_col, seed_col, task_col, model_col])
    ds = data.to_xarray()

    da = ds[score_col]

    dims = [bootstrap_col, seed_col, task_col, model_col]
    da = da.transpose(*dims)

    output = da.to_masked_array() if masked else da.values

    if masked:
        output = np.ma.masked_invalid(output)
    else:
        output = np.nan_to_num(output, nan=1e-10)

    return output


def infer_parameters(replications):
    _, _, T, M = replications.shape

    overall_means_hat = replications.reshape(-1, M).mean(axis=0)
    task_means_hat = replications.reshape(-1, T, M).mean(axis=0)

    between_task_sd_hat = replications.reshape(-1, T, M).mean(axis=0).std(axis=0)

    seed_sd_hat = replications.mean(axis=0).std(axis=0)

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
        masked=True,
    )
    inferred = infer_parameters(replications)

    out = out.copy()
    out["sd_seed"] = inferred["SD (seed)"].flatten()
    out["sd_boot"] = inferred["SD (boot)"].flatten()
    out["sd_repl"] = np.sqrt(out["sd_seed"] ** 2 + out["sd_boot"] ** 2)

    return out
