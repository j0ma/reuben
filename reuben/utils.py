from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.table import Table

from reuben.analysis import (
    compute_replication_suff_stats,
    compute_task_mean_and_std,
    standardize_data,
    winsorize,
)


def is_convertible_to_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def fisher_transform(arr: np.typing.NDArray) -> np.typing.NDArray:
    return np.arctanh(arr)


def inverse_fisher_transform(arr: np.typing.NDArray) -> np.typing.NDArray:
    return np.tanh(arr)


def estimate_correlation_ci_with_fisher_transform(
    arr: np.typing.NDArray,
    rounding: int = 2,
) -> np.typing.NDArray:
    n_models, _ = arr.shape

    model2model_corr_transf = fisher_transform(arr)
    model2model_corr_trasnf_std = 1 / np.sqrt(max(1, n_models - 3))
    model2model_corr_transf_lower = (
        model2model_corr_transf - 1.96 * model2model_corr_trasnf_std
    )
    model2model_corr_transf_upper = (
        model2model_corr_transf + 1.96 * model2model_corr_trasnf_std
    )
    model2model_corr_lower = inverse_fisher_transform(
        model2model_corr_transf_lower
    ).round(rounding)
    model2model_corr_upper = inverse_fisher_transform(
        model2model_corr_transf_upper
    ).round(rounding)

    return transform_lower_and_upper_to_ci(
        lower=model2model_corr_lower, upper=model2model_corr_upper, rounding=rounding
    )


def transform_lower_and_upper_to_ci(lower, upper, rounding: int = 2):
    out = create_lower_upper_ci_array(
        lower=lower,
        upper=upper,
        rounding=rounding,
    )
    np.fill_diagonal(out, "")
    return np.triu(out)[:-1, 1:]


def nested_convert_defaultdict_to_dict(dd):
    if isinstance(dd, dict):
        return {k: nested_convert_defaultdict_to_dict(v) for k, v in dd.items()}
    elif isinstance(dd, list):
        return [nested_convert_defaultdict_to_dict(v) for v in dd]
    else:
        return dd


def justify_minus_sign(x):
    if x.startswith("-"):
        return x
    else:
        return " " + x


def get_suffix(boolean: bool, suffix: str = " (*)"):
    if boolean:
        return "", suffix
    else:
        return "", ""


def apply_no_styling(boolean: bool):
    return "", ""


def format_min_max_columns(
    df: pd.DataFrame,
    items: List[Tuple[str, str, str]],
    rounding: int = 2,
) -> pd.DataFrame:
    out = {}
    for out_col, min_col, max_col in items:
        if min_col in df.columns and max_col in df.columns:
            left = df[min_col].round(rounding).astype(str)
            right = df[max_col].round(rounding).astype(str)
            out[out_col] = left + " - " + right
    return pd.DataFrame(out, index=df.index)


def build_varcomp_display(
    summary: pd.DataFrame,
    model_col: str,
    rounding: int = 2,
) -> tuple[list[dict], list[str], dict]:
    df = summary.copy()

    ranges = format_min_max_columns(
        df,
        items=[
            ("seed_sd", "seed_sd_min", "seed_sd_max"),
            ("boot_sd", "boot_sd_min", "boot_sd_max"),
            ("repl_sd", "repl_sd_min", "repl_sd_max"),
            ("boot_to_seed_ratio", "boot_to_seed_ratio_min", "boot_to_seed_ratio_max"),
        ],
        rounding=rounding,
    )

    between = df["between_sd"].round(rounding).astype(str)

    visible = pd.DataFrame(index=df.index)
    visible[model_col] = df[model_col].astype(str)
    visible["between_sd"] = between
    for col in ["seed_sd", "boot_sd", "repl_sd", "boot_to_seed_ratio"]:
        if col in ranges.columns:
            visible[col] = ranges[col]
        else:
            visible[col] = "-"

    if "boot_seed_corr" in df.columns:
        corr = df["boot_seed_corr"].round(rounding).astype(str)
        visible["boot_seed_corr"] = corr

    columns = [
        model_col,
        "between_sd",
        "repl_sd",
        "seed_sd",
        "boot_sd",
        "boot_to_seed_ratio",
        "boot_seed_corr",
    ]
    pretty = {
        model_col: "Model",
        "between_sd": "SD (between) - ν",
        "repl_sd": "SD (within) - η",
        "seed_sd": "SD (seed) - σ",
        "boot_sd": "SD (boot) - τ",
        "boot_to_seed_ratio": "Ratio - τ/σ",
        "boot_seed_corr": "Corr[τ, σ]",
    }

    return visible.to_dict(orient="records"), columns, pretty


def create_lower_upper_ci_array(
    lower,
    upper,
    rounding=2,
    postprocess_fn=get_suffix,
):
    contains_zero = (lower <= 0) & (0 <= upper)
    excludes_zero = ~contains_zero

    str_lower = lower.round(rounding).astype(str)
    str_upper = upper.round(rounding).astype(str)

    out_raw = "[" + str_lower + ", " + str_upper + "]"

    postprocess = np.vectorize(postprocess_fn)
    prefix, suffix = postprocess(excludes_zero)
    out = prefix + out_raw + suffix

    return out


def create_mean_plus_sd_array(
    means,
    sds,
    rounding=2,
    sd_multiplier=1,
    postprocess_fn=get_suffix,
):
    plusminus_sign = "±"

    moe = sd_multiplier * sds
    lb = means - moe
    ub = means + moe
    contains_zero = (lb <= 0) & (0 <= ub)
    excludes_zero = ~contains_zero

    str_mean = means.round(rounding).astype(str)
    str_moe = moe.round(rounding).astype(str)

    out_raw = str_mean + f" {plusminus_sign} " + str_moe

    postprocess = np.vectorize(postprocess_fn)
    prefix, suffix = postprocess(excludes_zero)
    out = prefix + out_raw + suffix

    return out


def grab_data_and_preprocess(
    data_path: Union[str, Path],
    score_col: str,
    model_col: str,
    task_col: str,
    replication_sd_col: str,
    seed_sd_col: str,
    boot_sd_col: str,
    replication_idx_col: str,
    seed_idx_col: str,
    boot_idx_col: str,
    standardized: bool,
    return_raw_data: bool = False,
    fix_outliers: bool = False,
) -> tuple[pd.DataFrame, str, str, str, Optional[pd.DataFrame]]:
    idx_column_names = [seed_idx_col, boot_idx_col, replication_idx_col]

    if any([col is not None for col in idx_column_names]):
        mode = "must-compute-sd"
    elif replication_sd_col is not None:
        mode = "replication-sd-provided"
    elif seed_sd_col is not None and boot_sd_col is not None:
        mode = "compute-overall-sd"
    else:
        raise ValueError("Must provide either SD columns or replication indices!")

    data = load_data(data_path)
    raw_data = data.copy()

    orig_replication_sd_col = replication_sd_col

    if mode == "must-compute-sd":
        data = compute_replication_suff_stats(
            data=data,
            score_col=score_col,
            model_col=model_col,
            task_col=task_col,
            replication_idx_col=replication_idx_col,
            seed_idx_col=seed_idx_col,
            boot_idx_col=boot_idx_col,
        ).reset_index()
        seed_sd_col = "sd_seed"
        boot_sd_col = "sd_boot"
        replication_sd_col = "sd_repl"
    elif mode == "compute-overall-sd":
        replication_sd_col = "SD_repl"
        data[replication_sd_col] = np.sqrt(
            data[boot_sd_col] ** 2 + data[seed_sd_col] ** 2
        )

    score_stats = compute_task_mean_and_std(
        data, score_col=score_col, task_col=task_col
    )
    if standardized:
        sd_cols_not_none = [
            c for c in [seed_sd_col, boot_sd_col, replication_sd_col] if c is not None
        ]
        data = standardize_data(
            data=data,
            stats=score_stats,
            score_col=score_col,
            score_sd_cols=sd_cols_not_none,
        )

    data = data[~data.isna().any(axis=1)]

    if fix_outliers:
        for sdcol in [seed_sd_col, boot_sd_col, replication_sd_col]:
            if sdcol is not None and sdcol in data:
                winsorized_sd, _ = winsorize(data[sdcol], method="iqr")
                data[sdcol] = winsorized_sd

    if mode == "must-compute-sd":
        data = data.rename(
            columns={replication_sd_col: orig_replication_sd_col or replication_sd_col}
        )

    return (
        data,
        seed_sd_col,
        boot_sd_col,
        orig_replication_sd_col
        if mode == "must-compute-sd" and orig_replication_sd_col is not None
        else replication_sd_col,
        (raw_data if return_raw_data else None),
    )


def make_rich_table(data, title, subset=None, columns=None, column_renames=None):
    if columns is None:
        columns = data[subset][0]

    if column_renames is None:
        column_renames = {}

    table = Table(title=title, min_width=80)
    for column in columns:
        table.add_column(column_renames.get(column, column))

    for row_dict in data[subset]:
        row = [str(row_dict.get(col, "-")) for col in columns]
        row = [f"{float(s):.2f}" if is_convertible_to_float(s) else s for s in row]
        table.add_row(*row)

    return table


def load_json(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def load_csv_tsv(path: Union[str, Path], sep=",") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in [".json", ".jsonl"]:
        out = load_json(path)
    elif path.suffix == ".csv":
        out = load_csv_tsv(path, sep=",")
    elif path.suffix == ".tsv":
        out = load_csv_tsv(path, sep="\t")
    else:
        raise TypeError(f"Bad file extension: {path}")

    return out
