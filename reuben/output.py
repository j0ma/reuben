from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule

import reuben.aggregators as agg
from reuben.utils import build_varcomp_display, justify_minus_sign, make_rich_table


class OutputFormatter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def format_results_for_output(*args, **kwargs):
        raise NotImplementedError

    @classmethod
    def output_rich_tables(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def output_csv(cls, results, output_dir):
        raise NotImplementedError

    @classmethod
    def handle_output(cls, results, config, metadata):
        raise NotImplementedError


class VarianceComponentOutputFormatter(OutputFormatter):
    @staticmethod
    def format_results_for_output(summary_df, per_task_df, model_col):
        return {
            "summary": summary_df.to_dict(orient="records"),
            "per_task": per_task_df.to_dict(orient="records"),
            "model_col": model_col,
        }

    @staticmethod
    def _compute_pretty(results, rounding: int):
        df = pd.DataFrame(results["summary"])
        rows, columns, pretty = build_varcomp_display(
            df, results["model_col"], rounding
        )
        return rows, columns, pretty

    @classmethod
    def output_rich_tables(cls, results, uniq_models, config, console):
        rows, columns, pretty = cls._compute_pretty(results, config.rounding)
        table = make_rich_table(
            data={"varcomp": rows},
            subset="varcomp",
            title="Variance Components (summary)",
            columns=columns,
            column_renames=pretty,
        )
        console.print(table)

    @classmethod
    def output_csv(cls, results, output_dir):
        d = Path(output_dir)
        for _subfolder in ["aggregate-analysis", "detailed"]:
            subfolder = d / _subfolder
            subfolder.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(results["summary"]).to_csv(
            d / "aggregate-analysis" / "variance_components_summary.csv", index=False
        )
        pd.DataFrame(results["per_task"]).to_csv(
            d / "detailed" / "variance_components_per_task_plus_results.csv",
            index=False,
        )

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "csv" and config.output_path:
            cls.output_csv(results, config.output_path)
        else:
            cls.output_rich_tables(
                results, metadata.get("uniq_models", []), config, console
            )


class AggregateAnalysisOutputFormatter(OutputFormatter):
    @staticmethod
    def format_results_for_output(stats, uniq_models, model_col):
        results = {}

        aggregated_results = pd.DataFrame(
            stats["agg_over_task"]["pretty"], index=uniq_models
        ).reset_index(names=model_col)
        results["aggregated_results"] = aggregated_results.to_dict(orient="records")

        for agg_name in stats["pairwise_diffs"]["effect_size"]:
            prettified = stats["ranks"]["crosstab"][agg_name]
            prettified["vals"] = prettified.apply(
                lambda row: "".join([str(x) for x in row]), axis=1
            )
            prettified = prettified.sort_values("vals", ascending=False)
            prettified = prettified.drop(columns="vals")
            prettified = prettified.reset_index(names=["Model"])
            prettified.columns = [str(c) for c in prettified.columns]
            results[f"ranks_{agg_name}"] = prettified.to_dict(orient="records")

        for agg_name, df in stats["pairwise_diffs"]["pretty"].items():
            results[f"pairwise_diffs_{agg_name}"] = (
                df.reset_index(names=["Model"])
                .map(justify_minus_sign)
                .to_dict(orient="records")
            )

        for agg_name, df in stats["pairwise_diffs"]["effect_size"].items():
            results[f"effect_size_{agg_name}"] = (
                df.reset_index(names=["Model"])
                .map(justify_minus_sign)
                .to_dict(orient="records")
            )

        return results

    @classmethod
    def output_rich_tables(cls, results, uniq_models, config, console):
        first_row = (
            results["aggregated_results"][0] if results["aggregated_results"] else {}
        )
        cols = [config.model_col] + [
            c for c in first_row.keys() if c != config.model_col
        ]

        ordered_cols = [
            c
            for c in cols
            if c in {config.model_col, "arithmetic_mean", "geometric_mean", "median"}
        ]

        agg_table_col_renames = {
            c: agg.get_pretty_name(c) for c in list(agg.AGGREGATOR_PRETTYNAME_REGISTRY)
        }

        table = make_rich_table(
            data=results,
            subset="aggregated_results",
            title="Aggregators",
            columns=ordered_cols,
            column_renames=agg_table_col_renames,
        )
        console.print(table)

        agg_names = []
        for k in list(results.keys()):
            if k.startswith("pairwise_diffs_"):
                agg_names.append(k.replace("pairwise_diffs_", ""))

        for agg_name in agg_names:
            subset = f"ranks_{agg_name}"
            if subset in results:
                table = make_rich_table(
                    data=results,
                    subset=subset,
                    title=f"Rank distribution ({agg.get_pretty_name(agg_name)})",
                    columns=None,
                    column_renames={
                        config.model_col: "",
                        **agg.AGGREGATOR_PRETTYNAME_REGISTRY,
                    },
                )
                console.print(table)

        for agg_name in agg_names:
            subset = f"pairwise_diffs_{agg_name}"
            if subset in results:
                table = make_rich_table(
                    data=results,
                    subset=subset,
                    title=f"Pairwise diffs ({agg.get_pretty_name(agg_name)})",
                    columns=None,
                    column_renames={
                        config.model_col: "",
                        **agg.AGGREGATOR_PRETTYNAME_REGISTRY,
                    },
                )
                console.print(table)

        for agg_name in agg_names:
            subset = f"effect_size_{agg_name}"
            if subset in results:
                table = make_rich_table(
                    data=results,
                    subset=subset,
                    title=f"Effect size ({agg.get_pretty_name(agg_name)})",
                    columns=None,
                    column_renames={
                        config.model_col: "",
                        **agg.AGGREGATOR_PRETTYNAME_REGISTRY,
                    },
                )
                console.print(table)

    @classmethod
    def output_csv(cls, results, output_dir):
        orig_output_dir = Path(output_dir)
        output_dir = orig_output_dir / "aggregate-analysis"
        output_dir.mkdir(exist_ok=True, parents=True)

        metrics = ["arithmetic_mean", "geometric_mean", "median"]
        output_types = ["pairwise_diffs", "effect_size", "ranks"]

        metric_name_output_type_tuples = [("", "aggregated_results")] + list(
            product(metrics, output_types)
        )
        for metric_name, output_type in metric_name_output_type_tuples:
            if output_type == "aggregated_results":
                key = output_type
                output_path = output_dir / f"{output_type}.csv"
            else:
                key = f"{output_type}_{metric_name}"
                output_path = output_dir / metric_name / f"{output_type}.csv"

            if key not in results:
                console = Console()
                console.print(
                    f"Warning: {key} not found in results, skipping CSV output."
                )

            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results[key]).to_csv(output_path, index=False)

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "csv" and config.output_path:
            cls.output_csv(results, config.output_path)
        else:
            cls.output_rich_tables(results, metadata["uniq_models"], config, console)


class PairwiseDiffOutputFormatter:
    @staticmethod
    def format_results_for_output(summary, details):
        return {"summary": summary, "details": details}

    @classmethod
    def output_rich_tables(
        cls,
        results,
        config,
        console,
    ):
        pretty_rows = []
        for row in results["summary"]:
            pretty_rows.append(
                {
                    "ModelA": row["ModelA"],
                    "ModelB": row["ModelB"],
                    "n_tasks": int(row["n_tasks"]),
                    "mean": row["mean"],
                    "predictive_sd": row["predictive_sd"],
                    "between_sd": row["between_sd"],
                    "within_sd_mean": row["within_sd_mean"],
                    "within_sd_sd": row["within_sd_sd"],
                }
            )

        table = make_rich_table(
            data={"summary": pretty_rows},
            subset="summary",
            title="Pairwise diff variance components",
            columns=[
                "ModelA",
                "ModelB",
                "n_tasks",
                "mean",
                "predictive_sd",
                "between_sd",
                "within_sd_mean",
                "within_sd_sd",
            ],
            column_renames={
                "ModelA": "Model A",
                "ModelB": "Model B",
                "n_tasks": "T (# Tasks)",
                "mean": "E[Δ]",
                "predictive_sd": "SD[Δ]",
                "between_sd": "ν (between-SD)",
                "within_sd_mean": "E[η] (avg. within-SD)",
                "within_sd_sd": "SD[η] (variability)",
            },
        )
        console.print(table)

    @classmethod
    def output_csv(cls, results, output_dir):
        d = Path(output_dir)
        for _subfolder in ["aggregate-analysis", "detailed"]:
            subfolder = d / _subfolder
            subfolder.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(results["summary"])
        summary_df = summary_df[
            [
                "ModelA",
                "ModelB",
                "n_tasks",
                "mean",
                "predictive_sd",
                "between_sd",
                "within_sd_mean",
                "within_sd_sd",
            ]
        ].rename(columns={
                "ModelA": "Model A",
                "ModelB": "Model B",
                "n_tasks": "T (# Tasks)",
                "mean": "E[Δ]",
                "predictive_sd": "SD_pred[Δ]",
                "between_sd": "ν (between-SD)",
                "within_sd_mean": "E[η] (avg. within-SD)",
                "within_sd_sd": "SD[η] (variability)",
        })

        summary_df.to_csv(
            d / "aggregate-analysis" / "variance_components_pairwise_diffs.csv",
            index=False,
        )

        details_rows = []
        for (a, b), det in results["details"].items():
            for outcome, lst in [
                ("A_beats_B", det["A_beats_B"]),
                ("B_beats_A", det["B_beats_A"]),
                ("inconclusive", det["inconclusive"]),
                ("outliers", det["outliers"]),
            ]:
                for item in lst:
                    details_rows.append(
                        {
                            "ModelA": a,
                            "ModelB": b,
                            "Task": item["task"],
                            "Result": outcome,
                            "Mean": item["mean"],
                            "StdErr": item["std_err_mean"],
                        }
                    )

        details_df = pd.DataFrame(details_rows)
        details_df.to_csv(d / "detailed" / "pairwise_diff_details.csv", index=False)

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "csv" and config.output_path:
            cls.output_csv(results, config.output_path)
        else:
            cls.output_rich_tables(
                results,
                config,
                console,
            )
