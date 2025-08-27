import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np
from rich.console import Console
from rich.rule import Rule

from reuben.utils import make_rich_table, justify_minus_sign

import reuben.aggregators as agg
from reuben.utils import build_varcomp_display


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

    @staticmethod
    def output_json(results, output_path):
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
        console.print("[dim](Full per-task dump available via JSON / CSV.)[/dim]")

    @staticmethod
    def output_json(results, output_path):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

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
            d / "detailed" / "variance_components_per_task.csv", index=False
        )

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "json" and config.output_path:
            cls.output_json(results, config.output_path)
        elif config.output_format == "csv" and config.output_path:
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

        results["corr_matrix_model2model"] = (
            stats["correlation_matrices"]["model2model"]["pretty"]
            .reset_index(names=["Model"])
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

        if config.standardized:
            ordered_cols = [
                c for c in cols if c in {config.model_col, "arithmetic_mean", "median"}
            ]
        else:
            ordered_cols = [
                c
                for c in cols
                if c
                in {config.model_col, "arithmetic_mean", "geometric_mean", "median"}
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

        should_use_correlations = False
        if "corr_matrix_model2model" in results and should_use_correlations:
            table = make_rich_table(
                data=results,
                subset="corr_matrix_model2model",
                title="CI for correlation matrix (model2model)",
                columns=[config.model_col, *uniq_models[1:]],
                column_renames={
                    config.model_col: "",
                    **agg.AGGREGATOR_PRETTYNAME_REGISTRY,
                },
            )
            console.print(table)

    @staticmethod
    def output_json(results, output_path):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @classmethod
    def output_csv(cls, results, output_dir):
        orig_output_dir = Path(output_dir)
        output_dir = orig_output_dir / "aggregate-analysis"
        output_dir.mkdir(exist_ok=True, parents=True)

        keys_to_output_as_csv = [
            "aggregated_results",
            "pairwise_diffs_arithmetic_mean",
            "pairwise_diffs_geometric_mean",
            "pairwise_diffs_median",
            "effect_size_arithmetic_mean",
            "effect_size_geometric_mean",
            "effect_size_median",
            "ranks_arithmetic_mean",
            "ranks_geometric_mean",
            "ranks_median",
            "corr_matrix_model2model",
        ]
        for key in keys_to_output_as_csv:
            if key not in results:
                console = Console()
                console.print(
                    f"Warning: {key} not found in results, skipping CSV output."
                )

            pd.DataFrame(results[key]).to_csv(output_dir / f"{key}.csv", index=False)

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "json" and config.output_path:
            cls.output_json(results, config.output_path)
        elif config.output_format == "csv" and config.output_path:
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
        *,
        show_details=True,
    ):
        pretty_rows = []
        for row in results["summary"]:
            pretty_rows.append(
                {
                    "Model_A": row["Model_A"],
                    "Model_B": row["Model_B"],
                    "n_tasks": int(row["n_tasks"]),
                    "delta": f"{row['mean']:.2f} ± {row['predictive_sd'] / np.sqrt(row['n_tasks']):.2f}",
                    "between_sd": row["between_sd"],
                    "within_sd_mean": row["within_sd_mean"],
                }
            )

        table = make_rich_table(
            data={"summary": pretty_rows},
            subset="summary",
            title="Pairwise diff variance components",
            columns=[
                "Model_A",
                "Model_B",
                "n_tasks",
                "delta",
                "between_sd",
                "within_sd_mean",
                "within_sd_sd",
            ],
            column_renames={
                "Model_A": "Model A",
                "Model_B": "Model B",
                "n_tasks": "# Tasks",
                # "delta": "E[Δ] +/- SD[Δ]",
                "delta": "E[Δ] ± SD[E[Δ]]",
                "between_sd": "ν (between-SD)",
                "within_sd_mean": "E[η] (avg. within-SD)",
                "within_sd_sd": "SD[η] (variability)",
            },
        )
        console.print(table)

        if not show_details:
            return

        for (a, b), det in results["details"].items():
            console.print(Rule(f"[bold]{a} vs {b}[/bold]", style="bright_white"))
            for label, color, key in [
                ("A beats B", "green", "A_beats_B"),
                ("B beats A", "red", "B_beats_A"),
                ("Inconclusive", "yellow", "inconclusive"),
                ("Outliers", "magenta", "outliers"),
            ]:
                tasks = det[key]
                header = f"[{color}]{label}:[/{color}]"
                body = "\n".join(f"  • {t}" for t in tasks) if tasks else "  –"
                console.print(f"{header}\n{body}\n")

    @staticmethod
    def output_json(results, output_path):
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @classmethod
    def output_csv(cls, results, output_dir):
        d = Path(output_dir)
        for _subfolder in ["aggregate-analysis", "detailed"]:
            subfolder = d / _subfolder
            subfolder.mkdir(parents=True, exist_ok=True)

        # Summary output as CSV
        summary_df = pd.DataFrame(results["summary"])
        summary_df.to_csv(
            d / "aggregate-analysis" / "variance_components_pairwise_diffs.csv",
            index=False,
        )

        # Details output as CSV,
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
                            "Model_A": a,
                            "Model_B": b,
                            "Task": item["task"],
                            "Result": outcome,
                            "Mean": item["mean"],
                            "StdErr": item["std_err_mean"],
                        }
                    )

        details_df = pd.DataFrame(details_rows)
        details_df.to_csv(
            d / "detailed" / "pairwise_diff_details.csv", index=False
        )

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        if config.output_format == "json" and config.output_path:
            cls.output_json(results, config.output_path)
        elif config.output_format == "csv" and config.output_path:
            cls.output_csv(results, config.output_path)
        else:
            cls.output_rich_tables(
                results,
                config,
                console,
                show_details=metadata.get("show_details", True),
            )
