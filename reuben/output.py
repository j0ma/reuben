import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from rich.console import Console

import reuben.aggregators as agg
from reuben.utils import justify_minus_sign, make_rich_table, build_varcomp_display


class OutputFormatter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def format_results_for_output(stats, uniq_models, model_col):
        raise NotImplementedError

    @classmethod
    def output_rich_tables(cls, results, uniq_models, config, console):
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
    def format_results_for_output(stats, uniq_models, model_col):
        if isinstance(stats, pd.DataFrame):
            summary_rows = stats.to_dict(orient="records")
        elif isinstance(stats, list):
            summary_rows = stats
        else:
            raise TypeError("stats must be a DataFrame or list[dict].")
        return {"summary": summary_rows, "model_col": model_col}

    @staticmethod
    def _compute_pretty(results, rounding: int):
        df = pd.DataFrame(results["summary"])
        model_col = results.get("model_col", "Model")
        rows, columns, pretty = build_varcomp_display(df, model_col, rounding=rounding)
        return rows, columns, pretty

    @classmethod
    def output_rich_tables(cls, results, uniq_models, config, console):
        rows, columns, pretty = cls._compute_pretty(results, rounding=config.rounding)
        table = make_rich_table(
            data={"varcomp": rows},
            subset="varcomp",
            title="Variance Components",
            columns=columns,
            column_renames=pretty,
        )
        console.print(table)

    @staticmethod
    def output_json(results, output_path):
        payload = results
        with open(output_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def output_csv(cls, results, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        if "varcomp" in results:
            pd.DataFrame(results["varcomp"]).to_csv(
                output_dir / "variance_components.csv", index=False
            )
        else:
            pd.DataFrame(results["summary"]).to_csv(
                output_dir / "variance_components_numeric.csv", index=False
            )

    @classmethod
    def handle_output(cls, results, config, metadata):
        console = Console()
        rows, columns, pretty = cls._compute_pretty(results, rounding=config.rounding)
        prepared = {
            "varcomp": rows,
            "varcomp_columns": columns,
            "varcomp_pretty": pretty,
            **results,
        }

        if config.output_format == "json" and config.output_path:
            cls.output_json(prepared, config.output_path)
        elif config.output_format == "csv" and config.output_path:
            cls.output_csv(prepared, config.output_path)
        else:
            cls.output_rich_tables(
                prepared, metadata.get("uniq_models", []), config, console
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
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    @classmethod
    def output_csv(cls, results, output_dir):
        output_dir = Path(output_dir)
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
