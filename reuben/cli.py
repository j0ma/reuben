import pickle
from pathlib import Path

import click
from rich.console import Console

import reuben.analysis_pipeline as ap
from reuben.cli_utils import (
    data_path_arg,
    fix_outliers_arg,
    idx_col_options,
    load_config_dict,
    merge_params_with_config,
    score_model_task_options,
    sd_col_options,
    task_and_repl_resampling_options,
    output_path_args,
)
from reuben.config import ConfigModel
from reuben.output import (
    AggregateAnalysisOutputFormatter as agg_out,
    VarianceComponentOutputFormatter as var_comp_out,
)
from reuben.resampling import ReplicationResamplingMethod, TaskResamplingMethod
from reuben.utils import (
    grab_data_and_preprocess,
    nested_convert_defaultdict_to_dict,
)


@click.group(context_settings={"show_default": True})
@click.option("--config-file", type=click.Path(exists=True), help="YAML/JSON config")
@click.pass_context
def main_cli(ctx, config_file):
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config_dict(config_file) if config_file else {}


@main_cli.command()
@data_path_arg
@score_model_task_options
@sd_col_options
@idx_col_options
@click.option("--rounding", help="Rounding precision", default=3)
@output_path_args
@fix_outliers_arg
@click.pass_context
def variance_components(
    ctx,
    **_,
):
    params = merge_params_with_config(ctx)
    pydantic_config_model = ConfigModel(
        **{k: v for k, v in params.items() if k in ConfigModel.model_fields}
    )
    config = pydantic_config_model.to_config()

    data, seed_sd_col, boot_sd_col, replication_sd_col, raw_data = (
        grab_data_and_preprocess(
            data_path=params["data_path"],
            score_col=config.score_col,
            model_col=config.model_col,
            task_col=config.task_col,
            replication_sd_col=config.replication_sd_col,
            seed_sd_col=config.seed_sd_col,
            boot_sd_col=config.boot_sd_col,
            replication_idx_col=config.replication_idx_col,
            seed_idx_col=config.seed_idx_col,
            boot_idx_col=config.boot_idx_col,
            standardized=False,
            return_raw_data=True,
            fix_outliers=params["fix_outliers"],
        )
    )

    # Update Config with new SD cols if necessary
    config.seed_sd_col = seed_sd_col
    config.boot_sd_col = boot_sd_col
    config.replication_sd_col = replication_sd_col

    # Variance components summary
    summary = ap.variance_components_summary(data, config, rounding=config.rounding)
    results = var_comp_out.format_results_for_output(summary, [], config.model_col)
    var_comp_out.handle_output(results, config, {})


@main_cli.command()
@data_path_arg
@task_and_repl_resampling_options
@score_model_task_options
@sd_col_options
@idx_col_options
@click.option("--standardized", is_flag=True)
@click.option("--rounding", help="Rounding precision", default=3)
@output_path_args
@fix_outliers_arg
@click.pass_context
def compare_models_aggregate(
    ctx,
    **_,
):
    params = merge_params_with_config(ctx)
    pydantic_config_model = ConfigModel(
        **{k: v for k, v in params.items() if k in ConfigModel.model_fields}
    )
    config = pydantic_config_model.to_config()

    data, seed_sd_col, boot_sd_col, replication_sd_col, raw_data = (
        grab_data_and_preprocess(
            params["data_path"],
            config.score_col,
            config.model_col,
            config.task_col,
            config.replication_sd_col,
            config.seed_sd_col,
            config.boot_sd_col,
            config.replication_idx_col,
            config.seed_idx_col,
            config.boot_idx_col,
            config.standardized,
            return_raw_data=True,
            fix_outliers=params.get("fix_outliers", False),
        )
    )

    # Update SD column names in config post-preprocessing
    config.seed_sd_col = seed_sd_col
    config.boot_sd_col = boot_sd_col
    config.replication_sd_col = replication_sd_col

    console = Console()

    leaderboard_task_means, orig_results, metadata = ap.compute_base_statistics(
        data, raw_data, config
    )

    if TaskResamplingMethod.is_not_none(
        config.task_resampling_method
    ) or ReplicationResamplingMethod.is_not_none(config.replication_resampling_method):
        leaderboard = ap.apply_resampling(
            leaderboard_task_means, raw_data, config, metadata
        )
    else:
        console.print("No resampling requested, computing raw results...")
        leaderboard = leaderboard_task_means[..., None]

    stats = ap.compute_all_statistics(
        leaderboard, orig_results, metadata["uniq_models"], config
    )
    results = agg_out.format_results_for_output(
        stats, metadata["uniq_models"], config.model_col
    )
    agg_out.handle_output(results, config, metadata)

    if config.pickle_output_folder:
        stats = nested_convert_defaultdict_to_dict(stats)
        pickle_output_folder = Path(config.pickle_output_folder)
        pickle_output_folder.mkdir(exist_ok=True, parents=True)
        with open(pickle_output_folder / "stats.pkl", "wb") as fout:
            console.print(
                f"[green][italic]Dumping statistics pickle to {pickle_output_folder / 'stats.pkl'}[/italic][/green]"
            )
            pickle.dump(stats, fout)
        with open(pickle_output_folder / "leaderboard.pkl", "wb") as fout:
            console.print(
                f"[green][italic]Dumping leaderboard pickle to {pickle_output_folder / 'leaderboard.pkl'}[/italic][/green]"
            )
            pickle.dump(leaderboard, fout)


if __name__ == "__main__":
    main_cli()
