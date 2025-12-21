import pickle
from pathlib import Path

import click
from rich.console import Console

from reuben.analysis_pipeline import run_analysis
from reuben.cli_utils import (
    command_with_all_args,
    load_config_dict,
    merge_params_with_config,
)
from reuben.config import ConfigModel
from reuben.output import (
    AggregateAnalysisOutputFormatter as agg_output_formatter,
    PairwiseDiffOutputFormatter as pdiff_output_formatter,
    VarianceComponentOutputFormatter as vc_output_formatter,
)
from reuben.utils import nested_convert_defaultdict_to_dict

console = Console()


@click.group(context_settings={"show_default": True})
@click.option("--config-file", type=click.Path(exists=True), help="YAML/JSON config")
@click.option("--debug-mode", envvar="REUBEN_DEBUG_MODE", is_flag=True, hidden=True)
@click.pass_context
def main_cli(ctx, config_file, debug_mode):
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config_dict(config_file) if config_file else {}
    ctx.obj["debug_mode"] = debug_mode


@main_cli.command(name="analyze")
@click.option("--skip-param-merge", is_flag=True, default=False, hidden=True)
@command_with_all_args
def analyze(
    ctx,
    aggregate_analysis,
    pairwise_diffs,
    variance_components,
    pairwise_diffs_task_level_details,
    skip_param_merge,
    **_,
):
    if not skip_param_merge:
        params = merge_params_with_config(ctx)
    else:
        params = ctx.params

    config = ConfigModel(
        **{k: v for k, v in params.items() if k in ConfigModel.model_fields}
    )

    results = run_analysis(
        data_path=params["data_path"],
        config=config,
        include_pairwise=pairwise_diffs,
        include_varcomp=variance_components,
        pairwise_diffs_task_level_details=pairwise_diffs_task_level_details,
        fix_outliers=params.get("fix_outliers", False),
        leaderboard_pkl=params.get("leaderboard_pkl", None),
    )

    agg_results = agg_output_formatter.format_results_for_output(
        results["aggregate"], results["metadata"]["uniq_models"], config.model_col
    )
    if aggregate_analysis:
        agg_output_formatter.handle_output(agg_results, config, results["metadata"])

    if variance_components:
        vc_output_formatter.handle_output(
            results["varcomp"], config, results["metadata"]
        )

    if pairwise_diffs:
        pdiff_output_formatter.handle_output(
            results["pairwise"],
            config,
            {
                "uniq_models": results["metadata"]["uniq_models"],
                "show_details": pairwise_diffs_task_level_details,
            },
        )

    if config.pickle_output_folder:
        Path(config.pickle_output_folder).mkdir(parents=True, exist_ok=True)
        with open(Path(config.pickle_output_folder) / "analysis.pkl", "wb") as f:
            pickle.dump(
                nested_convert_defaultdict_to_dict(results),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


if __name__ == "__main__":
    main_cli()
