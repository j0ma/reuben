import json
from pathlib import Path
from collections import defaultdict

from rich.console import Console
import click
from click.core import ParameterSource
import yaml

from reuben.resampling import ReplicationResamplingMethod, TaskResamplingMethod


def bundle_decorators(decorators):
    def combined(function):
        for decorator in reversed(decorators):
            function = decorator(function)

        return function

    return combined


def output_path_args(function):
    deco = bundle_decorators(
        [
            click.option(
                "--output-format",
                type=click.Choice(["rich", "json", "csv"]),
                default="rich",
            ),
            click.option(
                "--output-path", type=click.Path(), help="Output file/directory path"
            ),
            click.option(
                "--pickle-output-folder",
                type=click.Path(file_okay=False, path_type=Path),
                help="Path to dump leaderboard and simulations in .pkl format",
            ),
        ]
    )
    return deco(function)


def fix_outliers_arg(function):
    deco = bundle_decorators(
        [
            click.option(
                "--fix-outliers",
                help="Fix outlier SDs with winsorization (i.e. truncating extreme SD values)",
                is_flag=True,
            )
        ]
    )
    return deco(function)


def data_path_arg(function):
    deco = bundle_decorators([click.argument("data_path")])
    return deco(function)


def task_and_repl_resampling_options(function):
    decorator = bundle_decorators(
        [
            click.option(
                "--task-resampling-method",
                type=click.Choice(TaskResamplingMethod),
                default=TaskResamplingMethod.none,
            ),
            click.option("--task-resampling-with-replacement", is_flag=True),
            click.option(
                "--task-resampling-num-tasks",
                "-T",
                type=int,
                help="Number of tasks to sample",
                default=None,
            ),
            click.option(
                "--replication-resampling-method",
                type=click.Choice(ReplicationResamplingMethod),
                default=ReplicationResamplingMethod.none,
            ),
            click.option(
                "--num-bootstrap-resamples",
                "-B",
                type=int,
                help="Number of bootstrap resamples to draw",
                default=0,
            ),
        ]
    )
    return decorator(function)


def score_model_task_options(function):
    decorator = bundle_decorators(
        [
            click.option("--score-col", default="Mean"),
            click.option("--model-col", default="Model"),
            click.option("--task-col", default="Task"),
        ]
    )
    return decorator(function)


def idx_col_options(function):
    decorator = bundle_decorators(
        [
            click.option("--replication-idx-col", default=None),
            click.option("--seed-idx-col", default=None),
            click.option("--boot-idx-col", default=None),
        ]
    )
    return decorator(function)


def sd_col_options(function):
    decorator = bundle_decorators(
        [
            click.option("--replication-sd-col", default=None),
            click.option("--seed-sd-col", default=None),
            click.option("--boot-sd-col", default=None),
        ]
    )
    return decorator(function)


def analysis_pipeline_component_args(function):
    decorator = bundle_decorators(
        [
            click.option("--aggregate-analysis", is_flag=True),
            click.option("--variance-components", is_flag=True),
            click.option("--pairwise-diffs", is_flag=True),
            click.option("--pairwise-diffs-task-level-details", is_flag=True),
        ]
    )
    return decorator(function)


def existing_leaderboard_pickle_arg(function):
    decorator = bundle_decorators(
        [
            click.option("--leaderboard-pkl", type=click.Path(dir_okay=False)),
        ]
    )
    return decorator(function)


def command_with_all_args(function):
    @data_path_arg
    @analysis_pipeline_component_args
    @task_and_repl_resampling_options
    @score_model_task_options
    @sd_col_options
    @idx_col_options
    @output_path_args
    @fix_outliers_arg
    @existing_leaderboard_pickle_arg
    @click.pass_context
    def wrapper(ctx, **kwargs):
        return function(ctx, **kwargs)

    return wrapper


# NON OPTIONS


def load_config_dict(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        return yaml.safe_load(open(path, "r")) or {}
    if path.endswith(".json"):
        return json.load(open(path, "r")) or {}
    raise click.BadParameter(f"Unsupported config file type: {path}")


def merge_params_with_config(ctx):
    cfg = ctx.obj.get("cfg", {})
    debug_mode = ctx.obj.get("debug_mode", False)
    merged = {}

    taken_from = defaultdict(list)

    for param in ctx.command.params:
        name = param.name
        value = ctx.params.get(name)
        src = ctx.get_parameter_source(name)

        if src == ParameterSource.COMMANDLINE:
            merged[name] = value
            taken_from["command_line"].append(name)
        elif name in cfg:
            merged[name] = cfg[name]
            taken_from["config_file"].append(name)
        elif src == ParameterSource.DEFAULT:
            merged[name] = value
            taken_from["default"].append(name)
        else:
            merged[name] = value
            taken_from["other"].append(name)

    if debug_mode:
        console = Console(stderr=True)
        console.print("[yellow]Sources of parameters:[/yellow]")
        console.print(json.dumps(taken_from, indent=2, ensure_ascii=False))

    return merged
