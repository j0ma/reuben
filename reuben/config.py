from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, field_validator

from reuben.resampling import ReplicationResamplingMethod, TaskResamplingMethod


@dataclass
class Config:
    # Core data columns
    score_col: str = "Mean"
    model_col: str = "Model"
    task_col: str = "Task"

    # Standard deviation columns (optional)
    replication_sd_col: Optional[str] = None
    seed_sd_col: Optional[str] = None
    boot_sd_col: Optional[str] = None

    # Index columns (optional)
    replication_idx_col: Optional[str] = None
    seed_idx_col: Optional[str] = None
    boot_idx_col: Optional[str] = None

    # Task resampling options (use enums)
    task_resampling_method: TaskResamplingMethod = TaskResamplingMethod.none
    task_resampling_with_replacement: bool = False
    task_resampling_num_tasks: Optional[int] = None

    # Replication resampling options (use enums)
    replication_resampling_method: ReplicationResamplingMethod = (
        ReplicationResamplingMethod.none
    )
    num_bootstrap_resamples: int = 0

    # Analysis options
    standardized: bool = False
    rounding: int = 3

    # Output options
    output_format: str = "rich"
    output_path: Optional[Path] = None
    pickle_output_folder: Optional[Path] = None

    @classmethod
    def from_cli_args(cls, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if "output_path" in filtered_kwargs:
            filtered_kwargs["output_path"] = Path(filtered_kwargs["output_path"])
        if "pickle_output_folder" in filtered_kwargs:
            filtered_kwargs["pickle_output_folder"] = Path(
                filtered_kwargs["pickle_output_folder"]
            )
        return cls(**filtered_kwargs)


class ConfigModel(BaseModel):
    score_col: str = "Mean"
    model_col: str = "Model"
    task_col: str = "Task"

    replication_sd_col: Optional[str] = None
    seed_sd_col: Optional[str] = None
    boot_sd_col: Optional[str] = None

    replication_idx_col: Optional[str] = None
    seed_idx_col: Optional[str] = None
    boot_idx_col: Optional[str] = None

    task_resampling_method: TaskResamplingMethod = TaskResamplingMethod.none
    task_resampling_with_replacement: bool = False
    task_resampling_num_tasks: Optional[int] = None

    replication_resampling_method: ReplicationResamplingMethod = (
        ReplicationResamplingMethod.none
    )
    num_bootstrap_resamples: int = 0

    standardized: bool = False
    rounding: int = 3

    output_format: Literal["rich", "json", "csv"] = "rich"
    output_path: Optional[Union[str, Path]] = None
    pickle_output_folder: Optional[Union[str, Path]] = None

    @field_validator("task_resampling_method", mode="before")
    @classmethod
    def _coerce_task_method(cls, v):
        return TaskResamplingMethod(v) if isinstance(v, str) else v

    @field_validator("replication_resampling_method", mode="before")
    @classmethod
    def _coerce_repl_method(cls, v):
        return ReplicationResamplingMethod(v) if isinstance(v, str) else v

    def to_config(self) -> "Config":
        return Config.from_cli_args(**self.model_dump())
