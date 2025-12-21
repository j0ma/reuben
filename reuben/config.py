from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel

from reuben.resampling import ReplicationResamplingMethod, TaskResamplingMethod


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

    rounding: int = 2

    output_format: Literal["rich", "json", "csv"] = "rich"
    output_path: Optional[Union[str, Path]] = None
    pickle_output_folder: Optional[Union[str, Path]] = None
