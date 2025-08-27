from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class TaskResamplingConfig:
    num_tasks: int = field(default=-1)
    fraction: float = field(default=-1)
    replace: bool = field(default=False)
    subsample: bool = field(default=False)

    @property
    def should_resample(self):
        return bool(self.num_tasks) or bool(self.fraction)


class TaskResamplingMethod(Enum):
    none = "none"
    parametric = "parametric"
    nonparametric = "nonparametric"

    @classmethod
    def is_none(cls, method: str):
        return method == cls.none

    @classmethod
    def is_not_none(cls, method: str):
        return not cls.is_none(method)


class ReplicationResamplingMethod(Enum):
    none = "none"
    parametric = "parametric"
    nonparametric = "nonparametric"

    @classmethod
    def is_none(cls, method: "ReplicationResamplingMethod"):
        return method == cls.none

    @classmethod
    def is_not_none(cls, method: "ReplicationResamplingMethod"):
        return not cls.is_none(method)


def resample_residual_given_sds(
    standard_deviations: pd.Series, num_samples: int = 1
) -> NDArray:
    return np.random.normal(
        loc=0, scale=standard_deviations, size=(num_samples, *standard_deviations.shape)
    )
