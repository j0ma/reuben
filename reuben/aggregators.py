#!/usr/bin/env python

from typing import Union

import numpy as np
import pandas as pd

AGGREGATOR_REGISTRY = {}
AGGREGATOR_PRETTYNAME_REGISTRY = {
    "arithmetic_mean_numpy": "Arithmetic mean",
    "geometric_mean_numpy": "Geometric mean",
    "median_numpy": "Median",
    "sd_numpy": "SD",
    "arithmetic_mean": "Arithmetic mean",
    "geometric_mean": "Geometric mean",
    "median": "Median",
    "sd": "SD",
}


# Decorator to register functions in the aggregator registry
def register_aggregator(name, pretty_name=None):
    def actual_register_method(f):
        AGGREGATOR_REGISTRY[name] = f
        if isinstance(name, str) and name not in AGGREGATOR_PRETTYNAME_REGISTRY:
            AGGREGATOR_PRETTYNAME_REGISTRY[name] = pretty_name
        return f

    return actual_register_method


def apply_aggregator(name: str, arr: Union[pd.Series, np.ndarray], **kwargs):
    agg_func = AGGREGATOR_REGISTRY[name]
    return agg_func(arr, **kwargs)


def get_pretty_name(name: str) -> str:
    return AGGREGATOR_PRETTYNAME_REGISTRY.get(name, name)


@register_aggregator("arithmetic_mean_numpy")
def arithm_mean_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmean(arr, axis=axis)


@register_aggregator("geometric_mean_numpy")
def geom_mean_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.exp(np.nanmean(np.log(arr), axis=axis))


@register_aggregator("median_numpy")
def median_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.nanmedian(arr, axis=axis)


@register_aggregator("sd_numpy")
def sd_numpy(arr: np.ndarray, axis: int) -> np.ndarray:
    return np.nanstd(arr, axis=axis)
