from typing import Set

from dhsic import dhsic
from rdc import rdc
import dcor
import numpy as np

measures = {"hsic", "dcor", "rdc"}


def canonical_correlation(measure: str, data: np.ndarray, set_of_dims_1st: Set[int], set_of_dims_2nd: Set[int]) -> bool:
    if measure == "hsic":
        return hsic_cc(data, set_of_dims_1st, set_of_dims_2nd)
    elif measure == "dcor":
        return dcor_cc(data, set_of_dims_1st, set_of_dims_2nd)
    elif measure == "rdc":
        return rdc_cc(data, set_of_dims_1st, set_of_dims_2nd)
    else:
        raise Exception("Wrong test name!")


def hsic_cc(data: np.ndarray, set_of_dims_1st: Set[int], set_of_dims_2nd: Set[int]) -> bool:
    data_x = data[:, list(set_of_dims_1st)]
    data_y = data[:, list(set_of_dims_2nd)]
    return dhsic.dHSIC(data_x, data_y)


def dcor_cc(data: np.ndarray, set_of_dims_1st: Set[int], set_of_dims_2nd: Set[int]) -> bool:
    data_x = data[:, list(set_of_dims_1st)]
    data_y = data[:, list(set_of_dims_2nd)]
    return dcor.distance_correlation(data_x, data_y)


def rdc_cc(data: np.ndarray, set_of_dims_1st: Set[int], set_of_dims_2nd: Set[int]) -> bool:
    data_x = data[:, list(set_of_dims_1st)]
    data_y = data[:, list(set_of_dims_2nd)]
    return rdc.rdc(data_x, data_y)

