from typing import Set

from dhsic import dhsic
import numpy as np

test_names = {"dhsic"}


def general_dependency(estimator: str, data: np.ndarray, sets_of_dimensions: Set[Set[int]]) -> bool:
    if estimator == "dhsic":
        return dhsic_gd(data, sets_of_dimensions)
    else:
        raise Exception("Wrong test name!")


def dhsic_gd(data: np.ndarray, sets_of_dimensions: Set[Set[int]]) -> bool:
    list_of_sub_data = [data[:, list(set_of_dims)] for set_of_dims in sets_of_dimensions]
    return dhsic.dHSIC(*list_of_sub_data)

