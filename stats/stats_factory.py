from typing import Set

from hsic import hsic
import dcor
import numpy as np

test_names = {"hsic", "dcor"}


def test(test_name: str, data: np.ndarray, x: Set[int], y: Set[int], threshold: float = 0.05) -> bool:
    if test_name == "hsic":
        return hsic_test(data, x, y, threshold)
    elif test_name == "dcor":
        return dcor_test(data, x, y, threshold)
    else:
        raise Exception("Wrong test name!")


def cc(test_name: str, data: np.ndarray, x: Set[int], y: Set[int]) -> bool:
    if test_name == "hsic":
        return hsic_cc(data, x, y)
    elif test_name == "dcor":
        return dcor_cc(data, x, y)
    else:
        raise Exception("Wrong test name!")


def hsic_test(data: np.ndarray, x: Set[int], y: Set[int], threshold: float = 0.05) -> bool:
    data_x = data[:, list(x)]
    data_y = data[:, list(y)]
    (testStat, thresh) = hsic.hsic_gam(data_x, data_y, alph=threshold)
    return testStat > thresh


def hsic_cc(data: np.ndarray, x: Set[int], y: Set[int]) -> bool:
    data_x = data[:, list(x)]
    data_y = data[:, list(y)]
    (testStat, thresh) = hsic.hsic_gam(data_x, data_y, alph=0.05)
    return testStat


def dcor_test(data: np.ndarray, x: Set[int], y: Set[int], threshold: float = 0.05) -> bool:
    data_x = data[:, list(x)]
    data_y = data[:, list(y)]
    return dcor.independence.distance_correlation_t_test(data_x, data_y).p_value < threshold


def dcor_cc(data: np.ndarray, x: Set[int], y: Set[int]) -> bool:
    data_x = data[:, list(x)]
    data_y = data[:, list(y)]
    return dcor.independence.distance_correlation_t_test(data_x, data_y).statistic
