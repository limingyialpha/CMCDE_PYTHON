from generators.parameterizedDataGenerator import ParameterizedDataGenerator
import numpy as np


class GroupsOfLinears(ParameterizedDataGenerator):
    name = "groups_of_linears"

    def __init__(self, dim, noise, param):
        super().__init__(dim, noise, param)
        self.num_groups = int(param)
        assert self.dim % self.num_groups == 0, "The number of dimension should be multiple of number of groups."
        self.dims_in_each_group = self.dim / self.num_groups

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            values_of_each_group = [np.random.random() for _ in range(self.num_groups)]
            return [values_of_each_group[int(i / self.dims_in_each_group)] for i in range(0, self.dim)]

        return np.array([get_point() for _ in range(n)])
