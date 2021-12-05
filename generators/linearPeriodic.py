from generators.parameterizedDataGenerator import ParameterizedDataGenerator
import numpy as np


class LinearPeriodic(ParameterizedDataGenerator):
    name = "linearperiodic"

    def __init__(self, dim, noise, param=2):
        super().__init__(dim, noise, param)

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            pivot = np.random.random()
            period = self.param
            data = np.zeros(self.dim)
            data[0] = pivot
            for i in range(1, self.dim):
                data[i] = (data[i-1] % (1/period)) * period
            return data
        return np.array([get_point() for _ in range(n)])
