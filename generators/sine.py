from parameterizedDataGenerator import ParameterizedDataGenerator
import numpy as np


class Sine(ParameterizedDataGenerator):
    name = "sine"

    def __init__(self, dim, noise, param=1):
        super().__init__(dim, noise, param)

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            pivot = np.random.random()
            data = np.zeros(self.dim)
            data[0] = pivot
            period = self.param
            for i in range(1, self.dim):
                data[i] = (np.sin(data.sum() * 2 * np.pi * period) + 1) / 2
            return data

        return np.array([get_point() for _ in range(n)])
