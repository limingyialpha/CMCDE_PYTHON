from generators.parameterizedDataGenerator import ParameterizedDataGenerator
import numpy as np


class DoubleLinear(ParameterizedDataGenerator):
    name = "doublelinear"

    def __init__(self, dim, noise, param=0.25):
        super().__init__(dim, noise, param)

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            pivot = np.random.random()
            raw_points = np.repeat(pivot, self.dim)
            coef = self.param
            mask = np.array([1 if (i >= 2 and np.random.randint(2) == 0) else coef for i in range(1, self.dim + 1)])
            return raw_points * mask

        return np.array([get_point() for _ in range(n)])
