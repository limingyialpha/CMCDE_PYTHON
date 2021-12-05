from generators.parameterizedDataGenerator import ParameterizedDataGenerator
import numpy as np


class Parabola(ParameterizedDataGenerator):
    name = "parabola"

    def __init__(self, dim, noise, param=1):
        super().__init__(dim, noise, param)

    def get_points(self, n: int) -> np.ndarray:

        def noiseless_power_normalization(x, power):
            empty = np.zeros(self.dim)
            pivot = 1.0
            empty[0] = pivot
            for i in range(1, self.dim):
                empty[i] = np.power(empty.sum(), power)
            maxi = empty[-1]
            return x / maxi

        def get_point():
            scale = self.param
            data = np.zeros(self.dim)
            pivot = np.random.uniform(-1, 1)
            data[0] = pivot
            power = 2 + (2 * (scale - 1))
            for i in range(1, self.dim):
                data[i] = np.power(sum(data), power)
            return noiseless_power_normalization(data, power)

        return np.array([get_point() for _ in range(n)])
