from dataGenerator import DataGenerator
import numpy as np


class Linear(DataGenerator):
    name = "linear"

    def get_points(self, n: int) -> np.ndarray:
        return np.repeat(np.random.random((n, 1)), self.dim, axis=1)
