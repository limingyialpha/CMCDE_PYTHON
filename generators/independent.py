from dataGenerator import DataGenerator
import numpy as np


class Independent(DataGenerator):
    name = "independent"

    def get_points(self, n: int) -> np.ndarray:
        return np.random.random((n, self.dim))
