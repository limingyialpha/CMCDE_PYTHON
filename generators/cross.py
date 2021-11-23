from generators.dataGenerator import DataGenerator
import numpy as np


class Cross(DataGenerator):
    name = "cross"

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            pivot = np.random.random()
            return np.array([pivot if (np.random.randint(2) == 0) else 1-pivot for i in range(self.dim)])
        return np.array([get_point() for _ in range(n)])
