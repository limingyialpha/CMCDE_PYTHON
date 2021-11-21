from dataGenerator import DataGenerator
import numpy as np


# the uniform distribution is the key
class HyperSphere(DataGenerator):
    name = "hypersphere"

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            raw = np.random.normal(0, 1, self.dim)
            radius = np.sqrt(np.sum(raw ** 2))
            return raw / (2 * radius) + 0.5

        return np.array([get_point() for _ in range(n)])
