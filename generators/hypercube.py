from generators.dataGenerator import DataGenerator
import numpy as np


class Hypercube(DataGenerator):
    name = "hypercube"

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            # the value on this dimension does not change and is either 0 or 1
            face = np.random.randint(1, self.dim + 1)
            return np.array([np.random.random() if (i != face) else np.random.randint(2) for i in range(1, self.dim + 1)])
        return np.array([get_point() for _ in range(n)])
