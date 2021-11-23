from generators.dataGenerator import DataGenerator
import numpy as np


class Star(DataGenerator):
    name = "star"

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            pivot = np.random.random()
            data = np.array([pivot if np.random.randint(2) == 0 else 1 - pivot for i in range(1, self.dim+1)])
            if np.random.randint(2) == 0:
                dim = np.random.randint(self.dim)
                data[dim] = 0.5
            return data
        return np.array([get_point() for _ in range(n)])
