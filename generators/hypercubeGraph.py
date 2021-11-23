from generators.dataGenerator import DataGenerator
import numpy as np


class HypercubeGraph(DataGenerator):
    name = "hypercubegraph"

    def get_points(self, n: int) -> np.ndarray:
        def get_point():
            # the value along this dimension is uniform distributed thus forming an edge
            edge = np.random.randint(1, self.dim + 1)
            return np.array([np.random.randint(2) if (i != edge) else np.random.random() for i in range(1, self.dim + 1)])
        return np.array([get_point() for _ in range(n)])
