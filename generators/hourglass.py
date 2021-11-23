from generators.dataGenerator import DataGenerator
import numpy as np


class Hourglass(DataGenerator):
    name = "hourglass"

    def get_points(self, n: int) -> np.ndarray:
        def get_next_dim_value(head):
            toss = np.random.randint(4)
            value = 0
            if toss == 0:
                value = 0
            elif toss == 1:
                value = head
            elif toss == 2:
                value = 1 - head
            else:
                value = 1
            return value

        def get_point():
            pivot = np.random.random()
            return np.array([get_next_dim_value(pivot) if (i >= 2) else pivot for i in range(1, self.dim+1)])
        return np.array([get_point() for _ in range(n)])
