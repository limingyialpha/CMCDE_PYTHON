from dataGenerator import DataGenerator
import numpy as np


class Zinv(DataGenerator):
    name = "zinv"

    def get_points(self, n: int) -> np.ndarray:
        def add_uniform_noise(x: int, noise_value: float):
            return x + np.random.uniform(-noise_value/2, noise_value/2)

        def get_next_dim_value(head):
            toss = np.random.randint(3)
            value = 0
            if toss == 0:
                value = 0
            elif toss == 1:
                value = 1
            elif toss == 2:
                value = 1 - head
            return value

        def get_point():
            pivot = np.random.random()
            no_noise = np.array([get_next_dim_value(pivot) if (i >= 2) else pivot for i in range(1, self.dim+1)])
            return np.array([add_uniform_noise(x, self.noise) for x in no_noise])
        return np.array([get_point() for _ in range(n)])
