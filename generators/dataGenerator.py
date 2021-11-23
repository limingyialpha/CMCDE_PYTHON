import numpy as np

from generators import generatorFactory
import matplotlib.pyplot as plt


class DataGenerator:
    name = "data_generator"

    def __init__(self, dim: int, noise: float):
        self.dim = dim
        self.noise = noise

    @classmethod
    def get_short_name(cls):
        return generatorFactory.correspondances.get(cls.name)

    def get_id(self):
        return f"{self.name}-{self.get_short_name()}-{self.dim}-{self.noise}"

    def get_points(self, n: int) -> np.ndarray:
        pass

    # noise is actually standard deviation
    # in scala CMCDE, the noise is also standard deviation
    # as a result, we need to quad it to get variance and cov matrix
    def generate(self, n: int) -> np.ndarray:
        mean = np.zeros(self.dim)
        var = self.noise ** 2
        cov = np.diag(np.repeat(var, self.dim))
        noises = np.random.multivariate_normal(mean, cov, n)
        return self.get_points(n) + noises

    @classmethod
    def test_plot(cls, n: int):
        # 2d
        plt.figure(figsize=(12, 12))
        gen = cls(2, 0)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("2d")
        plt.show()

        # 3d
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        gen = cls(3, 0)
        data = gen.generate(n)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.title("3d")
        plt.show()

        # 2d with noise 0.05
        plt.figure(figsize=(12, 12))
        gen = cls(2, 0.05)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("2d noise 0.05")
        plt.show()

        # 3d with noise 0.05
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        gen = cls(3, 0.05)
        data = gen.generate(n)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.title("3d noise 0.05")
        plt.show()


