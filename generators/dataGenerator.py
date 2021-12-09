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
    def test_plot_2d(cls, n: int, noise: float):
        plt.figure(figsize=(12, 12))
        gen = cls(2, noise)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("2d")
        plt.show()

    @classmethod
    def test_plot_3d(cls, n: int, noise: float):
        plt.figure(figsize=(12, 12))
        gen = cls(3, noise)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.title("3d")
        plt.show()


