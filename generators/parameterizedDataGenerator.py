from generators.dataGenerator import DataGenerator
import matplotlib.pyplot as plt


class ParameterizedDataGenerator(DataGenerator):

    def __init__(self, dim: int, noise: float, param: float):
        super().__init__(dim, noise)
        self.param = param

    def get_id(self):
        return f"{self.name}-{self.get_short_name()}_{self.param}-{self.dim}-{self.noise}"

    @classmethod
    def test_plot_param(cls, param: float, n: int):
        # 2d
        plt.figure(figsize=(12, 12))
        gen = cls(2, 0, param)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("2d")
        plt.show()

        # 3d
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        gen = cls(3, 0, param)
        data = gen.generate(n)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.title("3d")
        plt.show()

        # 2d with noise 0.05
        plt.figure(figsize=(12, 12))
        gen = cls(2, 0.05, param)
        data = gen.generate(n)
        plt.scatter(data[:, 0], data[:, 1])
        plt.title("2d noise 0.05")
        plt.show()

        # 3d with noise 0.05
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        gen = cls(3, 0.05, param)
        data = gen.generate(n)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        plt.title("3d noise 0.05")
        plt.show()

