import multiprocessing
from functools import partial

import numpy as np
from multiprocessing import Pool
import itertools

from typing import Set

from experiment import experiment_template
from generators import *
import csv
import logging

from generators.independent import Independent
from stats import cc


class CanonicalCorrelationPowerCompare(experiment_template.Experiment):
    dimensions_of_interest = [4, 8, 12, 16]
    noise_level = 30
    noise_precision = 2
    observation_num_of_interest = [100, 1000]
    power_computation_iteration_num = 500
    gens = [
        linear.Linear,
        cross.Cross,
        partial(doubleLinear.DoubleLinear, param=0.25),
        hourglass.Hourglass,
        hypercube.Hypercube,
        hypercubeGraph.HypercubeGraph,
        hypersphere.HyperSphere,
        partial(parabola.Parabola, param=1),
        partial(sine.Sine, param=1),
        partial(sine.Sine, param=5),
        star.Star,
        zinv.Zinv
    ]
    threshold = 0.05
    estimator_names = {"hsic", "dcor", "rdc"}
    level_of_parallelism = multiprocessing.cpu_count() - 1

    def symmetric_task(self, estimator: str, obs_num: int, dim: int, noise: float,
                       t90: float, t95: float, t99: float):
        # call the config again, so that we are sure that, even if we are in a subprocess,
        # that the information will be logged centrally
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {type(self).__name__} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        self.info(
            f"now dealing with gens: symmetric, estimator: {estimator}, observation number: {obs_num}, dimension: {dim}, noise {noise}")
        set_of_dims_1st = set(range(0, int(dim / 2)))
        set_of_dims_2nd = set(range(int(dim / 2), dim))
        # to avoid some stupid warnings in dcor, we take 0 as 0.0001
        stub = 0.0001
        noise = stub if noise == 0 else noise
        for gen in self.gens:
            gen_instance = gen(dim, noise)
            results = np.zeros(self.power_computation_iteration_num)
            for rep in range(self.power_computation_iteration_num):
                data = gen_instance.generate(obs_num)
                results[rep] = cc.canonical_correlation(estimator, data, set_of_dims_1st, set_of_dims_2nd)
            power90 = sum([r > t90 for r in results]) / len(results)
            power95 = sum([r > t95 for r in results]) / len(results)
            power99 = sum([r > t99 for r in results]) / len(results)
            avg_cc = np.mean(results)
            std_cc = np.std(results)
            # in writing , we need to write 0.0001 back to 0
            noise_to_write = 0 if noise == stub else noise
            summary_content = [gen_instance.get_id(), dim, noise_to_write, obs_num, estimator, avg_cc, std_cc, power90,
                               power95,
                               power99]
            with open(self.summary_path, mode='a', newline='') as summary_file:
                summary_writer = csv.writer(summary_file, delimiter=',')
                summary_writer.writerow(summary_content)

    def asymmetric_task(self, estimator: str, obs_num: int, dim: int, noise: float,
                        t90: float, t95: float, t99: float):
        # call the config again, so that we are sure that, even if we are in a subprocess,
        # that the information will be logged centrally
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {type(self).__name__} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        self.info(
            f"now dealing with gens: asymmetric, estimator: {estimator}, observation number: {obs_num}, dimension: {dim}, noise {noise}")
        # to avoid some stupid warnings in dcor, we take 0 as 0.0001
        stub = 0.0001
        noise = stub if noise == 0 else noise
        for gen in self.gens:
            gen_instance = gen(int(dim / 2), noise)
            asymmetry = Independent(int(dim / 2), noise)
            results = np.zeros(self.power_computation_iteration_num)
            for rep in range(self.power_computation_iteration_num):
                data_sy = gen_instance.generate(obs_num)
                data_asy = asymmetry.generate(obs_num)
                data = np.concatenate((data_sy, data_asy), axis=1)
                set_of_dims_1st = set(range(0, int(dim / 4))).union(range(int(dim / 2), int(dim / 4 * 3)))
                set_of_dims_2nd = set(range(0, dim)) - set_of_dims_1st
                results[rep] = cc.canonical_correlation(estimator, data, set_of_dims_1st, set_of_dims_2nd)
            power90 = sum([r > t90 for r in results]) / len(results)
            power95 = sum([r > t95 for r in results]) / len(results)
            power99 = sum([r > t99 for r in results]) / len(results)
            avg_cc = np.mean(results)
            std_cc = np.std(results)
            # in writing , we need to write 0.0001 back to 0
            noise_to_write = 0 if noise == stub else noise
            summary_content = [gen_instance.get_id() + "_asy", dim, noise_to_write, obs_num, estimator, avg_cc, std_cc, power90,
                               power95,
                               power99]
            with open(self.summary_path, mode='a', newline='') as summary_file:
                summary_writer = csv.writer(summary_file, delimiter=',')
                summary_writer.writerow(summary_content)

    def benchmark_task(self, estimator: str, obs_num: int, dim: int):
        set_of_dims_1st = set(range(0, int(dim / 2)))
        set_of_dims_2nd = set(range(int(dim / 2), dim))
        benchmark_gen_ins = Independent(dim, 0)
        data = benchmark_gen_ins.generate(obs_num)
        return cc.canonical_correlation(estimator, data, set_of_dims_1st, set_of_dims_2nd)

    def run(self):
        self.info("Parameters:")
        self.info(f"Canonical Correlation estimators: {self.estimator_names}")
        self.info(f"dimensions of interest: {self.dimensions_of_interest}")
        self.info(f"noise levels: {self.noise_level}")
        self.info(f"observation numbers of interest: {self.observation_num_of_interest}")
        gen_names = [gen(2, 0).name for gen in self.gens]
        self.info(f"generators of interest: {gen_names}")
        self.info(f"number of iterations for power computation: {self.power_computation_iteration_num}")

        summary_header = ["genId", "dim", "noise", "obs_num", "estimator", "avg_cc", "std_cc", "power90", "power95",
                          "power99"]
        self.write_summary_header(summary_header)

        for estimator in self.estimator_names:
            for obs_num in self.observation_num_of_interest:
                for dim in self.dimensions_of_interest:
                    # computing threshold with uniform distributions
                    self.info(
                        f"now computing thresholds for estimator: {estimator}, observation number: {obs_num}, dimension: {dim}")
                    with Pool(processes=self.level_of_parallelism) as pool:
                        task_inputs = [(estimator, obs_num, dim)
                                       for _ in range(self.power_computation_iteration_num)]
                        results = pool.starmap(self.benchmark_task, task_inputs)
                    threshold90 = self.percentile_scala_breeze(results, 0.90)
                    threshold95 = self.percentile_scala_breeze(results, 0.95)
                    threshold99 = self.percentile_scala_breeze(results, 0.99)
                    self.info(
                        f"finished computing thresholds for estimator: {estimator}, observation number: {obs_num}, dimension: {dim}")

                    # # computing the symmetric data set
                    # with Pool(processes=self.level_of_parallelism) as pool:
                    #     noises = [round(i / self.noise_level, self.noise_precision) for i in
                    #               range(0, self.noise_level + 1)]
                    #     task_inputs = [(estimator, obs_num, dim, noise,
                    #                     threshold90, threshold95, threshold99) for noise in noises]
                    #     pool.starmap(self.symmetric_task, task_inputs)

                    # computing the asymmetric data set
                    with Pool(processes=self.level_of_parallelism) as pool:
                        noises = [round(i / self.noise_level, self.noise_precision) for i in
                                  range(0, self.noise_level + 1)]
                        task_inputs = [(estimator, obs_num, dim, noise,
                                        threshold90, threshold95, threshold99) for noise in noises]
                        pool.starmap(self.asymmetric_task, task_inputs)
