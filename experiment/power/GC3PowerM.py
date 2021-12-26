import logging
import multiprocessing
import sys
from functools import partial
from random import random

import numpy as np
from multiprocessing import Pool
import socket
from datetime import datetime

from experiment import experiment
from generators import *

from generators import linearPeriodic
from generators.independent import Independent
from stats import gc

"""
Compare the power of competitors in general case with other GMCDE.
The general case we are looking at is:
Dependency between 3 groups of dimensions.
GMCDE is implemented in Python. See partner Repo.
We look at different observation numbers, dimensions, noise levels,
symmetric data distributions of all kinds
"""


class GC3PowerM(experiment.Experiment):

    def __init__(self, output_folder):
        super().__init__(output_folder)
        self.noises_of_interest = [round(i / self.noise_levels, self.noise_precision) for i in
                                   range(0, self.noise_levels + 1)]

    # data specific params
    gens = [
        linear.Linear,
        partial(doubleLinear.DoubleLinear, param=0.25),
        partial(linearPeriodic.LinearPeriodic, param=2),
        partial(sine.Sine, param=1),
        partial(sine.Sine, param=5),
        hypercube.Hypercube,
        hypercubeGraph.HypercubeGraph,
        hypersphere.HyperSphere,
        cross.Cross,
        star.Star,
        hourglass.Hourglass,
        zinv.Zinv,
        Independent,
        partial(groupsOfLinears.GroupsOfLinears, param=3)
    ]
    dimensions_of_interest = [6, 9, 12, 15]
    noise_levels = 30
    noise_precision = 2
    observation_num_of_interest = [100, 1000]

    # measure specific params
    measures = ["dHSIC"]

    # methodology specific params
    power_computation_iteration_num = 500
    benchmark_iteration_num = 10000
    level_of_parallelism = multiprocessing.cpu_count() - 1

    def run(self):
        logger = logging.getLogger(self.class_name)
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger.info(f"{now} - Starting experiments - {type(self).__name__}")

        logger.info("Data specific params:")
        gen_names = [gen(2, 0).name for gen in self.gens]
        logger.info(f"generators of interest for both symmetric and asymmetric distributions: {gen_names}")
        logger.info(f"dimensions of interest: {self.dimensions_of_interest}")
        logger.info(f"noise levels: {self.noise_levels}")
        logger.info(f"observation numbers of interest: {self.observation_num_of_interest}")

        logger.info("Dependency measure specific params:")
        logger.info(f"Generalized Contrast measures: {self.measures}")

        logger.info("Methodology specific params:")
        logger.info(f"number of iterations for power computation: {self.power_computation_iteration_num}")
        logger.info(f"number of iterations for benchmark: {self.benchmark_iteration_num}")
        logger.info(f"Level of parallelism: {self.level_of_parallelism}")

        logger.info(f"Started on {socket.gethostname()}")

        summary_header = ["genId", "dim", "noise", "obs_num", "measure", "avg_gc", "std_gc", "power90", "power95",
                          "power99"]
        self.write_summary_header(summary_header)

        for measure in self.measures:
            for obs_num in self.observation_num_of_interest:
                for dim in self.dimensions_of_interest:
                    # computing threshold with uniform distributions
                    logger.info(
                        f"now computing thresholds for measure: {measure}, observation number: {obs_num}, dimension: {dim}")
                    with Pool(processes=self.level_of_parallelism) as pool:
                        task_inputs = [(measure, obs_num, dim)
                                       for _ in range(self.benchmark_iteration_num)]
                        results = pool.starmap(self.benchmark_task, task_inputs)
                    threshold90 = self.percentile_scala_breeze(results, 0.90)
                    threshold95 = self.percentile_scala_breeze(results, 0.95)
                    threshold99 = self.percentile_scala_breeze(results, 0.99)
                    logger.info(
                        f"finished computing thresholds for measure: {measure}, observation number: {obs_num}, dimension: {dim}")

                    # computing the symmetric data set
                    with Pool(processes=self.level_of_parallelism) as pool:
                        task_inputs = [(measure, obs_num, dim, noise,
                                        threshold90, threshold95, threshold99) for noise in self.noises_of_interest]
                        pool.starmap(self.symmetric_task, task_inputs)

        logger.info(f"{now} - Finished experiments - {type(self).__name__}")

    def benchmark_task(self, measure: str, obs_num: int, dim: int):
        set_of_dims_1st = frozenset(range(0, int(dim / 3)))
        set_of_dims_2nd = frozenset(range(int(dim / 3), int(dim / 3 * 2)))
        set_of_dims_3nd = frozenset(range(int(dim / 3 * 2), int(dim)))
        sets = {set_of_dims_1st, set_of_dims_2nd, set_of_dims_3nd}
        benchmark_gen_ins = Independent(dim, 0)
        data = benchmark_gen_ins.generate(obs_num)
        return gc.generalized_contrast(measure, data, sets)

    def symmetric_task(self, measure: str, obs_num: int, dim: int, noise: float,
                       t90: float, t95: float, t99: float):
        # This block of code is included to deal with logging with multiprocessing
        # We instantiate a new logger for each new process
        # the start of the block
        logger = logging.getLogger(str(random()))
        logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s {self.class_name} - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        f_handler = logging.FileHandler(self.log_path)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s {self.class_name} - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
        # the end of the block

        logger.info(
            f"now dealing with gens: symmetric, measure: {measure}, observation number: {obs_num}, dimension: {dim}, noise {noise}")
        set_of_dims_1st = frozenset(range(0, int(dim / 3)))
        set_of_dims_2nd = frozenset(range(int(dim / 3), int(dim / 3 * 2)))
        set_of_dims_3nd = frozenset(range(int(dim / 3 * 2), int(dim)))
        sets = {set_of_dims_1st, set_of_dims_2nd, set_of_dims_3nd}
        # to avoid some stupid warnings in dcor, we take 0 as 0.0001
        stub = 0.0001
        noise = stub if noise == 0 else noise
        for gen in self.gens:
            gen_instance = gen(dim, noise)
            results = np.zeros(self.power_computation_iteration_num)
            for rep in range(self.power_computation_iteration_num):
                data = gen_instance.generate(obs_num)
                results[rep] = gc.generalized_contrast(measure, data, sets)
            power90 = sum([r > t90 for r in results]) / len(results)
            power95 = sum([r > t95 for r in results]) / len(results)
            power99 = sum([r > t99 for r in results]) / len(results)
            avg_cc = np.mean(results)
            std_cc = np.std(results)
            # in writing , we need to write 0.0001 back to 0
            noise_to_write = 0 if noise == stub else noise
            summary_content = [gen_instance.get_id(), dim, noise_to_write, obs_num, measure, avg_cc, std_cc,
                               power90,
                               power95,
                               power99]
            self.write_summary_content(summary_content)
