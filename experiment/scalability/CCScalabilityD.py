import logging
import multiprocessing

import numpy as np
from multiprocessing import Pool
import socket
from datetime import datetime

from experiment import experiment
from generators import *

from generators.dataGenerator import DataGenerator
from stats import cc

from utils import stopwatch

"""
This experiment analyse the scalability(CPU time) of different canonical correlation measures,
with respect to dimensions.
Only GMCDE is in scala, others are in Python partner repo.
We look at Independent Uniform distribution.
Each group has equal number of dimensions 
Observation number is 1000.
"""


class CCScalabilityD(experiment.Experiment):

    def __init__(self, output_folder):
        super().__init__(output_folder)

    # data specific params
    gen = linear.Linear
    noise = 0
    dimensions_of_interest = [2, 4, 6, 8, 10, 12, 14]
    observation_num = 1000

    # measure specific params
    measures = {"RDC", "dCor", "dHSIC"}

    # methodology specific params
    repetitions = 10000
    level_of_parallelism = multiprocessing.cpu_count() - 1
    unit = "ms"

    def run(self):
        logger = logging.getLogger(self.class_name)
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger.info(f"{now} - Starting experiments - {type(self).__name__}")

        logger.info("Data specific params:")
        logger.info(f"generator: {self.gen.name}")
        logger.info(f"dimensions of interest: {self.dimensions_of_interest}")
        logger.info(f"observation number: {self.observation_num}")

        logger.info("Dependency measure specific params:")
        logger.info(f"Canonical Correlation measures: {self.measures}")

        logger.info("Methodology specific params:")
        logger.info(f"number of repetitions: {self.repetitions}")
        logger.info(f"Level of parallelism: {self.level_of_parallelism}")
        logger.info(f"unit of scalability (cpu time): {self.unit}")

        logger.info(f"Started on {socket.gethostname()}")

        summary_header = ["measure", "dim", "avg_cpu_time"]
        self.write_summary_header(summary_header)
        for measure in self.measures:
            for dim in self.dimensions_of_interest:
                logger.info(f"now dealing with measure: {measure}, dimension: {dim}")
                gen_ins = self.gen(dim, self.noise)
                set_of_dims_1st = set(range(0, int(dim / 2)))
                set_of_dims_2nd = set(range(int(dim / 2), dim))
                with Pool(processes=self.level_of_parallelism) as pool:
                    task_inputs = [(gen_ins, measure, set_of_dims_1st, set_of_dims_2nd) for _ in
                                   range(0, self.repetitions)]
                    cpu_times = pool.starmap(self.task, task_inputs)
                avg_cpu_time = np.mean(cpu_times)
                summary_content = [measure, dim, avg_cpu_time]
                self.write_summary_content(summary_content)

        logger.info(f"{now} - Finished experiments - {type(self).__name__}")

    def task(self, gen_ins: DataGenerator, measure, set_of_dims_1st, set_of_dims_2nd):
        data = gen_ins.generate(self.observation_num)
        start = stopwatch.start_ms()
        cc.canonical_correlation(measure, data, set_of_dims_1st, set_of_dims_2nd)
        end = stopwatch.stop_ms()
        cpu_time_ms = end - start
        return cpu_time_ms
