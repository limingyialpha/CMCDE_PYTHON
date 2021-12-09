import multiprocessing

import numpy as np
from multiprocessing import Pool
import socket
from datetime import datetime

from experiment import experiment_template
from generators import *

from stats import gc

from utils import stopwatch

"""
This experiment analyse the scalability(CPU time) of different generalized contrast measures,
with respect to different observation numbers.
Only GMCDE is in scala, others are in Python partner repo.
We look at Independent Uniform distribution.
We look at 3 groups of dimensions.
Each group has 2 dimensions, total 6.
We look at maximum 1000 observations.
"""


class GC3ScalabilityO(experiment_template.Experiment):
    # data specific params
    gen = linear.Linear(6, 0)
    set_of_dims_1st = frozenset([0, 1])
    set_of_dims_2nd = frozenset([2, 3])
    set_of_dims_3rd = frozenset([4, 5])
    dim_groups = {set_of_dims_1st, set_of_dims_2nd, set_of_dims_3rd}
    observation_nums_of_interest = [10, 20, 50, 100, 200, 300, 400, 500, 1000]

    # measure specific params
    measures = {"dHSIC"}

    # methodology specific params
    repetitions = 500
    level_of_parallelism = multiprocessing.cpu_count() - 1
    unit = "ms"

    def run(self):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.info(f"{now} - Starting experiments - {type(self).__name__}")

        self.info("Data specific params:")
        self.info(f"generator: {self.gen.get_id()}")
        self.info(f"observation number of interest: {self.observation_nums_of_interest}")

        self.info("Dependency measure specific params:")
        self.info(f"General Dependency measures: {self.measures}")

        self.info("Methodology specific params:")
        self.info(f"number of repetitions: {self.repetitions}")
        self.info(f"Level of parallelism: {self.level_of_parallelism}")
        self.info(f"unit of scalability (cpu time): {self.unit}")

        self.info(f"Started on {socket.gethostname()}")

        summary_header = ["measure", "obs_num", "avg_cpu_time"]
        self.write_summary_header(summary_header)
        for measure in self.measures:
            for obs_num in self.observation_nums_of_interest:
                self.info(f"now dealing with measure: {measure}, observation number: {obs_num}")
                with Pool(processes=self.level_of_parallelism) as pool:
                    task_inputs = [(measure, obs_num) for _ in
                                   range(0, self.repetitions)]
                    cpu_times = pool.starmap(self.task, task_inputs)
                avg_cpu_time = np.mean(cpu_times)
                summary_content = [measure, obs_num, avg_cpu_time]
                self.write_summary_content(summary_content)

        self.info(f"{now} - Finished experiments - {type(self).__name__}")

    def task(self, measure, obs_num):
        data = self.gen.generate(obs_num)
        start = stopwatch.start_ms()
        gc.generalized_contrast(measure, data, self.dim_groups)
        end = stopwatch.stop_ms()
        cpu_time_ms = end - start
        return cpu_time_ms
