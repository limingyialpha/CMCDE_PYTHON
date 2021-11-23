import multiprocessing

import numpy as np
from multiprocessing import Pool
import itertools

from experiment import experiment_template
from functools import partial
from generators import *
from stats import stats_factory
import csv
import logging


class PowerSymmetricHalf(experiment_template.Experiment):
    #dimensions_of_interest = [4, 6, 8, 10, 16]
    dimensions_of_interest = [4, 6]
    noise_level = 20
    #noise_level = 30
    precision = 2
    obs_num = 1000
    comparison_iteration_num = 500
    gens = [
        linear.Linear,
        # cross.Cross,
        # partial(doubleLinear.DoubleLinear, param=0.25),
        # hourglass.Hourglass,
        # hypercube.Hypercube,
        # hypercubeGraph.HypercubeGraph,
        # hypersphere.HyperSphere,
        # partial(parabola.Parabola, param=1),
        # partial(sine.Sine, param=1),
        # partial(sine.Sine, param=5),
        # star.Star,
        # zinv.Zinv
    ]
    threshold = 0.05
    test_names = stats_factory.test_names

    def task(self, dim, noise):
        # call the cofig again, so that we are sure that, even if we are in a subprocess,
        # that the information will be logged centrally
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {type(self).__name__} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        self.info(f"now dealing with dimension: {dim} and noise {noise}")
        x = set(range(0, int(dim / 2)))
        y = set(range(int(dim / 2), dim))
        # to avoid some stupid warnings in dcor, we take 0 as 0.0001
        noise = 0.0001 if noise == 0 else noise
        for gen in self.gens:
            gen_instance = gen(dim, noise)
            for test_name in self.test_names:
                results = np.zeros(self.comparison_iteration_num)
                for rep in range(self.comparison_iteration_num):
                    data = gen_instance.generate(self.obs_num)
                    results[rep] = stats_factory.test(test_name, data, x, y, threshold=self.threshold)
                power = results.sum() / len(results)
                # in writing , we need to write 0.0001 back to 0
                noise_to_write = 0 if noise == 0.0001 else noise
                summary_content = [gen_instance.get_id(), dim, noise_to_write, test_name, power]
                with open(self.summary_path, mode='a', newline='') as summary_file:
                    summary_writer = csv.writer(summary_file, delimiter=',')
                    summary_writer.writerow(summary_content)
                #self.write_summary_content(summary_content)


    def run(self):
        self.info("Parameters:")
        self.info(f"dimensions of interest: {self.dimensions_of_interest}")
        self.info(f"noise levels: {self.noise_level}")
        self.info(f"number of observations: {self.obs_num}")
        self.info(f"Canonical Correlation statistic methods: {self.test_names}")
        self.info(f"number of repetitions for comparison: {self.comparison_iteration_num}")

        summary_header = ["genId", "dim", "noise", "stats", "power"]
        self.write_summary_header(summary_header)

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            noises = [round(i / self.noise_level, 2) for i in range(0, self.noise_level + 1)]
            dim_noise_products = list(itertools.product(self.dimensions_of_interest, noises))
            pool.starmap(self.task, dim_noise_products)
