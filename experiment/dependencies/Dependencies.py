import logging
from functools import partial

import socket
from datetime import datetime

from experiment import experiment
from generators import *

from generators import linearPeriodic

"""
Plot dependencies
"""


class Dependencies(experiment.Experiment):

    def __init__(self, output_folder):
        super().__init__(output_folder)

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
    ]
    noise = 0.0
    num_obs = 1000

    def run(self):
        logger = logging.getLogger(self.class_name)
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logger.info(f"{now} - Starting experiments - {type(self).__name__}")

        logger.info("Now plotting dependencies in 2d and 3d spaces.")
        logger.info(f"Started on {socket.gethostname()}")

        summary_header = ["genId", "dim", "x", "y", "z"]
        self.write_summary_header(summary_header)

        # 2d
        dim = 2
        for gen in self.gens:
            gen_instance = gen(dim, self.noise)
            data = gen_instance.generate(self.num_obs)
            for row in data:
                self.write_summary_content([gen_instance.get_id(), dim, row[0], row[1], 0.0])
        # 3d
        dim = 3
        for gen in self.gens:
            gen_instance = gen(dim, self.noise)
            data = gen_instance.generate(self.num_obs)
            for row in data:
                self.write_summary_content([gen_instance.get_id(), dim, row[0], row[1], row[2]])

        logger.info(f"{now} - Finished experiments - {type(self).__name__}")
