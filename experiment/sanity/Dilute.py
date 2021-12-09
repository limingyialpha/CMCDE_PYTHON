import numpy as np
import socket
from datetime import datetime

from experiment import experiment_template

from generators.independent import Independent
from stats import gc

"""
This experiment checks the behaviour of the measure with increasing independent dimensions.
We start with 2 dimensions with identical observations, we then add independent dimensions.
We expect the generalized contrast to drop.
GMCDE is in scala repo and dHSIC is in python repo
"""


class Dilute(experiment_template.Experiment):
    # data specific params
    data_12 = np.array([np.array([i / 1000, i / 1000]) for i in range(1000)])

    # measure specific params
    measures = {"dHSIC"}

    # methodology specific params
    maximal_extra_independent_dimensions_num = 100

    def run(self):
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.info(f"{now} - Starting experiments - {type(self).__name__}")

        self.info("Data specific params:")
        self.info("benchmark: 2 dimensions of 0 to 0.999 with step 0.001")

        self.info("Dependency measure specific params:")
        self.info(f"Generalized Contrast measures: {self.measures}")

        self.info("Methodology specific params:")
        self.info(f"maximal extra independent dimensions: {self.maximal_extra_independent_dimensions_num}")

        self.info(f"Started on {socket.gethostname()}")

        summary_header = ["measure", "dilution", "gc"]
        self.write_summary_header(summary_header)

        for measure in self.measures:
            for i in range(0, self.maximal_extra_independent_dimensions_num + 1):
                data = self.data_12 if (i == 0) else np.concatenate(
                    (self.data_12, Independent(i, 0).generate(len(self.data_12))), axis=1)
                sets_of_dims = {frozenset([i]) for i in range(0, 2 + i)}
                generalized_contrast = gc.generalized_contrast(measure, data, sets_of_dims)
                summary_content = [measure, i, generalized_contrast]
                self.write_summary_content(summary_content)

        self.info(f"{now} - Finished experiments - {type(self).__name__}")
