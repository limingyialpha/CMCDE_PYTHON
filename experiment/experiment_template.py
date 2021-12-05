import logging
from pathlib import Path
from datetime import datetime
import csv

from typing import List


class Experiment:
    master_experiment_folder = "E:\work\Project\CMCDE_PYTHON\experiments_output"

    def __init__(self):
        Path(self.master_experiment_folder).mkdir(parents=True, exist_ok=True)
        class_name = type(self).__name__
        now = datetime.now()
        self.dir_name = now.strftime("%Y-%m-%d-%H-%M-%S") + "_" + class_name + "_"
        self.experiment_folder = self.master_experiment_folder + "/" + self.dir_name
        Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        self.summary_path = self.experiment_folder + "/" + class_name + "_python" + ".csv"
        self.log_path = self.experiment_folder + "/" + class_name + "_python" + ".log"
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {class_name} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

    def info(self, message: str):
        # call the config again, so that we are sure that, even if we are in a subprocess,
        # that the information will be logged centrally
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {type(self).__name__} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        logging.info(message)

    def run(self):
        pass

    def write_summary_header(self, header_array):
        with open(self.summary_path, mode='w+', newline='') as summary_file:
            summary_writer = csv.writer(summary_file, delimiter=',')
            summary_writer.writerow(header_array)

    def write_summary_content(self, content_array):
        with open(self.summary_path, mode='a', newline='') as summary_file:
            summary_writer = csv.writer(summary_file, delimiter=',')
            summary_writer.writerow(content_array)

    # python has another implementation of percentile as in scala, breeeze.
    # This implementation is identical to the one in scala, breeze
    # so that the experiment in scala and in python are comparable.
    def percentile_scala_breeze(self, list_of_floats: List[float], p: float):
        arr = sorted(list_of_floats)
        f = (len(arr) + 1) * p
        i = int(f)
        if i == 0:
            return arr[0]
        elif i >= len(arr):
            return arr[-1]
        else:
            return arr[i - 1] + (f - i) * (arr[i] - arr[i - 1])



