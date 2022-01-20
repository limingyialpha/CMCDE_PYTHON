import logging
import sys
from pathlib import Path
from datetime import datetime
import csv

from typing import List


class Experiment:

    def __init__(self, output_folder: str):
        self.output_folder = output_folder
        self.master_experiment_folder = self.output_folder + '/experiments_output'
        Path(self.master_experiment_folder).mkdir(parents=True, exist_ok=True)
        self.class_name = type(self).__name__
        now = datetime.now()
        self.dir_name = now.strftime("%Y-%m-%d-%H-%M") + "_" + self.class_name + "_python"
        self.experiment_folder = self.master_experiment_folder + "/" + self.dir_name
        Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        self.summary_path = self.experiment_folder + "/" + self.class_name + "_python" + ".csv"
        self.log_path = self.experiment_folder + "/" + self.class_name + "_python" + ".log"
        self.logger = logging.getLogger(self.class_name)
        self.logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s {self.class_name} - %(message)s')
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)
        f_handler = logging.FileHandler(self.log_path)
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s {self.class_name} - %(message)s')
        f_handler.setFormatter(f_format)
        self.logger.addHandler(f_handler)

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
