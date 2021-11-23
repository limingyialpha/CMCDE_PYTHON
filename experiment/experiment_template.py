import logging
from pathlib import Path
from datetime import datetime
import socket
import csv



class Experiment:
    master_experiment_folder = "E:\work\Project\CMCDE_PYTHON\experiments_output"

    def __init__(self):
        Path(self.master_experiment_folder).mkdir(parents=True, exist_ok=True)
        class_name = type(self).__name__
        now = datetime.now()
        self.dir_name = now.strftime("%Y-%m-%d-%H-%M-%S") + "_" + class_name + "_"
        self.experiment_folder = self.master_experiment_folder + "/" + self.dir_name
        Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        self.summary_path = self.experiment_folder + "/" + class_name + ".csv"
        self.log_path = self.experiment_folder + "/" + class_name + ".log"
        logging.basicConfig(level=logging.INFO, filename=self.log_path,
                            format=f'%(asctime)s (process)d %(levelname)s {class_name} - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self.info(f"Starting the experiment {class_name}")
        self.info(f"Started on {socket.gethostname()}")

    def info(self, message: str):
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



