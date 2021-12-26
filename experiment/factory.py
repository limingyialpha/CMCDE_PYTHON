import logging
import sys

from experiment.power.CCPowerM import CCPowerM
from experiment.power.GC3PowerM import GC3PowerM
from experiment.sanity.Sanity import Sanity
from experiment.scalability.Scalability import Scalability

experiments_dictionary = {
    "Sanity": Sanity,
    "Scalability": Scalability,
    "CCPowerM": CCPowerM,
    "GC3PowerM": GC3PowerM
}


def run(experiment: str, output_folder: str):
    if experiment == "all":
        names = list(experiments_dictionary.keys())
        logger = logging.getLogger("factory")
        logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s "facotry" - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        logger.info("Now running all the experiments:")
        logger.info(names)
        for name in names:
            exp = experiments_dictionary[name]
            exp(output_folder).run()
        logger.info("Finished all experiments.")
    elif experiment not in experiments_dictionary.keys():
        raise RuntimeError('Wrong experiment name!')
    else:
        experiments_dictionary[experiment](output_folder).run()
