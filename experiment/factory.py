import logging
import sys
from typing import List

from experiment.dependencies.Dependencies import Dependencies
from experiment.power.CCPowerM import CCPowerM
from experiment.power.GC3PowerM import GC3PowerM
from experiment.sanity.Sanity import Sanity
from experiment.scalability.Scalability import Scalability

experiments_names = [
    "Dependencies",
    "Sanity",
    "Scalability",
    "CCPowerM",
    "GC3PowerM"
]

experiments_dictionary = {
    "Dependencies": Dependencies,
    "Sanity": Sanity,
    "Scalability": Scalability,
    "CCPowerM": CCPowerM,
    "GC3PowerM": GC3PowerM
}


def run(experiments: List[str], output_folder: str):
    if len(experiments) == 0:
        raise RuntimeError('No experiments given!')
    elif len(experiments) == 1 and experiments[0] == "all":
        logger = logging.getLogger("factory")
        logger.setLevel(logging.INFO)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(f'%(asctime)s (process)d %(levelname)s "facotry" - %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        logger.info("Now running ALL experiments:")
        logger.info(experiments_names)
        for name in experiments_names:
            exp = experiments_dictionary[name]
            exp(output_folder).run()
        logger.info("Finished ALL experiments.")
    else:
        correct = all(e in experiments_names for e in experiments)
        if correct:
            for e in experiments:
                experiments_dictionary[e](output_folder).run()
        else:
            raise RuntimeError("Wrong experiment name (names)!")
