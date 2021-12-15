import sys

from experiment.sanity.Sanity import Sanity
from experiment.power.CCPowerM import CCPowerM
from experiment.power.GC3PowerM import GC3PowerM

if __name__ == '__main__':
    output_folder = sys.argv[1]
    GC3PowerM(output_folder).run()
