import sys
from experiment import factory

if __name__ == '__main__':
    experiment = sys.argv[1]
    output_folder = sys.argv[2]
    factory.run(experiment, output_folder)
