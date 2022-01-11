import sys
from experiment import factory

if __name__ == '__main__':
    output_folder = sys.argv[1]
    experiments = sys.argv[2:]
    factory.run(experiments, output_folder)
