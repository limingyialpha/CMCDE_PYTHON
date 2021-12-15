from experiment import experiment
from experiment.sanity.Dilute import Dilute
from experiment.sanity.Duplicate import Duplicate


class Sanity(experiment.Experiment):

    def __init__(self, output_folder):
        super().__init__(output_folder)

    def run(self):
        x = Dilute(self.output_folder)
        x.run()
        y = Duplicate(self.output_folder)
        y.run()
