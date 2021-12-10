from experiment import experiment_template
from experiment.sanity.Dilute import Dilute
from experiment.sanity.Duplicate import Duplicate


class Sanity(experiment_template.Experiment):
    def run(self):
        Dilute().run()
        Duplicate().run()
