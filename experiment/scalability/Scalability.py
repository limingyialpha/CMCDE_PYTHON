from experiment import experiment_template
from experiment.scalability.CCScalabilityD import CCScalabilityD
from experiment.scalability.CCScalabilityO import CCScalabilityO
from experiment.scalability.GC3ScalabilityD import GC3ScalabilityD
from experiment.scalability.GC3ScalabilityO import GC3ScalabilityO


class Scalability(experiment_template.Experiment):
    def run(self):
        CCScalabilityD().run()
        CCScalabilityO().run()
        GC3ScalabilityD().run()
        GC3ScalabilityO().run()
