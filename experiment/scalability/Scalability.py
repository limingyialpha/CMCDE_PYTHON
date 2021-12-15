from experiment import experiment
from experiment.scalability.CCScalabilityD import CCScalabilityD
from experiment.scalability.CCScalabilityO import CCScalabilityO
from experiment.scalability.GC3ScalabilityD import GC3ScalabilityD
from experiment.scalability.GC3ScalabilityO import GC3ScalabilityO


class Scalability(experiment.Experiment):

    def __init__(self, output_folder):
        super().__init__(output_folder)

    def run(self):
        CCScalabilityD(self.output_folder).run()
        CCScalabilityO(self.output_folder).run()
        GC3ScalabilityD(self.output_folder).run()
        GC3ScalabilityO(self.output_folder).run()
