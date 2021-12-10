from experiment import experiment_template
from experiment.scalability.CCScalabilityD import CCScalabilityD
from experiment.scalability.CCScalabilityO import CCScalabilityO
from experiment.scalability.GC3ScalabilityD import GC3ScalabilityD
from experiment.scalability.GC3ScalabilityO import GC3ScalabilityO


"""
This experiment analyse the scalability(CPU time) of different generalized contrast measures,
with respect to different observation numbers.
Only GMCDE is in scala, others are in Python partner repo.
We look at Independent Uniform distribution.
We look at 3 groups of dimensions.
Each group has 2 dimensions, total 6.
"""


class Scalability(experiment_template.Experiment):
    def run(self):
        CCScalabilityD().run()
        CCScalabilityO().run()
        GC3ScalabilityD().run()
        GC3ScalabilityO().run()
