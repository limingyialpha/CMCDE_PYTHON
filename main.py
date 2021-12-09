from experiment.power import GC3PowerM
from experiment.scalability import GC3ScalabilityD, GC3ScalabilityO

if __name__ == '__main__':
    GC3ScalabilityO.GC3ScalabilityO().run()
    GC3ScalabilityD.GC3ScalabilityD().run()
    GC3PowerM.GC3PowerM().run()

