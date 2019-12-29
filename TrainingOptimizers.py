"""These are classes that will be passed to GeneralCGPSolver and will be
notified of several variables after each epoch. They will then have the option
of modifying some of the training variables in GeneralCGPSolver with the goal
of minimizing training time and producing the best individual possible at
the end of training.
"""

import abc
import sys

import FitnessCollapseFunctions

class AbstractTrainingOptimizer(object, metaclass=abc.ABCMeta):
    """Initialize the class."""
    def __init__(
      self,
      startingPopulationSize=None,
      startingNumberParents=None,
      startingMaxEpochs=None,
      startingBatchSize=None,
      startingFitnessFunction=None,
      startingFitnessCollapseFunction=None,
      startingMutationStrategy=None
    ):
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))


    """Make any updates to the general solver based upon the current traing
    status. This function is called after fitness is calculated for an epoch,
    but before deciding to end training."""
    def oncePerEpochUpdate(
      self,
      genSolver,
      currEpochNumber,
      currBestFitness
    ):
        # Do nothing, and report that we did not make any changes:
        return False

class doNothingOptimizer(AbstractTrainingOptimizer):
    """The AbstractTrainingOptimizer's defaults do nothing. This is a
    concrete class which implements that interface."""
    def __init__(
      self,
      startingPopulationSize=None,
      startingNumberParents=None,
      startingMaxEpochs=None,
      startingBatchSize=None,
      startingFitnessFunction=None,
      startingFitnessCollapseFunction=None,
      startingMutationStrategy=None
    ):
        pass

class newCollapseFunctionOptimizer(AbstractTrainingOptimizer):
    """This optimizer will update the function used to collapse an individual's
    fitnesses to a single fitness."""
    def __init__(self, newCollapseFunction, epochChange=None,
                 fitnessChange=None):
        if epochChange is None and fitnessChange is None:
            raise ValueError("Either epochChange or fitnessChange must be defined.")

        self.newCollapseFunction = newCollapseFunction
        self.madeTheChange = False
        self.epochChange = epochChange
        self.fitnessChange = fitnessChange

    def oncePerEpochUpdate(
      self,
      genSolver,
      currEpochNumber,
      currBestFitness
    ):
        """Change the collapse function if we've hit one of the required
        conditions."""
        if self.madeTheChange:
            return False

        if self.epochChange is not None and currEpochNumber == self.epochChange:
            genSolver.fitnessCollapseFunction = self.newCollapseFunction
            print("Changing fitness collapse function to %s because epoch %d was hit." %
              (str(self.newCollapseFunction), currEpochNumber))
            self.madeTheChange = True

        if self.fitnessChange is not None and currBestFitness >= self.fitnessChange:
            genSolver.fitnessCollapseFunction = self.newCollapseFunction
            print("Changing fitness collapse function to %s because fitness \
%.2f was exceeded by fitness %.2f." %
              (str(self.newCollapseFunction), self.fitnessChange, currBestFitness))
            self.madeTheChange = True

        return self.madeTheChange

class newFitnessFunctionOptimizer(AbstractTrainingOptimizer):
    """This optimizer will replace the training function used to test
    individuals when certain conditions are met."""
    def __init__(self, newFitnessFunction, epochChange=None,
                 fitnessChange=None):
        if epochChange is None and fitnessChange is None:
            raise ValueError("Either epochChange or fitnessChange must be defined.")

        self.newFitnessFunction = newFitnessFunction
        self.madeTheChange = False
        self.epochChange = epochChange
        self.fitnessChange = fitnessChange

    def oncePerEpochUpdate(
      self,
      genSolver,
      currEpochNumber,
      currBestFitness
    ):
        """Change the training function if we've hit one of the required
        conditions."""
        if self.madeTheChange:
            return False

        if self.epochChange is not None and currEpochNumber == self.epochChange:
            genSolver.setFitnessFunction(self.newFitnessFunction)
            print("Changing fitness function to %s because epoch %d was hit." %
              (str(self.newFitnessFunction), currEpochNumber))
            self.madeTheChange = True

        if self.fitnessChange is not None and currBestFitness >= self.fitnessChange:
            genSolver.setFitnessFunction(self.newFitnessFunction)
            print("Changing fitness function to %s because fitness \
%.2f was exceeded by fitness %.2f on epoch %d." %
              (str(self.newFitnessFunction), self.fitnessChange,
              currBestFitness, currEpochNumber))
            self.madeTheChange = True

        return self.madeTheChange
