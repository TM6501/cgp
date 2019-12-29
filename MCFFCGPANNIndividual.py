import random
import copy
import inspect
import itertools
import math

import FFCGPANNIndividual

class MCFFCGPANNIndividual(FFCGPANNIndividual.FFCGPANNIndividual):
    """This class represents a Multi-Chromosomal Feed-Forward Cartesian
    Genetic Programmed Artificial Neural Network (MCFFCGPANN). This is a
    group of FFCGPANN's, one for each classification outputs."""

    def __init__(self, type=None, inputSize=None, outputSize=None, shape=None,
                 pRange=None, constraintRange=None, functionList=None,
                 MCFFCGPANNSpecificParameters=None):
        """Set all training variables and initialize the class."""

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.__activeGenes = [None] * outputSize
        self.__genotype = [None] * outputSize

        # Make these easier to access later:
        self.rows = shape['rowCount']
        self.cols = shape['colCount']
        self.maxColForward = shape['maxColForward']
        self.maxColBack = shape['maxColBack']

        self.integerConversion()

        self.totalInputCount = self.inputSize

        # Get MCFFCGPANN specific parameters:

        # Must be a list of integers representing the number of inputs into
        # each neuron. The list could be of length 1 if we aren't evolving the
        # number of inputs to each neuron:
        self._FFCGPANNIndividual__inputsPerNeuron = \
          MCFFCGPANNSpecificParameters['inputsPerNeuron']

        # Range of possible weights to apply to each neuron inputs. The list
        # must be of length 2, but the values could be identical to indicate
        # that we aren't evolving weights:
        self._FFCGPANNIndividual__weightRange = \
          MCFFCGPANNSpecificParameters['weightRange']

        # On/Off-switches allow specific inputs to a neuron to be switched
        # off or on. This list should be of at least length 1, and must only
        # contain the values 0 or 1 (which will be multiplied by inputs).
        # A list of only '0' will not allow the network to train at all.
        self._FFCGPANNIndividual__switchValues = \
          MCFFCGPANNSpecificParameters['switchValues']

        # Our strategy for producing crossover individuals:
        self.__crossoverStrategy = MCFFCGPANNSpecificParameters['crossoverstrategy']


    def resetTrainingVariable(self, type=None, inputSize=None, outputSize=None,
                              shape=None, pRange=None, constraintRange=None,
                              functionList=None,
                              MCFFCGPANNSpecificParameters=None):

        """Reset all training variables."""

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            if val is not None:
                setattr(self, arg, val)

        # Make these easier to access later:
        self.rows = self.shape['rowCount']
        self.cols = self.shape['colCount']
        self.maxColForward = self.shape['maxColForward']
        self.maxColBack = self.shape['maxColBack']

        self.integerConversion()

        self.totalInputCount = self.inputSize
        self.__activeGenes = [None] * self.outputSize

        # Get MCFFCGPANN specific parameters:

        # Must be a minimum and maximum number of inputs per neuron. The min
        # and max can be the same if we don't want to mutate on this.
        self._FFCGPANNIndividual__inputsPerNeuron = \
          self.MCFFCGPANNSpecificParameters['inputsPerNeuron']

        # Range of possible weights to apply to each neuron inputs. The list
        # must be of length 2, but the values could be identical to indicate
        # that we aren't evolving weights:
        self._FFCGPANNIndividual__weightRange = \
          self.MCFFCGPANNSpecificParameters['weightRange']

        # On/Off-switches allow specific inputs to a neuron to be switched
        # off or on. This list should be of at least length 1, and must only
        # contain the values 0 or 1 (which will be multiplied by inputs).
        # A list of only '0' will not allow the network to train at all.
        self._FFCGPANNIndividual__switchValues = \
          self.MCFFCGPANNSpecificParameters['switchValues']

        # Our strategy for producing crossover individuals:
        self.__crossoverStrategy = MCFFCGPANNSpecificParameters['crossoverstrategy']

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        super().integerConversion()

    def getPercentageNodesUsed(self):
        # Return a list of percentages:
        percentages = []

        for i in self.__genotype:
            activeGenes = self.calculateActiveGenes_genotype(i)
            percentages.append((len(activeGenes) / len(i)) * 100.0)

        return percentages

    def randomize(self):
        """Randomize our genotypes. This is normally only done to initialize
        before training."""
        self.__activeGenes = [None] * self.outputSize

        for i in range(len(self.__genotype)):
            self.__genotype[i] = self.randomize_genotype(self.__genotype[i], 1)

    def calculateActiveGenes(self, genotypeNum):
        """Determine the genes that are part of calculating the outputs."""
        self.__activeGenes[genotypeNum] = self.calculateActiveGenes_genotype(
              self.__genotype[genotypeNum])

    def getActiveGenes(self, genotypeNum):
        """Get the genes that are part of calculating the outputs."""
        self.calculateActiveGenes(genotypeNum)

        return self.__activeGenes[genotypeNum]

    def calculateOutputs(self, inputs):
        """Process inputs and produce outputs."""
        outputs = []
        for i in range(self.outputSize):
            actGenes = self.getActiveGenes(i)
            outputs.append(self.calculateOutputs_genotype(self.__genotype[i],
                                                          actGenes,
                                                          inputs, 1))
        return outputs

    def activeGeneMutate(self, numGenesToMutate):
        """Mutate ourselves in an active-gene way."""
        for i in range(len(self.__genotype)):
            actGenes = self.getActiveGenes(i)
            self.__genotype[i] = self.activeGeneMutate_genotype(
              self.__genotype[i], numGenesToMutate, actGenes)

        self.__activeGenes = [None] * self.outputSize

    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """Multi-chromosomal individuals do crossover mutations once per
        epoch."""
        return self.__produceCrossoverIndividuals(listAllIndividuals,
                                                  epochFitnesses)

    def __produceCrossoverIndividuals(self, listAllIndividuals,
                                      epochFitnesses):
        """Add to the list of individuals as many crossover individuals as we
        want to create."""
        # There are 2 possibilities with crossover indivdiuals:
        # 1. We have fitness functions for each output and we can pick the
        #    best of each to produce a single best individual.
        # 2. We have a single fitness function and we should crossover from
        #    random combinations of our best individuals.

        if self.__crossoverStrategy.lower() == "singlebest":
            return self.__produceNewPopulationFromParents_singleBest(
              listAllIndividuals, epochFitnesses)
        else:
            return self.__produceNewPopulationFromParents_randomCrossover(
              listAllIndividuals)

    def __produceNewPopulationFromParents_singleBest(self, listAllIndividuals,
                                                     epochFitnesses):
        """Change this individual's genotypes into the best of all available
        individuals."""

        # When creating only a single individual, it is assumed that we are
        # that individual. We need to modify ourselves, then set ourselves as
        # the only item in the all-individuals list that was passed in.
        bestFitnesses = copy.deepcopy(epochFitnesses[0])
        bestFitnessIndices = [0] * len(epochFitnesses[0])

        # Determine the best individual genotypes:
        for indNum in range(len(epochFitnesses)):
            for valNum in range(len(epochFitnesses[0])):
                if epochFitnesses[indNum][valNum] >= bestFitnesses[valNum]:
                    bestFitnesses[valNum] = epochFitnesses[indNum][valNum]
                    bestFitnessIndices[valNum] = indNum

        for valNum in range(len(epochFitnesses[0])):
            self.__genotype[valNum] = copy.deepcopy(
              listAllIndividuals[bestFitnessIndices[valNum]].__genotype[valNum])

        return [self]

    def __produceNewPopulationFromParents_randomCrossover(self, listAllIndividuals):
        """Assume that all given individuals are the parents of the new
        generation and produce some crossover individuals from them."""
        raise NotImplementedError("No random crossover, yet.")
    
