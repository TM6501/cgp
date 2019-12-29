import random
import copy
import inspect
import itertools
import math

import AbstractCGPIndividual

class MCCGPIndividual(AbstractCGPIndividual.AbstractCGPIndividual):

    """This class represents a Multi-Chromosomal Individual. MCCGP individuals
    will maintain a separate genotype for every output and can mutate using
    crossover.

    Traditionally, they have a fitness function for every output,
    but that will be optional with this class. If multiple fitness functions
    are provided, crossover will choose the best of each genotype. If not, then
    genotypes will be chosen randomly from the available population (assumed to
    be the best of the most recent epoch's generation)."""

    def __init__(
      self,
      type=None,
      inputSize=1,
      outputSize=1,
      shape=None,
      pRange=None,
      constraintRange=None,
      functionList=None,
      MC_CGPSpecificParameters=None):

      args, _, _, values = inspect.getargvalues(inspect.currentframe())
      values.pop("self")

      for arg, val in values.items():
          setattr(self, arg, val)

      self.integerConversion()

      self.__genotype = None

      self.crossoverStrategy = MC_CGPSpecificParameters['crossoverStrategy']

      if not self.checkAndSetParameters():
          raise ValueError("Found error in parameters.")

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        integerList = ['rows', 'cols', 'maxColForward', 'maxColBack',
                       'inputSize']

        for name in integerList:
            setattr(self, name, int(getattr(self, name)))

    def checkAndSetParameters(self):
        """Check all variables passed into this class and modify those that
        need to be changed.

        Arguments: None

        Returns:
            True if the class variables were able to be modified enough so
            that the class is ready."""

        retValue = True

        # First check all of the variables that must exist:
        if self.inputSize is None or self.outputSize is None \
          or self.shape is None or self.pRange is None \
          or self.functionList is None \
          or self.MC_CGPSpecificParameters is None:
            print("At least one required parameter was not provided.")
            retValue = False


        if self.inputSize is not None:
            self.totalInputCount = self.inputSize

        # For later simplicity, there will be separate values for every
        # shape, pRange, constraintRange, and functionList. For those
        # that the user only provided a single value, we'll create those lists:

        # Shape:
        if isinstance(self.shape, list):
            if len(self.shape) != outputSize:
                print("If a list of shapes is provided, its length must be the\
 number of outputs.")
                retValue = False
        else: # Make a list from the single provided shape:
            self.shape = [self.shape] * self.outputSize

        # To make accessing the values easier later:
        self.rows = []
        self.cols = []
        self.maxColForward = []
        self.maxColBack = []
        for shape in self.shape:
            self.rows.append(shape['rowCount'])
            self.cols.append(shape['colCount'])
            self.maxColForward.append(shape['maxColForward'])
            self.maxColBack.append(shape['maxColBack'])

        # pRanges:
        if isinstance(self.pRange[0], list):
            if len(self.pRange) != outputSize:
                print("If a list of pRanges is provided, its length must be \
number of outputs.")
                retValue = False
        else: # Make a list from the provided pRanges:
            self.pRange = [self.pRange] * self.outputSize

        # constraintRange:
        if self.constraintRange is not None:
            if isinstance(self.constraintRange[0], list):
                if len(self.pRange) != outputSize:
                    print("If a list of constraints is provided, its length \
must be number of outputs.")
                    retValue = False
            else: # Make a list from the provided constraint ranges:
                self.constraintRange = [self.constraintRange] * self.outputSize
        else:
            self.constraintRange = [None] * self.outputSize

        # Function lists:
        if isinstance(self.functionList[0], list):
            if len(self.functionList) != outputSize:
                print("If the functionlist is provided as a list of lists, the\
 top-level list's length must be the outputsize.")
                retValue = False
        else: # Just a list of functions, create our list of lists:
            self.functionList = [self.functionList] * self.outputSize

        return retValue

    def getPercentageNodesUsed(self):
        # Return a list of percentages:
        percentages = []

        for i in self.__genotype:
            activeGenes = self.getActiveGenes_generic(i, 1)
            percentages.append((len(activeGenes) / len(i)) * 100.0)

        return percentages

    def randomize(self):
        """Randomize this individual by creating all of our genotypes."""

        self.__genotype = []
        for i in range(self.outputSize):
            self.__genotype.append(self.getRandomizedGenotype(
              self.functionList[i], self.rows[i], self.cols[i],
              self.maxColForward[i], self.maxColBack[i], self.pRange[i],
              self.totalInputCount, 1))

    def calculateOutputs(self, inputs):
        outputs = []
        for i in range(self.outputSize):
            outputs.append(self.calculateSingleGenotypeOutput(
              self.__genotype[i], inputs, self.functionList[i]))

        return outputs

    def calculateSingleGenotypeOutput(self, genotype, inputs, functionList):
        """Calculate the output from a single genotype.

        Arguments:
            genotype - The genotype to use.
            inputs - A list of inputs, expected to be of the appropriate
                     length.
            functionList - The functionlist associated with this genotype.

        Returns: A single output value for this genotype"""

        # Start out with none of the values calculated:
        geneOutputs = [None] * len(genotype)

        # Fill in that the inputs have values available:
        inputNumber = 0
        for input in inputs:
            geneOutputs[inputNumber] = inputs[inputNumber]
            inputNumber += 1

        # Get all of the active genes:
        temp = self.getActiveGenes_generic(genotype, 1)
        activeGenes = copy.deepcopy(temp)

        # Remove the input and output genes from the active gene list:
        activeGenes = [x for x in activeGenes if x not in
                       range(self.totalInputCount)]
        activeGenes = [x for x in activeGenes if x not in
                       range(len(genotype) - 1,
                             len(genotype))]

        # Make sure they are in order:
        activeGenes = sorted(activeGenes)

        # To deal with the possibility of recurrent connections, we will move
        # forward in the active genes list, calculating every output we can
        # and repeat that process for as long as we are still making progress
        # (at least 1 gene output value is calculated). If we can't make any
        # more progress (circular connection), we'll set uncalculatable inputs
        # to zero and finish producing outputs.
        progressMade = True
        while progressMade:
            progressMade = False
            genesToRemove = []
            for geneNum in activeGenes:
                # Get the gene's inputs:
                X = geneOutputs[genotype[geneNum][1]]
                Y = geneOutputs[genotype[geneNum][2]]

                # Check if we can calculate our output:
                if X is not None and Y is not None:
                    # Calculate the value and set it into our outputs:
                    geneOutputs[geneNum] = self.constrain(
                      genotype[geneNum][3] * \
                      (functionList[genotype[geneNum][0]](
                      X, Y, genotype[geneNum][3])))

                    # Mark progress made:
                    progressMade = True

                    # Remove from future calculations:
                    genesToRemove.append(geneNum)

            activeGenes = [x for x in activeGenes if x not in genesToRemove]

        # No more progress being made, calculate the rest with 0 used for
        # uncalculatable inputs. Moving from left to right, some values may
        # cascade as one gene's output provides another's input:
        for geneNum in activeGenes:
            X = geneOutputs[genotype[geneNum][1]]
            Y = geneOutputs[genotype[geneNum][2]]

            if X is None:
                X = 0
            if Y is None:
                Y = 0

            geneOutputs[geneNum] = self.constrain(
              genotype[geneNum][3] * (functionList[genotype[geneNum][0]](
              X, Y, genotype[geneNum][3])))

        # Now, all gene outputs should be set, we can collect our output, which
        # should be the last value:
        output = None
        geneNum = len(genotype) - 1

        output = geneOutputs[genotype[geneNum][0]]

        if output is None:
            self.printNumberedGenotype()
            raise ValueError("Output for gene %d not available." %
                             (genotype[geneNum][0]))

        return output

    def __getProbabilisticMutatedChild(self, genMutationRate=0.1,
                                       outMutationRate=0.1,
                                       application='pergene'):
        child = copy.deepcopy(self)
        for i in range(len(child.__genotype)):
            child.probabilisticMutate(child.__genotype[i],
                                      child.functionList[i],
                                      child.pRange[i],
                                      child.maxColForward[i],
                                      child.maxColBack[i],
                                      totalInputCount=child.totalInputCount,
                                      outputSize=1,
                                      rows=child.rows[i],
                                      cols=child.cols[i],
                                      genMutationRate=genMutationRate,
                                      outMutationRate=outMutationRate,
                                      application=application)
        return child

    def __getActiveGeneMutatedChild(self, numGenesToMutate=1):
        child = copy.deepcopy(self)

        for i in range(len(child.__genotype)):
            activeGenes = child.getActiveGenes_generic(child.__genotype[i], 1)
            child.activeGeneMutate(child.__genotype[i],
                                   child.functionList[i],
                                   child.pRange[i],
                                   activeGenes,
                                   child.maxColForward[i],
                                   child.maxColBack[i],
                                   numGenesToMutate=numGenesToMutate,
                                   totalInputCount=child.totalInputCount,
                                   outputSize=1,
                                   rows=child.rows[i],
                                   cols=child.cols[i])

        return child

    def getOneMutatedChild(self, mutationStrategy):
        """This function will return a mutated child based upon this
           individual.

        Arguments:
            mutationStrategy - A dictionary with the name of the mutation
                               strategy as well as any parameters necessary
                               for that strategy.

        Returns:
            The new child."""

        # Mutation rate and number of genes to mutate are given as ranges.
        # We need to select a value from within the available range.

        # Apply a certain chance of mutation to all genes:
        if mutationStrategy['name'].lower() == 'probability':
            return self.__getProbabilisticMutatedChild(
                genMutationRate=random.uniform(mutationStrategy['genRate'][0],
                                               mutationStrategy['genRate'][1]),
                outMutationRate=random.uniform(mutationStrategy['outRate'][0],
                                               mutationStrategy['outRate'][1]),
                application=mutationStrategy['application'])

        # Mutate genes until at least X active genes are mutated. X is
        # normally 1.
        elif mutationStrategy['name'].lower() == 'activegene':
            return self.__getActiveGeneMutatedChild(
                numGenesToMutate=random.randint(
                    mutationStrategy['numGenes'][0],
                    mutationStrategy['numGenes'][1]))
        else:
            ValueError("Unknown mutation strategy.")

    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """Multi-chromosomal individuals do crossover mutations once every
        epoch."""
        return self.__produceCrossoverIndividuals(listAllIndividuals, epochFitnesses)


    def __becomeRandomCrossover(self, listOfParents):
        """Given a list of parents to pull from, turn our genome into a random
        crossover of all parents."""

        maxRand = len(listOfParents) - 1
        for i in range(len(self.__genotype)):
            parentNum = random.randint(0, maxRand)
            self.__genotype[i] = copy.deepcopy(listOfParents[parentNum].__genotype[i])

    def __produceNewPopulationFromParents_randomCrossover(self, listAllIndividuals):
        """Assume that all given individuals are the parents of the new
        generation and produce some crossover individuals from them."""

        childrenToProduce = self.MC_CGPSpecificParameters['numberCrossoverChildren']
        maxParent = len(listAllIndividuals)

        for i in range(childrenToProduce):
            newInd = copy.deepcopy(self)
            newInd.__becomeRandomCrossover(listAllIndividuals[:maxParent])
            listAllIndividuals.append(newInd)

        return listAllIndividuals

    def __produceCrossoverIndividuals(self, listAllIndividuals, epochFitnesses):
        """Add to the list of individuals as many crossover individuals as we
        want to create."""
        # There are 2 possibilities with crossover indivdiuals:
        # 1. We have fitness functions for each output and we can pick the
        #    best of each to produce a single best individual.
        # 2. We have a single fitness function and we should crossover from
        #    random combinations of our best individuals.

        if self.crossoverStrategy.lower() == "singlebest":
            return self.__produceNewPopulationFromParents_singleBest(listAllIndividuals, epochFitnesses)
        else:
            return self.__produceNewPopulationFromParents_randomCrossover(listAllIndividuals)

    def __produceNewPopulationFromParents_singleBest(self, listAllIndividuals,
                                                     epochFitnesses):
        """Change this individual's genotypes into the best of all available
        individuals."""

        # When creating only a single individual, it is assumed that we ar that
        # individual. We need to modify ourselves, then set ourselves as the
        # only item in the all-individuals list that was passed in.
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
