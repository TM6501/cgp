import random
import copy
import inspect
import itertools
import math
import numpy as np
import numbers

import AbstractCGPIndividual

class CGPIndividual(AbstractCGPIndividual.AbstractCGPIndividual):
    """This is a generic CGPIndividual. It assumes 2 inputs to every function
    and allows for very basic memory functionality."""
    selectedOuts = []

    def __init__(self,
                 type=None,
                 inputSize=1,
                 outputSize=1,
                 shape=None,
                 inputMemory=None,
                 pRange=None,
                 constraintRange=None,
                 functionList=None):
        # Set all of the initialization variables:
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.__genotype = []
        self.__activeGenes = None

        # To make the syntax easier, separate the shape parameter:
        self.rows = self.shape['rowCount']
        self.cols = self.shape['colCount']
        self.maxColForward = self.shape['maxColForward']
        self.maxColBack = self.shape['maxColBack']

        self.integerConversion()

        # Our total number of inputs will be the size of our input array plus
        # any stored memory:
        self.totalInputCount = self.inputSize
        if self.inputMemory is not None:
            self.totalInputCount += sum(self.inputMemory['memLength'])

        self.resetForNewTimeSeries()

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        integerList = ['rows', 'cols', 'maxColForward', 'maxColBack',
                       'inputSize']

        for name in integerList:
            setattr(self, name, int(getattr(self, name)))

    def printSelf(self):
        """Write a concise description of self out to screen. This function
        only has use in debugging and is never used during training."""
        print("inputSize: " + str(self.inputSize))
        print("outputSize: " + str(self.outputSize))
        print("rows: " + str(self.rows))
        print("cols: " + str(self.cols))
        print("maxColForward: " + str(self.maxColForward))
        print("maxColBack: " + str(self.maxColBack))
        print("pRange: " + str(self.pRange))

        print("Genotype: ")
        self.printGenotype()

    def randomize(self):
        """This function will randomize this individual."""
        self.__genotype = self.getRandomizedGenotype(self.functionList,
          self.rows, self.cols, self.maxColForward, self.maxColBack,
          self.pRange, self.totalInputCount, self.outputSize)

        self.__activeGenes = None

    def getPercentageNodesUsed(self):
        """Return the percentage of nodes in the active path through the
        genotype."""
        activeGenes = self.getActiveGenes()
        return (len(activeGenes) / len(self.__genotype)) * 100.0

    def printGenotype(self):
        """Utility function to print the genome for debugging purposes."""
        for row in range(self.rows):
            for col in range(self.cols):
                geneNumber = (col * (self.rows - 1)) + col + row + \
                             self.totalInputCount
                # If it is an input or output column, just print:
                if col == 0 or col == self.cols - 1:
                    print(str(self.__genotype[geneNumber]) + " ", end='')
                else:  # Need to format the floating point number:
                    print("[%d %d %d %.2f]" % (self.__genotype[geneNumber][0],
                                               self.__genotype[geneNumber][1],
                                               self.__genotype[geneNumber][2],
                                               self.__genotype[geneNumber][3]))
            print("\n")
        print("Out: " + str(self.__genotype[-self.outputSize:]))

    def printNumberedGenotype(self):
        """Utility function to just give an ordered list of all nodes."""
        for i in range(len(self.__genotype)):
            print(str(i) + ": " + str(self.__genotype[i]))

    def getActiveGenes(self):
        """This function is designed to get a list of the genes that are active
        without going through the process of calculating their values."""

        # Recalculate the active genes only if it has changed:
        if self.__activeGenes is None:
            self.__activeGenes = self.getActiveGenes_generic(self.__genotype,
                                                             self.outputSize)

        return self.__activeGenes

    def getActiveFunctionList(self):
        """This function will return a dictionary of every function and how
        many times it is actively used by this current individual."""
        self.getActiveGenes()

        retDict = {}

        for geneNum in self.__activeGenes:
            if len(self.__genotype[geneNum]) == 4:
                if self.__genotype[geneNum][0] in retDict:
                    retDict[self.__genotype[geneNum][0]] += 1
                else:
                    retDict[self.__genotype[geneNum][0]] = 1

        return retDict

    def calculateOutputs(self, inputs):
        """This function will generate the outputs of a single genotype
        using the provided inputs. If the individual is built with memory,
        this will add the input to the current memory values:

        Arguments:
            inputs - A list of inputs, expected to be of the appropriate
                     length.

        Returns: A list of output values of the same length as the number of
                 output genes in the genotype."""

        # Start out with none of the values calculated:
        self.__geneOutputs = [None] * len(self.__genotype)
        outputs = []

        # Fill in that the inputs have values available:
        inputNumber = 0
        for input in inputs:
            self.__geneOutputs[inputNumber] = inputs[inputNumber]
            inputNumber += 1

        # Fill in the memory values. We will always line up inputs into the
        # machine as such:
        # [All current iteration inputs] [memory for value 0]
        # [memory for value 1] [memory for value 2] [etc]
        if self.inputMemory is not None:
            for valueMemory in self.inputMemoryValues:
                if valueMemory is not None:
                    # Add the memory values to our inputs:
                    for i in range(len(valueMemory)):
                        self.__geneOutputs[inputNumber] = valueMemory[i]
                        inputNumber += 1

            # Push the value into the most recent memory spaces:
            for i in range(len(self.inputMemoryValues)):
                if self.inputMemoryValues[i] is not None:
                    # Drop one off the end, and one to the front:
                    del(self.inputMemoryValues[i][len(self.inputMemoryValues[i]) - 1])
                    self.inputMemoryValues[i].insert(0,
                                                     copy.deepcopy(inputs[i]))

        # Get all of the active genes:
        temp = self.getActiveGenes()
        activeGenes = copy.deepcopy(temp)

        # Remove the input and output genes from the active gene list:
        activeGenes = [x for x in activeGenes if x not in
                       range(self.totalInputCount)]
        activeGenes = [x for x in activeGenes if x not in
                       range(len(self.__genotype) - self.outputSize,
                             len(self.__genotype))]

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
                X = self.__geneOutputs[self.__genotype[geneNum][1]]
                Y = self.__geneOutputs[self.__genotype[geneNum][2]]

                # Check if we can calculate our output:
                if X is not None and Y is not None:
                    # Calculate the value and set it into our outputs:
                    self.__geneOutputs[geneNum] = \
                        self.constrain(
                           self.__genotype[geneNum][3] * \
                           (self.functionList[self.__genotype[geneNum][0]](
                           X, Y, self.__genotype[geneNum][3]))
                        )

                    # Mark progress made:
                    progressMade = True

                    # Remove from future calculations:
                    genesToRemove.append(geneNum)

            activeGenes = [x for x in activeGenes if x not in genesToRemove]

        # No more progress being made, calculate the rest with 0 used for
        # uncalculatable inputs. Moving from left to right, some values may
        # cascade as one gene's output provides another's input:
        for geneNum in activeGenes:
            X = self.__geneOutputs[self.__genotype[geneNum][1]]
            Y = self.__geneOutputs[self.__genotype[geneNum][2]]

            if X is None:
                X = 0
            if Y is None:
                Y = 0

            self.__geneOutputs[geneNum] = \
               self.constrain(
                  self.__genotype[geneNum][3] * \
                  (self.functionList[self.__genotype[geneNum][0]](
                  X, Y, self.__genotype[geneNum][3]))
               )

        # Now, all gene outputs should be set, we can collect our output:
        outputs = []
        for geneNum in range(len(self.__genotype) - self.outputSize,
                             len(self.__genotype)):
            self.__geneOutputs[geneNum] = \
               self.__geneOutputs[self.__genotype[geneNum][0]]

            outputs.append(self.__geneOutputs[geneNum])
            if outputs[len(outputs) - 1] is None:
                self.printNumberedGenotype()
                raise ValueError("Output for gene %d not available." %
                                 (self.__genotype[geneNum][0]))

        return outputs

    def __getProbabilisticMutatedChild(self, genMutationRate=0.1,
                                       outMutationRate=0.1,
                                       application='pergene'):
        child = copy.deepcopy(self)
        child.probabilisticMutate(child.__genotype, child.functionList,
          child.pRange, child.maxColForward, child.maxColBack,
          genMutationRate=genMutationRate, outMutationRate=outMutationRate,
          application=application)
        child.__activeGenes = None
        return child

    def __getActiveGeneMutatedChild(self, numGenesToMutate=1):
        child = copy.deepcopy(self)
        activeGenes = child.getActiveGenes()
        child.activeGeneMutate(self.__genotype, self.functionList, self.pRange,
                               activeGenes, self.maxColForward,
                               self.maxColBack,
                               numGenesToMutate=numGenesToMutate)
        child.__activeGenes = None
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

    def constrain(self, value):
        # No range? No constraining values:
        if self.constraintRange is None:
            return value

        min = self.constraintRange[0]
        max = self.constraintRange[1]

        # If we have a vector, we need to apply to every element:
        if isinstance(value, np.ndarray):
            retVal = copy.copy(value)
            initialShape = retVal.shape
            retVal = retVal.flatten()
            retVal = np.array(
               [(min if i < min else (max if i > max else i)) for i in retVal])
            retVal = retVal.reshape(initialShape)
        else:
            retVal = value
            if not math.isfinite(retVal):
                retVal = 0.0
            if retVal > max:
                retVal = max
            elif retVal < min:
                retVal = min

        return retVal

    def resetForNewTimeSeries(self):
        """This function will wipe the memory of inputs so the individual's
        next input is interpretted as the beginning of a new time series."""
        self.__activeGenes = None

        if self.inputMemory is None:
            return

        # constFill provides us with a constant value for each individual
        # input:
        if self.inputMemory['strategy'].lower() == 'constfill':
            self.inputMemoryValues = []

            # Go through all of our memory values and fill them with their
            # respective constants:
            for i in range(len(self.inputMemory['startValues'])):
                if self.inputMemory['startValues'] is not None:
                    tempMemory = [self.inputMemory['startValues'][i]] * \
                                 self.inputMemory['memLength'][i]
                    self.inputMemoryValues.append(tempMemory)
                else:
                    self.inputMemoryValues.append(None)
        else:
            raise ValueError("Unknown memory strategy: %s" %
                             (self.inputMemory['strategy']))

    def performOncePerEpochUpdates(self, listAllIndividuals, fitnesses):
        """The base individual doesn't have extra updates to perform, but
        this function needs to be implemented anyway."""
        return listAllIndividuals

    def setFunctionList(self, functionList):
        """Set the functions available to us.

        Argument:
            functionList - The (probably shared) list of available functions
                           that we can use."""

        self.functionList = functionList
