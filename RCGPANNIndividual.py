import random
import copy
import inspect

import FFCGPANNIndividual


class RCGPANNIndividual(FFCGPANNIndividual.FFCGPANNIndividual):
    """This class represents a Recurrent Cartesian Genetic Programmed
    Artificial Neural Network (RCGPANN). Here, instead of layers of neurons
    set out in a rigid structure (as is the case with a traditional FFANN),
    the overall complexity of the network develops as part of the training.
    Connections between neurons, weights, functions applied to inputs,
    the number of inputs, and the possibly being ingored are all up for grabs
    during the training process.

    In some literature RCGPANN or CGPRNN refers to CGPANN in which nodes are
    allowed to make recurrent connections. That is not the case with this
    class. This class takes outputs from one calculation and feeds them back
    as inputs to the next. It allows this to be done in several different ways.
    """

    # There are multiple ways that recurrency can be implemented. The "type"
    # argument in RCGPANNSpecificParameters will decide how handle it.
    # Type 1: All outputs are passed back in as parameters to the next
    #         evaluation of an input. self.__memorySteps determines how many
    #         steps this has. Meaning, if memorySteps is 5, then the outputs
    #         from the network from 5, 4, 3, 2, and 1 evaluation ago are all
    #         passed in.
    # Type 2: All outputs are used as inputs to an additional node. Those
    #         inputs are multiplied by weights and a function is applied to
    #         the sum, just like with any other node. The outputs from this
    #         function is used as an input on the subsequent input evaluation.
    #         The function and all weights into this node are evolveable.
    #         __memorySteps determines how many of these nodes exist. They do
    #         not hold results from more than 1 step back, but rather all use
    #         the same inputs, just with different weights and potentially
    #         different functions.
    # Type 3: A single node is created in the style of type 2 with
    #         __memorySteps = 1.  Anything greater than 1 duplicates the single
    #        node's output from 1 evaluation ago.  Meaning, a single evolvable
    #        node takes in all of the outputs from time t and creates a
    #        recurrent value to use on the evaluation of t+1. That same value
    #        is passed in as memoryStep #2 on evaluation t+2.  It is continues
    #        being passed in X times, where X is __memorySteps. 0's are put in
    #        place of all memory steps for which values haven't be determined.

    def __init__(self, type=None, inputSize=None, outputSize=None, shape=None,
                 pRange=None, constraintRange=None, functionList=None,
                 RCGPANNSpecificParameters=None):
        """Set all training variables and initialize the class."""
        super().__init__(type=type, inputSize=inputSize, outputSize=outputSize,
                         shape=shape, constraintRange=constraintRange,
                         functionList=functionList,
                         FFCGPANNSpecificParameters=RCGPANNSpecificParameters)

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        # Get RCGPANN specific parameters:

        # The number of previous network outputs to use as inputs to the
        # system. Effectively, this provides the network with a memory of the
        # X previous outputs.
        self.__memorySteps = RCGPANNSpecificParameters['memorySteps']
        self.__type = RCGPANNSpecificParameters['type']

        self.integerConversion()

        if self.__type != 1 and self.__type != 2 and self.__type !=3:
            raise ValueError("Type must be 1, 2, or 3.")

        if self.__type == 1:
            self.totalInputCount = self.inputSize + \
              (self.outputSize * self.__memorySteps)

            self.__previousOutputs = [[0.0] * self.outputSize] * \
                                     self.__memorySteps
        elif self.__type == 2:
            self.totalInputCount = self.inputSize + self.__memorySteps
            self.__recurrentNodes = [None] * self.__memorySteps
            self.__previousOutputs = [0.0] * self.__memorySteps

        else:  # type 3
            self.totalInputCount = self.inputSize + self.__memorySteps
            self.__recurrentNodes = [None]
            self.__previousOutputs = [0.0] * self.__memorySteps

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        super().integerConversion()

        # setattr / getattr won't let us set/get private variables. Do those
        # manually:
        self.__memorySteps = int(self.__memorySteps)
        self.__type = int(self.__type)

    def resetForNewTimeSeries(self):
        """Reset this individual for a new series of inputs.
        RCGPANN is primarily used for handling time series data and
        reinforcement learning. This means it needs to reset itself between
        each series of inputs."""
        if self.__type == 1:
            self.__previousOutputs = [[0.0] * self.outputSize] * \
                                     self.__memorySteps
        else:  # type 2 & 3:
            self.__previousOutputs = [0.0] * self.__memorySteps

    def getPercentageNodesUsed(self):
        """Get the percentage of nodes used to actually calculate the output."""
        activeGenes = self.getActiveGenes()
        return (len(activeGenes) / len(self._FFCGPANNIndividual__genotype)) * 100.0

    def calculateOutputs(self, inputs):
        """Process the inputs and produce outputs."""
        actGenes = self.getActiveGenes()

        # Build our input with memory of previous outputs:
        fullInput = copy.deepcopy(list(inputs))

        if self.__type == 1:
            for singleOutput in self.__previousOutputs:
                fullInput += singleOutput

        else:  # Type 2 / 3:
            fullInput += self.__previousOutputs

        newOutput = self.calculateOutputs_genotype(
          self._FFCGPANNIndividual__genotype, actGenes, fullInput,
          self.outputSize)

        if self.__type == 1:
            self.__previousOutputs += [newOutput]
            del self.__previousOutputs[0]
        elif self.__type == 2:
            self.__previousOutputs = self.calculateType2PreviousOutputs(newOutput)
        else:  # type 3
            # Insert the new calculated output:
            tempVal = self.calculateType2PreviousOutputs(newOutput)
            self.__previousOutputs.insert(0, tempVal[0])

            # Cut off the last value that just fell out of memory:
            self.__previousOutputs = self.__previousOutputs[:self.__memorySteps]

        return newOutput

    def calculateType2PreviousOutputs(self, networkOutputs):
        """Go through all of our type2 recurrent nodes and calculate their
        outputs."""
        # Rucurrent node structure is the same as standard node structure, but
        # with a known and unchangeable number of inputs. We will still
        # calculate as though we don't know the number of inputs, though, to
        # allow for future development that may change the inputs to recurrent
        # nodes.

        # For the purposes of this function, outputs must always be a list:
        tempNetworkOutputs = copy.deepcopy(networkOutputs)
        if not isinstance(tempNetworkOutputs, list):
            tempNetworkOutputs = [tempNetworkOutputs]

        fullOutputs = []
        for oneNode in self.__recurrentNodes:
            # Node structure: [function bias
            #                 [inputNodeNum1 weight1 activation1]
            #                 ...
            #                 [inputNodeNumN weightN activationN]]
            # For our purposes, the inputNodeNum will refer to the number in
            # the available outputs rather than nodes in the genotype.
            sum = oneNode[1]  # Start with the bias
            for i in range(2, len(oneNode)):
                # Add weight * input * activation
                sum += oneNode[i][1] * tempNetworkOutputs[oneNode[i][0]] * \
                       oneNode[i][2]

            # Apply the function:
            output = self.functionList[oneNode[0]](sum)
            fullOutputs.append(output)

        return fullOutputs

    def randomize(self):
        """Randomize our genotype."""
        # First do the FFCGPANN randomization:
        super(RCGPANNIndividual, self).randomize()

        # Now, randomize our recurrent connections:
        self.__recurrentNodes = []
        self._FFCGPANNIndividual__activeGenes = None
        if self.__type == 2 or self.__type == 3:
            memNodesToBuild = 1  # Assume type 3
            if self.__type == 2:
                memNodesToBuild = self.__memorySteps
            for i in range(memNodesToBuild):
                newNode = []
                # Add the function:
                newNode.append(random.randint(0, len(self.functionList) - 1))

                # Add the bias term:
                if self.pRange is None:
                    newNode.append(0.0)
                else:
                    newNode.append(random.uniform(self.pRange[0],
                                                  self.pRange[1]))

                # Add the list of weights/inputs:
                for j in range(self.outputSize):
                    nodeInput = [j]
                    # Weight on this node input:
                    nodeInput.append(random.uniform(
                      self._FFCGPANNIndividual__weightRange[0],
                      self._FFCGPANNIndividual__weightRange[1]))

                    # Add the switch value:
                    if len(self._FFCGPANNIndividual__switchValues) > 1:
                        nodeInput.append(random.randint(0, 1))
                    else:  # No choice, just append 1.
                        nodeInput.append(1)

                    newNode.append(nodeInput)

                self.__recurrentNodes.append(newNode)

    def activeGeneMutate(self, numGenesToMutate):
        """Mutate ourselves using an active gene strategy."""
        # If we're type 2 or 3, we'll need to take over the mutation to do it
        # properly.  Otherwise, we can let the base class handle it.

        if self.__type == 1:
            super(RCGPANNIndividual, self).activeGeneMutate(numGenesToMutate)

        else:
            self.doActiveGeneMutate_type2_3(self._FFCGPANNIndividual__genotype,
              self.__recurrentNodes, numGenesToMutate)
            self._FFCGPANNIndividual__activeGenes = None

    def doActiveGeneMutate_type2_3(self, genotype, recurrentGenes,
                                   numGenesToMutate):
        """Mutate our genome and recurrent connections with equal probability.
        """
        activeGenesMutated = 0
        activeGenes = copy.deepcopy(self.getActiveGenes())

        while activeGenesMutated < numGenesToMutate:
            # Choose between a gene in the standard genome or in the recurrent
            # connections:
            geneNum = random.randint(self.totalInputCount,
                                     len(genotype) - 1 + len(recurrentGenes))

            if geneNum >= len(genotype):
                self.mutateRecurrentNode(recurrentGenes,
                                         geneNum - len(genotype))

                # A recurrent gene is active if its input number is active:
                if (geneNum - len(genotype) + self.inputSize) in activeGenes:
                    activeGenesMutated += 1

                # Carry on in this loop so that the whole next section need not
                # not be in a big else block.
                continue

            # Output gene, select a new input:
            if len(genotype[geneNum]) == 1:
                genotype[geneNum][0] = \
                  self.getValidInputNodeNumber(geneNum, self.maxColForward,
                                               self.maxColBack,
                                               self.totalInputCount,
                                               self.outputSize, self.rows,
                                               self.cols)
            # Standard node, need to decide between many options:
            # Change Function, change the bias, change Switch, add an input,
            # remove an input, change an input's weight,
            # change an input's switch, and change an input's target.
            else:
                allOptions = []

                # Build a list of all possible options so that we can choose
                # between them randomly:
                if len(self.functionList) > 1:
                    allOptions.append('CF')  # Change function

                if self.pRange is not None:
                    allOptions.append('CB') # Change bias

                if len(genotype[geneNum]) <= self._FFCGPANNIndividual__inputsPerNeuron[1]:
                    allOptions.append('AI')  # Add input

                # Remove input isn't done on a per-input basis so that it has
                # the same chance as add-input
                if len(genotype[geneNum]) -1 > self._FFCGPANNIndividual__inputsPerNeuron[0]:
                    allOptions.append('RI')  # Remove input

                if len(self._FFCGPANNIndividual__switchValues) > 1:
                    for i in range(2, len(genotype[geneNum])):
                        allOptions.append('CS_%d' % (i))  # Change switch

                for i in range(2, len(genotype[geneNum])):
                    allOptions.append('CW_%d' % (i))  # Change weight
                    allOptions.append('CT_%d' % (i))  # Change target

                # Choose what we're mutating:
                selection = allOptions[random.randint(0, len(allOptions) - 1)]
                category = selection[:2]
                inputNum = None
                if len(selection) > 2:
                    inputNum = int(selection[3:])

                if category == 'CF':
                    self.chooseNewFunction(genotype, geneNum)
                elif category == 'CB':
                    self.chooseNewBias(genotype, geneNum)
                elif category == 'AI':
                    self.addInput(genotype, geneNum)
                elif category == 'RI':
                    self.removeInput(genotype, geneNum)
                elif category == 'CS':
                    self.changeSwitch(genotype, geneNum, inputNum)
                elif category == 'CW':
                    self.changeWeight(genotype, geneNum, inputNum)
                elif category == 'CT':
                    self.changeTarget(genotype, geneNum, inputNum)

            if geneNum in activeGenes:
                activeGenesMutated += 1

        return genotype, recurrentGenes

    def mutateRecurrentNode(self, recurrentGenes, geneNum):
        """Mutate a single recurrent gene, given the list of genes and the
        gene number to mutate."""
        allOptions = []

        # Build a list of all possible options so that we can choose
        # between them randomly. Recurrent connections cannot add inputs,
        # remove inputs, or change the target of the input. This limits the
        # available options significantly.
        if len(self.functionList) > 1:
            allOptions.append('CF')  # Change function

        if self.pRange is not None:
            allOptions.append('CB') # Change bias

        if len(self._FFCGPANNIndividual__switchValues) > 1:
            for i in range(2, len(recurrentGenes[geneNum])):
                allOptions.append('CS_%d' % (i))  # Change switch

        for i in range(2, len(recurrentGenes[geneNum])):
            allOptions.append('CW_%d' % (i))  # Change weight

        # Choose what we're mutating:
        selection = allOptions[random.randint(0, len(allOptions) - 1)]
        category = selection[:2]
        inputNum = None
        if len(selection) > 2:
            inputNum = int(selection[3:])

        if category == 'CF':
            self.chooseNewFunction(recurrentGenes, geneNum)
        elif category == 'CB':
            self.chooseNewBias(recurrentGenes, geneNum)
        elif category == 'CS':
            self.changeSwitch(recurrentGenes, geneNum, inputNum)
        elif category == 'CW':
            self.changeWeight(recurrentGenes, geneNum, inputNum)

    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """No processing needs to be done, so return the individuals."""
        return listAllIndividuals

#############################################
# Functions below here are provided for debugging purposes to allow the user
# to insect certain aspects of the genotype to confirm mutation as one would
# expect.
#############################################

    def printGenotype(self):
        self.printGivenGenotype(self._FFCGPANNIndividual__genotype)
        print("#### Recurrent Genes: ####")
        self.printGivenGenotype(self.__recurrentNodes)

    def getGenotype(self):
        return self._FFCGPANNIndividual__genotype

    def getRecurrentGenes(self):
        return self.__recurrentNodes
