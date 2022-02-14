import random
import copy
import inspect
import itertools
import math
import sys

import AbstractCGPIndividual

class FFCGPANNIndividual(AbstractCGPIndividual.AbstractCGPIndividual):
    """This class represents a Feed-Forward Cartesian Genetic Programmed
    Artificial Neural Network (FFCGPANN). Here, instead of layers of neurons
    set out in a rigid structure (as is the case with a traditional FFANN),
    the overall complexity of the network develops as part of the training.
    Changing connections between neurons, weights, functions applied to inputs,
    the number of inputs, and certain inputs being ignored entirely are all
    possible training paths."""

    def __init__(self, type=None, inputSize=None, outputSize=None, shape=None,
                 pRange=None, constraintRange=None, functionList=None,
                 FFCGPANNSpecificParameters=None):
        """Set all training variables and initialize the class."""

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.__genotype = None

        # Make these easier to access later:
        self.rows = shape['rowCount']
        self.cols = shape['colCount']
        self.maxColForward = shape['maxColForward']
        self.maxColBack = shape['maxColBack']

        # Make sure we are calling our version, not a subclass version.
        FFCGPANNIndividual.integerConversion(self)

        self.totalInputCount = self.inputSize
        self.__activeGenes = None

        # Get FFCGPANN specific parameters:

        # Must be a list of integers representing the number of inputs into
        # each neuron. The list could be of length 1 if we aren't evolving the
        # number of inputs to each neuron:
        self.__inputsPerNeuron = FFCGPANNSpecificParameters['inputsPerNeuron']

        # Range of possible weights to apply to each neuron inputs. The list
        # must be of length 2, but the values could be identical to indicate
        # that we aren't evolving weights:
        self.__weightRange = FFCGPANNSpecificParameters['weightRange']

        # On/Off-switches allow specific inputs to a neuron to be switched
        # off or on. This list should be of at least length 1, and must only
        # contain the values 0 or 1 (which will be multiplied by inputs).
        # A list of only '0' will not allow the network to train at all.
        self.__switchValues = FFCGPANNSpecificParameters['switchValues']

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        integerList = ['rows', 'cols', 'maxColForward', 'maxColBack',
                       'inputSize']

        for name in integerList:
            setattr(self, name, int(getattr(self, name)))

    def getRandomInput(self, nodeNum):
        """Produce a single input vector based upon the columnNumber where
        the input will be used."""

        # Get our input node number:
        inputNodeNumber = self.getValidInputNodeNumber(nodeNum,
          self.maxColForward, self.maxColBack, self.totalInputCount,
          self.outputSize, self.rows, self.cols)
        weight = random.uniform(self.__weightRange[0], self.__weightRange[1])

        # Default to switch 'on'. If we have options, then decide between
        # them:
        switch = 1
        if len(self.__switchValues) > 1:
            switch = random.randint(0, 1)

        return [inputNodeNumber, weight, switch]

    def getPercentageNodesUsed(self):
        """Report back the percentage of nodes used in our active path."""
        activeGenes = self.getActiveGenes()
        return (len(activeGenes) / len(self.__genotype)) * 100.0

    def randomize(self):
        """Randomize all trainable parameters. Effectively this starts us with
        no training at all."""
        self.__activeGenes = None
        self.__genotype = self.randomize_genotype(self.__genotype,
                                                  self.outputSize)

    def randomize_genotype(self, genotype, outputSize):
        """Generate an entirely random genotype based upon our number of
        inputs, outputs, and the shape provided to us."""
        genotype = []
        numFunctions = len(self.functionList)
        numNodes = (self.rows * self.cols) + self.totalInputCount

        for nodeNum in range(numNodes):
            colNum = self.getColumnNumber(
              nodeNum, totalInputCount=self.totalInputCount,
              outputSize=self.outputSize, rows=self.rows, cols=self.cols)

            if colNum == 0:
                gene = ['IN']
            else: # Standard gene
                # Full gene: [function, bias,
                #             [inputNeuron1, weight1, switch1],
                #             [inputNeuron2, weight2, switch2],
                #                       ...
                #             [inputNeuronN, weightN, switchN]]

                # Add the function:
                function = random.randint(0, numFunctions-1)

                # If a pRange is provided, use it as the bias value on the neuron:
                pValue = 0.0
                if self.pRange is not None:
                    pValue = random.uniform(self.pRange[0], self.pRange[1])

                gene = [function, pValue]

                # Add the inputs:
                numInputs = random.randint(self.__inputsPerNeuron[0],
                                           self.__inputsPerNeuron[1])

                for i in range(numInputs):
                    gene.append(self.getRandomInput(nodeNum))

            genotype.append(gene)

        # Add the outputs. First, get the acceptable range:
        minCol = max(self.cols + 1 - self.maxColBack, 0)
        maxCol = self.cols

        minNodeNum, _ = self.getNodeNumberRange(minCol,
          totalInputCount=self.totalInputCount, outputSize=outputSize,
          rows=self.rows, cols=self.cols)

        _, maxNodeNum = self.getNodeNumberRange(maxCol,
          totalInputCount=self.totalInputCount, outputSize=outputSize,
          rows=self.rows, cols=self.cols)

        for i in range(outputSize):
            genotype.append([random.randint(minNodeNum, maxNodeNum)])

        return genotype

    def setANNGenesActive(self, genotype, geneNumber, activeGenes):
        """Recursively add all genes upon which this one depends to the
        activeGenes list."""

        # Already marked?
        if geneNumber in activeGenes:
            return

        # Mark active:
        activeGenes.append(geneNumber)

        # Special case length = 1 (could be input or output):
        if len(genotype[geneNumber]) == 1:
            # Input gene:
            if geneNumber < self.totalInputCount:
                return

            # Output gene:
            self.setANNGenesActive(genotype, genotype[geneNumber][0],
                                   activeGenes)

        else:  # Standard gene
            for i in range(2, len(genotype[geneNumber])):
                self.setANNGenesActive(genotype, genotype[geneNumber][i][0],
                                       activeGenes)

    def calculateActiveGenes(self):
        """Calculate all of our active genes."""
        return self.calculateActiveGenes_genotype(self.__genotype)

    def calculateActiveGenes_genotype(self, genotype):
        """Calculate all of the active genes of the provided genotype."""
        activeGenes = []

        # Get the dependent genes for all outputs:
        for geneNumber in range(len(genotype) - 1, 0, -1):
            if len(genotype[geneNumber]) == 1: # Output gene, add dependents.
                self.setANNGenesActive(genotype, geneNumber, activeGenes)
            else:  # Done with output genes, break out.
                break

        # Return the list:
        return activeGenes

    def getActiveGenes(self):
        """Return a list of all active genes so we know what has to be
        calculated."""
        if self.__activeGenes is None:
            self.__activeGenes = self.calculateActiveGenes()

        return self.__activeGenes

    def calculateOutputs(self, inputs):
        """Given a set of inputs, calculate this individual's output value(s).
        """
        actGenes = self.getActiveGenes()
        return self.calculateOutputs_genotype(self.__genotype, actGenes,
                                              inputs, self.outputSize)

    def calculateOutputs_genotype(self, genotype, actGenes, inputs, outputSize):
        """Calculate all of the outputs for a given genotype and set of inputs.
        """

        # Start with no outputs calculated:
        geneOutputs = [None] * len(genotype)

        # Fill in the inputs:
        inputNumber = 0
        for inputNumber in range(len(inputs)):
            geneOutputs[inputNumber] = inputs[inputNumber]

        # Get our active genes:
        activeGenes = copy.deepcopy(actGenes)

        # Remove the input and output genes from the active gene list:
        activeGenes = [x for x in activeGenes if x not in
                       range(self.totalInputCount)]
        activeGenes = [x for x in activeGenes if x not in
                       range(len(genotype) - outputSize,
                             len(genotype))]

        # Put them in order:
        activeGenes = sorted(activeGenes)

        # Go through the genes in order so that the inputs should always be
        # available. The precludes recurrent connections. The feed forward
        # individual cannot allow using later nodes as inputs without changing
        # how outptus are calculated:
        for geneNum in activeGenes:
            sum = 0.0
            # Add each input to the sum coming into this neuron:
            for i in range(2, len(genotype[geneNum])):
                neuronNum = genotype[geneNum][i][0]
                if geneOutputs[neuronNum] is None:
                    self.printGivenGenotype(genotype)
                    print("Active Genes: %s" % (str(actGenes)))
                    raise ValueError("Output %d is not available." % (neuronNum))
                else:
                    # (Input * Weight) * Switch:
                    sum += (geneOutputs[neuronNum] * \
                            genotype[geneNum][i][1]) * \
                            genotype[geneNum][i][2]

            # Add the bias to the sum:
            sum += genotype[geneNum][1]

            # Apply the function to the sum:
            geneOutputs[geneNum] = \
              self.functionList[genotype[geneNum][0]](sum)

        # All should be calculated now.  Need to return the output genes.
        outputs = []
        for geneNum in range(len(genotype) - outputSize,
                             len(genotype)):
            geneOutputs[geneNum] = \
              geneOutputs[genotype[geneNum][0]]
            outputs.append(geneOutputs[geneNum])
            if outputs[len(outputs) - 1] is None:
                self.printGenotype()
                print("Active Genes: %s" % (str(actGenes)))
                raise ValueError("Output for gene %d not available." %
                                 (genotype[geneNum][0]))

        # If we only have a single output, don't return it as a single value:
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
    def getOneMutatedChild(self, mutationStrategy):
        """Return a mutated child based upon this individual and the given
        mutation strategy."""

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

    def __getActiveGeneMutatedChild(self, numGenesToMutate=1):
        """Create and return a new individual that is the result of applying
        active gene mutation to this individual."""
        child = copy.deepcopy(self)
        child.activeGeneMutate(numGenesToMutate)
        return child

    def __getProbabilisticMutatedChild(self, genMutationRate=0.01, outMutationRate=0.01, application='perGene'):
        """Create and return a new individual that is the result of applying
        probabilistic mutation to this individual."""
        child = copy.deepcopy(self)
        child.probabilisticMutate(child.__genotype, child.functionList,
                                  child.pRange, child.maxColForward,
                                  child.maxColBack, genMutationRate=genMutationRate,
                                  outMutationRate=outMutationRate, application=application)
        return child

    def probabilisticMutate(self, genotype, functionList, pRange,
                            maxColForward, maxColBack, genMutationRate=0.1,
                            outMutationRate=0.1,
                            totalInputCount=None, outputSize=None,
                            rows=None, cols=None, application='pergene'):
        """Mutate the provided genotype, given the provided parameters."""

        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        for geneNum in range(totalInputCount, len(genotype)):
            # Mutate outputs at a different rate than standard genes:
            if len(genotype[geneNum]) == 1:
                if random.random() <= outMutationRate:
                    startVal = genotype[geneNum]
                    attemptNumber = 0  # Rare case where there are no other choices
                    while startVal == genotype[geneNum] and attemptNumber < 10:
                        attemptNumber += 1
                        newOut = self.getValidInputNodeNumber(
                          geneNum, maxColForward, maxColBack,
                          totalInputCount=totalInputCount,
                          outputSize=outputSize, rows=rows, cols=cols)
                        genotype[geneNum] = [newOut]

            # Must be a generic node. Decide between applying the mutation rate
            # per gene or per value inside the gene:
            elif application.lower() == 'pergene':
                if random.random() <= genMutationRate:
                    allOptions = []

                    # Build a list of all possible options so that we can choose
                    # between them randomly:
                    if len(self.functionList) > 1:
                        allOptions.append('CF')  # Change function

                    if self.pRange is not None:
                        allOptions.append('CB') # Change bias

                    if len(genotype[geneNum]) <= self.__inputsPerNeuron[1]:
                        allOptions.append('AI')  # Add input

                    # Remove input isn't done on a per-input basis so that it has
                    # the same chance as add-input
                    if len(genotype[geneNum]) -1 > self.__inputsPerNeuron[0]:
                        allOptions.append('RI')  # Remove input

                    if len(self.__switchValues) > 1 and \
                      self.__switchValues[0] != self.__switchValues[1]:
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

                    # Mutate the gene we chose in the manner we found:
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

            elif application.lower() == 'pervalue':
                # Check the mutation once for each value in the entire gene:
                # Mutate function:
                if random.random() <= genMutationRate and len(functionList) > 1:
                    self.chooseNewFunction(genotype, geneNum)

                # Mutate the parameter (P):
                if random.random() <= genMutationRate:
                    self.chooseNewBias(genotype, geneNum)

                # Mutate all of our inputs' values separately:
                for inputNum in range(2, len(genotype[geneNum])):
                    if random.random() <= genMutationRate:
                        self.changeWeight(genotype, geneNum, inputNum)
                    if random.random() <= genMutationRate:
                        self.changeTarget(genotype, geneNum, inputNum)
                    if len(self.__switchValues) > 1 and random.random() <= genMutationRate:
                        self.changeSwitch(genotype, geneNum, inputNum)

                # Add input:
                if len(genotype[geneNum]) <= self.__inputsPerNeuron[1]:
                    if random.random() <= genMutationRate:
                        self.addInput(genotype, geneNum)

                # Remove input:
                if len(genotype[geneNum]) -1 > self.__inputsPerNeuron[0]:
                    if random.random() <= genMutationRate:
                        self.removeInput(genotype, geneNum)

            else:
                raise ValueError("Unknown mutation application strategy: %s" %
                                 (application))

        self.__activeGenes = None

    def activeGeneMutate(self, numGenesToMutate):
        """Mutate my own genotype in an active-gene mutation manner."""
        actGenes = self.getActiveGenes()
        self.activeGeneMutate_genotype(self.__genotype, numGenesToMutate,
                                       actGenes)
        self.__activeGenes = None

    def activeGeneMutate_genotype(self, genotype, numGenesToMutate, actGenes):
        """Mutate the provided genotype in an active-gene mutation manner."""

        activeGenesMutated = 0
        activeGenes = copy.deepcopy(actGenes)

        # Active gene mutation requires us to mutate genes randomly until we've
        # mutated a specific number of genes that are/were part of the active
        # path. Typically numGenesToMutate is 1, but not always.
        while activeGenesMutated < numGenesToMutate:
            geneNum = random.randint(self.totalInputCount,
                                     len(genotype) - 1)

            # Output gene, select a new input:
            if len(genotype[geneNum]) == 1:
                genotype[geneNum][0] = \
                  self.getValidInputNodeNumber(geneNum, self.maxColForward,
                                               self.maxColBack,
                                               self.totalInputCount,
                                               self.outputSize, self.rows,
                                               self.cols)
            # Standard node, need to decide between many options:
            # Change the function, change the bias, change a switch,
            # add an input, remove an input, change an input's weight,
            # change an input's switch, or change an input's target.
            else:
                allOptions = []

                # Build a list of all possible options so that we can choose
                # between them randomly:
                if len(self.functionList) > 1:
                    allOptions.append('CF')  # Change function

                if self.pRange is not None:
                    allOptions.append('CB') # Change bias

                if len(genotype[geneNum]) <= self.__inputsPerNeuron[1]:
                    allOptions.append('AI')  # Add input

                # Remove input isn't done on a per-input basis so that it has
                # the same chance as add-input
                if len(genotype[geneNum]) -1 > self.__inputsPerNeuron[0]:
                    allOptions.append('RI')  # Remove input

                if len(self.__switchValues) > 1 and \
                  self.__switchValues[0] != self.__switchValues[1]:
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

                # Mutate the gene we chose in the manner we found:
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

        return genotype

    def chooseNewFunction(self, genotype, geneNum):
        """Modify the function of the given gene."""
        if len(self.functionList) < 2:
            raise ValueError("Cannot modify the function; only one function available.")

        # Make sure it actually changes by choosing new random functions until
        # something other than what we started with is returned.
        currFunc = genotype[geneNum][0]
        while genotype[geneNum][0] == currFunc:
            genotype[geneNum][0] = random.randint(
              0, len(self.functionList) - 1)

    def chooseNewBias(self, genotype, geneNum):
        """Modify the bias on this gene."""
        if self.pRange is None or len(self.pRange) != 2:
            raise ValueError("To mutate bias, pRange must be of exactly length 2.")

        # No need to check starting value to confirm changes since we're
        # choosing a new floating point value from a range:
        genotype[geneNum][1] = random.uniform(self.pRange[0], self.pRange[1])

    def addInput(self, genotype, geneNum):
        """Add an input to the given gene."""
        genotype[geneNum].append(self.getRandomInput(geneNum))

    def removeInput(self, genotype, geneNum):
        """Remove an input from the given gene."""
        if len(genotype[geneNum]) < 3:
            raise ValueError("Cannot remove input from gene %d." % (geneNum))

        # Select a random input to remove:
        input = random.randint(2, len(genotype[geneNum]) - 1)
        del genotype[geneNum][input]

    def changeSwitch(self, genotype, geneNum, inputNum):
        """Change the switch value for given gene and input number."""
        if len(self.__switchValues) == 1:
            raise ValueError("Cannot change switch value.")

        # Switch value should only be 0 or 1.  However, we'll choose randomly
        # anyway in case of later changes allow more than just those 2
        # values:
        currSwitch = genotype[geneNum][inputNum][2]
        while currSwitch == genotype[geneNum][inputNum][2]:
            index = random.randint(0, len(self.__switchValues) - 1)
            genotype[geneNum][inputNum][2] = self.__switchValues[index]

    def changeWeight(self, genotype, geneNum, inputNum):
        """Change the weight of one of the inputs of a given gene."""
        genotype[geneNum][inputNum][1] = random.uniform(
          self.__weightRange[0], self.__weightRange[1])

    def changeTarget(self, genotype, geneNum, inputNum):
        """Change the input node number of a given gene's input."""
        currTarget = genotype[geneNum][inputNum][0]
        # Keep selecting new targets (inputs) until we get one different than
        # we started with. It is possible that this creates an infinite loop
        # if there is only a single input; we don't worry about that, though.
        # Single-input neural networks make very little sense.
        while currTarget == genotype[geneNum][inputNum][0]:
            genotype[geneNum][inputNum][0] = \
              self.getValidInputNodeNumber(geneNum, self.maxColForward,
                                           self.maxColBack,
                                           self.totalInputCount,
                                           self.outputSize, self.rows,
                                           self.cols)

    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """Do nothing because FFCGPANN doesn't need to do any population-wide
        processing each epoch."""
        return listAllIndividuals

    def printGenotype(self):
        """Print out our genotype for debug purposes."""
        self.printGivenGenotype(self.__genotype)

    def getGenotype(self):
        return self.__genotype
