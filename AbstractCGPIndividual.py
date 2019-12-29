import abc
import sys
import random

class AbstractCGPIndividual(object, metaclass=abc.ABCMeta):
    """The AbstractCGPIndividual is the individual upon which all individuals
    that GeneralCGPSolver uses should be based. It implements very little
    functionality and serves mostly as an interface which must be implemented
    for GeneralCGPSolver to function."""

    @abc.abstractmethod
    def __init__(
        self,
        type=None,
        maximumModuleLevel=None,
        compressProbability=None,
        expandProbability=None,
        maximumModuleSize=None,
        moduleMutationProbability=None,
        inputSize=None,
        outputSize=None,
        shape=None,
        constraintRange=None,
        functionList=None):
            """Every individual must provide a constructor."""
            raise NotImplementedError("%s must be defined." %
                                      (sys._getframe().f_code.co_name))

    @abc.abstractmethod
    def randomize(self):
        """The randomize function is called on every individual in order to
        generate a starting population. This function should create a valid,
        though untrained, individual."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    @abc.abstractmethod
    def calculateOutputs(self, inputs):
        """Run a set of inputs through the individual and get its outputs.
        This function must return the individual's calculated outputs."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    @abc.abstractmethod
    def getOneMutatedChild(self, mutationStrategy):
        """Get a new individual mutated from the current individual based upon
        the mutation strategy (dictionary) provided."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    @abc.abstractmethod
    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """Do any updates that must be done population-wide per epoch.
        This function will only be called on a single individual, but that
        individual will be provided with a list of all individuals. This
        breaks some fundamental OOP principles, but I felt the tradeoff was
        worth it to give specific CGP types more flexibility.

        GeneralCGPSolver takes care of calculating fitness, finding the new
        parent(s), and generating a new population. That doesn't need to be
        handled by this function."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    @abc.abstractmethod
    def integerConversion(self):
        """Convert all inputs that must be integers into integers. Parameters
        are passed in as floating point values. If the individual wants some
        of them converted to integers, it should do that here."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    def constrain(self, value):
        """If the individual wants outputs constrained to a certain value,
        it should do that here. Otherwise, the AbstractCGPIndividual has a
        do-nothing function to just return the unconstrained value."""
        return value

    def resetForNewTimeSeries(self):
        """If the individual needs to reset between runs in a training
        scenario, it should do that here. If this isn't needed,
        AbstractCGPIndividual provides a do-nothing function."""
        pass

    @abc.abstractmethod
    def getPercentageNodesUsed(self):
        """As part of data collection, the individual must be able to provide
        a percentage value of how many of its computational nodes are actually
        being used as part of determining its output."""
        raise NotImplementedError("%s must be defined." %
                                  (sys._getframe().f_code.co_name))

    def getColumnNumber(self, nodeNumber, totalInputCount=None,
                        outputSize=None, rows=None, cols=None):
        """Determine the column number of a node, based upon its node number.
        This generic version makes the assumption that the first N form column
        0, where N is the number of inputs into the algorithm.

        Arguments:
            nodeNumber - The node number to check.

        Optional Arguments:
            All optional arguments are class variables of the same name. If they
            aren't provided, the class variable will be substituted in.

        Return:
            The column number of that node."""

        # Substitute in any values not provided:
        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        # Need to calculate the genotype length we'll have because this
        # function could be called as we're building it:
        totalLength = totalInputCount + outputSize + (rows * cols)

        # Input node:
        if nodeNumber < totalInputCount:
            return 0

        # Output node:
        elif nodeNumber >= totalLength - outputSize:
            return cols + 1

        else:
            return int(((nodeNumber - totalInputCount) / rows)) + 1

    def getValidInputNodeNumber(self, nodeNumber, maxColsForward, maxColsBack,
                                totalInputCount=None, outputSize=None,
                                rows=None, cols=None):
        """Return a valid input node number based upon the given node number
        and how many columns back is acceptable.

        Arguments:
            nodeNumber - The node that needs a new input.
            minColumnsBack - Maximum number of acceptable columns to search
                             forward for inputs. Can be negative to not allow
                             any recurrent connections.
            maxColumnsBack - Maximum acceptable number of colums back to search
                             for an input.

        Returns:
            Input node number."""

        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        # Our column number:
        colNum = self.getColumnNumber(nodeNumber,
                                      totalInputCount=totalInputCount,
                                      outputSize=outputSize,
                                      rows=rows, cols=cols)

        # Acceptable columns to pull from:
        maxCol = min(colNum + maxColsForward, cols)
        minCol = max(colNum - maxColsBack, 0)

        # Get the node ranges for those columns:
        minNode, _ = self.getNodeNumberRange(minCol,
                                             totalInputCount=totalInputCount,
                                             outputSize=outputSize,
                                             rows=rows, cols=cols)
        _, maxNode = self.getNodeNumberRange(maxCol,
                                             totalInputCount=totalInputCount,
                                             outputSize=outputSize,
                                             rows=rows, cols=cols)

        # Finally, choose a new value, making sure we don't use ourselves as
        # an input:
        retVal = nodeNumber
        while retVal == nodeNumber:
            retVal = random.randint(minNode, maxNode)

        return retVal

    def getNodeNumberRange(self, columnNumber, totalInputCount=None,
                           outputSize=None, rows=None, cols=None):
        """Get the range of nodes in a provided column. Both returned values
        are inclusive.

        Arguments:
          columnNumber - The column number to check.

        Return:
          The minimum and maximum node numbers."""

        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        # Input node:
        if columnNumber == 0:
            return 0, totalInputCount - 1

        # Output node:
        elif columnNumber > cols:
            # Calculating expected genotype length in case this function is
            # called before it is filled in:
            maxNode = totalInputCount + outputSize + (rows * cols) - 1
            minNode = maxNode - outputSize + 1
            return minNode, maxNode

        else:
            minNode = ((columnNumber - 1) * rows) + totalInputCount
            return minNode, minNode + (rows - 1)

    def probabilisticMutate(self, genotype, functionList, pRange,
                            maxColForward, maxColBack, genMutationRate=0.1,
                            outMutationRate=0.1,
                            totalInputCount=None, outputSize=None,
                            rows=None, cols=None, application='pergene'):
        """Mutate the provided genotype, given the provided parameters.
        Individuals will almost definitely want to provide a replacement for
        this function as a generic mutate will almost never work for new
        individual types."""

        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        for i in range(totalInputCount, len(genotype)):
            # Mutate outputs at a different rate than standard genes:
            if len(genotype[i]) == 1:
                if random.random() <= outMutationRate:
                    startVal = genotype[i]
                    while startVal == genotype[i]:
                        newOut = self.getValidInputNodeNumber(
                          i, maxColForward, maxColBack,
                          totalInputCount=totalInputCount,
                          outputSize=outputSize, rows=rows, cols=cols)
                        genotype[i] = [newOut]

            # Must be a generic node. Decide between applying the mutation rate
            # per gene or per value inside the gene:
            elif application.lower() == 'pergene':
                # Check the mutation rate once for this gene, then is mutation
                # is selected, choose one value in the gene to mutate:
                if random.random() <= genMutationRate:
                    valToMutate = random.randint(0, 3)

                    # Mutate the function:
                    if valToMutate == 0:
                        startVal = genotype[i][0]
                        while startVal == genotype[i][0]:
                            genotype[i][0] = \
                              random.randint(0, len(functionList) - 1)
                    # Mutate the p-value:
                    elif valToMutate == 3:
                        genotype[i][3] = random.uniform(pRange[0], pRange[1])

                    # Otherwise, mutate one of the 2 inputs. Make sure it
                    # changes and we don't use ourselves as an input:
                    else:
                        startVal = genotype[i][valToMutate]
                        while startVal == genotype[i][valToMutate]:
                            newVal = self.getValidInputNodeNumber(
                              i, maxColForward, maxColBack,
                              totalInputCount=totalInputCount,
                              outputSize=outputSize, rows=rows, cols=cols)
                            genotype[i][valToMutate] = newVal

            elif application.lower() == 'pervalue':
                # Check the mutation once for each value in the gene:
                if random.random() <= genMutationRate:
                    startVal = genotype[i][0]
                    while startVal == genotype[i][0]:
                        genotype[i][0] = \
                            random.randint(0, len(functionList) - 1)
                # Mutate the first input:
                if random.random() <= genMutationRate:
                    startVal = genotype[i][1]
                    while startVal == genotype[i][1]:
                        genotype[i][1] = self.getValidInputNodeNumber(
                          i, maxColForward, maxColBack,
                          totalInputCount=totalInputCount,
                          outputSize=outputSize, rows=rows, cols=cols)

                # Mutate the second input:
                if random.random() <= genMutationRate:
                    startVal = genotype[i][2]
                    while startVal == genotype[i][2]:
                        genotype[i][2] = self.getValidInputNodeNumber(
                          i, maxColForward, maxColBack,
                          totalInputCount=totalInputCount,
                          outputSize=outputSize, rows=rows, cols=cols)

                # Mutate the parameter (P):
                if random.random() <= genMutationRate:
                    genotype[i][3] = random.uniform(pRange[0], pRange[1])

            else:
                raise ValueError("Unknown mutation application strategy: %s" %
                                 (application))

    def activeGeneMutate(self, genotype, functionList, pRange, activeGenes,
                         maxColForward, maxColBack, numGenesToMutate=1,
                         totalInputCount=None, outputSize=None,
                         rows=None, cols=None,):
        """Mutate the individual using an active-gene mutation strategy.
        Individuals will definitely want to provide a replacement for this
        function as a generic mutation function will almost never work
        properly for new individual types."""
        if totalInputCount is None:
            totalInputCount = self.totalInputCount

        if outputSize is None:
            outputSize = self.outputSize

        if rows is None:
            rows = self.rows

        if cols is None:
            cols = self.cols

        activeGenesMutated = 0
        loopCount = 0
        while activeGenesMutated < numGenesToMutate:
            geneNum = random.randint(totalInputCount, len(genotype) - 1)

            # If we selected an output gene, we only have one thing to change:
            if len(genotype[geneNum]) == 1:
                startVal = genotype[geneNum][0]
                while startVal == genotype[geneNum][0]:
                    genotype[geneNum][0] = self.getValidInputNodeNumber(
                      geneNum, maxColForward, maxColBack,
                      totalInputCount=totalInputCount, outputSize=outputSize,
                      rows=rows, cols=cols)

            else:  # Standard gene
                # Pick one of its characteristics:
                characteristic = random.randint(0, 3)

                # Pick a new function:
                if characteristic == 0:
                    startVal = genotype[geneNum][0]
                    while startVal == genotype[geneNum][0]:
                        genotype[geneNum][0] = random.randint(
                          0, len(functionList) - 1)

                # Pick a new parameter:
                elif characteristic == 3:
                    genotype[geneNum][3] = random.uniform(pRange[0], pRange[1])

                # Pick a new input:
                else:
                    startVal = genotype[geneNum][characteristic]
                    while startVal == genotype[geneNum][characteristic]:
                        genotype[geneNum][characteristic] = \
                          self.getValidInputNodeNumber(
                            geneNum, maxColForward, maxColBack,
                            totalInputCount=totalInputCount,
                            outputSize=outputSize,
                            rows=rows, cols=cols)

            if geneNum in activeGenes:
                activeGenesMutated += 1
                loopCount = 0

            loopCount += 1
            if loopCount > 100000:
                raise ValueError("Infinite loop.")

    def getRandomizedGenotype(self, functionList, rows, cols, maxColForward,
                              maxColBack, pRange, totalInputCount, outputSize):
        """Return a randomized genotype based upon the given parameters. It
        only uses primitive functions and won't create or use modules."""
        genotype = []
        numFunctions = len(functionList)
        numNodes = (rows * cols) + totalInputCount

        for nodeNum in range(numNodes):
            # Determine the acceptable inputs:
            colNum = self.getColumnNumber(nodeNum,
                                          totalInputCount=totalInputCount,
                                          outputSize=outputSize,
                                          rows=rows, cols=cols)

            # Specially mark input columns:
            if colNum == 0:
                gene = ['IN']
            else:  # Standard column:
                # Full gene: [function, input1(X), input2(Y), parameter(P)]

                # Add the function to the gene:
                gene = [random.randint(0, numFunctions-1)]

                # Add 2 random outputs from the node range:
                gene.append(self.getValidInputNodeNumber(nodeNum,
                  maxColForward, maxColBack,
                  totalInputCount=totalInputCount, outputSize=outputSize,
                  rows=rows, cols=cols))

                gene.append(self.getValidInputNodeNumber(nodeNum,
                  maxColForward, maxColBack,
                  totalInputCount=totalInputCount, outputSize=outputSize,
                  rows=rows, cols=cols))

                # Add our P parameter, Real number:
                gene.append(random.uniform(pRange[0], pRange[1]))

            # Add this gene to the genome:
            genotype.append(gene)

        # Add the outputs. First, get the acceptable range:
        minCol = max(cols + 1 - maxColBack, 0)
        maxCol = cols

        minNodeNum, _ = self.getNodeNumberRange(minCol,
          totalInputCount=totalInputCount, outputSize=outputSize, rows=rows,
          cols=cols)

        _, maxNodeNum = self.getNodeNumberRange(maxCol,
          totalInputCount=totalInputCount, outputSize=outputSize, rows=rows,
          cols=cols)

        for outNum in range(outputSize):
            # Output nodes only have a single value: A node number to output
            genotype.append([random.randint(minNodeNum, maxNodeNum)])

        return genotype

    def getActiveGenes_generic(self, genotype, outputSize):
        """This function is designed to get a list of the genes that are active
        without going through the process of calculating their values."""
        activeGenes = []

        # Get the dependent genes for all outputs:
        for geneNumber in range(len(genotype) - 1,
                                len(genotype) - outputSize - 1,
                                -1):
            self.__setDependentGenesActive(genotype, geneNumber, activeGenes)

        # Return the list:
        return activeGenes

    def __setDependentGenesActive(self, genotype, geneNumber, activeGenes):
        """This is a recursive function built to get a list of all genes that
        contribute to the output of the passed in gene number.

        Arguments:
            genotype - The genotype we're searching.
            geneNumber - The gene number to check.
            activeGenes - The list of active genes to be modified.
        """
        # If we've already checkd this one, we're done:
        if geneNumber in activeGenes:
            return

        # Mark this gene active:
        activeGenes.append(geneNumber)

        # If this is an output or input gene (length 1), they must be special
        # cased:
        if len(genotype[geneNumber]) == 1:
            # Input gene, mark as active, then return.
            if geneNumber < self.totalInputCount:
                return
            # Output gene, check its input:
            else:
                self.__setDependentGenesActive(genotype,
                                               genotype[geneNumber][0],
                                               activeGenes)

        # Standard gene, mark its dependent genes as active:
        else:
            self.__setDependentGenesActive(genotype,
                                           genotype[geneNumber][1],
                                           activeGenes)
            self.__setDependentGenesActive(genotype,
                                           genotype[geneNumber][2],
                                           activeGenes)




####################################
# Below here are functions provided for debugging purposes. They do not serve
# any purpose in training or using individuals, but can sometimes provide
# insight in to a misbehaving training module.
####################################

    def printGivenGenotype(self, genotype):
        for i in range(len(genotype)):
            if not isinstance(genotype[i][0], int) and \
              len(genotype[i]) != 1:
                # Do something different for modules:
                print("%d: " % (i))
                self.printModule(genotype[i])
            else:
                print(str(i) + ": " + str(genotype[i]))

    def getGenotype(self):
        return self.__genotype

    def printGeneOutputs(self, geneOutputs):
        for i in range(len(geneOutputs)):
            print("%d: %s" % (i, str(geneOutputs[i])))
