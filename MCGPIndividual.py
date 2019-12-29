import random
import copy
import inspect
import itertools
import math

from collections import deque

import AbstractCGPIndividual

class MCGPIndividual(AbstractCGPIndividual.AbstractCGPIndividual):
    """Modular CGP allows for the creation, usage, mutation, and deletion of
    modules during the training process. This list of modules is shared among
    an entire population."""
    # Modules are stored as such:
    # [
    #   [Ident Level InCount NodeCount OutCount Type]
    #   [[InNode1 OutNumFromInNode1] ... [InNodeX OutNumFromInNodeX]] {Must be length InCount}
    #   [[SubModule1] ... [SubModuleX]] {Must be length NodeCount}
    #   [[OutNodeNum1 OutNumFromNode1] ... [OutNodeNumX OutNumFromNodeX]] {Must be length OutCount}
    # ]
    # Each submodule can be a primitive function or another module.
    # Primitives within modules are saved as:
    # [FuncNumber [InputMod1 OutputNumberFromInputMod1] [InputMod2 OutputNumberFromInputMod2] P]

    # Static class variables:
    # Module header location definitions:
    MOD_IDENTIFIER = 0
    MOD_LEVEL = 1
    MOD_INPUTS = 2
    MOD_NODES = 3
    MOD_OUTPUTS = 4
    MOD_TYPE = 5 # Should be 1 or 2.  2's cannot be expanded.

    """This represents a single individual in a modular-CGP solution."""
    def __init__(
       self,
       type=None,
       inputSize=1,
       outputSize=1,
       shape=None,
       pRange=None,
       constraintRange=None,
       functionList=None,
       MCGPSpecificParameters=None,
       activeModuleList=None,
       inactiveModuleList=None):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.__genotype = []

        # Make values easier to access later. Check for None, first. It is
        # possible we're being created without these values available and
        # they will be filled in later:
        if self.shape is not None:
            self.cols = self.shape['colCount']
            self.maxColForward = self.shape['maxColForward']
            self.maxColBack = self.shape['maxColBack']
            self.rows = self.shape['rowCount']

        if self.MCGPSpecificParameters is not None:
            self.modulePointMutationProbability = \
              self.MCGPSpecificParameters['pointMutationProbability']
            self.pointMutationCreateNewModule = \
              self.MCGPSpecificParameters['pointMutationCreateNewModule']
            self.moduleAddInputProbability = \
              self.MCGPSpecificParameters['addInputMutationProbability']
            self.moduleRemoveInputProbability = \
              self.MCGPSpecificParameters['removeInputMutationProbability']
            self.moduleAddOutputProbability = \
              self.MCGPSpecificParameters['addOutputMutationProbability']
            self.moduleRemoveOutputProbability = \
              self.MCGPSpecificParameters['removeOutputMutationProbability']
            self.maximumModuleLevel = \
              self.MCGPSpecificParameters['maximumModuleLevel']
            self.maximumModuleSize = \
              self.MCGPSpecificParameters['maximumModuleSize']
            self.compressProbability = \
              self.MCGPSpecificParameters['compressProbability']
            self.expandProbability = \
              self.MCGPSpecificParameters['expandProbability']
            self.primitiveFunctionFocus = \
              self.MCGPSpecificParameters['primitiveFunctionFocus']

        self.integerConversion()

        if self.inputSize is not None:
            self.totalInputCount = self.inputSize

        # To make the syntax easier, separate the shape parameter:
        if self.shape['rowCount'] != 1:
            raise ValueError("MCGP allows only a single row.")

    def integerConversion(self):
        """Convert any values that are needed as integers, but may have been
        passed in as floating point values."""
        super().integerConversion()

        integerList = ['maximumModuleSize', 'maximumModuleLevel']
        for name in integerList:
            setattr(self, name, int(getattr(self, name)))

    def getPercentageNodesUsed(self):
        """Return a list of percentages of nodes used by each genotype in this
        individual."""
        percentages = []

        for i in self.__genotype:
            activeGenes = self.getActiveGenes_generic(i, 1)
            percentages.append((len(activeGenes) / len(i)) * 100.0)

        return percentages

    def getActiveGenes(self):
        """Fill our list of active genes."""
        self.__activeGenes = []

        # Get the dependent genes for all outputs:
        for geneNumber in range(len(self.__genotype) - 1,
                                len(self.__genotype) - \
                                   self.outputSize - 1,
                                -1):
            self.__setDependentGenesActive(geneNumber)

        # Return the list:
        return self.__activeGenes

    def __setDependentGenesActive(self, geneNumber):
        """Add the current gene and all genes it depends upon to our list of
        active genes."""

        if geneNumber in self.__activeGenes:
            return

        # Mark this gene active:
        self.__activeGenes.append(geneNumber)

        try:
            if len(self.__genotype[geneNumber]) == 1: # Input
                return
            elif len(self.__genotype[geneNumber]) == 2: # Output
                self.__setDependentGenesActive(self.__genotype[geneNumber][0])
            # Primitive function, mark its 2 inputs as active:
            elif isinstance(self.__genotype[geneNumber][0], int):
                self.__setDependentGenesActive(self.__genotype[geneNumber][1][0])
                self.__setDependentGenesActive(self.__genotype[geneNumber][2][0])
            else: # Module:
                for inputs in self.__genotype[geneNumber][1]:
                    self.__setDependentGenesActive(inputs[0])
        except:
            print("Hit error. geneNumber: %d" % (geneNumber))
            self.printNumberedGenotype()
            raise

    def getRandomFunctionNumber(self):
        """Return a new function or module number."""
        # If we aren't specifically focusing on primitive functions or
        # modules, or if there are no modules to choose from, just select
        # randomly:
        if self.primitiveFunctionFocus is None or \
          len(self.activeModuleList) == 0:
            funcNum = random.randint(0, self.getMaxFunctionNumber())
        else:
            # Decide between selecting a primitive function and a module:
            if random.random() < self.primitiveFunctionFocus:
                funcNum = random.randint(0, len(self.functionList) - 1)
            else:
                funcNum = random.randint(len(self.functionList),
                                         self.getMaxFunctionNumber())

        return funcNum

    def calculateOutputs(self, inputs):
        """Calculate the output of this individual given these inputs."""
        # Start out with none of the values calculated. Every output is a list
        # to allow for multiple outputs from functions.
        self.__geneOutputs = [[None]] * len(self.__genotype)
        outputs = []

        # Fill in that the inputs have values available:
        for inputNumber in range(len(inputs)):
            self.__geneOutputs[inputNumber] = [inputs[inputNumber]]
            # Bug: The below doesn't work. It fills in the input for the entire
            #      list every time. Need to use the above syntax.
            # self.__geneOutputs[inputNumber][0] = inputs[inputNumber]

        temp = self.getActiveGenes()
        activeGenes = copy.deepcopy(temp)

        # Remove the input and outputs from the activeGene list:
        activeGenes = [x for x in activeGenes if x not in
                       range(self.totalInputCount)]
        activeGenes = [x for x in activeGenes if x not in
                       range(len(self.__genotype) - self.outputSize,
                             len(self.__genotype))]
        activeGenes = sorted(activeGenes)

        # With no recurrent connections allowed, we should be able to just
        # evaluate the needed genes in order:
        for i in range(len(activeGenes)):
            geneNum = activeGenes[i]

            # Decide between a primitive function and a module, then calculate
            # the outputs:
            if isinstance(self.__genotype[geneNum][0], int):
                self.__calculateGeneOutput_primitive(geneNum)
            else: # Must be a module.
                self.__calculateGeneOutput_module(geneNum)

        # Now, we grab the outputs:
        outputs = []
        for geneNum in range(len(self.__genotype) - self.outputSize,
                             len(self.__genotype)):
            outputGeneNum = self.__genotype[geneNum][0]
            outputNumber = self.__genotype[geneNum][1]

            try:
                self.__geneOutputs[geneNum] = \
                  self.__geneOutputs[outputGeneNum][outputNumber]
            except:
                self.printGeneOutputs()
                self.printNumberedGenotype()
                print("geneNum: %s, outputGeneNum: %s, outputNumber: %s" %
                  (str(geneNum), str(outputGeneNum), str(outputNumber)))
                raise

            outputs.append(self.__geneOutputs[geneNum])
            if outputs[len(outputs) - 1] is None:
                self.printNumberedGenotype()
                raise ValueError("Output for gene %d:%d not available." %
                                 (outputGeneNum, outputNumber))

        return outputs

    def __calculateGeneOutput_primitive(self, geneNum):
        """Calculate a specifc gene's output values."""
        X_geneNum = self.__genotype[geneNum][1][0]
        X_outputNum = self.__genotype[geneNum][1][1]
        Y_geneNum = self.__genotype[geneNum][2][0]
        Y_outputNum = self.__genotype[geneNum][2][1]

        X,Y = None, None

        try:
            X = self.__geneOutputs[X_geneNum][X_outputNum]
        except:
            self.printNumberedGenotype()
            self.printGeneOutputs()
            print("geneNum: %d, X_genNum: %d, X_outputNum: %d" %
                  (geneNum, X_geneNum, X_outputNum))
            raise

        try:
            Y = self.__geneOutputs[Y_geneNum][Y_outputNum]
        except:
            self.printNumberedGenotype()
            self.printGeneOutputs()
            print("geneNum: %d, Y_genNum: %d, Y_outputNum: %d" %
                  (geneNum, Y_geneNum, Y_outputNum))
            raise

        if X is None:
            raise ValueError("Inputs not available for %d [%d:%d]" %
                             (geneNum, X_geneNum, X_outputNum))
        if Y is None:
            raise ValueError("Inputs not available for %d [%d:%d]" %
                             (geneNum, Y_geneNum, Y_outputNum))

        # Set our output equal to constrain(P * func(X, Y, P))
        output = self.__constrain(self.__genotype[geneNum][3] * \
           (self.functionList[self.__genotype[geneNum][0]](
              X, Y, self.__genotype[geneNum][3])))
        self.__geneOutputs[geneNum] = [output]

    def __calculateGeneOutput_module(self, geneNum):
        """Calculate all outputs for the given module. Module outputs must be
        calculated recursively to allow nesting."""

        # Gather the module and the inputs it will need:
        module = self.__genotype[geneNum]
        modInputs = []
        for i in module[1]:
            modInputs.append(copy.deepcopy([self.__geneOutputs[i[0]] [i[1]]]))

        modOutputs = self.__doCalculateGeneOutputs_module(module, modInputs)

        self.__geneOutputs[geneNum] = modOutputs

    def __doCalculateGeneOutputs_module(self, module, inputs):
        """Given a module and its inputs, calculate the outputs. This function
        can recursively call itself to calculate the values produced from
        submodules."""

        inputCount = module[0][2]
        if len(inputs) != inputCount:
            raise ValueError("Got %d inputs. Expected %d." %
                             (len(inputs), inputCount))


        # Calculated values of size inputs + nodes.
        moduleCalculatedValues = [[None]] * (inputCount + module[0][3])
        for i in range(inputCount):
            moduleCalculatedValues[i] = inputs[i]

        # Use a queue to "recursively" mark nodes as active.
        activeNodes = []
        markActiveDeque = deque()

        # Add all of our outputs:
        for output in module[3]:
            markActiveDeque.append(output[0])

        # Until the queue is empty, mark each as active and add its dependents
        while len(markActiveDeque) > 0:
            activeVal = markActiveDeque.popleft()

            # already added:
            if activeVal in activeNodes:
                continue

            activeNodes.append(activeVal)

            # Input? Just mark active, no need to add to queue:
            if activeVal < inputCount:
                pass
            # Primitive function:
            elif isinstance(module[2][activeVal - inputCount][0], int):
                markActiveDeque.append(module[2][activeVal - inputCount][1][0])
                markActiveDeque.append(module[2][activeVal - inputCount][2][0])
            else:
                for newInput in module[2][activeVal - inputCount][1]:
                    markActiveDeque.append(newInput[0])

        activeNodes = sorted(activeNodes)

        # Fill in the values of each active node as we go:
        for nodeNum in activeNodes:
            # Input? Go retrieve:
            if nodeNum < inputCount:
                pass # Should have been filled in at the top of the function.

            # Output. Skip, we'll grab our outputs at the end.
            elif nodeNum > inputCount + module[0][3]:
                pass

            # Primitve function:
            else:
                subNode = module[2][nodeNum - inputCount]
                if isinstance(subNode[0], int):
                    X = moduleCalculatedValues[subNode[1][0]][subNode[1][1]]
                    Y = moduleCalculatedValues[subNode[2][0]][subNode[2][1]]
                    P = subNode[3]
                    if X is None or Y is None:
                        raise ValueError("Failed to get value for " + str(subNode))

                    # Apply the function and store the output:
                    moduleCalculatedValues[nodeNum] = \
                      [self.__constrain(self.functionList[subNode[0]](X, Y, P))]

                else: # Submodule. Grab the inputs and pass them in:
                    subModInputs = []
                    for i in subNode[1]:
                        subModInputs.append([moduleCalculatedValues[i[0]][i[1]]])

                    moduleCalculatedValues[nodeNum] = \
                      self.__doCalculateGeneOutputs_module(subNode, subModInputs)


        # Grab our outputs and set them for the world to see:
        outputList = []
        for outNode in module[3]:
            outputList.append(moduleCalculatedValues[outNode[0]][outNode[1]])
            if outputList[len(outputList) - 1] is None:
                print("moduleCalculatedValues: " + str(moduleCalculatedValues))
                raise ValueError("Found a null output.")

        return outputList

    def __constrain(self, value):
        """Constrain the given value to the acceptable provided range."""
        # For now, we're assuming single values (not matrices), so we can just
        # do a bare comparison:
        if self.constraintRange is None:
            pass
        elif value < self.constraintRange[0]:
            value = self.constraintRange[0]
        elif value > self.constraintRange[1]:
            value = self.constraintRange[1]

        return value

    def checkParameters(self):
        """Throw an exception if any of the parameters passed to this class are
        obviously in error or make no sense."""
        if self.shape['rowCount'] != 1:
            raise ValueError("ModularCGP allows only a single row.")

        if self.shape['maxColForward'] > -1:
            raise ValueError("ModularCGP doesn't allow inputs from forward in the graph.")

        if self.activeModuleList is None or self.inactiveModuleList is None:
            raise ValueError("Module lists must be defined.")

    def randomize(self):
        """Randomize this individual. It is assumed that this function is only
        called when there are no modules (only primitive function types used.)
        """
        self.__genotype = []
        numFunctions = len(self.functionList)
        numNodes = self.cols + self.inputSize

        for nodeNum in range(numNodes):
            # Determine the acceptable inputs:
            colNum = self.getColumnNumber(nodeNum)

            # Specially mark input columns:
            if colNum == 0:
                gene = ['IN']
            else:  # Standard gene:
                # Full gene: [function,
                #            [inputFunc1 inputFunc1OutputNumber] (X)
                #            [inputFunc2 inputFunc2OutputNumber] (Y)
                #            parameter(P)]

                # Add the function to the gene:
                gene = [random.randint(0, numFunctions-1)]

                # Add 2 random outputs from the node range with 0s. Primitive
                # functions can only have 1 output:
                gene.append([self.getValidInputNodeNumber(nodeNum,
                                                          self.maxColForward,
                                                          self.maxColBack), 0])
                gene.append([self.getValidInputNodeNumber(nodeNum,
                                                          self.maxColForward,
                                                          self.maxColBack), 0])

                # Add our P parameter, Real number:
                gene.append(random.uniform(self.pRange[0], self.pRange[1]))

            # Add this gene to the genome:
            self.__genotype.append(copy.deepcopy(gene))

        # Add the outputs. First, get the acceptable range:
        minCol = max(self.cols + 1 - self.maxColBack, 0)
        maxCol = self.cols

        minNodeNum, _ = self.getNodeNumberRange(minCol)
        _ , maxNodeNum = self.getNodeNumberRange(maxCol)

        for outNum in range(self.outputSize):
            # Output nodes only have a single value: A node number to output
            self.__genotype.append([random.randint(minNodeNum, maxNodeNum), 0])

    def getOneMutatedChild(self, mutationStrategy):
        """Return a new child based upon a mutation of ourselves.

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

    def __getCopy(self):
        """We need a specialized copy operator because some class variables
        need a shallow copy while others need a deep copy."""
        retInd = copy.copy(self)
        retInd.__genotype = copy.deepcopy(self.__genotype)
        return retInd

    def getModuleDefinition(self, moduleNumber):
        """Get the definition of the module with the given module number. This
        definition will always be a list, and as such, the returned value can
        be modified in order to modify the actual module in the list."""

        if moduleNumber < len(self.functionList):
            raise ValueError("No module %d exists." % (moduleNumber))
        else:
            # Search the active module list:
            for module in self.activeModuleList:
                if module[0][self.MOD_IDENTIFIER] == moduleNumber:
                    return module

            # Search the inactive module list:
            for module in self.inactiveModuleList:
                if module[0][self.MOD_IDENTIFIER] == moduleNumber:
                    return module

        self.printNumberedGenotype()
        raise ValueError("No module %d exists." % (moduleNumber))

    def checkIfModuleExists(self, moduleNumber):
        """Return True if a module with that identifier exists. False
        otherwise."""
        for module in self.activeModuleList:
            if module[0][self.MOD_IDENTIFIER] == moduleNumber:
                return True

        for module in self.inactiveModuleList:
            if module[0][self.MOD_IDENTIFIER] == moduleNumber:
                return True

        return False

    def getNewModuleIdentifier(self):
        """Given the current list of functions and modules, return the next
        available module number. We need to search through the lists because it
        is possible for module X+1 to be created, then module X to be removed.
        The size of the lists does not tell us the module numbers that are
        available."""
        startValue = len(self.functionList)
        newModIdent = startValue
        while True:
            if not self.checkIfModuleExists(newModIdent):
                return newModIdent
            newModIdent += 1

    def getMaxFunctionNumber(self):
        """Determine the maximum value that a function can take on, based upon
        the number of primitive functions that have been provided and the
        current count of active modules.

        This only returns the count! One must be sure to select a module that
        is available within the active module list."""
        return len(self.functionList) + len(self.activeModuleList) - 1

    def __getNewModuleLevel(self, startGene, endGene):
        """Given the start and end genes that we're trying to compress, return
        the level the new module would have to take on. Level is equal to the
        maximum level of the modules to be compressed +1."""
        maximumLevel = 0
        for i in range(startGene, endGene + 1, 1):
            maximumLevel = max(maximumLevel,
                               self.getModuleLevelGivenNodeNumber(i))

        return maximumLevel + 1

    def getModuleNumber(self, nodeNumber):
        """Given a node number in the genotype, return the module or function
        number."""
        # If it is a primitive, just return the integer:
        if isinstance(self.__genotype[nodeNumber][0], int):
            return self.__genotype[nodeNumber][0]
        else: # Pull the module number out of the header:
            return self.__genotype[nodeNumber][0][self.MOD_IDENTIFIER]

    def getModuleLevelGivenNodeNumber(self, nodeNumber):
        """Given a node number in the genotype, return its module level."""
        if isinstance(self.__genotype[nodeNumber][0], int):
            return 0
        else:
            return self.__genotype[nodeNumber][0][self.MOD_LEVEL]

    def getInputsGivenNodeNumber(self, nodeNumber):
        """Get all of the inputs to a given node number."""
        if isinstance(self.__genotype[nodeNumber][0], int):
            return [self.__genotype[nodeNumber][1],
                    self.__genotype[nodeNumber][2]]
        else: # Modules keep a list of inputs:
            return self.__genotype[nodeNumber][1]

    def getMaxOutputNumber(self, nodeNumber):
        """Return the maximum output number allowed for a given module or
        function. All primitive functions are assumed to have a single output,
        meaning they should return 0.
        """
        # If this is an input, just return a single output (max of 0):
        if nodeNumber < self.totalInputCount:
            return 0

        # If this is a primitive function, return 0:
        if isinstance(self.__genotype[nodeNumber][0], int):
            return 0

        # Get the module number for this node:
        moduleNumber = self.getModuleNumber(nodeNumber)

        if moduleNumber < len(self.functionList):
            return 0
        else:
            module = self.getModuleDefinition(moduleNumber)
            return module[0][self.MOD_OUTPUTS] - 1

    def getInputCount(self, moduleNumber):
        if moduleNumber < len(self.functionList):
            return 2
        else:
            module = self.getModuleDefinition(moduleNumber)
            return module[0][self.MOD_INPUTS]

    def getIsPrimitiveFunc(self, moduleNumber):
        if moduleNumber < len(self.functionList):
            return True
        else:
            return False

    def __getProbabilisticMutatedChild(self, genMutationRate=0.1,
                                       outMutationRate=0.1,
                                       application='pergene'):
        child = self.__getCopy()
        child.__probabilisticMutateSelf(genMutationRate=genMutationRate,
                                        outMutationRate=outMutationRate,
                                        application=application)
        return child

    def __getActiveGeneMutatedChild(self, numGenesToMutate=1):
        child = self.__getCopy()
        child.__activeGeneMutateSelf(numGenesToMutate=numGenesToMutate)
        return child

    def __mutateFunctionOfPrimitive(self, geneNum):
        """Given the geneNum, assume it points to a primitive function and
        mutate it, possibly turning it into a module."""

        startFuncNum = self.__genotype[geneNum][0]
        funcNum = self.__genotype[geneNum][0]

        # Still a primitive function? Easy, just change the function number:
        while self.getIsPrimitiveFunc(funcNum) and self.__genotype[geneNum][0] == startFuncNum:
            funcNum = self.getRandomFunctionNumber()
            if self.getIsPrimitiveFunc(funcNum):
                self.__genotype[geneNum][0] = funcNum
            else:
                listIndex = funcNum - len(self.functionList)

                # Collect our current outputs:
                prevOutputs = [self.__genotype[geneNum][1],
                               self.__genotype[geneNum][2]]

                # Move a copy of the module into our genotype:
                self.__genotype[geneNum] = copy.deepcopy(
                  self.activeModuleList[listIndex])

                # Add our previous inputs:
                self.__genotype[geneNum].insert(1, [])
                self.__genotype[geneNum][1].append(copy.deepcopy(prevOutputs[0]))
                if self.__genotype[geneNum][0][self.MOD_INPUTS] > 1:
                    self.__genotype[geneNum][1].append(copy.deepcopy(prevOutputs[1]))

                # Add new inputs until we have enough:
                while len(self.__genotype[geneNum][1]) < \
                  self.__genotype[geneNum][0][self.MOD_INPUTS]:
                    self.__genotype[geneNum][1].append(self.getValidInput(geneNum))

    def __probabilisticMutateSelf(self, genMutationRate=0.1,
                                  outMutationRate=0.1, application='pergene'):

        # Update any modules in our genotype that may have been changed by
        # module mutations elsewhere. Only need to do if we're not creating
        # new modules with point mutations:
        if not self.pointMutationCreateNewModule:
            self.updateModulesFromModuleList()

        # Create or destroy modules:
        self.applyCompressAndExpand()

        for i in range(self.totalInputCount, len(self.__genotype)):
            # Mutate outputs at a different rate than standard genes:
            if len(self.__genotype[i]) == 2:
                if random.random() <= outMutationRate:
                    nodeNum = self.__genotype[i][0]
                    outputNum = self.__genotype[i][1]
                    while self.__genotype[i] == [nodeNum, outputNum]:
                        self.__genotype[i] = self.getValidInput(i)

            # Must be a generic node. Decide between applying the mutation rate
            # per gene or per value inside the gene:
            elif application.lower() == 'pergene':
                # Check the mutation rate once for this gene, then is mutation
                # is selected, choose one value in the gene to mutate:
                if random.random() < genMutationRate:
                    # 2 possibilities: We're found a module or we're
                    # mutating a primitive function.
                    # Primitive functions' first value will be an integer:
                    if isinstance(self.__genotype[i][0], int):
                        valToMutate = random.randint(0, 3)

                        # Mutate the function:
                        if valToMutate == 0:
                            self.__mutateFunctionOfPrimitive(i)

                        # Mutate the p-value:
                        elif valToMutate == 3:
                            self.__genotype[i][3] = random.uniform(self.pRange[0],
                                                                   self.pRange[1])

                        # Otherwise, mutate one of the 2 inputs. Make sure it
                        # changes and we don't use ourselves as an input:
                        else:
                            nodeNum = self.__genotype[i][valToMutate][0]
                            outputNum = self.__genotype[i][valToMutate][1]
                            while self.__genotype[i][valToMutate] == \
                              [nodeNum, outputNum]:
                                 self.__genotype[i][valToMutate] = copy.deepcopy(self.getValidInput(i))

                    else: # Mutating a module:
                        try:
                            self.__mutateModuleNode_perGene(i)
                        except:
                            print("Problem: %d" % (i))
                            # self.printNumberedGenotype()
                            raise

            elif application.lower() == 'pervalue':
                raise ValueError("PerValue mutation cannot be applied in ModularCGP.")

            else:
                raise ValueError("Unknown mutation application strategy: %s" %
                                 (application))

    def applyCompressAndExpand(self):
        """Apply the compress and expand operators to the genotype."""
        if random.random() < self.compressProbability:
            self.__applyCompressOperator()
        if random.random() < self.expandProbability:
            self.__applyExpandOperator()

    def __mutateModuleNode_perGene(self, nodeNum):
        """Call the proper mutation function based upon the module's type."""
        if self.__genotype[nodeNum][0][self.MOD_TYPE] == 1:
            self.__mutateModuleNode_perGene_type1(nodeNum)
        else:
            self.__mutateModuleNode_perGene_type2(nodeNum)

    def __mutateModuleNode_perGene_type2(self, nodeNum):
        """Given a node which is using a type2 module, mutate it."""

        # Choose between its function or one of its inputs:
        if random.random() > 0.5: # If we should modify an input
            # Get a value to mutate:
            valToMutate = random.randint(
              0, self.__genotype[nodeNum][0][self.MOD_INPUTS] - 1)

            self.__genotype[nodeNum][1][valToMutate] = copy.deepcopy(self.getValidInput(nodeNum))

        # Mutate the function:
        else:
            oldInputs = copy.deepcopy(self.__genotype[nodeNum][1])
            newFunc = self.getRandomFunctionNumber()

            # Collect what our current max output is:
            prevMaxOutput = self.__genotype[nodeNum][0][self.MOD_OUTPUTS] - 1
            newMaxOutput = 0 # Assume primitive function, for now.

            # If we just selected a primitive function, copy over our inputs
            # and choose a random P value.
            if self.getIsPrimitiveFunc(newFunc):
                self.__genotype[nodeNum] = [newFunc,
                  copy.deepcopy(oldInputs[0]),
                  copy.deepcopy(oldInputs[1]),
                  random.uniform(self.pRange[0], self.pRange[1])]
            else:
                # Copy over a definition of the module:
                moduleIndex = newFunc - len(self.functionList)
                self.__genotype[nodeNum] = copy.deepcopy(self.activeModuleList[moduleIndex])

                # Modify our old inputs to have the correct number to match
                # this new module's definition:
                while len(oldInputs) < self.__genotype[nodeNum][0][self.MOD_INPUTS]:
                    oldInputs.append(self.getValidInput(nodeNum))

                oldInputs = oldInputs[:self.__genotype[nodeNum][0][self.MOD_INPUTS]]

                # Insert the inputs to our gene:
                self.__genotype[nodeNum].insert(1, oldInputs)

                # Collect our new output count:
                newMaxOutput = self.__genotype[nodeNum][0][self.MOD_OUTPUTS] - 1

            # We reduced the number of available outputs. Any other gene
            # counting on those needs to be modified:
            if newMaxOutput < prevMaxOutput:
                for i in range(nodeNum + 1, len(self.__genotype)):
                    # Primitive function:
                    if isinstance(self.__genotype[i][0], int) and \
                      len(self.__genotype[i]) == 4:
                        # Check each input:
                        for j in range(1, 3):
                            if (self.__genotype[i][j][0] == nodeNum and \
                              self.__genotype[i][j][1] > newMaxOutput):
                                self.__genotype[i][j][1] = random.randint(0, newMaxOutput)
                    # Output:
                    elif len(self.__genotype[i]) == 2:
                        if self.__genotype[i][0] == nodeNum and \
                          self.__genotype[i][1] > newMaxOutput:
                            self.__genotype[i][1] = random.randint(0, newMaxOutput)
                    # Module:
                    else:
                        for modInput in self.__genotype[i][1]:
                            if modInput[0] == nodeNum and \
                              modInput[1] > newMaxOutput:
                                modInput[1] = random.randint(0, newMaxOutput)

    def __mutateModuleNode_perGene_type1(self, nodeNum):
        """Mutate the genotype given the nodenumber in the genotype, making the
        assumption that the given nodenumber is a module."""
        module = copy.deepcopy(self.__genotype[nodeNum])
        valToMutate = random.randint(1, 3)

        # Modify an input:
        if valToMutate == 1:
            inputNum = random.randint(0, module[0][self.MOD_INPUTS] - 1)
            module[1][inputNum] = self.getValidInput(nodeNum)

        # Modify an output:
        elif valToMutate == 2:
            outputNum = random.randint(0, module[0][self.MOD_OUTPUTS] - 1)
            inNodeNum = random.randint(0, module[0][self.MOD_NODES] - 1)

            # Decide on the maximum output from that node:
            maxOut = 0
            if not isinstance(module[2][inNodeNum][0], int):
                maxOut = module[2][inNodeNum][0][self.MOD_OUTPUTS] - 1

            outputSelection = random.randint(0, maxOut)

            # Outputs must be from one of our nodes, not an input:
            inNodeNum += module[0][self.MOD_INPUTS]

            # Set the new output value selection:
            module[3][outputNum] = [inNodeNum, outputSelection]

        # Modify a submodule:
        elif valToMutate == 3:
            # Pick the module number:
            subModNum = random.randint(0, len(module[2]) - 1)
            if isinstance(module[2][subModNum][0], int):
                # Decide to modify the submodule's function, pValue, or inputs:
                subValToMutate = random.randint(0, 3)

                # Mutate the function:
                if subValToMutate == 0:
                    module[2][subModNum][0] = random.randint(0, len(self.functionList) - 1)

                # Mutate the p-value:
                elif subValToMutate == 3:
                    module[2][subModNum][3] = random.uniform(self.pRange[0],
                                                          self.pRange[1])

                # Otherwise, mutate one of the 2 inputs. Make sure it
                # changes and we don't use ourselves as an input:
                else:
                    inNodeNum = random.randint(0,
                      module[0][self.MOD_INPUTS] + subModNum - 1)

                    # If we're pulling from an input, set as such.
                    if inNodeNum < module[0][self.MOD_INPUTS]:
                        module[2][subModNum][subValToMutate] = [inNodeNum, 0]

                    # Otherwise, get the acceptable input range:
                    else:
                        maxOut = 0
                        inNodeIndex = inNodeNum - module[0][self.MOD_INPUTS]
                        if not isinstance(module[2][inNodeIndex][0], int):
                            maxOut = module[2][inNodeIndex][0][self.MOD_INPUTS] - 1
                        outputSelection = random.randint(0, maxOut)

                        # Set the new input into the definition:
                        module[2][subModNum][subValToMutate] = [inNodeNum, outputSelection]
            else:
                print("Can't yet modify submodules from within the genotype point mutation operator.")

        # If we create new modules (don't just change them) we need a new
        # module identifier:
        if self.pointMutationCreateNewModule:
            module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()

        # Copy the new module back into the genotype and module list:
        self.__genotype[nodeNum] = copy.deepcopy(module)
        self.addModuleToActiveModules(module)

    def __activeGeneMutateSelf(self, numGenesToMutate=1):
        """Mutate ourselves using an active-gene mutation strategy."""

        # Update any modules in our genotype that may have been changed by
        # module mutations elsewhere. Only need to do if we're not creating
        # new modules with point mutations:
        if not self.pointMutationCreateNewModule:
            self.updateModulesFromModuleList()

        # Create or destroy modules:
        self.applyCompressAndExpand()

        activeGenesMutated = 0
        loopCount = 0
        activeGenes = self.getActiveGenes()

        while activeGenesMutated < numGenesToMutate:
            geneNum = random.randint(self.totalInputCount,
                                     len(self.__genotype) - 1)

            if geneNum in activeGenes:
                activeGenesMutated += 1

            # Output gene selected:
            if len(self.__genotype[geneNum]) == 2:
                self.__genotype[geneNum] = self.getValidInput(geneNum)

            # Primitive function:
            elif isinstance(self.__genotype[geneNum][0], int):
                valToMutate = random.randint(0, 3)

                if valToMutate == 0: # Function
                    self.__mutateFunctionOfPrimitive(geneNum)
                elif valToMutate == 3: # P-Value
                    self.__genotype[geneNum][3] = random.uniform(self.pRange[0],
                                                                 self.pRange[1])
                else:
                    self.__genotype[geneNum][valToMutate] = copy.deepcopy(
                      self.getValidInput(geneNum))

            # Module:
            if isinstance(self.__genotype[geneNum][0], list):
                self.__mutateModuleNode_perGene(geneNum)

            loopCount += 1
            if loopCount > 10000:
                raise ValueError("Infinite loop.")

    def __applyExpandOperator(self):
        """Pick one module and re-expand it into the genotype."""
        availableModules = []
        for i in range(self.totalInputCount, len(self.__genotype)):
            if not isinstance(self.__genotype[i][0], int):
                availableModules.append(i)

        # The genotype has no modules:
        if len(availableModules) == 0:
            return

        # Select the module to expand:
        nodeNum = availableModules[random.randint(0, len(availableModules)-1)]

        # For now, we only expand level 1, type 1 modules:
        module = self.__genotype[nodeNum]
        if module[0][self.MOD_LEVEL] == 1 and module[0][self.MOD_TYPE] == 1:
            self.__doApplyExpandOperator(nodeNum)
        else:
            pass

    def __doApplyExpandOperator(self, nodeNum):
        module = self.__genotype[nodeNum]
        modIdent = self.__genotype[nodeNum][0][self.MOD_IDENTIFIER]
        inputCount = module[0][2]

        # Change all of the input numbers on the modules in our mod list:
        for inputNum in range(len(module[2])):
            # Primitive function:
            if isinstance(module[2][inputNum][0], int):
                for i in range(1, 3, 1):
                    # If we're requesting a module input, just copy that input.
                    if module[2][inputNum][i][0] < inputCount:
                        module[2][inputNum][i] = copy.deepcopy(module[1][module[2][inputNum][i][0]])
                    else: # Input from somewhere inside the module.
                        module[2][inputNum][i][0] += nodeNum - inputCount
            # Submodule:
            else:
                subModule = node[0]
                for subInNum in range(len(subModule[1])):
                    # Input from outside the module? Just copy over the input:
                    if subModule[1][subInNum][0] < nodeNum:
                        subModule[1][subInNum] = copy.deepcopy(module[1][subModule[1][subInNum][0]])
                    else:
                        subModule[1][subInNum][0] += nodeNum - inputCount

        # Modify all of our outputs to what they will be post-expansion
        for outNode in module[3]:
            outNode[0] += nodeNum - inputCount

        # Modify the input number on functions after this module.
        modDiff = len(module[2]) - 1
        for node in range(nodeNum + 1, len(self.__genotype)):
            # Primitive function, change X and Y:
            if isinstance(self.__genotype[node][0], int)\
              and len(self.__genotype[node]) == 4:
                for i in range(1, 3, 1):
                    # 3 options: Before, after, or in our module:
                    if self.__genotype[node][i][0] < nodeNum: # before
                        pass # No changes necessary
                    elif self.__genotype[node][i][0] > nodeNum: # after
                        self.__genotype[node][i][0] += modDiff
                    else: # In our module
                        outputNumber = self.__genotype[node][i][1]
                        self.__genotype[node][i] = copy.deepcopy(module[3][outputNumber])
            elif len(self.__genotype[node]) == 2: # Output gene:
                # 3 options: Before, after, or in our module:
                if self.__genotype[node][0] < nodeNum: # Before
                    pass
                elif self.__genotype[node][0] > nodeNum: # after
                    self.__genotype[node][0] += modDiff
                else: # In
                    outputNumber = self.__genotype[node][1]
                    self.__genotype[node] = copy.deepcopy(module[3][outputNumber])
            else: # Module
                secondMod = self.__genotype[node]
                for modInNum in range(len(secondMod[1])):
                    # 3 Options: Before, after, or in our module:
                    if secondMod[1][modInNum][0] < nodeNum: # before
                        pass
                    elif secondMod[1][modInNum][0] > nodeNum: # After
                        secondMod[1][modInNum][0] += modDiff
                    else: # Inside
                        outputNumber = secondMod[1][modInNum][1]
                        secondMod[1][modInNum] = copy.deepcopy(module[3][outputNumber])

        # Copy the module's node list into the genotype
        for i in range(len(module[2]) - 1, -1, -1):
            self.__genotype.insert(nodeNum + 1, module[2][i])

        # Delete the module from the genotype.
        del(self.__genotype[nodeNum])

        # Update our number of columns:
        self.cols = len(self.__genotype) - self.totalInputCount - self.outputSize

    def __applyCompressOperator(self):
        """Do the actual compression in the genotype."""
        try:
            startGene = random.randint(self.totalInputCount,
              len(self.__genotype) - self.outputSize - 2)
        except ValueError:
            print("Problem with randint. inputCount: %d, len(geno): %d, outSize: %d"
              % (self.totalInputCount, len(self.__genotype), self.outputSize))
            raise ValueError("...what?")

        endGene = startGene + random.randint(1, self.maximumModuleSize - 1)
        endGene = min(endGene, len(self.__genotype) - self.outputSize - 1)

        newModuleLevel = self.__getNewModuleLevel(startGene, endGene)
        # We don't compress if it will create a module of too high of a level.
        if newModuleLevel > self.maximumModuleLevel:
            return

        # Our module definition:
        modHeader = []
        modInputs = []
        modNodes = []
        modOutputs = []
        modHeader.append(self.getNewModuleIdentifier())
        modHeader.append(newModuleLevel)
        modHeader.append(0) # Input count, fill it in later.
        modHeader.append(endGene - startGene + 1) # Node count
        modHeader.append(0) # Output count, fill it in later.
        modHeader.append(1) # Module type.  All created modules are type 1,
                            # which allows them to be expanded later. Module
                            # copies that get inserted are type 2.

        # Gather our inputs, which include any node input that doesn't come
        # from within the module.
        for nodeNum in range(startGene, endGene + 1, 1):
            geneInputs = self.getInputsGivenNodeNumber(nodeNum)
            for geneInput in geneInputs:
                # If the output is from outside of the new module, add it to
                # the list of our inputs:
                if geneInput[0] < startGene:
                    modInputs.append(copy.deepcopy(geneInput))

        # Set our number of inputs:
        modHeader[self.MOD_INPUTS] = len(modInputs)

        # Now, we can re-number our node inputs:
        # Node inputs from outside the module will be in order
        modInputIndex = 0
        for nodeNum in range(startGene, endGene + 1, 1):
            modNodes.append(copy.deepcopy(self.__genotype[nodeNum]))
            newNodeNum = len(modNodes) - 1

            # Compress primitive functions differently than modules:
            if isinstance(self.__genotype[nodeNum][0], int):
                # If the input comes from outside the module, modify the reference:
                if modNodes[newNodeNum][1][0] < startGene:
                    modNodes[newNodeNum][1] = [modInputIndex, 0]
                    modInputIndex += 1
                else: # Current value - startGene + number of inputs
                    modNodes[newNodeNum][1][0] += -startGene + len(modInputs)

                if modNodes[newNodeNum][2][0] < startGene:
                    modNodes[newNodeNum][2] = [modInputIndex, 0]
                    modInputIndex += 1
                else:
                    modNodes[newNodeNum][2][0] += -startGene + len(modInputs)
            else:
                raise NotImplementedError("Can't compress modules yet.")

        # Modify all geneInputNumbers after our module:
        modDiff = endGene - startGene
        for nodeNum in range(endGene + 1, len(self.__genotype)):
            # Primitive functions, change each X and Y:
            if isinstance(self.__genotype[nodeNum][0], int)\
              and len(self.__genotype[nodeNum]) == 4:
                for i in range(1, 3, 1): # Repeat for X and Y (index 1 and 2)
                    # 3 options: Before, in, or after our module:
                    if self.__genotype[nodeNum][i][0] < startGene: # before
                        pass
                    elif self.__genotype[nodeNum][i][0] > endGene: # after
                        self.__genotype[nodeNum][i][0] -= modDiff
                    else: # In
                        # Get what that output would look like in our module:
                        modifiedVal = copy.deepcopy(self.__genotype[nodeNum][i])
                        modifiedVal[0] += -startGene + len(modInputs)

                        # If it isn't in the list, add it:
                        if not modifiedVal in modOutputs:
                            modOutputs.append(copy.deepcopy(modifiedVal))

                        self.__genotype[nodeNum][i] = \
                          [startGene, modOutputs.index(modifiedVal)]
            elif len(self.__genotype[nodeNum]) == 2: # Output gene:
                if self.__genotype[nodeNum][0] < startGene: # before
                    pass # Do nothing.
                elif self.__genotype[nodeNum][0] > endGene: # after
                    self.__genotype[nodeNum][0] -= modDiff
                else: # In
                    # Get what that output would look like in our module:
                    modifiedVal = copy.deepcopy(self.__genotype[nodeNum])
                    modifiedVal[0] += -startGene + len(modInputs)

                    # If it isn't in the list, add it:
                    if not modifiedVal in modOutputs:
                        modOutputs.append(copy.deepcopy(modifiedVal))

                    self.__genotype[nodeNum] = \
                      [startGene, modOutputs.index(modifiedVal)]
            else: # Module
                # Change each of the module's inputs:
                for subIn in self.__genotype[nodeNum][1]:
                    if subIn[0] < startGene: # before
                        pass # Nothing to do.
                    elif subIn[0] > endGene: # after
                        subIn[0] -= modDiff
                    else: # In
                        # Get what that output would look like in our module:
                        modifiedVal = copy.deepcopy(subIn)
                        modifiedVal[0] += -startGene + len(modInputs)

                        # If it isn't in the list, add it:
                        if not modifiedVal in modOutputs:
                            modOutputs.append(copy.deepcopy(modifiedVal))

                        subIn[0] = startGene
                        subIn[1] = modOutputs.index(modifiedVal)

        # It is possible for a module to be built with 0 outputs. If that
        # happened, we will artificially add one:
        if len(modOutputs) == 0:
            # Select a random node:
            randSubMod = random.randint(0, len(modNodes) - 1)

            # Select one of its outputs at random:
            maxOutput = 0
            if not isinstance(modNodes[randSubMod][0], int):
                maxOutput = modNodes[randSubMod][0][self.MOD_OUTPUTS]

            randSubModOutput = random.randint(0, maxOutput)

            # Add the input count to get to the actual module number:
            randSubMod += len(modInputs)

            # Fill it in to our module:
            modOutputs = [[randSubMod, randSubModOutput]]

        # Set our number of outputs:
        modHeader[self.MOD_OUTPUTS] = len(modOutputs)

        # Build our full header:
        fullModule = [modHeader, modInputs, modNodes, modOutputs]

        # Add it to our list of modules:
        self.addModuleToActiveModules(fullModule)

        # Insert the module into our genotype:
        self.__genotype.insert(startGene, fullModule)

        # Delete the replaced modules:
        for i in range(startGene, endGene + 1):
            del self.__genotype[startGene + 1]

        # Update our number of columns:
        self.cols = len(self.__genotype) - self.totalInputCount - self.outputSize

    def getValidInput(self, nodeNum):
        """Get a new valid input ([nodeNum, outputNum]) given the provided
        nodenumber that needs an input."""
        newInNum = self.getValidInputNodeNumber(nodeNum, self.maxColForward,
                                                self.maxColBack)
        newInput = [newInNum,
          random.randint(0, self.getMaxOutputNumber(newInNum))]

        return newInput

    def updateModulesFromModuleList(self):
        """Update the modules in our genotype in case module mutations have
        modified them."""
        # For now, assume that adding/removing inputs/outputs will create new
        # modules rather than changing them in place.
        self.updateModulesFromModuleList_noInOutUpdates()
        #self.updateModulesFromModuleList_inOutUpdates()

    def updateModulesFromModuleList_noInOutUpdates(self):
        """Update the modules in our genotype from the module list while
        assuming that the input and output counts will remain the same."""
        for nodeNum in range(len(self.__genotype)):
            # Check that it is a type 2 mod, we don't want to replace type 1:
            if not isinstance(self.__genotype[nodeNum][0], int) and \
              len(self.__genotype[nodeNum]) == 4 and \
              self.__genotype[nodeNum][0][self.MOD_TYPE] == 2:

              # Copy our old inputs:
              oldInputs = copy.deepcopy(self.__genotype[nodeNum][1])

              # Copy in the module definition:
              self.__genotype[nodeNum] = copy.deepcopy(
                self.getModuleDefinition(self.__genotype[nodeNum][0][self.MOD_IDENTIFIER]))

              # Put the inputs back in place:
              self.__genotype[nodeNum].insert(1, oldInputs)

    def updateModulesFromModuleList_inOutUpdates(self):
        """Update the modules in our genotype from the module list while making
        no assumptions about changes to the input/output count."""
        for nodeNum in range(len(self.__genotype)):
            # Found a module:
            if isinstance(self.__genotype[nodeNum][0], int) and \
              len(self.__genotype) == 4 and \
              self.__genotype[nodeNum][0][self.MOD_TYPE] == 2:
                oldInputs = copy.deepcopy(self.__genotype[nodeNum][1])
                oldOutputCount = self.__genotype[nodeNum][0][self.MOD_OUTPUTS]

                # Copy in the module definition:
                self.__genotype[nodeNum] = copy.deepcopy(
                  self.getModuleDefinition(self.__genotype[nodeNum][0][self.MOD_IDENTIFIER]))

                # Modify our inputs to match the new number of inputs we need:
                oldInputs = oldInputs[:self.__genotype[nodeNum][0][self.MOD_INPUTS]]
                while len(oldInputs) < self.__genotype[nodeNum][0][self.MOD_INPUTS]:
                    oldInputs.append(self.getValidInput(nodeNum))

                # Put our inputs back in place:
                self.__genotype[nodeNum].insert(1, oldInputs)

                # If we've reduced the number of outputs this module has, we
                # need to modify any module down the line that was expecting
                # that output to be available:
                if oldOutputCount > self.__genotype[nodeNum][0][self.MOD_OUTPUTS]:
                    newMax = self.__genotype[nodeNum][0][self.MOD_OUTPUTS] - 1
                    for modNodeNum in range(nodeNum + 1,
                                            range(len(self.__genotype))):

                        # Primitive function:
                        if isinstance(self.__genotype[modNodeNum][0], int) and \
                          len(self.__genotype[modNodeNum]) == 4:
                            # Check both inputs:
                            for i in range(1, 3, 1):
                                if self.__genotype[modNodeNum][i][0] == nodeNum \
                                  and self.__genotype[modNodeNum][i][1] > newMax:
                                     # Choose a new random input:
                                     self.__genotype[modNodeNum][i] = self.getValidInput(modNodeNum)

                        # Output node:
                        elif len(self.__genotype[modNodeNum]) == 2:
                            if self.__genotype[modNodeNum][0] == nodeNum and \
                              self.__genotype[modNodeNum][1] > newMax:
                                # Choose new random value:
                                self.__genotype[modNodeNum] = self.getValidInput(modNodeNum)

                        # Module:
                        else:
                            for inNum in range(len(self.__genotype[modNodeNum][1])):
                                if self.__genotype[modNodeNum][1][inNum][0] == nodeNum \
                                  and self.__genotype[modNodeNum][1][inNum][1] > newMax:
                                    self.__genotype[modNodeNum][1][inNum] = \
                                      self.getValidInput(modNodeNum)


    def addModuleToActiveModules(self, module, removeInputs=True):
        """Add the provided module to our module list. It is assumed that we're
        being handed a newly-compressed module and that its definition needs
        to be modified somewhat to fit into our list."""
        listModule = copy.deepcopy(module)

        # Index 1 refers to the module's inputs, which are decided each time
        # that the module is used in a genotype.
        if removeInputs:
            del(listModule[1])

        # Set the module type. Default to 2 in the list because that's what
        # it will be if any genotype brings it in.
        listModule[0][self.MOD_TYPE] = 2

        # If we already have the module, update it:
        for i in range(len(self.activeModuleList)):
            if self.activeModuleList[i][0][self.MOD_IDENTIFIER] == \
              listModule[0][self.MOD_IDENTIFIER]:
                self.activeModuleList[i] = copy.deepcopy(listModule)
                return

        # Didn't find a module to update, add it:
        self.activeModuleList.append(listModule)

    def removeModuleFromActiveModules(self, moduleNumber):
        """Take a module currently on the active module list and remove it.
        We will always move active modules to the inactive module list. If it
        is found to not be needed, it can be fully deleted there."""

        for i in range(len(self.activeModuleList)):
            if self.activeModuleList[i][0][self.MOD_IDENTIFIER] == moduleNumber:
                self.inactiveModuleList.append(self.activeModuleList[i])
                del(self.activeModuleList[i])
                break

    def removeModuleFromInactiveModules(self, moduleNumber):
        """Fully delete a module from the inactive modules list."""
        for i in range(len(self.inactiveModuleList)):
            if self.inactiveModuleList[i][0][self.MOD_IDENTIFIER] == moduleNumber:
                del(self.inactiveModuleList[i])
                break

    def performOncePerEpochUpdates(self, listAllIndividuals, fitnesses):
        """This function is assumed to be called once per epoch and only on a
        single indivdiual in the population. It needs to update all of the
        variables shared across the population."""
        # Apply our module mutations:
        self.mutateAllModules()
        self.moduleListCleanup(listAllIndividuals)
        return listAllIndividuals

    def mutateAllModules(self):
        """Go through the list of active modules and apply each required
        mutation."""

        # For now, changes to the number of inputs or outputs will create
        # a new module. Later, we may try modifying the module in place.
        for i in range(len(self.activeModuleList)):
            # The point mutation operator is applied every time; the random
            # chance of changing or not changing is calculated within:
            self.applyModulePointMutation(i,
              createNewModule=self.pointMutationCreateNewModule)

            if random.random() < self.moduleAddInputProbability:
                self.applyModuleAddInput(i)

            if random.random() < self.moduleRemoveInputProbability:
                self.applyModuleRemoveInput(i)

            if random.random() < self.moduleAddOutputProbability:
                self.applyModuleAddOutput(i)

            if random.random() < self.moduleRemoveOutputProbability:
                self.applyModuleRemoveOutput(i)

    def applyModulePointMutation(self, indexNumber, createNewModule=True):
        """Apply the point mutation to the module in the active module list at
        the given index."""
        module = None
        if createNewModule:
            module = copy.deepcopy(self.activeModuleList[indexNumber])
        else:
            module = self.activeModuleList[indexNumber]

        numInputs = module[0][self.MOD_INPUTS]
        numSubModules = module[0][self.MOD_NODES]
        madeChange = False
        for subMod in range(numSubModules):
            # Submodule is a primitive function:
            if isinstance(module[1][subMod][0], int):
                # Check each input:
                for i in range(1, 3, 1):
                    if random.random() < self.modulePointMutationProbability:
                        madeChange = True
                        newInNum = random.randint(0, numInputs + subMod - 1)

                        # For now, assuming all submodules must be primitive
                        # functions with only a single output:
                        module[1][subMod][1] = [newInNum, 0]

        # Modify where the outputs are getting their values:
        for out in range(len(module[2])):
            if random.random() < self.modulePointMutationProbability:
                madeChange = True
                newInNum = random.randint(numInputs,
                                          numInputs + numSubModules - 1)

                # For now, assuming all submodules are primitive functions with
                # only a single output:
                module[2][out] = [newInNum, 0]

        # Update our module identifier if we are creating new modules:
        if madeChange and createNewModule:
            module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()

        # Report the change to the list:
        if madeChange:
            self.addModuleToActiveModules(module, removeInputs=False)

    def applyModuleAddInput(self, indexNumber):
        """Create a new module that is a duplicate of the active module at the
        given index with an extra input."""
        # If we can add an input, add it:
        if self.activeModuleList[indexNumber][0][self.MOD_INPUTS] < \
           2 * self.activeModuleList[indexNumber][0][self.MOD_NODES]:
            module = copy.deepcopy(self.activeModuleList[indexNumber])
            module[0][self.MOD_INPUTS] += 1
            module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()
            self.addModuleToActiveModules(module, removeInputs=False)

    def applyModuleRemoveInput(self, indexNumber):
        """Create a new module that is a duplicate of the active module at the
        given index with one input removed."""
        # If we can remove an input, remove it:
        if self.activeModuleList[indexNumber][0][self.MOD_INPUTS] > 2:
            module = copy.deepcopy(self.activeModuleList[indexNumber])
            module[0][self.MOD_INPUTS] -= 1
            module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()
            self.addModuleToActiveModules(module, removeInputs=False)

    def applyModuleAddOutput(self, indexNumber):
        """Create a new module that is a duplicate of the active module at the
        given index with an extra output."""
        module = copy.deepcopy(self.activeModuleList[indexNumber])

        # Find an output we can add, assuming for now that all submodules are
        # primitive functions:
        possibleOutputs = []
        for i in range(module[0][self.MOD_INPUTS],
                       module[0][self.MOD_INPUTS] + module[0][self.MOD_NODES],
                       1):
            if not [i, 0] in module[2]:
                possibleOutputs.append([i, 0])

        # No more outputs can be added:
        if len(possibleOutputs) == 0:
            return

        # Add one at random:
        outToAdd = random.randint(0, len(possibleOutputs) - 1)
        module[0][self.MOD_OUTPUTS] += 1
        module[2].append(possibleOutputs[outToAdd])

        # Put it in our list:
        module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()
        self.addModuleToActiveModules(module, removeInputs=False)

    def applyModuleRemoveOutput(self, indexNumber):
        """Create a new module that is a duplicate of the active module at the
        given index with an output removed."""
        # Make sure we can remove one:
        if self.activeModuleList[indexNumber][0][self.MOD_OUTPUTS] <= 1:
            return

        module = copy.deepcopy(self.activeModuleList[indexNumber])

        # Remove one at random:
        outToRemove = random.randint(0, module[0][self.MOD_OUTPUTS] - 1)
        del(module[2][outToRemove])
        module[0][self.MOD_OUTPUTS] -= 1

        # Add it to our active module list:
        module[0][self.MOD_IDENTIFIER] = self.getNewModuleIdentifier()
        self.addModuleToActiveModules(module, removeInputs=False)

    def moduleListCleanup(self, listAllIndividuals):
        """Remove all modules that are not needed anymore from the active and
        inactive module lists."""
        # Get lists of all modules used by the population:
        usedModules = []
        for ind in listAllIndividuals:
            for gene in ind.__genotype:
                if not isinstance(gene[0], int):
                    modIdent = gene[0][self.MOD_IDENTIFIER]
                    if not modIdent in usedModules:
                        usedModules.append(modIdent)

        # Create the list of modules to remove:
        modulesToRemove = []
        for mod in self.activeModuleList:
            if not mod[0][self.MOD_IDENTIFIER] in usedModules and \
              not mod[0][self.MOD_IDENTIFIER] in modulesToRemove:
                modulesToRemove.append(mod[0][self.MOD_IDENTIFIER])

        # Remove said modules:
        for mod in modulesToRemove:
            self.removeModuleFromActiveModules(mod)

        # When we implement higher level modules, we will need to clean up
        # the inactive module list.  For now, though, just empty it:
        self.inactiveModuleList = []

    def __addSubmoduleToRequiredList(self, module, reqSubMod):
        """Recursively add this module and all of its submodule to the list of
        modules which must be kept around."""
        if module[0][self.MOD_IDENTIFIER] not in reqSubMod:
            reqSubMod.append(module[0][self.MOD_IDENTIFIER])
            for subMod in module[1]:
                if not isinstance(subMod[0], int):
                    self.__addSubmoduleToRequiredList(subMod, reqSubMods)

    def getNumberOfModulesAndPrimitiveFunctions(self):
        """To make tracking module usage easier, get the number of modules and
        primitive functions in the genotype. This won't count number of unique
        modules, just the number of times that modules are used."""
        numModules = 0
        numPrimitives = 0
        for gene in self.__genotype:
            if isinstance(gene[0], list):
                numModules += 1
            elif isinstance(gene[0], int) and len(gene) == 4:
                numPrimitives += 1
        return numModules, numPrimitives

####################################
# Below here are functions only available for debugging. They allow for the
# testing of specific operators and printing of results to the screen.
####################################

    def printNumberedGenotype(self):
        """Utility function to just give an ordered list of all nodes."""
        for i in range(len(self.__genotype)):
            if not isinstance(self.__genotype[i][0], int) and \
              len(self.__genotype[i]) != 1:
                # Do something different for modules:
                print("%d: " % (i))
                self.printModule(self.__genotype[i])
            else:
                print(str(i) + ": " + str(self.__genotype[i]))

    def printModule(self, module):
        print("[")
        print("   " + str(module[0]))
        print("   " + str(module[1]))
        print("   [")
        for submod in module[2]:
            print("      " + str(submod))
        print("   " + str(module[3]))
        print("]")

    def printListModule(self, module):
        """Print a module from the module list (no inputs provided)."""
        print("[")
        print("   " + str(module[0]))
        print("   [")
        for submod in module[1]:
            print("      " + str(submod))
        print("   " + str(module[2]))
        print("]")

    def printGeneOutputs(self):
        for i in range(len(self.__geneOutputs)):
            print("%d: %s" % (i, str(self.__geneOutputs[i])))

    def applyCompress(self):
        self.__applyCompressOperator()

    def applyExpand(self):
        self.__applyExpandOperator()

    def applyProbabilisticMutation(self):
        self.__probabilisticMutateSelf()

    def getActiveModules(self):
        return self.activeModuleList

    def getInactiveModules(self):
        return self.inactiveModuleList

    def getAllInputsValid(self):
        # Step through each gene and make sure it is asking for valid inputs:
        for geneNum in range(len(self.__genotype)):
            # Input node:
            if self.__genotype[geneNum] == ['IN']:
                continue
            # Output node:
            elif len(self.__genotype[geneNum]) == 2:
                inGeneNum = self.__genotype[geneNum][0]
                maxOutVal = self.getMaxOutputNumber(inGeneNum)
                if self.__genotype[geneNum][1] > maxOutVal:
                    # print("A: %d requested output %d from %d." %
                    #       (geneNum, self.__genotype[geneNum][1],
                    #        self.__genotype[geneNum][0]))
                    return False
            # Primitive function:
            elif isinstance(self.__genotype[geneNum][0], int):
                for i in range(1, 3, 1):
                    inGeneNum = self.__genotype[geneNum][i][0]
                    maxOutVal = self.getMaxOutputNumber(inGeneNum)
                    if self.__genotype[geneNum][i][1] > maxOutVal:
                        print("B: %d requested output %d from %d." %
                              (geneNum, self.__genotype[geneNum][i][1],
                               self.__genotype[geneNum][i][0]))
                        return False
            # Module
            else:
                for input in self.__genotype[geneNum][1]:
                    maxOutVal = self.getMaxOutputNumber(input[0])
                    if input[1] > maxOutVal:
                        print("C: %d requested output %d from %d." %
                              (geneNum, input[1], input[0]))
                        return False

        return True

    def getFunctionTypeNumbers(self):
        numPrimitiveFunc, numModules = 0, 0
        for gene in self.__genotype:
            # Skip in and out:
            if gene == ['IN'] or len(gene) == 2:
                pass
            elif isinstance(gene[0], int):
                numPrimitiveFunc += 1
            else:
                numModules += 1

        return numPrimitiveFunc, numModules
