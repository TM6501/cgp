import random
import copy
import inspect

import AbstractCGPIndividual

class PCGP1DIndividual(AbstractCGPIndividual.AbstractCGPIndividual):
    """This class represents a PCGP individual as described by 'Positional
    Cartesian Genetic Programming' arXiv: 1810.04119v1

    It allows for various forms of mutation and crossover."""

    def __init__(self, type=None, inputSize=None, outputSize=None, pRange=None,
                 constraintRange=None, shape=None, functionList=None,
                 PCGP1DSpecificParameters=None):

        """Set all training variables and initialize the class."""

        # Collect all of our variables.
        # We must accept a shape parameter because all other individuals do,
        # though we don't use it.
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        # Nodes will be stored as dictionaries to make processing easier.
        # Input and computational nodes will be kept in the genotype. The
        # output nodes don't have positions and will be kept separate:
        self.__genotype = []
        self.__outputNodes = []

        self.totalInputCount = self.inputSize
        self.__activeGenes = None

        # Certain variables are required to be in our variation-specific
        # parameter.  Check that they are:
        requiredVSPParams = ['iStart',  # Istart
                             'recursive',  # r
                             'weights',  # w, kinda. List of len 2
                             'minComputationNodes',  # sizeMin
                             'maxComputationNodes',  # sizeMax
                             'inputsPerNode' # List of len 2
                            ]

        for param in requiredVSPParams:
            if param not in PCGP1DSpecificParameters:
                raise ValueError("%s must be in the variation specific parameters."
                                 % (param))

        # Gather the parameters into class variables:
        self.iStart = PCGP1DSpecificParameters['iStart']
        self.recursive = PCGP1DSpecificParameters['recursive']
        self.weights = PCGP1DSpecificParameters['weights']
        self.minComputationNodes = PCGP1DSpecificParameters['minComputationNodes']
        self.maxComputationNodes = PCGP1DSpecificParameters['maxComputationNodes']
        self.inputsPerNode = PCGP1DSpecificParameters['inputsPerNode']

        # There are many possibly parameters to modify how training proceeds.
        # GeneralCGPSolver will track our mutation strategy; we must track
        # crossover choices and how implement them.

        # Crossover will be an array of dictionaries.  Each dictionary is
        # expected to hold the name of a crossover technique to apply ('name'),
        # as well as, 'numIndividualsToCreate'. Any other variables that the
        # particular crossover technique requires can also be stored in this
        # dictionary.
        # The crossover techniques currently available are:
        #    singlePoint, randomNode, alignedNode, proportional, outputGraph,
        #    subgraph
        self.crossover = PCGP1DSpecificParameters['crossover']

        # Convert arguments to integer and check for errors.
        self.integerConversion()
        self.checkClassVariablesForErrors()

        # Distribute our functions between 0.0 and 1.0:
        self.createFunctionDistribution()

    def createFunctionDistribution(self):
        """Take our function list and position each function between 0.0 and
        1.0. Since nodes 'snap' to the nearest function, we'll add min/max to
        each function so that we don't need to calculate nearest, just find
        the function with the appropriate min/max."""
        rangePerFunction = 1.0 / float(len(self.functionList))
        minVal = 0.0
        self.functionDistribution = []
        for func in self.functionList:
            funcNode = {'min': minVal,
                        'max': minVal + rangePerFunction,
                        'func': func}
            self.functionDistribution.append(funcNode)
            minVal += rangePerFunction

        # To avoid math issues, make sure the final function goes up to 1.0:
        self.functionDistribution[-1]['max'] = 1.0

    def checkClassVariablesForErrors(self):
        """Go through our class variables and check for obvious errors or
        values which will cause problems in training. Throw an exception if
        a bad value is found."""
        # iStart defines the minimum location for inputs. 0-1.0 contains
        # computational nodes, while inputs are gathered from less than 0.
        # As the input space grows, so too does the chances of nodes taking
        # their inputs from problems inputs rather than other computational
        # nodes.  Min: -1.0.  Max -0.01
        if not (-1.0 <= self.iStart <= -0.01):
            raise ValueError("iStart must be between -1.0 and -0.01 inclusive.")

        # recursive vaguely defines how recursive nodes can be. With a value
        # of 0.0, nodes must always select inputs from nodes with lower
        # positions. With a value of 1.0, nodes are free to select inputs from
        # any other node.  Values between the two vary how much recursiveness
        # is used.
        if not (0.0 <= self.recursive <= 1.0):
            raise ValueError("recursive must be between 0.0 and 1.0 inclusive.")

        # Weights are applied in PCGP to the output of functions rather than
        # the input. The weights variable is a range of values that can be
        # multiplied by the output before passing it on to other nodes. To not
        # allow weights to be mutated or included as a factor, both could be
        # set to 1.0.
        if (not isinstance(self.weights, list)) or len(self.weights) != 2:
            raise ValueError("weights must be a list of length 2.")

        # Make sure they are in ascending order:
        self.weights[0], self.weights[1] = min(self.weights), max(self.weights)

        # minComputationNodes and maxComputationNodes define the size of
        # the genotype outside of the inputs and outputs. As long as they are
        # integers in the correct order, there are no further restrictions:
        if not (isinstance(self.minComputationNodes, int) and \
          isinstance(self.maxComputationNodes, int) and \
          self.minComputationNodes <= self.maxComputationNodes):
            raise ValueError("minComputationNodes must be <= maxComputationNodes.")

        # inputsPerNode Must be a list of length 2 defining the minimum and
        # maximum number of inputs per node. These values can be identical if
        # evolution should be used to change the number of inputs per node.
        # The minimum must be 1 or greater.
        if (not isinstance(self.inputsPerNode, list)) or len(self.inputsPerNode) != 2:
            raise ValueError("inputsPerNode must be a list of length 2.")

        # Make sure they are in ascending order:
        self.inputsPerNode[0], self.inputsPerNode[1] = \
          int(min(self.inputsPerNode)), int(max(self.inputsPerNode))

        # Crossover must be a list of dictionaries. Each dictionary must
        # include at least 'name' and 'numIndividualsToCreate':
        if not isinstance(self.crossover, list):
            raise ValueError("crossover must be a list of dictionaries.")

        for method in self.crossover:
            if not isinstance(method, dict):
                raise ValueError("crossover must be a list of dictionaries.")
            if not ('name' in method and 'numIndividualsToCreate' in method):
                raise ValueError("Every crossover method must include 'name' \
and 'numIndividualsToCreate'")

        if self.functionList is None or not isinstance(self.functionList, list):
            raise ValueError("functionList must be a list of functions.")

    def randomize(self):
        """Randomize our genotype by adding random nodes, then sorting those
        nodes into positional order."""
        pass

    def addComputationalNodeToGenotype(self, node):
        """Add a node to our genotype while maintaining the correct ordering."""
        # Use a few special cases to make the general algorithm easy to write:
        if len(self.__genotype) == 0:
            self.__genotype.append(node)

        elif len(self.__genotype) == 1:
            if node['position'] < self.__genotype[0]['position']:
                self.__genotype.insert(0, node)
            else:
                self.__genotype.append(node)

        elif len(self.__genotype) == 2:
            if node['position'] < self.__genotype[0]['position']:
                self.__genotype.insert(0, node)
            elif self.__genotype[0]['position'] < node['position'] < self.__genotype[1]['position']:
                self.__genotype.insert(1, node)
            else:
                self.__genotype.append(node)

        else:  # > 2 nodes:
            # Scan forward linearally to find the location to insert:
            inserted = False
            for i in range(len(self.__genotype)):
                if node['position'] < self.__genotype[i]['position']:
                    self.__genotype.insert(i, node)
                    inserted = True
                    break

            # It should be last because we didn't insert it elsewhere:
            if not inserted:
                self.__genotype.append(node)

    def createRandomComputationalNode(self):
        """Create and return a single computational node filled randomly
        within the bounds of the parameters passed to this class.
        Computational node structure:
        {
          'type': 'computational',
          'position': <float between 0.0 and 1.0>
          'functionPosition': <float indicating the function position>
          'inputPositions': [list of locations to gather inputs]
          'weight': <float to multiply our output by>
        }"""
        pos = random.uniform(0.0, 1.0)

        retNode = {
          'type': 'computational',
          'position': pos,
          'functionPosition': random.uniform(0.0, 1.0),
          'inputPositions': [],
          'weight': random.uniform(self.weights[0], self.weights[1])
        }

        # Add as many inputs as requested:
        inputsToAdd = random.randint(self.inputsPerNode[0], self.inputsPerNode[1])
        for i in range(inputsToAdd):
            # In order to take iStart and recursive into account, we can't
            # just select a random value:
            tempPos = random.random() * ((self.recursive * (1.0 - pos) + pos)\
              - self.iStart) + self.iStart
            retNode['inputPositions'].append(tempPos)

        return retNode

    def calculateOutputs(self, inputs):
        """Determine our output values using the available inputs and our
        genotype."""
        pass

    def getOneMutatedChild(self, mutationStrategy):
        """Return a new individual mutated from ourselves."""
        # PCGP allows 3 types of mutation: gene, mixedNode, and mixedSubgraph:
        strategy = mutationStrategy['name'].lower()
        if strategy not in ['gene', 'mixedNode', 'mixedSubgraph']:
            raise ValueError("Invalid mutation strategy.")

        # Allocate the child that will self-mutate:
        child = deepcopy(self)

        if strategy == 'gene':
            # Gene mutation modifies computational nodes, output nodes,
            # and the position of input nodes. If required by the active
            # variable, this process repeats until at least one active
            # node is mutated.
            child.geneMutate(nodeCompChance=mutationStrategy['mComp'],
                             nodeOutChance=mutationStrategy['mOut'],
                             nodeInChance=mutationStrategy['mIn'],
                             active=mutationStrategy['mActive'])
        elif strategy == 'mixedNode':
            # Mixed node uses mModify, mAdd, and the current size of the
            # genome to decide between performing a geneMutation, adding
            # nodes, or deleting nodes. mDelta helps determine how many to
            # add or remove.
            child.mixedNodeMutate(mModify=mutationStrategy['mModify'],
                                  mAdd=mutationStrategy['mAdd'],
                                  mDelta=mutationStrategy['mDelta'])
        elif strategy == 'mixedSubgraph':
            # Like mixedNode, except instead of using node addition and node
            # deletion, subgraph addition and subgraph deletion are used.
            child.mixedSubgraphMutate(mModify=mutationStrategy['mModify'],
                                      mAdd=mutationStrategy['mAdd'],
                                      mDelta=mutationStrategy['mDelta'])
        else:
            # Bad strategy should be caught above, not here.
            raise ValueError("Unrecognized mutation strategy: %s" % (strategy))

        return child

    def geneMutate(self, nodeCompChance=0.0, nodeOutChance=0.0, nodeInChance=0.0, active=False):
        """Mutate ourselves by randomly changing the position of input nodes,
        where output nodes pull their inputs from, and the various values
        in computational nodes."""
        pass

    def mixedNodeMutate(self, mModify=1.0, mAdd=0.0, mDelta=0.0):
        """Mutate ourselves, choosing between gene mutation, adding nodes, and
        deleting nodes."""
        pass

    def mixedSubgraphMutate(self, mModify=1.0, mAdd=0.0, mDelta=0.0):
        """Mutate ourselves, choosing between gene mutation, adding subtrees,
        and deleting subtrees."""
        pass

    def performOncePerEpochUpdates(self, listAllIndividuals, epochFitnesses):
        """Produce new crossover individuals and add them to our list of all
        individuals. This list will be treated as the parents by the main
        training algorithm."""
        return listAllIndividuals

    def integerConversion(self):
        """Convert the input parameters that must be converted to integers."""
        self.minComputationNodes = int(self.minComputationNodes)
        self.maxComputationNodes = int(self.maxComputationNodes)

    def constrain(self, value):
        """Constrain outputs of nodes to a specific range."""
        # If we have a constraint range, use it:
        if self.constraintRange is not None:
            return max(min(value, self.constraintRange[1]), self.constraintRange[0])
        else:  # Otherwise, just return the value:
            return value

    def resetForNewTimeSeries(self):
        """Reset ourselves for a new series of inputs.  This is important due
        to our possibly recursive nature; we need to set up default outputs
        for nodes that may not have calculated their outputs yet."""
        pass

    def getPercentageNodesUsed(self):
        """Return the percentage of our genotype that is being used to help
        calculate outputs."""
        return 0.0

    #########################
    # DEBUG METHODS #
    #########################
    def getGenotype(self):
        return self.__genotype

    def printGenotype(self):
        for node in self.__genotype:
            for k,v in node.items():
                print("'%s': %s" % (k, str(v)))
            print("-------------------")
