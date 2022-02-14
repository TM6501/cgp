import inspect
import itertools
import operator
import random
import datetime
import copy
import os
import time
import pickle

# Recording results:
import wandb

# Concurrent.futures.ProcessPoolExecutor is used instead of
# multiprocessing.Pool because it allows multiprocessing within multiprocessing.
# Gathering fitnesses is done in a multithreaded manner inside this class.
# By allowing Pools within Pools, individuals can use multiprocessing fitness
# functions.
from concurrent.futures import ProcessPoolExecutor as Pool

# CGP Miscellaneous:
import CGPFunctions
import GymFitnessFunctions
import FitnessCollapseFunctions
import TrainingOptimizers

# Individual types:
import CGPIndividual
import MCGPIndividual
import MCCGPIndividual
import FFCGPANNIndividual
import MCFFCGPANNIndividual
import RCGPANNIndividual

########## IMPORTANT NOTE 1 #########
# If multithreading is used, any fitness function MUST be included by this
# script and accept a single tuple as input. Said
# tuple will be the individual to be tested, the input, and the expected
# output. If it is a class function, you must allocate the class outside
# and pass in an instance of the function.
# If it cannot meet those requirements, numThreads must be set to 1.
# The arguments to the function must remain as a single tuple in any case.
######### IMPORTANT NOTE 1 ##########

######## IMPORTANT NOTE 2 ###########
# Due to a bug in Python3 (as of 2018-09-25), the fitness function
# CANNOT output to the screen and flush said output. It can lead to a deadlock
# situation.
######## IMPORTANT NOTE 2 ###########

# GeneralCGPSolver derives from BaseEstimator to allow ScikitLearn to use this
# class in their training Pipelines and other model-selectors.
from sklearn.base import BaseEstimator

GLOBAL_FITNESS_FUNC = None

class GeneralCGPSolver(BaseEstimator):
    """This is a generic Cartesian Genetic Programming algorithm
    developer. It is responsible for guiding the training of a population of
    individuals as well as recording statistics about them as training
    progresses."""

    # MultiCGPTester might not know the type of certain arguments and pass in
    # all numbers as float. We need a list of all arguments that must be
    # converted to integers:
    integerArguments = ['inputSize', 'outputSize', 'inputMemory',
                        'populationSize', 'numberParents', 'maxEpochs',
                        'numThreads', 'epochModOutput', 'wandbModelSave']

    def __init__(self,
                 type="BaseCGP",
                 variationSpecificParameters=None,
                 inputSize=1,
                 outputSize=1,
                 shape={'rowCount': 1,
                        'colCount': 10,
                        'maxColForward': -1,
                        'maxColBack': 10},
                 inputMemory=None,
                 fitnessFunction=None,
                 functionList=[],
                 populationSize=5,
                 numberParents=1,
                 parentSelectionStrategy='RoundRobin',
                 maxEpochs=100,
                 numThreads=1,
                 bestFitness=0,
                 epochModOutput=100,
                 pRange=[-1.0, 1.0],
                 constraintRange=None,
                 periodicSaving=None,
                 csvFileName=None,
                 wandbStatRecord=False,
                 wandbModelSave=False,
                 fitnessCollapseFunction=FitnessCollapseFunctions.mean,
                 completeFitnessCollapseFunction=None,
                 mutationStrategy={'name': 'probability',
                                   'genRate': [0.1, 0.1],
                                   'outRate': [0.1, 0.1],
                                   'application': 'pergene'},
                 trainingOptimizer=TrainingOptimizers.doNothingOptimizer()):

        """Create the class.

        Arguments:
            type - Defines the type of CGP that will be created / trained. As
                   this value changes, other parameters may be expected or may
                   be interpretted as a different type. As new varieties are
                   implemented, they must be added to __getNewIndividual()
                   in order to function properly.
            variationSpecificParameters - This is a dictionary filled with the
                                          parameters required by each specific
                                          CGP variation. Please check the
                                          variation's individiual class for
                                          the specific values required and
                                          their meanings.
            inputSize - The number of inputs into the algorithm.
            outputSize - The number of outputs from the algorithm.
            inputMemory - If not None, a dictionary describing the amount of
                          memory (old inputs) to store for each input value.
                          The dictionary values expected to be there:
                            - 'strategy' : How to deal with first few inputs
                                           where memory information won't be
                                           available.
                            - 'startValues' : If 'strategy' is 'constFill',
                                              this must be a list of length
                                              inputSize providing values which
                                              should be filled in as the
                                              initial memory for each input.
                                              None indicates that memory is
                                              not needed with this input.
                            - 'memLength' : A list integers of length inputSize
                                            which is used to describe how many
                                            inputs will be stored for each
                                            value. Total number of inputs into
                                            the algorithm will then be:
                                            inputSize + sum(memLength). 0
                                            in any space indicates that no
                                            memory is needed for this value.
            fitnessFunction - A function that must take an indidividual, an
                              input, and an expected output and return a list
                              of possible fitnesses for that individual.
                              Deterministic versions can return a list of
                              length 1. This function MUST obey the rule:
                              "Bigger is better"; Higher fitness values must
                              indicate greater fitness than lower values.
            functionList - A list of callable, simple functions. Different
                           individual types may require different numbers of
                           inputs to each of these functions.
            shape - A dictionary describing the number of rows / cols and
                    how far in each direction a gene can look for inputs. These
                    values are expected to be found in the dictionary:
                       - 'rowCount': Number of rows in the solution graph.
                       - 'colCount': Number of columns in the solution graph.
                       - 'maxColForward': Number of columns forward a node is
                                          allowed to search for inputs. The
                                          default is -1, meaning that recurrent
                                          connections aren't allowed.
                       - 'maxColBack': Number of columns backward a node is
                                       allowed to search for inputs. Generally,
                                       this is set to the number of columns to
                                       allow all previous columns to be used.
            populationSize - The total population size of each generation.
            numberParents - The number of parents to keep from each
                            generation to the next. If this is greater than
                            one, the parent(s) selected as the base for
                            each offspring of the next generation will be
                            determined by parentSelectionStrategy. If this
                            value is greater than or equal to populationSize,
                            the algorithm will never advance.
            parentSelectionStrategy - The manner by which the parent of each
                                      child is selected. The following options
                                      are valid:
                                      RoundRobin - Starting with the first
                                                   parent, each parent is
                                                   selected in order. This list
                                                   of parents repeats as often
                                                   as needed to generate the
                                                   new population.
                                      Random - Parents are selected randomly.
                                               No distribution is guaranteed;
                                               this can result in the most fit
                                               individual producing no
                                               offspring.
            maxEpochs - The number of epochs to train if no solution is found.
            numThreads - The number of threads to allocate to our thread pool
                         for processing samples. This is not an indication
                         of the total number of threads the training process
                         may allocate for itself because individual fitness
                         functions may choose to use multiple threads.
            bestFitness - The value that if returned from the fitness function,
                          after calculating for an entire epoch, which
                          indicates that the best possible program has been
                          found and further train in not needed.
            epochModOutput - The number % epoch number at which status should
                             be output to the screen.
            pRange - A list of length 2 indicating the minimum and maximum
                     values that the parameter (P) can take on in each node.
                     For most individual types, this is a bias. For others the
                     P-value can take on other meanings.
            constraintRange - If not none, expected to be a list of 2 values
                             representing the minimum and maximum values that
                             any given function will be allowed to return.
                             Values outside of that range will be constrained.
            periodicSaving - If not None, expected to be a dictionary that
                             defines 2 values:
                                 - fileName: A file name to which the epoch
                                             number will be appended to create
                                             full save-file names.
                                 - epochMod: An integer. When the epoch % this
                                             value is zero, the current model
                                             will be saved to file.
            csvFileName - If not None, expected to be a filename to which
                          stats will be saved in CSV format on a per-epoch
                          basis.
            wandbStatRecord - If True, the following details will get recorded
                              to WeightsAndBiases on each epoch:
                                 - seconds to calculate that epoch
                                 - Best individual's fitness
                                 - Percentage nodes used by the best individual
            wandbModelSave - If True, whenever there is a model saved off by
                             the periodic model saving capability, that model
                             will also be saved to WandB.
            fitnessCollapseFunction - The function used to take a list of
                                      output fitnesses from an individual and
                                      turn them into a single value that can be
                                      compared to the fitnesses of other
                                      individuals.
            completeFitnessCollapseFunction - This function works exactly the
                                              same way as
                                              fitnessCollapseFunction. However,
                                              this function is used to determine
                                              if an individual has completed
                                              training. It is only used if
                                              fitnessCollapseFunction reports
                                              max fitness has been hit. Thus,
                                              training is only considered
                                              complete if both of the fitness
                                              collapse functions, when applied
                                              to the list of possible fitnesses,
                                              return values greater than or
                                              equal to bestFitness.
            mutationStrategy - A dictionary of values that must at least have
                              'name' as the name of the mutation strategy. All
                              other required parameters associated with that
                              strategy are expected to be present in the
                              dictionary.  Current possibilities:
                              'probability'. Requires: 'genRate', 'outRate',
                                                       'application'
                              'activegene'. Requires: 'numGenes'
            trainingOptimizer - A TrainingOptimizer responsible for modifying
                                some training variables during training.
        """

        # Simplify setting all of the initialization variables:
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        # Read in, converting to integers as we go:
        for arg, val in values.items():
            if arg in self.integerArguments and val is not None:
                setattr(self, arg, int(val))
            else:
                setattr(self, arg, val)

        # If the complete fitness collapse function isn't provided, set it
        # equal to the standard collapse function:
        if self.completeFitnessCollapseFunction is None:
            self.completeFitnessCollapseFunction = self.fitnessCollapseFunction

        # Convert the shape values to integers, too:
        for key, val in shape.items():
            shape[key] = int(val)

        self.__alreadyTrained = False
        self.__epochsToSolution = None
        self.processPool = None
        self.__csvFileHandle = None

    def printCurrentParameters(self):
        print("Current parameters:")
        print("Network type: " + self.type)
        print("Variation specific parameters: " +
              str(self.variationSpecificParameters))
        print("\t Shape:" + str(self.shape))
        print("\t Input Size: " + str(self.inputSize))
        print("\t Output Size: " + str(self.outputSize))
        print("\t Input memory details: " + str(self.inputMemory))
        if self.functionList is not None:
            print("\t Len(functionList): " + str(len(self.functionList)))
        else:
            print("\t No function list defined.")
        print("\t Fitness Function: " + str(self.fitnessFunction))
        print("\t Population Size: " + str(self.populationSize))
        print("\t Number of parents: " + str(self.numberParents))
        print("\t Parent selection strategy: " + str(self.parentSelectionStrategy))
        print("\t Max Epochs: " + str(self.maxEpochs))
        print("\t Num Threads: " + str(self.numThreads))
        print("\t Parameter Range: " + str(self.pRange))
        print("\t Range: " + str(self.constraintRange))
        print("\t Epoch mod output: " + str(self.epochModOutput))
        print("\t Best Possible Fitness: " + str(self.bestFitness))
        print("\t Periodic saving: " + str(self.periodicSaving))
        print("\t CSV file name: " + str(self.csvFileName))
        print("\t Fitness collapse function: " + str(self.fitnessCollapseFunction))
        print("\t Comp. Fitness collapse function: " + str(self.completeFitnessCollapseFunction))
        print("\t Mutation Strategy: " + str(self.mutationStrategy))
        print("\t Training Optimizer: " + str(self.trainingOptimizer))

    def copyFrom(self, otherGenSolver):
        """Copy all variables from otherGenSolver into this class."""
        self.type = otherGenSolver.type
        self.variationSpecificParameters = \
          otherGenSolver.variationSpecificParameters
        self.shape = otherGenSolver.shape
        self.inputSize = otherGenSolver.inputSize
        self.outputSize = otherGenSolver.outputSize
        self.inputMemory = otherGenSolver.inputMemory
        self.functionList = otherGenSolver.functionList
        self.fitnessFunction = otherGenSolver.fitnessFunction
        self.populationSize = otherGenSolver.populationSize
        self.numberParents = otherGenSolver.numberParents
        self.parentSelectionStrategy = otherGenSolver.parentSelectionStrategy
        self.maxEpochs = otherGenSolver.maxEpochs
        self.numThreads = otherGenSolver.numThreads
        self.pRange = otherGenSolver.pRange
        self.constraintRange = otherGenSolver.constraintRange
        self.epochModOutput = otherGenSolver.epochModOutput
        self.bestFitness = otherGenSolver.bestFitness
        self.periodicSaving = otherGenSolver.periodicSaving
        self.csvFileName = otherGenSolver.csvFileName
        self.fitnessCollapseFunction = otherGenSolver.fitnessCollapseFunction
        self.completeFitnessCollapseFunction = otherGenSolver.completeFitnessCollapseFunction
        self.mutationStrategy = otherGenSolver.mutationStrategy
        self.trainingOptimizer = otherGenSolver.trainingOptimizer

        self.__population = copy.deepcopy(otherGenSolver.__population)

    def __setGlobalFitnessFunc(self, func):
        """Set our global fitness function."""
        global GLOBAL_FITNESS_FUNC
        GLOBAL_FITNESS_FUNC = func

    def __getGlobalFitnessFunc(self):
        """Get the global fitness function."""
        return GLOBAL_FITNESS_FUNC

    def __getNewIndividual(self):
        """Return a new individual based upon the type required by our class
        variables."""
        if self.type == None:
            raise ValueError("Type must be defined to generate individuals.")

        elif self.type == "BaseCGP":
            retInd = CGPIndividual.CGPIndividual(
                type=self.type,
                inputSize=self.inputSize,
                outputSize=self.outputSize,
                shape=self.shape,
                inputMemory=self.inputMemory,
                pRange=self.pRange,
                constraintRange=self.constraintRange,
                functionList=self.functionList,
                baseSpecificParameters=self.variationSpecificParameters)
            return retInd

        elif self.type == "ModularCGP":
            # We need to make sure the module lists are created and shared
            # between all individuals:
            if not hasattr(self, "activeModuleList"):
                self.activeModuleList = []
            if not hasattr(self, "inactiveModuleList"):
                self.inactiveModuleList = []

            retInd = MCGPIndividual.MCGPIndividual(
               type=self.type,
               inputSize=self.inputSize,
               outputSize=self.outputSize,
               shape=self.shape,
               pRange=self.pRange,
               constraintRange=self.constraintRange,
               functionList=self.functionList,
               MCGPSpecificParameters=self.variationSpecificParameters,
               activeModuleList=self.activeModuleList,
               inactiveModuleList=self.inactiveModuleList)
            return retInd

        elif self.type == "MultiGenomeCGP":
            retInd = MCCGPIndividual.MCCGPIndividual(
               type=self.type,
               inputSize=self.inputSize,
               outputSize=self.outputSize,
               shape=self.shape,
               pRange=self.pRange,
               constraintRange=self.constraintRange,
               functionList=self.functionList,
               MC_CGPSpecificParameters=self.variationSpecificParameters)
            return retInd

        elif self.type == 'FFCGPANN':
            retInd = FFCGPANNIndividual.FFCGPANNIndividual(
               type=self.type,
               inputSize=self.inputSize,
               outputSize=self.outputSize,
               shape=self.shape,
               pRange=self.pRange,
               constraintRange=self.constraintRange,
               functionList=self.functionList,
               FFCGPANNSpecificParameters=self.variationSpecificParameters)
            return retInd

        elif self.type == 'MCFFCGPANN':
            retInd = MCFFCGPANNIndividual.MCFFCGPANNIndividual(
               type=self.type,
               inputSize=self.inputSize,
               outputSize=self.outputSize,
               shape=self.shape,
               pRange=self.pRange,
               constraintRange=self.constraintRange,
               functionList=self.functionList,
               MCFFCGPANNSpecificParameters=self.variationSpecificParameters)
            return retInd

        elif self.type == 'RCGPANN':
            retInd = RCGPANNIndividual.RCGPANNIndividual(
               type=self.type,
               inputSize=self.inputSize,
               outputSize=self.outputSize,
               shape=self.shape,
               pRange=self.pRange,
               constraintRange=self.constraintRange,
               functionList=self.functionList,
               RCGPANNSpecificParameters=self.variationSpecificParameters)
            return retInd

        else:
            raise ValueError("Unrecognized individual-type: " + str(self.type))

    def generateStartingPopulation(self, continueTraining=False):
        """This function will generate the starting population for our
        algorithm.

        Arguments:
            continueTraining - If True, the first member of our population
                               is already filled with an individual that we
                               want to keep; it shouldn't be randomized.

        Returns:
            None."""
        randStart = 0
        # If we're continuing training, we need to keep the current best
        # individual:
        if continueTraining:
            randStart = 1
            self.__population = self.__population[:1]

            # That individual may have been loaded from file. We need to
            # fill him in with any details not provided by the file-load or
            # that may have changed:
            self.__population[0].functionList = self.functionList
            self.__population[0].pRange = self.pRange

            for i in range(randStart, self.populationSize):
                individual = self.__population[0].getOneMutatedChild(self.mutationStrategy)
                self.__population.append(individual)
        else:
            self.__population = []
            for i in range(randStart, self.populationSize):
                individual = self.__getNewIndividual()
                individual.randomize()
                self.__population.append(individual)

    def __calculateAllFitness_noData(self):
        """This function will calculate the fitness values for every member
        of the population based on the idea that there is no input or output
        data, but rather the fitness function will make multiple calls to the
        individual, asking it to process.  This is typically used with control
        type problems.

        Return:
            A list of the fitnesses of all of our population."""

        allFitness = []
        allCompleteFitness = []
        if self.numThreads == 1:
            allFitness = [self.__getGlobalFitnessFunc()(X) for X in
                          zip(self.__population,
                            itertools.repeat(None),
                            itertools.repeat(None))]

        else:
            allFitness = list(self.processPool.map(self.__getGlobalFitnessFunc(),
                                  zip(self.__population,
                                      itertools.repeat(None),
                                      itertools.repeat(None))))

        # Replace the list of fitnesses with the value calculated from them:
        for i in range(len(allFitness)):
            # Don't break functionality with older fitness functions that
            # returned single values:
            if isinstance(allFitness[i], list):
                allCompleteFitness.append(self.completeFitnessCollapseFunction(allFitness[i]))
                allFitness[i] = self.fitnessCollapseFunction(allFitness[i])

        return allFitness, allCompleteFitness

    def __calculateAllFitness_data(self, X, Y):
        """This will calculate the fitness of every member of the population
        given that we have inputs and ground truth expected outputs.

        X and Y are expected to be lists of the same length, with X
        representing the inputs and Y the expected outputs."""
        allFitness = []
        allCompleteFitness = []
        if self.numThreads == 1:
            allFitness = [self.__getGlobalFitnessFunc()(X) for X in
                          zip(self.__population,
                              itertools.repeat(X),
                              itertools.repeat(Y))]

        else:
            allFitness = list(self.processPool.map(self.__getGlobalFitnessFunc(),
                                  zip(self.__population,
                                      itertools.repeat(X),
                                      itertools.repeat(Y))))

        # Replace the list of fitnesses with the value calculated from them:
        for i in range(len(allFitness)):
            # Don't break functionality with older fitness functions that
            # returned single values:
            if isinstance(allFitness[i], list):
                allCompleteFitness.append(self.completeFitnessCollapseFunction(allFitness[i]))
                allFitness[i] = self.fitnessCollapseFunction(allFitness[i])

        return allFitness, allCompleteFitness

    def __cutPopulationDownToNewParents(self, allFitnesses, allCompleteFitnesses):
        """Cut the population down to the parents we want to keep."""
        return self.__cutPopulationDownToNewParents_singleFitness(allFitnesses,
          allCompleteFitnesses)

    def __cutPopulationDownToNewParents_singleFitness(self, allFitnesses, allCompleteFitnesses):
        """Many evolution strategies involve 'killing off' all but the best
        individuals. This function implements that.

        Arguments:
            allFitnesses - A list of the same length as our population
                           containing all of their calucalted fitnesses."""

        # High fitness is always better. Cut down our population until the
        # highest fitnesses are all that remain. This function will favor
        # children over parents of equal fitness (as per the standard for the
        # 1 + lambda evolution strategy)
        while len(allFitnesses) > self.numberParents:
            # Find the minimum index.  This will return the earliest minimum
            # in the case of ties, which will favor killing off parents rather
            # than offspring.
            min_index, _ = min(enumerate(allFitnesses),
                               key=operator.itemgetter(1))

            # Remove that individual from the population:
            del allFitnesses[min_index]
            del allCompleteFitnesses[min_index]
            del self.__population[min_index]

        # Other parts of the algorithm make the assumption that the 0th
        # indivdiual is always the best.  We need to move it there:
        max_index, _ = max(enumerate(allFitnesses), key=operator.itemgetter(1))
        allFitnesses.insert(0, allFitnesses.pop(max_index))
        allCompleteFitnesses.insert(0, allCompleteFitnesses.pop(max_index))
        self.__population.insert(0, self.__population.pop(max_index))
        return allFitnesses, allCompleteFitnesses

    def __produceNewPopulationFromParents(self):
        """This function assumes that the individuals currently available in
        population are all parents that can be used for creating new
        individuals. This allows it to be used both when we have many parents
        and when we've re-loaded from file with a single parent."""
        roundRobin = False
        parentNum = -1
        maxParent = len(self.__population) - 1

        # If the user wants round-robin parent selection, mark as such:
        if self.parentSelectionStrategy.lower() == 'roundrobin':
            roundRobin = True

        while len(self.__population) < self.populationSize:
            if roundRobin:
                parentNum += 1
                if parentNum > maxParent:
                    parentNum = 0
            else:  # Random
                parentNum = random.randint(0, maxParent)

            self.__population.append(
                self.__population[parentNum].getOneMutatedChild(
                    self.mutationStrategy))

    def periodicOutputs(self, epoch, fitnesses, calcSec):
        """Output to the screen and file as often as requested."""
        # Find the best (max fitness) individual:
        max_index, max_fitness = max(enumerate(fitnesses),
                                     key=operator.itemgetter(1))

        percentNodesUsed = self.__population[max_index].getPercentageNodesUsed()

        if self.__csvFileHandle is not None:
            # Write out all of the stats we're tracking:
            # epochNumber, calc seconds, min, max, mean, median fitnesses,
            # and percent nodes used by the best individual:
            self.__csvFileHandle.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" %
              (epoch,
               calcSec,
               float(FitnessCollapseFunctions.minimum(fitnesses)),
               float(max_fitness),
               float(FitnessCollapseFunctions.mean(fitnesses)),
               float(FitnessCollapseFunctions.median(fitnesses)),
               float(percentNodesUsed)))

            # Make sure the line gets flushed out to disk, in case we are
            # interrupted later:
            self.__csvFileHandle.flush()
            os.fsync(self.__csvFileHandle.fileno())

        # Save to WeightsAndBiases. The calling script is expected to
        # initialize our WandB connection for us:
        if self.wandbStatRecord:
            wandb.log({'Fitness': max_fitness,
                       'Epoch Calculation Seconds': calcSec,
                       'Percentage of Nodes Used': percentNodesUsed}, step=epoch)

        # Output to the screen:
        epochModElapsedTime = datetime.datetime.now() - self.lastEpochModTimestamp
        if self.epochModOutput is not None and epoch % self.epochModOutput == 0:
            print("Type: %s. Epoch %s. Percent nodes used: %.2f, Time since last output: %.2f minutes, Fitness: %.2f."
                  % (self.type, str(epoch),
                     float(self.__population[max_index].getPercentageNodesUsed()),
                     epochModElapsedTime.total_seconds() / 60.0,
                     float(fitnesses[max_index])))

            self.lastEpochModTimestamp = datetime.datetime.now()

            if self.type == 'ModularCGP':
                modules, primitives = self.__population[max_index].\
                  getNumberOfModulesAndPrimitiveFunctions()
                print("Number of active modules: %d, Used by best individual: \
Modules - %d, Primitives - %d. Primitive focus: %f." %
                  (len(self.__population[max_index].activeModuleList), modules,
                  primitives, self.__population[max_index].primitiveFunctionFocus))

        # Occasional save to file:
        if epoch != 0 and \
           self.periodicSaving is not None and \
           epoch % self.periodicSaving['epochMod'] == 0:
            fullFileName = self.periodicSaving['fileName'] + "_" + \
                           str(epoch)
            self.save(fullFileName, forceSave=True)

            # Record the file to WandB as well:
            if self.wandbModelSave:
                wandb.save(fullFileName)

            fullFileName = self.periodicSaving['fileName'] + '_' + \
                           'mostRecent'
            self.save(fullFileName, forceSave=True)

    def fit_NoData(self, startingEpoch=0):
        """Run the full training algorithm using a fitness function which does
        not take inputs or have ground-truth outputs."""
        epochFitnesses, completeEpochFitnesses = self.__calculateAllFitness_noData()

        # No success to start.
        success = False

        # Update based upon this batch:
        self.__cutPopulationDownToNewParents(epochFitnesses, completeEpochFitnesses)

        for epoch in range(startingEpoch, self.maxEpochs, 1):
            # Tell the leading individual to do the updates that are needed
            # once per population per epoch. Some individual types won't need
            # to do any processing.
            self.__population = self.__population[0].performOncePerEpochUpdates(
              self.__population, epochFitnesses)

            self.__produceNewPopulationFromParents()

            # This should get us a list of all fitnesses for the individual
            # across all samples in the epoch:
            start = datetime.datetime.now()
            epochFitnesses, completeEpochFitnesses = self.__calculateAllFitness_noData()
            elapsed = datetime.datetime.now() - start

            if self.epochModOutput is not None and (epoch % self.epochModOutput) == 0:
                print("\nMost recent single fitness calculation time: %f seconds (%f minutes)" \
                      %(elapsed.total_seconds(), elapsed.total_seconds() / 60.0))

            # Occasional output to screen and file(s):
            self.periodicOutputs(epoch, epochFitnesses, elapsed.total_seconds())

            # Update based upon this batch and put the best fitness at the
            # front of the list:
            self.__cutPopulationDownToNewParents(epochFitnesses, completeEpochFitnesses)

            # Let the optimizer update things if need be:
            madeChange = self.trainingOptimizer.oncePerEpochUpdate(self, epoch,
              epochFitnesses[0])

            # We can't end on an epoch where the training optimizer made a
            # change in case it was important to the training:
            if epochFitnesses[0] >= self.bestFitness and \
              completeEpochFitnesses[0] >= self.bestFitness and \
              not madeChange:
                success = True

            # If we succeeded, set a few variables and break out of the loop.
            if success:
                print("Epoch " + str(epoch) + ": Found solution.")
                self.__epochsToSolution = epoch

                if self.wandbStatRecord:
                    wandb.log({'epochsToSolution': self.__epochsToSolution,
                                             'bestFitness': epochFitnesses[0]})

                if self.periodicSaving is not None:
                    # Save off our solution:
                    fullFileName = self.periodicSaving['fileName'] + '_' + 'solution'
                    self.save(fullFileName, forceSave=True)

                    if self.wandbModelSave:
                        wandb.save(fullFileName)

                break


        print("Done. Best fitness: %s. Secondary collapse fitness: %s" %
          (str(epochFitnesses[0]), str(completeEpochFitnesses[0])))

        # Record a too-high epochsToSolution since this may be what we
        # try to minimize:
        if self.wandbStatRecord:
            solutionEpochs = epoch
            if not success:
                solutionEpochs += 100
            wandb.log({'epochsToSolution': solutionEpochs,
                       'bestFitness': epochFitnesses[0]})

        self.fitted_ = True
        self.__alreadyTrained = True
        self.__finalFitness = epochFitnesses[0]
        if self.__csvFileHandle is not None:
            self.__csvFileHandle.close()

        # GridSearchCV requires this return value:
        return self

    def fit(self, X, Y, startingEpoch=0):
        """Fit will train the algorithm with the given dataset.

        Arguments:
            X - A numpy array of inputs of the same length as Y.
            Y - A numpy array of expected outputs with the same length as X.
            startingEpoch - The starting epoch number when training. Since the
                            model can output itself to file periodically and
                            uses the epoch number as part of the file name,
                            this allows for training to "continue" rather than
                            starting at 0 every time.

        If X and Y are None, then it is assumed that we have a fitness function
        that doesn't require inputs.
        """
        self.printCurrentParameters()

        if self.csvFileName is not None:
            try:
                self.__csvFileHandle = open(self.csvFileName, 'w')
                self.__csvFileHandle.write('epoch,seconds,min,max,mean,median,percentNodesUsed\n')
            except:
                print("Error opening %s. Not recording information to csv." \
                  % (self.csvFileName))
                self.__csvFileHandle = None

        if self.numThreads != 1:
            self.processPool = Pool(self.numThreads)
        else:
            self.processPool = None

        self.__setGlobalFitnessFunc(self.fitnessFunction)

        # Confirm that our most basic needs are met:
        if self.functionList is None or len(self.functionList) == 0:
            raise ValueError("A function list of at least one function must \
be provided.")

        if self.fitnessFunction is None:
            raise ValueError("A fitness function must be provided.")

        # Set our processing start time:
        self.lastEpochModTimestamp = datetime.datetime.now()

        # If we've already been fitted, then make sure we keep the current
        # best individiual:
        self.generateStartingPopulation(continueTraining=self.__alreadyTrained)

        success = False
        epochFitnesses = None
        self.__epochsToSolution = -1

        # If no data was provided, we may be doing some reinforcement learning;
        # call the modified fit function:
        if X is None:
            return self.fit_NoData(startingEpoch=startingEpoch)

        self.allFitTimes = []
        self.allMutTimes = []

        # Start with this so that it can be the bottom of the loop. This makes
        # sure we only have parents when we exit this loop:
        epochFitnesses, completeEpochFitnesses = self.__calculateAllFitness_data(X, Y)

        self.__cutPopulationDownToNewParents(epochFitnesses, completeEpochFitnesses)

        for epoch in range(startingEpoch, self.maxEpochs, 1):
            # Tell the leading individual to do the updates that are needed
            # once per population per epoch. Some individual types won't need
            # to do any processing.
            self.__population = self.__population[0].\
              performOncePerEpochUpdates(self.__population, epochFitnesses)

            self.__produceNewPopulationFromParents()

            # This should get us a list of all fitnesses for the individual
            # across all samples in the epoch:
            start = datetime.datetime.now()
            epochFitnesses, completeEpochFitnesses = self.__calculateAllFitness_data(X, Y)
            elapsed = datetime.datetime.now() - start

            # Occasional output to screen and file:
            self.periodicOutputs(epoch, epochFitnesses, elapsed.total_seconds())

            # Update based upon this batch:
            epochFitnesses = self.__cutPopulationDownToNewParents(epochFitnesses, completeEpochFitnesses)

            # Check for success.
            if epochFitnesses[0] >= self.bestFitness and \
              completeEpochFitnesses[0] >= self.bestFitness:
                success = True

            # If we succeeded, set a few variables and break out of the loop.
            if success:
                print("Epoch " + str(epoch) + ": Found solution.")
                self.__epochsToSolution = epoch

                if self.periodicSaving is not None:
                    # Save off our solution:
                    fullFileName = self.periodicSaving['fileName'] + '_' + 'solution'
                    self.save(fullFileName, forceSave=True)

                break

        print("Done. Best fitness: %s. Secondary collapse fitness: %s" %
          (str(epochFitnesses[0]), str(completeEpochFitnesses[0])))
        self.fitted_ = True
        self.__alreadyTrained = True
        self.__finalFitness = epochFitnesses[0]

        # Clean-up:
        # No need to clean up the concurrent.futures Pool, just drop our link
        # to it and let it get cleaned up by garbage collection.
        self.processPool = None
        self.__setGlobalFitnessFunc(None)
        if self.__csvFileHandle is not None:
            self.__csvFileHandle.close()

        # GridSearchCV requires this return value:
        return self

    def score(self, X, Y, fitnessFunction):
        """This function is meant to score the model to allow GridSearchCV to
        decide which model is best. It expects and array/list of inputs and
        an array/list of expected outputs of the same size. It will return
        the current model's fitness where bigger is better.

        Arguments:
            X - Array/List of inputs.
            Y - Array/List of expected outputs of the same length as X UNLESS
                X is None, then Y will be interpretted as the number of times
                to record our best individual's fitness before averaging them
                together.

        Return:
            The model's fitness (bigger is better)"""
        # Run all of the inputs through our parent, which should be our best:
        fitness = fitnessFunction((self.__population[0], X, Y))
        return fitness

    def predict(self, X, y=None):
        """This function will return the model's output for a single input.

        Arguments:
            X - A single input.

        Return:
            The model's output for that input."""
        return self.__population[0].calculateOutputs(X)

    def save(self, fileName, forceSave=False):
        """Save off the class as is.

        Arguments:
            fileName - The name of the file to which to save this class.
            forceSave - If True, saving will be allowed even if training
                        hasn't been completed.

        Returns:
            None
        """
        # We can't pickle threads, set them aside:
        with open(fileName, "wb") as file:
            # Temporarily store some non-pickle-able variables:
            tempThreads = self.processPool
            self.processPool = None

            tempFileHandle = self.__csvFileHandle
            self.__csvFileHandle = None

            tempFitnessFunction = self.fitnessFunction
            self.fitnessFunction = None

            pickle.dump(self, file)

            # Put them back:
            self.processPool = tempThreads
            self.__csvFileHandle = tempFileHandle
            self.fitnessFunction = tempFitnessFunction

    def load(self, fileName):
        """Load ourselves from file."""

        # Pickle has issues loading self from file, so we'll create a
        # new instance and copy over our variables:
        with open(fileName, "rb") as file:
            temp = pickle.load(file)
            self.copyFrom(temp)

        # Set variables that hold functions to None; it is unsafe to assume
        # those function pointers are still valid after loading from memory.
        # The user will need to reset them in order to continue training.
        self.functionList = None
        self.fitnessFunction = None
        self.processPool = None
        self.fitted_ = True

        # Need to mark ourselves as trained:
        self.__alreadyTrained = True

    def saveLoadTest(self, fileName):
        """For debugging the save / load functionality."""
        print("**** Pre-save: ****")
        self.printCurrentParameters()
        startGenotype = copy.deepcopy(self.__population[0].getGenotype())
        self.save(fileName)
        self.__population = None

        self.load(fileName)
        print("**** Post-Load: ****")
        self.printCurrentParameters()
        endGenotype = copy.deepcopy(self.__population[0].getGenotype())

        if startGenotype == endGenotype:
            print("Genotype success")
        else:
            print("Genotype failure")

    def setFitnessFunction(self, newFitnessFunction):
        """Set the global fitness function."""
        self.__setGlobalFitnessFunc(newFitnessFunction)

    def getEpochsToSolution(self):
        """Return how many epochs it took to find a solution."""
        return self.__epochsToSolution

    def getFinalFitness(self):
        """Return what our final fitness was."""
        return self.__finalFitness

    def getBestIndividual(self):
        """Return a reference to our best individual."""
        return self.__population[0]
