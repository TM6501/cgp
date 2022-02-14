import MultiCGPTester
import FitnessCollapseFunctions
import TrainingOptimizers
import GeneralCGPSolver
import functionLists
import GymFitnessFunctions

import gym
import gym_jsbsim

doNothing = TrainingOptimizers.doNothingOptimizer()

actionSize = 3
observationSize = 9

def levelFlightFitness(inputTuple):
    return GymFitnessFunctions.getGeneralGymFitness(
      inputTuple[0],  # Individual
      3, # Times to repeat
      'GymJsbsim-GetToAltitudeAndHeadingSmallTask-v0',  # Name of task
       False,  # Use argmax
       maxStepsPerRun=8000, renderSpeed=None, numThreads=1)

cgpArgs = {
  'type': ['FFCGPANN'],
  'inputSize': [observationSize],
  'outputSize': [actionSize],
  'shape__rowCount': [1],
  'shape__colCount': [500],
  'shape__maxColForward': [-1],
  'shape__maxColBack': [5001],
  'inputMemory': [None],
  'fitnessFunction': [levelFlightFitness],
  'functionList': [functionLists.funcListANN_singleTan],
  'populationSize': [7],
  'numberParents': [1],
  'maxEpochs': [15000],
  'epochModOutput': [10],
  'bestFitness': [6250],  # 8000 time steps, 1.0 each is best theoretical score.
  'pRange': [[-1.0, 1.0]],
  'constraintRange': [[-1.0, 1.0]],
  'trainingOptimizer': [doNothing],
  'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
  'completeFitnessCollapseFunction': [None],
  'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 1]}],
  'numThreads': [1],
  # We're experimenting on different arity values and ranges. Put in all
  # varieties:
  'vsp__inputsPerNeuron': [[2, 9]],
  'vsp__weightRange': [[-1.0, 1.0]],
  'vsp__switchValues': [[1, 1]]
}

tester = MultiCGPTester.MultiCGPTester(cgpArgs, runsPerVariation=1,
                                       periodicModelSaving=None)

_ = tester.runTests(None, None, confirmPrompt=False)
