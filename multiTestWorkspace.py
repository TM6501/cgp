import sys
sys.path.insert(0, '/home/jlwilli5/pythonScripts/CGP') # Work
# sys.path.insert(0, '/home/joseph/pythonScripts/CGP') # Home
mainExperimentFolder = '/home/jlwilli5/CGPExperiments'  # Work
#mainExperimentFolder = '/home/joseph/CGPExperiments' # Home

import MultiCGPTester
import GymFitnessFunctions
import FitnessCollapseFunctions
import TrainingOptimizers
import GeneralCGPSolver
import functionLists
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

doNothing = TrainingOptimizers.doNothingOptimizer()

# envs = ["CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0",
#         "LunarLander-v2", "LunarLanderContinuous-v2"]

envs = ["CartPole-v1"]

fullTestList = []
for env in envs:
    tempAct, tempObs = GymFitnessFunctions.getActionObservationSizes(env)
    fullTestList.append(
      {
      'type': ['FFCGPANN'],
      'inputSize': [tempObs],
      'outputSize': [tempAct],
      'shape__rowCount': [1],
      'shape__colCount': [300],
      'shape__maxColForward': [-1],
      'shape__maxColBack': [1001],
      'inputMemory': [None],
      'fitnessFunction': [GymFitnessFunctions.getEnvironmentTestFunction_tuple(env)],
      'functionList': [functionLists.funcListANN_singleTan],
      'populationSize': [7],
      'numberParents': [1],
      'maxEpochs': [50],
      'epochModOutput': [5],
      'bestFitness': [GymFitnessFunctions.getMaxScore(env)],
      'pRange': [[-1.0, 1.0]],
      'constraintRange': [[-1.0, 1.0]],
      'trainingOptimizer': [doNothing],
      'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
      'completeFitnessCollapseFunction': [None],
      'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 1]},
                           {'name': 'activeGene', 'numGenes': [1, 3]}],
      'vsp__inputsPerNeuron': [[2, 9]],
      'vsp__weightRange': [[-1.0, 1.0]],
      'vsp__switchValues': [[1]],
      }
    )

# Dynamically decide on a folder name to make life easier. Need to convert
# the now() result to local time:
tempNow = pd.datetime.now() - pd.Timedelta(hours=5)

# experimentFolder = '%s/%s_%d-%d-%d_%d-%d-%d' % (mainExperimentFolder,
#   environmentName, tempNow.year, tempNow.month, tempNow.day, tempNow.hour,
#   tempNow.minute, tempNow.second)
experimentFolder = "%s/multiEnvTest_2019_12_29" % (mainExperimentFolder)

tester = MultiCGPTester.MultiCGPTester(
  fullTestList,
  runsPerVariation = 2,
  periodicModelSaving = None,
  experimentFolder = None)

df = tester.runTests(None, None)

# Testing:
tester = GeneralCGPSolver.GeneralCGPSolver()
tester.load('filename')
ind = tester.getBestIndividual()
allFitness = GymFitnessFunctions.getGeneralGymFitness(
  ind, 5, 'BipedalWalker-v2', False, maxStepsPerRun=2000, renderSpeed=0, numThreads=1)

allFits = GymFitnessFunctions.getGeneralGymFitness(ind, 5, environmentName,
  True,  # argmax
  maxStepsPerRun=1000, renderSpeed=None, numThreads=10)
