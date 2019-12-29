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
minSwitchOptimizer = TrainingOptimizers.newCollapseFunctionOptimizer(
  FitnessCollapseFunctions.minimum, fitnessChange=190.0)
hardcoreFitnessChanger = TrainingOptimizers.newFitnessFunctionOptimizer(
  GymFitnessFunctions.bipedalWalkerHardcoreFitness, fitnessChange=175.0)

# Get environment details from GymFitnessFunctions to help prevent errors:
environmentName = "BipedalWalker-v2_modifiedReward"
act, obs = GymFitnessFunctions.getActionObservationSizes(environmentName)
testFunc = GymFitnessFunctions.getEnvironmentTestFunction_tuple(environmentName)
# atariTestFunc = GymFitnessFunctions.genericAtariFitness_ram(
#   environmentName=environmentName,
#   timesToRepeat=10,
#   numThreads=10,
#   renderSpeed=None
# )
# testFunc = atariTestFunc.fitnessFunction
maxScore = GymFitnessFunctions.getMaxScore(environmentName)

if None in [act, obs, testFunc, maxScore]:
    print("Act: %s, Obs: %s, testFunc: %s, maxScore: %s" % (str(act), str(obs),
      str(testFunc), str(maxScore)))
    raise ValueError("Error collecting necessary data for '%s'" % (environmentName))

# Dynamically decide on a folder name to make life easier. Need to convert
# the now() result to local time:
tempNow = pd.datetime.now() - pd.Timedelta(hours=5)

experimentFolder = '%s/%s_%d-%d-%d_%d-%d-%d' % (mainExperimentFolder,
  environmentName, tempNow.year, tempNow.month, tempNow.day, tempNow.hour,
  tempNow.minute, tempNow.second)
experimentFolder = "%s/multiEnvTest" % (mainExperimentFolder)

tester = MultiCGPTester.MultiCGPTester(
[
{
'type': ['FFCGPANN'],
'inputSize': [obs],
'outputSize': [act]],
'shape__rowCount': [1],
'shape__colCount': [300],
'shape__maxColForward': [-1],
'shape__maxColBack': [1001],
'inputMemory': [None],
'fitnessFunction': [testFuncs[0]],
'functionList': [functionLists.funcListANN_singleTan],
'populationSize': [7],
'numberParents': [1],
'maxEpochs': [15000],
'epochModOutput': [100],
'bestFitness': [maxScore],
'pRange': [[-1.0, 1.0]],
'constraintRange': [[-2.0, 2.0]],
'trainingOptimizer': [doNothing],
'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
'completeFitnessCollapseFunction': [None],
'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 1]},
                     {'name': 'activeGene', 'numGenes': [1, 3]}],
'vsp__inputsPerNeuron': [[2, 9]],
'vsp__weightRange': [[-1.0, 1.0]],
'vsp__switchValues': [[1]],
},


# ,
# {
# 'type': ['RCGPANN'],
# 'inputSize': [obs],
# 'outputSize': [act],
# 'shape__rowCount': [1],
# 'shape__colCount': [100, 500],
# 'shape__maxColForward': [-1],
# 'shape__maxColBack': [501],
# 'inputMemory': [None],
# 'fitnessFunction': [testFunc],
# 'functionList': [functionLists.funcListANN_single],
# 'populationSize': [5],
# 'numberParents': [1],
# 'maxEpochs': [10000],
# 'epochModOutput': [25],
# 'bestFitness': [maxScore],
# 'pRange': [[-1.0, 1.0]],
# 'constraintRange': [None],
# 'trainingOptimizer': [doNothing],
# 'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
# 'completeFitnessCollapseFunction': [None],
# 'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 4]}],
# 'vsp__inputsPerNeuron': [[2, 9]],
# 'vsp__weightRange': [[-1.0, 1.0]],
# 'vsp__switchValues': [[1]],
# 'vsp__memorySteps': [3],
# 'vsp__type': [1, 2, 3]
# }
],
runsPerVariation = 2,
periodicModelSaving = 100,
experimentFolder = experimentFolder)

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
