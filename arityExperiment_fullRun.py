import sys
sys.path.insert(0, '/home/jlwilli5/pythonScripts/CGP')
mainExperimentFolder = '/home/jlwilli5/CGPExperiments'

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

envs = ["CartPole-v1",  # Already done.
        "LunarLander-v2",
        "MountainCar-v0_modifiedReward",
        "LunarLanderContinuous-v2",
        "MountainCarContinuous-v0_modifiedReward"
        ]

argList = []
for env in envs:
    tempAct, tempObs = GymFitnessFunctions.getActionObservationSizes(env)
    argList.append(
      {
      'type': ['FFCGPANN'],
      'inputSize': [tempObs],
      'outputSize': [tempAct],
      'shape__rowCount': [1],
      'shape__colCount': [300],
      'shape__maxColForward': [-1],
      'shape__maxColBack': [301],
      'inputMemory': [None],
      'fitnessFunction': [GymFitnessFunctions.getEnvironmentTestFunction_tuple(env)],
      'functionList': [functionLists.funcListANN_singleTan],
      'populationSize': [7],
      'numberParents': [1],
      'maxEpochs': [20000],
      'epochModOutput': [25],
      'bestFitness': [GymFitnessFunctions.getMaxScore(env)],
      'pRange': [[-1.0, 1.0]],
      'constraintRange': [[-1.0, 1.0]],
      'trainingOptimizer': [doNothing],
      'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
      'completeFitnessCollapseFunction': [None],
      'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 1]}],
      # We're experimenting on different arity values and ranges. Put in all
      # varieties:
      'vsp__inputsPerNeuron': [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6],
                               [7, 7], [8, 8], [9, 9], [2, 3], [2, 4],
                               [2, 5], [2, 6], [2, 7], [2, 8], [2, 9],
                               [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
                               [3, 9], [4, 5], [4, 6], [4, 7], [4, 8],
                               [4, 9], [5, 6], [5, 7], [5, 8], [5, 9],
                               [6, 7], [6, 8], [6, 9], [7, 8], [7, 9],
                               [8, 9]],
      'vsp__weightRange': [[-1.0, 1.0]],
      'vsp__switchValues': [[1, 1]]
      }
    )

# Run each environment as its own MultiCGPTester so that if we hit an error
# or processing is stopped for any reason we have already saved off all
# results from previous environments:
for expTup in zip(envs, argList):
    experimentFolder = "%s/arityCollapseTest/%s" % (mainExperimentFolder, expTup[0])

    tester = MultiCGPTester.MultiCGPTester(
      expTup[1],
      runsPerVariation = 5,
      periodicModelSaving = 25,
      experimentFolder = experimentFolder)

    _ = tester.runTests(None, None, confirmPrompt=False)

# Started at 9:20AM, 2019-12-30
