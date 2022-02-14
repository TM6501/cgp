import CGPFunctions as f
import GymFitnessFunctions
import FitnessCollapseFunctions
import TrainingOptimizers
from GeneralCGPSolver import GeneralCGPSolver
import functionLists
import warnings
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import gym

def main():
    doNothing = TrainingOptimizers.doNothingOptimizer()

    # envName = 'LunarLanderContinuous-v2'
    envName = 'CartPole-v1'

    fitnessFunction = GymFitnessFunctions.getEnvironmentTestFunction_tuple(envName)
    bestFitness = GymFitnessFunctions.getMaxScore(envName)
    act, obs = GymFitnessFunctions.getActionObservationSizes(envName)
    functionList = functionLists.gymFuncList

    functionList = [f.ADD,  # X + Y
                    f.AMINUS,  # X - Y
                    f.CMINUS,
                    f.MULT,  # X * Y
                    f.CMULT,  # X * P
                    f.ABS,  # abs(X)
                    f.SQRT,  # abs(X) ** 0.5
                    f.CPOW,  # X ** (P + 1)
                    f.YPOW,  # abs(X) ** abs(Y)
                    f.EXPX,  # e ** X - 1 / e - 1
                    f.GTE,  # X >= Y
                    f.LTE,  # X <= Y
                    f.LTEP,  # X <= P
                    f.GTEP,  # X >= P
                    f.MAX2,  # max(X, Y)
                    f.MIN2,  # min(X, Y)
                    f.ROUND,  # -1, 0, or 1
                    # f.float_And,  # X and Y both > 0.0
                    # f.float_Or,  # X or Y > 0.0
                    # f.float_Nand,  # Not float_and(X, Y)
                    # f.float_Nor,  # Not float_Or(X, Y)
                    # f.float_Xor,  # X or Y > 0.0, but not both
                    # f.float_AndNotY,  # X > 0.0 and Y <= 0.0
                    f.CONST # P
                    # f.YWIRE,  # Y
                    # f.NOP     # X
                   ]

    cgpKwargs = {
      'type': 'BaseCGP',
      'inputSize': obs,
      'outputSize': act,
      'shape': {'rowCount': 1, 'colCount': 1000,
                'maxColForward': -1, 'maxColBack': 1001},
      'inputMemory': None,
      'fitnessFunction': fitnessFunction,
      'functionList': functionList,
      'populationSize': 10,
      'numberParents': 1,
      'parentSelectionStrategy': 'RoundRobin',
      'maxEpochs': 1000,
      'epochModOutput': 1,
      'bestFitness': bestFitness,
      'pRange': [-1.0, 1.0],
      'constraintRange': [-2.0, 2.0],
      'trainingOptimizer': doNothing,
      'fitnessCollapseFunction': FitnessCollapseFunctions.minOfMeanMedian,
      'completeFitnessCollapseFunction': FitnessCollapseFunctions.minimum,
      'mutationStrategy': {'name': 'activeGene', 'numGenes': [1, 1]},
      'numThreads': 5,
      'variationSpecificParameters': {'useSeparateScaleValues': True,
                                      'scaleRange': [0.0, 1.0]}
    }

    solver = GeneralCGPSolver(**cgpKwargs)
    solver.fit(None, None)

    ind = solver.getBestIndividual()
    actFuncList = ind.getActiveFunctionList()

    print("Active functions:")
    for funcNum, usageCount in actFuncList.items():
        print(f"\t{functionList[funcNum]}: {usageCount}")

    print("Genotype:")
    ind.printGenotype()


if __name__ == "__main__":
    main()
