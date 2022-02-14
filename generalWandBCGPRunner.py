"""This script is responsible for taking in the parameters passed to it on the
command line from WeightsAndBiases and convert them into the parameters that
our PAINTTask and GeneralCGPSolver require."""
# General:
import sys
import os
import pandas as pd
import random
import functionLists
import warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings("ignore")

# Set the WandB timeout before we do anything WandB related:
# os.environ['WANDB_HTTP_TIMEOUT'] = 60  # Safer to set in bash
import wandb

# Task
import gym
import gym_jsbsim
import gym_jsbsim.properties as prp

# CGP:
import GymFitnessFunctions
import FitnessCollapseFunctions
import TrainingOptimizers
import functionLists
from GeneralCGPSolver import GeneralCGPSolver

# Start sweep with:
# wandb sweep --project projectName --name sweepName yamlFileName

# If running simple sweeps, okay to start them in the background with:
# wandb agent tm6501/projectName/IDProvidedBywandbSweep &

# Writing our own argument parser because argparse requires we name every
# possible parameter. All arguments are expected to be in the form:
# --argName=argValue
def parseSingleArgument(inString):
    try:
        name, inVal = inString.split('=', 1)
    except:
        print(f"Couldn't parse: {inString}")
    name = name[2:]
    gotGoodVal = False
    outVal = None
    # If we can convert it, do so:
    try:
        outVal = int(inVal)
        gotGoodVal = True
    except:
        pass
    if not gotGoodVal:
        try:
            outVal = float(inVal)
            gotGoodVal = True
        except:
            pass
    if not gotGoodVal and inVal == "True":
        outVal = True
        gotGoodVal = True
    if not gotGoodVal and inVal == "False":
        outVal = False
        gotGoodVal = True
    if not gotGoodVal:
        outVal = str(inVal)
    return name, outVal

def getParametersByPrefix(startDict, prefix, match=True, removeFromOriginalDict=False):
    retDict = {}
    originalsFound = []
    for key, val in startDict.items():
        # We either want all of those that start with the prefix:
        if match and key.startswith(prefix):
            if isinstance(val, str):
                val = stringConverter(val)
            retDict[key[len(prefix):]] = val
            originalsFound.append(key)
        # Or all of those that don't start with the prefix:
        elif not match and not key.startswith(prefix):
            if isinstance(val, str):
                val = stringConverter(val)
            retDict[key] = val
            originalsFound.append(key)

    # If we want to remove them from the original dictionary, do so here:
    if removeFromOriginalDict:
        for keyName in originalsFound:
            del startDict[keyName]

    return retDict

def stringConverter(inputString):
    # Conversion to a list:
    retVal = None
    if inputString.startswith("["):
        code = f"retVal = {inputString}"
        _locals = locals()
        exec(code, globals(), _locals)
        retVal = _locals['retVal']
    elif inputString.lower() == 'none':
        pass  # retVal is already None
    elif inputString.lower() == 'false':
        retVal = False
    elif inputString.lower() == 'true':
        retVal = True
    # A module and function name:
    elif inputString.find('.') != -1:
        module, val = inputString.split('.', 1)
        code = f"""retVal = getattr({module}, "{val}", None)"""
        _locals = locals()
        exec(code, globals(), _locals)
        retVal = _locals['retVal']
        if retVal is None:
            raise ValueError(f"Module {module} has no value {val}.")
    else:  # Just leave it as a string
        retVal = inputString
    return retVal

def main():
    # Get all the parameters passed in:
    argsDict = {}
    for i in range(1, len(sys.argv), 1):
        name, val = parseSingleArgument(sys.argv[i])
        argsDict[name] = val

    CGPArgs = getParametersByPrefix(argsDict, "CGP_", removeFromOriginalDict=True)
    ENVArgs = getParametersByPrefix(argsDict, "ENV_", removeFromOriginalDict=True)

    # Break out specific dictionaries from our CGPArgs:
    neededBreakouts = ['shape', 'mutationStrategy', 'variationSpecificParameters']
    for name in neededBreakouts:
        CGPArgs[name] = getParametersByPrefix(CGPArgs, name + "_", removeFromOriginalDict=True)

    wandb.init()
    runName = wandb.run.name
    # runName = "testRun"

    # Convert any strings we received:
    for k,v in argsDict.items():
        if isinstance(v, str):
            argsDict[k] = stringConverter(v)

    # Create our fitness function:
    print("\n\nArgs to fitnessClass: ")
    for k,v in argsDict.items():
        print(f"\t{k}: {v} of type {type(v)}")

    fitnessClass = GymFitnessFunctions.generalTupleFitnessFunction(
      argsDict['timesToRepeat'],
      argsDict['envName'],
      argsDict['useArgmax'],
      argsDict['maxStepsPerRun'],
      renderSpeed=argsDict['renderSpeed'],
      numThreads=argsDict['numThreads'],
      envParams=ENVArgs,
      npConvert=argsDict['npConvert'])

    CGPArgs['fitnessFunction'] = fitnessClass.fitnessFunc

    # Add outputs to WandB:
    resultsDirectory = f"./CGPResults/{runName}"
    Path(resultsDirectory).mkdir(parents=True, exist_ok=True)
    CGPArgs['periodicSaving'] = {'fileName': resultsDirectory + "/model",
                                 'epochMod': argsDict['epochModelSave']}

    # Add a doNothing optimizer:
    doNothing = TrainingOptimizers.doNothingOptimizer()
    CGPArgs['trainingOptimizer'] = doNothing

    # DEBUG:
    # CGPArgs['wandbModelSave'] = False
    # CGPArgs['wandbStatRecord'] = False

    print(f"\n\nCGPArgs: ")
    for k,v in CGPArgs.items():
        print(f"\t{k}: {v} of type {type(v)}")

    print(f"\n\nENVArgs: ")
    for k,v in ENVArgs.items():
        print(f"\t{k}: {v} of type {type(v)}")


    model = GeneralCGPSolver(**CGPArgs)
    model.fit(None, None)

if __name__ == "__main__":
    main()
