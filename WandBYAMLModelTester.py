"""This script is designed to take in a YAML file designed for WandB, read it
in, and test it against a loaded model."""

import yaml

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
    else:  # Try to convert it to a float to handle scientific notation:
        try:
            retVal = float(inputString)
        except:
            retVal = inputString
    return retVal

def main():
    if len(sys.argv) < 3:
        print("Usage: python WandBYAMLModelTester.py modelFileName YamlFileName outputCSVFileName")
        exit()

    # yamlFileName = 'GeneralCGPSweep.yaml'
    # modelFileName = 'D:\GoogleDrive\CGP\scripts\CGPResults\change3_modelSolution.cgpModel'
    # outputCSVFileName = 'D:\GoogleDrive\CGP\scripts\CGPResults\modelOutput.csv'

    yamlFileName = sys.argv[2]
    modelFileName = sys.argv[1]
    outputCSVFileName = None
    if len(sys.argv) > 3:
        outputCSVFileName = sys.argv[3]

    if outputCSVFileName is None:
        print(f"Testing model at {modelFileName} with yaml: {yamlFileName}.")
    else:
        print(f"Testing model at {modelFileName} with yaml: {yamlFileName}. Writing results to {outputCSVFileName}")

    # Get the yaml data first since some of it will be required to recreate the model:
    yamlFile = open(yamlFileName)
    yamlData = yaml.load(yamlFile, Loader=yaml.FullLoader)
    yamlParameters = yamlData['parameters']

    # Get the backwards compatibility YAML data and add it to the loaded data:
    backwardsData = yaml.load(open('backwardsCompatibility.yaml'), Loader=yaml.FullLoader)
    backParams = backwardsData['parameters']
    for k, v in backParams.items():
        if k not in yamlParameters:
            yamlParameters[k] = v

    # Convert all "values" to parameters, doing string conversion as we go:
    for k, v in yamlParameters.items():
        if isinstance(v, dict):
            if 'value' in v:
                value = v['value']
            elif 'values' in v:  # Hopefully this wasn't important...
                value = v['values'][0]
            else:
                raise ValueError(f"Don't know how to parse dictionary: {k}: {v}")
            if isinstance(value, str):
                value = stringConverter(value)
            yamlParameters[k] = value
        else:
            print(f"Non-dictionay found. {k}: {v}")

    ENVArgs = getParametersByPrefix(yamlParameters, "ENV_", removeFromOriginalDict=True)

    neededBreakouts = ['shape', 'mutationStrategy', 'variationSpecificParameters']

    # Load the model:
    model = GeneralCGPSolver()
    model.load(modelFileName)
    individual = model.getBestIndividual()

    # Create our fitness function (not saved as part of the model):
    fitnessClass = GymFitnessFunctions.generalTupleFitnessFunction(
      1,  #yamlParameters['timesToRepeat'],
      "PAINTTask-FG-v0",# yamlParameters['envName'],
      yamlParameters['useArgmax'],
      yamlParameters['maxStepsPerRun'],
      0, # renderSpeed=yamlParameters['renderSpeed'],
      renderMode="flightgear",
      numThreads=1, # numThreads=yamlParameters['numThreads'],
      envParams=ENVArgs,
      npConvert=yamlParameters['npConvert'],
      csvFileName=outputCSVFileName)
    fitnessFunc = fitnessClass.fitnessFunc

    # Run the test:
    print("Beginning run...")
    score = fitnessFunc((individual, ))
    print(f"Final score: {score}")

if __name__ == "__main__":
    main()
