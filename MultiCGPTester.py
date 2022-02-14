import pandas as pd
import datetime
import os

import GeneralCGPSolver

# Individual types:
import CGPIndividual
import MCGPIndividual
import MCCGPIndividual
import FFCGPANNIndividual
import MCFFCGPANNIndividual
import RCGPANNIndividual

class MultiCGPTester():
    """This class mimics somewhat ScikitLearn's GridSearchCV. It will take
    multiple parameters and try many solutions. It allows more information
    to be collected and is better specialized to CGP testing than GridSearchCV.
    Over the course of one experiment (series of runs) multiple
    GeneralCGPSolvers will be created and destroyed to run the tests."""

    def __init__(
      self,
      inputParams,
      runsPerVariation = 5,
      csvFileFolder = None,
      experimentFolder = None,
      periodicModelSaving = None
    ):
        """Create the class.

        Arguments:
            inputParams - A dictionary in which every key represents the
                          name of the keyword argument to be passed to
                          GeneralCGPSolver. Every value is expected to be a
                          list of different values to try; this is true even
                          if only a single value is to be used.  Parameters
                          that should be passed in as part of the
                          variationSpecificParameters should have names
                          prefixed with 'vsp__'. "shape" is not expected to
                          be passed in, but rather the four values which
                          make up shape should be provided, each with a
                          'shape__' prefix.  Those values: 'rowCount',
                          'colCount', 'maxColForward', 'maxColBack'. If
                          numThreads is not provided, it will be set to
                          populationSize.
                          This argument can also be a list of dictionaries
                          that all meet these requirements if options need to
                          be used for which all permutations don't make sense.
            runsPerVariation - This is the number of times to try each
                               permutation.
            experimentFolder - If not None, a folder to which all outputs from
                               this run should be stored and organized.
            periodicModelSavingEpoch - If provided, how often models should
                                       output saved versions of themselves.
                                       This only happens if there is an
                                       experiment folder."""

        self.__inputParams = inputParams
        self.__runsPerVariation = runsPerVariation
        self.__experimentFolder = experimentFolder
        self.__periodicModelSavingEpoch = periodicModelSaving

        # We always want lists of input parameter dictionaries:
        if not isinstance(self.__inputParams, list):
            self.__inputParams = [self.__inputParams]

    def runTests(self, X, Y, confirmPrompt=True):
        """Run the actual tests, given the variables that were provided to
        us in the initialization function.

        Arguments:
            X - A numpy array of inputs.
            Y - The corresponding outputs.

        Note:
             - X and Y can be set to None to perform training that doesn't
               require data."""

        # Get all input variations as a dataframe:
        df = self.getAllVariations()

        # Add the columns we'll want to fill:
        temp = [None] * len(df)
        df['finalFitness'] = temp
        df['epochsToSolution'] = temp
        df['minutesOfProcessing'] = temp
        df['savedSolver'] = temp

        nonArgumentColumns = ['index', 'finalFitness', 'epochsToSolution',
                              'runNumber', 'minutesOfProcessing',
                              'savedSolver']

        # Let the user know what we're about to do and give them the chance
        # to back out:
        if confirmPrompt:
            carryOn = input("Total runs: %d.  Start the tests (y/n)?" % (len(df)))
            if carryOn.lower() != 'y':
                print("Stopping.")
                return
        else:
            print("Beginning training of %d total runs." % len(df))

        tempRet = []

        # Make the directory where we'll store our CSV files:
        infoFileHandle = None
        modelsFileHandle = None
        if self.__experimentFolder is not None:
            try:
                os.makedirs(self.__experimentFolder, exist_ok=True)

                # Human-readable information:
                filename = "%s/csvInfo.txt" % self.__experimentFolder
                infoFileHandle = open(filename, 'w')
                infoFileHandle.write("%d total to train.\n" % (len(df)))

                # Information in CSV format:
                filename2 = "%s/modelInfo.csv" % self.__experimentFolder
                modelsFileHandle = open(filename2, 'w')
                modelsFileHandle.write("modelNumber,bestFitness,epochsToSolution,trainingMinutes\n")
            except:
                print("Error opening %s. Not recording csv information."
                  % (filename))
                infoFileHandle = None
                modelsFileHandle = None

        for i in range(len(df)):
            # Reset our dictionaries:
            keyArgs = {'variationSpecificParameters': {}, 'shape': {}}
            # Add each column to the full dictionary:
            for col in df.columns:
                # Ignore several of the columns:
                if col in nonArgumentColumns:
                    continue

                # Special case the variationSpecificParameters:
                elif col[:5].lower() == 'vsp__':
                    keyArgs['variationSpecificParameters'][col[5:]] = df.iloc[i][col]

                # Special case the shape parameters:
                elif col[:7].lower() == 'shape__':
                    keyArgs['shape'][col[7:]] = df.iloc[i][col]

                # Add to our dictionary of inputs:
                else:
                    keyArgs[col] = df.iloc[i][col]

            # If the user didn't provide a number of threads, we set it
            # equal to the number of parents:
            if not 'numThreads' in keyArgs:
                keyArgs['numThreads'] = keyArgs['populationSize']

            # Add in the csv file-writing information:
            if self.__experimentFolder is not None:
                # Tell the GeneralCGPSolver that we want it to output to csv:
                csvFileName = "%s/%d.csv" % (self.__experimentFolder, i)
                keyArgs['csvFileName'] = csvFileName

                # And tell it to save periodic model outputs:
                if self.__periodicModelSavingEpoch:
                    pSaveFolder = "%s/model_%d" % (self.__experimentFolder, i)
                    os.makedirs(pSaveFolder, exist_ok=True)
                    pSaveFileName = "%s/epoch" % (pSaveFolder)
                    keyArgs['periodicSaving'] = {'fileName': pSaveFileName,
                                                 'epochMod': self.__periodicModelSavingEpoch}
                else:
                    keyArgs['periodicSaving'] = None

                infoFileHandle.write("#####  %d  #####" % (i))
                keys = sorted(keyArgs)
                for key in keys:
                    infoFileHandle.write("  %s: %s" % (str(key), str(keyArgs[key])))
                infoFileHandle.write("\n")
                infoFileHandle.flush()
                os.fsync(infoFileHandle.fileno())

            else:
                keyArgs['csvFileName'] = None
                keyArgs['periodicSaving'] = None

            print("###################################")
            print("Beginning training of %d / %d" % (i+1, len(df)))
            print("###################################")

            # Collect the processing time:
            start = datetime.datetime.now()
            solver = GeneralCGPSolver.GeneralCGPSolver(**keyArgs)
            solver.fit(X, Y)
            elapsed = datetime.datetime.now() - start

            # Get our results:
            fitness = solver.getFinalFitness()
            epochs = solver.getEpochsToSolution()

            print("Elapsed test time: %f seconds (%f minutes). Final fitness: \
%s. Epochs to solution (-1 -> no solution found): %d" %
  (elapsed.total_seconds(), elapsed.total_seconds() / 60.0, str(fitness),
   epochs))

            # Store our results:
            df.at[i, 'finalFitness'] = solver.getFinalFitness()
            df.at[i, 'epochsToSolution'] = solver.getEpochsToSolution()
            df.at[i, 'minutesOfProcessing'] = elapsed.total_seconds() / 60.0
            df.at[i, 'savedSolver'] = solver

            # Write those results out to our models csv file:
            # modelNumber, bestFitness, epochsToSolution, trainingMinutes
            if modelsFileHandle is not None:
                modelsFileHandle.write("%d,%.2f,%d,%.2f\n" % (i,
                  float(solver.getFinalFitness()), solver.getEpochsToSolution(),
                  float(elapsed.total_seconds() / 60.0)))
                modelsFileHandle.flush()
                os.fsync(modelsFileHandle.fileno())

            # Write results out to the human-readable file:
            if infoFileHandle is not None:
                infoFileHandle.write("finalFitness: %s\n" \
                  % (str(solver.getFinalFitness())))
                infoFileHandle.write("epochsToSolution: %s\n" \
                  % (str(solver.getEpochsToSolution())))
                infoFileHandle.write("minutesOfProcessing: %s\n\n" \
                  % (str(elapsed.total_seconds() / 60.0)))
                infoFileHandle.flush()
                os.fsync(infoFileHandle.fileno())

        # Cleanup our files:
        if modelsFileHandle is not None:
            modelsFileHandle.close()

        if infoFileHandle is not None:
            infoFileHandle.close()

        return df

    def getAllVariations(self):
        """Get a dataframe of all input variations.

        Assumptions:
            - self.__inputParams and self.__runsPerVariation are properly
              filled.
        """

        fullDf = pd.DataFrame(columns=['runNumber'])

        for inputParams in self.__inputParams:
            # Run number is always our first key:
            temp = [i for i in range(1, self.__runsPerVariation + 1)]
            df = pd.DataFrame(temp, columns=['runNumber'])

            for key, value in inputParams.items():
                # We need to duplicate our whole dataframe for every new value's
                # possibility to create all permutations:
                startLen = len(df)
                for i in range(1, len(value)):
                    df = df.append(df[:startLen])

                # Now we add as many copies of each of our inputs as we need:
                fullValueSet = []
                for i in value:
                    fullValueSet += [i] * startLen

                # Add the whole list as a new column:
                df[key] = fullValueSet

            # Add to our full list of inputs:
            fullDf = fullDf.append(df, ignore_index=True).reset_index(drop=True)

        return fullDf
