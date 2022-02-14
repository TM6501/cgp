import gym
# import pybulletgym
import numpy as np
import pandas as pd
import time
import random
import math
import itertools

# Prevent the annoying warning with every environment created:
gym.logger.set_level(40)

# Concurrent.futures.ProcessPoolExecutor is used instead of
# multiprocessing.Pool because it allows multiprocessing within multiprocessing.
# This allows fitness functions to use multiple threads even through the
# overall training process also uses multiple threads.
from concurrent.futures import ProcessPoolExecutor as Pool

########## IMPORTANT NOTE 1 #########
# We've hit multiple problems trying to run Atari environments with screen
# inputs.  It is on the (very long) list of bugs / features to be dealt with.
# For now, use them at your own risk. Atari with RAM-input seems to be
# functioning properly.
########## IMPORTANT NOTE 1 #########

# Build a list of all environments:
all_envs = gym.envs.registry.all()
G_ENV_IDS = [env_spec.id for env_spec in all_envs]

# We keep track of Mujoco environments in order to remove them from our
# list of available environments. They require special licensing which most
# people don't have.
mujocoEnvironments = ['HandManipulateBlockTouchSensorsDense-v1',
  'FetchSlide-v1', 'Ant-v3', 'HandManipulateBlockRotateXYZTouchSensors-v0',
  'Swimmer-v3', 'HalfCheetah-v3',
  'HandManipulatePenRotateTouchSensorsDense-v0', 'FetchReach-v1',
  'HandManipulateBlockRotateZTouchSensors-v1',
  'HandManipulateBlockRotateParallelTouchSensorsDense-v1',
  'FetchReachDense-v1', 'Ant-v2',
  'HandManipulateBlockRotateZTouchSensorsDense-v1', 'Hopper-v3',
  'HandManipulateBlockRotateZTouchSensors-v0', 'HandManipulateBlockRotateZ-v0',
  'HandManipulatePenTouchSensors-v1', 'HandManipulatePenTouchSensorsDense-v0',
  'FetchPushDense-v1', 'Pusher-v2',
  'HandManipulateBlockRotateXYZTouchSensorsDense-v1', 'HalfCheetah-v2',
  'HandManipulateBlockRotateParallelTouchSensors-v1',
  'HandManipulateBlockTouchSensors-v1', 'HandManipulatePen-v0',
  'HandManipulateEggDense-v0', 'HandManipulateBlockTouchSensorsDense-v0',
  'HandManipulateBlockFull-v0', 'HandManipulateEggTouchSensorsDense-v0',
  'HandManipulateEggRotateTouchSensors-v1',
  'HandManipulateBlockRotateZDense-v0', 'Reacher-v2', 'HandManipulateEgg-v0',
  'Thrower-v2', 'HandReach-v0', 'HandManipulatePenRotateTouchSensors-v1',
  'Humanoid-v3', 'HandManipulatePenFull-v0',
  'HandManipulateBlockRotateParallel-v0',
  'HandManipulateEggRotateTouchSensorsDense-v0',
  'HandManipulateBlockRotateParallelDense-v0',
  'HandManipulateEggTouchSensors-v0', 'HandManipulatePenRotateDense-v0',
  'FetchPickAndPlaceDense-v1',
  'HandManipulateBlockRotateZTouchSensorsDense-v0', 'Walker2d-v2',
  'HandManipulatePenDense-v0', 'HandManipulateEggRotateTouchSensors-v0',
  'HandManipulateEggTouchSensorsDense-v1', 'Humanoid-v2',
  'HandManipulatePenFullDense-v0', 'FetchPickAndPlace-v1',
  'HandManipulatePenTouchSensorsDense-v1', 'Walker2d-v3',
  'HandManipulatePenTouchSensors-v0', 'HandManipulateBlockDense-v0',
  'FetchSlideDense-v1', 'HandManipulateBlockRotateXYZDense-v0',
  'FetchPush-v1', 'HandManipulateEggRotateTouchSensorsDense-v1',
  'HandManipulateBlockRotateParallelTouchSensorsDense-v0',
  'HandManipulateBlockRotateXYZTouchSensorsDense-v0', 'HandReachDense-v0',
  'HandManipulateEggFull-v0', 'InvertedDoublePendulum-v2',
  'HandManipulateBlock-v0', 'HandManipulateEggRotate-v0',
  'HandManipulateBlockTouchSensors-v0', 'HumanoidStandup-v2', 'Swimmer-v2',
  'HandManipulateBlockRotateXYZ-v0', 'HandManipulateEggTouchSensors-v1',
  'Striker-v2', 'HandManipulatePenRotateTouchSensorsDense-v1',
  'HandManipulateBlockFullDense-v0', 'HandManipulateEggRotateDense-v0',
  'HandManipulateBlockRotateParallelTouchSensors-v0',
  'HandManipulatePenRotate-v0', 'Hopper-v2',
  'HandManipulatePenRotateTouchSensors-v0', 'HandManipulateEggFullDense-v0',
  'HandManipulateBlockRotateXYZTouchSensors-v1', 'InvertedPendulum-v2']

# Environments that hang during creation as of 2019-10-16. We will also remove
# these from our list of available environments until they are fixed:
makeErrorEnvironments = ['Defender-RamNoFrameskip-v4', 'Defender-ram-v4',
                         'Defender-v0', 'Defender-ramNoFrameskip-v4',
                         'DefenderNoFrameskip-v4', 'Defender-ram-v0',
                         'DefenderNoFrameskip-v0',
                         'Defender-ramDeterministic-v0',
                         'DefenderDeterministic-v4', 'Defender-v4',
                         'Defender-ramDeterministic-v4',
                         'DefenderDeterministic-v0',
                         'Defender-ramNoFrameskip-v0']

# Remove mujoco and error environments from our global list:
for val in mujocoEnvironments + makeErrorEnvironments:
    if val in G_ENV_IDS:
        G_ENV_IDS.remove(val)

# Sort them for the benefit of anybody that actually looks through the list:
G_ENV_IDS = sorted(G_ENV_IDS)

def debugOutputAllEnvironments():
    """Output all environments to the screen for debugging purposes."""
    for id in G_ENV_IDS:
        act, obs = getActionObservationSizes(id)
        print("%s: Action Size: %s, Observation Size: %s" % (id, str(act), str(obs)))

def getIfEnvironmentIsAvailable(env_id):
    """Return if an environment is currently available."""
    if env_id in G_ENV_IDS:
        return True
    else:
        return False

def convertAtariObservation(observation):
    """Convert the Atari observation into the format we want."""
    # This function needs to convert the observation space to have the RGB axis
    # first and turn the [0-255] into [-1.0, 1.0]:
    observation = np.moveaxis(observation, -1, 0)
    observation = ((observation / 255.0) * 2.0) - 1.0

    return observation

def convertAtariObservation_GreyScaleMax(observation):
    """Convert an Atari observation to a greyscale version using a simple
    averaging of all the RGB components."""
    # This function will convert the [0-255] three axis RGB values to a single
    # matrix of grey-scale values. The simple averaging of values conversion
    # will be used:
    observation = np.moveaxis(observation, -1, 0)
    observation = ((observation / 255.0) * 2.0) - 1.0
    greyObservation = (observation[0] + observation[1] + observation[2]) / 3.0

    return greyObservation

def convertAtariObservation_colorToMax(observations, obsToUse):
    """Take 1 or more observations and return a new observation for which each
    pixel equals the maximum value found in any observations equivalent space.
    For instance:
    [[0,1,2,3], [3,2,1,0]] -> [3,2,2,3]

    Arguments:
        observations - A list of observations. Each observation is expected to
                       be 3 numpy arrays representing the Red, Green, and Blue.
        obsToUse - An array of integers representing the frame numbers to pull
                   in from the observations array; -1 is the current frame.
                   For instance: [-1, -3, -5, -7, -9] would bring in every
                   other frame.  If None, just return the current frame.
    channels of the observation.

    Atari observations are known to be of shape: (210, 160)
    """

    # If no processing to do, return the current frame:
    if obsToUse is None:
        return observations[-1]

    # Grab the first frome:
    firstFrameNum = obsToUse[0]
    retObservation = [np.copy(observations[firstFrameNum][0]),
                      np.copy(observations[firstFrameNum][1]),
                      np.copy(observations[firstFrameNum][2])]

    # Get the max of each observation:
    for i in range(1, len(obsToUse), 1):
        frameNum = obsToUse[i]
        # Red, green, blue:
        retObservation[0] = np.maximum(retObservation[0], observations[frameNum][0])
        retObservation[1] = np.maximum(retObservation[1], observations[frameNum][1])
        retObservation[2] = np.maximum(retObservation[2], observations[frameNum][2])

    # atariFrameRecording.outputRGBAsColorImage(retObservation)
    return retObservation

def convertAtariObservation_greyToMax(observations, obsToUse):
    """Take 1 or more observations and return a new observation for which each
    pixel equals the maximum value found in any observations equivalent space.
    For instance:
    [[0,1,2,3], [3,2,1,0]] -> [3,2,2,3]

    Arguments:
        observations - A list of observations. Each observation is expected to
                       be a single numpy greyscale image of the frame.
        obsToUse - An array of integers representing the frame numbers to pull
                   in from the observations array; -1 is the current frame.
                   For instance: [-1, -3, -5, -7, -9] would bring in every
                   other frame.  If None, just return the current frame.
    channels of the observation.

    Atari observations are known to be of shape: (210, 160)
    """
    # If no processing to do, return the current frame:
    if obsToUse is None:
        return observations[-1]

    print("Length 1: %s" % (str(len(observations))))

    # Grab the first frome:
    firstFrameNum = obsToUse[0]
    retObservation = np.copy(observations[firstFrameNum])
    # retObservation = np.copy(observations[firstFrameNum][0])

    # Get the max of each observation:
    for i in range(len(obsToUse), 1, 1):
        frameNum = obsToUse[i]
        np.maximum(retObservation, observations[frameNum], out=retObservation)

    print(str(retObservation))
    print("Shape: %s" % (str(retObservation.shape)))

    # atariFrameRecording.outputGreyImage(retObservation)
    return retObservation

def convertAtariAction(action):
    """Convert one possible way Atari actions could be output to the type of
    input the environment is expecting."""
    # Matrix outputs must be converted to the average of the matrix:
    retAction = []
    for val in action:
        if isinstance(val, np.ndarray):
            retAction.append(val.mean())
        else:
            retAction.append(val)

    return retAction

def getInputObservationFromListOfFrames(allObservations, greyScaleConversionMethod, stepCombinations, stepCombinationMethod):
    """Get the Atari frame observation into a format we want."""
    # Choose the correct method of conversion of multiple frames to a single
    # input:
    if len(allObservations) == 1: # No conversion, just return:
        return allObservations[0]

    if greyScaleConversionMethod is None:  # Using full color, not greyscale:
        if stepCombinationMethod.lower() == 'max':
            return convertAtariObservation_colorToMax(allObservations,
                                                      stepCombinations)
        else:
            raise ValueError("Unrecognized combination method: %s" %
              (str(stepCombinationMethod)))
    else:
        if stepCombinationMethod.lower() == 'max':
            return convertAtariObservation_greyToMax(allObservations,
                                                     stepCombinations)
        else:
            raise ValueError("Unrecognized combination method: %s" %
              (str(stepCombinationMethod)))


# Once we have models performing well, we need to make sure that they aren't
# just learning to repeat a few steps to get a good score, but rather are
# actually taking input from the Atari environment and making decisions.
# To that end, we will test high-performing models with random input. If their
# scores remain the same, we know they are effectively ignoring input.
def getRandomAtariObservation_ram():
    """Return an input of the equivalent size, shape, and ranges as the Atari
    RAM inputs, but filled with random values."""
    # Atari ram-observations are unsigned byte arrays of length 128.
    randObs = np.random.randint(0, high=255, size=(128,), dtype='uint8')
    return randObs

def getRandomAtariObservation_screen():
    """Return an input of the equivalent size, shape, and ranges as the Atari
    screen inputs, but filled with random values."""
    # Atari screen-observations are unsigned byte arrays of shape:
    # (210, 160, 3) to represent the whole screen in red, green, and blue.
    randObs = np.random.randint(0, high=255, size=(210, 160, 3), dtype='uint8')
    return randObs

def getGeneralGymSingleRunFitness(tup):
    """Get a single fitness from a standard openAI gym environment.
    The input tuple is expected to be defined as such:
    0: The individual
    1: Render speed. None for no rendering.
    2: Render mode. None if no rendering, or no mode needed for rendering.
    2: Maximum steps in the simulation.
    3: Take max. True if argmax should be applied to the action, False otherwise.
    4: Environment name.
    5: Environment parameters.
    6: Convert to np.array()
    7: A filename to which to write out the run's data
    """
    individual, renderSpeed, renderMode, maxSteps, takeMax, envName, envParams, npConvert, csvFileName = \
      tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6], tup[7], tup[8]

    # Use them if they passed in parameters:
    env = None
    if envParams is not None:
        env = gym.make(envName, **envParams)
    else:
        env = gym.make(envName)

    # Ready our dataframe if needed:
    df = None
    if csvFileName is not None:
        df = pd.DataFrame()

    observation = env.reset()
    runReward = 0
    individual.resetForNewTimeSeries()

    for j in range(maxSteps):
        if takeMax:
            action = np.argmax(individual.calculateOutputs(observation))
        else:
            action = individual.calculateOutputs(observation)
            # Some individuals return a single value instead of a list if
            # the list would only have 1 value; gym needs a list.  If this
            # action isn't a list, convert it to one:
            if isinstance(action, int) or isinstance(action, float):
                action = [action]

            if npConvert:
                action = np.array(action)

        if renderSpeed is not None:
            if renderMode is None:
                env.render()
            else:
                env.render(mode=renderMode)
            if renderSpeed != 0:
                time.sleep(renderSpeed)

        observation, reward, done, info = env.step(action)
        runReward += reward

        if csvFileName is not None:
            df = df.append(info, ignore_index=True)

        if done:
            break

    env.close()
    if csvFileName is not None:
        df.to_csv(csvFileName)

    return runReward

def getGeneralGymFitness(individual, timesToRepeat, envName, useArgmax, maxStepsPerRun=1000, renderSpeed=None, renderMode=None, numThreads=10, envParams=None, npConvert=False, csvFileName=None):
    """Get multiple fitnesses for a single individual in a multithreaded
    manner. Because fitnesses can vary dramatically from run to run, often many
    runs are combined into a single fitness for an individual.
    """
    # Build a list of times to repeat to make zipping easier:
    repeats = [i for i in range(timesToRepeat)]

    allScores = []
    if numThreads > 1:
        # Get our process pool:
        pool = Pool(numThreads)

        # See getGeneralGymSingleRunFitness for a list of the arguments expected in
        # each tuple:
        allScores = pool.map(getGeneralGymSingleRunFitness,
                             zip(itertools.repeat(individual),
                                 itertools.repeat(renderSpeed),
                                 itertools.repeat(renderMode),
                                 itertools.repeat(maxStepsPerRun),
                                 itertools.repeat(useArgmax),
                                 itertools.repeat(envName),
                                 itertools.repeat(envParams),
                                 itertools.repeat(npConvert),
                                 itertools.repeat(csvFileName),
                                 repeats))
    else:
        # Run without threads:
        allScores = map(getGeneralGymSingleRunFitness,
                        zip(itertools.repeat(individual),
                            itertools.repeat(renderSpeed),
                            itertools.repeat(renderMode),
                            itertools.repeat(maxStepsPerRun),
                            itertools.repeat(useArgmax),
                            itertools.repeat(envName),
                            itertools.repeat(envParams),
                            itertools.repeat(npConvert),
                            itertools.repeat(csvFileName),
                            repeats))

    # Requesting the list will cause this line to block until all threads have
    # finished and returned a result:
    return list(allScores)

def getGeneralAtariRamFitness(individual, timesToRepeat=10, envName=None,
                              useArgmax=True, maxStepsPerRun=20000,
                              scaleObservations=True, renderSpeed=None,
                              numThreads=10):
    """Get multiple fitnesses for a single Atari individual with RAM as input
    in a multithreaded manner."""
    if envName is None:
        return [-10000]

    # Build a list of times to repeat to make zipping easier:
    repeats = [i for i in range(timesToRepeat)]

    # Get our process pool:
    pool = Pool(numThreads)

    # See getAtariRamSingleRunFitness for a list of the arguments expected in
    # each tuple:
    allScores = pool.map(getAtariRamSingleRunFitness,
                         zip(itertools.repeat(individual),
                             itertools.repeat(renderSpeed),
                             itertools.repeat(maxStepsPerRun),
                             itertools.repeat(useArgmax),
                             itertools.repeat(envName),
                             itertools.repeat(scaleObservations),
                             repeats))

    # Requesting the list will cause this line to block until all threads have
    # finished and returned a result:
    return list(allScores)

def getMountainCarFitness_modifiedReward(individual, timesToRepeat, renderSpeed=None, continuous=False, numThreads=10):
    """Handle calling the modified-reward mountain car fitness function in
    a multithreaded manner. A modified-reward version is provided because the
    standard environment doesn't reward partial progress towards the goal.

    Parameters:
      individual - The model to be tested.
      timesToRepeat - The number of fitnesses to return.
      renderSpeed - If None, don't render. Otherwise the number of seconds
                    to pause between each frame's rendering.
      continuous - True if we should use MountainCarContinuous, false to use
                   MountainCar.
      numThreads - Number of threads/processes to use."""

    # Build a list of times to repeat to make zipping easier:
    repeats = [i for i in range(timesToRepeat)]

    # Allocate our process pool:
    pool = Pool(numThreads)

    # Send in the arguments expected by the single-run fitness function:
    allScores = pool.map(getOneRunMountainCarFitness_modifiedReward,
                         zip(itertools.repeat(individual),
                             itertools.repeat(continuous),
                             itertools.repeat(renderSpeed),
                             repeats))

    # Requesting the list will cause this line to block until all threads have
    # returned their results:
    return list(allScores)

def getOneRunMountainCarFitness_modifiedReward(tup):
    """Get one fitness from the MountainCar or MountainCarContinuous
    environment while modifying its reward function.

    The MountainCar environments reward only success, not progress towards
    success.  This means that individuals that are trying to drive up the
    hill, but not succeeding will get the exact same fitness as individuals
    that do nothing at all. This function provides some reward to the
    individual based on the maximum distance it made it up the hill.

    Parameters: A tuple expected to contain the following:
      0: individual - The model,
      1: continuous - True if using MountainCarContinuous, false to use
                      MountainCar.
      2: renderSpeed - None to not render, otherwise the number of seconds to
                        sleep between each frame; this can be a floating point
                        value."""

    individual, continuous, renderSpeed = tup[0], tup[1], tup[2]

    env = None
    if continuous:
        env = gym.make('MountainCarContinuous-v0')
    else:
        env = gym.make('MountainCar-v0')

    maxFrames = 2000
    runReward = 0
    maxPosition = -1.2  # 1.2 is the minimum for this environment.
    observation = env.reset()
    individual.resetForNewTimeSeries()

    for j in range(maxFrames):
        # The continuous version doesn't required argmax, but it does need
        # a conversion from a single value to the list that the environment
        # expects:
        if continuous:
            action = [individual.calculateOutputs(observation)]
        else:
            action = np.argmax(individual.calculateOutputs(observation))

        if renderSpeed is not None:
            env.render()
            if renderSpeed != 0:
                time.sleep(renderSpeed)

        observation, reward, done, info = env.step(action)
        runReward += reward

        # Record the furthest we made it up the hill:
        maxPosition = max(observation[0], maxPosition)

        if done:
            break

    env.close()

    # Return the fitness, modified by the maxPosition attained. The position
    # weighs heavier with the continuous version:
    if continuous:
        return runReward + (1000.0 * maxPosition)
    else:
        return runReward + (10.0 * maxPosition)

def getBipedalWalkerFitness_modifiedReward(individual, timesToRepeat, renderSpeed=None, hardcore=False, numThreads=10):
    """Handle calling the modified-reward bipedal walker fitness function in
    a multithreaded manner. A modified-reward version is provided because the
    standard environment heavily punishes falling over, which discourages
    experimentation in the early stages of training.

    Parameters:
      individual - The model to be tested.
      timesToRepeat - The number of fitnesses to return.
      renderSpeed - If None, don't render. Otherwise the number of seconds
                    to pause between each frame's rendering.
      hardcore - True if running in hardcore mode, False otherwise.
      numThreads - Number of threads/processes to use."""

    # Build a list of times to repeat to make zipping easier:
    repeats = [i for i in range(timesToRepeat)]

    # Allocate our process pool:
    pool = Pool(numThreads)

    # Send in the arguments expected by the single-run fitness function:
    allScores = pool.map(getOneRunBipedalWalkerFitness_modifiedReward,
                         zip(itertools.repeat(individual),
                             itertools.repeat(hardcore),
                             itertools.repeat(renderSpeed),
                             repeats))

    # Requesting the list will cause this line to block until all threads have
    # returned their results:
    return list(allScores)

def getOneRunBipedalWalkerFitness_modifiedReward(tup):
    """Get one fitness from the bipedal walker, modifying the reward function.

    The bipedal walker fitness function heavily penalizes falling over, which
    discourages early generations of many AI models from experimenting and
    encourages them to hold perfectly still to avoid the massive penalty for
    losing its balance. This function removes that penalty to encourage the
    learning model to experiment with ways of moving forward, even if the early
    results involve falling over.

    Parameters: A tuple expected to contain the following:
      0: individual - The model,
      1: hardcore - True if running in hardcore mode, false otherwise.
      2: renderSpeed - None to not render, otherwise the number of seconds to
                        sleep between each frame; this can be a floating point
                        value."""

    individual, hardcore, renderSpeed = tup[0], tup[1], tup[2]

    env = None
    if hardcore:
        env = gym.make('BipedalWalkerHardcore-v2')
    else:
        env = gym.make('BipedalWalker-v2')

    maxFrames = 2000
    runReward = 0
    observation = env.reset()
    individual.resetForNewTimeSeries()

    for j in range(maxFrames):
        action = individual.calculateOutputs(observation)
        if renderSpeed is not None:
            env.render()
            if renderSpeed != 0:
                time.sleep(renderSpeed)

        observation, reward, done, info = env.step(action)

        # Modify the massive penalty for falling over to encourage
        # the walker to take chances:
        if reward < -50:
            reward = 0

        runReward += reward
        if done:
            break

    env.close()

    # Return the fitness:
    return runReward

def getAtariRamSingleRunFitness(tup):
    """Get a single fitness from a standard openAI gym environment.
    The input tuple is expected to be defined as such:
    0: The individual
    1: Render speed. None for no rendering.
    2: Maximum steps in the simulation.
    3: Take max. True if argmax should be applied to the action, False otherwise.
    4: Environment name.
    5: If the integer observations should be scaled between -1 and 1.
    """
    individual, renderSpeed, maxSteps, takeMax, envName, scaleObservations = \
      tup[0], tup[1], tup[2], tup[3], tup[4], tup[5]

    env = gym.make(envName)
    observation = env.reset()
    runReward = 0
    individual.resetForNewTimeSeries()

    for j in range(maxSteps):
        # Convert to floats between -1 and 1:
        if scaleObservations:
            observation = (observation / 128.0) - 1.0

        if takeMax:
            action = np.argmax(individual.calculateOutputs(observation))
        else:
            action = individual.calculateOutputs(observation)
            # Some individuals return a single value instead of a list if
            # the list would only have 1 value; gym needs a list.  If this
            # action isn't a list, convert it to one:
            if isinstance(action, int) or isinstance(action, float):
                action = [action]

        if renderSpeed is not None:
            env.render()
            if renderSpeed != 0:
                time.sleep(renderSpeed)

        observation, reward, done, info = env.step(action)
        runReward += reward
        if done:
            break

    env.close()
    return runReward

def getOneAtariFitness_screen(individual, environmentName=None, maxSteps=18000,
                              timesToRepeat=10, renderSpeed=None,
                              greyScaleConversionMethod=None,
                              numObservationsToKeep=1, stepCombinations=[-1],
                              stepCombinationMethod='max',
                              randomizedObservations=False):
    """Get the fitness of an individual as it plays an Atari game with screen
    inputs.

    THIS CURRENTLY DOESN'T FUNCTION PROPERLY."""
    allScores = []
    if environmentName is None:
        raise ValueError("Environment name must be specified.")

    env = None
    try:
        env = gym.make(environmentName)
    except:
        raise ValueError("Unrecognized environment name: %s" %
                         (str(environmentName)))

    inputObservation = []

    # Average multiple scores for this individual:
    for i in range(timesToRepeat):
        observation = env.reset()
        # print(observation)
        print("Shape: %s" % (str(observation.shape)))
        print("^^^ observation from env.reset() ^^^")
        #####  Losing shape as we go. Need to figure out where/why. #####

        if randomizedObservations:
            observation = getRandomAtariObservation_screen()

        individual.resetForNewTimeSeries()
        runReward = 0

        # Fill in our array of observations:
        allObservations = []
        if greyScaleConversionMethod is not None:
            if greyScaleConversionMethod.lower() == 'max':
                observation = convertAtariObservation_GreyScaleMax(observation)
            else:
                raise ValueError("Unknown greyscale conversion method: %s"
                  % (str(greyScaleConversionMethod)))
        else:
            observation = convertAtariObservation(observation)

        allObservations = [observation] * numObservationsToKeep

        # Run through one time:
        for j in range(maxSteps):
            # If we're using random observations, gather it:
            if randomizedObservations:
                observation = getRandomAtariObservation_screen()

            # Convert the axes and units:
            if greyScaleConversionMethod is not None:
                if greyScaleConversionMethod.lower() == 'max':
                    observation = convertAtariObservation_GreyScaleMax(observation)
                # Errors around unknown conversion methods should be caught above.
            else:
                observation = convertAtariObservation(observation)

            # Add to the end, remove from the front:
            allObservations.append(observation)
            allObservations = allObservations[1:]

            # Get the multiple steps:
            inputObservation = getInputObservationFromListOfFrames(
              allObservations, greyScaleConversionMethod,
              stepCombinations, stepCombinationMethod)

            # Turn our single screen into an array of screens, like the CGP
            # algorithm expects:
            inputObservation = [inputObservation]

            # Get the controller action we want:
            action = np.argmax(
                        convertAtariAction(
                           individual.calculateOutputs(inputObservation)))

            if renderSpeed is not None:
                env.render()
                print("Frame number: {:7d}, Run reward: {:7.2f}\r"\
                  .format(j, runReward), end='')
                if renderSpeed != 0:
                    time.sleep(renderSpeed)

            observation, reward, done, info = env.step(action)
            runReward += reward

            if done:
                # Step beyond the line we were modifying:
                if renderSpeed is not None:
                    print("\nScore: %d" % (runReward))
                allScores.append(runReward)
                break
        else:
            # Step beyond the line we were modifying:
            if renderSpeed is not None:
                print("\nScore: %d" % (runReward))
            allScores.append(runReward)

    env.close()

    # Return the full list of all fitnesses:
    return allScores

# Below many functions which take a single tuple as input (as required by
# GeneralCGPSolver) are provided which call the general fitness functions.
# Currently, all Gym fitness functions run 50 trials using 10 threads.

# We can't just return a single function; we must create a class so that
# Python can pickle it to facilitate multiprocessing:
class generalTupleFitnessFunction:
    def __init__(self, timesToRepeat, envName, useArgmax, maxStepsPerRun=1000, renderSpeed=None, renderMode=None, numThreads=10, envParams=None, npConvert=False, csvFileName=None):
        self.timesToRepeat = timesToRepeat
        self.envName = envName
        self.useArgmax = useArgmax
        self.maxStepsPerRun = maxStepsPerRun
        self.renderSpeed = renderSpeed
        self.renderMode = renderMode
        self.numThreads = numThreads
        self.envParams = envParams
        self.npConvert = npConvert
        self.csvFileName = csvFileName

    def fitnessFunc(self, inputTuple):
        return getGeneralGymFitness(inputTuple[0], self.timesToRepeat,
                                    self.envName, self.useArgmax,
                                    maxStepsPerRun=self.maxStepsPerRun,
                                    renderSpeed=self.renderSpeed,
                                    renderMode=self.renderMode,
                                    numThreads=self.numThreads,
                                    envParams=self.envParams,
                                    npConvert=self.npConvert,
                                    csvFileName=self.csvFileName)

def cartPoleFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 5, 'CartPole-v1', True,
                                maxStepsPerRun=500, renderSpeed=None,
                                numThreads=10)

def pendulumFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'Pendulum-v0', False,
                                maxStepsPerRun=1000, renderSpeed=None,
                                numThreads=10)

def lunarLanderFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'LunarLander-v2', True,
                                maxStepsPerRun=1000, renderSpeed=None,
                                numThreads=10)

def lunarLanderContinuousFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'LunarLanderContinuous-v2',
                                False, maxStepsPerRun=1000, renderSpeed=None,
                                numThreads=10)

def mountainCarFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'MountainCar-v0',
                                True, maxStepsPerRun=1000, renderSpeed=None,
                                numThreads=10)

def mountainCarFitness_modifiedReward(inputTuple):
    return getMountainCarFitness_modifiedReward(inputTuple[0], 50,
                                                renderSpeed=None,
                                                continuous=False,
                                                numThreads=10)

def mountainCarContinuousFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'MountainCarContinuous-v0',
                                False, maxStepsPerRun=1000, renderSpeed=None,
                                numThreads=10)

def mountainCarContinuousFitness_modifiedReward(inputTuple):
    return getMountainCarFitness_modifiedReward(inputTuple[0], 50,
                                                renderSpeed=None,
                                                continuous=True,
                                                numThreads=10)

def bipedalWalkerFitness_modifiedReward(inputTuple):
    return getBipedalWalkerFitness_modifiedReward(inputTuple[0], 50,
                                                  renderSpeed=None,
                                                  hardcore=False,
                                                  numThreads=10)

def bipedalWalkerHardcoreFitness_modifiedReward(inputTuple):
    return getBipedalWalkerFitness_modifiedReward(inputTuple[0], 50,
                                                  renderSpeed=None,
                                                  hardcore=True,
                                                  numThreads=10)

def bipedalWalkerHardcoreFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'BipedalWalkerHardcore-v2',
                                False, maxStepsPerRun=2000, renderSpeed=None,
                                numThreads=10)

def bipedalWalkerFitness(inputTuple):
    return getGeneralGymFitness(inputTuple[0], 50, 'BipedalWalker-v2',
                                False, maxStepsPerRun=2000, renderSpeed=None,
                                numThreads=10)

# Provide a function to get input and output sizes of an environment:
def getActionObservationSizes(env_id):
    """Give the user the size of inputs and outputs expected by an
    environment."""

    # Make sure any modified reward functions check the original names:
    if env_id.endswith('_modifiedReward'):
        env_id = env_id[:-(len('_modifiedReward'))]

    # Check for valid ID:
    if not getIfEnvironmentIsAvailable(env_id):
        return None, None

    tempEnv = gym.make(env_id)
    inputs = tempEnv.action_space
    outputs = tempEnv.observation_space

    # There are a few options for the return types of each.  We need to process
    # them differently:
    if isinstance(inputs, gym.spaces.discrete.Discrete):
        inputs = inputs.n
    elif isinstance(inputs, gym.spaces.box.Box):
        shape = inputs.shape
        if len(shape) == 1:
            inputs = shape[0]
        else:
            inputs = shape

    if isinstance(outputs, gym.spaces.discrete.Discrete):
        outputs = outputs.n
    elif isinstance(outputs, gym.spaces.box.Box):
        shape = outputs.shape
        if len(shape) == 1:
            outputs = shape[0]
        else:
            outputs = shape

    return inputs, outputs

def getEnvironmentTestFunction_tuple(env_id):
    """Return the function that takes an input tuple from GeneralCGPSolver and
    tests the environment provided by the given ID."""
    funcSwitcher = {
      'CartPole-v1' : cartPoleFitness,
      'MountainCar-v0' : mountainCarFitness,
      'MountainCar-v0_modifiedReward' : mountainCarFitness_modifiedReward,
      'MountainCarContinuous-v0' : mountainCarContinuousFitness,
      'MountainCarContinuous-v0_modifiedReward' : mountainCarContinuousFitness_modifiedReward,
      'Pendulum-v0' : pendulumFitness,
      # 'Acrobot-v1' : NEED TO DEFINE,
      'LunarLander-v2' : lunarLanderFitness,
      'LunarLanderContinuous-v2' : lunarLanderContinuousFitness,
      'BipedalWalker-v2' : bipedalWalkerFitness,
      'BipedalWalkerHardcore-v2' : bipedalWalkerHardcoreFitness,
      'BipedalWalker-v2_modifiedReward': bipedalWalkerFitness_modifiedReward,
      'BipedalWalkerHardcore-v2_modifiedReward': bipedalWalkerHardcoreFitness_modifiedReward
    }
    return funcSwitcher.get(env_id, None)

def getMaxScore(env_id):
    """Return a max score for the environment. If a max score isn't defined,
    return a reasonable goal score for the agent to aim for."""
    scoreSwitch = {
      'CartPole-v1' : 500,
      'MountainCar-v0' : -110,
      'MountainCarContinuous-v0' : 90,
      # Maximum position is 0.5, meaning (0.5 * 10) can be added to the score.
      # Require that in the max score:
      'MountainCar-v0_modifiedReward' : -105,
      # Maximum position is 0.45, meaning (0.45 * 1000) can be added to the score.
      # Require that in the max score:
      'MountainCarContinuous-v0_modifiedReward' : 540.0,
      'Pendulum-v0' : -130,
      # 'Acrobot-v1' : NEED TO DEFINE,
      'LunarLander-v2' : 200,
      'LunarLanderContinuous-v2' : 200,
      'BipedalWalker-v2' : 300,
      'BipedalWalkerHardcore-v2' : 300,
      'BipedalWalker-v2_modifiedReward': 300,
      'BipedalWalkerHardcore-v2_modifiedReward': 300,
      'Skiing-ram-v0': 300,  # Not known, just filling in a number.
      'SpaceInvaders-ram-v0': 5000  # Not known, just filling in a number.
    }

    return scoreSwitch.get(env_id, None)

class genericAtariFitness_ram(object):
    """This class allows for the creation of a generic Atari ram fitness
    function while allocating it in such a way to allow multithreading.
    With the proper variables passed in at instantiation, the instance's
    fitnessFunction can be provided to GeneralCGPSolver as the training
    fitness function."""
    def __init__(self,
      environmentName=None, # Name of the Atari environment, must be provided.
      maxSteps=18000, # The maximum number of steps each playthrough gets.
      useArgmax=True, # If True, the individual's output array will be converted
                      # to an integer representing the location in said array
                      # of the max value before being passed into the environment.
      timesToRepeat=10, # Times to test each individual (averaging results)
      numThreads=10, # The number of threads to use when calculating fitnesses.
      renderSpeed=None, # Seconds between frames, None to not render to screen.
      scaleObservations=True): # If True, convert the integer array to floating
                              # point values between -1.0 and 1.0 inclusive.
        self.inputKwargs = {'envName': environmentName,
                            'maxStepsPerRun': maxSteps,
                            'useArgmax': useArgmax,
                            'timesToRepeat': timesToRepeat,
                            'numThreads': numThreads,
                            'renderSpeed': renderSpeed,
                            'scaleObservations': scaleObservations}

    def fitnessFunction(self, inputTuple):
        return getGeneralAtariRamFitness(inputTuple[0], **self.inputKwargs)

class genericAtariFitness_screen(object):
    """This duplicates the RAM version of this class, but for screen inputs.
    Just as a reminder, Atari screen inputs don't function properly yet."""
    def __init__(
      self,
      environmentName=None, # Name of the Atari environment, must be provided.
      maxSteps=18000, # The maximum number of steps each playthrough gets.
      timesToRepeat=10, # Times to test each individual (averaging results)
      renderSpeed=None, # Seconds between frames, None to not render to screen.
      greyScaleConversionMethod=None, # If None, observations aren't converted to
                                      # greyscale. Otherwise expected to be a
                                      # string representing a conversion method.
      numObservationsToKeep=1, # The number of observations/frames to keep for
                               # processing into single inputs.
      stepCombinations=[-1], # The values in the observations array to use in
                             # creating a single input
      stepCombinationMethod=None,  # If not None, a string representing how the
                                   # frames should be combined.
      randomizeObservations=False): # If True, the observation from the environment
                                    # should be discarded and randomized
                                    # observations should be used.
        self.inputKwargs = {
          'environmentName': environmentName,
          'maxSteps': maxSteps,
          'timesToRepeat': timesToRepeat,
          'renderSpeed': renderSpeed,
          'greyScaleConversionMethod': greyScaleConversionMethod,
          'numObservationsToKeep': numObservationsToKeep,
          'stepCombinations': stepCombinations,
          'stepCombinationMethod': stepCombinationMethod,
          'randomizedObservations': randomizeObservations}

    def fitnessFunction(self, inputTuple):
        return getOneAtariFitness_screen(inputTuple[0], **self.inputKwargs)
