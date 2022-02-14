# import sys
# sys.path.insert(0, '/home/jlwilli5/pythonScripts/CGP') # Work
# sys.path.insert(0, '/home/joseph/pythonScripts/CGP') # Home
# mainExperimentFolder = '/home/jlwilli5/CGPExperiments'  # Work
# mainExperimentFolder = '/home/joseph/CGPExperiments' # Home

import MultiCGPTester
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
import gym_jsbsim
import gym_jsbsim.properties as prp

testParams = {
  # Print info:
  'info-aileron_cmd': True,
  'info-airspeed_knots': True,
  'info-altitude_sl_ft': True,
  'info-delta_altitude': True,
  'info-delta_heading': True,
  'info-elevator_cmd': True,
  'info-engine_thrust_lbs': True,
  'info-heading_deg': True,
  'info-rudder_cmd': True,
  'info-sim_time_s': True,
  'info-throttle_cmd': True,
  'info-total_fuel_pounds': True,
  'info-used_fuel_pounds': True,

  # Actions:
  'action-0': 'aileron_cmd',
  'action-1': 'elevator_cmd',
  'action-10': '',
  'action-2': 'throttle_cmd',
  'action-3': '',
  'action-4': '',
  'action-5': '',
  'action-6': '',
  'action-7': '',
  'action-8': '',
  'action-9': '',

  # Delta altitude:
  'delta_altitude_customScaling_max': 1000,
  'delta_altitude_customScaling_min': -1000,
  'delta_altitude_max': 12000,
  'delta_altitude_min': -12000,
  'delta_altitude_scenario_max': 1000,

  # Delta heading:
  'delta_heading_customScaling_max': 60,
  'delta_heading_customScaling_min': -60,
  'delta_heading_max': 180,
  'delta_heading_min': -180,
  'delta_heading_scenario_max': 60,

  # Delta airspeed:
  'delta_airspeed_customScaling_max': 140,
  'delta_airspeed_customScaling_min': 60,
  'delta_airspeed_max': 4400,
  'delta_airspeed_min': 0,
  'delta_airspeed_scenario_max': 40,

  # Initials:
  'initial-aileron_cmd': 0,
  'initial-all_engine_running': -1,
  'initial-elevator_cmd': 0,
  'initial-gear': 0,
  'initial-gear_all_cmd': 0,
  'initial-initial_airspeed_knots': 100,
  'initial-initial_altitude_ft': 5000,
  'initial-initial_heading_deg': 270,
  'initial-initial_latitude_geod_deg': 41.6611,
  'initial-initial_longitude_geoc_deg': -91.5302,
  'initial-initial_p_radps': 0,
  'initial-initial_q_radps': 0,
  'initial-initial_r_radps': 0,
  'initial-initial_roc_fpm': 0,
  'initial-initial_terrain_altitude_ft': 1e-08,
  'initial-initial_v_fps': 0,
  'initial-initial_w_fps': 0,
  'initial-mixture_cmd': 1,
  'initial-rudder_cmd': 0,
  'initial-throttle_cmd': 0.7,
  'max_time_seconds': 1000,

  # Observations:
  'observation-0': 'delta_altitude',
  'observation-1': 'delta_heading',
  'observation-10': '',
  'observation-11': '',
  'observation-12': '',
  'observation-13': '',
  'observation-14': '',
  'observation-15': '',
  'observation-2': 'v_fps',
  'observation-3': 'airspeed_knots',
  'observation-4': 'p_radps',
  'observation-5': 'q_radps',
  'observation-6': 'r_radps',
  'observation-7': 'pitch_rad',
  'observation-8': 'roll_rad',
  'observation-9': 'delta_airspeed',

  # Altitude:
  'target_altitude': 5000,
  'rewardAltitudeWorth': 0.33,
  'worstCaseAltitudeDiff': 1000,
  'numAltitudeStaggerLevels': 10,

  # Airspeed:
  'target_airspeed': 100,
  'rewardAirspeedWorth': 0.33,
  'worstCaseAirspeedDiff': 40,
  'numAirspeedStaggerLevels': 10,

  # Heading:
  'target_heading': 270,
  'rewardHeadingWorth': 0.34,
  'worstCaseHeadingDiff': 60,
  'numHeadingStaggerLevels': 10,

  'sim_steps_per_agent_step': 15,
  'terminalUpdateFunction': 'holdAltitudeHeadingAndAirspeed',
  'rewardFunction': 'staggeredRewardAltitudeHeadingAndAirspeed',
  'print_trace_messages': False,
  'print_task_debug_messages': False
  }

doNothing = TrainingOptimizers.doNothingOptimizer()
act, obs = 3, 10

fitnessFuncClass = GymFitnessFunctions.generalTupleFitnessFunction(
  5, "PAINTTask-v0", False, 4100, renderSpeed=None, numThreads=1, envParams=testParams, npConvert=True)

cgpKwargs = {
  'type': 'FFCGPANN',
  'inputSize': obs,
  'outputSize': act,
  'shape' : {'rowCount': 1, 'colCount': 500, 'maxColForward': -1, 'maxColBack': 501},
  'inputMemory': None,
  'fitnessFunction': fitnessFuncClass.fitnessFunc,
  'functionList': functionLists.funcListANN_singleTan,
  'populationSize': 8,
  'numberParents': 1,
  'parentSelectionStrategy': 'RoundRobin',
  'maxEpochs': 10000,
  'epochModOutput': 10,
  'bestFitness': 3200,
  'pRange': [-1.0, 1.0],
  'constraintRange': [-1.0, 1.0],
  'trainingOptimizer': doNothing,
  'fitnessCollapseFunction': FitnessCollapseFunctions.minOfMeanMedian,
  'completeFitnessCollapseFunction': None,
  'mutationStrategy': {'name': 'activeGene', 'numGenes': [1, 3]},
  'variationSpecificParameters': {'inputsPerNeuron': [2, 9],
                                  'weightRange': [-1.0, 1.0],
                                  'switchValues': [1]},
  'numThreads': 4
}

# Save some results based on our runName
resultsDirectory = "/data/CGPResults/"
runName = "holdAll3"
experimentDirectory = resultsDirectory + runName
csvFileName = experimentDirectory + "/trainingResults.csv"
experimentDescriptionFileName = experimentDirectory + "/description.txt"
periodicSavingFile = experimentDirectory + "/model_"
Path(experimentDirectory).mkdir(parents=True, exist_ok=True)

# Add these details to our cgpKwargs:
cgpKwargs['csvFileName'] = csvFileName
cgpKwargs['periodicSaving'] = {'fileName': periodicSavingFile,
                               'epochMod': cgpKwargs['epochModOutput']}

# Write out all of these details to a file to explain this experiment:
experimentOut = open(experimentDescriptionFileName, 'w')
_ = experimentOut.write("Task Parameters:\n")
for k,v in testParams.items():
    _ = experimentOut.write(f"\t{k}: {v}\n")

_ = experimentOut.write("\n\nCGP Parameters:\n")
for k,v in cgpKwargs.items():
    _ = experimentOut.write(f"\t{k}: {v}\n")

experimentOut.close()

model = GeneralCGPSolver(**cgpKwargs)
model.fit(None, None)

df = pd.DataFrame()
env = gym.make("PAINTTask-v0", **testParams)
done = False
obs = env.reset()

while not done:
    action = model.predict(obs)
    action = np.array(action)
    obs, reward, done, info = env.step(action)
    df = df.append(info, ignore_index=True)

import PCGP1D

ind = PCGP1D.PCGP1DIndividual(
  type='PCGP1D',
  inputSize=5,
  outputSize=10,
  pRange=[0.0, 1.0],
  constraintRange=[-1.0, 1.0],
  shape=None,
  functionList=functionLists.atariFuncList,
  PCGP1DSpecificParameters={
    'iStart': -0.5,
    'recursive': 1.0,
    'weights': [1.0, 1.0],
    'minComputationNodes': 5,
    'maxComputationNodes': 10,
    'inputsPerNode': [2, 2],
    'crossover': [{'name': 'test', 'numIndividualsToCreate': 5}]
  }
)



logging.debug("Stuff %s, and other stuff %d", firstString, secondInt)


ind.addComputationalNodeToGenotype(ind.createRandomComputationalNode())

doNothing = TrainingOptimizers.doNothingOptimizer()
minSwitchOptimizer = TrainingOptimizers.newCollapseFunctionOptimizer(
  FitnessCollapseFunctions.minimum, fitnessChange=190.0)
hardcoreFitnessChanger = TrainingOptimizers.newFitnessFunctionOptimizer(
  GymFitnessFunctions.bipedalWalkerHardcoreFitness, fitnessChange=175.0)

# Get environment details from GymFitnessFunctions to help prevent errors:
environmentName = "MountainCarContinuous-v0_modifiedReward"
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
experimentFolder = "%s/mountainCarContinuous_SanityCheck3" % (mainExperimentFolder)

tester = MultiCGPTester.MultiCGPTester(
[
{
'type': ['FFCGPANN'],
'inputSize': [obs],
'outputSize': [act],
'shape__rowCount': [1],
'shape__colCount': [300],
'shape__maxColForward': [-1],
'shape__maxColBack': [1001],
'inputMemory': [None],
'fitnessFunction': [testFunc],
'functionList': [functionLists.funcListANN_singleTan],
'populationSize': [7],
'numberParents': [1],
'maxEpochs': [20000],
'epochModOutput': [10],
'bestFitness': [maxScore],
'pRange': [[-1.0, 1.0]],
'constraintRange': [[-2.0, 2.0]],
'trainingOptimizer': [doNothing],
'fitnessCollapseFunction': [FitnessCollapseFunctions.minOfMeanMedian],
'completeFitnessCollapseFunction': [None],
'mutationStrategy': [{'name': 'activeGene', 'numGenes': [1, 1]}],
'vsp__inputsPerNeuron': [[2, 2], [3, 3], [4, 4]],
'vsp__weightRange': [[-1.0, 1.0]],
'vsp__switchValues': [[1]],
}
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
runsPerVariation = 1,
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



################# MESSO DEBUGGING ###################
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy
import pandas as pd
import FlightDatabaseProcessor
import adaptive2019ManeuverGrades as grade

cluster = Cluster(['build_cassandra-1_1'], load_balancing_policy=RoundRobinPolicy())
session = cluster.connect('adaptive_2019')

clear_run_data_query = 'TRUNCATE run_data;'
session.execute(clear_run_data_query)

tableColumns = [
  ['flightstate', ['altitude_msl', 'body_pitch_rate', 'body_roll_rate',
                   'body_yaw_rate', 'calibrated_airspeed',
                   'computed_airspeed', 'down_velocity',
                   'gps_ground_speed', 'latitude', 'mach',
                   'longitude', 'mag_heading', 'pitch', 'roll',
                   'true_airspeed', 'vertical_speed',
                   'load_factor_normal_body']],
  ['controlinput', ['rawx', 'rawy', 'rangex', 'rangey']],
  ['workload', ['logworkload']],
  ['masterarm', ['currentstate', 'previousstate']],
  ['bomb', ['distime', 'eventid', 'firingid', 'locationx', 'locationy',
            'locationz', 'munitionsid', 'pdutype', 'targetid',
            'velocityx', 'velocityy', 'velocityz']]
]

def get_tag_indices(run, tags, start_time, end_time):
    bomb_run_indices = grade.findBombRuns(run, tags, start_time, end_time)
    if len(bomb_run_indices) > 0:
        print("Found %d possible bombing runs." % (len(bomb_run_indices)))
    else:
        print("Found no bombing runs in the data.")

    # There should really only be one bombing run, so we'll take the first one with the tags we need:
    indices_to_use = None
    for tempIndices in bomb_run_indices:
        if 'ABEAM' in tempIndices and 'ARC' in tempIndices:
            indices_to_use = tempIndices
            break

    if indices_to_use is None and len(bomb_run_indices) > 0:
        print("No bombing run with 'ARC' and 'ABEAM' indices found. Using middle index.")
        indices_to_use = bomb_run_indices[int(len(bomb_run_indices) / 2.0)]
    elif indices_to_use is None:
        print("No bombing run indices found.")

    bomb_run_indices = indices_to_use
    print('Found bomb run indices: %s', bomb_run_indices)

    # Convert all timestamps to relative timestamps
    #for k,v in bomb_run_indices.items():
    #    if k not in ['start', 'stop']:
    #        bomb_run_indices[k] -= bomb_run_indices['start']
    return bomb_run_indices

processor = FlightDatabaseProcessor.FlightDatabaseProcessor(session)
time_query = 'SELECT * FROM run WHERE subject_id = {} AND start_time = {} and end_time = {} allow filtering;'.format(subjectId, startTime, endTime)
processor.fillFullGradingTable(time_query, tableColumns, 'run_data', 0, nullOrDataTables=['bomb'])

run_query = 'SELECT * FROM run_data;'
run_data = pd.DataFrame(list(session.execute(run_query)))

tags_query = 'SELECT * FROM tags where subject_id = {} AND start_time >= {} AND end_time <= {} allow filtering;'.format(subjectId, startTime, endTime)
tags_data = pd.DataFrame(list(session.execute(tags_query)))

if maneuver == 'popupbombingrun':
    indices = get_tag_indices(run, tags, start_time, end_time)
    print('tag indices: %s', indices)
    result, ignoredIndices = grade.gradePopupBombingRun(run, indices)
elif maneuver == 'bombrun':
    indices = get_tag_indices(run, tags, start_time, end_time)
    print('tag indices: %s', indices)
    result, ignoredIndices = grade.gradeBombRun(run, indices)
elif maneuver == 'turnpatternright':
    result = grade.gradeTurnPatternRight(run)
elif maneuver == 'turnpatternleft':
    result = grade.gradeTurnPatternLeft(run)
elif maneuver == 'diveturnpull':
    result = grade.gradeDiveTurnPull(run)
elif maneuver == 'aileronrollright':
    result = grade.gradeAileronRollRight(run)
elif maneuver == 'aileronrollleft':
    result = grade.gradeAileronRollLeft(run)
elif maneuver == 'popup15degreediveright':
    result, ignoredSectionDict = grade.gradePopUp15DegreeDiveRight(run)
elif maneuver == 'popup15degreediveleft':
    result, ignoredSectionDict = grade.gradePopUp15DegreeDiveLeft(run)
else:
    raise ValueError("Unrecognized maneuver: %s" % (maneuver))

print("Results:")
if isinstance(result, list):
    for oneResult in result:
        print("--------------------")
        grade.printOrderedDict(oneResult)
else:
    grade.printOrderedDict(result)
