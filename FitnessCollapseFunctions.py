"""These are functions used to take a list of fitnesses from a single
individual and return a single value to compare with others. Many fitness
functions return a list of fitnesses calculated for an individual. These
functions provide various ways in which we can reduce that list down to a
single value which can be used for determining the best individual(s)."""
import statistics as stat

def mean(fitnessList):
    return stat.mean(fitnessList)

def median(fitnessList):
    return stat.median(fitnessList)

def avgMeanMedian(fitnessList):
    return (stat.mean(fitnessList) + stat.median(fitnessList)) / 2.0

def minOfMeanMedian(fitnessList):
    return min(stat.mean(fitnessList), stat.median(fitnessList))

def minimum(fitnessList):
    return min(fitnessList)

def maximum(fitnessList):
    return max(fitnessList)

# Need to be careful with this one. It may help push the evolution in the
# right direction, but the user needs to either change the maximum possible
# fitness or use a TrainingOptimizer to change this collapse function during
# training:
def minOfMeanMedianPlusMin(fitnessList):
    return min(stat.mean(fitnessList), stat.median(fitnessList)) + min(fitnessList)
