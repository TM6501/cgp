"""This file provides a list of functions currently being used in computational
nodes in various CGP individuals. Functions aren't all each given their own
header and description since most are self-explanatory and the comments would
end up being redundant while cluttering the file."""
import numpy as np
import scipy
import math
import copy
import numbers
import random

# Binary (boolean) inputs / outputs.
def And(X, Y, P):
    return int(X and Y)

def Or(X, Y, P):
    return int(X or Y)

def Nand(X, Y, P):
    return int(not (X and Y))

def Nor(X, Y, P):
    return int(not (X or Y))

def Xor(X, Y, P):
    return int((X or Y) and not (X and Y))

def AndNotY(X, Y, P):
    return X and not Y

# Floating point logical-equivalents. They will consider > 0 as "True" and
# <= 0 as "False". They will return 1.0 for "True" and -1.0 for "False"
def float_And(X, Y, P):
    if X > 0.0 and Y > 0.0:
        return 1.0
    else:
        return -1.0

def float_Or(X, Y, P):
    if X > 0.0 or Y > 0.0:
        return 1.0
    else:
        return -1.0

def float_Nand(X, Y, P):
    return -float_And(X, Y, P)

def float_Nor(X, Y, P):
    return -float_Or(X, Y, P)

def float_Xor(X, Y, P):
    # X or Y, not both:
    if (X > 0.0 and Y <= 0.0) or (X <= 0.0 and Y > 0.0):
        return 1.0
    else:
        return -1.0

def float_AndNotY(X, Y, P):
    if X > 0.0 and Y <= 0.0:
        return 1.0
    else:
        return -1.0


# Functions used by neurons in neural networks. They take a single input, which
# is usually the sum of their inputs, each multiplied by a weight:
def ANN_Sigmoid(X):
    return 1 / (1 + math.exp(-X))

def ANN_Tanh(X):
    return math.tanh(X)

def ANN_Relu(X):
    return X if X >=0 else 0

################ Functions with some measure of randomness: #############
def randInt_0_X(X, Y, P):
    return random.randint(min(0, X), max(0, X))

def randInt_X_Y(X, Y, P):
    return random.randint(min(X, Y), max(X, Y))

def randFloat_0_X(X, Y, P):
    return random.uniform(0.0, X)

def randFloat_X_1(X, Y, P):
    return random.uniform(X, 1.0)

def randFloat_X_Y(X, Y, P):
    return random.uniform(X, Y)

def randXChance1(X, Y, P):
    return 1 if random.random() <= X else 0

def randXorY(X, Y, P):
    return X if random.random() <= 0.5 else Y

def randPChanceXElseY(X, Y, P):
    return X if random.random() <= P else Y

# This is the general functian caller. It allows a simple function to call
# the appropriate real functian based upon the input types:
def callRealFunc(X, Y, P, typeList):
    for option in typeList:
        # If the first type matches and the second either matches or is None:
        if (option[0] is None or isinstance(X, option[0]) and \
           (option[1] is None or isinstance(Y, option[1]))):

            # Call the provided function. If we hit an error, pass X through:
            val = None
            try:
                val = option[2](X, Y, P)
                # Only check for nan with numbers:
                if isinstance(val, numbers.Number) and math.isnan(val):
                    val = X
            except (ZeroDivisionError, OverflowError, ValueError):
                val = X

            return val

    # If we made it through the list of available options and didn't find a
    # match, then we just pass through X (treat this function as a wire).
    # print("Couldn't find match with types %s and %s. Function acting as wire." %
    #       (str(type(X)), str(type(Y))))
    # print(typeList)
    return X

def applyFunctionToMatrix(matrix, func):
    retVal = copy.copy(matrix)
    initialShape = retVal.shape
    retVal = retVal.flatten()
    for i in range(retVal.size):
        try:
            retVal[i] = func(retVal[i])
        except (ZeroDivisionError, OverflowError):
            # Leave it unchanged:
            pass

    retVal = retVal.reshape(initialShape)
    return retVal

def closeToZero(value):
    if value >= -0.0001 and value <= 0.0001:
        return True
    else:
        return False

def getMatricesMinimumDimensions(X, Y):
    """Reduce the 2 input matrices to the same dimensionality and size,
    focusing on the beginning/top of each matrix.

    Arguments:
        X - A numpy.ndarray with dimensionality of 1 or 2.
        Y - A numpy.ndarray with dimensionality of 1 or 2.

    Return:
        X and Y with their dimensionality and size each reduced the maximum
        that can be achieved by both."""

    minDimOne = min(X.shape[0], Y.shape[0])

    # Option 1: Both have 2 dimensions:
    if len(X.shape) == 2 and len(Y.shape) == 2:
        minDimTwo = min(X.shape[1], Y.shape[1])
        return X[0:minDimOne, 0:minDimTwo], Y[0:minDimOne, 0:minDimTwo]

    # Option 2: X has dimension 2, Y has dimension 1:
    elif len(X.shape) == 2 and len(Y.shape) == 1:
        return X[0:minDimOne, 0], Y[0:minDimOne]

    # Option 3: X has dimension 1, Y has dimension 2:
    elif len(X.shape) == 1 and len(Y.shape) == 2:
        return X[0:minDimOne], Y[0:minDimOne, 0]

    # Option 4: Both X and Y have only a single dimension:
    elif len(X.shape) == 1 and len(Y.shape) == 1:
        return X[0:minDimOne], Y[0:minDimOne]

    raise ValueError("X and Y need to be numpy.ndarrays, each with \
dimensionality between 1 and 2 inclusive.")


# Below here are the functions defined by https://arxiv.org/pdf/1806.05695.pdf
# for playing Atari games. They should also prove useful for many computer
# vision tasks. They are all designed to work between -1.0 and 1.0. Outputs
# should be constrained to that range if the user wants to use these functions.

# Many mathematical operations (thanks to np.ndarray's nice syntax) don't
# require separate functions depending upon the types of the arguments:

########## Mathematical Functions: ##########
def ADD_ATARI_float(X, Y, P):
    return (X + Y) / 2.0

def ADD_ATARI_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (X1 + Y1) / 2.0

def ADD_ATARI(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, ADD_ATARI_matrix),
                                  (np.ndarray, numbers.Number, ADD_ATARI_float),
                                  (numbers.Number, np.ndarray, ADD_ATARI_float),
                                  (numbers.Number, numbers.Number, ADD_ATARI_float)])

def do_ADD(X, Y, P):
    return X + Y

def ADD(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_ADD)])

def AMINUS_ATARI_float(X, Y, P):
    return abs(X-Y) / 2.0

def AMINUS_ATARI_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return abs(X1 - Y1) / 2.0

def AMINUS_ATARI(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray,
                                   AMINUS_ATARI_matrix),
                                  (np.ndarray, numbers.Number, AMINUS_ATARI_float),
                                  (numbers.Number, np.ndarray, AMINUS_ATARI_float),
                                  (numbers.Number, numbers.Number, AMINUS_ATARI_float)])

def do_AMINUS(X, Y, P):
    return X - Y

def AMINUS(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_AMINUS)])

def do_CMINUS(X, Y, P):
    return X - P

def CMINUS(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_CMINUS)])

def MULT_ATARI_float(X, Y, P):
    return X * Y

def MULT_ATARI_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return X1 * Y1

def MULT_ATARI(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray,
                                   MULT_ATARI_matrix),
                                  (np.ndarray, numbers.Number, MULT_ATARI_float),
                                  (numbers.Number, np.ndarray, MULT_ATARI_float),
                                  (numbers.Number, numbers.Number, MULT_ATARI_float)])

def do_MULT(X, Y, P):
    # This is an element-by-element multiplication, not matrix multiplication:
    return X * Y

def MULT(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_MULT)])

def do_CMULT(X, Y, P):
    return X * P

def CMULT(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_CMULT)])

def do_INV(X, Y, P):
    # If there is an error, treat this function as a wire (pass X through)
    try:
        return 1.0 / X
    except ZeroDivisionError:
        return X

def INV(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_INV)])

def do_ABS(X, Y, P):
    return abs(X)

def ABS(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_ABS)])

def do_SQRT(X, Y, P):
    return abs(X) ** 0.5

def SQRT(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_SQRT)])

def do_CPOW(X, Y, P):
    return abs(X) ** (P + 1)

def CPOW(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_CPOW)])

def YPOW_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return abs(X1) ** abs(Y1)

def YPOW_float(X, Y, P):
    return abs(X) ** abs(Y)

def YPOW(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, YPOW_matrix),
                                  (np.ndarray, numbers.Number, YPOW_float),
                                  (numbers.Number, np.ndarray, YPOW_float),
                                  (numbers.Number, numbers.Number, YPOW_float)])

def do_EXPX(X, Y, P):
    # If there is an error, treat this function as a wire (pass X through)
    return ((math.e ** X) - 1) / (math.e - 1)

def EXPX(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_EXPX)])

def SINX_matrix(X, Y, P):
    return applyFunctionToMatrix(X, math.sin)

def SINX_float(X, Y, P):
    return math.sin(X)

def SINX(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, SINX_matrix),
                                  (numbers.Number, None, SINX_float)])

def SQRTXY_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (((X1 ** 2.0) + (Y1 ** 2.0)) ** 0.5) / (2 ** 0.5)

def SQRTXY_float(X, Y, P):
    return (((X ** 2.0) + (Y ** 2.0)) ** 0.5) / (2 ** 0.5)

def SQRTXY(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, SQRTXY_matrix),
                                  (np.ndarray, numbers.Number, SQRTXY_float),
                                  (numbers.Number, np.ndarray, SQRTXY_float),
                                  (numbers.Number, numbers.Number, SQRTXY_float)])

def ACOS_matrix_atari(X, Y, P):
    return applyFunctionToMatrix(X, math.acos) / math.pi

def ACOS_matrix(X, Y, P):
    return applyFunctionToMatrix(X, math.acos)

def ACOS_float(X, Y, P):
    return math.acos(X)

def ACOS_float_atari(X, Y, P):
    return math.acos(X) / math.pi

def ACOS(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ACOS_matrix),
                                  (numbers.Number, None, ACOS_float)])

def ACOS_atari(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ACOS_matrix_atari),
                                  (numbers.Number, None, ACOS_float_atari)])

def ASIN_matrix_atari(X, Y, P):
    return applyFunctionToMatrix(X, math.asin) * (2.0 / math.pi)

def ASIN_matrix(X, Y, P):
    return applyFunctionToMatrix(X, math.asin)

def ASIN_float(X, Y, P):
    return math.asin(X)

def ASIN_float_atari(X, Y, P):
    return (math.asin(X) * 2.0) / math.pi

def ASIN(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ASIN_matrix),
                                  (numbers.Number, None, ASIN_float)])

def ASIN_atari(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ASIN_matrix_atari),
                                  (numbers.Number, None, ASIN_float_atari)])

def ATAN_matrix(X, Y, P):
    return applyFunctionToMatrix(X, math.atan)

def ATAN_matrix_atari(X, Y, P):
    return applyFunctionToMatrix(X, math.atan) * (4.0 / math.pi)

def ATAN_float(X, Y, P):
    return math.atan(X)

def ATAN_float_atari(X, Y, P):
    return (4.0 * math.atan(X)) / math.pi

def ATAN_atari(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ATAN_matrix_atari),
                                  (numbers.Number, None, ATAN_float_atari)])

def ATAN(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ATAN_matrix),
                                  (numbers.Number, None, ATAN_float)])

def do_COS(X, Y, P):
    return math.cos(X)

def COS(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_COS)])

def do_TAN(X, Y, P):
    return math.tan(X)

def TAN(X, Y, P):
    return callRealFunc(X, Y, P, [(None, None, do_TAN)])

########## Statistical Functions: ##########
# Statistical functions are built to only allow matrix input. Other inputs
# are passed through.

def STDDEV_matrix(X, Y, P):
    return np.std(X)

def STDDEV(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, STDDEV_matrix)])

def SKEW_matrix(X, Y, P):
    return scipy.stats.skew(X)

def SKEW(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, SKEW_matrix)])

def KURTOSIS_matrix(X, Y, P):
    return scipy.stats.kurtosis(X)

def KURTOSIS(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, KURTOSIS_matrix)])

def MEAN_matrix(X, Y, P):
    return X.mean()

def MEAN(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, MEAN_matrix)])

def RANGE_matrix(X, Y, P):
    return np.max(X) - np.min(X)

def RANGE_matrix_atari(X, Y, P):
    return np.max(X) - np.min(X) - 1

def RANGE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, RANGE_matrix)])

def RANGE_atari(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, RANGE_matrix_atari)])

def ROUND_matrix(X, Y, P):
    return np.around(X)

def ROUND_float(X, Y, P):
    if X >= 0:
        return int(X + 0.5)
    else:
        return int(X - 0.5)

def ROUND(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ROUND_matrix),
                                  (numbers.Number, None, ROUND_float)])

def CEIL_matrix(X, Y, P):
    return np.ceil(X)

def CEIL(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, CEIL_matrix)])

def FLOOR_matrix(X, Y, P):
    return np.floor(X)

def FLOOR(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, FLOOR_matrix)])

def MAX1_matrix(X, Y, P):
    return X.max()

def MAX1(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, MAX1_matrix)])

def MIN1_matrix(X, Y, P):
    return X.min()

def MIN1(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, MIN1_matrix)])

########## Comparison Functions: ##########

def LT_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (X1 < Y1).astype(float)

def LT_matrix_float(X, Y, P):
    return (X < Y).astype(float)

def LT_float(X, Y, P):
    return float(X < Y)

def LT(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, LT_matrix_matrix),
                                  (np.ndarray, numbers.Number, LT_matrix_float),
                                  (numbers.Number, numbers.Number, LT_float),
                                  (numbers.Number, np.ndarray, LT_matrix_float)])

def LTE_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (X1 <= Y1).astype(float)

def LTE_matrix_float(X, Y, P):
    return (X <= Y).astype(float)

def LTE_float(X, Y, P):
    return float(X <= Y)

def LTE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, LTE_matrix_matrix),
                                  (np.ndarray, numbers.Number, LTE_matrix_float),
                                  (numbers.Number, numbers.Number, LTE_float),
                                  (numbers.Number, np.ndarray, LTE_matrix_float)])

def GT_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (X1 > Y1).astype(float)

def GT_matrix_float(X, Y, P):
    return (X > Y).astype(float)

def GT_float(X, Y, P):
    return float(X > Y)

def GT(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, GT_matrix_matrix),
                                  (np.ndarray, numbers.Number, GT_matrix_float),
                                  (numbers.Number, numbers.Number, GT_float),
                                  (numbers.Number, np.ndarray, GT_matrix_float)])

def GTE_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    return (X1 >= Y1).astype(float)

def GTE_matrix_float(X, Y, P):
    return (X >= Y).astype(float)

def GTE_float(X, Y, P):
    return float(X >= Y)

def GTE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, GTE_matrix_matrix),
                                  (np.ndarray, numbers.Number, GTE_matrix_float),
                                  (numbers.Number, numbers.Number, GTE_float),
                                  (numbers.Number, np.ndarray, GTE_matrix_float)])

def GTEP(X, Y, P):
    return float(X >= P)

def LTEP(X, Y, P):
    return float(X <= P)

def MAX2_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    # Use numpy to apply the max function element-wise to a matrix:
    def myMax(inX, inY):
        return max(inX, inY)

    vfunc = np.vectorize(myMax)

    return vfunc(X1, Y1)

def MAX2_matrix_float(X, Y, P):
    # Use numpy to apply the max function element-wise to a matrix:
    def myMax(inX, inY):
        return max(inX, inY)

    vfunc = np.vectorize(myMax)

    return vfunc(X, Y)

def MAX2_float(X, Y, P):
    return max(X, Y)

def MAX2(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, MAX2_matrix_matrix),
                                  (np.ndarray, numbers.Number, MAX2_matrix_float),
                                  (numbers.Number, numbers.Number, MAX2_float),
                                  (numbers.Number, np.ndarray, MAX2_matrix_float)])

def MIN2_matrix_matrix(X, Y, P):
    X1, Y1 = getMatricesMinimumDimensions(X, Y)
    # Use numpy to apply the min function element-wise to a matrix:
    def myMin(inX, inY):
        return min(inX, inY)

    vfunc = np.vectorize(myMin)

    return vfunc(X1, Y1)

def MIN2_matrix_float(X, Y, P):
    # Use numpy to apply the min function element-wise to a matrix:
    def myMin(inX, inY):
        return min(inX, inY)

    vfunc = np.vectorize(myMin)

    return vfunc(X, Y)

def MIN2_float(X, Y, P):
    return min(X, Y)

def MIN2(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, MIN2_matrix_matrix),
                                  (np.ndarray, numbers.Number, MIN2_matrix_float),
                                  (numbers.Number, numbers.Number, MIN2_float),
                                  (numbers.Number, np.ndarray, MIN2_matrix_float)])

########## List Processing Functions: ##########
# These functions are described as list-processing in the paper; however,
# many can be applied to matrices as well.

def SPLIT_BEFORE_list(X, Y, P):
    amount = (P + 1.0) / 2.0
    retVal = copy.copy(X)

    cut = max(math.ceil(amount * retVal.shape[0]), 1)
    retVal = retVal[:cut]
    return retVal

def SPLIT_AFTER_list(X, Y, P):
    amount = (P + 1.0) / 2.0
    retVal = copy.copy(X)

    cut = min(math.floor(amount * retVal.shape[0]), retVal.shape[0] - 1)
    retVal = retVal[cut:]
    return retVal

def SPLIT_BEFORE_X_matrix(X, Y, P):
    # Single dimension, easier split:
    if len(X.shape) == 1:
        return SPLIT_BEFORE_list(X, Y, P)

    # P is between -1.0 and 1.0. We use it to get the percentage of the matrix
    # we want.
    amount = (P + 1.0) / 2.0
    retVal = copy.copy(X)

    # If we have only a single dimension, we only cut in that dimension:
    if retVal.shape[0] == 1 or retVal.shape[1] == 1:
        transposed = False
        if retVal.shape[1] == 1:
            transposed = True  # We'll re-transpose on the way out
            retVal = retVal.transpose()

        # Get the amount we're cutting off:
        cut = max(math.ceil(amount * retVal.shape[0]), 1)
        retVal = retVal[:cut, :]
        if transposed:
            retVal = retVal.transpose()

    # We leave the number of columns (X-dimension) intact and cut the Y
    # dimensions by the amount provedided:
    else:
        cut = max(math.ceil(amount * retVal.shape[0]), 1)
        retVal = retVal[:cut, :]
    return retVal

def SPLIT_AFTER_X_matrix(X, Y, P):
    # Single dimension, easier split:
    if len(X.shape) == 1:
        return SPLIT_AFTER_list(X, Y, P)

    amount = (P + 1.0) / 2.0
    retVal = copy.copy(X)

    # If we have only a single dimension, we only cut in that dimension:
    if retVal.shape[0] == 1 or retVal.shape[1] == 1:
        transposed = False
        if retVal.shape[1] == 1:
            transposed = True  # We'll re-transpose on the way out
            retVal = retVal.transpose()

        # Get the amount we're cutting off:
        cut = min(math.floor(amount * retVal.shape[0]), retVal.shape[0] - 1)
        retVal = retVal[cut:, :]
        if transposed:
            retVal = retVal.transpose()

    # We leave the number of columns (X-dimension) intact and cut the Y
    # dimensions by the amount provedided:
    else:
        cut = min(math.floor(amount * retVal.shape[0]), retVal.shape[0] - 1)
        retVal = retVal[cut:, :]
    return retVal

def SPLIT_BEFORE_X(X, Y, P):
    # This function will return a percentage of the given matrix while keeping
    # the X dimension at the same size. If a single dimension array is
    # provided, it will be split at that same boundary. Any scalar in the X
    # position means we'll treat this as a pass-through.
    return callRealFunc(X, Y, P, [(np.ndarray, None, SPLIT_BEFORE_X_matrix)])

def SPLIT_AFTER_X(X, Y, P):
    # The equivalent of SPLIT_BEFORE_X, but taking the bottomt of the matrix
    # rather than the beginning.
    return callRealFunc(X, Y, P, [(np.ndarray, None, SPLIT_AFTER_X_matrix)])

def SPLIT_BEFORE_Y_matrix(X, Y, P):
    # Flip our input and output and we can get the same result from the
    # X-split function:
    return SPLIT_BEFORE_X_matrix(X.transpose(), Y, P).transpose()

def SPLIT_AFTER_Y_matrix(X, Y, P):
    # Flip our input and output and we can get the same result from the
    # X-split function:
    return SPLIT_AFTER_X_matrix(X.transpose(), Y, P).transpose()

def SPLIT_BEFORE_Y(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, SPLIT_BEFORE_Y_matrix)])

def SPLIT_AFTER_Y(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, SPLIT_AFTER_Y_matrix)])

def SPLIT_BEFORE_SQUARE_matrix(X, Y, P):
    # If we have a 1-D array, let a different function handle it:
    if len(X.shape) == 1 or X.shape[0] == 1 or X.shape[1] == 1:
        return SPLIT_BEFORE_X_matrix(X, Y, P)

    amount = (P + 1.0) / 2.0
    # Force a minimum of 1 return value:
    numReturnVals = max(float(X.shape[0] * X.shape[1]) * amount, 1.0)
    retVal = copy.copy(X)

    # Make sure we know who is smallest:
    xSmaller = X.shape[1] < X.shape[0]

    # Get our dimensions and the proposed cut dimensions:
    minDim = min(retVal.shape)
    maxDim = max(retVal.shape)
    minCut = min(math.floor(math.sqrt(numReturnVals)), minDim)

    if minCut < 1:
        # Max out our min dimension and set the max dimension such that we still
        # get the number of returned values we want:
        minCut = 1

    # We can't always round / floor / ceil the maxCut.  Need to compare to see
    # what would get us closest to the number of return values requested:
    maxCut = numReturnVals / minCut
    ceilDist = abs((minCut * math.ceil(maxCut)) - numReturnVals)
    floorDist = abs((minCut * math.floor(maxCut)) - numReturnVals)
    if ceilDist < floorDist:
        maxCut = math.ceil(maxCut)
    else:
        maxCut = math.floor(maxCut)

    # Return based upon which dimension is smaller:
    if xSmaller:
        retVal = retVal[:maxCut, :minCut]
    else:
        retVal = retVal[:minCut, :maxCut]

    return retVal

def SPLIT_AFTER_SQUARE_matrix(X, Y, P):
    # If we have a 1-D array, let a different function handle it:
    if len(X.shape) == 1 or X.shape[0] == 1 or X.shape[1] == 1:
        return SPLIT_AFTER_X_matrix(X, Y, P)

    amount = (P + 1.0) / 2.0
    # Force a minimum of 1 return value:
    numReturnVals = max(float(X.shape[0] * X.shape[1]) * amount, 1.0)
    retVal = copy.copy(X)

    # Make sure we know who is smallest:
    xSmaller = X.shape[1] < X.shape[0]

    # Get our dimensions and the proposed cut dimensions:
    minDim = min(retVal.shape)
    maxDim = max(retVal.shape)
    minCut = min(math.floor(math.sqrt(numReturnVals)), minDim)

    if minCut < 1:
        # Max out our min dimension and set the max dimension such that we still
        # get the number of returned values we want:
        minCut = 1

    # We can't always round / floor / ceil the maxCut.  Need to compare to see
    # what would get us closest to the number of return values requested:
    maxCut = numReturnVals / minCut
    ceilDist = abs((minCut * math.ceil(maxCut)) - numReturnVals)
    floorDist = abs((minCut * math.floor(maxCut)) - numReturnVals)
    if ceilDist < floorDist:
        maxCut = math.ceil(maxCut)
    else:
        maxCut = math.floor(maxCut)

    # Return based upon which dimension is smaller:
    if xSmaller:
        retVal = retVal[-maxCut:, -minCut:]
    else:
        retVal = retVal[-minCut:, -maxCut:]

    return retVal

def SPLIT_BEFORE_SQUARE(X, Y, P):
    # This function will return a percentage of the total elements in the given
    # matrix as a square starting in the top left. For example, a 10x10 matrix
    # has 100 inputs. If we want 30% of those, that would be 30 inputs. A
    # 5x6 matrix would then be returned. The percentage is not guaranteed, but
    # it will be close. Square is not guaranteed, but it will be within one
    # step, if possible.  If we cannot return the requested amount of inputs
    # because one dimension is too small to return something square-ish, that
    # dimension will get maxed out.
    return callRealFunc(X, Y, P, [(np.ndarray, None,
                                   SPLIT_BEFORE_SQUARE_matrix)])

def SPLIT_AFTER_SQUARE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None,
                                   SPLIT_AFTER_SQUARE_matrix)])

def RANGE_IN_list(X, Y, P):
    minPercent = min((Y + 1.0) / 2.0, (P + 1.0) / 2.0)
    maxPercent = max((Y + 1.0) / 2.0, (P + 1.0) / 2.0)

    minCut = int(max(round((X.shape[0]) * minPercent), 0))
    maxCut = int(min(round((X.shape[0]) * maxPercent), X.shape[0]))

    # Always need to return a matrix with at least one value in it:
    while minCut >= maxCut:
        # Skew towards the beginning of the list rather than the end. Only
        # increas the maxCut if the minCut shrank and it didn't fix the
        # problem:
        if minCut > 0:
            minCut -= 1

        if minCut >= maxCut and maxCut < X.shape[0]:
            maxCut += 1

    return X[minCut:maxCut]

def RANGE_INX_matrix_float(X, Y, P):
    # If we only have a single dimension, return the range in that:
    if len(X.shape) == 1:
        return RANGE_IN_list(X, Y, P)

    # Otherwise we return a section of the x-dimension (all of Y):
    minPercent = min((Y + 1.0) / 2.0, (P + 1.0) / 2.0)
    maxPercent = max((Y + 1.0) / 2.0, (P + 1.0) / 2.0)

    minCut = int(max(round((X.shape[0]) * minPercent), 0))
    maxCut = int(min(round((X.shape[0]) * maxPercent), X.shape[0]))

    # Always need to return a matrix with at least one value in it:
    while minCut >= maxCut:
        # Skew towards the beginning of the list rather than the end. Only
        # increas the maxCut if the minCut shrank and it didn't fix the
        # problem:
        if minCut > 0:
            minCut -= 1

        if minCut >= maxCut and maxCut < X.shape[0]:
            maxCut += 1

    return X[minCut:maxCut, :]

def RANGE_INY_matrix_float(X, Y, P):
    return RANGE_INX_matrix_float(X.transpose(), Y, P).transpose()

def RANGE_INX_matrix_matrix(X, Y, P):
    # Use the average of the Y matrix:
    return RANGE_INX_matrix_float(X, np.mean(Y), P)

def RANGE_INY_matrix_matrix(X, Y, P):
    return RANGE_INY_matrix_float(X, np.mean(Y), P)

def RANGE_INX(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray,
                                   RANGE_INX_matrix_matrix),
                                  (np.ndarray, numbers.Number, RANGE_INX_matrix_float)])

def RANGE_INY(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray,
                                   RANGE_INY_matrix_matrix),
                                  (np.ndarray, numbers.Number, RANGE_INY_matrix_float)])

def INDEX_Y_float(X, Y, P):
    percent = (Y + 1.0) / 2.0
    return X.flatten()[round(percent * X.size)]

def INDEX_Y_matrix(X, Y, P):
    return INDEX_Y_float(X, np.mean(Y), P)

def INDEX_Y(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, np.ndarray, INDEX_Y_matrix),
                                  (np.ndarray, numbers.Number, INDEX_Y_float)])

def INDEX_P(X, Y, P):
    percent = (P + 1.0) / 2.0
    return X.flatten()[round(percent * X.size)]

def VECTORIZE_matrix(X, Y, P):
    return X.flatten()

def VECTORIZE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, VECTORIZE_matrix)])

def FIRST_matrix(X, Y, P):
    return X.flatten()[0]

def FIRST(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, FIRST_matrix)])

def LAST_matrix(X, Y, P):
    return X.flatten()[X.size - 1]

def LAST(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, FIRST_matrix)])

def DIFFERENCES_matrix(X, Y, P):
    retVal = np.diff(X.flatten())
    # Must always return a matrix of at least size one; np.diff could give
    # an empty array.
    if retVal.size == 0:
        retVal = np.array([0.0])

    return retVal

def DIFFERENCES(X, Y, P):
    # "Computational derivative of the 1D vector of X"?
    # Since the function is called, "DIFFERENCES", I'm going with the
    # assumption that they want the difference between each subsequent value.
    return callRealFunc(X, Y, P, [(np.ndarray, None, DIFFERENCES_matrix)])

def AVG_DIFFERENCES_matrix(X, Y, P):
    return np.mean(np.diff(X.flatten()))

def AVG_DIFFERENCES(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, AVG_DIFFERENCES_matrix)])

def ROTATE_matrix(X, Y, P):
    return np.roll(X, round(P))

def ROTATE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ROTATE_matrix)])

def REVERSE_matrix(X, Y, P):
    return X.flatten()[::-1].reshape(X.shape)

def REVERSE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, REVERSE_matrix)])

def PUSH_BACK_XY(X, Y, P):
    # This works properly with any combination of array / scalar.
    return np.append(X, Y)

def PUSH_BACK_YX(X, Y, P):
    return np.append(Y, X)

def SET_XY(X, Y, P):
    retVal = np.empty(X.size)
    retVal.fill(Y)
    return retVal

def SET_YX(X, Y, P):
    retVal = np.empty(Y.size)
    retVal.fill(X)
    return retVal

def SET(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, numbers.Number, SET_XY),
                                  (numbers.Number, np.ndarray, SET_YX)])

def SUM_matrix(X, Y, P):
    return np.sum(X)

def SUM(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, SUM_matrix)])

def TRANSPOSE_matrix(X, Y, P):
    return X.transpose()

def TRANSPOSE(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, TRANSPOSE_matrix)])

def VECFROMDOUBLE_matrix(X, Y, P):
    retVal = np.empty(1)
    retVal.fill(X)
    return retVal

def VECFROMDOUBLE(X, Y, P):
    return callRealFunc(X, Y, P, [(numbers.Number, None, VECFROMDOUBLE_matrix)])

def YWIRE(X, Y, P):
    return Y

def NOP(X, Y, P):
    return X

def CONST(X, Y, P):
    return P

def CONSTVECTORD_matrix(X, Y, P):
    retVal = copy.copy(X)
    retVal.fill(P)
    return retVal

def CONSTVECTORD(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, CONSTVECTORD_matrix)])

def ZEROES_matrix(X, Y, P):
    retVal = copy.copy(X)
    retVal.fill(0.0)
    return retVal

def ZEROES(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ZEROES_matrix)])

def ONES_matrix(X, Y, P):
    retVal = copy.copy(X)
    retVal.fill(1.0)
    return retVal

def ONES(X, Y, P):
    return callRealFunc(X, Y, P, [(np.ndarray, None, ONES_matrix)])
