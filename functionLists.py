import CGPFunctions as funcs

funcList = [funcs.And, funcs.Or, funcs.Nand, funcs.Xor]
funcList2 = [funcs.And, funcs.AndNotY, funcs.Or, funcs.Xor]
funcList3 = [funcs.And, funcs.Nand, funcs.AndNotY, funcs.Or, funcs.Xor]
funcListANN = [funcs.ANN_Sigmoid, funcs.ANN_Tanh, funcs.ANN_Relu]
funcListANN_singleTan = [funcs.ANN_Tanh]
funcListANN_singleSigmoid = [funcs.ANN_Sigmoid]

brainBasicList1 = [funcs.ADD, funcs.AMINUS, funcs.CMINUS, funcs.MULT,
                   funcs.CMULT, funcs.ABS, funcs.SQRT, funcs.LT, funcs.GT,
                   funcs.GTEP, funcs.LTEP, funcs.MIN2, funcs.MAX2]

# brainListInput1 =

vilFuncList = [funcs.ADD, funcs.AMINUS, funcs.MULT, funcs.CMULT, funcs.INV,
               funcs.ABS, funcs.SQRT, funcs.CPOW, funcs.YPOW, funcs.EXPX,
               funcs.SQRTXY, funcs.LT, funcs.GT, funcs.YWIRE, funcs.CONST,
               funcs.NOP, funcs.ROUND, funcs.MIN2, funcs.MAX2]

gymFuncList = [funcs.ADD, funcs.MULT_ATARI, funcs.AMINUS, funcs.CMULT,
               funcs.INV, funcs.ABS, funcs.SQRT, funcs.CPOW, funcs.YPOW,
               funcs.EXPX, funcs.SQRTXY, funcs.GT, funcs.LT, funcs.MAX2,
               funcs.MIN2, funcs.ASIN, funcs.SINX, funcs.ACOS, funcs.COS,
               funcs.ATAN, funcs.TAN]

testMountainCarFunctionList = [funcs.ADD, funcs.AMINUS, funcs.MULT_ATARI,
                               funcs.CMULT, funcs.INV, funcs.ABS, funcs.SQRT,
                               funcs.CPOW, funcs.GT, funcs.LT, funcs.MAX2,
                               funcs.MIN2]

cartPoleFunctionList = [funcs.ADD, funcs.MULT_ATARI, funcs.AMINUS, funcs.CMULT,
                        funcs.INV, funcs.ABS, funcs.CPOW, funcs.YPOW,
                        funcs.EXPX, funcs.SQRTXY, funcs.GT, funcs.LT,
                        funcs.MAX2, funcs.MIN2, funcs.ASIN, funcs.SINX,
                        funcs.ACOS, funcs.COS, funcs.ATAN, funcs.TAN]

pendulumFuncList = [funcs.ADD, funcs.MULT_ATARI, funcs.AMINUS, funcs.CMULT,
                    funcs.INV, funcs.ABS, funcs.SQRT, funcs.CPOW, funcs.YPOW,
                    funcs.EXPX, funcs.SQRTXY, funcs.GT, funcs.LT, funcs.MAX2,
                    funcs.MIN2, funcs.ASIN, funcs.ACOS, funcs.ATAN]

lunarLanderFuncList = [funcs.ADD, funcs.MULT_ATARI, funcs.AMINUS, funcs.CMULT,
                       funcs.INV, funcs.ABS, funcs.SQRT, funcs.CPOW,
                       funcs.YPOW, funcs.EXPX, funcs.SQRTXY, funcs.GT,
                       funcs.LT, funcs.MAX2, funcs.MIN2]

atariFuncList = [funcs.ADD_ATARI, funcs.AMINUS_ATARI, funcs.MULT_ATARI,
                 funcs.CMULT, funcs.INV, funcs.ABS, funcs.SQRT, funcs.CPOW,
                 funcs.YPOW, funcs.EXPX, funcs.SINX, funcs.SQRTXY,
                 funcs.ACOS_atari, funcs.ASIN_atari, funcs.ATAN_atari,
                 funcs.STDDEV, funcs.SKEW, funcs.KURTOSIS, funcs.MEAN,
                 funcs.RANGE_atari, funcs.ROUND, funcs.CEIL, funcs.FLOOR,
                 funcs.MAX1, funcs.MIN1, funcs.LT, funcs.GT, funcs.MAX2,
                 funcs.MIN2, funcs.SPLIT_BEFORE_X, funcs.SPLIT_AFTER_X,
                 funcs.SPLIT_BEFORE_Y, funcs.SPLIT_AFTER_Y,
                 funcs.SPLIT_BEFORE_SQUARE, funcs.SPLIT_AFTER_SQUARE,
                 funcs.RANGE_INX, funcs.RANGE_INY, funcs.FIRST,
                 funcs.LAST, funcs.DIFFERENCES, funcs.AVG_DIFFERENCES,
                 funcs.ROTATE, funcs.REVERSE, funcs.PUSH_BACK_XY,
                 funcs.PUSH_BACK_YX, funcs.SET, funcs.SUM,
                 funcs.TRANSPOSE, funcs.VECFROMDOUBLE, funcs.YWIRE, funcs.NOP,
                 funcs.CONST, funcs.CONSTVECTORD, funcs.ZEROES, funcs.ONES]

juliaGymFuncList = []
#[
 # f_split_before:
 #    - "x"
 #    - "if length(x) > 1; return x[1:f2ind(x, (c+1)/2.0)]; else; return x; end"
 #  f_split_after:
 #    - "x"
 #    - "if length(x) > 1; return x[f2ind(x, (c+1)/2.0):end]; else; return x; end"
 #  f_range_in:
 #    - "x"
 #    - "range_in(x, (y+1)/2.0, (c+1)/2.0)"
 #  f_index_y:
 #    - "x"
 #    - "index_in(x, (y+1)/2.0)"
 #  f_index_c:
 #    - "x"
 #    - "index_in(x, (c+1)/2.0)"
 #  f_vectorize:
 #    - "x"
 #    - "x[:]"
 #  f_first:
 #    - "x"
 #    - "x[1]"
 #  f_last:
 #    - "x"
 #    - "x[end]"
 #  f_differences:
 #    - "x"
 #    - "if length(x) > 1; return scaled(diff(x[:])); else; return 0.0; end"
 #  f_avgdifferences:
 #    - "x"
 #    - "if length(x) > 1; return scaled(mean(diff(x[:]))); else; return 0.0; end"
 #  f_rotate:
 #    - "x"
 #    - "circshift(x, ceil(c))"
 #  f_reverse:
 #    - "x"
 #    - "reverse(x[:])"
 #  f_pushback:
 #    - "[x; y]"
 #    - "[x; y[:]]"
 #    - "[x[:]; y]"
 #    - "[x[:]; y[:]]"
 #  f_pushfront:
 #    - "[y; x]"
 #    - "[y[:]; x]"
 #    - "[y; x[:]]"
 #    - "[y[:]; x[:]]"
 #  f_set:
 #    - "x"
 #    - "x*ones(size(y))"
 #    - "y*ones(size(x))"
 #    - "mean(x)*ones(size(y))"
 #  f_sum:
 #    - "x"
 #    - "scaled(sum(x))"
 #  f_transpose:
 #    - "x"
 #    - "if ndims(x) < 3; return Array{Float64}(ctranspose(x)); else; return x; end"
 #  # mathematical
 #  f_add:
 #    - "(x+y)/2.0"
 #    - "(x.+y)/2.0"
 #    - "(x.+y)/2.0"
 #    - ".+(eqsize(x,y,c)...)/2.0"
 #  f_aminus:
 #    - "abs(x-y)/2.0"
 #    - "abs.(x.-y)/2.0"
 #    - "abs.(x.-y)/2.0"
 #    - "abs.(.-(eqsize(x,y,c)...))/2.0"
 #  f_mult:
 #    - "x*y"
 #    - "x.*y"
 #    - "x.*y"
 #    - ".*(eqsize(x,y,c)...)"
 #  f_cmult:
 #    - "x.*c"
 #  f_inv:
 #    - "scaled(1./x)"
 #  f_abs:
 #    - "abs.(x)"
 #  f_sqrt:
 #    - "sqrt.(abs.(x))"
 #  f_cpow:
 #    - "abs.(x).^(c+1.0)"
 #  f_ypow:
 #    - "abs(x)^abs(y)"
 #    - "abs.(x).^abs.(y)"
 #    - "abs.(x).^abs.(y)"
 #    - ".^(eqsize(abs.(x),abs.(y),c)...)"
 #  f_expx:
 #    - "(exp.(x)-1.0)/(exp(1.0)-1.0)"
 #  f_sinx:
 #    - "sin.(x)"
 #  f_cosx:  [NOT LISTED IN PAPER]
 #    - "cos.(x)"
 #  f_sqrtxy:
 #    - "sqrt.(x*x+y*y)/sqrt(2.0)"
 #    - "sqrt.(x*x+y.*y)/sqrt(2.0)"
 #    - "sqrt.(x.*x+y*y)/sqrt(2.0)"
 #    - "sqrt.(.+(eqsize(x.*x, y.*y, c)...))/sqrt(2.0)"
 #  f_acos:
 #    - "acos.(x)/pi"
 #  f_asin:
 #    - "2*asin.(x)/pi"
 #  f_atan:
 #    - "4*atan.(x)/pi"
 #  # Comparison
 #  f_lt:
 #    - "Float64(x < y)"
 #    - "Float64.(x.<y)"
 #    - "Float64.(x.<y)"
 #    - "Float64.(.<(eqsize(x,y,c)...))"
 #  f_gt:
 #    - "Float64(x > y)"
 #    - "Float64.(x.>y)"
 #    - "Float64.(x.>y)"
 #    - "Float64.(.>(eqsize(x,y,c)...))"
 #  # Statistical
 #  f_stddev:
 #    - "0.0"
 #    - "scaled(std(x[:]))"
 #  f_skew:
 #    - "x"
 #    - "scaled(skewness(x[:]))"
 #  f_kurtosis:
 #    - "x"
 #    - "scaled(kurtosis(x[:]))"
 #  f_mean:
 #    - "x"
 #    - "mean(x)"
 #  f_range:
 #    - "x"
 #    - "maximum(x)-minimum(x)-1.0"
 #  f_round:
 #    - "round.(x)"
 #  f_ceil:
 #    - "ceil.(x)"
 #  f_floor:
 #    - "floor.(x)"
 #  f_max1:
 #    - "x"
 #    - "maximum(x)"
 #  f_max2:
 #    - "max(x,y)"
 #    - "max.(x,y)"
 #    - "max.(x,y)"
 #    - "max.(eqsize(x, y, c)...)"
 #  f_min1:
 #    - "x"
 #    - "minimum(x)"
 #  f_min2:
 #    - "min(x,y)"
 #    - "min.(x,y)"
 #    - "min.(x,y)"
 #    - "min.(eqsize(x, y, c)...)"
 #  # Misc
 #  f_vecfromdouble:
 #    - "[x]"
 #    - "x"
 #  f_ywire:
 #    - "y"
 #  f_nop:
 #    - "x"
 #  f_const:
 #    - "c"
 #  f_constvectord:
 #    - "c"
 #    - "c.*ones(size(x))"
 #  f_zeros:
 #    - "0.0"
 #    - "zeros(size(x))"
 #  f_ones:
 #    - "1.0"
 #    - "ones(size(x))"
#]
