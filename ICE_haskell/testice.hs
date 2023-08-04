import ICE

-- Test data and computations for illustration

-- testdata :: Dataset
-- testdata = [([4.1,2.6],-1),([-3.2,4.7],1),([1.1,-7.6],1),([0.3,0.4],-1)]

testdata :: Dataset
testdata = [
 ([0.5904,0.2234,0.4528], 1),
 ([0.6653,0.0357,0.9646], 1),
 ([0.8118,0.5466,0.5848],-1),
 ([0.2354,0.3451,0.2995],-1),
 ([0.0145,0.0248,0.6320],-1),
 ([0.0231,0.1222,0.5583],-1),
 ([0.6966,0.0101,0.5258], 1),
 ([0.3553,0.2129,0.6028], 1),
 ([0.7534,0.6306,0.0776],-1),
 ([0.6318,0.5370,0.5876], 1),
 ([0.5016,0.8751,0.8357],-1),
 ([0.4406,0.0367,0.1077],-1),
 ([0.2083,0.8233,0.3827],-1),
 ([0.8459,0.5828,0.9089], 1),
 ([0.9042,0.4960,0.4492], 1),
 ([0.7942,0.6902,0.1646], 1),
 ([0.2427,0.6052,0.0123], 1),
 ([0.4027,0.4970,0.0628],-1),
 ([0.5278,0.5112,0.8225],-1),
 ([0.6168,0.0350,0.0064], 1)]


-- check data items
testpoints = map point testdata

-- check labels
testlabels = map label testdata

approxub = 5

-- generate all configurations
testgen = (e01gen (1) approxub testdata) ++ (e01gen (-1) approxub testdata)

-- select the optimal configuration
testopt = e01class approxub testdata

maybef f = maybe Nothing (\x -> Just (f x))

-- check configuration's model only
testgenmodel = map model testgen

-- check the 0-1 loss of each configuration
testgenloss = map (maybef modell) testgenmodel

-- check the normal vector of each configuration
testgenw = map (maybef modelw) testgenmodel

-- check the combination of each configuration
testgencomb = map comb testgen

-- check the sequence of each configuration
testgenseq = map seqn testgen

-- check the combination of the optimal configuration
optcomb  = comb testopt

-- check the model of the optimal configuration
optmodel = model testopt

-- check the 0-l loss of the optimal configuration
optloss = maybef modell optmodel

-- Predict labels from optimal model ...
optw = (\(Just (w,l)) -> w) optmodel
optv = evalw testpoints optw
optlabels = plabel optv
