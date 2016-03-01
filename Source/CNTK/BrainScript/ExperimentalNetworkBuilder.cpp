#if 0 // this entire file can be removed once CNTK.core.bs works
// ExperimentalNetworkBuilder.cpp -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS   // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include <string>

using namespace std;

// TODO: move to actual text files to be included

wstring standardFunctions =
    L"Print(value, format='') = new PrintAction [ what = value /*; how = format*/ ] \n"
    L"Debug(value, say = '', enabled = true) = new Debug [ /*macro arg values*/ ] \n"
    L"Format(value, format) = new StringFunction [ what = 'Format' ; arg = value ; how = format ] \n"
    L"Replace(s, from, to) = new StringFunction [ what = 'Replace' ; arg = s ; replacewhat = from ; withwhat = to ] \n"
    L"Substr(s, begin, num) = new StringFunction [ what = 'Substr' ; arg = s ; pos = begin ; chars = num ] \n"
    L"Chr(c) = new StringFunction [ what = 'Chr' ;  arg = c ] \n"
    L"Floor(x)  = new NumericFunction [ what = 'Floor' ;  arg = x ] \n"
    L"Length(x) = new NumericFunction [ what = 'Length' ; arg = x ] \n"
    L"Ceil(x) = -Floor(-x) \n"
    L"Round(x) = Floor(x+0.5) \n"
    L"Sign(x) = if x > 0 then 1 else if x < 0 then -1 else 0 \n"
    L"Min(a,b) = if a < b then a else b \n"
    L"Max(a,b) = if a > b then a else b \n"
    L"Fac(n) = if n > 1 then Fac(n-1) * n else 1 \n";

wstring commonMacros =
    L"BFF(in, rows, cols) = [ B = Parameter(rows, 1, init = 'fixedValue', value = 0) ; W = Parameter(rows, cols) ; z = W*in+B ] \n"
    L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
    L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
    L"LogPrior(labels) = Log(Mean(labels)) \n";

wstring computationNodes = // TODO: use actual TypeName() here? would first need to make it a wide string; we should also extract those two methods into the base macro
L"LearnableParameter(rows, cols, learningRateMultiplier = 1.0, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', initOnCPUOnly=true, randomSeed=-1, tag='') = new ComputationNode [ operation = 'LearnableParameter' ; shape = new TensorShape [ dims = (rows : cols) ] /*plus the function args*/ ]\n"
    L"Parameter = LearnableParameter // deprecated \n"
L"ParameterTensor(dims, learningRateMultiplier = 1.0, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', initOnCPUOnly=true, randomSeed=-1, tag='') = new ComputationNode [ operation = 'LearnableParameter' ; shape = new TensorShape [ /*dims*/ ] /*plus the function args*/ ]\n"
    // TODO: ImageParameter?
    // ^^ already works; vv untested
    L"Input(dims, tag='feature') = new ComputationNode [ operation = 'InputValue' ; shape = new TensorShape [ /*dims*/ ] ; isImage = false /*plus the function args*/ ]\n" // note: naming a little inconsistent  // TODO: re-test after flag change
    L"SparseInput(dims, tag='feature') = new ComputationNode [ operation = 'SparseInputValue' ; shape = new TensorShape [ /*dims*/ ] ; isImage = false /*plus the function args*/ ]\n"
    L"ImageInput(imageWidth, imageHeight, imageChannels, imageLayout='CHW', tag='feature') = new ComputationNode [ operation = 'InputValue' ; isImage = true /*plus the function args*/ ]\n"
    L"SparseImageInput(imageWidth, imageHeight, imageChannels, imageLayout='CHW', tag='feature') = new ComputationNode [ operation = 'SparseInputValue' ; isImage = true /*plus the function args*/ ]\n"
    L"Constant(val, rows = 1, cols = 1, tag='') = Parameter(rows, cols, learningRateMultiplier = 0, init = 'fixedValue', value = val) \n"
    L"PastValue(dims, input, timeStep = 1, defaultHiddenActivation = 0.1, tag='') = new ComputationNode [ operation = 'PastValue' ; inputs = input ; shape = new TensorShape [ /*dims*/ ] /*plus the function args*/ ]\n"
    L"FutureValue(dims, input, timeStep = 1, defaultHiddenActivation = 0.1, tag='') = new ComputationNode [ operation = 'FutureValue' ; inputs = input ; shape = new TensorShape [ /*dims*/ ] /*plus the function args*/ ]\n"
    // TODO: ^^ DelayedValues no longer need to know their dimension. That is inferred in Validation.
    L"Shift(input, fromOffset, boundaryValue, boundaryMode=-1/*context*/, dim=-1, tag='') = new ComputationNode [ operation = 'Shift' ; inputs = (input : boundaryValue) /*plus the function args*/ ]\n"
    L"RowSlice(startIndex, numRows, input, tag='') = new ComputationNode [ operation = 'RowSlice' ; inputs = input /*plus the function args*/ ]\n"
    L"RowRepeat(input, numRepeats, tag='') = new ComputationNode [ operation = 'RowRepeat' ; inputs = input /*plus the function args*/ ]\n"
    L"RowStack(inputs, tag='') = new ComputationNode [ operation = 'RowStack' /*plus the function args*/ ]\n"
    L"Reshape(input, numRows, imageWidth = 0, imageHeight = 0, imageChannels = 0, tag='') = new ComputationNode [ operation = 'LegacyReshape' ; inputs = input /*plus the function args*/ ]\n"
    L"NewReshape(input, dims, beginDim=0, endDim=0, tag='') = new ComputationNode [ operation = 'Reshape' ; inputs = input ; shape = new TensorShape [ /*dims*/ ] /*plus the function args*/ ]\n"
    L"ReshapeDimension(x, dim, tensorShape) = NewReshape(x, tensorShape, beginDim=dim, endDim=dim + 1) \n"
    L"FlattenDimensions(x, dim, num) = NewReshape(x, 0, beginDim=dim, endDim=dim + num) \n"
    L"SplitDimension(x, dim, N) = ReshapeDimension(x, dim, 0:N) \n"
    L"TransposeDimensions(input, dim1, dim2, tag='') = new ComputationNode [ operation = 'TransposeDimensions' ; inputs = input /*plus the function args*/ ]\n"
    L"Transpose(x) = TransposeDimensions(x, 1, 2)\n"
    L"Times(A, B, outputRank=1, tag='') = new ComputationNode [ operation = 'Times' ; inputs = ( A : B ) /*plus the function args*/ ]\n"
    // TODO: Logistic should be generated with with BinaryStandardNode macro below.
    L"Logistic(label, probability, tag='') = new ComputationNode [ operation = 'Logistic' ; inputs = (label : probability) /*plus the function args*/ ]\n"
    L"WeightedLogistic(label, probability, instanceWeight, tag='') = new ComputationNode [ operation = 'Logistic' ; inputs = (label : probability : instanceWeight) /*plus the function args*/ ]\n"
    L"ReconcileMBLayout(dataInput, layoutInput, tag='') = new ComputationNode [ operation = 'ReconcileMBLayout' ; inputs = (dataInput : layoutInput) /*plus the function args*/ ]\n"
    L"Convolution(weightNode, inputValueNode, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'Convolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]\n"
    L"NDConvolution(weightNode, inputValueNode, kernelDims, mapDims, stride=1, sharing = true, autoPadding = true, lowerPad = 0, upperPad = 0, imageLayout='CHW', maxTempMemSizeInSamples = 0, tag='') = new ComputationNode [ operation = 'NDConvolution' ; inputs = (weightNode : inputValueNode); kernelShape = new TensorShape [ dims = kernelDims ] ; mapShape = new TensorShape [ dims = mapDims ] ; strideShape = new TensorShape [ dims = stride ] ; dimSharing = new BoolVector [ items = sharing ] ; dimPadding = new BoolVector [ items = autoPadding ] ; dimPadLower = new TensorShape [ dims = lowerPad ] ; dimPadUpper = new TensorShape [ dims = upperPad ] /*plus the function args*/ ]\n"
    //L"NDConvolution(weightNode, inputValueNode, kernelShape = new TensorShape [ /*dims*/ ], mapShape = new TensorShape[ /*dims*/ ], stride = new TensorShape [ /*dims*/ ], sharing = new BoolVector [ /*dims*/ ], padding = new BoolVector [ /*dims*/ ], lowerPad = new TensorShape [ /*dims*/ ], upperPad = new TensorShape [ /*dims*/ ], imageLayout='CHW', maxTempMemSizeInSamples = 0, tag='') = new ComputationNode [ operation = 'NDConvolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]\n"
    //L"NDConvolution(weightNode, inputValueNode, kernelShape, mapShape, stride, sharing, padding, lowerPad, upperPad, imageLayout='CHW', maxTempMemSizeInSamples = 0, tag='') = new ComputationNode [ operation = 'NDConvolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]\n"
    L"MaxPooling(input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'MaxPooling' ; inputs = input /*plus the function args*/ ]\n"
    L"AveragePooling(input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'AveragePooling' ; inputs = input /*plus the function args*/ ]\n"
    // TODO: define DelayedValue, with negative delay for future; cannot do this yet, need to be able to say something like delay = -(^.delay)
    // aliases
    L"ColumnwiseCrossProduct = KhatriRaoProduct // deprecated \n" // TODO: should it be deprecated? It is described as easier to understand in the CNTKBook.
    L"ClassificationError = ErrorPrediction \n"
    L"Delay = PastValue \n" // TODO: should it allow negative offsets and an if test here?
    L"BatchNormalization(input, scale, bias, runMean, runInvStdDev, eval, spatial, normalizationTimeConstant = 0, epsilon = 0.00001, useCntkEngine = true, imageLayout='CHW', tag='') = new ComputationNode [ operation = 'BatchNormalization' ; inputs = (input : scale : bias : runMean : runInvStdDev) /*plus the function args*/ ]\n"
// standard nodes. We use macros to define these strings.
#define UnaryStandardNode(Op, a) L## #Op L"(" L## #a L", tag='') = new ComputationNode [ operation = '" L## #Op L"' ; inputs = " L## #a L" /*plus the function args*/ ]\n"
#define BinaryStandardNode(Op, a, b) L## #Op L"(" L## #a L", " L## #b L", tag='') = new ComputationNode [ operation = '" L## #Op L"' ; inputs = (" L## #a L" : " L## #b L") /*plus the function args*/ ]\n"
#define TernaryStandardNode(Op, a, b, c) L## #Op L"(" L## #a L", " L## #b L", " L## #c L", tag='') = new ComputationNode [ operation = '" L## #Op L"' ; inputs = (" L## #a L" : " L## #b L" : " L## #c L") /*plus the function args*/ ]\n"
#define QuaternaryStandardNode(Op, a, b, c, d) L## #Op L"(" L## #a L", " L## #b L", " L## #c L", " L## #d L", tag='') = new ComputationNode [ operation = '" L## #Op L"' ; inputs = (" L## #a L" : " L## #b L" : " L## #c L" : " L## #d L") /*plus the function args*/ ]\n"
#ifdef COMING_SOON
    TernaryStandardNode(CRF, labelVectorSequence, positionDependenScoreVectorSequence, transitionScores) // TODO: better names
#endif
    UnaryStandardNode(Abs, x)
    QuaternaryStandardNode(ClassBasedCrossEntropyWithSoftmax, labelClassDescriptorVectorSequence, mainInputInfo, mainWeight, classLogProbsBeforeSoftmax)
    // BUGBUG: the commented-out ones are not mentioned in the CNTK book, nor are their parameters documented in the source code
    BinaryStandardNode(ColumnElementTimes, aVectorSequence, anotherVectorSequence)
    BinaryStandardNode(CosDistance, aVectorSequence, anotherVectorSequence)
    QuaternaryStandardNode(CosDistanceWithNegativeSamples, aVectorSequence, anotherVectorSequence, numShifts, numNegSamples)
    //BinaryStandardNode(CosDistanceWithNegativeSamplesNode)
    UnaryStandardNode(Cosine, x)
    BinaryStandardNode(CrossEntropy, refProbVectorSequence, outProbVectorSequence)
    BinaryStandardNode(CrossEntropyWithSoftmax, labelVectorSequence, outProbVectorSequence)
    BinaryStandardNode(DiagTimes, diagonalMatrixAsColumnVector, matrix)
    UnaryStandardNode(Dropout, activationVectorSequence)
    //BinaryStandardNode(DummyCriterionNode)
    BinaryStandardNode(ElementTimes, aMatrix, anotherMatrix)
    BinaryStandardNode(ErrorPrediction, labelVectorSequence, outVectorSequence) // CNTKBook: ClassificationError?
    UnaryStandardNode(Exp, x)
    QuaternaryStandardNode(GMMLogLikelihood, unnormalizedPriorVector, meansAsRows, logStdDevAsRows, dataVectorSequence)
    UnaryStandardNode(InvStdDev, dataVectorSequence)
    BinaryStandardNode(KhatriRaoProduct, leftMatrix, rightMatrix)
    //BinaryStandardNode(LSTMNode)
    UnaryStandardNode(Log, x)
    UnaryStandardNode(LogSoftmax, z)
    //BinaryStandardNode(LookupTableNode)
    UnaryStandardNode(MatrixL1Reg, matrix)
    UnaryStandardNode(MatrixL2Reg, matrix)
    // BUGBUG: CNTKBook also mentions L1Norm and L2Norm
    UnaryStandardNode(Mean, dataVectorSequence)
    BinaryStandardNode(Minus, leftMatrix, rightMatrix)
    UnaryStandardNode(Negate, input)
    TernaryStandardNode(PerDimMeanVarDeNormalization, dataVectorSequence, meanVector, invStdDevVector) // TODO: correct?
    TernaryStandardNode(PerDimMeanVarNormalization, dataVectorSequence, meanVector, invStdDevVector)
    BinaryStandardNode(Plus, leftMatrix, rightMatrix)
    UnaryStandardNode(RectifiedLinear, z)
    //BinaryStandardNode(RowElementTimesNode)
    BinaryStandardNode(Scale, scalarScalingFactor, matrix)
#ifdef COMING_SOON
    //BinaryStandardNode(SequenceDecoderNode)
#endif
    UnaryStandardNode(Sigmoid, z)
    UnaryStandardNode(Softmax, z)
    UnaryStandardNode(Hardmax, z)
    BinaryStandardNode(SquareError, aMatrix, anotherMatrix)
    UnaryStandardNode(SumColumnElements, z)
    UnaryStandardNode(SumElements, matrix)
    UnaryStandardNode(Tanh, z)
    UnaryStandardNode(TimeReverse, vectorSequence)
    BinaryStandardNode(TransposeTimes, leftMatrix, rightMatrix)
    // those nodes are deprecated, we won't implement them in BS:
    //BinaryStandardNode(NoiseContrastiveEstimationNode)
    //BinaryStandardNode(ParallelNode)
    //BinaryStandardNode(StrideTimesNode)
    ;
#endif
