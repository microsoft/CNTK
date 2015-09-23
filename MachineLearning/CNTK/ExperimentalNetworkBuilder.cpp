// ExperimentalNetworkBuilder.cpp -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ExperimentalNetworkBuilder.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#include <string>

#ifndef let
#define let const auto
#endif


namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace Microsoft::MSR;

    wstring standardFunctions =
        L"Print(value, format='') = new PrintAction [ what = value /*; how = format*/ ] \n"
        L"Format(value, format) = new StringFunction [ what = 'Format' ; arg = value ; how = format ] \n"
        L"Replace(s, from, to) = new StringFunction [ what = 'Replace' ; arg = s ; replacewhat = from ; withwhat = to ] \n"
        L"Substr(s, begin, num) = new StringFunction [ what = 'Substr' ; arg = s ; pos = begin ; chars = num ] \n"
        L"Chr(c) = new StringFunction [ what = 'Chr' ;  arg = c ] \n"
        L"Floor(x)  = new NumericFunction [ what = 'Floor' ;  arg = x ] \n"
        L"Length(x) = new NumericFunction [ what = 'Length' ; arg = x ] \n"
        L"Ceil(x) = -Floor(-x) \n"
        L"Round(x) = Floor(x+0.5) \n"
        L"Abs(x) = if x >= 0 then x else -x \n"
        L"Sign(x) = if x > 0 then 1 else if x < 0 then -1 else 0 \n"
        L"Min(a,b) = if a < b then a else b \n"
        L"Max(a,b) = if a > b then a else b \n"
        L"Fac(n) = if n > 1 then Fac(n-1) * n else 1 \n"
        ;

    wstring commonMacros =
        L"BFF(in, rows, cols) = [ B = Parameter(rows, 1, init = 'fixedValue', value = 0) ; W = Parameter(rows, cols) ; z = W*in+B ] \n"
        L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
        L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
        L"LogPrior(labels) = Log(Mean(labels)) \n"
        ;

    // TODO: must be moved to ComputationNodeBase.h
    // a ComputationNode that derives from MustFinalizeInit does not resolve some args immediately (just keeps ConfigValuePtrs),
    // assuming they are not ready during construction.
    // This is specifically meant to be used by DelayNode, see comments there.
    struct MustFinalizeInit { virtual void FinalizeInit() = 0; };   // derive from this to indicate ComputationNetwork should call FinalizeIitlate initialization

    wstring computationNodes =  // TODO: use actual TypeName() here? would first need to make it a wide string; we should also extract those two methods into the base macro
        L"LearnableParameter(rows, cols, needGradient = true, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', initOnCPUOnly=true, randomSeed=-1, tag='') = new ComputationNode [ operation = 'LearnableParameter' /*plus the function args*/ ]\n"
        L"Parameter = LearnableParameter // deprecated \n"
        // ^^ already works; vv untested
        L"Input(rows, cols, tag='feature') = new ComputationNode [ operation = 'InputValue' ; isSparse = false ; isImage = false /*plus the function args*/ ]\n" // note: naming a little inconsistent  // TODO: re-test after flag change
        L"SparseInput(rows, cols, tag='feature') = new ComputationNode [ operation = 'InputValue' ; isSparse = true ; isImage = false /*plus the function args*/ ]\n"
        L"ImageInput(imageWidth, imageHeight, imageChannels, numImages, tag='feature') = new ComputationNode [ operation = 'InputValue' ; isSparse = true ; isImage = true /*plus the function args*/ ]\n"
        L"SparseImageInput(imageWidth, imageHeight, imageChannels, numImages, tag='feature') = new ComputationNode [ operation = 'InputValue' ; isSparse = true ; isImage = true /*plus the function args*/ ]\n"
        L"Constant(value, rows = 1, cols = 1, tag='') = Parameter(rows, cols, needGradient = false, init = 'fixedValue') \n"
        L"PastValue(rows, cols, input, timeStep = 1, defaultHiddenActivation = 0.1, tag='') = new ComputationNode [ operation = 'PastValue' ; inputs = input /*plus the function args*/ ]\n"
        L"FutureValue(rows, cols, input, timeStep = 1, defaultHiddenActivation = 0.1, tag='') = new ComputationNode [ operation = 'FutureValue' ; inputs = input /*plus the function args*/ ]\n"
        L"RowSlice(startIndex, numRows, input, needGradient = false, tag='') = new ComputationNode [ operation = 'RowSlice' ; inputs = input /*plus the function args*/ ]\n"
        L"RowRepeat(input, numRepeats, needGradient = false, tag='') = new ComputationNode [ operation = 'RowRepeat' ; inputs = input /*plus the function args*/ ]\n"
        L"RowStack(inputs, tag='') = new ComputationNode [ operation = 'RowStack' /*plus the function args*/ ]\n"
        L"Reshape(input, numRows, imageWidth = 0, imageHeight = 0, imageChannels = 0, needGradient = false, tag='') = new ComputationNode [ operation = 'Reshape' ; inputs = input /*plus the function args*/ ]\n"
        L"ConvolutionNode(weightNode, inputValueNode, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0, tag='') = new ComputationNode [ operation = 'Convolution' ; inputs = (weightNode : inputValueNode) /*plus the function args*/ ]\n"
        L"MaxPoolingNode(input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, tag='') = new ComputationNode [ operation = 'MaxPooling' ; inputs = input /*plus the function args*/ ]\n"
        L"AveragePoolingNode(input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, tag='') = new ComputationNode [ operation = 'AveragePoolingNode' ; inputs = input /*plus the function args*/ ]\n"
        // TODO: define DelayedValue, with negative delay for future; cannot do this yet, need to be able to say something like delay = -(^.delay)
        // aliases
        L"ColumnwiseCrossProduct = KhatriRaoProduct // deprecated \n"   // TODO: should it be deprecated? It is described as easier to understand in the CNTKBook.
        L"ClassificationError = ErrorPrediction \n"
        L"Delay = PastValue \n" // TODO: should it allow negative offsets and an if test here?
        // standard nodes. We use macros to define these strings.
#define UnaryStandardNode(Op,a) L ## #Op L"(" L ## #a L", tag='') = new ComputationNode [ operation = '" L ## #Op  L"' ; inputs = " L ## #a L" /*plus the function args*/ ]\n"
#define BinaryStandardNode(Op,a,b) L ## #Op L"(" L ## #a L", " L ## #b L", tag='') = new ComputationNode [ operation = '" L ## #Op  L"' ; inputs = (" L ## #a L" : " L ## #b L") /*plus the function args*/ ]\n"
#define TernaryStandardNode(Op,a,b,c) L ## #Op L"(" L ## #a L", " L ## #b L", " L ## #c L", tag='') = new ComputationNode [ operation = '" L ## #Op  L"' ; inputs = (" L ## #a L" : " L ## #b L" : " L ## #c L") /*plus the function args*/ ]\n"
#define QuaternaryStandardNode(Op,a,b,c,d) L ## #Op L"(" L ## #a L", " L ## #b L", " L ## #c L", " L ## #d L", tag='') = new ComputationNode [ operation = '" L ## #Op  L"' ; inputs = (" L ## #a L" : " L ## #b L" : " L ## #c L" : " L ## #d L") /*plus the function args*/ ]\n"
        TernaryStandardNode(CRF, labelVectorSequence, positionDependenScoreVectorSequence, transitionScores)    // TODO: better names
        QuaternaryStandardNode(ClassBasedCrossEntropyWithSoftmax, labelClassDescriptorVectorSequence, mainInputInfo, mainWeight, classLogProbsBeforeSoftmax)
        // BUGBUG: the commented-out ones are not mentioned in the CNTK book, nor are their parameters documented in the source code
        //BinaryStandardNode(ColumnElementTimesNode)
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
        //BinaryStandardNode(NoiseContrastiveEstimationNode)
        //BinaryStandardNode(PairNetworkNode)
        //BinaryStandardNode(ParallelNode)
        TernaryStandardNode(PerDimMeanVarDeNormalization, dataVectorSequence, meanVector, invStdDevVector)   // TODO: correct?
        TernaryStandardNode(PerDimMeanVarNormalization, dataVectorSequence, meanVector, invStdDevVector)
        BinaryStandardNode(Plus, leftMatrix, rightMatrix)
        UnaryStandardNode(RectifiedLinear, z)
        //BinaryStandardNode(RowElementTimesNode)
        BinaryStandardNode(Scale, scalarScalingFactor, matrix)
        //BinaryStandardNode(SequenceDecoderNode)
        UnaryStandardNode(Sigmoid, z)
        UnaryStandardNode(Softmax, z)
        BinaryStandardNode(SquareError, aMatrix, anotherMatrix)
        //BinaryStandardNode(StrideTimesNode)
        //BinaryStandardNode(SumColumnElementsNode)
        UnaryStandardNode(SumElements, matrix)
        UnaryStandardNode(Tanh, z)
        UnaryStandardNode(TimeReverse, vectorSequence)
        BinaryStandardNode(Times, leftMatrix, rightMatrix)
        UnaryStandardNode(Transpose, matrix)
        //BinaryStandardNode(TransposeTimesNode)
    ;

    // helper that returns 'float' or 'double' depending on ElemType
    template<class ElemType> static const wchar_t * ElemTypeName();
    template<> /*static*/ const wchar_t * ElemTypeName<float>()  { return L"float"; }
    template<> /*static*/ const wchar_t * ElemTypeName<double>() { return L"double"; }

    // build a ComputationNetwork from BrainScript source code
    template<class ElemType>
    /*virtual*/ /*IComputationNetBuilder::*/ComputationNetwork* ExperimentalNetworkBuilder<ElemType>::BuildNetworkFromDescription(ComputationNetwork*)
    {
        if (!m_net || m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            // We interface with outer old CNTK config by taking the inner part, which we get as a string, as BrainScript.
            // We prepend a few standard definitions, and also definition of deviceId and precision, which all objects will pull out again when they are being constructed.
            // BUGBUG: We are not getting TextLocations right in this way! Do we need to inject location markers into the source?
            let expr = BS::ParseConfigString(BS::standardFunctions + BS::computationNodes + BS::commonMacros
                + msra::strfun::wstrprintf(L"deviceId = %d ; precision = '%s' ; network = new ComputationNetwork ", (int)m_deviceId, ElemTypeName<ElemType>())  // TODO: check if typeid needs postprocessing
                + m_sourceCode);    // source code has the form [ ... ]
            // evaluate the parse tree--specifically the top-level field 'network'--which will create the network
            let object = EvaluateField(expr, L"network");                               // this comes back as a BS::Object
            let network = dynamic_pointer_cast<ComputationNetwork>(object);   // cast it
            // This should not really fail since we constructed the source code above such that this is the right type.
            // However, it is possible (though currently not meaningful) to locally declare a different 'precision' value.
            // In that case, the network might come back with a different element type. We need a runtime check for that.
            if (!network)
                RuntimeError("BuildNetworkFromDescription: network has the wrong element type (float vs. double)");
            // success
            m_net = network;
            // TODO: old CNTK code seems to be able to load the network in-place--is that important; is it OK to just replace the pointer?
        }
        m_net->ResetEvalTimeStamp();
        return m_net.get();
    }

    template class ExperimentalNetworkBuilder<float>;
    template class ExperimentalNetworkBuilder<double>;

}}}
