// ExperimentalNetworkBuilder.cpp -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ExperimentalNetworkBuilder.h"
#include "BrainScriptEvaluator.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "RecurrentNodes.h"
#include "NonlinearityNodes.h"
#include "LinearAlgebraNodes.h"
#include "ConvolutionalNodes.h"

#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"

#include <memory>
#include <deque>
#include <set>
#include <string>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace BS {

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
        L"Reshape(input, numRows, imageWidth = 0, imageHeight = 0, imageChannels = 0, tag='') = new ComputationNode [ operation = 'Reshape' ; inputs = input /*plus the function args*/ ]\n"
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
        //BinaryStandardNode(RowStackNode)
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

    // The following class(es) implement the MakeRuntimeObject() function for different types. Sorry for the strange template dance.

    // -------------------------------------------------------------------
    // basic function template, for classes that can instantiate themselves from IConfigRecordPtr  TODO: do we even have any?
    // -------------------------------------------------------------------

    template<typename ElemType, class C>
    struct DualPrecisionHelpers
    {
        static shared_ptr<Object> MakeRuntimeObject(const IConfigRecordPtr config) { return make_shared<C>(config); }
    };

    // -------------------------------------------------------------------
    // ComputationNode -- covers all standard nodes
    // -------------------------------------------------------------------

    // helper wrapper class for ComputationNodes that must AttachInputs() late due to circular references
    // Instantiate with LateAttachingNode<node type>(lambda, args for node constructor).
    // To resolve, call AttachInputs()
    // TODO: This is a bit indirect. Can it be done more nicely?
    struct ILateAttachingNode { virtual void LateAttachInputs() = 0; };
    template<class N>
    class LateAttachingNode : public N, public ILateAttachingNode
    {
        typedef typename N::OurElemType ElemType;
        function<void(ComputationNode<ElemType>*)> attachInputs;
    public:
        // constructor
        template<class... _Types>
        LateAttachingNode(DEVICEID_TYPE deviceId, const wstring & name, const function<void(ComputationNode<ElemType>*)> & attachInputs, _Types&&... _Args) : attachInputs(attachInputs), N(deviceId, name, forward<_Types>(_Args)...) {}
        // the one member that does the work
        void /*ILateAttachingNode::*/LateAttachInputs()
        {
            attachInputs(dynamic_cast<N*>(this));
            attachInputs = [](ComputationNode<ElemType>*){ LogicError("LateAttachingNode::AttachInputs: must only be called once"); };
        }
    };

    template<typename ElemType>
    struct DualPrecisionHelpers<ElemType, ComputationNode<ElemType>>
    {
        // create ComputationNode
        // This is the equivalent of the old SynchronousNodeEvaluator::Evaluate(), and we duplicate code from there.
        static shared_ptr<Object> MakeRuntimeObject(const IConfigRecordPtr configp)
        {
            let & config = *configp;
            wstring operationName = config[L"operation"];
            wstring nodeName = L"<placeholder>";   // name will be overwritten by caller upon return (TODO: fix this here? pass expression name in?)
            DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
            static unsigned long m_randomSeedOffset = 0;    // TODO: this is held in the ComputationNetwork, but we don't have one yet
            // TODO" ^^ actually it seems only used by initialization of LearnableParameters--check that again; in that case, we can have a local

            // note on optional parameters
            // Instead of defining optional parameters here in code, they are defined as optional args to the creating macro.

            ComputationNodeBasePtr node;

#define OpIs(op) (operationName == msra::strfun::utf16(op<ElemType>::TypeName()))

            // TODO: in the code below, for reference, each block is preceded by an #if-0'ed out copy of the respective code from SynchronousNodeEvaluator::Evaluate()--remove these when this all works

            // first group: nodes without inputs
#if 0
            if (InputValue<ElemType>::TypeName() == cnNodeType)
            {
                if (parameter.size() < 1 || parameter.size() > 2)
                    RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                    // first look for this node already existing in the network
                    if (m_net.NodeNameExist(name))
                        nodePtr = m_net.GetNodeFromName(name);
                    else
                        nodePtr = m_net.CreateInputNode(name, rows, cols);
                }
            }
            else if (InputValue<ElemType>::SparseTypeName() == cnNodeType)
            {
                if (parameter.size() < 1 || parameter.size() > 2)
                    RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                    // first look for this node already existing in the network
                    if (m_net.NodeNameExist(name))
                        nodePtr = m_net.GetNodeFromName(name);
                    else
                        nodePtr = m_net.CreateSparseInputNode(name, rows, cols);
                }
            }
            else if (cnNodeType == L"ImageInput")
            {
                if (parameter.size() < 3 || parameter.size() > 4)
                    RuntimeError("%ls should have 3 or 4 parameters[imageWidth, imageHeight, imageChannels, [numImages=1]].", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                    size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                    size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;

                    nodePtr = m_net.CreateInputNode(name, imageWidth, imageHeight, imageChannels, numImages);
                }
            }
            else if (cnNodeType == L"SparseImageInput")
            {
                if (parameter.size() < 3 || parameter.size() > 4)
                    RuntimeError("%ls should have 3 or 4 parameters[imageWidth, imageHeight, imageChannels, [numImages=1]].", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                    size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                    size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;

                    nodePtr = m_net.CreateSparseInputNode(name, imageWidth, imageHeight, imageChannels, numImages);
                }
            }
#endif
            if (OpIs(InputValue))
            {
                let isSparse = config(L"isSparse");
                let isImage = config(L"isImage");
                if (!isImage)
                    node = New<InputValue<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"], isSparse);
                else
                    node = New<InputValue<ElemType>>(deviceId, nodeName, (size_t)config[L"imageWidth"], (size_t)config[L"imageHeight"], (size_t)config[L"imageChannels"], (size_t)config[L"numImages"], isSparse);
            }
#if 0
            else if (LearnableParameter<ElemType>::TypeName() == cnNodeType)
            {
                if (parameter.size() < 1 || parameter.size() > 2)
                    RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                    bool needGradient = node->GetOptionalParameter("needGradient", "true");

                    nodePtr = m_net.CreateLearnableParameter(name, rows, cols);

                    nodePtr->NeedGradient() = needGradient;
                }
                else if (pass == ndlPassFinal)
                {
                    static int randomSeed = 1;
                    std::string initString = node->GetOptionalParameter("init", "uniform");
                    ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                    ElemType value = node->GetOptionalParameter("value", "0");

                    msra::strfun::tolower_ascii(initString);
                    if (initString == "fixedvalue")
                        nodePtr->FunctionValues().SetValue(value);
                    else if (initString == "uniform")
                        m_net.InitLearnableParameters(nodePtr, true, randomSeed++, initValueScale);
                    else if (initString == "gaussian")
                        m_net.InitLearnableParameters(nodePtr, false, randomSeed++, initValueScale);
                    else if (initString == "fromfile")
                    {
                        std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                        if (initFromFilePath == "")
                            RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                        if (initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size() - 1] == '\"')
                            // remove the opening and closing double quotes
                            initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size() - 2);
                        if (!fexists(initFromFilePath))
                            RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                        m_net.InitLearnableParametersFromFile(nodePtr, initFromFilePath);
                    }
                    else
                        RuntimeError("init must be one of the values of [uniform|gaussian|fixedvalue]");
                }
            }
            else if (SparseLearnableParameter<ElemType>::TypeName() == cnNodeType)
            {
                if (parameter.size() < 1 || parameter.size() > 2)
                    RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                    bool needGradient = node->GetOptionalParameter("needGradient", "true");

                    nodePtr = m_net.CreateSparseLearnableParameter(name, rows, cols);

                    nodePtr->NeedGradient() = needGradient;
                }
                else if (pass == ndlPassFinal)
                {
                    static int randomSeed = 1;
                    std::string initString = node->GetOptionalParameter("init", "uniform");
                    ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                    ElemType value = node->GetOptionalParameter("value", "0");

                    msra::strfun::tolower_ascii(initString);
                    if (initString == "fixedvalue")
                        nodePtr->FunctionValues().SetValue(value);
                    else if (initString == "uniform")
                        m_net.InitLearnableParameters(nodePtr, true, randomSeed++, initValueScale);
                    else if (initString == "gaussian")
                        m_net.InitLearnableParameters(nodePtr, false, randomSeed++, initValueScale);
                    else if (initString == "fromfile")
                    {
                        std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                        if (initFromFilePath == "")
                            RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                        if (initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size() - 1] == '\"')
                            // remove the opening and closing double quotes
                            initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size() - 2);
                        if (!fexists(initFromFilePath))
                            RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                        m_net.InitLearnableParametersFromFile(nodePtr, initFromFilePath);
                    }
                    else
                        RuntimeError("init must be one of the values of [uniform|gaussian|fixedvalue]");
                }
            }
#endif
            else if (OpIs(LearnableParameter) || OpIs(SparseLearnableParameter))
            {
                // parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float])
                // TODO: do we need a default value mechanism? How to make sure it does not pop upwards? Current functions do not allow overloads.
                // TODO: test this with random init for QuickE2E on CPU against SimpleNetworkBuilder
                let isSparse = (operationName.find(L"Sparse") != wstring::npos);
                if (!isSparse)
                    node = New<LearnableParameter<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"]);
                else
                    node = New<SparseLearnableParameter<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"], 0/*size*/);    // TODO: what is size?
                node->NeedGradient() = config[L"needGradient"];
                static int randomSeed = 1;
                wstring initString = config[L"init"];
                if (initString == L"fixedValue")
                    dynamic_pointer_cast<LearnableParameter<ElemType>>(node)->FunctionValues().SetValue((ElemType)config[L"value"]);
                else if (initString == L"uniform" || initString == L"gaussian")
                {
                    // TODO: add these options also to old NDL
                    int forcedRandomSeed = config[L"randomSeed"];   // forcing a specific random seed is useful for testing to get repeatable initialization independent of evaluation order
                    dynamic_pointer_cast<LearnableParameter<ElemType>>(node)->InitRandom((initString == L"uniform"), forcedRandomSeed < 0 ? (randomSeed++ + m_randomSeedOffset) : (unsigned long)forcedRandomSeed, config[L"initValueScale"], config[L"initOnCPUOnly"]);
                }
                else if (initString == L"fromFile")
                {
                    wstring initFromFilePath = config[L"initFromFilePath"];
                    if (initFromFilePath.empty())
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    ComputationNetwork<ElemType>::InitLearnableParametersFromFile(dynamic_pointer_cast<LearnableParameter<ElemType>>(node), initFromFilePath, node->GetDeviceId());
                }
                else
                    RuntimeError("init must be one of the values of [uniform|gaussian|fixedValue|fromFile]");
            }
#if 0
            else if (cnNodeType == L"Constant")
            {
                if (parameter.size() != 1)
                    RuntimeError("Constant should have 1 fixed parameter [val] and two optional parameters [rows=[1|yourvalue], cols=[1|yourvalue]].");

                if (pass == ndlPassInitial)
                {
                    size_t rows = node->GetOptionalParameter("rows", "1");
                    size_t cols = node->GetOptionalParameter("cols", "1");

                    nodePtr = m_net.CreateLearnableParameter(name, rows, cols);
                    nodePtr->NeedGradient() = false;
                }
                else if (pass == ndlPassFinal || nodePtr->FunctionValues().GetNumElements() != 0)
                {
                    ElemType val = parameter[0]->GetScalar();
                    nodePtr->FunctionValues().SetValue(val);
                }
            }
#endif
            // Constant is implemented as a LearnableParameter with initializion as fixedValue with needGradient false, on script level
#if 0
            else if (cnNodeType == PastValueNode<ElemType>::TypeName() ||
                cnNodeType == FutureValueNode<ElemType>::TypeName())
            {
                if (parameter.size() <2 || parameter.size() >3)
                    RuntimeError("PastValue or FutureValue should have two to three fixed parameters. Usage: PastValue(rows, [cols], m, [timeStep=1, defaultPastValue=0.1]).");

                nodeParamCount = 1;
                nodeParamStart = parameter.size() > 2 ? 2 : 1;

                if (pass == ndlPassInitial)
                {
                    // evaluate only scalar parameters
                    vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                    size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                    // if we have three parameters the second is columns
                    size_t cols = parameter.size() > 2 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                    bool needGradient = node->GetOptionalParameter("needGradient", "false");
                    float defaultHiddenActivity = node->GetOptionalParameter("defaultHiddenActivity", "0.1");

                    //for backward compatibility we check timeStep first
                    size_t timeStep = node->GetOptionalParameter("timeStep", "1");
                    if (timeStep == 1)
                    {
                        timeStep = node->GetOptionalParameter("delayTime", "1");
                    }

                    if (cnNodeType == PastValueNode<ElemType>::TypeName())
                    {
                        nodePtr = m_net.PastValue(NULL, defaultHiddenActivity, rows, cols, name);
                        static_pointer_cast<PastValueNode<ElemType>>(nodePtr)->SetTimeStep(timeStep);
                    }
                    else
                    {
                        nodePtr = m_net.FutureValue(NULL, defaultHiddenActivity, rows, cols, name);
                        static_pointer_cast<FutureValueNode<ElemType>>(nodePtr)->SetTimeStep(timeStep);
                    }

                    nodePtr->NeedGradient() = needGradient; // TODO: What for?
                }
            }
#endif
            // nodes with delayed inputs, where we cannot yet resolve inputs due to circular references
            else if (OpIs(PastValueNode) || OpIs(FutureValueNode)) // TODO: untested
            {
                // rows, cols, input, [timeStep=1, defaultHiddenActivation=0.1]
                // Note: changed names of optional args compared to current NDL
                // TODO: we really should NOT have to specify the dimensions; network builder can figure it out. Keep it for now, fix when it is time.
                // We instantiate not the node directly, but a wrapped version that can cast to LateAttachingNode, which holds a lambda to complete the attachment process at the appropriate time.
                function<void(ComputationNode<ElemType>*)> completeAttachInputs = [configp](ComputationNode<ElemType>* node)   // This is the lambda to complete the process. Note that config captured as a shared_ptr.
                {
                    node->AttachInputs(GetInputs(*configp));    // this is executed by network builder while iterating the nodes
                };
                if (OpIs(PastValueNode))
                    node = New<LateAttachingNode<PastValueNode<ElemType>>>(deviceId, nodeName, completeAttachInputs, (ElemType)config[L"defaultHiddenActivation"], (size_t)config[L"rows"], (size_t)config[L"cols"], (size_t)config[L"timeStep"]);
                else
                    node = New<LateAttachingNode<FutureValueNode<ElemType>>>(deviceId, nodeName, completeAttachInputs, (ElemType)config[L"defaultHiddenActivation"], (size_t)config[L"rows"], (size_t)config[L"cols"], (size_t)config[L"timeStep"]);
            }
            else        // nodes with inputs
            {
                let inputs = GetInputs(config);
                // second group: nodes with special initializers
#if 0
                /*else*/ if (cnNodeType == RowSliceNode<ElemType>::TypeName())
                {
                    if (parameter.size() != 3)
                        RuntimeError("RowSlice should have three parameters. Usage: RowSlice(startRowIndex, numRows, origNodeName.");

                    nodeParamCount = 1;
                    nodeParamStart = 2;

                    if (pass == ndlPassInitial)
                    {
                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                        size_t start_index = ((NDLNode<ElemType>*)params[0])->GetScalar();
                        size_t num_rows = ((NDLNode<ElemType>*)params[1])->GetScalar();

                        bool needGradient = node->GetOptionalParameter("needGradient", "false");
                        nodePtr = m_net.RowSlice(NULL, start_index, num_rows, name);
                        nodePtr->NeedGradient() = needGradient;
                    }
                }
#endif
                if (OpIs(RowSliceNode)) // TODO: untested
                {
                    // startIndex, numRows, inputs /*one*/, needGradient=false
                    node = New<RowSliceNode<ElemType>>(deviceId, nodeName, (size_t)config[L"startIndex"], (size_t)config[L"numRows"]);
                    node->NeedGradient() = config[L"needGradient"];
                }
#if 0
                else if (cnNodeType == RowRepeatNode<ElemType>::TypeName())
                {
                    if (parameter.size() != 2)
                        RuntimeError("RowRepeat should have two parameters. Usage: RowRepeat(origNodeName, numRepeats.");

                    nodeParamCount = 1;
                    nodeParamStart = 0;

                    if (pass == ndlPassInitial)
                    {
                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                        size_t num_repeat = ((NDLNode<ElemType>*)params[1])->GetScalar();

                        bool needGradient = node->GetOptionalParameter("needGradient", "false");
                        nodePtr = m_net.RowRepeat(NULL, num_repeat, name);
                        nodePtr->NeedGradient() = needGradient;
                    }
                }
#endif
                else if (OpIs(RowRepeatNode)) // TODO: untested
                {
                    // inputs /*one*/, numRepeats, needGradient=false
                    node = New<RowRepeatNode<ElemType>>(deviceId, nodeName, (size_t)config[L"numRepeats"]);
                    node->NeedGradient() = config[L"needGradient"];
                }
#if 0
                else if (cnNodeType == ReshapeNode<ElemType>::TypeName())
                {
                    if (parameter.size() < 2 || parameter.size() > 5)
                        RuntimeError("Reshape should have two to five parameters. Usage: Reshape(origNodeName, numRows, [imageWidth=], [imageHeight=], [imageChannels=].");

                    nodeParamCount = 1;
                    nodeParamStart = 0;

                    if (pass == ndlPassInitial)
                    {
                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                        size_t num_rows = ((NDLNode<ElemType>*)params[1])->GetScalar();
                        size_t img_width = node->GetOptionalParameter("imageWidth", "0");
                        size_t img_height = node->GetOptionalParameter("imageHeight", "0");
                        size_t img_channels = node->GetOptionalParameter("imageChannels", "0");

                        bool needGradient = node->GetOptionalParameter("needGradient", "false");
                        nodePtr = m_net.Reshape(NULL, num_rows, img_width, img_height, img_channels, name);
                        nodePtr->NeedGradient() = needGradient;
                    }
                }
#endif
                else if (OpIs(ReshapeNode)) // TODO: untested
                {
                    // inputs /*one*/, numRows, imageWidth = 0, imageHeight = 0, imageChannels = 0
                    node = New<ReshapeNode<ElemType>>(deviceId, nodeName, (size_t)config[L"numRows"], (size_t)config[L"imageWidth"], (size_t)config[L"imageHeight"], (size_t)config[L"imageChannels"]);
                    node->NeedGradient() = config[L"needGradient"];
                    //nodePtr = m_net.Reshape(NULL, num_rows, img_width, img_height, img_channels, name);
                    // BUGBUG: ^^ how to implement this?? We got no network here. What is this for?
                    LogicError("ReshapeNode not working with BS because init code needs access to network which we don't haveyet--to be fixed elsewhere");
                }
#if 0
                else if (cnNodeType == ConvolutionNode<ElemType>::TypeName())
                {
                    if (parameter.size() != 7)
                        RuntimeError("%ls should have 7 fixed parameters[weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels,horizontalSubsample, verticalSubsample] and two optional parameters [zeroPadding = [false|yourvalue], maxTempMemSizeInSamples = [0|yourvalue]].", cnNodeType.c_str());

                    // setup the parameter position of children so we can hook them up later
                    nodeParamCount = 2;
                    nodeParamStart = 0;

                    if (pass == ndlPassInitial)
                    {
                        int id = 2; // skip weightNode and inputValueNode

                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
                        id = 0; // reset counter because the params array starts at zero
                        size_t kernelWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t kernelHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t outputChannels = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();

                        assert(id == 5);

                        //optional
                        bool zeroPadding = node->GetOptionalParameter("zeroPadding", "false");
                        size_t maxTempMemSizeInSamples = node->GetOptionalParameter("maxTempMemSizeInSamples", "0");


                        nodePtr = m_net.Convolution(NULL, NULL, kernelWidth, kernelHeight, outputChannels,
                            horizontalSubsample, verticalSubsample, zeroPadding, name, maxTempMemSizeInSamples);
                    }
                }
#endif
                else if (OpIs(ConvolutionNode)) // TODO: untested
                {
                    // weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0
                    node = New<ConvolutionNode<ElemType>>(deviceId, nodeName, (size_t)config[L"kernelWidth"], (size_t)config[L"kernelHeight"], (size_t)config[L"outputChannels"],
                                                                              (size_t)config[L"horizontalSubsample"], (size_t)config[L"verticalSubsample"],
                                                                              (bool)config[L"zeroPadding"], (size_t)config[L"maxTempMemSizeInSamples"]);
                }
#if 0
                else if (cnNodeType == MaxPoolingNode<ElemType>::TypeName())
                {
                    if (parameter.size() != 5)
                        RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

                    // setup the parameter position of children so we can hook them up later
                    nodeParamCount = 1;
                    nodeParamStart = 0;

                    if (pass == ndlPassInitial)
                    {
                        int id = 1; // skip inputValueNode

                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
                        id = 0; // reset counter because the params array starts at zero
                        size_t windowWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t windowHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();

                        assert(id == 4);

                        nodePtr = m_net.MaxPooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight,
                            horizontalSubsample, verticalSubsample, name);
                    }
                }
#endif
                else if (OpIs(MaxPoolingNode)) // TODO: untested
                {
                    // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
                    node = New<MaxPoolingNode<ElemType>>(deviceId, nodeName, (size_t)config[L"windowWidth"], (size_t)config[L"windowHeight"], (size_t)config[L"horizontalSubsample"], (size_t)config[L"verticalSubsample"]);
                }
#if 0
                else if (cnNodeType == AveragePoolingNode<ElemType>::TypeName())
                {
                    if (parameter.size() != 5)
                        RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

                    // setup the parameter position of children so we can hook them up later
                    nodeParamCount = 1;
                    nodeParamStart = 0;

                    if (pass == ndlPassInitial)
                    {
                        int id = 1; // skip inputValueNode

                        // evaluate only scalar parameters
                        vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
                        id = 0; // reset counter because the params array starts at zero
                        size_t windowWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t windowHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                        size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();

                        assert(id == 4);

                        nodePtr = m_net.AveragePooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight,
                            horizontalSubsample, verticalSubsample, name);
                    }
                }
#endif
                else if (OpIs(AveragePoolingNode)) // TODO: untested
                {
                    // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
                    node = New<AveragePoolingNode<ElemType>>(deviceId, nodeName, (size_t)config[L"windowWidth"], (size_t)config[L"windowHeight"], (size_t)config[L"horizontalSubsample"], (size_t)config[L"verticalSubsample"]);
                }
                // last group: standard nodes that only take 'inputs'
                else
                {
                    node = ComputationNetworkBuilder<ElemType>::NewStandardNode(operationName, deviceId, nodeName);
                }
                node->AttachInputs(inputs); // TODO: where to check the number of inputs? Should be a template parameter to ComputationNode!
            }
            // add a tag
            let nodeWithTag = dynamic_pointer_cast<WithTag>(node);
            if (nodeWithTag)
                nodeWithTag->SetTag(config[L"tag"]);
            // and done
            return node;
        }
    private:
        // helper for the factory function for ComputationNodes
        static vector<ComputationNodeBasePtr> GetInputs(const IConfigRecord & config)
        {
            vector<ComputationNodeBasePtr> inputs;
            let inputsArg = config[L"inputs"];
            if (inputsArg.Is<ComputationNodeBase>())                // single arg
                inputs.push_back(inputsArg);
            else                                                    // a whole vector
            {
                ConfigArrayPtr inputsArray = (ConfigArrayPtr&)inputsArg;
                let range = inputsArray->GetIndexRange();
                for (int i = range.first; i <= range.second; i++)   // pull them. This will resolve all of them.
                    inputs.push_back(inputsArray->At(i, inputsArg.GetLocation()));
            }
            return inputs;
        }
    };

    // -------------------------------------------------------------------
    // ComputationNetwork
    // -------------------------------------------------------------------

    template<typename ElemType>
    struct DualPrecisionHelpers<ElemType, ComputationNetwork<ElemType>>
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

        // initialize a ComputationNetwork<ElemType> from a ConfigRecord
        static shared_ptr<Object> MakeRuntimeObject(const IConfigRecordPtr configp)
        {
            let & config = *configp;

            DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
            auto net = make_shared<ComputationNetwork<ElemType>>(deviceId);

            auto & m_nameToNodeMap = net->GetNameToNodeMap();

            deque<ComputationNodeBasePtr> workList;
            // flatten the set of all nodes
            // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
            // TODO: This currently only collects nodes of the same ElemType. We could allow conversion operators.
            // TODO: Can we even make the ComputationNetwork independent of ElemType?? As long as the nodes themselves are hooked up properly that should be OK!
            for (let & id : config.GetMemberIds())
            {
                let & value = config[id];
                if (value.Is<ComputationNode<ElemType>>())
                    workList.push_back((ComputationNodePtr&)value);
            }
            // process work list
            // Also call FinalizeInit where we must.
            while (!workList.empty())
            {
                let node = workList.front();
                workList.pop_front();

                // add to set
                let res = m_nameToNodeMap.insert(make_pair(node->NodeName(), node));
                if (!res.second)        // not inserted: we already got this one
                    if (res.first->second == node)
                        continue;       // the same
                    else                // oops, a different node with the same name
                        LogicError("ComputationNetwork: multiple nodes with the same NodeName() '%ls'", node->NodeName().c_str());

                // If node derives from MustFinalizeInit() then it has unresolved inputs. Resolve them now.
                // This may generate a whole new load of nodes, including nodes which in turn have late init.
                // TODO: think this through whether it may generate circular references nevertheless
                let lateAttachingNode = dynamic_pointer_cast<ILateAttachingNode>(node);
                if (lateAttachingNode)
                    lateAttachingNode->LateAttachInputs();

                // add it to the respective node group based on the tag
                let nodeWithTag = dynamic_pointer_cast<WithTag>(node);
                if (nodeWithTag)
                {
                    wstring tag = nodeWithTag->GetTag();
                    if (tag == L"feature")                              net->FeatureNodes().push_back(node);
                    else if (tag == L"label")                           net->LabelNodes().push_back(node);
                    else if (tag == L"criterion" || tag == L"criteria") net->FinalCriterionNodes().push_back(node); // 'criteria' is wrong (plural); we keep it for compat
                    else if (!_wcsnicmp(tag.c_str(), L"eval", 4))       net->EvaluationNodes().push_back(node);     // eval*
                    else if (tag == L"output")                          net->OutputNodes().push_back(node);
                    else if (tag == L"pair")                            net->PairNodes().push_back(node);           // TODO: I made this up; the original code in SynchronousExecutionEngine did not have this
                    else if (tag == L"multiseq")                        net->NodesReqMultiSeqHandling().push_back(node);
                    else if (!tag.empty())
                        RuntimeError("ComputationNetwork: unknown tag '%ls'", tag.c_str());
                    // TODO: are there nodes without tag? Where do they go?
                }

                // TODO: ...can we do stuff like propagating dimensions here? Or still too early?

                // traverse children: append them to the end of the work list
                let children = node->GetChildren();
                for (auto child : children)
                    workList.push_back(child);  // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
            }

            // TODO: what is missing is the dimensions
#if 1
            wstring args = net->ToString();
            fprintf(stderr, "%ls\n", args.c_str());
#endif
            // these post-processing steps are done by the other network builders, but I don't know why they are necessary
            net->FixupInputMinibatchSize();         // make sure dimensions are set up correctly
            net->ResetEvalTimeStamp();              // (should not really be needed)
            return net;
        }

        // -------------------------------------------------------------------
        // ... more specialized node types that have extra constructor parameters
        // -------------------------------------------------------------------

        // fragment from original NDL--optional params are evaluated afterwards, such as initvalue
        // node->EvaluateMacro(nodeEval, baseName, pass);
        // nodeEval.ProcessOptionalParameters(node);
    };

    // creates the lambda for creating an object that can exist as 'float' or 'double'
    // Pass both types as the two template args.
    template<class Cfloat, class Cdouble>
    static ConfigurableRuntimeType MakeRuntimeTypeConstructorDualPrecision()
    {
        ConfigurableRuntimeType rtInfo;
        rtInfo.construct = [](const IConfigRecordPtr config)        // lambda to construct--this lambda can construct both the <float> and the <double> variant based on config parameter 'precision'
        {
            wstring precision = (*config)[L"precision"];            // dispatch on ElemType
            if (precision == L"float")
                return DualPrecisionHelpers<float, Cfloat>::MakeRuntimeObject(config);
            else if (precision == L"double")
                return DualPrecisionHelpers<double, Cdouble>::MakeRuntimeObject(config);
            else
                RuntimeError("invalid value for 'precision', must be 'float' or 'double'");
        };
        rtInfo.isConfigRecord = is_base_of<IConfigRecord, Cfloat>::value;
        static_assert(is_base_of<IConfigRecord, Cfloat>::value == is_base_of<IConfigRecord, Cdouble>::value, "");   // we assume that both float and double have the same behavior
        return rtInfo;
    }

    //#define DefineRuntimeType(T) { L ## #T, MakeRuntimeTypeConstructors<T>() } }
#define DefineRuntimeTypeDualPrecision(T) { L ## #T, MakeRuntimeTypeConstructorDualPrecision<T<float>,T<double>>() }

    // get information about configurable runtime types
    // This returns a ConfigurableRuntimeType structure which primarily contains a lambda to construct a runtime object from a ConfigRecord ('new' expression).
    const ConfigurableRuntimeType * FindExternalRuntimeTypeInfo(const wstring & typeId)
    {
        // lookup table for "new" expression
        // This table lists all C++ types that can be instantiated from "new" expressions, and gives a constructor lambda and type flags.
        static map<wstring, ConfigurableRuntimeType> configurableRuntimeTypes =
        {
            // ComputationNodes
            DefineRuntimeTypeDualPrecision(ComputationNode),
            DefineRuntimeTypeDualPrecision(ComputationNetwork),
#if 0
            DefineRuntimeType(RecurrentComputationNode),
            // In this experimental state, we only have Node and Network.
            // Once BrainScript becomes the driver of everything, we will add other objects like Readers, Optimizers, and Actions here.
#endif
        };

        // first check our own
        let newIter = configurableRuntimeTypes.find(typeId);
        if (newIter != configurableRuntimeTypes.end())
            return &newIter->second;
        return nullptr; // not found
    }

}}}

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace Microsoft::MSR;

    // helper that returns 'float' or 'double' depending on ElemType
    template<typename ElemType> static const wchar_t * ElemTypeName();
    template<> /*static*/ const wchar_t * ElemTypeName<float>()  { return L"float"; }
    template<> /*static*/ const wchar_t * ElemTypeName<double>() { return L"double"; }

    // build a ComputationNetwork from BrainScript source code
    template<typename ElemType>
    /*virtual*/ /*IComputationNetBuilder::*/ComputationNetwork<ElemType>* ExperimentalNetworkBuilder<ElemType>::BuildNetworkFromDescription(ComputationNetwork<ElemType>*)
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
            let network = dynamic_pointer_cast<ComputationNetwork<ElemType>>(object);   // cast it
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
