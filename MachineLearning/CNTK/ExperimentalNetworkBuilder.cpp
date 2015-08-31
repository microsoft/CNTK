// ExperimentalNetworkBuilder.cpp -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ExperimentalNetworkBuilder.h"
#include "BrainScriptEvaluator.h"

#include "ComputationNode.h"
#include "ComputationNetwork.h"

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

    // TODO: must be moved to ComputationNode.h
    // a ComputationNode that derives from MustFinalizeInit does not resolve some args immediately (just keeps ConfigValuePtrs),
    // assuming they are not ready during construction.
    // This is specifically meant to be used by DelayNode, see comments there.
    struct MustFinalizeInit { virtual void FinalizeInit() = 0; };   // derive from this to indicate ComputationNetwork should call FinalizeIitlate initialization

    wstring computationNodes =
        L"Parameter(rows, cols, needGradient = true, init = 'uniform'/*|fixedValue|gaussian|fromFile*/, initValueScale = 1, value = 0, initFromFilePath = '', tag='') = new ComputationNode [ operation = 'LearnableParameter' /*plus the function args*/ ]\n"
        L"Input(rows, cols, tag='feature') = new ComputationNode [ operation = 'InputValue' /*plus the function args*/ ]\n" // note: naming a little inconsistent
        // untested:
        L"SparseInput(rows, cols, tag='feature') = new ComputationNode [ operation = 'SparseInputValue' /*plus the function args*/ ]\n"
        // ^^ already works; vv not yet working
        L"RowSlice(firstRow, rows, features, tag='') = new ComputationNode [ operation = 'RowSlice' ; inputs = features ; first = firstRow ; num = rows /* ; tag = tag */ ]\n"
        L"Delay(in, delay, tag='') = new ComputationNode [ operation = 'Delay' ; input = in ; deltaT = -delay /* ; tag = tag */ ]\n"
        // standard nodes, tested
        L"Mean(z, tag='') = new ComputationNode [ operation = 'Mean' ; inputs = z /* ; tag = tag */ ]\n"
        L"InvStdDev(z, tag='') = new ComputationNode [ operation = 'InvStdDev' ; inputs = z /* ; tag = tag */ ]\n"
        L"PerDimMeanVarNormalization(feat,mean,invStdDev, tag='') = new ComputationNode [ operation = 'PerDimMeanVarNormalization' ; inputs = feat:mean:invStdDev /* ; tag = tag */ ]\n"
        L"Sigmoid(z, tag='') = new ComputationNode [ operation = 'Sigmoid' ; inputs = z /* ; tag = tag */ ]\n"
        L"CrossEntropyWithSoftmax(labels, outZ, tag='criterion') = new ComputationNode [ operation = 'CrossEntropyWithSoftmax' ; inputs = labels:outZ ]\n"
        L"ErrorPrediction(labels, outZ, tag='') = new ComputationNode [ operation = 'ErrorPrediction' ; inputs = labels:outZ /* ; tag = tag */ ]\n"
        // standard nodes, untested
        L"Log(z, tag='') = new ComputationNode [ operation = 'Log' ; inputs = z /* ; tag = tag */ ]\n"
        ;

    template<typename ElemType>
    struct DualPrecisionHelpers
    {
        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

        // basic function template, for classes that can instantiate themselves from IConfigRecordPtr
        // TODO: do we even have any?
        template<class C>
        static shared_ptr<Object> MakeRuntimeObject(const IConfigRecordPtr config)
        {
            return make_shared<C>(config);
        }

        // -------------------------------------------------------------------
        // ComputationNetwork
        // -------------------------------------------------------------------

        // initialize a ComputationNetwork<ElemType> from a ConfigRecord
        template<>
        static shared_ptr<Object> MakeRuntimeObject<ComputationNetwork<ElemType>>(const IConfigRecordPtr configp)
        {
            let & config = *configp;

            DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
            auto net = make_shared<ComputationNetwork<ElemType>>(deviceId);

            auto & m_nameToNodeMap = net->GetNameToNodeMap();

            deque<ComputationNodePtr> workList;
            // flatten the set of all nodes
            // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
            // TODO: This currently only collects nodes of the same ElemType. We could allow conversion operators.
            // TODO: Can we even make the ComputationNetwork independent of ElemType?? As long as the nodes themselves are hooked up properly that should be OK!
            for (let & id : config.GetMemberIds())
            {
                let & value = config[id];
                if (value.Is<ComputationNode<ElemType>>())
                    workList.push_back((ComputationNodePtr)value);
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
                let mustFinalizeInit = dynamic_pointer_cast<MustFinalizeInit>(node);
                if (mustFinalizeInit)
                    mustFinalizeInit->FinalizeInit();

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
            return net;
        }

        // -------------------------------------------------------------------
        // ComputationNode -- covers all standard nodes
        // -------------------------------------------------------------------

    private:
        // helper for the factory function for ComputationNodes
        static vector<ComputationNodePtr> GetInputs(const IConfigRecord & config)
        {
            vector<ComputationNodePtr> inputs;
            let inputsArg = config[L"inputs"];
            if (inputsArg.Is<ComputationNode<ElemType>>())          // single arg
                inputs.push_back(inputsArg);
            else                                                    // a whole vector
            {
                let inputsArray = (ConfigArrayPtr)inputsArg;
                let range = inputsArray->GetIndexRange();
                for (int i = range.first; i <= range.second; i++)   // pull them. This will resolve all of them.
                    inputs.push_back(inputsArray->At(i, inputsArg.GetLocation()));
            }
            return inputs;
        }
    public:
        // create ComputationNode
        // This is the equivalent of the old SynchronousNodeEvaluator::Evaluate(), and we duplicate code from there.
        template<>
        static shared_ptr<Object> MakeRuntimeObject<ComputationNode<ElemType>>(const IConfigRecordPtr configp)
        {
            let & config = *configp;
            wstring operationName = config[L"operation"];
            wstring nodeName = L"<placeholder>";   // name will be overwritten by caller upon return (TODO: fix this here? pass expression name in?)
            DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
            static unsigned long m_randomSeedOffset = 0;    // TODO: this is held in the ComputationNetwork, but we don't have one yet
            // TODO" ^^actually it seems only used by initialization of LearnableParameters--check that again; in that case, we can have a local

            // note on optional parameters
            // Instead of defining optional parameters here in code, they are defined as optional args to the creating macro.

            ComputationNodePtr node;

            // first group: nodes without inputs
            // TODO: each block is preceded by the respective code from SynchronousNodeEvaluator::Evaluate()--remove these when this all works
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
#endif
            if (operationName == L"InputValue" || operationName == L"SparseInputValue") // TODO: sparse case untested
            {
                let isSparse = (operationName == L"SparseInputValue");
                node = New<InputValue<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"], isSparse);
            }
            if (operationName == L"ImageInput" || operationName == L"SparseImageInput") // TODO: untested
            {
                let isSparse = (operationName == L"SparseImageInput");
                //size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                //size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                //size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                //size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;
                node = New<InputValue<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"], isSparse);
            }
#if 0
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
#endif
            if (operationName == L"LearnableParameter")
            {
                // parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float])
                // TODO: do we need a default value mechanism? How to make sure it does not pop upwards? Current functions do not allow overloads.
                // TODO: test this with random init for QuickE2E on CPU against SimpleNetworkBuilder
                node = New<LearnableParameter<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"]);
                node->NeedGradient() = config[L"needGradient"];
                static int randomSeed = 1;
                wstring initString = config[L"init"];
                if (initString == L"fixedValue")
                    node->FunctionValues().SetValue((ElemType)config[L"value"]);
                else if (initString == L"uniform" || initString == L"gaussian")
                    ComputationNetwork<ElemType>::InitLearnableParameters(node, (initString == L"uniform"), randomSeed++, config[L"initValueScale"], m_randomSeedOffset);
                else if (initString == L"fromFile")
                {
                    wstring initFromFilePath = config[L"initFromFilePath"];
                    if (initFromFilePath.empty())
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    ComputationNetwork<ElemType>::InitLearnableParametersFromFile(node, initFromFilePath, node->GetDeviceId());
                }
                else
                    RuntimeError("init must be one of the values of [uniform|gaussian|fixedValue|fromFile]");
            }
#if 0
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
            else if (cnNodeType == RowSliceNode<ElemType>::TypeName())
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

                    nodePtr->NeedGradient() = needGradient;
                }
            }
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
            else        // nodes with inputs
            {
                let inputs = GetInputs(config);
                // second group: nodes with special initializers
                // third group: 
                node = ComputationNetwork<ElemType>::NewStandardNode(operationName, deviceId, nodeName);
                node->AttachInputs(inputs); // TODO: where to check the number of inputs? Should be a template parameter to ComputationNode!
            }
            // add a tag
            let nodeWithTag = dynamic_pointer_cast<WithTag>(node);
            if (nodeWithTag)
                nodeWithTag->SetTag(config[L"tag"]);
            // and done
            return node;
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
                return DualPrecisionHelpers<float>::MakeRuntimeObject<Cfloat>(config);
            else if (precision == L"double")
                return DualPrecisionHelpers<double>::MakeRuntimeObject<Cdouble>(config);
            else
                RuntimeError("invalid value for 'precision', must be 'float' or 'double'");
        };
        rtInfo.isConfigRecord = is_base_of<IConfigRecord, Cfloat>::value;
        static_assert(is_base_of<IConfigRecord, Cfloat>::value == is_base_of<IConfigRecord, Cdouble>::value, "");   // we assume that both float and double have the same behavior
        return rtInfo;
    }

    //#define DefineRuntimeType(T) { L#T, MakeRuntimeTypeConstructors<T>() } }
#define DefineRuntimeTypeDualPrecision(T) { L#T, MakeRuntimeTypeConstructorDualPrecision<T<float>,T<double>>() }

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
    template<> static const wchar_t * ElemTypeName<float>()  { return L"float"; }
    template<> static const wchar_t * ElemTypeName<double>() { return L"double"; }

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
                + wstrprintf(L"deviceId = %d ; precision = '%s' ; network = new ComputationNetwork ", (int)m_deviceId, ElemTypeName<ElemType>())  // TODO: check if typeid needs postprocessing
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
        }
        m_net->ResetEvalTimeStamp();
        return m_net.get();
    }

    template class ExperimentalNetworkBuilder<float>;
    template class ExperimentalNetworkBuilder<double>;

}}}
