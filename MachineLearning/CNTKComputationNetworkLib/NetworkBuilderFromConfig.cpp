// NetworkBuilderFromConfig.cpp -- interface to node and network creation from glue languages through config record parameters  --fseide

#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ScriptableObjects.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "RecurrentNodes.h"
#include "NonlinearityNodes.h"
#include "LinearAlgebraNodes.h"
#include "ConvolutionalNodes.h"
#include "ReshapingNodes.h"

#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"

#include <memory>
#include <deque>
#include <set>
#include <string>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

    using namespace Microsoft::MSR;

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

    template<class ElemType>
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

#define OpIs(op) (operationName == msra::strfun::utf16(OperationNameOf(op)))

            // first group: nodes without inputs
            if (OpIs(InputValue))
            {
                let isSparse = config[L"isSparse"];
                let isImage  = config[L"isImage"];
                if (!isImage)
                    node = New<InputValue<ElemType>>(deviceId, nodeName, (size_t)config[L"rows"], (size_t)config[L"cols"], isSparse);
                else
                    node = New<InputValue<ElemType>>(deviceId, nodeName, ImageLayoutWHC(config[L"imageWidth"], config[L"imageHeight"], config[L"imageChannels"]), (size_t)config[L"numImages"], isSparse);
            }
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
                // TODO: "needGradient" should be renamed to better match m_parameterUpdateRequired
                node->SetParameterUpdateRequired(config[L"needGradient"]);
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
                    dynamic_pointer_cast<LearnableParameter<ElemType>>(node)->InitFromFile(initFromFilePath);
                }
                else
                    RuntimeError("init must be one of the values of [uniform|gaussian|fixedValue|fromFile]");
            }
            // Constant is implemented as a LearnableParameter with initializion as fixedValue with needGradient false, on script level
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
                // legacy: bad spelling. Warn users who may have converted.
                if (config.Find(L"defaultHiddenActivity"))
                    config[L"defaultHiddenActivity"].Fail(L"Past/FutureValueNode: Optional NDL parameter 'defaultHiddenActivity' should be spelled 'defaultHiddenActivation'. Please update your script.");
                if (OpIs(PastValueNode))
                    node = New<LateAttachingNode<PastValueNode<ElemType>>>(deviceId, nodeName, completeAttachInputs, (ElemType)config[L"defaultHiddenActivation"], (size_t)config[L"rows"], (size_t)config[L"cols"], (size_t)config[L"timeStep"]);
                else
                    node = New<LateAttachingNode<FutureValueNode<ElemType>>>(deviceId, nodeName, completeAttachInputs, (ElemType)config[L"defaultHiddenActivation"], (size_t)config[L"rows"], (size_t)config[L"cols"], (size_t)config[L"timeStep"]);
            }
            else        // nodes with inputs
            {
                let inputs = GetInputs(config);
                // second group: nodes with special initializers
                if (OpIs(RowSliceNode)) // TODO: untested
                {
                    // startIndex, numRows, inputs /*one*/, needGradient=false
                    node = New<RowSliceNode<ElemType>>(deviceId, nodeName, (size_t)config[L"startIndex"], (size_t)config[L"numRows"]);
                    node->SetParameterUpdateRequired(config[L"needGradient"]);
                }
                else if (OpIs(RowRepeatNode)) // TODO: untested
                {
                    // inputs /*one*/, numRepeats, needGradient=false
                    node = New<RowRepeatNode<ElemType>>(deviceId, nodeName, (size_t)config[L"numRepeats"]);
                    node->SetParameterUpdateRequired(config[L"needGradient"]);
                }
                else if (OpIs(DiagonalNode))
                {
                    // inputs /*one*/, numRepeats, needGradient=false
                    node = New<DiagonalNode<ElemType>>(deviceId, nodeName);
                    node->SetParameterUpdateRequired(config[L"needGradient"]);
                }
                else if (OpIs(ReshapeNode)) // TODO: untested
                {
                    // inputs /*one*/, numRows, imageWidth = 0, imageHeight = 0, imageChannels = 0
                    node = New<ReshapeNode<ElemType>>(deviceId, nodeName, (size_t)config[L"numRows"], ImageLayoutWHC(config[L"imageWidth"], config[L"imageHeight"], config[L"imageChannels"]));
                }
                else if (OpIs(ConvolutionNode)) // TODO: untested
                {
                    // weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, zeroPadding = false, maxTempMemSizeInSamples = 0
                    node = New<ConvolutionNode<ElemType>>(deviceId, nodeName, (size_t)config[L"kernelWidth"], (size_t)config[L"kernelHeight"], (size_t)config[L"outputChannels"],
                                                                              (size_t)config[L"horizontalSubsample"], (size_t)config[L"verticalSubsample"],
                                                                              (bool)config[L"zeroPadding"], (size_t)config[L"maxTempMemSizeInSamples"]);
                }
                else if (OpIs(MaxPoolingNode)) // TODO: untested
                {
                    // input, windowWidth, windowHeight, horizontalSubsample, verticalSubsample
                    node = New<MaxPoolingNode<ElemType>>(deviceId, nodeName, (size_t)config[L"windowWidth"], (size_t)config[L"windowHeight"], (size_t)config[L"horizontalSubsample"], (size_t)config[L"verticalSubsample"]);
                }
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
                    inputs.push_back(inputsArray->At(i, [](const wstring &){ LogicError("GetInputs: out of bounds index while iterating??"); }));
            }
            return inputs;
        }
    };

    // -------------------------------------------------------------------
    // ComputationNetwork
    // -------------------------------------------------------------------

    // initialize a ComputationNetwork from a ConfigRecord
    template<>
    /*static*/ shared_ptr<Object> MakeRuntimeObject<ComputationNetwork>(const IConfigRecordPtr configp)
    {
        let & config = *configp;

        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        auto net = make_shared<ComputationNetwork>(deviceId);

        auto & m_nameToNodeMap = net->GetNameToNodeMap();

        deque<ComputationNodeBasePtr> workList;
        // flatten the set of all nodes
        // we collect all root ComputationNodes from the config record, and then expand into all their children by work-list processing
        // TODO: This currently only collects nodes of the same ElemType. We could allow conversion operators.
        // TODO: Can we even make the ComputationNetwork independent of ElemType?? As long as the nodes themselves are hooked up properly that should be OK!
        for (let & id : config.GetMemberIds())
        {
            let & value = config[id];
            if (value.Is<ComputationNodeBase>())
                workList.push_back((const ComputationNodeBasePtr&)value);
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
#if 0           // deprecated
                else if (tag == L"pair")                            net->PairNodes().push_back(node);           // TODO: I made this up; the original code in SynchronousExecutionEngine did not have this
#endif
                else if (!tag.empty())
                    RuntimeError("ComputationNetwork: unknown tag '%ls'", tag.c_str());
                // TODO: are there nodes without tag? Where do they go?
            }

            // traverse children: append them to the end of the work list
            let & children = node->GetChildren();
            for (auto & child : children)
                workList.push_back(child);  // (we could check whether c is in 'nodes' already here to optimize, but this way it is cleaner)
        }

        net->ValidateNetwork();
#if 1
        wstring args = net->ToString();
        fprintf(stderr, "%ls\n", args.c_str());
#endif
        // these post-processing steps are done by the other network builders, but I don't know why they are necessary
        net->FixupInputMinibatchSize();         // make sure dimensions are set up correctly
        net->ResetEvalTimeStamp();              // (should not really be needed)
        return net;
    }

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
                RuntimeError("invalid value '%ls' for 'precision', must be 'float' or 'double'", precision.c_str());
        };
        rtInfo.isConfigRecord = is_base_of<IConfigRecord, Cfloat>::value;
        static_assert(is_base_of<IConfigRecord, Cfloat>::value == is_base_of<IConfigRecord, Cdouble>::value, "");   // we assume that both float and double have the same behavior
        return rtInfo;
    }

    // and the regular one without ElemType dependency
    template<class C>
    static ConfigurableRuntimeType MakeRuntimeTypeConstructor()
    {
        ConfigurableRuntimeType rtInfo;
        rtInfo.construct = [](const IConfigRecordPtr config)        // lambda to construct--this lambda can construct both the <float> and the <double> variant based on config parameter 'precision'
        {
            return MakeRuntimeObject<C>(config);
        };
        rtInfo.isConfigRecord = is_base_of<IConfigRecord, C>::value;
        return rtInfo;
    }

#define DefineRuntimeType(T) { L ## #T, MakeRuntimeTypeConstructor<T>() }
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
            DefineRuntimeType(ComputationNetwork),
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
