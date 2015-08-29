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

namespace Microsoft { namespace MSR { namespace CNTK { namespace BS {   // new config parsing lives in a sub-namespace, as to avoid conflict with existing ones which get implicitly pulled in by some headers we need

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

    wstring computationNodes =      // BUGBUG: optional args not working yet, some scope problem causing a circular reference
        L"Mean(z, tag='') = new ComputationNode [ class = 'MeanNode' ; inputs = z /* ; tag = tag */ ]\n"
        L"InvStdDev(z, tag='') = new ComputationNode [ class = 'InvStdDevNode' ; inputs = z /* ; tag = tag */ ]\n"
        L"PerDimMeanVarNormalization(feat,mean,invStdDev, tag='') = new ComputationNode [ class = 'PerDimMeanVarNormalizationNode' ; inputs = feat:mean:invStdDev /* ; tag = tag */ ]\n"
        L"Parameter(outD, inD/*, tag=''*/) = new ComputationNode [ class = 'LearnableParameterNode' ; outDim = outD ; inDim = inD /*; optionalTag = 'tag'*/ ]\n"
        L"Input(dim) = Parameter(dim,1/*,tag='features'*/)   // TODO: for now \n"
        L"RowSlice(firstRow, rows, features, tag='') = new ComputationNode [ class = 'RowSliceNode' ; inputs = features ; first = firstRow ; num = rows /* ; tag = tag */ ]\n"
        L"Delay(in, delay, tag='') = new ComputationNode [ class = 'DelayNode' ; input = in ; deltaT = -delay /* ; tag = tag */ ]\n"
        L"Sigmoid(z, tag='') = new ComputationNode [ class = 'SigmoidNode' ; inputs = z /* ; tag = tag */ ]\n"
        L"Log(z, tag='') = new ComputationNode [ class = 'LogNode' ; inputs = z /* ; tag = tag */ ]\n"
        L"CrossEntropyWithSoftmax(labels, outZ, tag='') = new ComputationNode [ class = 'CrossEntropyWithSoftmaxNode' ; inputs = labels:outZ /* ; tag = tag */ ]\n"
        L"ErrorPrediction(labels, outZ, tag='') = new ComputationNode [ class = 'ErrorPredictionNode' ; inputs = labels:outZ /* ; tag = tag */ ]\n"
        ;

    wstring commonMacros =  // TODO: rename rows and cols to inDim and outDim or vice versa, whichever it is
        L"BFF(in, rows, cols) = [ B = Parameter(rows, 1/*init = fixedvalue, value = 0*/) ; W = Parameter(rows, cols) ; z = /*W*in+B*/Log(in) ] \n" // TODO: fix this once we got the ComputationNode type connected correctly
        L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
        L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
        L"LogPrior(labels) = Log(Mean(labels)) \n"
        ;

    // TODO: must be moved to ComputationNode.h
    // a ComputationNode that derives from MustFinalizeInit does not resolve some args immediately (just keeps ConfigValuePtrs),
    // assuming they are not ready during construction.
    // This is specifically meant to be used by DelayNode, see comments there.
    struct MustFinalizeInit { virtual void FinalizeInit() = 0; };   // derive from this to indicate ComputationNetwork should call FinalizeIitlate initialization

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
            for (let & id : config.GetMemberIds())
            {
                let & value = config[id];
                if (value.Is<ComputationNode<ElemType>>())
                    workList.push_back((ComputationNodePtr)value);
            }
            // process work list
            // Also call FinalizeInit where we must.
            set<ComputationNodePtr> inputs;         // all input nodes
            set<ComputationNodePtr> outputs;        // all output nodes
            set<ComputationNodePtr> parameters;     // all parameter nodes
            set<ComputationNodePtr> allChildren;    // all nodes that are children of others (those that are not are output nodes)
            while (!workList.empty())
            {
                let n = workList.front();
                workList.pop_front();
                // add to set
                let res = m_nameToNodeMap.insert(make_pair(n->NodeName(), n));
                if (!res.second)        // not inserted: we already got this one
                    if (res.first->second == n)
                        continue;       // the same
                    else                // oops, a different node with the same name
                        LogicError("NDLComputationNetwork: multiple nodes with the same NodeName()");
                // If node derives from MustFinalizeInit() then it has unresolved inputs. Resolve them now.
                // This may generate a whole new load of nodes, including nodes which in turn have late init.
                // TODO: think this through whether it may generate circular references nevertheless
                let mustFinalizeInit = dynamic_pointer_cast<MustFinalizeInit>(n);
                if (mustFinalizeInit)
                    mustFinalizeInit->FinalizeInit();
                // TODO: ...can we do stuff like propagating dimensions here? Or still too early?
                // traverse children: append them to the end of the work list
                let children = n->GetChildren();
                for (auto c : children)
                {
                    workList.push_back(c);  // (we could check whether c is in 'nodes' here to optimize, but this way it is cleaner)
                    allChildren.insert(c);  // also keep track of all children, for computing the 'outputs' set below
                }
            }
            // build sets of special nodes
            // TODO: figure out the rule. This is somehow based on the tags.
            for (auto iter : m_nameToNodeMap)
            {
                let n = iter.second;
                //if (n->GetChildren().empty())
                //{
                //    if (dynamic_pointer_cast<InputValue>(n))
                //        inputs.insert(n);
                //    else if (dynamic_pointer_cast<LearnableParameter>(n))
                //        parameters.insert(n);
                //    else
                //        LogicError("ComputationNetwork: found child-less node that is neither InputValue nor LearnableParameter");
                //}
                if (allChildren.find(n) == allChildren.end())
                    outputs.insert(n);
            }
            ///*HasToString::*/ wstring ToString() const
            //{
            wstring args;
            bool first = true;
            for (auto & iter : m_nameToNodeMap)
            {
                let node = iter.second;
                if (first)
                    first = false;
                else
                    args.append(L"\n");
                args.append(node->ToString());
            }
            fprintf(stderr, "ExperimentalComputationNetwork = [\n%ls\n]\n", NestString(args, L'[', true, ']').c_str());
            //return L"NDLComputationNetwork " + NestString(args, L'[', true, ']');
            //}
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
        template<>
        static shared_ptr<Object> MakeRuntimeObject<ComputationNode<ElemType>>(const IConfigRecordPtr configp)
        {
            let & config = *configp;
            wstring nodeType = config[L"class"];
            let inputs = GetInputs(config);
            DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
            auto node = ComputationNetwork<ElemType>::NewStandardNode(nodeType, deviceId, L"placeholder");   // name will be overwritten by caller upon return (TODO: fix this here? pass expression name in?)
            node->AttachInputs(inputs); // TODO: where to check the number of inputs?
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
        rtInfo.construct = [](const IConfigRecordPtr config)    // lambda to construct--this lambda can construct both the <float> and the <double> variant based on config parameter 'precision'
        {
            wstring precision = (*config)[L"precision"];           // dispatch on ElemType
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

}}}}

namespace Microsoft { namespace MSR { namespace CNTK {

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
                + wstrprintf(L"deviceId = %d ; precision = '%s' ; network = new ExperimentalComputationNetwork ", (int)m_deviceId, typeid(ElemType).name())  // TODO: check if typeid needs postprocessing
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
