// ExperimentalNetworkBuilder.h -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ExperimentalNetworkBuilder.h"
#include "ConfigEvaluator.h"

#include "ComputationNode.h"
#include "ComputationNetwork.h"

#include <memory>
#include <deque>
#include <set>
#include <string>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK { namespace Config {   // new config parsing lives in a sub-namespace, as to avoid conflict with existing ones which get implicitly pulled in by some headers we need

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
    shared_ptr<ComputationNetwork<ElemType>> /*ComputationNetworkPtr*/ CreateNetwork(const wstring & sourceCode, DEVICEID_TYPE deviceId, const wchar_t * precision)
    {
        // we pass deviceId and precision in as dictionary entries, which the constructor below will pull out again
        let expr = ParseConfigString(standardFunctions + computationNodes + commonMacros
                                     + wstrprintf(L"deviceId = %d ; precision = '%s' ; network = new ExperimentalComputationNetwork", (int)deviceId, precision)
                                     + sourceCode);
        let network = dynamic_pointer_cast<ComputationNetwork<ElemType>>(EvaluateField(expr, L"network"));
        return network;
    }

    // initialize a ComputationNetwork<ElemType> from a ConfigRecord
    template<typename ElemType>
    shared_ptr<ComputationNetwork<ElemType>> CreateComputationNetwork(const ConfigRecord & config)
    {
        DEVICEID_TYPE deviceId = -1; // (DEVICEID_TYPE)(int)config[L"deviceId"];
        auto net = make_shared<ComputationNetwork<ElemType>>(deviceId);

        typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;   // this is only needed in this experimental setup; will go away once this function becomes part of ComputationNetwork itself
        auto & m_nameToNodeMap = net->GetNameToNodeMap();

        deque<ComputationNodePtr> workList;
        // flatten the set of all nodes
        // we collect all ComputationNodes from the config; that's it
        for (auto & iter : config.GetMembers())
            if (iter.second.Is<ComputationNode<ElemType>>())
                workList.push_back((ComputationNodePtr)config[iter.first]);
        // process work list
        // Also call FinalizeInit where we must.
        set<ComputationNodePtr> inputs;     // all input nodes
        set<ComputationNodePtr> outputs;    // all output nodes
        set<ComputationNodePtr> parameters; // all parameter nodes
        set<ComputationNodePtr> allChildren;    // all nodes that are children of others (those that are not are output nodes)
        while (!workList.empty())
        {
            let n = workList.front();
            workList.pop_front();
            // add to set
            let res = m_nameToNodeMap.insert(make_pair(n->NodeName(), n));
            if (!res.second)        // not inserted: we already got this one
            if (res.first->second != n)
                LogicError("NDLComputationNetwork: multiple nodes with the same NodeName()");
            else
                continue;
            // If node derives from MustFinalizeInit() then it has unresolved ConfigValuePtrs. Resolve them now.
            // This may generate a whole new load of nodes, including nodes which in turn have late init.
            // TODO: think this through whether it may generate delays nevertheless
            let mustFinalizeInit = dynamic_pointer_cast<MustFinalizeInit>(n);
            if (mustFinalizeInit)
                mustFinalizeInit->FinalizeInit();
            // TODO: ...can we do stuff like propagating dimensions here? Or still too early?
            // get children
            // traverse children (i.e., append them to the work list)
            let children = n->GetChildren();
            for (auto c : children)
            {
                workList.push_back(c);  // (we could check whether c is in 'nodes' here to optimize, but this way it is cleaner)
                allChildren.insert(c);  // also keep track of all children, for computing the 'outputs' set below
            }
        }
        // build sets of special nodes
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

    // create a ComputationNetwork<ElemType> from a config--this implements "new ExperimentalComputationNetwork [ ... ]" in the added config snippet above
    shared_ptr<Object> MakeExperimentalComputationNetwork(const ConfigRecord & config)
    {
        wstring precision = config[L"precision"];   // TODO: we need to look those up while traversing upwards
        if (precision == L"float")
            return CreateComputationNetwork<float>(config);
        else if (precision == L"double")
            return CreateComputationNetwork<double>(config);
        else
            LogicError("MakeExperimentalComputationNetwork: precision must be 'float' or 'double'");
    }

    // initialize a ComputationNetwork<ElemType> from a ConfigRecord
    template<typename ElemType>
    shared_ptr<ComputationNode<ElemType>> CreateComputationNode(const ConfigRecord & config)
    {
        DEVICEID_TYPE deviceId = -1;// (DEVICEID_TYPE)(int)config[L"deviceId"];
        wstring classId = config[L"class"];
        auto node = make_shared<TimesNode<ElemType>>(deviceId);
        config;
        return node;
    }

    // create a ComputationNetwork<ElemType> from a config--this implements "new ExperimentalComputationNetwork [ ... ]" in the added config snippet above
    shared_ptr<Object> MakeExperimentalComputationNode(const ConfigRecord & config)
    {
        wstring precision = L"float"; // config[L"precision"];   // TODO: we need to look those up while traversing upwards
        if (precision == L"float")
            return CreateComputationNode<float>(config);
        else if (precision == L"double")
            return CreateComputationNode<double>(config);
        else
            LogicError("MakeExperimentalComputationNetwork: precision must be 'float' or 'double'");
    }

}}}}

namespace Microsoft { namespace MSR { namespace CNTK {

    // sorry for code dup--this will be made nicer when this gets fully integrated
    /*virtual*/ /*IComputationNetBuilder::*/ComputationNetwork<float>* ExperimentalNetworkBuilder<float>::BuildNetworkFromDescription(ComputationNetwork<float>*)
    {
        if (!m_net || m_net->GetTotalNumberOfNodes() < 1) //not built yet
            m_net = Config::CreateNetwork<float>(m_sourceCode, m_deviceId, L"float");
        m_net->ResetEvalTimeStamp();
        return m_net.get();
    }
    /*virtual*/ /*IComputationNetBuilder::*/ComputationNetwork<double>* ExperimentalNetworkBuilder<double>::BuildNetworkFromDescription(ComputationNetwork<double>*)
    {
        if (!m_net || m_net->GetTotalNumberOfNodes() < 1) //not built yet
            m_net = Config::CreateNetwork<double>(m_sourceCode, m_deviceId, L"float");
        m_net->ResetEvalTimeStamp();
        return m_net.get();
    }

}}}
