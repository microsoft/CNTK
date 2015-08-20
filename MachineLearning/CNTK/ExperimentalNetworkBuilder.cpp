// ExperimentalNetworkBuilder.h -- interface to new version of NDL (and config) parser  --fseide

#define _CRT_NONSTDC_NO_DEPRECATE   // make VS accept POSIX functions without _
#define _CRT_SECURE_NO_WARNINGS     // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ExperimentalNetworkBuilder.h"
#include "ConfigEvaluator.h"

#include "ComputationNetwork.h"

#include <memory>

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
        L"Mean(z, tag='') = new ComputationNode [ class = 'MeanNode' ; inputs = z ; optionalTag = 'tag' ]\n"
        L"InvStdDev(z, tag='') = new ComputationNode [ class = 'InvStdDevNode' ; inputs = z ; optionalTag = 'tag' ]\n"
        L"PerDimMeanVarNormalization(feat,mean,invStdDev, tag='') = new ComputationNode [ class = 'PerDimMeanVarNormalizationNode' ; inputs = feat:mean:invStdDev ; optionalTag = 'tag' ]\n"
        L"Parameter(outD, inD/*, tag=''*/) = new ComputationNode [ class = 'LearnableParameterNode' ; outDim = outD ; inDim = inD /*; optionalTag = 'tag'*/ ]\n"
        L"Input(dim) = Parameter(dim,1/*,tag='features'*/)   // TODO: for now \n"
        L"RowSlice(firstRow, rows, features, tag='') = new ComputationNode [ class = 'RowSliceNode' ; inputs = features ; first = firstRow ; num = rows ; optionalTag = 'tag' ]\n"
        L"Delay(in, delay, tag='') = new ComputationNode [ class = 'DelayNode' ; input = in ; deltaT = -delay ; optionalTag = 'tag' ]\n"
        L"Sigmoid(z, tag='') = new ComputationNode [ class = 'SigmoidNode' ; inputs = z ; optionalTag = 'tag' ]\n"
        L"Log(z, tag='') = new ComputationNode [ class = 'LogNode' ; inputs = z ; optionalTag = 'tag' ]\n"
        L"CrossEntropyWithSoftmax(labels, outZ, tag='') = new ComputationNode [ class = 'CrossEntropyWithSoftmaxNode' ; inputs = labels:outZ ; optionalTag = 'tag' ]\n"
        L"ErrorPrediction(labels, outZ, tag='') = new ComputationNode [ class = 'ErrorPredictionNode' ; inputs = labels:outZ ; optionalTag = 'tag' ]\n"
        ;

    wstring commonMacros =  // TODO: rename rows and cols to inDim and outDim or vice versa, whichever it is
        L"BFF(in, rows, cols) = [ B = Parameter(rows, 1/*init = fixedvalue, value = 0*/) ; W = Parameter(rows, cols) ; z = W*in+B ] \n"
        L"SBFF(in, rows, cols) = [ Eh = Sigmoid(BFF(in, rows, cols).z) ] \n "
        L"MeanVarNorm(feat) = PerDimMeanVarNormalization(feat, Mean(feat), InvStdDev(feat)) \n"
        L"LogPrior(labels) = Log(Mean(labels)) \n"
        ;

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
    shared_ptr<ComputationNetwork<ElemType>> InitComputationNetwork(const ConfigRecord & config, shared_ptr<ComputationNetwork<ElemType>> net)
    {
        config;
        return net;
    }

    // create a ComputationNetwork<ElemType> from a config--this implements "new ExperimentalComputationNetwork [ ... ]" in the added config snippet above
    shared_ptr<Object> MakeExperimentalComputationNetwork(const ConfigRecord & config)
    {
        DEVICEID_TYPE deviceId = (DEVICEID_TYPE)(int)config[L"deviceId"];
        wstring precision = config[L"precision"];
        if (precision == L"float")
            return InitComputationNetwork(config, make_shared<ComputationNetwork<float>>(deviceId));
        else if (precision == L"double")
            return InitComputationNetwork(config, make_shared<ComputationNetwork<double>>(deviceId));
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
