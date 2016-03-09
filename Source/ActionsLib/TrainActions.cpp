//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TrainActions.cpp -- CNTK training-related actions
//

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "stdafx.h"
#include "Basics.h"
#include "Actions.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"
#include "SynchronousExecutionEngine.h"
#include "ModelEditLanguage.h"
#include "SGD.h"
#include "Config.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "BestGpu.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include <memory>

#ifndef let
#define let const auto
#endif

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

// ===========================================================================
// DoTrain() - implements CNTK "train" command
// ===========================================================================

// TODO: decide where these should go. Also, do we need three variables?
//extern wstring standardFunctions;
//extern wstring commonMacros;
//extern wstring computationNodes;

// helper that returns 'float' or 'double' depending on ElemType
template <class ElemType> static const wchar_t* ElemTypeName();
template <> /*static*/ const wchar_t* ElemTypeName<float>()  { return L"float"; }
template <> /*static*/ const wchar_t* ElemTypeName<double>() { return L"double"; }

function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ScriptableObjects::IConfigRecord& config)
{
    // createNetwork() is a BrainScript lambda that creates the model
    // We create a C++ wrapper around it, which we then pass to Train().
    auto createNetworkConfigLambda = config[L"createNetwork"].AsPtr<ScriptableObjects::ConfigLambda>();
    return [createNetworkConfigLambda](DEVICEID_TYPE /*deviceId*/)
    {
        // execute the lambda
        vector<ScriptableObjects::ConfigValuePtr> args; // this lambda has no arguments
        ScriptableObjects::ConfigLambda::NamedParams namedArgs;
        let netValue = createNetworkConfigLambda->Apply(move(args), move(namedArgs), L"BuildNetworkFromDescription");
        // typecast the result to the desired type
        return netValue.AsPtr<ComputationNetwork>();
    };
}
function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ConfigParameters&)
{
    NOT_IMPLEMENTED;
} // old CNTK config does not support lambdas

// function to create an object of a certain type, using both old CNTK config and BrainScript
template <class C>
shared_ptr<C> CreateObject(const ScriptableObjects::IConfigRecord& config, const wchar_t* id)
{
    // TODO: CNTK config added "traceLevel = 0" to 'config'. In BS, we cannot do that (IConfigRecord is immutable). Solution: Just say "traceLevel = 0" in the BS macros for readers.
    return config[id].AsPtr<C>(); // BS instantiates this object through this call
}
template <class C>
shared_ptr<C> CreateObject(const ConfigParameters& config, const wchar_t* id)
{
    ConfigParameters readerConfig(config(id));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0")); // TODO: fix this by adding it to all config blocks. Easy to fix in BS as 'config with [ traceLevel = 0 ]'.
    return make_shared<C>(readerConfig);                           // old CNTK config specifies a dictionary which then must be explicitly instantiated
}

template <class ConfigRecordType, typename ElemType>
void DoTrain(const ConfigRecordType& config)
{
    bool makeMode = config(L"makeMode", true);
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    // determine the network-creation function
    // We have several ways to create that network.
    function<ComputationNetworkPtr(DEVICEID_TYPE)> createNetworkFn;

    if (config.Exists(L"createNetwork"))
    {
        createNetworkFn = GetCreateNetworkFn(config); // (we need a separate function needed due to template code)
    }
    else if (config.Exists(L"SimpleNetworkBuilder"))
    {
        const ConfigRecordType& simpleNetworkBuilderConfig(config(L"SimpleNetworkBuilder"));
        auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(simpleNetworkBuilderConfig); // parses the configuration and stores it in the SimpleNetworkBuilder object
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription()); // this operates based on the configuration saved above
        };
    }
    // legacy NDL
    else if (config.Exists(L"NDLNetworkBuilder"))
    {
        const ConfigRecordType& ndlNetworkBuilderConfig(config(L"NDLNetworkBuilder"));
        shared_ptr<NDLBuilder<ElemType>> netBuilder = make_shared<NDLBuilder<ElemType>>(ndlNetworkBuilderConfig);
        createNetworkFn = [netBuilder](DEVICEID_TYPE deviceId)
        {
            return shared_ptr<ComputationNetwork>(netBuilder->BuildNetworkFromDescription());
        };
    }
    // legacy test mode for BrainScript. Will go away once we fully integrate with BS.
    else if (config.Exists(L"BrainScriptNetworkBuilder") || config.Exists(L"ExperimentalNetworkBuilder" /*legacy name*/))
    {
        // We interface with outer old CNTK config by taking the inner part, which we get as a string, as BrainScript.
        // We prepend a few standard definitions, and also definition of deviceId and precision, which all objects will pull out again when they are being constructed.
        // BUGBUG: We are not getting TextLocations right in this way! Do we need to inject location markers into the source? Moot once we fully switch to BS
        wstring sourceCode = config.Exists(L"BrainScriptNetworkBuilder") ? config(L"BrainScriptNetworkBuilder") : config(L"ExperimentalNetworkBuilder");
        auto configDirs = ConfigParameters::GetBrainScriptNetworkBuilderIncludePaths();
        let expr = BS::ParseConfigDictFromString(L"include \'cntk.core.bs\'"     // Note: Using lowercase here to match the Linux name of the CNTK exe.
                                                 + msra::strfun::wstrprintf(L"deviceId = %d ; precision = '%ls' ; network = new ComputationNetwork ", (int)deviceId, ElemTypeName<ElemType>())
                                                 + sourceCode,      // source code has the form [ ... ] with brackets in the string
                                                 move(configDirs)); // set include paths to all paths that configs were read from; no additional configurable include paths are supported by BrainScriptNetworkBuilder
        createNetworkFn = [expr](DEVICEID_TYPE /*deviceId*/)
        {
            // evaluate the parse tree, particularly the top-level field 'network'
            // Evaluating it will create the network.
            let object = EvaluateField(expr, L"network");                   // this comes back as a BS::Object
            let network = dynamic_pointer_cast<ComputationNetwork>(object); // cast it
            if (!network)
                LogicError("BuildNetworkFromDescription: ComputationNetwork not what it was meant to be");
            // success
            return network;
        };
    }
    else
    {
        RuntimeError("No network builder found in the config file. NDLNetworkBuilder or SimpleNetworkBuilde must be specified");
    }

    auto dataReader = CreateObject<DataReader>(config, L"reader");

    shared_ptr<DataReader> cvDataReader;
    if (config.Exists(L"cvReader"))
        cvDataReader = CreateObject<DataReader>(config, L"cvReader");

    shared_ptr<SGD<ElemType>> optimizer;
    if (config.Exists(L"optimizer"))
    {
        optimizer = CreateObject<SGD<ElemType>>(config, L"optimizer");
    }
    else // legacy CNTK config syntax: needs a record called 'SGD'
    {
        const ConfigRecordType& configSGD(config(L"SGD"));
        optimizer = make_shared<SGD<ElemType>>(configSGD);
    }

    optimizer->Train(createNetworkFn, deviceId, dataReader.get(), cvDataReader.get(), makeMode);
}

namespace Microsoft { namespace MSR { namespace ScriptableObjects {

using namespace Microsoft::MSR::CNTK;

// -----------------------------------------------------------------------
// register ComputationNode with the ScriptableObject system
// -----------------------------------------------------------------------

class TrainAction
{
};
template <>
shared_ptr<Object> MakeRuntimeObject<TrainAction>(const IConfigRecordPtr configp)
{
    const IConfigRecord& config = *configp;
    wstring precision = config[L"precision"]; // dispatch on ElemType
    if (precision == L"float")
        DoTrain<IConfigRecord, float>(config);
    else if (precision == L"double")
        DoTrain<IConfigRecord, double>(config);
    else
        RuntimeError("invalid value '%ls' for 'precision', must be 'float' or 'double'", precision.c_str());

    return make_shared<Object>(); // return a dummy object
}

// register ComputationNode with the ScriptableObject system
ScriptableObjects::ConfigurableRuntimeTypeRegister::Add<TrainAction> registerTrainAction(L"TrainAction");
}}}

template void DoTrain<ScriptableObjects::IConfigRecord, float>(const ScriptableObjects::IConfigRecord& config);
template void DoTrain<ScriptableObjects::IConfigRecord, double>(const ScriptableObjects::IConfigRecord& config);
template void DoTrain<ConfigParameters, float>(const ConfigParameters& config);
template void DoTrain<ConfigParameters, double>(const ConfigParameters& config);

// ===========================================================================
// DoAdapt() - implements CNTK "adapt" command
// ===========================================================================

template <typename ElemType>
void DoAdapt(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    ConfigParameters configSGD(config(L"SGD"));
    bool makeMode = config(L"makeMode", "true");

    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    auto dataReader = make_shared<DataReader>(readerConfig);

    shared_ptr<DataReader> cvDataReader;
    ConfigParameters cvReaderConfig(config(L"cvReader", L""));

    if (cvReaderConfig.size() != 0)
    {
        cvReaderConfig.Insert("traceLevel", config(L"traceLevel", "0"));
        cvDataReader = make_shared<DataReader>(cvReaderConfig);
    }

    wstring origModelFileName = config(L"origModelFileName", L"");
    wstring refNodeName = config(L"refNodeName", L"");

    SGD<ElemType> sgd(configSGD);

    sgd.Adapt(origModelFileName, refNodeName, dataReader.get(), cvDataReader.get(), deviceId, makeMode);
}

template void DoAdapt<float>(const ConfigParameters& config);
template void DoAdapt<double>(const ConfigParameters& config);

// ===========================================================================
// DoEdit() - implements CNTK "edit" command
// ===========================================================================

template <typename ElemType>
void DoEdit(const ConfigParameters& config)
{
    wstring editPath = config(L"editPath");
    wstring ndlMacros = config(L"ndlMacros", "");
    NDLScript<ElemType> ndlScript;
    if (!ndlMacros.empty())
    {
        ndlScript.LoadConfigFile(ndlMacros);
    }
    MELScript<ElemType> melScript;
    melScript.LoadConfigFileAndResolveVariables(editPath, config);
}

template void DoEdit<double>(const ConfigParameters& config);
template void DoEdit<float>(const ConfigParameters& config);
