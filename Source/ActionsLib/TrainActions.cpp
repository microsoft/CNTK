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
    if (!readerConfig.ExistsCurrent("traceLevel")) // do not overwrite "traceLevel" if it's already present
    {
        readerConfig.Insert("traceLevel", config(L"traceLevel", "0")); // TODO: fix this by adding it to all config blocks. Easy to fix in BS as 'config with [ traceLevel = 0 ]'.
    }
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

    createNetworkFn = GetNetworkFactory<ConfigRecordType, ElemType>(config);

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

    optimizer->InitMPI(MPIWrapper::GetInstance());
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

    sgd.InitMPI(MPIWrapper::GetInstance());
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
