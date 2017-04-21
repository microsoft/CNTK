//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalActions.cpp -- CNTK evaluation-related actions
//

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "stdafx.h"
#include "Basics.h"
#include "Actions.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "Config.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "Criterion.h"
#include "BestGpu.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"

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

bool GetDistributedMBReadingDefaultValue(const ConfigParameters& config, const IDataReader& reader)
{
    // Return 'true' if we're running a parallel training with a v2 reader, 'false' otherwise.
    return (MPIWrapper::GetInstance() != nullptr && !reader.IsLegacyReader());
}

// ===========================================================================
// DoEvalBase() - implements CNTK "eval" command
// ===========================================================================

template <typename ElemType>
static void DoEvalBase(const ConfigParameters& config, IDataReader& reader)
{
    //DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", 0);
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");
    size_t firstMBsToShowResult = config(L"firstMBsToShowResult", "0");
    size_t maxSamplesInRAM = config(L"maxSamplesInRAM", (size_t)SIZE_MAX);
    size_t numSubminiBatches = config(L"numSubminibatches", (size_t)1);

    bool enableDistributedMBReading = config(L"distributedMBReading", GetDistributedMBReadingDefaultValue(config, reader));

    vector<wstring> evalNodeNamesVector;

    let net = GetModelFromConfig<ConfigParameters, ElemType>(config, L"evalNodeNames", evalNodeNamesVector);

    // set tracing flags
    net->EnableNodeTracing(config(L"traceNodeNamesReal",     ConfigParameters::Array(stringargvector())),
                           config(L"traceNodeNamesCategory", ConfigParameters::Array(stringargvector())),
                           config(L"traceNodeNamesSparse",   ConfigParameters::Array(stringargvector())));

    SimpleEvaluator<ElemType> eval(net, MPIWrapper::GetInstance(), enableDistributedMBReading, numMBsToShowResult, 
                                   firstMBsToShowResult, traceLevel, maxSamplesInRAM, numSubminiBatches);
    eval.Evaluate(&reader, evalNodeNamesVector, mbSize[0], epochSize);
}

template <typename ElemType>
void DoEval(const ConfigParameters& config)
{
    // test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));
    if (!readerConfig.ExistsCurrent(L"randomize"))
    {
        readerConfig.Insert("randomize", "None");
    }

    DataReader testDataReader(readerConfig);
    DoEvalBase<ElemType>(config, testDataReader);
}

template void DoEval<double>(const ConfigParameters& config);
template void DoEval<float>(const ConfigParameters& config);

// ===========================================================================
// DoCrossValidate() - implements CNTK "cv" command
// ===========================================================================

template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config)
{
    // test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    ConfigArray cvIntervalConfig = config(L"crossValidationInterval");
    intargvector cvInterval = cvIntervalConfig;

    size_t sleepSecondsBetweenRuns = config(L"sleepTimeBetweenRuns", "0");

    int traceLevel = config(L"traceLevel", 0);
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");
    size_t firstMBsToShowResult = config(L"firstMBsToShowResult", "0");
    size_t maxSamplesInRAM    = config(L"maxSamplesInRAM", (size_t)SIZE_MAX);
    size_t numSubminiBatches  = config(L"numSubminibatches", (size_t)1);

    ConfigArray evalNodeNames = config(L"evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    std::vector<std::wstring> cvModels;

    DataReader cvDataReader(readerConfig);

    bool enableDistributedMBReading = config(L"distributedMBReading", GetDistributedMBReadingDefaultValue(config, cvDataReader));

    vector<double> minErrors;
    vector<int>    minErrIds;

    bool finalModelEvaluated = false;
    for (size_t i = cvInterval[0]; i <= cvInterval[2]; i += cvInterval[1])
    {
        wstring cvModelPath = msra::strfun::wstrprintf(L"%ls.%lld", modelPath.c_str(), i);

        if (!fexists(cvModelPath))
        {
            LOGPRINTF(stderr, "Model %ls does not exist.\n", cvModelPath.c_str());
            if (finalModelEvaluated || !fexists(modelPath))
                continue; // file missing
            else
            {
                cvModelPath = modelPath;
                finalModelEvaluated = true;
            }
        }

        cvModels.push_back(cvModelPath);
        auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, cvModelPath);
        // BUGBUG: ^^ Should use GetModelFromConfig()

        SimpleEvaluator<ElemType> eval(net, MPIWrapper::GetInstance(), enableDistributedMBReading, numMBsToShowResult,
            firstMBsToShowResult, traceLevel, maxSamplesInRAM, numSubminiBatches);

        // Get requsted eval node names actually present in current network.
        // If no nodes are requested, this will return all eval and criterion nodes in current network.
        const vector<ComputationNodeBasePtr> evalNodes = net->GetEvalNodesWithName(evalNodeNamesVector);
        vector<wstring> evalNodeNamesForCurrentNet(evalNodes.size());
        transform(evalNodes.begin(), evalNodes.end(), evalNodeNamesForCurrentNet.begin(),
            [](const ComputationNodeBasePtr& node) { return node->GetName(); });

        if (evalNodeNamesVector.empty())
        {
            // If no nodes are requested currently, it means that this is the first network that we are
            // evaluating. We set requested nodes to all eval and criterion nodes from current network.
            // These will be requested for this and all subsequent models.
            evalNodeNamesVector = evalNodeNamesForCurrentNet;
        }
        else
        {
            // Verify that current model has all requested nodes.
            if (evalNodeNamesVector != evalNodeNamesForCurrentNet)
            {
                LogicError("Model %ls does not have all requested evaluation nodes.", cvModelPath.c_str());
            }
        }

        LOGPRINTF(stderr, "Model %ls --> \n", cvModelPath.c_str());
        auto evalErrors = eval.Evaluate(&cvDataReader, evalNodeNamesVector, mbSize[0], epochSize);

        // Update best model.
        for (int j = 0; j < evalErrors.size(); ++j)
        {
            if (cvModels.size() == 1 || evalErrors[j].Average() < minErrors[j])
            {
                minErrors[j] = evalErrors[j].Average();
                minErrIds[j] = cvModels.size() - 1;
            }
            LOGPRINTF(stderr, "Based on %ls: Best model so far = %ls with min err %.8g\n",
                evalNodeNamesVector[i].c_str(), cvModels[minErrIds[i]].c_str(), minErrors[i]);
        }

        ::Sleep(1000 * sleepSecondsBetweenRuns);
    }

    if (cvModels.size() == 0)
        LogicError("No model is evaluated.");

    // Flag which indicates if we need to save best models for each evaluation metric
    // under a new name. For example, these can be used by other tasks that follow
    // this one.
    const bool saveBestModelPerCriterion = config(L"saveBestEpochPerCriterion", false);

    LOGPRINTF(stderr, "Best models:\n");
    LOGPRINTF(stderr, "------------\n");
    for (int i = 0; i < minErrors.size(); ++i)
    {
        LOGPRINTF(stderr, "Based on %ls: Best model = %ls with min err %.8g\n",
            evalNodeNamesVector[i].c_str(), cvModels[minErrIds[i]].c_str(), minErrors[i]);

        // Save best model for each criterion if we are required to do so, and if we
        // are the main worker (in case of distributed execution).
        if (saveBestModelPerCriterion && 
            (MPIWrapper::GetInstance() == nullptr || MPIWrapper::GetInstance()->IsMainNode()))
        {
            const wstring modelNameOriginal = cvModels[minErrIds[i]];
            const wstring modelNameCopied = modelPath + L"_" + evalNodeNamesVector[i];
            copyOrDie(modelNameOriginal, modelNameCopied);
            LOGPRINTF(stderr, "Copying best model for %ls: %ls -> %ls\n",
                evalNodeNamesVector[i].c_str(), modelNameOriginal.c_str(), modelNameCopied.c_str());
        }
    }

}

template void DoCrossValidate<float>(const ConfigParameters& config);
template void DoCrossValidate<double>(const ConfigParameters& config);

// ===========================================================================
// DoWriteOutput() - implements CNTK "write" command
// ===========================================================================

template <typename ElemType>
void DoWriteOutput(const ConfigParameters& config)
{
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("randomize", "None"); // we don't want randomization when output results

    DataReader testDataReader(readerConfig);

    ConfigArray minibatchSize = config(L"minibatchSize", "2048");
    intargvector mbSize = minibatchSize;

    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    vector<wstring> outputNodeNamesVector;

    let net = GetModelFromConfig<ConfigParameters, ElemType>(config, L"outputNodeNames", outputNodeNamesVector);

    // set tracing flags
    net->EnableNodeTracing(config(L"traceNodeNamesReal",     ConfigParameters::Array(stringargvector())),
                           config(L"traceNodeNamesCategory", ConfigParameters::Array(stringargvector())),
                           config(L"traceNodeNamesSparse",   ConfigParameters::Array(stringargvector())));

    SimpleOutputWriter<ElemType> writer(net, 1);

    if (config.Exists("writer"))
    {
        ConfigParameters writerConfig(config(L"writer"));
        bool writerUnittest = writerConfig(L"unittest", "false");
        DataWriter testDataWriter(writerConfig);
        writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize, writerUnittest);
    }
    else if (config.Exists("outputPath"))
    {
        wstring outputPath = config(L"outputPath");
        bool writeSequenceKey = config(L"writeSequenceKey", false);
        WriteFormattingOptions formattingOptions(config);
        bool nodeUnitTest = config(L"nodeUnitTest", "false");
        writer.WriteOutput(testDataReader, mbSize[0], outputPath, outputNodeNamesVector, formattingOptions, epochSize, nodeUnitTest, writeSequenceKey);
    }
    else
        InvalidArgument("write command: You must specify either 'writer'or 'outputPath'");
}

template void DoWriteOutput<float>(const ConfigParameters& config);
template void DoWriteOutput<double>(const ConfigParameters& config);
