// EvalActions.cpp -- CNTK evaluation-related actions
//
// <copyright file="EvalActions.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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
#include "MultiNetworksSGD.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "MultiNetworksEvaluator.h"
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
// DoEvalBase() - implements CNTK "eval" command
// ===========================================================================

template <typename ElemType>
static void DoEvalBase(const ConfigParameters& config, IDataReader<ElemType>& reader)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config(L"evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);
    eval.Evaluate(&reader, evalNodeNamesVector, mbSize[0], epochSize);
}

template <typename ElemType>
void DoEval(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBase(config, testDataReader);
}

template void DoEval<double>(const ConfigParameters& config);
template void DoEval<float>(const ConfigParameters& config);

// ===========================================================================
// DoCrossValidate() - implements CNTK "cv" command
// ===========================================================================

template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config)
{
    //test
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

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    ConfigArray evalNodeNames = config(L"evalNodeNames", "");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    std::vector<std::vector<double>> cvErrorResults;
    std::vector<std::wstring> cvModels;

    DataReader<ElemType> cvDataReader(readerConfig);

    bool finalModelEvaluated = false;
    for (size_t i = cvInterval[0]; i <= cvInterval[2]; i += cvInterval[1])
    {
        wstring cvModelPath = msra::strfun::wstrprintf(L"%ls.%lld", modelPath.c_str(), i);

        if (!fexists(cvModelPath))
        {
            fprintf(stderr, "model %ls does not exist.\n", cvModelPath.c_str());
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

        SimpleEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);

        fprintf(stderr, "model %ls --> \n", cvModelPath.c_str());
        auto evalErrors = eval.Evaluate(&cvDataReader, evalNodeNamesVector, mbSize[0], epochSize);
        cvErrorResults.push_back(evalErrors);

        ::Sleep(1000 * sleepSecondsBetweenRuns);
    }

    //find best model
    if (cvErrorResults.size() == 0)
    {
        LogicError("No model is evaluated.");
    }

    std::vector<double> minErrors;
    std::vector<int> minErrIds;
    std::vector<double> evalErrors = cvErrorResults[0];
    for (int i = 0; i < evalErrors.size(); ++i)
    {
        minErrors.push_back(evalErrors[i]);
        minErrIds.push_back(0);
    }

    for (int i = 0; i < cvErrorResults.size(); i++)
    {
        evalErrors = cvErrorResults[i];
        for (int j = 0; j < evalErrors.size(); j++)
        {
            if (evalErrors[j] < minErrors[j])
            {
                minErrors[j] = evalErrors[j];
                minErrIds[j] = i;
            }
        }
    }

    fprintf(stderr, "Best models:\n");
    fprintf(stderr, "------------\n");
    for (int i = 0; i < minErrors.size(); ++i)
    {
        fprintf(stderr, "Based on Err[%d]: Best model = %ls with min err %.8g\n", i, cvModels[minErrIds[i]].c_str(), minErrors[i]);
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
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));
    readerConfig.Insert("randomize", "None"); //we don't want randomization when output results

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "2048");
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;

    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames", "");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    SimpleOutputWriter<ElemType> writer(net, 1);

    if (config.Exists("writer"))
    {
        ConfigParameters writerConfig(config(L"writer"));
        bool bWriterUnittest = writerConfig(L"unittest", "false");
        DataWriter<ElemType> testDataWriter(writerConfig);
        writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize, bWriterUnittest);
    }
    else if (config.Exists("outputPath"))
    {
        wstring outputPath = config(L"outputPath"); // crashes if no default given?
        writer.WriteOutput(testDataReader, mbSize[0], outputPath, outputNodeNamesVector, epochSize);
    }
    //writer.WriteOutput(testDataReader, mbSize[0], testDataWriter, outputNodeNamesVector, epochSize);
}

template void DoWriteOutput<float>(const ConfigParameters& config);
template void DoWriteOutput<double>(const ConfigParameters& config);
