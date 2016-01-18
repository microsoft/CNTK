// EsotericActions.cpp -- CNTK actions that are deprecated
//
// <copyright file="EsotericActions.cpp" company="Microsoft">
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
// DoConvertFromDbn() - implements CNTK "convertdbn" command
// ===========================================================================

template <typename ElemType>
void DoConvertFromDbn(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring dbnModelPath = config(L"dbnModelPath");

    auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(config);
    ComputationNetworkPtr net = netBuilder->BuildNetworkFromDbnFile(dbnModelPath);
    net->Save(modelPath);
}

template void DoConvertFromDbn<float>(const ConfigParameters& config);
template void DoConvertFromDbn<double>(const ConfigParameters& config);

// ===========================================================================
// DoEvalUnroll() - implements CNTK "testunroll" command
// ===========================================================================

// Special early implementation of RNNs by emulating them as a DNN.
// The code is very restricted to simple RNNs.
// The idea can be used for more complicated network but need to know which nodes are stateful or time-dependent so that unroll is done in a correct way to represent recurrent networks.
// TODO: can probably be removed.
template <typename ElemType>
void DoEvalUnroll(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig(config(L"reader"));
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DEVICEID_TYPE deviceId = DeviceFromConfig(config);
    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }
    wstring modelPath = config(L"modelPath");
    intargvector mbSize = minibatchSize;
    wstring path2EvalResults = config(L"path2EvalResults", L"");

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    MultiNetworksEvaluator<ElemType> eval(net);
    double evalEntropy;
    eval.EvaluateUnroll(&testDataReader, mbSize[0], evalEntropy, path2EvalResults == L"" ? nullptr : path2EvalResults.c_str(), epochSize);
}

template void DoEvalUnroll<float>(const ConfigParameters& config);
template void DoEvalUnroll<double>(const ConfigParameters& config);

// ===========================================================================
// DoEncoderDecoder() - implements CNTK "trainEncoderDecoder" command
// ===========================================================================

/**
This implements sequence to sequence translation paper in
http://arxiv.org/pdf/1409.3215.pdf

*/
template <typename ElemType>
void DoEncoderDecoder(const ConfigParameters& config)
{
    vector<IComputationNetBuilder<ElemType>*> netBuilders;
    vector<IDataReader<ElemType>*> trainDataReader;
    vector<IDataReader<ElemType>*> validationDataReader;

    ConfigParameters configSGD = config(L"SGD");
    bool makeMode = config(L"makeMode", "true");
    IComputationNetBuilder<ElemType>* encoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* decoderNetBuilder = NULL;

    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType>* encoderDataReader = new DataReader<ElemType>(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    DataReader<ElemType>* decoderDataReader = new DataReader<ElemType>(decoderReaderConfig);

    ConfigParameters cvEncoderReaderConfig = config(L"encoderCVReader");
    DataReader<ElemType>* cvEncoderDataReader = new DataReader<ElemType>(cvEncoderReaderConfig);

    ConfigParameters cvDecoderReaderConfig = config(L"decoderCVReader");
    DataReader<ElemType>* cvDecoderDataReader = new DataReader<ElemType>(cvDecoderReaderConfig);

    if (config.Exists("EncoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"EncoderNetworkBuilder");
        encoderNetBuilder = (IComputationNetBuilder<ElemType>*) new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need encoder network");
    }

    if (config.Exists("DecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"DecoderNetworkBuilder");
        decoderNetBuilder = (IComputationNetBuilder<ElemType>*) new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    MultiNetworksSGD<ElemType> sgd(configSGD);

    sgd.InitTrainEncoderDecoderWithHiddenStates(configSGD);

    netBuilders.push_back(encoderNetBuilder);
    netBuilders.push_back(decoderNetBuilder);
    trainDataReader.push_back(encoderDataReader);
    trainDataReader.push_back(decoderDataReader);
    validationDataReader.push_back(cvEncoderDataReader);
    validationDataReader.push_back(cvDecoderDataReader);

    sgd.EncoderDecoder(netBuilders, (int) config(L"deviceId"), trainDataReader, validationDataReader, makeMode);

    delete encoderDataReader;
    delete decoderDataReader;
    delete cvEncoderDataReader;
    delete cvDecoderDataReader;
}

template void DoEncoderDecoder<float>(const ConfigParameters& config);
template void DoEncoderDecoder<double>(const ConfigParameters& config);

// ===========================================================================
// DoBidirectionEncoderDecoder() - implements CNTK "trainBidirectionEncoderDecoder" command
// ===========================================================================

/**
DoBidirecionEncoderDecoder
*/
template <typename ElemType>
void DoBidirectionEncoderDecoder(const ConfigParameters& config)
{

    ConfigParameters configSGD = config(L"SGD");
    bool makeMode = config(L"makeMode", "true");
    IComputationNetBuilder<ElemType>* encoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* forwardDecoderNetBuilder = NULL;
    IComputationNetBuilder<ElemType>* backwardDecoderNetBuilder = NULL;
    vector<IComputationNetBuilder<ElemType>*> netBuilders;
    vector<IDataReader<ElemType>*> trainDataReader;
    vector<IDataReader<ElemType>*> validationDataReader;

    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType>* encoderDataReader = new DataReader<ElemType>(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    DataReader<ElemType>* decoderDataReader = new DataReader<ElemType>(decoderReaderConfig);

    ConfigParameters backwardDecoderReaderConfig = config(L"backwardDecoderReader");
    DataReader<ElemType>* backwardDecoderDataReader = new DataReader<ElemType>(backwardDecoderReaderConfig);

    ConfigParameters cvEncoderReaderConfig = config(L"encoderCVReader");
    DataReader<ElemType>* cvEncoderDataReader = new DataReader<ElemType>(cvEncoderReaderConfig);

    ConfigParameters cvDecoderReaderConfig = config(L"decoderCVReader");
    DataReader<ElemType>* cvDecoderDataReader = new DataReader<ElemType>(cvDecoderReaderConfig);

    ConfigParameters cvBackwardDecoderReaderConfig = config(L"BackwardDecoderCVReader");
    DataReader<ElemType>* cvBackwardDecoderDataReader = new DataReader<ElemType>(cvBackwardDecoderReaderConfig);

    if (config.Exists("EncoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"EncoderNetworkBuilder");
        encoderNetBuilder = (IComputationNetBuilder<ElemType>*) new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
        LogicError("Need encoder network");

    if (config.Exists("DecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"DecoderNetworkBuilder");
        forwardDecoderNetBuilder = (IComputationNetBuilder<ElemType>*) new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    if (config.Exists("BackwardDecoderNetworkBuilder"))
    {
        ConfigParameters configSNB = config(L"BackwardDecoderNetworkBuilder");
        backwardDecoderNetBuilder = (IComputationNetBuilder<ElemType>*) new SimpleNetworkBuilder<ElemType>(configSNB);
    }
    else
    {
        LogicError("Need decoder networks");
    }

    MultiNetworksSGD<ElemType> sgd(configSGD);

    sgd.InitTrainEncoderDecoderWithHiddenStates(configSGD);

    netBuilders.push_back(encoderNetBuilder);
    netBuilders.push_back(forwardDecoderNetBuilder);
    netBuilders.push_back(backwardDecoderNetBuilder);
    trainDataReader.push_back(encoderDataReader);
    trainDataReader.push_back(decoderDataReader);
    trainDataReader.push_back(backwardDecoderDataReader);
    validationDataReader.push_back(cvEncoderDataReader);
    validationDataReader.push_back(cvDecoderDataReader);
    validationDataReader.push_back(cvBackwardDecoderDataReader);

    sgd.EncoderDecoder(netBuilders, (int) config(L"deviceId"), trainDataReader, validationDataReader, makeMode);

    delete encoderDataReader;
    delete decoderDataReader;
    delete cvEncoderDataReader;
    delete cvDecoderDataReader;
    delete backwardDecoderDataReader;
    delete cvBackwardDecoderDataReader;
}

template void DoBidirectionEncoderDecoder<float>(const ConfigParameters& config);
template void DoBidirectionEncoderDecoder<double>(const ConfigParameters& config);

// ===========================================================================
// DoEvalEncodingBeamSearchDecoding() - implements CNTK "testEncoderDecoder" command
// ===========================================================================

/**
Originally, this is for testing models trained using the sequence to sequence translation method below
http://arxiv.org/pdf/1409.3215.pdf
Later on, it is extended to be more general to include a sequence of network operations. 
*/
template <typename ElemType>
void DoEvalEncodingBeamSearchDecoding(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceId = DeviceFromConfig(config);

    vector<IDataReader<ElemType>*> readers;
    ConfigParameters readerConfig = config(L"encoderReader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> encoderReader(readerConfig);

    ConfigParameters decoderReaderConfig = config(L"decoderReader");
    decoderReaderConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> decoderReader(decoderReaderConfig);

    readers.push_back(&encoderReader);
    readers.push_back(&decoderReader);

    ConfigArray minibatchSize = config(L"minibatchSize", "40960");
    size_t epochSize = config(L"epochSize", "0");
    if (epochSize == 0)
    {
        epochSize = requestDataSize;
    }

    wstring encoderModelPath = config(L"encoderModelPath");
    wstring decoderModelPath = config(L"decoderModelPath");

    intargvector mbSize = minibatchSize;

    int traceLevel = config(L"traceLevel", "0");
    size_t numMBsToShowResult = config(L"numMBsToShowResult", "100");

    vector<ComputationNetworkPtr> nets;
    auto encoderNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, encoderModelPath, FileOptions::fileOptionsBinary, true);

    auto decoderNet = ComputationNetwork::CreateFromFile<ElemType>(deviceId, decoderModelPath, FileOptions::fileOptionsBinary, false, encoderNet.get());

    nets.push_back(encoderNet);
    nets.push_back(decoderNet);
    ConfigArray evalNodeNames = config(L"evalNodeNames");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ElemType beamWidth = config(L"beamWidth", "1");

    ConfigParameters writerConfig = config(L"writer");
    DataWriter<ElemType> testDataWriter(writerConfig);

    MultiNetworksEvaluator<ElemType> eval(decoderNet, numMBsToShowResult, traceLevel);
    eval.InitTrainEncoderDecoderWithHiddenStates(config);

    eval.EncodingEvaluateDecodingBeamSearch(nets, readers,
                                            testDataWriter, evalNodeNamesVector,
                                            outputNodeNamesVector,
                                            mbSize[0], beamWidth, epochSize);
}

template void DoEvalEncodingBeamSearchDecoding<float>(const ConfigParameters& config);
template void DoEvalEncodingBeamSearchDecoding<double>(const ConfigParameters& config);

// ===========================================================================
// DoBeamSearchDecoding() - implements CNTK "beamSearch" command
// ===========================================================================

template <typename ElemType>
static void DoEvalBeamSearch(const ConfigParameters& config, IDataReader<ElemType>& reader)
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

    auto net = ComputationNetwork::CreateFromFile<ElemType>(deviceId, modelPath);

    ConfigArray evalNodeNames = config(L"evalNodeNames");
    vector<wstring> evalNodeNamesVector;
    for (int i = 0; i < evalNodeNames.size(); ++i)
    {
        evalNodeNamesVector.push_back(evalNodeNames[i]);
    }

    ConfigArray outputNodeNames = config(L"outputNodeNames");
    vector<wstring> outputNodeNamesVector;
    for (int i = 0; i < outputNodeNames.size(); ++i)
    {
        outputNodeNamesVector.push_back(outputNodeNames[i]);
    }

    ElemType beamWidth = config(L"beamWidth", "1");

    ConfigParameters writerConfig = config(L"writer");
    DataWriter<ElemType> testDataWriter(writerConfig);

    MultiNetworksEvaluator<ElemType> eval(net, numMBsToShowResult, traceLevel);
    eval.BeamSearch(&reader, testDataWriter, evalNodeNamesVector, outputNodeNamesVector, mbSize[0], beamWidth, epochSize);
}

/**
This is beam search decoder.

Developed by Kaisheng Yao.

It is used in the following work:
K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion" in Interspeech 2015
*/
template <typename ElemType>
void DoBeamSearchDecoding(const ConfigParameters& config)
{
    //test
    ConfigParameters readerConfig = config(L"reader");
    readerConfig.Insert("traceLevel", config(L"traceLevel", "0"));

    DataReader<ElemType> testDataReader(readerConfig);

    DoEvalBeamSearch(config, testDataReader);
}

template void DoBeamSearchDecoding<float>(const ConfigParameters& config);
template void DoBeamSearchDecoding<double>(const ConfigParameters& config);
