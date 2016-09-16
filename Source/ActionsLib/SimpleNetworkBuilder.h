//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "BestGpu.h"

#include "ComputationNetwork.h"
#include "Config.h"

// TODO: giving up moving stuff for now, running out of time. The following #includes should not be necessary once the hard-working code in here gets moved to .cpp
#include "InputAndParamNodes.h"

#include <stdexcept>
#include <regex>
#include <string>

#pragma warning(disable : 4661)

using namespace std; // TODO: ugh!

/// this is for sparse input, useful when input dimension is very large and sparse such as language modeling
/// to-do: need to use it guided by argument
#define SPARSE_INPUT

namespace Microsoft { namespace MSR { namespace CNTK {

#define MAX_DEPTH 20

// the standard network kinds that can be built with SimpleNetworkBuilder
enum StandardNetworkKind
{
    // basic
    FFDNNKind                  = 0,     // basic feed-forward
    RNNKind                    = 1,     // basic RNN
    LSTMKind                   = 2,     // basic LSTM
    // class-based
    ClassEntropyRNNKind        = 8,     // class-based RNN
    ClassLSTMNetworkKind       = 64,    // class-based LSTM
    // advanced
    LogBilinearNetworkKind     = 16,    // log-bilinear model for language modeling
    DNNLMNetworkKind           = 32,    // DNN-based LM
    NCELSTMNetworkKind         = 128,   // NCE LSTM
    ConditionalLSTMNetworkKind = 256,   // conditional LM for text generation
    CRFLSTMNetworkKind         = 512,   // sequential LSTM
};

enum class TrainingCriterion : int // TODO: camel-case these
{
    CrossEntropyWithSoftmax,
    CrossEntropy,
    SquareError,
    Logistic,
    ClassCrossEntropyWithSoftmax,
    NCECrossEntropyWithSoftmax,
    CRF,
    SequenceWithSoftmax,
    CTCwithSoftmax
};

enum class EvalCriterion : int
{
    CrossEntropyWithSoftmax,
    CrossEntropy,
    SquareError,
    Logistic,
    ClassificationError,
	PhoneError,
    ClassCrossEntropyWithSoftmax,
    NCECrossEntropyWithSoftmax,
    CRF,
    SequenceWithSoftmax,
    CTCwithSoftmax
};

TrainingCriterion ParseTrainingCriterionString(wstring s);
EvalCriterion ParseEvalCriterionString(wstring s);

template <class ElemType>
class SimpleNetworkBuilder
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

private:
    SimpleNetworkBuilder() // disable default constructor from being called
    {
    }

public:
    SimpleNetworkBuilder(const ConfigParameters& config)
        : m_net(nullptr)
    {
        Init(config);
    }
    SimpleNetworkBuilder(const ScriptableObjects::IConfigRecord&)
    {
        NOT_IMPLEMENTED;
    }

    // full parameter Init routine
    void Init(const intargvector& layerSizes, const TrainingCriterion trainCriterion, const EvalCriterion evalCriterion,
              DEVICEID_TYPE deviceId,
              int outputLayerSize = -1,
              const stringargvector nonLinearFunctions = L"Sigmoid",
              const bool addDropoutNodes = false,
              const bool uniformInit = true, const ElemType initValueScale = 1.0f,
              const bool applyMeanVarNorm = false, bool needPrior = false)
    {
        m_deviceId = deviceId;
        m_net = make_shared<ComputationNetwork>(m_deviceId);

        if (m_deviceId < 0)
            fprintf(stderr, "SimpleNetworkBuilder Using CPU\n");
        else
            fprintf(stderr, "SimpleNetworkBuilder Using GPU %d\n", m_deviceId);

        m_outputLayerSize = outputLayerSize;
        m_layerSizes = layerSizes;
        m_applyMeanVarNorm = applyMeanVarNorm;
        m_trainCriterion = trainCriterion;
        m_evalCriterion = evalCriterion;
        m_addDropoutNodes = addDropoutNodes;
        m_needPrior = needPrior;
        m_nonLinearFunctions = nonLinearFunctions;
        m_uniformInit = uniformInit;
        m_initValueScale = initValueScale;
        if (m_layerSizes.size() < 2)
            InvalidArgument("A network should have at least two layers (one input and one output)");
    }

    void InitAttentionNetworkConfig(const ConfigParameters& config)
    {
        m_auxFeatDim = config("auxfeatdim", "20");
    }

    virtual void InitRecurrentConfig(const ConfigParameters& config)
    {
        ConfigArray rLayerSizes = config("recurrentLayer", "");
        intargvector recurrentLayers = rLayerSizes;
        m_recurrentLayers = recurrentLayers;
        m_defaultHiddenActivity = config("defaultHiddenActivity", "0.1"); // TODO: spelling, should be -Activation
        ConfigArray str_rnnType = config("rnnType", L"SIMPLENET"); // TODO: camelCase

        m_maOrder = config("maOrder", "0");
        m_lookupTableOrder = config("lookupTableOrder", "0"); // TODO: What is this?

        ConfigArray sSizes = config("streamSizes", "");
        m_streamSizes = sSizes;
        sSizes = config("lookupTableOrderSizes", ""); // this allows having a multiple streams of inputs with
        // different lookuptable order sizes. the older one lookupTableOrder is still kept to have backward
        // support.
        m_lookupTabelOrderSizes = sSizes;

        m_labelEmbeddingSize   = config("labelEmbeddingSize",   "10");
        m_constForgetGateValue = config("constForgetGateValue", "false");
        m_constInputGateValue  = config("constInputGateValue",  "false");
        m_constOutputGateValue = config("constOutputGateValue", "false");

        m_forgetGateInitVal = config("forgetGateInitVal", "-1");
        m_inputGateInitVal  = config("inputGateInitVal",  "-1");
        m_outputGateInitVal = config("outputGateInitVal", "-1");

        m_sparse_input = config("sparseinput", "false");

        // TODO: use EqualCI(), and use camelCase, e.g. classLSTM
        stringargvector strType = str_rnnType;
             if (std::find(strType.begin(), strType.end(), L"SIMPLENET") != strType.end()) // TODO: camelCase
            m_standardNetworkKind = FFDNNKind;
        else if (std::find(strType.begin(), strType.end(), L"SIMPLERNN") != strType.end()) // TODO: camelCase
            m_standardNetworkKind = RNNKind;
        else if (std::find(strType.begin(), strType.end(), L"LSTM") != strType.end())
            m_standardNetworkKind = LSTMKind;
        else if (std::find(strType.begin(), strType.end(), L"CLASSLM") != strType.end()) // TODO: camelCase
            m_standardNetworkKind = ClassEntropyRNNKind;
        else if (std::find(strType.begin(), strType.end(), L"LBLM") != strType.end())
            m_standardNetworkKind = LogBilinearNetworkKind;
        else if (std::find(strType.begin(), strType.end(), L"NPLM") != strType.end())
            m_standardNetworkKind = DNNLMNetworkKind;
        else if (std::find(strType.begin(), strType.end(), L"CLASSLSTM") != strType.end()) // TODO: camelCase
            m_standardNetworkKind = ClassLSTMNetworkKind;
        else if (std::find(strType.begin(), strType.end(), L"NCELSTM") != strType.end())
            m_standardNetworkKind = NCELSTMNetworkKind;
        else if (std::find(strType.begin(), strType.end(), L"CLSTM") != strType.end())
            m_standardNetworkKind = ConditionalLSTMNetworkKind;
        else if (std::find(strType.begin(), strType.end(), L"CRF") != strType.end())
            m_standardNetworkKind = CRFLSTMNetworkKind;
        else
            InvalidArgument("InitRecurrentConfig: unknown value for rnnType parameter '%ls'", strType[0].c_str());
    }

    // Init - Builder Initialize for multiple data sets
    // config - [in] configuration parameters for the network builder
    virtual void Init(const ConfigParameters& config)
    {
        DEVICEID_TYPE deviceId = DeviceFromConfig(config);

        ElemType initValueScale = config("initValueScale", "1.0");

        ConfigArray layerTypes = config("layerTypes", L"Sigmoid"); // TODO: camelCase
        stringargvector nonlinearFunctions = layerTypes;

        bool uniformInit = config("uniformInit", "true");
        bool applyMeanVarNorm = config("applyMeanVarNorm", "false");
        bool needPrior = config("needPrior", "false");

        bool addDropoutNodes = config("addDropoutNodes", "false");

        int outputLayerSize;
        ConfigArray layerSizes;
        intargvector layers;
        TrainingCriterion trainingCriterion;
        EvalCriterion evalCriterion;

        outputLayerSize = config("outputLayerSize", "-1");
        layerSizes = config("layerSizes", "100");
        layers = layerSizes;
        trainingCriterion = ParseTrainingCriterionString(config("trainingCriterion"));
        evalCriterion = ParseEvalCriterionString(config("evalCriterion"));

        ConfigArray rDirect = config("directConnect", "");
        m_directConnect = rDirect;

        m_word2class = config("word2cls", "");
        m_cls2index = config("cls2index", "");
        m_vocabSize = (int) config("vocabSize", "-1");
        m_nbrCls = (int) config("nbrClass", "-1");
        nce_noises = (int) config("noise_number", "-1"); // nce noise

        Init(layers, trainingCriterion, evalCriterion, deviceId, outputLayerSize,
             nonlinearFunctions, addDropoutNodes,
             uniformInit, initValueScale, applyMeanVarNorm, needPrior);

        InitRecurrentConfig(config);

        InitAttentionNetworkConfig(config);
    }

    ComputationNetworkPtr BuildNetworkFromDescription();

    ComputationNetworkPtr BuildNetworkFromDbnFile(const std::wstring& dbnModelFileName); // legacy support for fseide's Microsoft-internal tool "DBN.exe"

protected:

    ComputationNetworkPtr BuildFFDNNFromDescription();
    ComputationNetworkPtr BuildRNNFromDescription();
    ComputationNetworkPtr BuildClassEntropyRNNFromDescription();
    ComputationNetworkPtr BuildLogBilinearNetworkFromDescription();
    ComputationNetworkPtr BuildDNNLMNetworkFromDescription();
    ComputationNetworkPtr BuildLSTMNetworkFromDescription();
#ifdef COMING_SOON
    ComputationNetworkPtr BuildCRFLSTMNetworkFromDescription();
#endif
    ComputationNetworkPtr BuildClassLSTMNetworkFromDescription();
    ComputationNetworkPtr BuildConditionalLSTMNetworkFromDescription();
    ComputationNetworkPtr BuildNCELSTMNetworkFromDescription();

    // mulitply used components
    ComputationNodePtr BuildLSTMComponent(unsigned long& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input);
    ComputationNodePtr BuildLSTMNodeComponent(ULONG& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input);
    ComputationNodePtr BuildDirectConnect(unsigned long& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode);

    // layer is 0 based
    ComputationNodePtr ApplyNonlinearFunction(ComputationNodePtr input, const size_t layer, const std::wstring nodeName = L"");
    ComputationNodePtr AddTrainAndEvalCriterionNodes(ComputationNodePtr input, ComputationNodePtr label, ComputationNodePtr matrix = nullptr, const std::wstring trainNodeName = L"", const std::wstring evalNodeName = L"", ComputationNodePtr clspostprob = nullptr, ComputationNodePtr trans = nullptr);

    static bool CheckDbnTag(File& fstream, const std::string expectedTag)
    {
        char tag[5];
        for (int i = 0; i < 4; i++)
            fstream >> tag[i];
        tag[4] = 0;
        return std::string(tag) == expectedTag;
    }

    Matrix<ElemType> ReadMatrixFromDbnFile(File& fstream, const std::string expectedName)
    {
        int numRows, numCols;
        std::string name;
        if (!CheckDbnTag(fstream, "BMAT"))
            RuntimeError("Error reading DBN file - did not find expected tag BMAT\n");
        // fstream.GetMarker(FileMarker::fileMarkerBeginSection, "BMAT");
        fstream >> name >> numRows >> numCols;
        if (name != expectedName)
        {
            InvalidArgument("ERROR reading pretrained DBN file, expected name %s, found name %s\n", expectedName.c_str(), name.c_str());
        }

        if (numCols > 1) // transpose W because dbn stores that way apparently
        {
            int origRows = numRows;
            numRows = numCols;
            numCols = origRows;
        }

        Matrix<ElemType> mat(numRows, numCols, m_deviceId);

        // dbn operates on row vectors not column vectors. x*W + b, so need to read in as W'
        // ElemType* d_array = new ElemType[numRows*numCols];
        float tmp;
        for (long i = 0; i < numRows; i++)
            for (long j = 0; j < numCols; j++)
            {
                fstream >> tmp;
                mat(i, j) = tmp;
                // d_array[i] = (ElemType)tmp;
            }
        if (!CheckDbnTag(fstream, "EMAT"))
            RuntimeError("Error reading DBN file - did not find expected tag EMAT\n");
        // fstream.GetMarker(FileMarker::fileMarkerBeginSection, "EMAT");

        return mat;
    }

protected:
    ComputationNetworkPtr m_net;

    int m_outputLayerSize;
    intargvector m_layerSizes;
    bool m_applyMeanVarNorm;
    bool m_needPrior;

    DEVICEID_TYPE m_deviceId;
    bool m_uniformInit;

    ElemType m_initValueScale;
    bool m_addDropoutNodes;

    stringargvector m_nonLinearFunctions;

    TrainingCriterion m_trainCriterion;
    EvalCriterion m_evalCriterion;

    intargvector m_directConnect; // connect those layers directly in a sequence order
    // for example: 1:2:3 will connect 1 to 2 and then 2 to 3

    // recurrent network
    intargvector m_recurrentLayers;
    float m_defaultHiddenActivity;
    StandardNetworkKind m_standardNetworkKind;
    int m_maOrder; // MA model order

    bool m_constForgetGateValue;
    bool m_constInputGateValue;
    bool m_constOutputGateValue;

    ElemType m_forgetGateInitVal;
    ElemType m_inputGateInitVal;
    ElemType m_outputGateInitVal;

    intargvector m_streamSizes;           // for multiple stream data
    intargvector m_lookupTabelOrderSizes; // each stream has its own projection, so need to provide with the lookup table order size for each stream

    int m_lookupTableOrder;
    int m_labelEmbeddingSize;

    // these are the file names for word 2 class mapping and class to word index mapping
    // these are used for class-based language modeling
    string m_cls2index;
    string m_word2class;
    int m_nbrCls;    // number of classes
    int m_vocabSize; // vocabulary size
    int nce_noises;

    bool m_sparse_input;

    /**
        for attention network development
        */
    size_t m_auxFeatDim;
};
} } }
