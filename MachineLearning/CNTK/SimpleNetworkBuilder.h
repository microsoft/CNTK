//
// <copyright file="SimpleNetworkBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <stdexcept>
#include <regex>
#include <string>

#include "Basics.h"
#include "Matrix.h"
#include "BestGpu.h"

#include "ComputationNetwork.h"
#include "IComputationNetBuilder.h"
#include "commandArgUtil.h"

#pragma warning (disable: 4661)

using namespace std;

/// this is for sparse input, useful when input dimension is very large and sparse such as language modeling
/// to-do: need to use it guided by argument
#define SPARSE_INPUT

namespace Microsoft { namespace MSR { namespace CNTK {

#define MAX_DEPTH 20

    typedef enum tpRNNType { SIMPLENET=0, /// no recurrent connections
            SIMPLERNN = 1, LSTM=2, DEEPRNN=4, CLASSLM = 8, 
            LBLM=16,
			LSTMENCODER = 18,
            NPLM = 32, CLASSLSTM = 64, NCELSTM = 128, 
			CLSTM = 256, RCRF = 512, 
            UNIDIRECTIONALLSTM=19,
            BIDIRECTIONALLSTM= 20} RNNTYPE;


    enum class TrainingCriterion : int
    {
        CrossEntropyWithSoftmax,
        CrossEntropy,
        SquareError,
        ClassCrossEntropyWithSoftmax,
        NCECrossEntropyWithSoftmax,
		CRF
    };

    enum class EvalCriterion : int
    {
        CrossEntropyWithSoftmax,
        CrossEntropy,
        SquareError,
        ErrorPrediction,
        ClassCrossEntropyWithSoftmax,

        NCECrossEntropyWithSoftmax,
        CRF
    };

    extern TrainingCriterion ParseTrainingCriterionString(wstring s);
    extern EvalCriterion ParseEvalCriterionString(wstring s);

    template<class ElemType>
    class SimpleNetworkBuilder : public IComputationNetBuilder<ElemType>
    {
    protected:
        typedef ComputationNode<ElemType>* ComputationNodePtr;

    private:
        SimpleNetworkBuilder() //disable default constructor from being called
        {        
        } 

    public:
        SimpleNetworkBuilder(const ConfigParameters& config) : m_net(nullptr)  
        {
            Init(config);
        }

        // full parameter Init routine
        void Init(const intargvector& layerSizes, const TrainingCriterion trainCriterion, const EvalCriterion evalCriterion,
            int outputLayerSize = -1,
            const stringargvector nonLinearFunctions=L"Sigmoid", 
            const bool addDropoutNodes=false,
            const bool uniformInit = true, const ElemType initValueScale = 1.0f,
            const bool applyMeanVarNorm = false, bool needPrior = false, DEVICEID_TYPE deviceId = AUTOPLACEMATRIX)
        {
            m_deviceId=deviceId;
            m_net = new ComputationNetwork<ElemType>(m_deviceId);

            m_outputLayerSize = outputLayerSize;
            m_layerSizes=layerSizes;
            m_applyMeanVarNorm=applyMeanVarNorm;
            m_trainCriterion=trainCriterion;
            m_evalCriterion=evalCriterion;
            m_addDropoutNodes=addDropoutNodes;
            m_needPrior=needPrior;
            m_nonLinearFunctions=nonLinearFunctions;
            m_uniformInit=uniformInit;
            m_initValueScale=initValueScale;
            if (m_layerSizes.size() < 2)
                throw std::invalid_argument("A network should have at least two layers (one input and one output)");

            if (m_deviceId == AUTOPLACEMATRIX)
                m_deviceId = Matrix<ElemType>::GetBestGPUDeviceId();

            m_net->SetDeviceID(m_deviceId);
            if (m_deviceId < 0)
                fprintf(stderr,"SimpleNetworkBuilder Using CPU\n");
            else
                fprintf(stderr,"SimpleNetworkBuilder Using GPU %d\n", m_deviceId);

        }

        void InitAttentionNetworkConfig(const ConfigParameters& config)
        {
            m_auxFeatDim= config("auxfeatdim", "20");
        }
        
        virtual void InitRecurrentConfig(const ConfigParameters& config)
        {
            ConfigArray rLayerSizes = config("recurrentLayer", "");
            intargvector recurrentLayers = rLayerSizes;
            m_recurrentLayers=recurrentLayers;
            m_defaultHiddenActivity = config("defaultHiddenActivity", "0.1");
            ConfigArray str_rnnType = config("rnnType", L"SIMPLENET");

            m_maOrder = config("maOrder", "0");
            m_lookupTableOrder = config("lookupTableOrder","0");

            ConfigArray sSizes = config("streamSizes", "");
            m_streamSizes = sSizes;
            sSizes = config("lookupTableOrderSizes", "");  /// this allows having a multiple streams of inputs with
            /// different lookuptable order sizes. the older one lookupTableOrder is still kept to have backward
            /// support.
            m_lookupTabelOrderSizes = sSizes;

            m_labelEmbeddingSize = config("labelEmbeddingSize","10");
            m_constForgetGateValue = config("constForgetGateValue","false");
            m_constInputGateValue = config("constInputGateValue","false");
            m_constOutputGateValue = config("constOutputGateValue","false");

            m_forgetGateInitVal = config("forgetGateInitVal", "-1");
            m_inputGateInitVal = config("inputGateInitVal", "-1");
            m_outputGateInitVal = config("outputGateInitVal", "-1");

            m_sparse_input = config("sparseinput", "false");

            stringargvector strType = str_rnnType; 
            if (std::find(strType.begin(), strType.end(), L"SIMPLERNN") != strType.end())
                m_rnnType = SIMPLERNN;
            if (std::find(strType.begin(), strType.end(), L"LSTM")!= strType.end())
                m_rnnType = LSTM;
            if (std::find(strType.begin(), strType.end(), L"DEEPRNN")!= strType.end())
                m_rnnType = DEEPRNN;
            if (std::find(strType.begin(), strType.end(), L"CLASSLM")!= strType.end())
                m_rnnType = CLASSLM;
            if (std::find(strType.begin(), strType.end(), L"LBLM") != strType.end())
                m_rnnType= LBLM;
            if (std::find(strType.begin(), strType.end(), L"NPLM") != strType.end())
                m_rnnType= NPLM;
            if (std::find(strType.begin(), strType.end(), L"CLASSLSTM") != strType.end())
                m_rnnType= CLASSLSTM;
            if (std::find(strType.begin(), strType.end(), L"NCELSTM") != strType.end())
                m_rnnType = NCELSTM;
            if (std::find(strType.begin(), strType.end(), L"CLSTM") != strType.end())
                m_rnnType= CLSTM;
			if (std::find(strType.begin(), strType.end(), L"CRF") != strType.end())
				m_rnnType = RCRF;
            if (std::find(strType.begin(), strType.end(), L"LSTMENCODER") != strType.end())
                m_rnnType = LSTMENCODER;
            if (std::find(strType.begin(), strType.end(), L"TRANSDUCER") != strType.end() ||
                std::find(strType.begin(), strType.end(), L"UNIDIRECTIONALLSTMWITHPASTPREDICTION") != strType.end())
                m_rnnType = UNIDIRECTIONALLSTM;
            if (std::find(strType.begin(), strType.end(), L"JOINTCONDITIONALBILSTMSTREAMS") != strType.end() ||
                std::find(strType.begin(), strType.end(), L"BIDIRECTIONALLSTMWITHPASTPREDICTION") != strType.end())
                m_rnnType = BIDIRECTIONALLSTM;
        }

        // Init - Builder Initialize for multiple data sets
        // config - [in] configuration parameters for the network builder
        virtual void Init(const ConfigParameters& config)
        {
            DEVICEID_TYPE deviceId = DeviceFromConfig(config);

            ElemType initValueScale = config("initValueScale", "1.0");

            ConfigArray layerTypes = config("layerTypes", L"Sigmoid");
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
            layerSizes = config("layerSizes","100");
            layers = layerSizes;
            trainingCriterion = ParseTrainingCriterionString(config("trainingCriterion"));
            evalCriterion = ParseEvalCriterionString(config("evalCriterion"));

            ConfigArray rDirect = config("directConnect", "");
            m_directConnect = rDirect;

            m_word2class = config("word2cls", "");
            m_cls2index = config("cls2index", "");
            m_vocabSize = (int)config("vocabSize", "-1");
            m_nbrCls = (int)config("nbrClass", "-1");
            nce_noises = (int)config("noise_number", "-1");//nce noise
 
            Init(layers, trainingCriterion, evalCriterion, outputLayerSize,
                nonlinearFunctions, addDropoutNodes,
                uniformInit, initValueScale, applyMeanVarNorm, needPrior, deviceId);   

            InitRecurrentConfig(config);

            InitAttentionNetworkConfig(config);

        }

        virtual ~SimpleNetworkBuilder()
        {
            delete m_net;
        }

        virtual ComputationNetwork<ElemType>& LoadNetworkFromFile(const wstring& modelFileName, bool forceLoad = true,
            bool bAllowNoCriterion = false) 
        {
            if (m_net->GetTotalNumberOfNodes() == 0 || forceLoad) //not built or force load
            {
                bool isDBN = false; 

                {  //force fstream to close when out of range
                    File fstream(modelFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
                    isDBN = CheckDbnTag(fstream,"DBN\n");
                }

                if (isDBN)
                {
                    BuildNetworkFromDbnFile(modelFileName);
                }
                else
                {
                    m_net->LoadFromFile(modelFileName, FileOptions::fileOptionsBinary, bAllowNoCriterion);
                }
            }

            m_net->ResetEvalTimeStamp();
            return *m_net;
        }

        ComputationNetwork<ElemType>& BuildNetworkFromDescription()
        {
            size_t mbSize = 1; 

            if (m_rnnType == SIMPLERNN)
                return BuildSimpleRNN(mbSize);
            if (m_rnnType == LSTM)
                return BuildLSTMNetworkFromDescription(mbSize);
            if (m_rnnType == CLASSLSTM)
                return BuildCLASSLSTMNetworkFromDescription(mbSize);
            if (m_rnnType == NCELSTM)
                return BuildNCELSTMNetworkFromDescription(mbSize);
            if (m_rnnType == CLASSLM)
                return BuildClassEntropyNetwork(mbSize);
            if (m_rnnType == LBLM)
                return BuildLogBilinearNetworkFromDescription(mbSize);
            if (m_rnnType == NPLM)
                return BuildNeuralProbNetworkFromDescription(mbSize);
            if (m_rnnType == CLSTM)
                return BuildConditionalLSTMNetworkFromDescription(mbSize);
			if (m_rnnType == RCRF)
				return BuildSeqTrnLSTMNetworkFromDescription(mbSize);
            if (m_rnnType == LSTMENCODER)
                return BuildLSTMEncoderNetworkFromDescription(mbSize);
            if (m_rnnType == UNIDIRECTIONALLSTM)
                return BuildUnidirectionalLSTMNetworksFromDescription(mbSize);
            if (m_rnnType == BIDIRECTIONALLSTM)
                return BuildBiDirectionalLSTMNetworksFromDescription(mbSize);

            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                unsigned long randomSeed = 1;

                size_t mbSize = 3; //this is not the actual minibatch size. only used in the validataion process

                size_t numHiddenLayers = m_layerSizes.size()-2;
                ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, output=nullptr, label=nullptr, prior=nullptr, scaledLogLikelihood=nullptr;

                input = m_net->Input(m_layerSizes[0], mbSize, L"features");
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {                  
                    w = m_net->Mean(input, L"MeanOfFeatures");
                    b = m_net->InvStdDev(input, L"InvStdOfFeatures");
                    output = m_net->PerDimMeanVarNormalization(input, w, b, L"MVNormalizedFeatures");
                    
                    input = output;
                }

                if (numHiddenLayers > 0)
                {
                    w = m_net->Parameter( m_layerSizes[1], m_layerSizes[0], L"W0");
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                    b = m_net->Parameter(m_layerSizes[1], 1, L"B0");
                    output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(w, input, L"W0*features"), b, L"W0*features+B0"), 0, L"H1");

                    if (m_addDropoutNodes)
                        input = m_net->Dropout(output, L"DropH1");
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        wstring nameOfW = msra::strfun::wstrprintf (L"W%d", i);
                        wstring nameOfB = msra::strfun::wstrprintf (L"B%d", i);
                        wstring nameOfPrevH = msra::strfun::wstrprintf (L"H%d", i);
                        wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                        wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
                        wstring nameOfH = msra::strfun::wstrprintf (L"H%d", i+1);

                        w = m_net->Parameter(m_layerSizes[i+1], m_layerSizes[i], nameOfW);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                        b = m_net->Parameter(m_layerSizes[i+1], 1, nameOfB);
                        output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus), i, nameOfH);

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output, L"Drop" + nameOfH);
                        else
                            input = output;
                    }
                }

                wstring nameOfW = msra::strfun::wstrprintf (L"W%d", numHiddenLayers);
                wstring nameOfB = msra::strfun::wstrprintf (L"B%d", numHiddenLayers);
                wstring nameOfPrevH = msra::strfun::wstrprintf (L"H%d", numHiddenLayers-1);
                wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;

                w = m_net->Parameter(m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers], nameOfW);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                b = m_net->Parameter(m_layerSizes[numHiddenLayers+1], 1, nameOfB);
                output = m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus);   
                m_net->RenameNode(output, L"HLast");

                label = m_net->Input(m_layerSizes[numHiddenLayers+1], mbSize, L"labels");
                
                AddTrainAndEvalCriterionNodes(output, label);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label, L"Prior");
                    input = m_net->Log(prior, L"LogOfPrior");

                    //following two lines are needed only if true probability is needed
                    //output = m_net->Softmax(output);
                    //output = m_net->Log(output);

                    scaledLogLikelihood = m_net->Minus(output, input, L"ScaledLogLikelihood");
                    m_net->OutputNodes().push_back(scaledLogLikelihood);
                }
                else
                {
                    m_net->OutputNodes().push_back(output);
                }

                //add softmax layer (if prob is needed or KL reg adaptation is needed)
                output = m_net->Softmax(output, L"PosteriorProb");
                //m_net->OutputNodes().push_back(output);
            }

            m_net->ResetEvalTimeStamp();
            return *m_net;
        }

        RNNTYPE RnnType(){ return m_rnnType;}

    protected:

        ComputationNetwork<ElemType>& BuildSimpleRNN(size_t mbSize  = 1);

        ComputationNetwork<ElemType>& BuildClassEntropyNetwork(size_t mbSize = 1);

        ComputationNodePtr BuildLSTMComponent(unsigned long &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input);

        ComputationNodePtr BuildLSTMNodeComponent(ULONG &randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input);

        ComputationNodePtr BuildLSTMComponentWithMultiInputs(ULONG &randomSeed, size_t mbSize, size_t iLayer, const vector<size_t>& inputDim, size_t outputDim, const vector<ComputationNodePtr>& inputObs, bool inputWeightSparse = false);

        ComputationNode<ElemType>* BuildDirectConnect(unsigned long &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode);

        ComputationNetwork<ElemType>& BuildLogBilinearNetworkFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildNeuralProbNetworkFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildLSTMNetworkFromDescription(size_t mbSize = 1);

		ComputationNetwork<ElemType>& BuildSeqTrnLSTMNetworkFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildLSTMEncoderNetworkFromDescription(size_t mbSize = 1);
        
        ComputationNetwork<ElemType>& BuildUnidirectionalLSTMNetworksFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildBiDirectionalLSTMNetworksFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildCLASSLSTMNetworkFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildConditionalLSTMNetworkFromDescription(size_t mbSize = 1);

        ComputationNetwork<ElemType>& BuildNCELSTMNetworkFromDescription(size_t mbSize = 1);
 
        
        ComputationNetwork<ElemType>& BuildNetworkFromDbnFile(const std::wstring& dbnModelFileName)
        {
            
            std::string hdr,comment,name; 
            int version;
            int numLayers,i;
            std::string layerType;

            unsigned long randomSeed = 1;

            ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, output=nullptr, label=nullptr, prior=nullptr, scaledLogLikelihood=nullptr;
            PreComputedNode<ElemType>* pcNodePtr=nullptr;
            size_t mbSize = 3; //this is not the actual minibatch size. only used in the validataion process

            File fstream(dbnModelFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);

            if (!CheckDbnTag(fstream,"DBN\n"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag DBN\n");
            fstream >> comment;
            if (!CheckDbnTag(fstream,"BDBN"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag BDBN\n");
            fstream >> version >> numLayers;

            Matrix<ElemType> globalMean = ReadMatrixFromDbnFile(fstream,std::string("gmean"));
            Matrix<ElemType> globalStdDev = ReadMatrixFromDbnFile(fstream,std::string("gstddev"));
            assert(globalMean.GetNumCols()==1);
            assert(globalStdDev.GetNumCols()==1);

            //move to CPU since element-wise operation is expensive and can go wrong in GPU
            int curDevId = globalStdDev.GetDeviceId();
            globalStdDev.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
            for (int i=0;i<globalStdDev.GetNumRows();i++)
                globalStdDev(i,0)=(ElemType)1.0/(const ElemType)globalStdDev(i,0);          
            globalStdDev.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);
        
            if (!CheckDbnTag(fstream,"BNET"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag BNET\n");

            for (i=0;i<numLayers;i++) //0th index is for input layer, 
            {    
                fstream >> layerType;
                
                Matrix<ElemType> wts = ReadMatrixFromDbnFile(fstream,std::string("W"));
                Matrix<ElemType> bias = ReadMatrixFromDbnFile(fstream,std::string("a")); // remnant from pretraining, not needed
                Matrix<ElemType> A = ReadMatrixFromDbnFile(fstream,std::string("b"));
                if (i==0)
                {
                    input = m_net->Input(wts.GetNumCols(), mbSize, L"features");
                    m_net->FeatureNodes().push_back(input);

                    size_t frameDim = globalMean.GetNumRows();
                    size_t numContextFrames = wts.GetNumCols()/frameDim;
                    size_t contextDim = numContextFrames*frameDim;
                    Matrix<ElemType> contextMean(contextDim, 1, m_deviceId);
                    Matrix<ElemType> contextStdDev(contextDim, 1, m_deviceId);

                    //move to CPU since element-wise operation is expensive and can go wrong in GPU
                    contextMean.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
                    contextStdDev.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
                    for (size_t j=0;j<frameDim;j++)
                    {
                        for (size_t k=0;k<numContextFrames;k++)
                        {
                            contextMean(j+k*frameDim,0)=(const ElemType)globalMean(j,0);
                            contextStdDev(j+k*frameDim,0)=(const ElemType)globalStdDev(j,0);
                        }
                    }
                    contextMean.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);
                    contextStdDev.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);

                    w = m_net->Mean(input, L"MeanOfFeatures");                    
                    w->FunctionValues().SetValue(contextMean);
                    w->NeedGradient() = false;
                    pcNodePtr = static_cast<PreComputedNode<ElemType>*>(w);
                    pcNodePtr->MarkComputed(true);

                    b = m_net->InvStdDev(input, L"InvStdOfFeatures");
                    b->FunctionValues().SetValue(contextStdDev);
                    b->NeedGradient() = false;
                    pcNodePtr = static_cast<PreComputedNode<ElemType>*>(b);
                    pcNodePtr->MarkComputed(true);

                    output = m_net->PerDimMeanVarNormalization(input, w, b, L"MVNormalizedFeatures");
                    input = output;
                }
                if (i == numLayers - 1)
                {
                    m_outputLayerSize = wts.GetNumRows();
                }
                wstring nameOfW = msra::strfun::wstrprintf (L"W%d", i);
                wstring nameOfB = msra::strfun::wstrprintf (L"B%d", i);
                wstring nameOfPrevH = msra::strfun::wstrprintf (L"H%d", i);
                wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
                wstring nameOfH = msra::strfun::wstrprintf (L"H%d", i+1);

                w = m_net->Parameter(wts.GetNumRows(),wts.GetNumCols(), nameOfW);
                w->FunctionValues().SetValue(wts);
                        
                b = m_net->Parameter(bias.GetNumRows(), 1, nameOfB);
                b->FunctionValues().SetValue(bias);
                
                if (layerType == "perceptron")
                {
                    fprintf(stderr, "DBN: Reading (%lu x %lu) perceptron\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                    output = m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus);
                }
                else if (layerType == "rbmisalinearbernoulli" )
                {
                    fprintf(stderr, "DBN: Reading (%lu x %lu) linear layer\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                    output = m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus);
                }
                else // assume rbmbernoullibernoulli
                {
                    fprintf(stderr, "DBN: Reading (%lu x %lu) non-linear layer\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                    output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus), i, nameOfH);
                    if (m_addDropoutNodes)
                        input = m_net->Dropout(output, L"Drop" + nameOfH);
                }
    
                input = output;
            }

            if (!CheckDbnTag(fstream,"ENET"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag ENET\n");
            //size_t outputLayerSize =  m_layerSizes[m_layerSizes.size()-1];

            label = m_net->Input(m_outputLayerSize, mbSize, L"labels");

            if (layerType == "perceptron") // complete network
            {
                m_net->RenameNode(output, L"HLast");
#if 0
                assert(numLayers+1==m_layerSizes.size());
#endif
                Matrix<ElemType> priorVals = ReadMatrixFromDbnFile(fstream,std::string("Pu"));
                assert(priorVals.GetNumCols()==1 && priorVals.GetNumRows()==m_outputLayerSize);

                w = m_net->Mean(label, L"Prior");
                w->FunctionValues().SetValue(priorVals);
                w->NeedGradient() = false;
                pcNodePtr = static_cast<PreComputedNode<ElemType>*>(w);
                pcNodePtr->MarkComputed(true);
            }
            else // pretrained network - need to add output layer, initalize
            {
                size_t outputLayerSize = 0;
                if (this->m_outputLayerSize >= 0)
                    outputLayerSize = this->m_outputLayerSize;
                else if (m_layerSizes.size() > 0)
                    m_layerSizes[m_layerSizes.size() - 1];
                else
                    std::runtime_error("Output layer size must be specified when converting pretrained network, use outputLayerSize=");
                    
                size_t penultimateSize = input->FunctionValues().GetNumRows();

                wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
                wstring nameOfB = msra::strfun::wstrprintf (L"B%d", i);
                wstring nameOfPrevH = msra::strfun::wstrprintf (L"H%d", i);
                wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
                wstring nameOfH = msra::strfun::wstrprintf (L"H%d", i+1);

                w = m_net->Parameter(outputLayerSize, penultimateSize, nameOfW);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                b = m_net->Parameter(outputLayerSize, 1, nameOfB);
                output = m_net->Plus(m_net->Times(w, input, nameOfTimes), b, nameOfPlus);
                m_net->RenameNode(output, L"HLast");

                if (m_needPrior)
                {
                    Matrix<ElemType> zeros = Matrix<ElemType>::Zeros(outputLayerSize, 1, m_deviceId);
                    prior = m_net->Mean(label, L"Prior");
                    prior->FunctionValues().SetValue(zeros);
                    pcNodePtr = static_cast<PreComputedNode<ElemType>*>(prior);
                    pcNodePtr->MarkComputed(false);
                }
            } 

            AddTrainAndEvalCriterionNodes(output, label);

            if (layerType=="perceptron" || m_needPrior) 
            {
                input = m_net->Log(pcNodePtr, L"LogOfPrior");

                //following two lines is needed only if true probability is needed
                //output = m_net->Softmax(output);
                //output = m_net->Log(output);

                scaledLogLikelihood = m_net->CreateComputationNode(MinusNode<ElemType>::TypeName(), L"ScaledLogLikelihood");
                scaledLogLikelihood->AttachInputs(output, input);
                m_net->OutputNodes().push_back(scaledLogLikelihood);
            }
            else
            {
                m_net->OutputNodes().push_back(output);
            }

            if (!CheckDbnTag(fstream,"EDBN"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag ENET\n");
            return *m_net;
        }

        //layer is 0 based
        ComputationNodePtr ApplyNonlinearFunction(ComputationNodePtr input, const size_t layer, const std::wstring nodeName = L"")
        {
            ComputationNodePtr output;
            wstring nonLinearFunction = m_nonLinearFunctions[layer];
            if (nonLinearFunction == SigmoidNode<ElemType>::TypeName())
                output = m_net->Sigmoid(input, nodeName);
            else if (nonLinearFunction == RectifiedLinearNode<ElemType>::TypeName())
                output = m_net->RectifiedLinear(input, nodeName);
            else if (nonLinearFunction == TanhNode<ElemType>::TypeName())
                output = m_net->Tanh(input, nodeName);
            else if (nonLinearFunction ==  L"None" || nonLinearFunction == L"none" || nonLinearFunction == L"")
            {
                output = input;  //linear layer
                if (nodeName != L"") 
                    m_net->RenameNode(output, nodeName);
            }
            else
                throw std::logic_error("Unsupported nonlinear function.");

            return output;
        }

        ComputationNodePtr AddTrainAndEvalCriterionNodes(ComputationNodePtr input, ComputationNodePtr label, ComputationNodePtr matrix = nullptr, const std::wstring trainNodeName = L"", const std::wstring evalNodeName = L"", ComputationNodePtr clspostprob = nullptr, ComputationNodePtr trans = nullptr)
        {
            m_net->LabelNodes().push_back(label);

            ComputationNodePtr output;
            ComputationNodePtr tinput = input;
            if (matrix != nullptr)
            {
                tinput = m_net->Times(matrix, input);   
            }

            switch (m_trainCriterion)
            {
            case TrainingCriterion::CrossEntropyWithSoftmax:
                output = m_net->CrossEntropyWithSoftmax(label, tinput, (trainNodeName == L"")?L"CrossEntropyWithSoftmax":trainNodeName);
                break;
            case TrainingCriterion::SquareError:
                output = m_net->SquareError(label, tinput, (trainNodeName == L"")?L"SquareError":trainNodeName);
                break;
                case TrainingCriterion::CRF:
                    assert(trans != nullptr);
                    output = m_net->CRF(label, input, trans, (trainNodeName == L"") ? L"CRF" : trainNodeName);
                    break;
            case TrainingCriterion::ClassCrossEntropyWithSoftmax:
                output = m_net->ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (trainNodeName == L"")?L"ClassCrossEntropyWithSoftmax":trainNodeName);
                break;
            case TrainingCriterion::NCECrossEntropyWithSoftmax:
                output = m_net->NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
                //output = m_net->NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
                break;
         default:
                throw std::logic_error("Unsupported training criterion.");
            }
            m_net->FinalCriterionNodes().push_back(output);

            if (!((m_evalCriterion == EvalCriterion::CrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::CrossEntropyWithSoftmax) ||
                (m_evalCriterion == EvalCriterion::SquareError && m_trainCriterion == TrainingCriterion::SquareError) ||
                (m_evalCriterion == EvalCriterion::CRF && m_trainCriterion == TrainingCriterion::CRF) ||
                (m_evalCriterion == EvalCriterion::ClassCrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::ClassCrossEntropyWithSoftmax) ||
                (m_evalCriterion == EvalCriterion::NCECrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::NCECrossEntropyWithSoftmax)))
            {
                switch (m_evalCriterion)
                {
                case EvalCriterion::CrossEntropyWithSoftmax:
                    //output = m_net->CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"")?L"EvalCrossEntropyWithSoftmax":evalNodeName);
                    output = m_net->CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"") ? L"CrossEntropyWithSoftmax" : evalNodeName);
                    break;
                case EvalCriterion::ClassCrossEntropyWithSoftmax:
                    //output = m_net->ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"EvalClassCrossEntropyWithSoftmax" : evalNodeName);
                    output = m_net->ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"ClassCrossEntropyWithSoftmax" : evalNodeName);
                    break;  
                case EvalCriterion::NCECrossEntropyWithSoftmax:
                    output = m_net->NoiseContrastiveEstimation(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"NoiseContrastiveEstimationNode" : evalNodeName);
                    break;
                case EvalCriterion::SquareError:
                    //output = m_net->SquareError(label, tinput, (evalNodeName == L"")?L"EvalSquareError":evalNodeName);
                    output = m_net->SquareError(label, tinput, (evalNodeName == L"") ? L"SquareError" : evalNodeName);
                    break;
                case EvalCriterion::ErrorPrediction:
					output = m_net->ErrorPrediction(label, tinput, (evalNodeName == L"") ? L"EvalErrorPrediction" : evalNodeName);
					break;
				case EvalCriterion::CRF:
					assert(trans != nullptr);
					output = m_net->CRF(label, tinput, trans, (evalNodeName == L"") ? L"EvalCRF" : evalNodeName);
                    break;
                default:
                    throw std::logic_error("Unsupported training criterion.");
                }
            }

            m_net->EvaluationNodes().push_back(output);

            return output;
       }

        Matrix<ElemType> ReadMatrixFromDbnFile(File &fstream, const std::string expectedName)
        {
            int numRows,numCols;
            std::string name;
            if (!CheckDbnTag(fstream,"BMAT"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag BMAT\n");
            //fstream.GetMarker(FileMarker::fileMarkerBeginSection, "BMAT");
            fstream >> name >> numRows >> numCols;
            if (name != expectedName)
            {
                throw std::invalid_argument(msra::strfun::strprintf("ERROR reading pretrained DBN file, expected name %s, found name %s\n", expectedName.c_str(), name.c_str()));
            }

            if (numCols>1) // transpose W because dbn stores that way apparently
            {        
                int origRows = numRows;
                numRows = numCols;
                numCols = origRows;
            }
        

            Matrix<ElemType> mat(numRows, numCols, m_deviceId);

            // dbn operates on row vectors not column vectors. x*W + b, so need to read in as W'
            //ElemType* d_array = new ElemType[numRows*numCols];
            float tmp;
            for (long i=0;i<numRows;i++)
                for (long j=0;j<numCols;j++)
                {
                    fstream>>tmp;
                    mat(i,j)=tmp;
                //d_array[i] = (ElemType)tmp;                
                }
            if (!CheckDbnTag(fstream,"EMAT"))
                throw std::runtime_error("Error reading DBN file - did not find expected tag EMAT\n");
            //fstream.GetMarker(FileMarker::fileMarkerBeginSection, "EMAT");

            return mat;

        }

        bool CheckDbnTag(File &fstream, const std::string expectedTag)
        {
            char tag[5];
            for (int i=0;i<4;i++)
                fstream >> tag[i];
            tag[4] = 0;

            if (std::string(tag)!=expectedTag)
            {
                return false;
            }

            return true;
        }
    protected:
        ComputationNetwork<ElemType>* m_net;

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

        intargvector m_directConnect; /// connect those layers directly in a sequence order
        /// for example: 1:2:3 will connect 1 to 2 and then 2 to 3

        /// recurrent network 
        intargvector m_recurrentLayers; 
        float m_defaultHiddenActivity; 
        RNNTYPE m_rnnType;
        int   m_maOrder; /// MA model order

        bool m_constForgetGateValue;
        bool m_constInputGateValue;
        bool m_constOutputGateValue;

        ElemType m_forgetGateInitVal;
        ElemType m_inputGateInitVal;
        ElemType m_outputGateInitVal;
  
        intargvector m_streamSizes;  /// for multiple stream data
        intargvector m_lookupTabelOrderSizes; /// each stream has its own projection, so need to provide with the lookup table order size for each stream

        int m_lookupTableOrder;
        int m_labelEmbeddingSize;

        /// these are the file names for word 2 class mapping and class to word index mapping
        /// these are used for class-based language modeling
        string m_cls2index;
        string m_word2class;
        int m_nbrCls;  /// number of classes
        int m_vocabSize; /// vocabulary size
        int nce_noises;

        bool m_sparse_input; 

        /**
        for attention network development
        */
        size_t m_auxFeatDim;
    };

}}}
