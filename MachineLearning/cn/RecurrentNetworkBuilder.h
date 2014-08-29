//
// <copyright file="RecurrentNetworkBuilder.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ComputationNetwork.h"
#include "IComputationNetBuilder.h"
#include "basetypes.h"
#include <string>
#include "commandArgUtil.h"
#include "DataReader.h"
#include "matrix.h"
#include <stdexcept>
#include "SimpleNetworkBuilder.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { 

    typedef enum tpRNNType { SIMPLERNN = 0, LSTM=1, DEEPRNN=2, DTSRNN=4, DOTSRNN = 8, CLASSLM = 16, 
        TENSORLSTM=32, MEANSGDLSTM=64, ARMALSTM=128, ARLSTM=256, PRJLSTM=512, LBLM=1024,
        NPLM=2048, JORDANLSTM=4096} RnnType; 

    template<class ElemType>
    class RecurrentNetworkBuilder : public SimpleNetworkBuilder<ElemType> 
    {
        float m_defaultHiddenActivity; 
        RnnType m_rnnType;
        int   m_arOrder; /// AR model order
        int   m_maOrder; /// MA model order
        bool m_lookupTableUnittest;
        int m_lookupTableOrder;
        
    public:
        RecurrentNetworkBuilder() { m_rnnType = SIMPLERNN; } // empty constructor, call Init immediately hereafter
        RecurrentNetworkBuilder(const ConfigParameters& config)
        {
            Init(config);
        }

        // Init - Builder Initialize for multiple data sets
        // config - [in] configuration parameters for the network builder
        virtual void Init(const ConfigParameters& config)
        {
            ConfigArray rLayerSizes = config("recurrentLayer", "");
		    intargvector recurrentLayers = rLayerSizes;
            SimpleNetworkBuilder<ElemType>::Init(config);
            m_recurrentLayers=recurrentLayers;
            m_defaultHiddenActivity = config("defaultHiddenActivity", "0.1");
            ConfigArray str_rnnType = config("rnnType", L"SIMPLERNN");

            m_arOrder = config("arOrder", "0");
            m_maOrder = config("maOrder", "0");
            m_lookupTableOrder = config("lookupTableOrder","0");
            m_lookupTableUnittest = config("lookupTableUnittest","false");

            stringargvector strType = str_rnnType; 
            if (std::find(strType.begin(), strType.end(), L"SIMPLERNN") != strType.end())
                m_rnnType = SIMPLERNN;
            if (std::find(strType.begin(), strType.end(), L"LSTM")!= strType.end())
                m_rnnType = LSTM;
            if (std::find(strType.begin(), strType.end(), L"TENSORLSTM")!=strType.end())
                m_rnnType = TENSORLSTM;
            if (std::find(strType.begin(), strType.end(),  L"MEANSGDLSTM")!=strType.end())
                m_rnnType = MEANSGDLSTM;
            if (std::find(strType.begin(), strType.end(),  L"ARMALSTM")!=strType.end())
                m_rnnType = ARMALSTM;
            if (std::find(strType.begin(), strType.end(),  L"ARLSTM")!=strType.end())
                m_rnnType = ARLSTM;
            if (std::find(strType.begin(), strType.end(), L"DEEPRNN")!= strType.end())
                m_rnnType = DEEPRNN;
            if (std::find(strType.begin(), strType.end(), L"CLASSLM")!= strType.end())
                m_rnnType = CLASSLM;
            if (std::find(strType.begin(), strType.end(), L"DOTSRNN")!= strType.end())
                m_rnnType = DOTSRNN;
            if (std::find(strType.begin(), strType.end(), L"PRJLSTM") != strType.end())
                m_rnnType = PRJLSTM;
            if (std::find(strType.begin(), strType.end(), L"LBLM") != strType.end())
                m_rnnType= LBLM;
            if (std::find(strType.begin(), strType.end(), L"NPLM") != strType.end())
                m_rnnType= NPLM;
            if (std::find(strType.begin(), strType.end(), L"JORDANLSTM") != strType.end())
                m_rnnType= JORDANLSTM;

		}

        virtual ~RecurrentNetworkBuilder()
        {}

        virtual ComputationNetwork<ElemType>& BuildNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word)
        {
            if (m_rnnType == LSTM)
                return BuildLSTMNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == TENSORLSTM)
                return BuildLSTMTensorNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == MEANSGDLSTM)
                return BuildMeanNormalizedLSTMNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == ARMALSTM)
                return BuildARMALSTMNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == ARLSTM)
                return BuildARLSTMNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == CLASSLM)
                return BuildClassEntropyNetwork(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == DOTSRNN)
                return BuildDOTSRNNClassEntropyNetwork(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == DTSRNN)
                return BuildDTSRNNClassEntropyNetwork(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == PRJLSTM)
                return BuildProjLSTMNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == LBLM)
                return BuildLogBilinearNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == NPLM)
                return BuildNeuralProbNetworkFromDescription(mbSize, rnnlm_file, word4idx, idx4word);
            if (m_rnnType == JORDANLSTM)
                return BuildJordanLSTM(mbSize, rnnlm_file, word4idx, idx4word);

            if (m_net.GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                input = m_net.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, mbSize);
                m_net.FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net.Mean(input);
                    b = m_net.InvStdDev(input);
                    output = m_net.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

				int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    u = m_net.CreateSparseLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0], mbSize*m_layerSizes[1]);
                    m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
					{
	                    w = m_net.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                        m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						delay = m_net.Delay(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize); 
						/// unless there is a good algorithm to detect loops, use this explicit setup
						output = ApplyNonlinearFunction(
							m_net.Plus(
								m_net.Times(u, input), m_net.Times(w, delay)), 0);
						delay->AttachInputs(output);
                        ((DelayNode<ElemType>*) delay)->SetDelay(1);
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), 0);
					    //output = m_net.Times(u, input);
                    }

					if (m_addDropoutNodes)
						input = m_net.Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        //TODO: to figure out sparse matrix size
                        u = m_net.CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i], 0);
                        m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						    delay = m_net.Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize); 
						    /// unless there is a good algorithm to detect loops, use this explicit setup
						    output = ApplyNonlinearFunction(
							    m_net.Plus(
								    m_net.Times(u, input), m_net.Times(w, delay)), 0);
						    delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net.Dropout(output);
                        else
                            input = output;
                    }
                }

				w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                /*m_net.MatrixL2Reg(w , L"L1w");*/

                //TODO: to verify sparse matrix size
                label = m_net.CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize, mbSize*2);
                AddTrainAndEvalCriterionNodes(input, label, w, L"criterion", L"eval");

                output = m_net.Times(w, input, L"outputs"); 
                
                m_net.OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net.Mean(label);
                }

            }

            m_net.ResetEvalTimeStamp();

            
            return m_net;
        }

        virtual ComputationNetwork<ElemType>& BuildClassEntropyNetwork(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word)
        {
            if (m_net.GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                input = m_net.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, mbSize);
                m_net.FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net.Mean(input);
                    b = m_net.InvStdDev(input);
                    output = m_net.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

				int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    u = m_net.CreateSparseLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0], m_layerSizes[numHiddenLayers]*mbSize);
                    m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                    u->FunctionValues()(0,0) = 0.1;
                    u->FunctionValues()(0,1) = 0.2;
                    u->FunctionValues()(1,0) = -0.1;
                    u->FunctionValues()(1,1) = 0.3;
#endif
//                    b = m_net.CreateLearnableParameter(L"B0", m_layerSizes[1], 1);
#ifdef RNN_DEBUG
                    b->FunctionValues()(0,0) = 0.1;
                    b->FunctionValues()(1,0) = -0.1;
#endif
					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
					{
	                    w = m_net.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                        m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                        w->FunctionValues()(0,0) = 0.2;
                        w->FunctionValues()(0,1) = 0.3;
                        w->FunctionValues()(1,0) = -0.1;
                        w->FunctionValues()(1,1) = -0.1;
#endif

						delay = m_net.Delay(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize); 
						/// unless there is a good algorithm to detect loops, use this explicit setup
						output = ApplyNonlinearFunction(
							m_net.Plus(
								m_net.Times(u, input), m_net.Times(w, delay)), 0);
						delay->AttachInputs(output);
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), 0);
					}

					if (m_addDropoutNodes)
						input = m_net.Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        //TODO: to figure out sparse matrix size
                        u = m_net.CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i], 0);
                        m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                        u->FunctionValues()(0,0) = 0.1;
                        u->FunctionValues()(0,1) = 0.2;
                        u->FunctionValues()(1,0) = -0.1;
                        u->FunctionValues()(1,1) = 0.3;
#endif
//                        b = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", i), m_layerSizes[i+1], 1);
#ifdef RNN_DEBUG
                        b->FunctionValues()(0,0) = 0.1;
                        b->FunctionValues()(1,0) = -0.1;
#endif
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						    delay = m_net.Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize); 
						    /// unless there is a good algorithm to detect loops, use this explicit setup
						    output = ApplyNonlinearFunction(
							    m_net.Plus(
								    m_net.Times(u, input), m_net.Times(w, delay)), 0);
						    delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net.Dropout(output);
                        else
                            input = output;
                    }
                }

                w = m_net.CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers]*(MAX_WORDS_PER_CLASS+MAX_CLASSES)*mbSize);
                m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                label = m_net.CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize, mbSize*2);

                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net.Times(w, input, L"outputs");
                
                m_net.OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net.Mean(label);
                }
            }

            m_net.ResetEvalTimeStamp();

           
            return m_net;
        }

        virtual ComputationNetwork<ElemType>& BuildDOTSRNNClassEntropyNetwork(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildDTSRNNClassEntropyNetwork(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildMeanNormalizedLSTMNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildARMALSTMNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildARLSTMNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildLogBilinearNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildNeuralProbNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        ComputationNetwork<ElemType>& BuildJordanLSTM(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);

        virtual ComputationNetwork<ElemType>& BuildLSTMNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word)
        {
            if (m_net.GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, e=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                ComputationNodePtr Wxo = nullptr, Who=nullptr, Wco=nullptr, bo = nullptr, Wxi=nullptr, Whi=nullptr, Wci=nullptr, bi=nullptr;
                ComputationNodePtr Wxf=nullptr, Whf=nullptr, Wcf=nullptr, bf=nullptr, Wxc=nullptr, Whc=nullptr, bc=nullptr;
                ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, ct=nullptr, ht=nullptr;
                ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC=nullptr, delayCF=nullptr, delayCC=nullptr;
                ComputationNodePtr directWIO = nullptr, directInput=nullptr, directOutput=nullptr;

                //TODO: to figure out sparse matrix size
                input = m_net.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, 0);
                m_net.FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net.Mean(input);
                    b = m_net.InvStdDev(input);
                    output = m_net.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                if(m_lookupTableOrder > 0)
                {
                    e = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                    m_net.InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    output = m_net.LookupTable(e, input, m_lookupTableOrder, m_layerSizes[1], mbSize, L"LookupTable");

                    if (m_addDropoutNodes)
                        input = m_net.Dropout(output);
                    else
                        input = output;
                }

                /// direct connect from input node to output node
                if (m_directConnect.size() > 1)
                {
                    directInput = input;
                    directWIO = m_net.CreateLearnableParameter(msra::strfun::wstrprintf(L"D%d", 0), m_layerSizes[m_directConnect[1]], m_layerSizes[m_directConnect[0]]*(m_lookupTableOrder > 0?m_lookupTableOrder :1));
                    m_net.InitLearnableParameters(directWIO, m_uniformInit, randomSeed++, m_initValueScale);
                    directOutput = m_net.Times(directWIO, directInput);
                }

				int recur_idx = 0;
                int offset = m_lookupTableOrder > 0? 1 : 0;
                if (numHiddenLayers > 0)
                {
                    //TODO: to figure out sparse matrix size
                    Wxo = m_net.CreateSparseLearnableParameter(L"WXO0", m_layerSizes[1 + offset], m_layerSizes[0 + offset] * (offset > 0 ? m_lookupTableOrder : 1), 0);
                    m_net.InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
                    //TODO: to figure out sparse matrix size
                    Wxi = m_net.CreateSparseLearnableParameter(L"WXI0", m_layerSizes[1 + offset], m_layerSizes[0 + offset] * (offset > 0 ? m_lookupTableOrder : 1), 0);
                    m_net.InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
                    bo = m_net.CreateLearnableParameter(L"bo0", m_layerSizes[1 + offset], 1);
                    bc = m_net.CreateLearnableParameter(L"bc0", m_layerSizes[1 + offset], 1);
                    bi = m_net.CreateLearnableParameter(L"bi0", m_layerSizes[1 + offset], 1);
                    bf = m_net.CreateLearnableParameter(L"bf0", m_layerSizes[1 + offset], 1);
                    Wxf = m_net.CreateLearnableParameter(L"WXF0", m_layerSizes[1 + offset], m_layerSizes[0 + offset] * (offset > 0 ? m_lookupTableOrder : 1));
                    m_net.InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
                    Wxc = m_net.CreateLearnableParameter(L"WXC0", m_layerSizes[1 + offset], m_layerSizes[0 + offset] * (offset > 0 ? m_lookupTableOrder : 1));
                    m_net.InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);
					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1 + offset)
					{
	                    Whi = m_net.CreateLearnableParameter(L"WHI0", m_layerSizes[1 + offset], m_layerSizes[1 + offset]);
                        m_net.InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
	                    Wci = m_net.CreateLearnableParameter(L"WCI0", m_layerSizes[1 + offset], 1);
                        m_net.InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

	                    Whf = m_net.CreateLearnableParameter(L"WHF0", m_layerSizes[1 + offset], m_layerSizes[1 + offset]);
                        m_net.InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
	                    Wcf = m_net.CreateLearnableParameter(L"WCF0", m_layerSizes[1 + offset], 1);
                        m_net.InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

	                    Who = m_net.CreateLearnableParameter(L"WHO0", m_layerSizes[1 + offset], m_layerSizes[1 + offset]);
                        m_net.InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
	                    Wco = m_net.CreateLearnableParameter(L"WCO0", m_layerSizes[1 + offset], 1);
                        m_net.InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

	                    Whc = m_net.CreateLearnableParameter(L"WHC0", m_layerSizes[1 + offset], m_layerSizes[1 + offset]);
                        m_net.InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

                        size_t layer1 = m_layerSizes[1 + offset];
                        delayHI = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                        delayHF = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                        delayHO = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                        delayHC = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
						delayCI = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
						delayCF = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
						delayCC = m_net.Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
						/// unless there is a good algorithm to detect loops, use this explicit setup
						it = ApplyNonlinearFunction(
							m_net.Plus(
                                m_net.Plus(
                                    m_net.Plus(
                                        m_net.Times(Wxi, input), 
                                        bi), 
                                    m_net.Times(Whi, delayHI)),
                                m_net.DiagTimes(Wci, delayCI)), 0);
						ft = ApplyNonlinearFunction(
							m_net.Plus(
                                m_net.Plus(
                                    m_net.Plus(
                                        m_net.Times(Wxf, input), 
                                        bf), 
                                    m_net.Times(Whf, delayHF)),
                                m_net.DiagTimes(Wcf, delayCF)), 0);
                        ct = m_net.Plus(
                                m_net.ElementTimes(ft, delayCC),
                                m_net.ElementTimes(it, 
                                    m_net.Tanh(
                                        m_net.Plus(
                                            m_net.Times(Wxc, input),
                                            m_net.Plus(
                                                m_net.Times(Whc, delayHC),
                                                bc
                                            )
                                        )
                                    )
                                )
                             );
						ot = ApplyNonlinearFunction(
							m_net.Plus(
                                m_net.Plus(
                                    m_net.Plus(
                                        m_net.Times(Wxo, input), 
                                        bo), 
                                    m_net.Times(Who, delayHO)),
                                m_net.DiagTimes(Wco, ct)), 0);
                        output = m_net.ElementTimes(ot, m_net.Tanh(ct));
						delayHO->AttachInputs(output);
						delayHI->AttachInputs(output);
						delayHF->AttachInputs(output);
						delayHC->AttachInputs(output);
						delayCI->AttachInputs(ct);
						delayCF->AttachInputs(ct);
						delayCC->AttachInputs(ct);
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), 0);
					}

					if (m_addDropoutNodes)
						input = m_net.Dropout(output);
                    else
                        input = output;

                    for (int i=1 + offset; i<numHiddenLayers; i++)
                    {
                        u = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                        m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
							std::list<ComputationNodePtr> recurrent_loop;
							delay = m_net.Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize);
							output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), m_net.Times(w, delay)), i);
							delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Plus(m_net.Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net.Dropout(output);
                        else
                            input = output;
                    }
                }

                if (directOutput)
                {
                    output = m_net.Plus(input, directOutput); 
                    input = output;
                }

				w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                //TODO: to figure out sparse matrix size
                label = m_net.CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize, 0);
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net.Times(w, input, L"outputs");  
                
                m_net.OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net.Mean(label);
                }

            }

            m_net.ResetEvalTimeStamp();

            
            return m_net;
        }

        virtual ComputationNetwork<ElemType>& BuildLSTMTensorNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);
        
        virtual ComputationNetwork<ElemType>& BuildProjLSTMNetworkFromDescription(size_t mbSize, const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word);
        
        void LoadSimpleRNNPersistableParametersFromFile(const std::wstring& rnnlm_file, std::map<std::string, unsigned>& word4idx, std::map<unsigned, std::string>& idx4word)    //will read whole network structure
        {
            FILE *fi;
            int a, b, ver;
            float starting_alpha, alpha;
            char str[100];
            char strFileName[2048];
            double d;
            size_t sz;
            int version = 15;
            int filetype;
            int layer0_size, layer1_size, fea_size, fea_matrix_used, layerc_size, layer2_size, direct_order;
            int bptt, bptt_block, vocab_size, class_size, old_classes, uses_class_file;
            int independent, alpha_divide;
            long long direct_size;

            std::wstring nodeName = L"U0";
            ComputationNodePtr nodePtrInput = m_net.GetNodeFromName(nodeName);
            nodeName = L"W0";
            ComputationNodePtr nodePtrH2H = m_net.GetNodeFromName(nodeName);
            nodeName = L"W1";
            ComputationNodePtr nodePtrH2O = m_net.GetNodeFromName(nodeName);

            assert(rnnlm_file.length() < 2048);
            wcstombs_s(&sz, strFileName, 2048, rnnlm_file.c_str(), rnnlm_file.length());
            fi=fopen(strFileName, "rt");
            if (fi==NULL) {
                printf("ERROR: model file '%s' not found!\n", rnnlm_file);
                exit(1);
            }

            goToDelimiter(':', fi);
            fscanf_s(fi, "%d", &ver);
            if ((ver==4) && (version==5)) /* we will solve this later.. */ ; else
                if (ver!=version) {
                    printf("Unknown version of file %s\n", rnnlm_file);
                    exit(1);
                }
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &filetype);

                goToDelimiter(':', fi);
                fscanf_s(fi, "%s", str, _countof(str));
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%s", str, _countof(str));
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%lf", &d);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &a);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &a);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%lf", &d);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &a);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &a);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &layer0_size);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &fea_size);
                assert(fea_size == 0);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &fea_matrix_used);
                assert(fea_matrix_used == 0);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%lf", &d);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &layer1_size);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &layerc_size);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &layer2_size);
                //
                if (ver>5) {
                    goToDelimiter(':', fi);
                    fscanf_s(fi, "%lld", &direct_size);
                    assert(direct_size == 0);
                }
                //
                if (ver>6) {
                    goToDelimiter(':', fi);
                    fscanf_s(fi, "%d", &direct_order);
                    assert(direct_order >= 0);
                }
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &bptt);
                //
                if (ver>4) {
                    goToDelimiter(':', fi);
                    fscanf_s(fi, "%d", &bptt_block);
                } else bptt_block=10;
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &vocab_size);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &class_size);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &old_classes);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &uses_class_file);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &independent);
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%lf", &d);
                starting_alpha=(float)d;
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%lf", &d);
                alpha=(float)d;
                //
                goToDelimiter(':', fi);
                fscanf_s(fi, "%d", &alpha_divide);
                //


                //read normal vocabulary
                goToDelimiter(':', fi);
                for (a=0; a<vocab_size; a++) {
                    int cn, class_index;
                    char word[32];
                    std::string strloc; 
                    //fscanf_s(fi, "%d%d%s%d", &b, &vocab[a].cn, vocab[a].word, &vocab[a].class_index);
                    fscanf_s(fi, "%d%d", &b, &cn);
                    readWord(word, fi);
                    fscanf_s(fi, "%d", &class_index);
                    strloc = word; 
                    // printf("%d  %d  %s  %d\n", b, vocab[a].cn, vocab[a].word, vocab[a].class_index);
                    idx4word[b] = strloc;
                    word4idx[strloc] = b;
                }


                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                a=fea_matrix_used;
                fea_matrix_used=0;

                fea_matrix_used=a;
                ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                {
                    goToDelimiter(':', fi);
                    for (a=0; a<layer1_size; a++) {
                        fscanf_s(fi, "%lf", &d, &b);
//                        neu1[a].ac=d;
                    }
                }
                //

                {
                    goToDelimiter(':', fi);
                    assert(nodePtrInput->FunctionValues().GetNumCols() == layer0_size - layer1_size);
                    assert(nodePtrInput->FunctionValues().GetNumRows() == layer1_size);
                    assert(nodePtrH2H->FunctionValues().GetNumCols() == layer1_size);
                    assert(nodePtrH2H->FunctionValues().GetNumRows() == layer1_size);

                    for (b=0; b<layer1_size; b++) {
                        for (a=0; a<layer0_size; a++) {
                            fscanf_s(fi, "%lf", &d, &b);
                            if (a < layer0_size - layer1_size)
                                nodePtrInput->FunctionValues()(b, a) = (ElemType)d; 
                            else 
                                nodePtrH2H->FunctionValues()(b, a - (layer0_size - layer1_size)) = (ElemType)d; 
                        }
                    }
                }
                //
                goToDelimiter(':', fi);
                goToDelimiter(':', fi);
                {
                    goToDelimiter(':', fi);
                    {	//no compress layer
                        for (b=0; b<layer2_size; b++) {
                            for (a=0; a<layer1_size; a++) {
                                fscanf_s(fi, "%lf", &d, &b);
                                if (b < layer2_size - 1)
                                    nodePtrH2O->FunctionValues()(b,a) = (ElemType)d;
                            }
                        }
                    }
                }

                fclose(fi);
        }

		/// unroll recurent networks
		virtual ComputationNetwork<ElemType>& Unroll(size_t mbSize)
        {
            if (m_net.GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
				std::list<ComputationNodePtr> outputNodes; 
				std::list<ComputationNodePtr> inputNodes; 

				for (int rc = 0; rc < mbSize; rc++)
				{
					input = m_net.CreateInputNode(msra::strfun::wstrprintf (L"feature%d", rc), m_layerSizes[0], 1);
					m_net.FeatureNodes().push_back(input);
					inputNodes.push_back(input);
				}

                if (m_applyMeanVarNorm)
                {
					throw("unsupported for RNN");
                    w = m_net.Mean(input);
                    b = m_net.InvStdDev(input);
                    output = m_net.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

				int recur_idx = 0; 
                for (int i=0; i<numHiddenLayers; i++)
                {
                    u = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                    m_net.InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
//                    b = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", i), m_layerSizes[i+1], 1);

					if (m_recurrentLayers[recur_idx] == i+1)
					{
	                    w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
		                ComputationNodePtr initHiddenActNode = m_net.CreateInputNode(msra::strfun::wstrprintf (L"inithiddenact%d", i+1), m_layerSizes[i+1], 1);
						for (int rc = 0; rc < m_layerSizes[i+1]; rc++) 
							initHiddenActNode->FunctionValues()(rc,0) = m_defaultHiddenActivity;
						for (std::list<ComputationNodePtr>::iterator iterNode = inputNodes.begin(); iterNode != inputNodes.end(); iterNode++)
						{
							if (iterNode == inputNodes.begin())
							{
								output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(
									m_net.Plus(
										m_net.Times(u, *iterNode), 
										m_net.Times(w, initHiddenActNode) ) , recur_idx);
							}
							else
							{
								output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(
									m_net.Plus(
										m_net.Times(u, *iterNode), 
										m_net.Times(w, output) ),  recur_idx );
							}
							outputNodes.push_back(output);
						}
						recur_idx ++;
					}
					else
					{
						for (std::list<ComputationNodePtr>::iterator iterNode = inputNodes.begin(); iterNode != inputNodes.end(); iterNode++)
						{
		                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net.Times(u, *iterNode), recur_idx);
							outputNodes.push_back(output); 
						}
					}

					inputNodes.clear();
					for (std::list<ComputationNodePtr>::iterator iterNode = outputNodes.begin(); iterNode != outputNodes.end(); iterNode++)
					{
						inputNodes.push_back(*iterNode); 
					}
					outputNodes.clear();
                }

				w = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net.InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
//                b = m_net.CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], 1);

				int cr = 0;
				for (std::list<ComputationNodePtr>::iterator iterNode = inputNodes.begin(); iterNode != inputNodes.end(); iterNode++)
				{
	                label = m_net.CreateInputNode(msra::strfun::wstrprintf (L"label%d", cr), m_layerSizes[numHiddenLayers+1], 1);
					output = m_net.Times(w, (*iterNode));
	                m_net.OutputNodes().push_back(output);
	                AddTrainAndEvalCriterionNodes(output, label, nullptr, msra::strfun::wstrprintf (L"CrossEntropyWithSoftmax%d", cr), msra::strfun::wstrprintf (L"CrossEntropy%d", cr));
                    cr++;
				}

            }

            m_net.ResetEvalTimeStamp();
            return m_net;
        }

        void goToDelimiter(int delim, FILE *m_file)
        {
            int ch=0;

            while (ch!=delim) {
                ch=fgetc(m_file);
                if (feof(m_file)) {
                    printf("Unexpected end of file\n");
                    throw("Unexpected end of file\n");
                }
            }
        }

        void readWord(char *word, FILE *fin)
        {
            int a=0, ch;
            int MAX_STRING = 2048;

            while (!feof(fin)) {
                ch=fgetc(fin);

                if (ch==13) continue;

                if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
                    if (a>0) {
                        if (ch=='\n') ungetc(ch, fin);
                        break;
                    }

                    if (ch=='\n') {
                        strcpy_s(word, 4, (char *)"</s>");
                        return;
                    }
                    else continue;
                }

                word[a]=(char)ch;
                a++;

                if (a>=MAX_STRING) {
                    //printf("Too long word found!\n");   //truncate too long words
                    a--;
                }
            }
            word[a]=0;
        }

    private:
        intargvector m_recurrentLayers; 
    };

    template class RecurrentNetworkBuilder<float>; 
    template class RecurrentNetworkBuilder<double>;

}}}
