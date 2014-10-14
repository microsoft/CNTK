//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "ComputationNode.h"
#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"
#include "SimpleNetworkBuilder.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildSimpleRNN(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                //TODO: to figure out sparse matrix size
                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, 0);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

				int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    //TODO: to figure out sparse matrix size
                    u = m_net->CreateSparseLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0], 0);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
					{
	                    w = m_net->CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize); 
						/// unless there is a good algorithm to detect loops, use this explicit setup
						output = ApplyNonlinearFunction(
							m_net->Plus(
								m_net->Times(u, input), m_net->Times(w, delay)), 0);
						delay->AttachInputs(output);
                        ((DelayNode<ElemType>*) delay)->SetDelay(1);
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
                        //output = m_net->Times(u, input);
					}

					if (m_addDropoutNodes)
						input = m_net->Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        //TODO: to figure out sparse matrix size
                        u = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i], 0);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                        if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						    delay = m_net->Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize); 
						    /// unless there is a good algorithm to detect loops, use this explicit setup
						    output = ApplyNonlinearFunction(
							    m_net->Plus(
								    m_net->Times(u, input), m_net->Times(w, delay)), 0);
						    delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                }

				w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                /*m_net->MatrixL2Reg(w , L"L1w");*/

                label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
                AddTrainAndEvalCriterionNodes(input, label, w, L"criterion", L"eval");

                output = m_net->Times(w, input, L"outputs");   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                }

            }

            m_net->ResetEvalTimeStamp();

            return *m_net;
    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildClassEntropyNetwork(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

				int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    u = m_net->CreateSparseLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                    u->FunctionValues()(0,0) = 0.1;
                    u->FunctionValues()(0,1) = 0.2;
                    u->FunctionValues()(1,0) = -0.1;
                    u->FunctionValues()(1,1) = 0.3;
#endif
//                    b = m_net->CreateLearnableParameter(L"B0", m_layerSizes[1], 1);
#ifdef RNN_DEBUG
                    b->FunctionValues()(0,0) = 0.1;
                    b->FunctionValues()(1,0) = -0.1;
#endif
					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
					{
	                    w = m_net->CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                        w->FunctionValues()(0,0) = 0.2;
                        w->FunctionValues()(0,1) = 0.3;
                        w->FunctionValues()(1,0) = -0.1;
                        w->FunctionValues()(1,1) = -0.1;
#endif

						delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize); 
						/// unless there is a good algorithm to detect loops, use this explicit setup
						output = ApplyNonlinearFunction(
							m_net->Plus(
								m_net->Times(u, input), m_net->Times(w, delay)), 0);
						delay->AttachInputs(output);
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
					}

					if (m_addDropoutNodes)
						input = m_net->Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        u = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef RNN_DEBUG
                        u->FunctionValues()(0,0) = 0.1;
                        u->FunctionValues()(0,1) = 0.2;
                        u->FunctionValues()(1,0) = -0.1;
                        u->FunctionValues()(1,1) = 0.3;
#endif
//                        b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", i), m_layerSizes[i+1], 1);
#ifdef RNN_DEBUG
                        b->FunctionValues()(0,0) = 0.1;
                        b->FunctionValues()(1,0) = -0.1;
#endif
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

						    delay = m_net->Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize); 
						    /// unless there is a good algorithm to detect loops, use this explicit setup
						    output = ApplyNonlinearFunction(
							    m_net->Plus(
								    m_net->Times(u, input), m_net->Times(w, delay)), 0);
						    delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                }

                w = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                label = m_net->CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);

                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net->Times(w, input, L"outputs");   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                }
            }

            m_net->ResetEvalTimeStamp();

            return *m_net;

    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildLSTMInputOutputTensorNetworkFromDescription(size_t mbSize)
    {
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size()-2;

			size_t numRecurrentLayers = m_recurrentLayers.size(); 

			ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr, pastEmbedding=nullptr;
            ComputationNodePtr Wxo = nullptr, Who=nullptr, Wco=nullptr, bo = nullptr, Wxi=nullptr, Whi=nullptr, Wci=nullptr, bi=nullptr;
            ComputationNodePtr Wxf=nullptr, Whf=nullptr, Wcf=nullptr, bf=nullptr, Wxc=nullptr, Whc=nullptr, bc=nullptr;
            ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, ct=nullptr, ht=nullptr;
            ComputationNodePtr einput=nullptr, elabel=nullptr;
            ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC=nullptr, delayCF=nullptr, delayCC=nullptr, delayYI=nullptr;
            ComputationNodePtr Wtensoroi = nullptr;
            ComputationNodePtr stateInput = nullptr;
            ComputationNodePtr beforeSoftMax = nullptr, Wti = nullptr, Wtf = nullptr, Wtc = nullptr, Wto= nullptr, Wxt = nullptr, Wyt = nullptr , reducedOutput = nullptr, reducedInput = nullptr; 
            size_t tensorSize = 10;

            input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = m_net->Mean(input);
                b = m_net->InvStdDev(input);
                output = m_net->PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if(m_lookupTableOrder > 0)
            {
                einput = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"EINPUT%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                m_net->InitLearnableParameters(einput, m_uniformInit, randomSeed++, m_initValueScale);
                output = m_net->LookupTable(einput, input, L"LookupTable");

                if (m_addDropoutNodes)
                    input = m_net->Dropout(output);
                else
                    input = output;

                //outputFromEachLayer[1] = input;
            }

            int recur_idx = 0; 
            int offset = m_lookupTableOrder > 0? 1 : 0;
            if (numHiddenLayers > 0)
            {
                //TODO: to figure out sparse matrix size
                Wxo = m_net->CreateLearnableParameter(L"WXO0", m_layerSizes[offset+1], m_layerSizes[offset]*(offset?m_lookupTableOrder:1));
                m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
                //TODO: to figure out sparse matrix size
                Wxi = m_net->CreateLearnableParameter(L"WXI0", m_layerSizes[offset+1], m_layerSizes[offset]*(offset?m_lookupTableOrder:1));
                m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
                bo = m_net->CreateLearnableParameter(L"bo0", m_layerSizes[offset+1], 1);
                bc = m_net->CreateLearnableParameter(L"bc0", m_layerSizes[offset+1], 1);
                bi = m_net->CreateLearnableParameter(L"bi0", m_layerSizes[offset+1], 1);
                bf = m_net->CreateLearnableParameter(L"bf0", m_layerSizes[offset+1], 1);
                //TODO: to figure out sparse matrix size
                Wxf = m_net->CreateLearnableParameter(L"WXF0", m_layerSizes[offset+1], m_layerSizes[offset]*(offset?m_lookupTableOrder:1));
                m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
                //TODO: to figure out sparse matrix size
                Wxc = m_net->CreateLearnableParameter(L"WXC0", m_layerSizes[offset+1], m_layerSizes[offset]*(offset?m_lookupTableOrder:1));
                m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == offset+1)
				{
	                Whi = m_net->CreateLearnableParameter(L"WHI0", m_layerSizes[offset+1], m_layerSizes[offset+1]);
                    m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
	                Wci = m_net->CreateLearnableParameter(L"WCI0", m_layerSizes[offset+1], 1);
                    m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

                    Whf = m_net->CreateLearnableParameter(L"WHF0", m_layerSizes[offset+1], m_layerSizes[offset+1]);
                    m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
	                Wcf = m_net->CreateLearnableParameter(L"WCF0", m_layerSizes[offset+1], 1);
                    m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

                    Who = m_net->CreateLearnableParameter(L"WHO0", m_layerSizes[offset+1], m_layerSizes[offset+1]);
                    m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
	                Wco = m_net->CreateLearnableParameter(L"WCO0", m_layerSizes[offset+1], 1);
                    m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

                    Whc = m_net->CreateLearnableParameter(L"WHC0", m_layerSizes[offset+1], m_layerSizes[offset+1]);
                    m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

                    Wtensoroi = m_net->CreateLearnableParameter(L"WTENSOR0", tensorSize, tensorSize * tensorSize);
                    m_net->InitLearnableParameters(Wtensoroi, m_uniformInit, randomSeed++, m_initValueScale);

                    size_t layer1 = m_layerSizes[offset+1];
                    size_t layerOutput = m_layerSizes[numHiddenLayers+1];
                    elabel = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"ELABEL%d", 0), m_labelEmbeddingSize, layerOutput);
                    m_net->InitLearnableParameters(elabel, m_uniformInit, randomSeed++, m_initValueScale);

                    
                    delayHI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                    delayHF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                    delayHO = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
                    delayHC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
					delayCI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
					delayCF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
					delayCC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);


                    delayYI = m_net->Delay(NULL, 0, layerOutput, mbSize);                    
                    /// reduce dimension
                    Wyt = m_net->CreateLearnableParameter(L"WYT0", tensorSize, m_layerSizes[numHiddenLayers+1]);
                    m_net->InitLearnableParameters(Wyt, m_uniformInit, randomSeed++, m_initValueScale);
                    Wxt = m_net->CreateLearnableParameter(L"WXT0", tensorSize, m_layerSizes[offset]*(offset?m_lookupTableOrder:1));
                    m_net->InitLearnableParameters(Wxt, m_uniformInit, randomSeed++, m_initValueScale);

                    reducedOutput = m_net->Times(Wyt, delayYI, L"ReduceDimensionFromOutput");
                    reducedInput = m_net->Times(Wxt, input, L"ReduceDimensionFromInput");
                    Wti = m_net->CreateLearnableParameter(L"WTI0", m_layerSizes[offset], tensorSize);
                    m_net->InitLearnableParameters(Wti, m_uniformInit, randomSeed++, m_initValueScale);

                    input = m_net->Plus(input, 
                        m_net->Times(Wti, m_net->Times(Wtensoroi,m_net->KhatriRaoProduct(reducedInput, reducedOutput))));

                    it = ApplyNonlinearFunction(
					    m_net->Plus(
                            m_net->Plus(
                                m_net->Plus(
                                    m_net->Times(Wxi, input), 
                                        bi), 
                                    m_net->Times(Whi, delayHI)),
                                m_net->DiagTimes(Wci, delayCI)), 0);
                    ft = ApplyNonlinearFunction(
					    m_net->Plus(
                            m_net->Plus(
                                m_net->Plus(
                                    m_net->Times(Wxf, input), 
                                        bf), 
                                    m_net->Times(Whf, delayHF)),
                                m_net->DiagTimes(Wcf, delayCF)), 0);
                    stateInput = m_net->Tanh(
                                        m_net->Plus(
                                            m_net->Times(Wxc, input),
                                                m_net->Plus(
                                                    m_net->Times(Whc, delayHC),
                                                    bc
                                                )
                                            )
                                 );
                    ct = m_net->Plus(
                            m_net->ElementTimes(ft, delayCC),
                              m_net->ElementTimes(it, 
                                    stateInput
                                )
                              );
                    ot = ApplyNonlinearFunction(
					    m_net->Plus(
                            m_net->Plus(
                                m_net->Plus(
                                    m_net->Times(Wxo, input), 
                                        bo), 
                                    m_net->Times(Who, delayHO)),
                                m_net->DiagTimes(Wco, ct)), 0);
                    output = m_net->ElementTimes(ot, m_net->Tanh(ct));
					
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
	                output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
			    }

				if (m_addDropoutNodes)
				    input = m_net->Dropout(output);
                else
                    input = output;

                for (int i=1+offset; i<numHiddenLayers; i++)
                {
                    u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
					{
					    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
						std::list<ComputationNodePtr> recurrent_loop;
						delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize);
						output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), m_net->Times(w, delay)), i);
						delay->AttachInputs(output);
						recur_idx++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
					}

					if (m_addDropoutNodes)
                        input = m_net->Dropout(output);
                    else
                        input = output;
                }
            }

			w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);

            AddTrainAndEvalCriterionNodes(input, label, w);
                
            beforeSoftMax = m_net->Times(w, input, L"BeforeSoftMax");
            output = m_net->Softmax(beforeSoftMax,L"outputs"); // triggered bugs

            delayYI->AttachInputs(beforeSoftMax);
//            delayYI->NeedGradient()=false;
            m_net->OutputNodes().push_back(output);

            if (m_needPrior)
            {
                prior = m_net->Mean(label);
            }

        }
        m_net->ResetEvalTimeStamp();

        return *m_net;
    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildLogBilinearNetworkFromDescription(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

                size_t numRecurrentLayers = m_recurrentLayers.size();

                ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr, featin = nullptr, e = nullptr;
                ComputationNodePtr bi=nullptr;
                ComputationNodePtr Wxi1=nullptr, Wxi=nullptr;
                ComputationNodePtr Wxi2=nullptr, Wxi3=nullptr, Wxi4=nullptr;
                ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, gt=nullptr, ct=nullptr, ht=nullptr;
                ComputationNodePtr delayXI = nullptr, delayXII = nullptr, delayXIII = nullptr, delayXIV = nullptr;

//                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);
                featin = input;
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                //used for lookuptable node unittest, will delete
                if(m_lookupTableOrder > 0)
                {
                    e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                    m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    output = m_net->LookupTable(e, input, L"Lookuptatble");

                    if (m_addDropoutNodes)
                        input = m_net->Dropout(output);
                    else
                        input = output;
                }

                int recur_idx = 0; 
    			/// unless there is a good algorithm to detect loops, use this explicit setup
                int ik = 1; 
                output = input;
                while (ik <= m_maOrder)
                {
                    delayXI = 
                        m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, 
                        msra::strfun::wstrprintf(L"DELAY%d", ik)); 
                    delayXI->NeedGradient() = false; 
                    delayXI->AttachInputs(input);
                    ((DelayNode<ElemType>*) delayXI)->SetDelay(ik);
                    //TODO: to figure out sparse matrix size
                    Wxi = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"DD%d", ik), m_layerSizes[0], m_layerSizes[0], 0);
                    m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

                    it = m_net->Plus(output, m_net->Times(Wxi, delayXI));
                    output = it;

                    ik++;
                }
                
                if (m_addDropoutNodes)
				    input = m_net->Dropout(output);
                else
                    input = output;

                for (int i = m_lookupTableOrder > 0 ? 1 : 0; i<numHiddenLayers; i++)
                {
                    u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i] * (m_lookupTableOrder > 0 ? m_lookupTableOrder : 1));
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                    output= m_net->Times(u, input);
                    input = output;
					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
					{
					    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"R%d", i+1), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
						delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize);
						output = m_net->Plus(m_net->Times(w, delay), input);

                        delay->AttachInputs(output);
                        input = output;
						recur_idx++;
			        }

                    bi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"bi%d", i), m_layerSizes[i+1], 1);
                    output = m_net->Plus(input, bi);

					if (m_addDropoutNodes)
						input = m_net->Dropout(output);
                    else
                        input = output;
                }
            
				w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net->Times(w, input, L"outputs");   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                }
            }

            m_net->ResetEvalTimeStamp();

            return *m_net;
    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildNeuralProbNetworkFromDescription(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                ULONG randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

				size_t numRecurrentLayers = m_recurrentLayers.size(); 

				ComputationNodePtr input=nullptr, w=nullptr, b=nullptr, u=nullptr, delay = nullptr, output=nullptr, label=nullptr, prior=nullptr;
                ComputationNodePtr bi=nullptr;
                ComputationNodePtr Wxi1=nullptr, Wxi=nullptr;
                ComputationNodePtr Wxi2=nullptr, Wxi3=nullptr, Wxi4=nullptr;
                ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, gt=nullptr, ct=nullptr, ht=nullptr;
                ComputationNodePtr delayXI = nullptr, delayXII = nullptr, delayXIII = nullptr, delayXIV = nullptr;

                //TODO: to figure out sparse matrix size
                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, 0);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    bi = m_net->CreateLearnableParameter(L"bi0", m_layerSizes[1], 1);

                    delayXI = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize); 
                    delayXII = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize); 
                    delayXIII = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize); 
                    delayXIV = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize); 
                    delayXI->AttachInputs(input);
                    delayXII->AttachInputs(input);
                    delayXIII->AttachInputs(input);
                    delayXIV->AttachInputs(input);

					if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
					{
                        //TODO: to figure out sparse matrix size
	                    Wxi2 = m_net->CreateSparseLearnableParameter(L"WXI2", m_layerSizes[1], m_layerSizes[0], 0);
                        m_net->InitLearnableParameters(Wxi2, m_uniformInit, randomSeed++, m_initValueScale);
	                    //TODO: to figure out sparse matrix size
                        Wxi3 = m_net->CreateSparseLearnableParameter(L"WXI3", m_layerSizes[1], m_layerSizes[0], 0);
                        m_net->InitLearnableParameters(Wxi3, m_uniformInit, randomSeed++, m_initValueScale);
	                    //TODO: to figure out sparse matrix size
                        Wxi4 = m_net->CreateSparseLearnableParameter(L"WXI4", m_layerSizes[1], m_layerSizes[0], 0);
                        m_net->InitLearnableParameters(Wxi4, m_uniformInit, randomSeed++, m_initValueScale);
	                    //TODO: to figure out sparse matrix size
                        Wxi1 = m_net->CreateSparseLearnableParameter(L"WXI1", m_layerSizes[1], m_layerSizes[0], 0);
                        m_net->InitLearnableParameters(Wxi1, m_uniformInit, randomSeed++, m_initValueScale);
	                    //TODO: to figure out sparse matrix size
                        Wxi = m_net->CreateSparseLearnableParameter(L"WXI", m_layerSizes[1], m_layerSizes[0], 0);
                        m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

						/// unless there is a good algorithm to detect loops, use this explicit setup
						it = m_net->Plus(
                                m_net->Tanh(
                                m_net->Plus(
                                m_net->Times(Wxi4, delayXIV),
                                m_net->Plus(
                                m_net->Times(Wxi3, delayXIII),
                                m_net->Plus(
                                    m_net->Times(Wxi2, delayXII),
                                    m_net->Plus(
                                        m_net->Times(Wxi1, delayXI),
                                        m_net->Times(Wxi, input))
                                        )
                                    )
                                )),
                                bi);
                        output = it;
                        ((DelayNode<ElemType>*) delayXII)->SetDelay(2);
                        ((DelayNode<ElemType>*) delayXIII)->SetDelay(3);
                        ((DelayNode<ElemType>*) delayXIV)->SetDelay(4);
                        delayXI->NeedGradient() = false;
                        delayXII->NeedGradient() = false;
                        delayXIII->NeedGradient() = false;
                        delayXIV->NeedGradient() = false;
						recur_idx ++;
					}
					else
					{
	                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
					}

					if (m_addDropoutNodes)
						input = m_net->Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
						{
							w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
							std::list<ComputationNodePtr> recurrent_loop;
							delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize);
							output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), m_net->Times(w, delay)), i);
							delay->AttachInputs(output);
							recur_idx++;
						}
						else
						{
	                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                }

                //TODO: to figure out sparse matrix size
				w = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers], 0);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
//                b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], 1);
                //TODO: to figure out sparse matrix size
                label = m_net->CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize, 0);
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net->Times(w, input);   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                }
            }

            m_net->ResetEvalTimeStamp();

            return *m_net;
    }

    template<class ElemType>
    ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildDirectConnect(ULONG &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode)
    {
        ComputationNodePtr directOutput = nullptr, mergedNode = nullptr;

        for (size_t i = 0; i < m_directConnect.size(); i++)
        {
            if (m_directConnect[i] == iLayer)
            {
                ComputationNodePtr directWIO = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"D%d", i), outputDim, inputDim);
                m_net->InitLearnableParameters(directWIO, m_uniformInit, randomSeed++, m_initValueScale);
                directOutput = ApplyNonlinearFunction(m_net->Times(directWIO, input),i);

                ComputationNodePtr scalar = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"SV%d", i), 1, 1);
                scalar->FunctionValues().SetValue((ElemType)0.01);
                ComputationNodePtr scaled = m_net->Scale(scalar, directOutput, msra::strfun::wstrprintf(L"S%d", i));

                mergedNode = m_net->Plus(toNode, scaled);
            }
        }

        return mergedNode;
    }


    template<class ElemType>
    ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMComponent(ULONG &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs, bool inputWeightSparse)
    {

        size_t numHiddenLayers = m_layerSizes.size()-2;

        ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
        ComputationNodePtr Wxo = nullptr, Who=nullptr, Wco=nullptr, bo = nullptr, Wxi=nullptr, Whi=nullptr, Wci=nullptr, bi=nullptr;
        ComputationNodePtr Wxf=nullptr, Whf=nullptr, Wcf=nullptr, bf=nullptr, Wxc=nullptr, Whc=nullptr, bc=nullptr;
        ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, ct=nullptr, ht=nullptr;
        ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC=nullptr, delayCF=nullptr, delayCC=nullptr;
        ComputationNodePtr directWIO = nullptr, directInput=nullptr, directOutput=nullptr;
        ComputationNodePtr bit=nullptr, bft=nullptr, bct=nullptr;

        input = inputObs;
        if(inputWeightSparse)
        {
            Wxo = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"WXO%d", iLayer), outputDim, inputDim);
            Wxi = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"WXI%d", iLayer), outputDim, inputDim);
            Wxf = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"WXF%d", iLayer), outputDim, inputDim);
            Wxc = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"WXC%d", iLayer), outputDim, inputDim);
        }
        else
        {
            Wxo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WXO%d", iLayer), outputDim, inputDim);        
            Wxi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WXI%d", iLayer), outputDim, inputDim);
            Wxf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WXF%d", iLayer), outputDim, inputDim);
            Wxc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WXC%d", iLayer), outputDim, inputDim);
        }
        m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

        bo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"bo%d", iLayer), outputDim, 1);
        bc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"bc%d", iLayer), outputDim, 1);
        bi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"bi%d", iLayer), outputDim, 1);
        bf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"bf%d", iLayer), outputDim, 1);
        //if (m_forgetGateInitVal > 0)
            bf->FunctionValues().SetValue(m_forgetGateInitVal);
        //if (m_inputGateInitVal > 0)
            bi->FunctionValues().SetValue(m_inputGateInitVal);
        //if (m_outputGateInitVal > 0)
            bo->FunctionValues().SetValue(m_outputGateInitVal);

        Whi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WHI%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
        Wci = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WCI%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

	    Whf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WHF%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
	    Wcf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WCF%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

	    Who = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WHO%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
	    Wco = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WCO%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

	    Whc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"WHC%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

        size_t layer1 = outputDim;
        
        delayHI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
        delayHF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
        delayHO = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
        delayHC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
		delayCI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
		delayCF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
		delayCC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize); 
		
        if(m_constInputGateValue)
        {
            //it = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"CONSTIT%d", iLayer), outputDim, mbSize);
            //it->NeedGradient() = false;
            //it->FunctionValues().SetValue(m_constInputGateValue);
            it = nullptr;
        }
        else
            it = ApplyNonlinearFunction(
		        m_net->Plus(
                    m_net->Plus(
                        m_net->Plus(
                            m_net->Times(Wxi, input), 
                                bi), 
                            m_net->Times(Whi, delayHI)),
                        m_net->DiagTimes(Wci, delayCI)), 0);

        if(it == nullptr)
        {
             bit = m_net->Tanh(
                            m_net->Plus(
                                m_net->Times(Wxc, input),
                                    m_net->Plus(
                                        m_net->Times(Whc, delayHC),
                                        bc
                                    )
                                )
                            );
        }
        else
        {
            bit = m_net->ElementTimes(it, 
                        m_net->Tanh(
                            m_net->Plus(
                                m_net->Times(Wxc, input),
                                    m_net->Plus(
                                        m_net->Times(Whc, delayHC),
                                        bc
                                    )
                                )
                            )
                        );
        }

        if(m_constForgetGateValue)
        {
            ft = nullptr;
        }
        else
            ft = ApplyNonlinearFunction(
		        m_net->Plus(
                    m_net->Plus(
                        m_net->Plus(
                            m_net->Times(Wxf, input), 
                            bf), 
                        m_net->Times(Whf, delayHF)),
                    m_net->DiagTimes(Wcf, delayCF)), 0);


        if(ft == nullptr)
        {
            bft = delayCC;
        }
        else
        {
            bft = m_net->ElementTimes(ft, delayCC);
        }

        ct = m_net->Plus(bft,bit);


        if(m_constOutputGateValue)
        {
            ot = nullptr;
        }
        else
            ot = ApplyNonlinearFunction(
		        m_net->Plus(
                    m_net->Plus(
                        m_net->Plus(
                            m_net->Times(Wxo, input), 
                            bo), 
                        m_net->Times(Who, delayHO)),
                    m_net->DiagTimes(Wco, ct)), 0);

        if (ot == nullptr)
        {
            output = m_net->Tanh(ct);
        }
        else
        {
            output = m_net->ElementTimes(ot, m_net->Tanh(ct));
        }
		
        delayHO->AttachInputs(output);
		delayHI->AttachInputs(output);
		delayHF->AttachInputs(output);
		delayHC->AttachInputs(output);
		delayCI->AttachInputs(ct);
		delayCF->AttachInputs(ct);
		delayCC->AttachInputs(ct);
		
        if (m_addDropoutNodes)
		    input = m_net->Dropout(output);
        else
            input = output;
        output = input;

        return (ComputationNode<ElemType>*) output; 
    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildLSTMNetworkFromDescription(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
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
                ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = {nullptr}; 

                input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                if(m_lookupTableOrder > 0)
                {
                    e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                    m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    output = m_net->LookupTable(e, input, L"LookupTable");

                    if (m_addDropoutNodes)
                        input = m_net->Dropout(output);
                    else
                        input = output;

                    outputFromEachLayer[1] = input;
                }

                /// direct connect from input node to output node

				int recur_idx = 0;
                int offset = m_lookupTableOrder > 0? 1 : 0;
                if (numHiddenLayers > 0)
                {
                    output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset]*(offset?m_lookupTableOrder:1), m_layerSizes[offset+1], input);
                    input = output;
                    outputFromEachLayer[offset+1]  = input;

                    for (int i=1 + offset; i<numHiddenLayers; i++)
                    {
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
						{
                            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i+1], input);

                            recur_idx++;
						}
						else
						{
                            u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                            b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", i), m_layerSizes[i+1], 1);
	                        output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net->Dropout(output);
                        else
                            input = output;

                        outputFromEachLayer[i+1] = input;
                    }
                }

                for (size_t i = offset; i < m_layerSizes.size(); i++)
                {
                    /// add direct connect from each layers' output to the layer before the output layer
                    output = BuildDirectConnect(randomSeed, mbSize, i, (i > 1)?m_layerSizes[i]:((offset == 0)?m_layerSizes[i]:m_layerSizes[i] * m_lookupTableOrder), m_layerSizes[numHiddenLayers], outputFromEachLayer[i], input);
                    if (output != nullptr) 
                        input = output;
                }

				w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net->Times(w, input, L"outputs");   

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                    input = m_net->Log(prior, L"LogOfPrior");
                    ComputationNodePtr 
                        scaledLogLikelihood = m_net->Minus(output, input, L"ScaledLogLikelihood");
                    m_net->OutputNodes().push_back(scaledLogLikelihood);
                }
                else 
                    m_net->OutputNodes().push_back(output);

                //add softmax layer (if prob is needed or KL reg adaptation is needed)
                output = m_net->Softmax(output, L"PosteriorProb");
                
            }

            m_net->ResetEvalTimeStamp();

            return *m_net;
    }

    template<class ElemType>
    ComputationNetwork<ElemType>& SimpleNetworkBuilder<ElemType>::BuildCLASSLSTMNetworkFromDescription(size_t mbSize)
    {
            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
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
                ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = {nullptr}; 

                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize, m_layerSizes[0] * mbSize);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = m_net->Mean(input);
                    b = m_net->InvStdDev(input);
                    output = m_net->PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                if(m_lookupTableOrder > 0)
                {
                    e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                    m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    output = m_net->LookupTable(e, input, L"LookupTable");

                    if (m_addDropoutNodes)
                        input = m_net->Dropout(output);
                    else
                        input = output;

                    outputFromEachLayer[1] = input;
                }

                /// direct connect from input node to output node

				int recur_idx = 0;
                int offset = m_lookupTableOrder > 0? 1 : 0;
                if (numHiddenLayers > 0)
                {
                    output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset]*(offset?m_lookupTableOrder:1), m_layerSizes[offset+1], input, true);
                    input = output;
                    outputFromEachLayer[offset+1]  = input;

                    for (int i=1 + offset; i<numHiddenLayers; i++)
                    {
						if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
						{
                            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i+1], input);

                            recur_idx++;
						}
						else
						{
                            u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                            b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", i), m_layerSizes[i+1], 1);
	                        output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
						}

					    if (m_addDropoutNodes)
						    input = m_net->Dropout(output);
                        else
                            input = output;

                        outputFromEachLayer[i+1] = input;
                    }
                }

                for (size_t i = offset; i < m_layerSizes.size(); i++)
                {
                    /// add direct connect from each layers' output to the layer before the output layer
                    output = BuildDirectConnect(randomSeed, mbSize, i, (i > 1)?m_layerSizes[i]:((offset == 0)?m_layerSizes[i]:m_layerSizes[i] * m_lookupTableOrder), m_layerSizes[numHiddenLayers], outputFromEachLayer[i], input);
                    if (output != nullptr) 
                        input = output;
                }

                // TODO: verify the change is okay
                // w = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers]*(MAX_WORDS_PER_CLASS+MAX_CLASSES)*mbSize*NUM_UTTS_IN_RECURRENT_ITER);
                w = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                // TODO: verify the change is okay
                //label = m_net->CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize, 2*mbSize*NUM_UTTS_IN_RECURRENT_ITER);
                label = m_net->CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
                
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = m_net->Times(w, input, L"outputs");   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = m_net->Mean(label);
                }

            }

            m_net->ResetEvalTimeStamp();

            return *m_net;
    }

}}}
