//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"

#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"
#include "SimpleNetworkBuilder.h"

#pragma warning (disable: 4189)     // (we have lots of unused variables to show how variables can be set up)

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildSimpleRNN(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

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
                        u = m_net->CreateLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
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
                            recur_idx++;
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

                        for (int i = 1; i<numHiddenLayers; i++)
                        {
                            //TODO: to figure out sparse matrix size
                            u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                            {
                                w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                                delay = m_net->Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i + 1], mbSize);
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

                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                    /*m_net->MatrixL2Reg(w , L"L1w");*/

                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
                    AddTrainAndEvalCriterionNodes(input, label, w, L"criterion", L"eval");

                    output = m_net->Times(w, input, L"outputs");

                    m_net->OutputNodes()->push_back(output);

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                    }

                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildClassEntropyNetwork(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr wrd2cls = nullptr, cls2idx = nullptr, clslogpostprob = nullptr, clsweight = nullptr;

                    if (m_vocabSize != m_layerSizes[numHiddenLayers + 1])
                        RuntimeError("BuildClassEntropyNetwork : vocabulary size should be the same as the output layer size");

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

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
                        u = m_net->CreateLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
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
                            recur_idx++;
                        }
                        else
                        {
                            b = m_net->CreateLearnableParameter(L"B0", m_layerSizes[1], 1);
                            m_net->InitLearnableParameters(b, m_uniformInit, randomSeed++, m_initValueScale);
                            output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
                        }

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;

                        for (int i = 1; i<numHiddenLayers; i++)
                        {
                            u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                            {
                                w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                                delay = m_net->Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i + 1], mbSize);
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

                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = m_net->CreateInputNode(L"labels", 4, mbSize);

                    clsweight = m_net->CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = m_net->Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                        clslogpostprob);

                    m_net->OutputNodes()->push_back(output);

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                    }
                }

                m_net->ResetEvalTimeStamp();

                return m_net;

            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildConditionalLSTMNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr gt = nullptr;
                    ComputationNodePtr clslogpostprob = nullptr;
                    ComputationNodePtr clsweight = nullptr;

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, L"LookupTable");

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                    else
                    {
                        LogicError("BuildCLASSLSTMNetworkFromDescription: LSTMNode cannot take sparse input. Need to project sparse input to continuous vector using LookupTable. Suggest using setups below\n layerSizes=$VOCABSIZE$:100:$HIDDIM$:$VOCABSIZE$ \nto have 100 dimension projection, and lookupTableOrder=1\n to project to a single window. To use larger context window, set lookupTableOrder=3 for example with width-3 context window.\n ");
                    }

                    int recur_idx = 0;
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        /// previously used function. now uses LSTMNode which is correct and fast
                        input = output;
                        for (int i = 1 + offset; i < numHiddenLayers; i++)
                        {
                            //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;
                        }
                    }

                    /// serve as a global bias term
                    gt = m_net->CreateInputNode(L"binaryFeature", m_auxFeatDim, 1);
                    m_net->FeatureNodes()->push_back(gt);
                    e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"AuxTrans%d", 0),
                        m_layerSizes[numHiddenLayers], m_auxFeatDim);
                    m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    u = ApplyNonlinearFunction(m_net->Times(e, gt), numHiddenLayers, L"TimesToGetGlobalBias");
                    output = m_net->Plus(input, u, L"PlusGlobalBias");
                    input = output;

                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = m_net->CreateInputNode(L"labels", 4, mbSize);

                    clsweight = m_net->CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = m_net->Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                        clslogpostprob);

                    output = m_net->Times(m_net->Transpose(w), input, L"outputs");

                    m_net->OutputNodes()->push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(output, L"PosteriorProb");
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            /**
            this builds an alignment based LM generator
            the aligment node takes a variable length input and relates each element to a variable length output
            */
            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildAlignmentDecoderNetworkFromDescription(ComputationNetwork<ElemType>* encoderNet,
                size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, encoderOutput = nullptr, e = nullptr,
                        b = nullptr, w = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, alignoutput = nullptr;
                    ComputationNodePtr clslogpostprob = nullptr;
                    ComputationNodePtr clsweight = nullptr;
                    ComputationNodePtr columnStride = nullptr, rowStride = nullptr;

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, L"LookupTable");

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                    else
                    {
                        LogicError("BuildCLASSLSTMNetworkFromDescription: LSTMNode cannot take sparse input. Need to project sparse input to continuous vector using LookupTable. Suggest using setups below\n layerSizes=$VOCABSIZE$:100:$HIDDIM$:$VOCABSIZE$ \nto have 100 dimension projection, and lookupTableOrder=1\n to project to a single window. To use larger context window, set lookupTableOrder=3 for example with width-3 context window.\n ");
                    }

                    int recur_idx = 0;
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;

                    /// the source network side output dimension needs to match the 1st layer dimension in the decoder network
                    std::vector<ComputationNodePtr> * encoderPairNodes = encoderNet->PairNodes();
                    if (encoderPairNodes->size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one pairoutput node as source node for the decoder network: ");

                    encoderOutput = m_net->PairNetwork((*encoderPairNodes)[0], L"pairNetwork");

                    /// the source network side output dimension needs to match the 1st layer dimension in the decoder network
                    std::vector<ComputationNodePtr> * encoderEvaluationNodes = encoderNet->OutputNodes();
                    if (encoderEvaluationNodes->size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one output node as source node for the decoder network: ");

                    if (numHiddenLayers > 0)
                    {
                        int i = 1 + offset;
                        u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i], m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1));
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                        delay = m_net->Delay(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i], mbSize);
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        //                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);

                        /// alignment node to get weights from source to target
                        /// this aligment node computes weights of the current hidden state after special encoder ending symbol to all 
                        /// states before the special encoder ending symbol. The weights are used to summarize all encoder inputs. 
                        /// the weighted sum of inputs are then used as the additional input to the LSTM input in the next layer
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"MatForSimilarity%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);

                        columnStride = m_net->CreateConstParameter(L"columnStride", 1, 1);
                        columnStride->FunctionValues().SetValue(1);
                        rowStride = m_net->CreateConstParameter(L"rowStride", 1, 1);
                        rowStride->FunctionValues().SetValue(0);
                        alignoutput = m_net->StrideTimes(encoderOutput, m_net->Softmax(m_net->StrideTimes(m_net->Times(m_net->Transpose(encoderOutput), e), delay, rowStride)), columnStride);

                        //                alignoutput = m_net->Times(encoderOutput, m_net->Softmax(m_net->Times(m_net->Times(m_net->Transpose(encoderOutput), e), delay)));

                        output = ApplyNonlinearFunction(
                            m_net->Plus(
                            m_net->Times(u, input), m_net->Times(w, alignoutput)), 0);
                        delay->AttachInputs(output);
                        input = output;

                        for (; i < numHiddenLayers; i++)
                        {
                            output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            //output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;
                        }

                    }


                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"OW%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = m_net->CreateInputNode(L"labels", 4, mbSize);

                    clsweight = m_net->CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = m_net->Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                        clslogpostprob);

                    output = m_net->Times(m_net->Transpose(w), input, L"outputs");

                    m_net->PairNodes()->push_back(input);

                    m_net->OutputNodes()->push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(output, L"PosteriorProb");
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLogBilinearNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr, featin = nullptr, e = nullptr;
                    ComputationNodePtr bi = nullptr;
                    ComputationNodePtr Wxi1 = nullptr, Wxi = nullptr;
                    ComputationNodePtr Wxi2 = nullptr, Wxi3 = nullptr, Wxi4 = nullptr;
                    ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, gt = nullptr, ct = nullptr, ht = nullptr;
                    ComputationNodePtr delayXI = nullptr, delayXII = nullptr, delayXIII = nullptr, delayXIV = nullptr;

                    //                input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);
                    featin = input;
                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    //used for lookuptable node unittest, will delete
                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
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
                        Wxi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"DD%d", ik), m_layerSizes[0], m_layerSizes[0]);
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
                        u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i] * (m_lookupTableOrder > 0 ? m_lookupTableOrder : 1));
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->Times(u, input);
                        input = output;
                        if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                        {
                            w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"R%d", i + 1), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                            delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i + 1], mbSize);
                            output = m_net->Plus(m_net->Times(w, delay), input);

                            delay->AttachInputs(output);
                            input = output;
                            recur_idx++;
                        }

                        bi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", i), m_layerSizes[i + 1], 1);
                        output = m_net->Plus(input, bi);

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;
                    }

                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
                    AddTrainAndEvalCriterionNodes(input, label, w);

                    output = m_net->Times(w, input, L"outputs");

                    m_net->OutputNodes()->push_back(output);

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                    }
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildNeuralProbNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr bi = nullptr;
                    ComputationNodePtr Wxi1 = nullptr, Wxi = nullptr;
                    ComputationNodePtr Wxi2 = nullptr, Wxi3 = nullptr, Wxi4 = nullptr;
                    ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, gt = nullptr, ct = nullptr, ht = nullptr;
                    ComputationNodePtr delayXI = nullptr, delayXII = nullptr, delayXIII = nullptr, delayXIV = nullptr;

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

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
                            Wxi2 = m_net->CreateLearnableParameter(L"WXI2", m_layerSizes[1], m_layerSizes[0]);
                            m_net->InitLearnableParameters(Wxi2, m_uniformInit, randomSeed++, m_initValueScale);
                            //TODO: to figure out sparse matrix size
                            Wxi3 = m_net->CreateLearnableParameter(L"WXI3", m_layerSizes[1], m_layerSizes[0]);
                            m_net->InitLearnableParameters(Wxi3, m_uniformInit, randomSeed++, m_initValueScale);
                            //TODO: to figure out sparse matrix size
                            Wxi4 = m_net->CreateLearnableParameter(L"WXI4", m_layerSizes[1], m_layerSizes[0]);
                            m_net->InitLearnableParameters(Wxi4, m_uniformInit, randomSeed++, m_initValueScale);
                            //TODO: to figure out sparse matrix size
                            Wxi1 = m_net->CreateLearnableParameter(L"WXI1", m_layerSizes[1], m_layerSizes[0]);
                            m_net->InitLearnableParameters(Wxi1, m_uniformInit, randomSeed++, m_initValueScale);
                            //TODO: to figure out sparse matrix size
                            Wxi = m_net->CreateLearnableParameter(L"WXI", m_layerSizes[1], m_layerSizes[0]);
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
                            recur_idx++;
                        }
                        else
                        {
                            output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), 0);
                        }

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;

                        for (int i = 1; i<numHiddenLayers; i++)
                        {
                            u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                            {
                                w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                                std::list<ComputationNodePtr> recurrent_loop;
                                delay = m_net->Delay(NULL, m_defaultHiddenActivity, m_layerSizes[i + 1], mbSize);
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
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                    //                b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], 1);
                    label = m_net->CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
                    AddTrainAndEvalCriterionNodes(input, label, w);

                    output = m_net->Times(w, input);

                    m_net->OutputNodes()->push_back(output);

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                    }
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildDirectConnect(unsigned long &randomSeed, size_t /*mbSize*/, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode)
            {
                ComputationNodePtr directOutput = nullptr, mergedNode = nullptr;

                for (size_t i = 0; i < m_directConnect.size(); i++)
                {
                    if (m_directConnect[i] == iLayer)
                    {
                        ComputationNodePtr directWIO = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"D%d", i), outputDim, inputDim);
                        m_net->InitLearnableParameters(directWIO, m_uniformInit, randomSeed++, m_initValueScale);
                        directOutput = ApplyNonlinearFunction(m_net->Times(directWIO, input), i);

                        ComputationNodePtr scalar = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"SV%d", i), 1, 1);
                        scalar->FunctionValues().SetValue((ElemType)0.01);
                        ComputationNodePtr scaled = m_net->Scale(scalar, directOutput, msra::strfun::wstrprintf(L"S%d", i));

                        mergedNode = m_net->Plus(toNode, scaled);
                    }
                }

                return mergedNode;
            }


            template<class ElemType>
            ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMComponent(unsigned long &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
            {

                size_t numHiddenLayers = m_layerSizes.size() - 2;

                ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                ComputationNodePtr Wxf = nullptr, Whf = nullptr, Wcf = nullptr, bf = nullptr, Wxc = nullptr, Whc = nullptr, bc = nullptr;
                ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, ct = nullptr, ht = nullptr;
                ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC = nullptr, delayCF = nullptr, delayCC = nullptr;
                ComputationNodePtr directWIO = nullptr, directInput = nullptr, directOutput = nullptr;
                ComputationNodePtr bit = nullptr, bft = nullptr, bct = nullptr;

                input = inputObs;
                Wxo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXO%d", iLayer), outputDim, inputDim);
                Wxi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXI%d", iLayer), outputDim, inputDim);
                Wxf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXF%d", iLayer), outputDim, inputDim);
                Wxc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXC%d", iLayer), outputDim, inputDim);

                m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
                m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
                m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
                m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

                bo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bo%d", iLayer), outputDim, 1);
                bc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bc%d", iLayer), outputDim, 1);
                bi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", iLayer), outputDim, 1);
                bf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bf%d", iLayer), outputDim, 1);
                //if (m_forgetGateInitVal > 0)
                bf->FunctionValues().SetValue(m_forgetGateInitVal);
                //if (m_inputGateInitVal > 0)
                bi->FunctionValues().SetValue(m_inputGateInitVal);
                //if (m_outputGateInitVal > 0)
                bo->FunctionValues().SetValue(m_outputGateInitVal);

                Whi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHI%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
                Wci = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCI%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

                Whf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHF%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
                Wcf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCF%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

                Who = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHO%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
                Wco = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCO%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

                Whc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHC%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

                size_t layer1 = outputDim;

                delayHI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHO = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);

                if (m_constInputGateValue)
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

                if (it == nullptr)
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

                if (m_constForgetGateValue)
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


                if (ft == nullptr)
                {
                    bft = delayCC;
                }
                else
                {
                    bft = m_net->ElementTimes(ft, delayCC);
                }

                ct = m_net->Plus(bft, bit);


                if (m_constOutputGateValue)
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
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildSeqTrnLSTMNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    ULONG randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                    ComputationNodePtr Wxf = nullptr, Whf = nullptr, Wcf = nullptr, bf = nullptr, Wxc = nullptr, Whc = nullptr, bc = nullptr;
                    ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, ct = nullptr, ht = nullptr;
                    ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC = nullptr, delayCF = nullptr, delayCC = nullptr;
                    ComputationNodePtr directWIO = nullptr, directInput = nullptr, directOutput = nullptr;
                    ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };
                    ComputationNodePtr trans = nullptr;

                    input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
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
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {
                        for (int i = offset; i<numHiddenLayers; i++)
                        {
                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                            {
                                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i] * (offset ? m_lookupTableOrder : 1), m_layerSizes[i + 1], input);
                                input = output;

                                recur_idx++;
                            }
                            else
                            {
                                u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i] * (offset ? m_lookupTableOrder : 1));
                                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                                b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                                output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
                            }

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;
                        }
                    }

                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"TimesBeforeSoftMax%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    output = m_net->Times(w, input, L"outputsBeforeSoftmax");

                    trans = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"TransProb%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers + 1]);
                    trans->FunctionValues().SetValue((ElemType)1.0 / m_layerSizes[numHiddenLayers + 1]);
                    //          m_net->InitLearnableParameters(trans, m_uniformInit, randomSeed++, m_initValueScale);
                    trans->NeedGradient() = true;
                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
                    AddTrainAndEvalCriterionNodes(output, label, nullptr, L"CRFTrainCriterion", L"CRFEvalCriterion", nullptr, trans);

                    input = output;
                    output = m_net->SequenceDecoder(label, input, trans, L"outputs");
                    m_net->OutputNodes()->push_back(output);

                    output = m_net->Softmax(input, L"PosteriorProb");

                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildCLASSLSTMNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                    ComputationNodePtr clslogpostprob = nullptr;
                    ComputationNodePtr clsweight = nullptr;

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, L"LookupTable");

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;
                    }
                    else
                    {
                        LogicError("BuildCLASSLSTMNetworkFromDescription: LSTMNode cannot take sparse input. Need to project sparse input to continuous vector using LookupTable. Suggest using setups below\n layerSizes=$VOCABSIZE$:100:$HIDDIM$:$VOCABSIZE$ \nto have 100 dimension projection, and lookupTableOrder=1\n to project to a single window. To use larger context window, set lookupTableOrder=3 for example with width-3 context window.\n ");
                    }

                    int recur_idx = 0;
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        /// previously used function. now uses LSTMNode which is correct and fast
                        input = output;
                        for (int i = 1 + offset; i <numHiddenLayers; i++)
                        {
                            //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;
                        }
                    }

                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = m_net->CreateInputNode(L"labels", 4, mbSize);

                    clsweight = m_net->CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = m_net->Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                        clslogpostprob);

                    output = m_net->Times(m_net->Transpose(w), input, L"outputs");

                    m_net->OutputNodes()->push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(output, L"PosteriorProb");
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMNodeComponent(ULONG &randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
            {

                size_t numHiddenLayers = m_layerSizes.size() - 2;

                ComputationNodePtr input = nullptr, output = nullptr;
                ComputationNodePtr wInputGate = nullptr, wForgetGate = nullptr, wOutputGate = nullptr, wMemoryCellMatrix = nullptr;

                input = inputObs;
                size_t nDim = inputDim + outputDim + 2;
                wInputGate = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WINPUTGATE%d", iLayer), outputDim, nDim);
                m_net->InitLearnableParameters(wInputGate, m_uniformInit, randomSeed++, m_initValueScale);
                wInputGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_inputGateInitVal);  /// init to input gate bias
                wForgetGate = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WFORGETGATE%d", iLayer), outputDim, nDim);
                m_net->InitLearnableParameters(wForgetGate, m_uniformInit, randomSeed++, m_initValueScale);
                wForgetGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_forgetGateInitVal); /// init to forget gate bias
                wOutputGate = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WOUTPUTGATE%d", iLayer), outputDim, nDim);
                m_net->InitLearnableParameters(wOutputGate, m_uniformInit, randomSeed++, m_initValueScale);
                wOutputGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_outputGateInitVal);/// init to output gate bias
                wMemoryCellMatrix = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WMEMORYCELLWEIGHT%d", iLayer), outputDim, inputDim + outputDim + 1);
                m_net->InitLearnableParameters(wMemoryCellMatrix, m_uniformInit, randomSeed++, m_initValueScale);
                wMemoryCellMatrix->FunctionValues().ColumnSlice(0, 1).SetValue(0);/// init to memory cell bias

                output = m_net->LSTM(inputObs, wInputGate, wForgetGate, wOutputGate, wMemoryCellMatrix, msra::strfun::wstrprintf(L"LSTM%d", iLayer));

#ifdef DEBUG_DECODER
                wInputGate->FunctionValues().SetValue((ElemType)0.01);
                wForgetGate->FunctionValues().SetValue((ElemType)0.01);
                wOutputGate->FunctionValues().SetValue((ElemType)0.01);
                wMemoryCellMatrix->FunctionValues().SetValue((ElemType)0.01);
#endif

                if (m_addDropoutNodes)
                    input = m_net->Dropout(output);
                else
                    input = output;
                output = input;

                return (ComputationNode<ElemType>*) output;
            }

            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    ULONG randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                    ComputationNodePtr Wxf = nullptr, Whf = nullptr, Wcf = nullptr, bf = nullptr, Wxc = nullptr, Whc = nullptr, bc = nullptr;
                    ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, ct = nullptr, ht = nullptr;
                    ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC = nullptr, delayCF = nullptr, delayCC = nullptr;
                    ComputationNodePtr directWIO = nullptr, directInput = nullptr, directOutput = nullptr;
                    ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };

                    if (m_sparse_input)
                        input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    else
                        input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);

                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, L"LookupTable");
#ifdef DEBUG_DECODER
                        e->FunctionValues().SetValue((ElemType)0.01);
#endif

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;

                        outputFromEachLayer[1] = input;
                    }

                    /// direct connect from input node to output node

                    int recur_idx = 0;
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {

                        output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        //                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        /// previously used function. now uses LSTMNode which is correct and fast
                        input = output;
                        outputFromEachLayer[offset + 1] = input;

                        for (int i = 1 + offset; i<numHiddenLayers; i++)
                        {
                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
                            {

                                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                                //                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                                // previously used function, now uses LSTMnode, which is fast and correct

                                recur_idx++;
                            }
                            else
                            {
                                u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                                b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                                output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
                            }

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;

                            outputFromEachLayer[i + 1] = input;
                        }
                    }

                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef DEBUG_DECODER
                    w->FunctionValues().SetValue((ElemType)0.01);
#endif
                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
                    AddTrainAndEvalCriterionNodes(input, label, w);

                    output = m_net->Times(w, input, L"outputs");

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                        input = m_net->Log(prior, L"LogOfPrior");
                        ComputationNodePtr
                            scaledLogLikelihood = m_net->Minus(output, input, L"ScaledLogLikelihood");
                        m_net->OutputNodes()->push_back(scaledLogLikelihood);
                    }
                    else
                        m_net->OutputNodes()->push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(output, L"PosteriorProb");

                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            /**
            This is encoder LSTM described in the following papers:
            H. Sutskever, O. Vinyals and Q. V. Le, "Sequence to sequence learning with neural networks", http://arxiv.org/abs/1409.3215

            The following code constructs the encoder and, to construct decoder, use BuildLSTMNetworkFromDescription

            Developed by Kaisheng Yao
            This is used in the following works:
            K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion, submitted to Interspeech 2015
            */
            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMEncoderNetworkFromDescription(size_t mbSize)
            {

                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    ULONG randomSeed = 1;

                    size_t i = 0;
                    size_t numHiddenLayers = m_layerSizes.size() - 1;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;

                    if (m_sparse_input)
                        input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    else
                        input = m_net->CreateInputNode(L"features", m_layerSizes[0], mbSize);

                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"EncoderE%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, L"EncoderLookupTable");
#ifdef DEBUG_DECODER
                        e->FunctionValues().SetValue((ElemType)0.01);
#endif

                        if (m_addDropoutNodes)
                            input = m_net->Dropout(output);
                        else
                            input = output;
                        i++;
                    }

                    /// direct connect from input node to output node

                    int recur_idx = 0;
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        input = output;
                        i++;

                        for (; i<numHiddenLayers; i++)
                        {
                            //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;
                        }
                    }

                    m_net->OutputNodes()->push_back(output);
                    m_net->PairNodes()->push_back(output);

                }

                m_net->ResetEvalTimeStamp();
                return m_net;
            }


            /**
            Build unidirectional LSTM p(y_t | y_t-1, x_1^t)

            Because the past prediction is used, decoding requires beam search decoder

            Developed by Kaisheng Yao
            This is used in the following work
            K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion" submitted to Interspeech 2015
            */
            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildUnidirectionalLSTMNetworksFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    ULONG randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();
                    size_t dims = 0;

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, Wxo = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    vector<ComputationNodePtr> streams;
                    vector<size_t> streamdims;
                    ComputationNodePtr inputforward = nullptr, inputbackward = nullptr, inputletter = nullptr;
                    ComputationNodePtr transcription_prediction = nullptr;

                    map<wstring, size_t> featDim;

                    assert(m_streamSizes.size() > 0);
                    inputbackward = m_net->CreateInputNode(L"featureDelayedTarget", m_streamSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(inputbackward);
                    featDim[L"featureDelayedTarget"] = m_streamSizes[0];

                    inputletter = m_net->CreateInputNode(L"ltrForward", m_streamSizes[1], mbSize);
                    m_net->FeatureNodes()->push_back(inputletter);
                    featDim[L"ltrForward"] = m_streamSizes[1];

                    size_t layerIdx = 0;
                    size_t idx = 0;
                    int recur_idx = 0;
                    for (typename vector<ComputationNodePtr>::iterator p = m_net->FeatureNodes()->begin();
                        p != m_net->FeatureNodes()->end(); p++, idx++)
                    {
                        layerIdx = 0;  /// reset layer id because each input stream starts from layer 0
                        input = *p;
                        if (m_applyMeanVarNorm)
                        {
                            input = *p;
                            w = m_net->Mean(input);
                            b = m_net->InvStdDev(input);
                            output = m_net->PerDimMeanVarNormalization(input, w, b);

                            input = output;
                        }

                        size_t idim = input->FunctionValues().GetNumRows();
                        assert(m_lookupTabelOrderSizes.size() == m_streamSizes.size());

                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"Embedding%d", idx), m_layerSizes[1], idim / m_lookupTabelOrderSizes[idx]);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, msra::strfun::wstrprintf(L"LOOKUP%d", idx));

                        streamdims.push_back(m_layerSizes[1] * m_lookupTabelOrderSizes[idx]);
                        input = output;
                        streams.push_back(input);
                    }

                    layerIdx++;

                    output = (ComputationNodePtr)m_net->Parallel(streams[0], streams[1], L"Parallel0");
                    input = output;
                    dims = streamdims[0] + streamdims[1];

                    /// now merge the streams
                    if (numHiddenLayers > 0)
                    {
                        while (layerIdx < numHiddenLayers)
                        {
                            switch (m_rnnType){
                            case UNIDIRECTIONALLSTM:
                                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx, dims, m_layerSizes[layerIdx + 1], input);
                                break;
                            default:
                                LogicError("This is for unidorectional LSTM model. Check rnntype to see whether it is UNIDIRECTIONALLSTMWITHPASTPREDICTION or TRANSDUCER");
                            }

                            layerIdx++;
                            dims = m_layerSizes[layerIdx];
                            input = output;
                        }
                    }

                    /// directly connect transcription model output/feature to the output layer
                    Wxo = m_net->CreateLearnableParameter(L"ConnectToLowerLayers", m_layerSizes[numHiddenLayers + 1], m_layerSizes[layerIdx]);
                    m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);

                    output = m_net->Times(Wxo, input);
                    input = output;

                    /// here uses "labels", so only one label from multiple stream inputs are used.
                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);

                    AddTrainAndEvalCriterionNodes(input, label, w);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(input, L"outputs");

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                        input = m_net->Log(prior, L"LogOfPrior");
                        ComputationNodePtr
                            scaledLogLikelihood = m_net->Minus(output, input, L"ScaledLogLikelihood");
                        m_net->OutputNodes()->push_back(scaledLogLikelihood);
                    }
                    else
                        m_net->OutputNodes()->push_back(output);

                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template<class ElemType>
            ComputationNode<ElemType>* SimpleNetworkBuilder<ElemType>::BuildLSTMComponentWithMultiInputs(ULONG &randomSeed, size_t mbSize, size_t iLayer, const vector<size_t>& inputDim, size_t outputDim, const vector<ComputationNodePtr>& inputObs, bool inputWeightSparse)
            {

                size_t numHiddenLayers = m_layerSizes.size() - 2;

                ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                ComputationNodePtr Wxf = nullptr, Whf = nullptr, Wcf = nullptr, bf = nullptr, Wxc = nullptr, Whc = nullptr, bc = nullptr;
                ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, ct = nullptr, ht = nullptr;
                ComputationNodePtr delayHI = nullptr, delayCI = nullptr, delayHO = nullptr, delayHF = nullptr, delayHC = nullptr, delayCF = nullptr, delayCC = nullptr;
                ComputationNodePtr directWIO = nullptr, directInput = nullptr, directOutput = nullptr;
                ComputationNodePtr bit = nullptr, bft = nullptr, bct = nullptr;
                ComputationNodePtr streamsxi = nullptr, streamsxo = nullptr, streamsxf = nullptr, streamsxc = nullptr;

                for (size_t sidx = 0; sidx < inputObs.size(); sidx++)
                {
                    input = inputObs[sidx];
                    if (inputWeightSparse)
                    {
                        Wxo = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXO%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxi = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXI%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxf = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXF%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxc = m_net->CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXC%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                    }
                    else
                    {
                        Wxo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXO%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXI%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXF%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                        Wxc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WXC%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                    }
                    m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
                    m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
                    m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
                    m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

                    streamsxi = (streamsxi == nullptr) ? m_net->Times(Wxi, input) : m_net->Plus(streamsxi, m_net->Times(Wxi, input));
                    streamsxf = (streamsxf == nullptr) ? m_net->Times(Wxf, input) : m_net->Plus(streamsxf, m_net->Times(Wxf, input));
                    streamsxc = (streamsxc == nullptr) ? m_net->Times(Wxc, input) : m_net->Plus(streamsxc, m_net->Times(Wxc, input));
                    streamsxo = (streamsxo == nullptr) ? m_net->Times(Wxo, input) : m_net->Plus(streamsxo, m_net->Times(Wxo, input));
                }


                bo = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bo%d", iLayer), outputDim, 1);
                bc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bc%d", iLayer), outputDim, 1);
                bi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", iLayer), outputDim, 1);
                bf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"bf%d", iLayer), outputDim, 1);
                //if (m_forgetGateInitVal > 0)
                bf->FunctionValues().SetValue(m_forgetGateInitVal);
                //if (m_inputGateInitVal > 0)
                bi->FunctionValues().SetValue(m_inputGateInitVal);
                //if (m_outputGateInitVal > 0)
                bo->FunctionValues().SetValue(m_outputGateInitVal);

                Whi = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHI%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
                Wci = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCI%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

                Whf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHF%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
                Wcf = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCF%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

                Who = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHO%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
                Wco = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WCO%d", iLayer), outputDim, 1);
                m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

                Whc = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"WHC%d", iLayer), outputDim, outputDim);
                m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

                size_t layer1 = outputDim;

                delayHI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHO = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayHC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCI = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCF = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);
                delayCC = m_net->Delay(NULL, m_defaultHiddenActivity, layer1, mbSize);

                if (m_constInputGateValue)
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
                    streamsxi,
                    bi),
                    m_net->Times(Whi, delayHI)),
                    m_net->DiagTimes(Wci, delayCI)), 0);

                if (it == nullptr)
                {
                    bit = m_net->Tanh(
                        m_net->Plus(
                        streamsxc,
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
                        streamsxc,
                        m_net->Plus(
                        m_net->Times(Whc, delayHC),
                        bc
                        )
                        )
                        )
                        );
                }

                if (m_constForgetGateValue)
                {
                    ft = nullptr;
                }
                else
                    ft = ApplyNonlinearFunction(
                    m_net->Plus(
                    m_net->Plus(
                    m_net->Plus(
                    streamsxf,
                    bf),
                    m_net->Times(Whf, delayHF)),
                    m_net->DiagTimes(Wcf, delayCF)), 0);


                if (ft == nullptr)
                {
                    bft = delayCC;
                }
                else
                {
                    bft = m_net->ElementTimes(ft, delayCC);
                }

                ct = m_net->Plus(bft, bit);


                if (m_constOutputGateValue)
                {
                    ot = nullptr;
                }
                else
                    ot = ApplyNonlinearFunction(
                    m_net->Plus(
                    m_net->Plus(
                    m_net->Plus(
                    streamsxo,
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

            /**
            Build a bi-directional LSTM network to compute the following
            p(y_t | y_1^{t-1}, x_1^T)
            The target side for y_t is a LSTM language model with past prediction y_{t-1} as its input. This language model also uses
            the outputs from the forwawrd direction LSTM and the output from the backward direction LSTM that are operated on the source side.

            Developed by Kaisheng Yao.
            This is used in the following works:
            K. Yao, G. Zweig, "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion, submitted to Interspeech 2015
            */
            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildBiDirectionalLSTMNetworksFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    ULONG randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr, Wxo;
                    ComputationNodePtr forwardInput = nullptr, forwardOutput = nullptr, backwardInput = nullptr, backwardOutput = nullptr;
                    vector<ComputationNodePtr> streams;
                    vector<size_t> streamdims;
                    ComputationNodePtr inputprediction = nullptr, inputletter = nullptr, ngram = nullptr;
                    ComputationNodePtr ltrSource = nullptr;
                    size_t ltrDim = 0;

                    map<wstring, size_t> featDim;

                    size_t ltrSrcIdx = 1;
                    /// create projections to use delay predictions
                    inputprediction = m_net->CreateInputNode(L"featureDelayedTarget", m_streamSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(inputprediction);

                    inputletter = m_net->CreateInputNode(L"ltrForward", m_streamSizes[1], mbSize);
                    m_net->FeatureNodes()->push_back(inputletter);
                    featDim[L"ltrForward"] = m_streamSizes[1];

                    size_t layerIdx = 0;
                    size_t idx = 0;
                    int recur_idx = 0;
                    for (typename vector<ComputationNodePtr>::iterator p = m_net->FeatureNodes()->begin();
                        p != m_net->FeatureNodes()->end(); p++, idx++)
                    {
                        layerIdx = 0;  /// reset layer id because each input stream starts from layer 0
                        input = *p;
                        if (m_applyMeanVarNorm)
                        {
                            input = *p;
                            w = m_net->Mean(input);
                            b = m_net->InvStdDev(input);
                            output = m_net->PerDimMeanVarNormalization(input, w, b);

                            input = output;
                        }

                        size_t idim = input->FunctionValues().GetNumRows();
                        assert(m_lookupTabelOrderSizes.size() == m_streamSizes.size());

                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"Embedding%d", idx), m_layerSizes[1], idim / m_lookupTabelOrderSizes[idx]);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = m_net->LookupTable(e, input, msra::strfun::wstrprintf(L"LOOKUP%d", idx));

                        streamdims.push_back(m_layerSizes[1] * m_lookupTabelOrderSizes[idx]);
                        input = output;
                        streams.push_back(input);

                        if (idx == ltrSrcIdx)
                        {
                            ltrSource = input;
                            ltrDim = m_layerSizes[1] * m_lookupTabelOrderSizes[idx];
                        }
                    }

                    layerIdx++;

                    /// glue the two streams
                    forwardInput = (ComputationNodePtr)m_net->Parallel(streams[0], streams[1], L"Parallel0");

                    if (numHiddenLayers > 0)
                    {
                        /// forward direction
//                        forwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 100, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                        forwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 100, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                        forwardInput = forwardOutput;

                        backwardInput = (ComputationNodePtr)m_net->TimeReverse(ltrSource);
//                        backwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 200, ltrDim, m_layerSizes[layerIdx + 1], backwardInput);
                        backwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 200, ltrDim, m_layerSizes[layerIdx + 1], backwardInput);
                        backwardInput = backwardOutput;

                        layerIdx++;

                        while (layerIdx < numHiddenLayers - 1)
                        {
//                            forwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 100, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], forwardInput);
                            forwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 100, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], forwardInput);
                            forwardInput = forwardOutput;

//                            backwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 200, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], backwardInput);
                            backwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 200, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], backwardInput);
                            backwardInput = backwardOutput;

                            layerIdx++;
                        }

                        backwardOutput = (ComputationNodePtr)m_net->TimeReverse(backwardInput);
                    }

                    streams.clear();
                    streamdims.clear();
                    streams.push_back(forwardOutput);
                    streamdims.push_back(m_layerSizes[layerIdx]);
                    streams.push_back(backwardOutput);
                    streamdims.push_back(m_layerSizes[layerIdx]);

                    /// glue the two streams
                    forwardInput = (ComputationNodePtr)m_net->Parallel(streams[0], streams[1], L"Parallel1");

//                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                    output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);

                    input = output;
                    layerIdx++;

                    /// directly connect transcription model output/feature to the output layer
                    Wxo = m_net->CreateLearnableParameter(L"ConnectToLowerLayers", m_layerSizes[numHiddenLayers + 1], m_layerSizes[layerIdx]);
                    m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);

                    output = m_net->Times(Wxo, input);
                    input = output;

                    /// here uses "labels", so only one label from multiple stream inputs are used.
                    label = m_net->CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);

                    AddTrainAndEvalCriterionNodes(input, label);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = m_net->Softmax(input, L"outputs");

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                        input = m_net->Log(prior, L"LogOfPrior");
                        ComputationNodePtr
                            scaledLogLikelihood = m_net->Minus(output, input, L"ScaledLogLikelihood");
                        m_net->OutputNodes()->push_back(scaledLogLikelihood);
                    }
                    else
                        m_net->OutputNodes()->push_back(output);

                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }
            template<class ElemType>
            ComputationNetwork<ElemType>* SimpleNetworkBuilder<ElemType>::BuildNCELSTMNetworkFromDescription(size_t mbSize)
            {
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;
                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, e = nullptr, delay = nullptr, output = nullptr, label = nullptr, prior = nullptr;
                    ComputationNodePtr Wxo = nullptr, Who = nullptr, Wco = nullptr, bo = nullptr, Wxi = nullptr, Whi = nullptr, Wci = nullptr, bi = nullptr;
                    ComputationNodePtr clslogpostprob = nullptr;
                    ComputationNodePtr bias = nullptr;
                    ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };

                    input = m_net->CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes()->push_back(input);

                    if (m_applyMeanVarNorm)
                    {
                        w = m_net->Mean(input);
                        b = m_net->InvStdDev(input);
                        output = m_net->PerDimMeanVarNormalization(input, w, b);

                        input = output;
                    }

                    if (m_lookupTableOrder > 0)
                    {
                        e = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
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
                    int offset = m_lookupTableOrder > 0 ? 1 : 0;
                    if (numHiddenLayers > 0)
                    {
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        input = output;
                        outputFromEachLayer[offset + 1] = input;

                        for (int i = 1 + offset; i<numHiddenLayers; i++)
                        {
                            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
                            {
                                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                                recur_idx++;
                            }
                            else
                            {
                                u = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                                b = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                                output = ApplyNonlinearFunction(m_net->Plus(m_net->Times(u, input), b), i);
                            }

                            if (m_addDropoutNodes)
                                input = m_net->Dropout(output);
                            else
                                input = output;

                            outputFromEachLayer[i + 1] = input;
                        }
                    }

                    for (size_t i = offset; i < m_layerSizes.size(); i++)
                    {
                        /// add direct connect from each layers' output to the layer before the output layer
                        output = BuildDirectConnect(randomSeed, mbSize, i, (i > 1) ? m_layerSizes[i] : ((offset == 0) ? m_layerSizes[i] : m_layerSizes[i] * m_lookupTableOrder), m_layerSizes[numHiddenLayers], outputFromEachLayer[i], input);
                        if (output != nullptr)
                            input = output;
                    }

                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = m_net->CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = m_net->CreateInputNode(L"labels", 2 * (this->nce_noises + 1), mbSize);

                    bias = m_net->CreateLearnableParameter(L"BiasVector", 1, m_layerSizes[m_layerSizes.size() - 1]);
                    bias->FunctionValues().SetValue((ElemType)-std::log(m_layerSizes[m_layerSizes.size() - 1]));
                    //m_net->InitLearnableParameters(bias, m_uniformInit, randomSeed++, std::log(m_layerSizes[m_layerSizes.size() - 1])* m_initValueScale);
                    //clslogpostprob = m_net->Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeNCEBasedCrossEntropy", L"EvalNodeNCEBasedCrossEntrpy", bias);

                    m_net->OutputNodes()->push_back(output);

                    if (m_needPrior)
                    {
                        prior = m_net->Mean(label);
                    }
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
            }

            template class SimpleNetworkBuilder<float>;
            template class SimpleNetworkBuilder<double>;

        }
    }
}
