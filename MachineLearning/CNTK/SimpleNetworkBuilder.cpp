//
// <copyright file="SimpleNetworkBuilder.cpp" company="Microsoft">
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
#include "ComputationNetworkBuilder.h"
#include "SGD.h"
#include "SimpleNetworkBuilder.h"

#pragma warning (disable: 4189)     // (we have lots of unused variables to show how variables can be set up)

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNetworkFromDescription(ComputationNetwork* encoderNet)
    {
        size_t mbSize = 1;
        ComputationNetworkPtr net;

        // TODO: this seems to call for a switch statement
        if (m_rnnType == SIMPLENET)
            net = BuildSimpleDNN();
        else if (m_rnnType == SIMPLERNN)
            net = BuildSimpleRNN(mbSize);
        else if (m_rnnType == LSTM)
            net = BuildLSTMNetworkFromDescription(mbSize);
        else if (m_rnnType == CLASSLSTM)
            net = BuildCLASSLSTMNetworkFromDescription(mbSize);
        else if (m_rnnType == NCELSTM)
            net = BuildNCELSTMNetworkFromDescription(mbSize);
        else if (m_rnnType == CLASSLM)
            net = BuildClassEntropyNetwork(mbSize);
        else if (m_rnnType == LBLM)
            net = BuildLogBilinearNetworkFromDescription(mbSize);
        else if (m_rnnType == NPLM)
            net = BuildNeuralProbNetworkFromDescription(mbSize);
        else if (m_rnnType == CLSTM)
            net = BuildConditionalLSTMNetworkFromDescription(mbSize);
        else if (m_rnnType == RCRF)
            net = BuildSeqTrnLSTMNetworkFromDescription(mbSize);
        else if (m_rnnType == LSTMENCODER)
            net = BuildLSTMEncoderNetworkFromDescription(mbSize);
        else if (m_rnnType == UNIDIRECTIONALLSTM)
            net = BuildUnidirectionalLSTMNetworksFromDescription(mbSize);
        else if (m_rnnType == BIDIRECTIONALLSTM)
            net = BuildBiDirectionalLSTMNetworksFromDescription(mbSize);
        else if (m_rnnType == ALIGNMENTSIMILARITYGENERATOR)
            net = BuildAlignmentDecoderNetworkFromDescription(encoderNet, mbSize);
        else if (m_rnnType == ALIGNMENTSIMILARITYGFORWARDDECODER)
            net = BuildAlignmentForwardDecoderNetworkFromDescription(encoderNet, mbSize);
        else
            LogicError("BuildNetworkFromDescription: invalid m_rnnType %d", (int)m_rnnType);

        net->ValidateNetwork(false/*allowFragment*/, true/*bAllowNoCriterion*/);	// no criterion possible because  ...TODO: what's the reason?
        return net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildSimpleDNN()
    {

        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t mbSize = 3; //this is not the actual minibatch size. only used in the validataion process

            size_t numHiddenLayers = m_layerSizes.size() - 2;
            ComputationNodePtr input, w, b, output, label, prior, scaledLogLikelihood;

            input = builder.Input(m_layerSizes[0], mbSize, L"features");
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input, L"MeanOfFeatures");
                b = builder.InvStdDev(input, L"InvStdOfFeatures");
                output = builder.PerDimMeanVarNormalization(input, w, b, L"MVNormalizedFeatures");

                input = output;
            }

            if (numHiddenLayers > 0)
            {
                w = builder.Parameter(m_layerSizes[1], m_layerSizes[0], L"W0");
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                b = builder.Parameter(m_layerSizes[1], 1, L"B0");
                output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, L"W0*features"), b, L"W0*features+B0"), 0, L"H1");

                if (m_addDropoutNodes)
                    input = builder.Dropout(output, L"DropH1");
                else
                    input = output;

                for (int i = 1; i<numHiddenLayers; i++)
                {
                    wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
                    wstring nameOfB = msra::strfun::wstrprintf(L"B%d", i);
                    wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", i);
                    wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                    wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
                    wstring nameOfH = msra::strfun::wstrprintf(L"H%d", i + 1);

                    w = builder.Parameter(m_layerSizes[i + 1], m_layerSizes[i], nameOfW);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                    b = builder.Parameter(m_layerSizes[i + 1], 1, nameOfB);
                    output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus), i, nameOfH);

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output, L"Drop" + nameOfH);
                    else
                        input = output;
                }
            }

            wstring nameOfW = msra::strfun::wstrprintf(L"W%d", numHiddenLayers);
            wstring nameOfB = msra::strfun::wstrprintf(L"B%d", numHiddenLayers);
            wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", numHiddenLayers - 1);
            wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
            wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;

            w = builder.Parameter(m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers], nameOfW);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            b = builder.Parameter(m_layerSizes[numHiddenLayers + 1], 1, nameOfB);
            output = builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus);
            m_net->RenameNode(output, L"HLast");

            label = builder.Input(m_layerSizes[numHiddenLayers + 1], mbSize, L"labels");

            AddTrainAndEvalCriterionNodes(output, label);

            if (m_needPrior)
            {
                prior = builder.Mean(label, L"Prior");
                input = builder.Log(prior, L"LogOfPrior");

                //following two lines are needed only if true probability is needed
                //output = builder.Softmax(output);
                //output = builder.Log(output);

                scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
                m_net->OutputNodes().push_back(scaledLogLikelihood);
            }
            else
            {
                m_net->OutputNodes().push_back(output);
            }

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(output, L"PosteriorProb");
            //m_net->OutputNodes().push_back(output);
        }

        m_net->ResetEvalTimeStamp();
        return m_net;
    }

    // Note: while ComputationNode and CompuationNetwork are (supposed to be) independent of ElemType, it is OK to keep this class dependent.
    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildSimpleRNN(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, pastValue, output, label, prior;

            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            int recur_idx = 0; 
            if (numHiddenLayers > 0)
            {
                //TODO: to figure out sparse matrix size
                u = builder.CreateLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
                {
                    w = builder.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize, 1); 
                    /// unless there is a good algorithm to detect loops, use this explicit setup
                    output = ApplyNonlinearFunction(
                        builder.Plus(
                            builder.Times(u, input), builder.Times(w, pastValue)), 0);
                    pastValue->AttachInputs(output);
                    recur_idx ++;
                }
                else
                {
                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), 0);
                    //output = builder.Times(u, input);
                }

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;

                for (int i=1; i<numHiddenLayers; i++)
                {
                    //TODO: to figure out sparse matrix size
                    u = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
                    {
                        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize, 1);
                        /// unless there is a good algorithm to detect loops, use this explicit setup
                        output = ApplyNonlinearFunction(
                            builder.Plus(
                                builder.Times(u, input), builder.Times(w, pastValue)), 0);
                        pastValue->AttachInputs(output);
                        recur_idx++;
                    }
                    else
                    {
                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            /*m_net->MatrixL2Reg(w , L"L1w");*/

            label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
            AddTrainAndEvalCriterionNodes(input, label, w, L"criterion", L"eval");

            output = builder.Times(w, input, L"outputs");   
                
            m_net->OutputNodes().push_back(output);

            if (m_needPrior)
                prior = builder.Mean(label);

            }

            m_net->ResetEvalTimeStamp();

            return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildClassEntropyNetwork(size_t mbSize)
    {
            ComputationNetworkBuilder<ElemType> builder(*m_net);

            if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                unsigned long randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

                size_t numRecurrentLayers = m_recurrentLayers.size(); 

                ComputationNodePtr input, w, b, u, pastValue, output, label, prior;
                ComputationNodePtr wrd2cls, cls2idx, clslogpostprob, clsweight;

                if (m_vocabSize != m_layerSizes[numHiddenLayers + 1])
                    RuntimeError("BuildClassEntropyNetwork : vocabulary size should be the same as the output layer size");

                input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = builder.Mean(input);
                    b = builder.InvStdDev(input);
                    output = builder.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                int recur_idx = 0; 
                if (numHiddenLayers > 0)
                {
                    u = builder.CreateLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
                    {
                        w = builder.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[1], mbSize, 1); 
                        /// unless there is a good algorithm to detect loops, use this explicit setup
                        output = ApplyNonlinearFunction(
                            builder.Plus(
                                builder.Times(u, input), builder.Times(w, pastValue)), 0);
                        pastValue->AttachInputs(output);
                        recur_idx ++;
                    }
                    else
                    {
                        b = builder.CreateLearnableParameter(L"B0", m_layerSizes[1], 1);
                        m_net->InitLearnableParameters(b, m_uniformInit, randomSeed++, m_initValueScale);
                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), 0);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;

                    for (int i=1; i<numHiddenLayers; i++)
                    {
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
                        {
                            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                            pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i+1], mbSize, 1); 
                            /// unless there is a good algorithm to detect loops, use this explicit setup
                            output = ApplyNonlinearFunction(
                                builder.Plus(
                                    builder.Times(u, input), builder.Times(w, pastValue)), 0);
                            pastValue->AttachInputs(output);
                            recur_idx++;
                        }
                        else
                        {
                            output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                        }

                        if (m_addDropoutNodes)
                            input = builder.Dropout(output);
                        else
                            input = output;
                    }
                }

                /// need to have [input_dim x output_dim] matrix
                /// e.g., [200 x 10000], where 10000 is the vocabulary size
                /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                /// the label is a dense matrix. each element is the word index
                label = builder.CreateInputNode(L"labels", 4, mbSize);

                clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

                output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy", 
                    clslogpostprob);
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = builder.Mean(label);
                }
            }

            m_net->ResetEvalTimeStamp();

            return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildConditionalLSTMNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
            ComputationNodePtr gt;
            ComputationNodePtr clslogpostprob;
            ComputationNodePtr clsweight;

            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if (m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"LookupTable");

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
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
                //           output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                /// previously used function. now uses LSTMNode which is correct and fast
                input = output;
                for (int i = 1 + offset; i < numHiddenLayers; i++)
                {
                    //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                    output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            /// serve as a global bias term
            gt = builder.CreateInputNode(L"binaryFeature", m_auxFeatDim, 1);
            m_net->FeatureNodes().push_back(gt);
            e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"AuxTrans%d", 0),
                m_layerSizes[numHiddenLayers], m_auxFeatDim);
            m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
            u = ApplyNonlinearFunction(builder.Times(e, gt), numHiddenLayers, L"TimesToGetGlobalBias");
            output = builder.Plus(input, u, L"PlusGlobalBias");
            input = output;

            /// need to have [input_dim x output_dim] matrix
            /// e.g., [200 x 10000], where 10000 is the vocabulary size
            /// this is for speed-up issue as per word matrix can be simply obtained using column slice
            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

            /// the label is a dense matrix. each element is the word index
            label = builder.CreateInputNode(L"labels", 4, mbSize);

            clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
            clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

            output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                clslogpostprob);

            output = builder.Times(builder.Transpose(w), input, L"outputs");

            m_net->OutputNodes().push_back(output);

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(output, L"PosteriorProb");
        }

        m_net->ResetEvalTimeStamp();

        return m_net;
            }

            /**
            this builds an alignment based LM generator
            the aligment node takes a variable length input and relates each element to a variable length output
            */
    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildAlignmentForwardDecoderNetworkFromDescription(ComputationNetwork* encoderNet, size_t mbSize)
    {
                ComputationNetworkBuilder<ElemType> builder(*m_net);
                if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input, encoderOutput, e,
                        b, w, u, pastValue, output, label, alignoutput;
                    ComputationNodePtr clslogpostprob;
                    ComputationNodePtr clsweight;
                    ComputationNodePtr columnStride, rowStride;

                    input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes().push_back(input);

                    if (m_lookupTableOrder > 0)
                    {
                        e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = builder.LookupTable(e, input, L"LookupTable");

                        if (m_addDropoutNodes)
                            input = builder.Dropout(output);
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
                    std::vector<ComputationNodeBasePtr>& encoderPairNodes = encoderNet->PairNodes();
                    if (encoderPairNodes.size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one pairoutput node as source node for the decoder network: ");

                    encoderOutput = builder.PairNetwork(dynamic_pointer_cast<ComputationNode<ElemType>>(encoderPairNodes[0]), L"pairNetwork");

                    /// the source network side output dimension needs to match the 1st layer dimension in the decoder network
                    std::vector<ComputationNodeBasePtr>& encoderEvaluationNodes = encoderNet->OutputNodes();
                    if (encoderEvaluationNodes.size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one output node as source node for the decoder network: ");

                    if (numHiddenLayers > 0)
                    {
                        int i = 1 + offset;
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i], m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1));
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i], mbSize, 1);
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        //                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);

                        /// alignment node to get weights from source to target
                        /// this aligment node computes weights of the current hidden state after special encoder ending symbol to all 
                        /// states before the special encoder ending symbol. The weights are used to summarize all encoder inputs. 
                        /// the weighted sum of inputs are then used as the additional input to the LSTM input in the next layer
                        e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"MatForSimilarity%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);

                        columnStride = builder.CreateLearnableParameter(L"columnStride", 1, 1);
                        columnStride->FunctionValues().SetValue(1);
                        columnStride->SetParameterUpdateRequired(false); 
                        rowStride = builder.CreateLearnableParameter(L"rowStride", 1, 1);
                        rowStride->FunctionValues().SetValue(0);
                        rowStride->SetParameterUpdateRequired(false);
                        alignoutput = builder.StrideTimes(encoderOutput, builder.Softmax(builder.StrideTimes(builder.Times(builder.Transpose(encoderOutput), e), pastValue, rowStride)), columnStride);

                        //                alignoutput = builder.Times(encoderOutput, builder.Softmax(builder.Times(builder.Times(builder.Transpose(encoderOutput), e), pastValue)));

                        output = ApplyNonlinearFunction(
                            builder.Plus(
                            builder.Times(u, input), builder.Times(w, alignoutput)), 0);
                        pastValue->AttachInputs(output);
                        input = output;

                        for (; i < numHiddenLayers; i++)
                        {
                            //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = builder.Dropout(output);
                            else
                                input = output;
                        }

                    }


                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"OW%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = builder.CreateInputNode(L"labels", 4, mbSize);

                    clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

                    output = builder.Times(builder.Transpose(w), input, L"outputs");

                    m_net->PairNodes().push_back(input);

                    m_net->OutputNodes().push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = builder.Softmax(output, L"PosteriorProb");
                }

                m_net->ResetEvalTimeStamp();

        return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildAlignmentDecoderNetworkFromDescription(ComputationNetwork* encoderNet, size_t mbSize)
    {
                ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
                {
                    unsigned long randomSeed = 1;

                    size_t numHiddenLayers = m_layerSizes.size() - 2;

                    size_t numRecurrentLayers = m_recurrentLayers.size();

                    ComputationNodePtr input, encoderOutput, e,
                        b, w, u, pastValue, output, label, alignoutput;
                    ComputationNodePtr clslogpostprob;
                    ComputationNodePtr clsweight;
                    ComputationNodePtr columnStride, rowStride;

                    input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                    m_net->FeatureNodes().push_back(input);

                    if (m_lookupTableOrder > 0)
                    {
                        e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                        output = builder.LookupTable(e, input, L"LookupTable");

                        if (m_addDropoutNodes)
                            input = builder.Dropout(output);
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
                    std::vector<ComputationNodeBasePtr>& encoderPairNodes = encoderNet->PairNodes();
                    if (encoderPairNodes.size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one pairoutput node as source node for the decoder network: ");

                    encoderOutput = builder.PairNetwork(dynamic_pointer_cast<ComputationNode<ElemType>>(encoderPairNodes[0]), L"pairNetwork");

                    /// the source network side output dimension needs to match the 1st layer dimension in the decoder network
                    std::vector<ComputationNodeBasePtr>& encoderEvaluationNodes = encoderNet->OutputNodes();
                    if (encoderEvaluationNodes.size() != 1)
                        LogicError("BuildAlignmentDecoderNetworkFromDescription: encoder network should have only one output node as source node for the decoder network: ");

                    if (numHiddenLayers > 0)
                    {
                        int i = 1 + offset;
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i], m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1));
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t)m_layerSizes[i], mbSize, 1);
                        //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                        //                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);

                        /// alignment node to get weights from source to target
                        /// this aligment node computes weights of the current hidden state after special encoder ending symbol to all 
                        /// states before the special encoder ending symbol. The weights are used to summarize all encoder inputs. 
                        /// the weighted sum of inputs are then used as the additional input to the LSTM input in the next layer
                        e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"MatForSimilarity%d", i), m_layerSizes[i], m_layerSizes[i]);
                        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);

                        columnStride = builder.CreateLearnableParameter(L"columnStride", 1, 1);
                        columnStride->FunctionValues().SetValue(1);
                        columnStride->SetParameterUpdateRequired(false); 
                        rowStride = builder.CreateLearnableParameter(L"rowStride", 1, 1);
                        rowStride->FunctionValues().SetValue(0);
                        rowStride->SetParameterUpdateRequired(false); 
                        alignoutput = builder.StrideTimes(encoderOutput, builder.Softmax(builder.StrideTimes(builder.Times(builder.Transpose(encoderOutput), e), pastValue, rowStride)), columnStride);

                        //                alignoutput = builder.Times(encoderOutput, builder.Softmax(builder.Times(builder.Times(builder.Transpose(encoderOutput), e), pastValue)));

                        output = ApplyNonlinearFunction(
                            builder.Plus(
                            builder.Times(u, input), builder.Times(w, alignoutput)), 0);
                        pastValue->AttachInputs(output);
                        input = output;

                        for (; i < numHiddenLayers; i++)
                        {
                            //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                            output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                            if (m_addDropoutNodes)
                                input = builder.Dropout(output);
                            else
                                input = output;
                        }

                    }


                    /// need to have [input_dim x output_dim] matrix
                    /// e.g., [200 x 10000], where 10000 is the vocabulary size
                    /// this is for speed-up issue as per word matrix can be simply obtained using column slice
                    w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"OW%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    /// the label is a dense matrix. each element is the word index
                    label = builder.CreateInputNode(L"labels", 4, mbSize);

                    clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
                    m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
                    clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

                    output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                        clslogpostprob);

                    output = builder.Times(builder.Transpose(w), input, L"outputs");

                    m_net->PairNodes().push_back(input);

                    m_net->OutputNodes().push_back(output);

                    //add softmax layer (if prob is needed or KL reg adaptation is needed)
                    output = builder.Softmax(output, L"PosteriorProb");
                }

                m_net->ResetEvalTimeStamp();

                return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildLogBilinearNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
            {
                unsigned long randomSeed = 1;

                size_t numHiddenLayers = m_layerSizes.size()-2;

                size_t numRecurrentLayers = m_recurrentLayers.size();

                ComputationNodePtr input, w, b, u, pastValue, output, label, prior, featin, e;
                ComputationNodePtr bi=nullptr;
                ComputationNodePtr Wxi1=nullptr, Wxi=nullptr;
                ComputationNodePtr Wxi2=nullptr, Wxi3=nullptr, Wxi4=nullptr;
                ComputationNodePtr ot=nullptr, it=nullptr, ft=nullptr, gt=nullptr, ct=nullptr, ht=nullptr;
                ComputationNodePtr pastValueXI, pastValueXII, pastValueXIII, pastValueXIV;

//                input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
                input = builder.CreateInputNode(L"features", m_layerSizes[0], mbSize);
                featin = input;
                m_net->FeatureNodes().push_back(input);

                if (m_applyMeanVarNorm)
                {
                    w = builder.Mean(input);
                    b = builder.InvStdDev(input);
                    output = builder.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                //used for lookuptable node unittest, will delete
                if(m_lookupTableOrder > 0)
                {
                    e = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                    m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                    output = builder.LookupTable(e, input, L"Lookuptatble");

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }

                int recur_idx = 0; 
                /// unless there is a good algorithm to detect loops, use this explicit setup
                int ik = 1; 
                output = input;
                while (ik <= m_maOrder)
                {
                    pastValueXI = 
                        builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, ik, msra::strfun::wstrprintf(L"pastValue%d", ik)); 
                    pastValueXI->SetParameterUpdateRequired(false);
                    pastValueXI->AttachInputs(input);
                    //TODO: to figure out sparse matrix size
                    Wxi = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"DD%d", ik), m_layerSizes[0], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

                    it = builder.Plus(output, builder.Times(Wxi, pastValueXI));
                    output = it;

                    ik++;
                }
                
                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;

                for (int i = m_lookupTableOrder > 0 ? 1 : 0; i<numHiddenLayers; i++)
                {
                    u = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i] * (m_lookupTableOrder > 0 ? m_lookupTableOrder : 1));
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                    output= builder.Times(u, input);
                    input = output;
                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
                    {
                        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"R%d", i+1), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize, 1);
                        output = builder.Plus(builder.Times(w, pastValue), input);

                        pastValue->AttachInputs(output);
                        input = output;
                        recur_idx++;
                    }

                    bi = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"bi%d", i), m_layerSizes[i+1], 1);
                    output = builder.Plus(input, bi);

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            
                w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
                AddTrainAndEvalCriterionNodes(input, label, w);
                
                output = builder.Times(w, input, L"outputs");   
                
                m_net->OutputNodes().push_back(output);

                if (m_needPrior)
                {
                    prior = builder.Mean(label);
                }
            }

            m_net->ResetEvalTimeStamp();

                return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNeuralProbNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input = nullptr, w = nullptr, b = nullptr, u = nullptr, pastValue, output = nullptr, label = nullptr, prior = nullptr;
            ComputationNodePtr bi = nullptr;
            ComputationNodePtr Wxi1 = nullptr, Wxi = nullptr;
            ComputationNodePtr Wxi2 = nullptr, Wxi3 = nullptr, Wxi4 = nullptr;
            ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, gt = nullptr, ct = nullptr, ht = nullptr;
            ComputationNodePtr pastValueXI, pastValueXII, pastValueXIII, pastValueXIV;

            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            int recur_idx = 0;
            if (numHiddenLayers > 0)
            {
                bi = builder.CreateLearnableParameter(L"bi0", m_layerSizes[1], 1);

                pastValueXI = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, 1);
                pastValueXII = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, 2);
                pastValueXIII = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, 3);
                pastValueXIV = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], mbSize, 4);
                pastValueXI->AttachInputs(input);
                pastValueXII->AttachInputs(input);
                pastValueXIII->AttachInputs(input);
                pastValueXIV->AttachInputs(input);

                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
                {
                    //TODO: to figure out sparse matrix size
                    Wxi2 = builder.CreateLearnableParameter(L"WXI2", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi2, m_uniformInit, randomSeed++, m_initValueScale);
                    //TODO: to figure out sparse matrix size
                    Wxi3 = builder.CreateLearnableParameter(L"WXI3", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi3, m_uniformInit, randomSeed++, m_initValueScale);
                    //TODO: to figure out sparse matrix size
                    Wxi4 = builder.CreateLearnableParameter(L"WXI4", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi4, m_uniformInit, randomSeed++, m_initValueScale);
                    //TODO: to figure out sparse matrix size
                    Wxi1 = builder.CreateLearnableParameter(L"WXI1", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi1, m_uniformInit, randomSeed++, m_initValueScale);
                    //TODO: to figure out sparse matrix size
                    Wxi = builder.CreateLearnableParameter(L"WXI", m_layerSizes[1], m_layerSizes[0]);
                    m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

                    /// unless there is a good algorithm to detect loops, use this explicit setup
                    it = builder.Plus(
                        builder.Tanh(
                        builder.Plus(
                        builder.Times(Wxi4, pastValueXIV),
                        builder.Plus(
                        builder.Times(Wxi3, pastValueXIII),
                        builder.Plus(
                        builder.Times(Wxi2, pastValueXII),
                        builder.Plus(
                        builder.Times(Wxi1, pastValueXI),
                        builder.Times(Wxi, input))
                        )
                        )
                        )),
                        bi);
                    output = it;
                    pastValueXI->SetParameterUpdateRequired(false);
                    pastValueXII->SetParameterUpdateRequired(false);
                    pastValueXIII->SetParameterUpdateRequired(false);
                    pastValueXIV->SetParameterUpdateRequired(false);
                    recur_idx++;
                }
                else
                {
                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), 0);
                }

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;

                for (int i=1; i<numHiddenLayers; i++)
                {
                    u = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"U%d", i), m_layerSizes[i+1], m_layerSizes[i]);
                    m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
                    {
                        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", i), m_layerSizes[i+1], m_layerSizes[i+1]);
                        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                        std::list<ComputationNodeBasePtr> recurrent_loop;
                        pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[i+1], mbSize, 1);
                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), builder.Times(w, pastValue)), i);
                        pastValue->AttachInputs(output);
                        recur_idx++;
                    }
                    else
                    {
                        output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            //TODO: to figure out sparse matrix size
            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            //                b = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], 1);
            label = builder.CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers+1], mbSize);
            AddTrainAndEvalCriterionNodes(input, label, w);

            output = builder.Times(w, input);

            m_net->OutputNodes().push_back(output);

            if (m_needPrior)
            {
                prior = builder.Mean(label);
            }
        }

        m_net->ResetEvalTimeStamp();

        return m_net;
    }

    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildDirectConnect(unsigned long &randomSeed, size_t /*mbSize*/, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        ComputationNodePtr directOutput, mergedNode;

        for (size_t i = 0; i < m_directConnect.size(); i++)
        {
            if (m_directConnect[i] == iLayer)
            {
                ComputationNodePtr directWIO = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"D%d", i), outputDim, inputDim);
                m_net->InitLearnableParameters(directWIO, m_uniformInit, randomSeed++, m_initValueScale);
                directOutput = ApplyNonlinearFunction(builder.Times(directWIO, input),i);

                ComputationNodePtr scalar = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"SV%d", i), 1, 1);
                scalar->FunctionValues().SetValue((ElemType)0.01);
                ComputationNodePtr scaled = builder.Scale(scalar, directOutput, msra::strfun::wstrprintf(L"S%d", i));

                mergedNode = builder.Plus(toNode, scaled);
            }
        }

        return mergedNode;
    }


    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMComponent(unsigned long &randomSeed, size_t mbSize, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        size_t numHiddenLayers = m_layerSizes.size()-2;

        ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
        ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
        ComputationNodePtr Wxf, Whf, Wcf, bf, Wxc, Whc, bc;
        ComputationNodePtr ot, it, ft, ct, ht;
        ComputationNodePtr pastValueHI, pastValueCI, pastValueHO, pastValueHF, pastValueHC, pastValueCF, pastValueCC;
        ComputationNodePtr directWIO, directInput, directOutput;
        ComputationNodePtr bit, bft, bct;

        input = inputObs;
        Wxo = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WXO%d", iLayer), outputDim, inputDim);        
        Wxi = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WXI%d", iLayer), outputDim, inputDim);
        Wxf = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WXF%d", iLayer), outputDim, inputDim);
        Wxc = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WXC%d", iLayer), outputDim, inputDim);

        m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
        m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

        bo = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"bo%d", iLayer), outputDim, 1);
        bc = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"bc%d", iLayer), outputDim, 1);
        bi = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"bi%d", iLayer), outputDim, 1);
        bf = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"bf%d", iLayer), outputDim, 1);
        //if (m_forgetGateInitVal > 0)
        bf->FunctionValues().SetValue(m_forgetGateInitVal);
        //if (m_inputGateInitVal > 0)
        bi->FunctionValues().SetValue(m_inputGateInitVal);
        //if (m_outputGateInitVal > 0)
        bo->FunctionValues().SetValue(m_outputGateInitVal);

        Whi = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WHI%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
        Wci = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WCI%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

        Whf = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WHF%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
        Wcf = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WCF%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

        Who = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WHO%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
        Wco = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WCO%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

        Whc = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"WHC%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

        size_t layer1 = outputDim;
        
        pastValueHI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHO = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        
        if(m_constInputGateValue)
        {
            //it = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"CONSTIT%d", iLayer), outputDim, mbSize);
            //it->SetParameterUpdateRequired(false);
            //it->FunctionValues().SetValue(m_constInputGateValue);
            it = nullptr;
        }
        else
            it = ApplyNonlinearFunction(
                builder.Plus(
                    builder.Plus(
                        builder.Plus(
                            builder.Times(Wxi, input), 
                                bi), 
                            builder.Times(Whi, pastValueHI)),
                        builder.DiagTimes(Wci, pastValueCI)), 0);

        if(it == nullptr)
        {
             bit = builder.Tanh(
                            builder.Plus(
                                builder.Times(Wxc, input),
                                    builder.Plus(
                                        builder.Times(Whc, pastValueHC),
                                        bc
                                    )
                                )
                            );
        }
        else
        {
            bit = builder.ElementTimes(it, 
                        builder.Tanh(
                            builder.Plus(
                                builder.Times(Wxc, input),
                                    builder.Plus(
                                        builder.Times(Whc, pastValueHC),
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
                builder.Plus(
                    builder.Plus(
                        builder.Plus(
                            builder.Times(Wxf, input), 
                            bf), 
                        builder.Times(Whf, pastValueHF)),
                    builder.DiagTimes(Wcf, pastValueCF)), 0);


        if(ft == nullptr)
        {
            bft = pastValueCC;
        }
        else
        {
            bft = builder.ElementTimes(ft, pastValueCC);
        }

        ct = builder.Plus(bft,bit);


        if(m_constOutputGateValue)
        {
            ot = nullptr;
        }
        else
            ot = ApplyNonlinearFunction(
                builder.Plus(
                    builder.Plus(
                        builder.Plus(
                            builder.Times(Wxo, input), 
                            bo), 
                        builder.Times(Who, pastValueHO)),
                    builder.DiagTimes(Wco, ct)), 0);

        if (ot == nullptr)
        {
            output = builder.Tanh(ct);
        }
        else
        {
            output = builder.ElementTimes(ot, builder.Tanh(ct));
        }
        
        pastValueHO->AttachInputs(output);
        pastValueHI->AttachInputs(output);
        pastValueHF->AttachInputs(output);
        pastValueHC->AttachInputs(output);
        pastValueCI->AttachInputs(ct);
        pastValueCF->AttachInputs(ct);
        pastValueCC->AttachInputs(ct);
        
        if (m_addDropoutNodes)
            input = builder.Dropout(output);
        else
            input = output;
        output = input;

        return output; 
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildSeqTrnLSTMNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
            ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
            ComputationNodePtr Wxf, Whf, Wcf, bf, Wxc, Whc, bc;
            ComputationNodePtr ot, it, ft, ct, ht;
            ComputationNodePtr pastValueHI, pastValueCI, pastValueHO, pastValueHF, pastValueHC, pastValueCF, pastValueCC;
            ComputationNodePtr directWIO, directInput, directOutput;
            ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };
            ComputationNodePtr trans;

            input = builder.CreateInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if (m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"LookupTable");

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
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
                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i+1)
                    {
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i] * (offset ? m_lookupTableOrder : 1), m_layerSizes[i + 1], input);
                        input = output;
 
                        recur_idx++;
                    }
                    else
                    {
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i] * (offset ? m_lookupTableOrder : 1));
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        b = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                        output = ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"TimesBeforeSoftMax%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

            output = builder.Times(w, input, L"outputsBeforeSoftmax");

            trans = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"TransProb%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers + 1]);
            trans->FunctionValues().SetValue((ElemType)1.0 / m_layerSizes[numHiddenLayers + 1]);
//          m_net->InitLearnableParameters(trans, m_uniformInit, randomSeed++, m_initValueScale);
            trans->SetParameterUpdateRequired(true);
            label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
            AddTrainAndEvalCriterionNodes(output, label, nullptr, L"CRFTrainCriterion", L"CRFEvalCriterion", nullptr, trans);

            input = output;
            output = builder.SequenceDecoder(label, input, trans, L"outputs");
            m_net->OutputNodes().push_back(output);

            output = builder.Softmax(input, L"PosteriorProb");
        }

        m_net->ResetEvalTimeStamp();

        return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildCLASSLSTMNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size()-2;

            size_t numRecurrentLayers = m_recurrentLayers.size(); 

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
            ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
            ComputationNodePtr clslogpostprob;
            ComputationNodePtr clsweight;

            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if(m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"E%d", 0), m_layerSizes[1], m_layerSizes[0]/m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"LookupTable");

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
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
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            /// need to have [input_dim x output_dim] matrix
            /// e.g., [200 x 10000], where 10000 is the vocabulary size
            /// this is for speed-up issue as per word matrix can be simply obtained using column slice
            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

            /// the label is a dense matrix. each element is the word index
            label = builder.CreateInputNode(L"labels", 4, mbSize);

            clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
            clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

            output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                    clslogpostprob);

            output = builder.Times(builder.Transpose(w), input, L"outputs");

            m_net->OutputNodes().push_back(output);

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(output, L"PosteriorProb");
        }

        m_net->ResetEvalTimeStamp();

                return m_net;
    }

#if 1
    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMNodeComponent(ULONG &, size_t , size_t , size_t , ComputationNodePtr )
    {
        InvalidArgument("BuildLSTMNodeComponent: LSTMNode is no longer available. You should not get here.");
    }
#else
    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMNodeComponent(ULONG &randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        size_t numHiddenLayers = m_layerSizes.size() - 2;

        ComputationNodePtr input, output;
        ComputationNodePtr wInputGate, wForgetGate, wOutputGate, wMemoryCellMatrix;

        input = inputObs;
        size_t nDim = inputDim + outputDim + 2;
        wInputGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WINPUTGATE%d", iLayer), outputDim, nDim);
        m_net->InitLearnableParameters(wInputGate, m_uniformInit, randomSeed++, m_initValueScale);
        wInputGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_inputGateInitVal);  /// init to input gate bias
        wForgetGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WFORGETGATE%d", iLayer), outputDim, nDim);
        m_net->InitLearnableParameters(wForgetGate, m_uniformInit, randomSeed++, m_initValueScale);
        wForgetGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_forgetGateInitVal); /// init to forget gate bias
        wOutputGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WOUTPUTGATE%d", iLayer), outputDim, nDim);
        m_net->InitLearnableParameters(wOutputGate, m_uniformInit, randomSeed++, m_initValueScale);
        wOutputGate->FunctionValues().ColumnSlice(0, 1).SetValue(m_outputGateInitVal);/// init to output gate bias
        wMemoryCellMatrix = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WMEMORYCELLWEIGHT%d", iLayer), outputDim, inputDim + outputDim + 1);
        m_net->InitLearnableParameters(wMemoryCellMatrix, m_uniformInit, randomSeed++, m_initValueScale);
        wMemoryCellMatrix->FunctionValues().ColumnSlice(0, 1).SetValue(0);/// init to memory cell bias

        output = builder.LSTM(inputObs, wInputGate, wForgetGate, wOutputGate, wMemoryCellMatrix, msra::strfun::wstrprintf(L"LSTM%d", iLayer));

#ifdef DEBUG_DECODER
        wInputGate->FunctionValues().SetValue((ElemType)0.01);
        wForgetGate->FunctionValues().SetValue((ElemType)0.01);
        wOutputGate->FunctionValues().SetValue((ElemType)0.01);
        wMemoryCellMatrix->FunctionValues().SetValue((ElemType)0.01);
#endif

        if (m_addDropoutNodes)
            input = builder.Dropout(output);
        else
            input = output;
        output = input;

        return output;
    }
#endif

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildLSTMNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
            ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
            ComputationNodePtr Wxf, Whf, Wcf, bf, Wxc, Whc, bc;
            ComputationNodePtr ot, it, ft, ct, ht;
            ComputationNodePtr pastValueHI, pastValueCI, pastValueHO, pastValueHF, pastValueHC, pastValueCF, pastValueCC;
            ComputationNodePtr directWIO, directInput, directOutput;
            ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };

            if (m_sparse_input)
                input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            else
                input = builder.CreateInputNode(L"features", m_layerSizes[0], mbSize);

            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if (m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"LookupTable");
#ifdef DEBUG_DECODER
                e->FunctionValues().SetValue((ElemType)0.01);
#endif

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;

                outputFromEachLayer[1] = input;
            }

            /// direct connect from input node to output node

            int recur_idx = 0;
            int offset = m_lookupTableOrder > 0 ? 1 : 0;
            if (numHiddenLayers > 0)
            {

                //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                /// previously used function. now uses LSTMNode which is correct and fast
                input = output;
                outputFromEachLayer[offset + 1] = input;

                for (int i = 1 + offset; i<numHiddenLayers; i++)
                {
                    if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
                    {

                        //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                        // previously used function, now uses LSTMnode, which is fast and correct

                        recur_idx++;
                    }
                    else
                    {
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        b = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                        output = ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;

                    outputFromEachLayer[i + 1] = input;
                }
            }

            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
#ifdef DEBUG_DECODER
            w->FunctionValues().SetValue((ElemType)0.01);
#endif
            label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);
            AddTrainAndEvalCriterionNodes(input, label, w);

            output = builder.Times(w, input, L"outputs");

            if (m_needPrior)
            {
                prior = builder.Mean(label);
                input = builder.Log(prior, L"LogOfPrior");
                ComputationNodePtr
                    scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
                m_net->OutputNodes().push_back(scaledLogLikelihood);
            }
            else
                m_net->OutputNodes().push_back(output);

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(output, L"PosteriorProb");

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
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildLSTMEncoderNetworkFromDescription(size_t mbSize)
    {

        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t i = 0;
            size_t numHiddenLayers = m_layerSizes.size() - 1;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;

            if (m_sparse_input)
                input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            else
                input = builder.CreateInputNode(L"features", m_layerSizes[0], mbSize);

            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if (m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"EncoderE%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"EncoderLookupTable");
#ifdef DEBUG_DECODER
                e->FunctionValues().SetValue((ElemType)0.01);
#endif

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;
                i++;
            }

            /// direct connect from input node to output node

            int recur_idx = 0;
            int offset = m_lookupTableOrder > 0 ? 1 : 0;
            if (numHiddenLayers > 0)
            {
                //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
                input = output;
                i++;

                for (; i<numHiddenLayers; i++)
                {
                    //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                    output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
                    else
                        input = output;
                }
            }

            m_net->OutputNodes().push_back(output);
            m_net->PairNodes().push_back(output);  /// need to provide pairnodes so that the next layer of network can connect to this network
            m_net->EvaluationNodes().push_back(output);

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
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildUnidirectionalLSTMNetworksFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();
            size_t dims = 0;

            ComputationNodePtr input, w, b, u, e, Wxo, output, label, prior;
            vector<ComputationNodePtr> streams;
            vector<size_t> streamdims;
            ComputationNodePtr inputforward, inputbackward, inputletter;
            ComputationNodePtr transcription_prediction;

            map<wstring, size_t> featDim;

            assert(m_streamSizes.size() > 0);
            inputbackward = builder.CreateInputNode(L"featurepastValueedTarget", m_streamSizes[0], mbSize);
            m_net->FeatureNodes().push_back(inputbackward);
            featDim[L"featurepastValueedTarget"] = m_streamSizes[0];

            inputletter = builder.CreateInputNode(L"ltrForward", m_streamSizes[1], mbSize);
            m_net->FeatureNodes().push_back(inputletter);
            featDim[L"ltrForward"] = m_streamSizes[1];

            size_t layerIdx = 0;
            size_t idx = 0;
            int recur_idx = 0;
            for (auto p = m_net->FeatureNodes().begin(); p != m_net->FeatureNodes().end(); p++, idx++)
            {
                layerIdx = 0;  /// reset layer id because each input stream starts from layer 0
                input = dynamic_pointer_cast<ComputationNode<ElemType>>(*p);
                if (m_applyMeanVarNorm)
                {
                    input = dynamic_pointer_cast<ComputationNode<ElemType>>(*p);
                    w = builder.Mean(input);
                    b = builder.InvStdDev(input);
                    output = builder.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                size_t idim = input->GetNumRows();
                assert(m_lookupTabelOrderSizes.size() == m_streamSizes.size());

                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"Embedding%d", idx), m_layerSizes[1], idim / m_lookupTabelOrderSizes[idx]);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, msra::strfun::wstrprintf(L"LOOKUP%d", idx));

                streamdims.push_back(m_layerSizes[1] * m_lookupTabelOrderSizes[idx]);
                input = output;
                streams.push_back(input);
            }

            layerIdx++;

            output = (ComputationNodePtr)builder.Parallel(streams[0], streams[1], L"Parallel0");
            input = output;
            dims = streamdims[0] + streamdims[1];

            /// now merge the streams
            if (numHiddenLayers > 0)
            {
                while (layerIdx < numHiddenLayers)
                {
                    switch (m_rnnType){
                    case UNIDIRECTIONALLSTM:
                        //output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx, dims, m_layerSizes[layerIdx + 1], input);
                        output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx, dims, m_layerSizes[layerIdx + 1], input);
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
            Wxo = builder.CreateLearnableParameter(L"ConnectToLowerLayers", m_layerSizes[numHiddenLayers + 1], m_layerSizes[layerIdx]);
            m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);

            output = builder.Times(Wxo, input);
            input = output;

            /// here uses "labels", so only one label from multiple stream inputs are used.
            label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);

            AddTrainAndEvalCriterionNodes(input, label, w);

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(input, L"outputs");

            if (m_needPrior)
            {
                prior = builder.Mean(label);
                input = builder.Log(prior, L"LogOfPrior");
                ComputationNodePtr scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
                m_net->OutputNodes().push_back(scaledLogLikelihood);
            }
            else
                m_net->OutputNodes().push_back(output);
        }

        m_net->ResetEvalTimeStamp();

        return m_net;
    }

    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMComponentWithMultiInputs(ULONG &randomSeed, size_t mbSize, size_t iLayer, const vector<size_t>& inputDim, size_t outputDim, const vector<ComputationNodePtr>& inputObs, bool inputWeightSparse)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
        ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
        ComputationNodePtr Wxf, Whf, Wcf, bf, Wxc, Whc, bc;
        ComputationNodePtr ot, it, ft, ct, ht;
        ComputationNodePtr pastValueHI, pastValueCI, pastValueHO, pastValueHF, pastValueHC, pastValueCF, pastValueCC;
        ComputationNodePtr directWIO, directInput, directOutput;
        ComputationNodePtr bit, bft, bct;
        ComputationNodePtr streamsxi, streamsxo, streamsxf, streamsxc;

        for (size_t sidx = 0; sidx < inputObs.size(); sidx++)
        {
            input = inputObs[sidx];
            if (inputWeightSparse)
            {
                Wxo = builder.CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXO%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxi = builder.CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXI%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxf = builder.CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXF%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxc = builder.CreateSparseLearnableParameter(msra::strfun::wstrprintf(L"WXC%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
            }
            else
            {
                Wxo = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXO%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXI%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXF%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
                Wxc = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXC%dI%d", iLayer, sidx), outputDim, inputDim[sidx]);
            }
            m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
            m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
            m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
            m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

            streamsxi = (streamsxi == nullptr) ? builder.Times(Wxi, input) : builder.Plus(streamsxi, builder.Times(Wxi, input));
            streamsxf = (streamsxf == nullptr) ? builder.Times(Wxf, input) : builder.Plus(streamsxf, builder.Times(Wxf, input));
            streamsxc = (streamsxc == nullptr) ? builder.Times(Wxc, input) : builder.Plus(streamsxc, builder.Times(Wxc, input));
            streamsxo = (streamsxo == nullptr) ? builder.Times(Wxo, input) : builder.Plus(streamsxo, builder.Times(Wxo, input));
        }


        bo = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bo%d", iLayer), outputDim, 1);
        bc = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bc%d", iLayer), outputDim, 1);
        bi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", iLayer), outputDim, 1);
        bf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bf%d", iLayer), outputDim, 1);
        //if (m_forgetGateInitVal > 0)
        bf->FunctionValues().SetValue(m_forgetGateInitVal);
        //if (m_inputGateInitVal > 0)
        bi->FunctionValues().SetValue(m_inputGateInitVal);
        //if (m_outputGateInitVal > 0)
        bo->FunctionValues().SetValue(m_outputGateInitVal);

        Whi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WHI%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whi, m_uniformInit, randomSeed++, m_initValueScale);
        Wci = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WCI%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wci, m_uniformInit, randomSeed++, m_initValueScale);

        Whf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WHF%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whf, m_uniformInit, randomSeed++, m_initValueScale);
        Wcf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WCF%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wcf, m_uniformInit, randomSeed++, m_initValueScale);

        Who = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WHO%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Who, m_uniformInit, randomSeed++, m_initValueScale);
        Wco = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WCO%d", iLayer), outputDim, 1);
        m_net->InitLearnableParameters(Wco, m_uniformInit, randomSeed++, m_initValueScale);

        Whc = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WHC%d", iLayer), outputDim, outputDim);
        m_net->InitLearnableParameters(Whc, m_uniformInit, randomSeed++, m_initValueScale);

        size_t layer1 = outputDim;

        pastValueHI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHO = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueHC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);
        pastValueCC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, mbSize, 1);

        if (m_constInputGateValue)
        {
            //it = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"CONSTIT%d", iLayer), outputDim, mbSize);
            //it->SetParameterUpdateRequired(false);
            //it->FunctionValues().SetValue(m_constInputGateValue);
            it = nullptr;
        }
        else
            it = ApplyNonlinearFunction(
            builder.Plus(
            builder.Plus(
            builder.Plus(
            streamsxi,
            bi),
            builder.Times(Whi, pastValueHI)),
            builder.DiagTimes(Wci, pastValueCI)), 0);

        if (it == nullptr)
        {
            bit = builder.Tanh(
                builder.Plus(
                streamsxc,
                builder.Plus(
                builder.Times(Whc, pastValueHC),
                bc
                )
                )
                );
        }
        else
        {
            bit = builder.ElementTimes(it,
                builder.Tanh(
                builder.Plus(
                streamsxc,
                builder.Plus(
                builder.Times(Whc, pastValueHC),
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
            builder.Plus(
            builder.Plus(
            builder.Plus(
            streamsxf,
            bf),
            builder.Times(Whf, pastValueHF)),
            builder.DiagTimes(Wcf, pastValueCF)), 0);


        if (ft == nullptr)
        {
            bft = pastValueCC;
        }
        else
        {
            bft = builder.ElementTimes(ft, pastValueCC);
        }

        ct = builder.Plus(bft, bit);


        if (m_constOutputGateValue)
        {
            ot = nullptr;
        }
        else
            ot = ApplyNonlinearFunction(
            builder.Plus(
            builder.Plus(
            builder.Plus(
            streamsxo,
            bo),
            builder.Times(Who, pastValueHO)),
            builder.DiagTimes(Wco, ct)), 0);

        if (ot == nullptr)
        {
            output = builder.Tanh(ct);
        }
        else
        {
            output = builder.ElementTimes(ot, builder.Tanh(ct));
        }

        pastValueHO->AttachInputs(output);
        pastValueHI->AttachInputs(output);
        pastValueHF->AttachInputs(output);
        pastValueHC->AttachInputs(output);
        pastValueCI->AttachInputs(ct);
        pastValueCF->AttachInputs(ct);
        pastValueCC->AttachInputs(ct);

        if (m_addDropoutNodes)
            input = builder.Dropout(output);
        else
            input = output;
        output = input;

        return output;
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
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildBiDirectionalLSTMNetworksFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            ULONG randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;

            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior, Wxo;
            ComputationNodePtr forwardInput, forwardOutput, backwardInput, backwardOutput;
            vector<ComputationNodePtr> streams;
            vector<size_t> streamdims;
            ComputationNodePtr inputprediction, inputletter, ngram;
            ComputationNodePtr ltrSource;
            size_t ltrDim = 0;

            map<wstring, size_t> featDim;

            size_t ltrSrcIdx = 1;
            /// create projections to use pastValue predictions
            inputprediction = builder.CreateInputNode(L"featurepastValueedTarget", m_streamSizes[0], mbSize);
            m_net->FeatureNodes().push_back(inputprediction);

            inputletter = builder.CreateInputNode(L"ltrForward", m_streamSizes[1], mbSize);
            m_net->FeatureNodes().push_back(inputletter);
            featDim[L"ltrForward"] = m_streamSizes[1];

            size_t layerIdx = 0;
            size_t idx = 0;
            int recur_idx = 0;
            for (auto p = m_net->FeatureNodes().begin(); p != m_net->FeatureNodes().end(); p++, idx++)
            {
                layerIdx = 0;  /// reset layer id because each input stream starts from layer 0
                input = dynamic_pointer_cast<ComputationNode<ElemType>>(*p);
                if (m_applyMeanVarNorm)
                {
                    input = dynamic_pointer_cast<ComputationNode<ElemType>>(*p);
                    w = builder.Mean(input);
                    b = builder.InvStdDev(input);
                    output = builder.PerDimMeanVarNormalization(input, w, b);

                    input = output;
                }

                size_t idim = input->GetNumRows();
                assert(m_lookupTabelOrderSizes.size() == m_streamSizes.size());

                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"Embedding%d", idx), m_layerSizes[1], idim / m_lookupTabelOrderSizes[idx]);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, msra::strfun::wstrprintf(L"LOOKUP%d", idx));

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
            forwardInput = (ComputationNodePtr)builder.Parallel(streams[0], streams[1], L"Parallel0");

            if (numHiddenLayers > 0)
            {
                /// forward direction
                //forwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 100, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                forwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 100, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                forwardInput = forwardOutput;

                backwardInput = (ComputationNodePtr)builder.TimeReverse(ltrSource);
                //backwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 200, ltrDim, m_layerSizes[layerIdx + 1], backwardInput);
                backwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 200, ltrDim, m_layerSizes[layerIdx + 1], backwardInput);
                backwardInput = backwardOutput;

                layerIdx++;

                while (layerIdx < numHiddenLayers - 1)
                {
                    //forwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 100, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], forwardInput);
                    forwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 100, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], forwardInput);
                    forwardInput = forwardOutput;

                    //backwardOutput = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx + 200, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], backwardInput);
                    backwardOutput = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx + 200, m_layerSizes[layerIdx], m_layerSizes[layerIdx + 1], backwardInput);
                    backwardInput = backwardOutput;

                    layerIdx++;
                }

                backwardOutput = (ComputationNodePtr)builder.TimeReverse(backwardInput);
            }

            streams.clear();
            streamdims.clear();
            streams.push_back(forwardOutput);
            streamdims.push_back(m_layerSizes[layerIdx]);
            streams.push_back(backwardOutput);
            streamdims.push_back(m_layerSizes[layerIdx]);

            /// glue the two streams
            forwardInput = (ComputationNodePtr)builder.Parallel(streams[0], streams[1], L"Parallel1");

//                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, layerIdx, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);
                    output = (ComputationNodePtr)BuildLSTMComponent(randomSeed, mbSize, layerIdx, streamdims[0] + streamdims[1], m_layerSizes[layerIdx + 1], forwardInput);

            input = output;
            layerIdx++;

            /// directly connect transcription model output/feature to the output layer
            Wxo = builder.CreateLearnableParameter(L"ConnectToLowerLayers", m_layerSizes[numHiddenLayers + 1], m_layerSizes[layerIdx]);
            m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);

            output = builder.Times(Wxo, input);
            input = output;

            /// here uses "labels", so only one label from multiple stream inputs are used.
            label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1], mbSize);

            AddTrainAndEvalCriterionNodes(input, label);

            //add softmax layer (if prob is needed or KL reg adaptation is needed)
            output = builder.Softmax(input, L"outputs");

            if (m_needPrior)
            {
                prior = builder.Mean(label);
                input = builder.Log(prior, L"LogOfPrior");
                ComputationNodePtr
                    scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
                m_net->OutputNodes().push_back(scaledLogLikelihood);
            }
            else
                m_net->OutputNodes().push_back(output);

        }

        m_net->ResetEvalTimeStamp();

                return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNCELSTMNetworkFromDescription(size_t mbSize)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);
        if (m_net->GetTotalNumberOfNodes() < 1) //not built yet
        {
            unsigned long randomSeed = 1;

            size_t numHiddenLayers = m_layerSizes.size() - 2;
            size_t numRecurrentLayers = m_recurrentLayers.size();

            ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
            ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
            ComputationNodePtr clslogpostprob;
            ComputationNodePtr bias;
            ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = { nullptr };

            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0], mbSize);
            m_net->FeatureNodes().push_back(input);

            if (m_applyMeanVarNorm)
            {
                w = builder.Mean(input);
                b = builder.InvStdDev(input);
                output = builder.PerDimMeanVarNormalization(input, w, b);

                input = output;
            }

            if (m_lookupTableOrder > 0)
            {
                e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
                m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
                output = builder.LookupTable(e, input, L"LookupTable");

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
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
                        u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                        m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                        b = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"B%d", i), m_layerSizes[i + 1], 1);
                        output = ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), i);
                    }

                    if (m_addDropoutNodes)
                        input = builder.Dropout(output);
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
            w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

            /// the label is a dense matrix. each element is the word index
            label = builder.CreateInputNode(L"labels", 2 * (this->nce_noises + 1), mbSize);

            bias = builder.CreateLearnableParameter(L"BiasVector", 1, m_layerSizes[m_layerSizes.size() - 1]);
            bias->FunctionValues().SetValue((ElemType)-std::log(m_layerSizes[m_layerSizes.size() - 1]));
            //m_net->InitLearnableParameters(bias, m_uniformInit, randomSeed++, std::log(m_layerSizes[m_layerSizes.size() - 1])* m_initValueScale);
            //clslogpostprob = builder.Times(clsweight, input, L"ClassPostProb");

            output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeNCEBasedCrossEntropy", L"EvalNodeNCEBasedCrossEntrpy", bias);

            m_net->OutputNodes().push_back(output);

            if (m_needPrior)
            {
                prior = builder.Mean(label);
            }
        }

        m_net->ResetEvalTimeStamp();

                return m_net;
    }

    template<class ElemType>
    ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNetworkFromDbnFile(const std::wstring& dbnModelFileName)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        std::string hdr, comment, name;
        int version;
        int numLayers, i;
        std::string layerType;

        unsigned long randomSeed = 1;

        ComputationNodePtr input, w, b, output, label, prior, scaledLogLikelihood;
        shared_ptr<PreComputedNode<ElemType>> pcNodePtr;
        size_t mbSize = 3; //this is not the actual minibatch size. only used in the validataion process

        File fstream(dbnModelFileName, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);

        if (!CheckDbnTag(fstream, "DBN\n"))
            RuntimeError("Error reading DBN file - did not find expected tag DBN\n");
        fstream >> comment;
        if (!CheckDbnTag(fstream, "BDBN"))
            RuntimeError("Error reading DBN file - did not find expected tag BDBN\n");
        fstream >> version >> numLayers;

        Matrix<ElemType> globalMean = ReadMatrixFromDbnFile(fstream, std::string("gmean"));
        Matrix<ElemType> globalStdDev = ReadMatrixFromDbnFile(fstream, std::string("gstddev"));
        assert(globalMean.GetNumCols() == 1);
        assert(globalStdDev.GetNumCols() == 1);

        //move to CPU since element-wise operation is expensive and can go wrong in GPU
        int curDevId = globalStdDev.GetDeviceId();
        globalStdDev.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
        for (int i = 0; i<globalStdDev.GetNumRows(); i++)
            globalStdDev(i, 0) = (ElemType)1.0 / (const ElemType)globalStdDev(i, 0);
        globalStdDev.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

        if (!CheckDbnTag(fstream, "BNET"))
            RuntimeError("Error reading DBN file - did not find expected tag BNET\n");

        for (i = 0; i<numLayers; i++) //0th index is for input layer, 
        {
            fstream >> layerType;

            Matrix<ElemType> wts = ReadMatrixFromDbnFile(fstream, std::string("W"));
            Matrix<ElemType> bias = ReadMatrixFromDbnFile(fstream, std::string("a")); // remnant from pretraining, not needed
            Matrix<ElemType> A = ReadMatrixFromDbnFile(fstream, std::string("b"));
            if (i == 0)
            {
                input = builder.Input(wts.GetNumCols(), mbSize, L"features");
                m_net->FeatureNodes().push_back(input);

                size_t frameDim = globalMean.GetNumRows();
                size_t numContextFrames = wts.GetNumCols() / frameDim;
                size_t contextDim = numContextFrames*frameDim;
                Matrix<ElemType> contextMean(contextDim, 1, m_deviceId);
                Matrix<ElemType> contextStdDev(contextDim, 1, m_deviceId);

                //move to CPU since element-wise operation is expensive and can go wrong in GPU
                contextMean.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
                contextStdDev.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
                for (size_t j = 0; j<frameDim; j++)
                {
                    for (size_t k = 0; k<numContextFrames; k++)
                    {
                        contextMean(j + k*frameDim, 0) = (const ElemType)globalMean(j, 0);
                        contextStdDev(j + k*frameDim, 0) = (const ElemType)globalStdDev(j, 0);
                    }
                }
                contextMean.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);
                contextStdDev.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);

                w = builder.Mean(input, L"MeanOfFeatures");
                static_pointer_cast<PreComputedNode<ElemType>>(w)->SideLoadFromMatrix(contextMean);
                w->SetParameterUpdateRequired(false);

                b = builder.InvStdDev(input, L"InvStdOfFeatures");
                static_pointer_cast<PreComputedNode<ElemType>>(b)->SideLoadFromMatrix(contextStdDev);
                b->SetParameterUpdateRequired(false);

                output = builder.PerDimMeanVarNormalization(input, w, b, L"MVNormalizedFeatures");
                input = output;
            }
            if (i == numLayers - 1)
            {
                m_outputLayerSize = wts.GetNumRows();
            }
            wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
            wstring nameOfB = msra::strfun::wstrprintf(L"B%d", i);
            wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", i);
            wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
            wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
            wstring nameOfH = msra::strfun::wstrprintf(L"H%d", i + 1);

            w = builder.Parameter(wts.GetNumRows(), wts.GetNumCols(), nameOfW);
            w->FunctionValues().SetValue(wts);

            b = builder.Parameter(bias.GetNumRows(), 1, nameOfB);
            b->FunctionValues().SetValue(bias);

            if (layerType == "perceptron")
            {
                fprintf(stderr, "DBN: Reading (%lu x %lu) perceptron\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                output = builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus);
            }
            else if (layerType == "rbmisalinearbernoulli")
            {
                fprintf(stderr, "DBN: Reading (%lu x %lu) linear layer\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                output = builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus);
            }
            else // assume rbmbernoullibernoulli
            {
                fprintf(stderr, "DBN: Reading (%lu x %lu) non-linear layer\n", (unsigned long)wts.GetNumRows(), (unsigned long)wts.GetNumCols());
                output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus), i, nameOfH);
                if (m_addDropoutNodes)
                    input = builder.Dropout(output, L"Drop" + nameOfH);
            }

            input = output;
        }

        if (!CheckDbnTag(fstream, "ENET"))
            RuntimeError("Error reading DBN file - did not find expected tag ENET\n");
        //size_t outputLayerSize =  m_layerSizes[m_layerSizes.size()-1];

        label = builder.Input(m_outputLayerSize, mbSize, L"labels");

        if (layerType == "perceptron") // complete network
        {
            m_net->RenameNode(output, L"HLast");
#if 0
            assert(numLayers + 1 == m_layerSizes.size());
#endif
            Matrix<ElemType> priorVals = ReadMatrixFromDbnFile(fstream, std::string("Pu"));
            assert(priorVals.GetNumCols() == 1 && priorVals.GetNumRows() == m_outputLayerSize);

            w = builder.Mean(label, L"Prior");
            static_pointer_cast<PreComputedNode<ElemType>>(w)->SideLoadFromMatrix(priorVals);
            w->SetParameterUpdateRequired(false);
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

            size_t penultimateSize = input->GetNumRows();

            wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
            wstring nameOfB = msra::strfun::wstrprintf(L"B%d", i);
            wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", i);
            wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
            wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
            wstring nameOfH = msra::strfun::wstrprintf(L"H%d", i + 1);

            w = builder.Parameter(outputLayerSize, penultimateSize, nameOfW);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            b = builder.Parameter(outputLayerSize, 1, nameOfB);
            output = builder.Plus(builder.Times(w, input, nameOfTimes), b, nameOfPlus);
            m_net->RenameNode(output, L"HLast");

            if (m_needPrior)
            {
                Matrix<ElemType> zeros = Matrix<ElemType>::Zeros(outputLayerSize, 1, m_deviceId);
                prior = builder.Mean(label, L"Prior");
                static_pointer_cast<PreComputedNode<ElemType>>(prior)->MarkComputed(false);
                prior->FunctionValues().SetValue(zeros);
            }
        }

        AddTrainAndEvalCriterionNodes(output, label);

        if (layerType == "perceptron" || m_needPrior)
        {
            input = builder.Log(pcNodePtr, L"LogOfPrior");

            //following two lines is needed only if true probability is needed
            //output = builder.Softmax(output);
            //output = builder.Log(output);

            scaledLogLikelihood = builder.CreateComputationNode(OperationNameOf(MinusNode), L"ScaledLogLikelihood");
            scaledLogLikelihood->AttachInputs(output, input);
            m_net->OutputNodes().push_back(scaledLogLikelihood);
        }
        else
        {
            m_net->OutputNodes().push_back(output);
        }

        if (!CheckDbnTag(fstream, "EDBN"))
            RuntimeError("Error reading DBN file - did not find expected tag ENET\n");
        return m_net;
    }

    //layer is 0 based
    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(ComputationNodePtr input, const size_t layer, const std::wstring nodeName)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        ComputationNodePtr output;
        wstring nonLinearFunction = m_nonLinearFunctions[layer];
        if (nonLinearFunction == OperationNameOf(SigmoidNode))
            output = builder.Sigmoid(input, nodeName);
        else if (nonLinearFunction == OperationNameOf(RectifiedLinearNode))
            output = builder.RectifiedLinear(input, nodeName);
        else if (nonLinearFunction == OperationNameOf(TanhNode))
            output = builder.Tanh(input, nodeName);
        else if (nonLinearFunction == L"None" || nonLinearFunction == L"none" || nonLinearFunction == L"")
        {
            output = input;  //linear layer
            if (nodeName != L"")
                m_net->RenameNode(output, nodeName);
        }
        else
            LogicError("Unsupported nonlinear function.");

        return output;
    }

    template<class ElemType>
    shared_ptr<ComputationNode<ElemType>> SimpleNetworkBuilder<ElemType>::AddTrainAndEvalCriterionNodes(ComputationNodePtr input, ComputationNodePtr label, ComputationNodePtr matrix, const std::wstring trainNodeName, const std::wstring evalNodeName, ComputationNodePtr clspostprob, ComputationNodePtr trans)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

        m_net->LabelNodes().push_back(label);

        ComputationNodePtr output;
        ComputationNodePtr tinput = input;
        if (matrix != nullptr)
            tinput = builder.Times(matrix, input);

        switch (m_trainCriterion)
        {
        case TrainingCriterion::CrossEntropyWithSoftmax:
            output = builder.CrossEntropyWithSoftmax(label, tinput, (trainNodeName == L"") ? L"CrossEntropyWithSoftmax" : trainNodeName);
            break;
        case TrainingCriterion::SquareError:
            output = builder.SquareError(label, tinput, (trainNodeName == L"") ? L"SquareError" : trainNodeName);
            break;
        case TrainingCriterion::Logistic:
            output = builder.Logistic(label, tinput, (trainNodeName == L"") ? L"Logistic" : trainNodeName);
            break;
        case TrainingCriterion::CRF:
            assert(trans != nullptr);
            output = builder.CRF(label, input, trans, (trainNodeName == L"") ? L"CRF" : trainNodeName);
            break;
        case TrainingCriterion::ClassCrossEntropyWithSoftmax:
            output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"ClassCrossEntropyWithSoftmax" : trainNodeName);
            break;
        case TrainingCriterion::NCECrossEntropyWithSoftmax:
            output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
            //output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
            break;
        default:
            LogicError("Unsupported training criterion.");
        }
        m_net->FinalCriterionNodes().push_back(output);

        if (!((m_evalCriterion == EvalCriterion::CrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::CrossEntropyWithSoftmax) ||
            (m_evalCriterion == EvalCriterion::SquareError && m_trainCriterion == TrainingCriterion::SquareError) ||
            (m_evalCriterion == EvalCriterion::Logistic && m_trainCriterion == TrainingCriterion::Logistic) ||
            (m_evalCriterion == EvalCriterion::CRF && m_trainCriterion == TrainingCriterion::CRF) ||
            (m_evalCriterion == EvalCriterion::ClassCrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::ClassCrossEntropyWithSoftmax) ||
            (m_evalCriterion == EvalCriterion::NCECrossEntropyWithSoftmax && m_trainCriterion == TrainingCriterion::NCECrossEntropyWithSoftmax)))
        {
            switch (m_evalCriterion)
            {
            case EvalCriterion::CrossEntropyWithSoftmax:
                //output = builder.CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"")?L"EvalCrossEntropyWithSoftmax":evalNodeName);
                output = builder.CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"") ? L"CrossEntropyWithSoftmax" : evalNodeName);
                break;
            case EvalCriterion::ClassCrossEntropyWithSoftmax:
                //output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"EvalClassCrossEntropyWithSoftmax" : evalNodeName);
                output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"ClassCrossEntropyWithSoftmax" : evalNodeName);
                break;
            case EvalCriterion::NCECrossEntropyWithSoftmax:
                output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"NoiseContrastiveEstimationNode" : evalNodeName);
                break;
            case EvalCriterion::SquareError:
                //output = builder.SquareError(label, tinput, (evalNodeName == L"")?L"EvalSquareError":evalNodeName);
                output = builder.SquareError(label, tinput, (evalNodeName == L"") ? L"SquareError" : evalNodeName);
                break;
            case EvalCriterion::Logistic:
                //output = builder.SquareError(label, tinput, (evalNodeName == L"")?L"EvalSquareError":evalNodeName);
                output = builder.Logistic(label, tinput, (evalNodeName == L"") ? L"Logistic" : evalNodeName);
                break;
            case EvalCriterion::ErrorPrediction:
                output = builder.ErrorPrediction(label, tinput, (evalNodeName == L"") ? L"EvalErrorPrediction" : evalNodeName);
                break;
            case EvalCriterion::CRF:
                assert(trans != nullptr);
                output = builder.CRF(label, tinput, trans, (evalNodeName == L"") ? L"EvalCRF" : evalNodeName);
                break;
            default:
                LogicError("Unsupported training criterion.");
            }
            output->SetParameterUpdateRequired(false);
        }

        m_net->EvaluationNodes().push_back(output);

        return output;
    }

    template class SimpleNetworkBuilder<float>;
    template class SimpleNetworkBuilder<double>;

}}}
