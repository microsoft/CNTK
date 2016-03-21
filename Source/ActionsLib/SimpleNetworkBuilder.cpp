//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SimpleNetworkBuilder.h"
#include "ComputationNetworkBuilder.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "PreComputeNodes.h"

#pragma warning(disable : 4189) // (we have lots of unused variables to show how variables can be set up)

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNetworkFromDescription()
{
    ComputationNetworkPtr net;
    switch (m_standardNetworkKind)
    {
    case FFDNNKind:
        net = BuildFFDNNFromDescription();
        break;
    case RNNKind:
        net = BuildRNNFromDescription();
        break;
    case LSTMKind:
        net = BuildLSTMNetworkFromDescription();
        break;
    case ClassLSTMNetworkKind:
        net = BuildClassLSTMNetworkFromDescription();
        break;
    case NCELSTMNetworkKind:
        net = BuildNCELSTMNetworkFromDescription();
        break;
    case ClassEntropyRNNKind:
        net = BuildClassEntropyRNNFromDescription();
        break;
    case LogBilinearNetworkKind:
        net = BuildLogBilinearNetworkFromDescription();
        break;
    case DNNLMNetworkKind:
        net = BuildDNNLMNetworkFromDescription();
        break;
    case ConditionalLSTMNetworkKind:
        net = BuildConditionalLSTMNetworkFromDescription();
        break;
#ifdef COMING_SOON
    case CRFLSTMNetworkKind:
        net = BuildCRFLSTMNetworkFromDescription();
        break;
#endif
    default:
        LogicError("BuildNetworkFromDescription: invalid m_standardNetworkKind %d", (int) m_standardNetworkKind);
    }

    // post-process the network
    net->CompileNetwork();

    return net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildFFDNNFromDescription()
{

    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;
        ComputationNodePtr input, w, b, output, label, prior, scaledLogLikelihood;

        input = builder.CreateInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

        if (m_applyMeanVarNorm)
        {
            w = builder.Mean(input, L"MeanOfFeatures");
            b = builder.InvStdDev(input, L"InvStdOfFeatures");
            output = builder.PerDimMeanVarNormalization(input, w, b, L"MVNormalizedFeatures");

            input = output;
        }

        if (numHiddenLayers > 0)
        {
            w = builder.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[0]);
            m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
            b = builder.CreateLearnableParameter(L"B0", m_layerSizes[1], 1);
            output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, 1, L"W0*features"), b, L"W0*features+B0"), 0, L"H1");

            if (m_addDropoutNodes)
                input = builder.Dropout(output, L"DropH1");
            else
                input = output;

            for (int i = 1; i < numHiddenLayers; i++)
            {
                wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
                wstring nameOfB = msra::strfun::wstrprintf(L"B%d", i);
                wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", i);
                wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
                wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
                wstring nameOfH = msra::strfun::wstrprintf(L"H%d", i + 1);

                w = builder.CreateLearnableParameter(nameOfW, m_layerSizes[i + 1], m_layerSizes[i]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                b = builder.CreateLearnableParameter(nameOfB, m_layerSizes[i + 1], 1);
                output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus), i, nameOfH);

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

        w = builder.CreateLearnableParameter(nameOfW, m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
        b = builder.CreateLearnableParameter(nameOfB, m_layerSizes[numHiddenLayers + 1], 1);
        output = builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus);
        m_net->RenameNode(output, L"HLast");

        label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);

        AddTrainAndEvalCriterionNodes(output, label);

        if (m_needPrior)
        {
            prior = builder.Mean(label, L"Prior");
            input = builder.Log(prior, L"LogOfPrior");

            // following two lines are needed only if true probability is needed
            // output = builder.Softmax(output);
            // output = builder.Log(output);

            scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
            m_net->AddToNodeGroup(L"output", scaledLogLikelihood);
        }
        else
        {
            m_net->AddToNodeGroup(L"output", output);
        }

        // add softmax layer (if prob is needed or KL reg adaptation is needed)
        output = builder.Softmax(output, L"PosteriorProb");
        // m_net->AddToNodeGroup(L"output", output);
    }

    return m_net;
}

// Note: while ComputationNode and CompuationNetwork are (supposed to be) independent of ElemType, it is OK to keep this class dependent.
template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildRNNFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, pastValue, output, label, prior;

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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
            // TODO: to figure out sparse matrix size
            u = builder.CreateLearnableParameter(L"U0", m_layerSizes[1], m_layerSizes[0]);
            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
            {
                w = builder.CreateLearnableParameter(L"W0", m_layerSizes[1], m_layerSizes[1]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[1], 1);
                // unless there is a good algorithm to detect loops, use this explicit setup
                output = ApplyNonlinearFunction(
                    builder.Plus(
                        builder.Times(u, input), builder.Times(w, pastValue)),
                    0);
                pastValue->AttachInputs({ output });
                recur_idx++;
            }
            else
            {
                output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), b), 0);
                // TODO: Why the ^^ namespace?
                // output = builder.Times(u, input);
            }

            if (m_addDropoutNodes)
                input = builder.Dropout(output);
            else
                input = output;

            for (int i = 1; i < numHiddenLayers; i++)
            {
                // TODO: to figure out sparse matrix size
                u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);

                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                {
                    w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t) m_layerSizes[i + 1], 1);
                    // unless there is a good algorithm to detect loops, use this explicit setup
                    output = ApplyNonlinearFunction(
                        builder.Plus(
                            builder.Times(u, input), builder.Times(w, pastValue)),
                        0);
                    pastValue->AttachInputs({ output });
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

        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
        /*m_net->MatrixL2Reg(w , L"L1w");*/

        label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);
        AddTrainAndEvalCriterionNodes(input, label, w, L"criterion", L"eval");

        output = builder.Times(w, input, 1, L"outputs");

        m_net->AddToNodeGroup(L"output", output);

        if (m_needPrior)
            prior = builder.Mean(label);
    }

    return m_net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildClassEntropyRNNFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);

    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, pastValue, output, label, prior;
        ComputationNodePtr wrd2cls, cls2idx, clslogpostprob, clsweight;

        if (m_vocabSize != m_layerSizes[numHiddenLayers + 1])
            RuntimeError("BuildClassEntropyRNNFromDescription : vocabulary size should be the same as the output layer size");

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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

                pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[1], 1);
                // unless there is a good algorithm to detect loops, use this explicit setup
                output = ApplyNonlinearFunction(
                    builder.Plus(
                        builder.Times(u, input), builder.Times(w, pastValue)),
                    0);
                pastValue->AttachInputs({ output });
                recur_idx++;
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

            for (int i = 1; i < numHiddenLayers; i++)
            {
                u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                {
                    w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

                    pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, (size_t) m_layerSizes[i + 1], 1);
                    // unless there is a good algorithm to detect loops, use this explicit setup
                    output = ApplyNonlinearFunction(
                        builder.Plus(
                            builder.Times(u, input), builder.Times(w, pastValue)),
                        0);
                    pastValue->AttachInputs({ output });
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

        // need to have [input_dim x output_dim] matrix
        // e.g., [200 x 10000], where 10000 is the vocabulary size
        // this is for speed-up issue as per word matrix can be simply obtained using column slice
        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

        // the label is a dense matrix. each element is the word index
        label = builder.CreateInputNode(L"labels", 4);

        clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
        clslogpostprob = builder.Times(clsweight, input, 1, L"ClassPostProb");

        output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                                               clslogpostprob);

        m_net->AddToNodeGroup(L"output", output);

        if (m_needPrior)
        {
            prior = builder.Mean(label);
        }
    }

    return m_net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildConditionalLSTMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
        ComputationNodePtr gt;
        ComputationNodePtr clslogpostprob;
        ComputationNodePtr clsweight;

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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
            LogicError("BuildClassLSTMNetworkFromDescription: LSTMNode cannot take sparse input. Need to project sparse input to continuous vector using LookupTable. Suggest using setups below\n layerSizes=$VOCABSIZE$:100:$HIDDIM$:$VOCABSIZE$ \nto have 100 dimension projection, and lookupTableOrder=1\n to project to a single window. To use larger context window, set lookupTableOrder=3 for example with width-3 context window.\n ");
        }

        int recur_idx = 0;
        int offset = m_lookupTableOrder > 0 ? 1 : 0;
        if (numHiddenLayers > 0)
        {
            //           output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            // previously used function. now uses LSTMNode which is correct and fast
            input = output;
            for (int i = 1 + offset; i < numHiddenLayers; i++)
            {
                //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;
            }
        }

        // serve as a global bias term
        gt = builder.CreateInputNode(L"binaryFeature", m_auxFeatDim);
        m_net->AddToNodeGroup(L"feature", gt);
        e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"AuxTrans%d", 0),
                                             m_layerSizes[numHiddenLayers], m_auxFeatDim);
        m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
        u = ApplyNonlinearFunction(builder.Times(e, gt), numHiddenLayers, L"TimesToGetGlobalBias");
        output = builder.Plus(input, u, L"PlusGlobalBias");
        input = output;

        // need to have [input_dim x output_dim] matrix
        // e.g., [200 x 10000], where 10000 is the vocabulary size
        // this is for speed-up issue as per word matrix can be simply obtained using column slice
        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

        // the label is a dense matrix. each element is the word index
        label = builder.CreateInputNode(L"labels", 4);

        clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
        clslogpostprob = builder.Times(clsweight, input, 1, L"ClassPostProb");

        output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                                               clslogpostprob);

        output = builder.TransposeTimes(w, input, L"outputs");

        m_net->AddToNodeGroup(L"output", output);

        // add softmax layer (if prob is needed or KL reg adaptation is needed)
        output = builder.Softmax(output, L"PosteriorProb");
    }

    return m_net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildLogBilinearNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, pastValue, output, label, prior, featin, e;
        ComputationNodePtr bi = nullptr;
        ComputationNodePtr Wxi1 = nullptr, Wxi = nullptr;
        ComputationNodePtr Wxi2 = nullptr, Wxi3 = nullptr, Wxi4 = nullptr;
        ComputationNodePtr ot = nullptr, it = nullptr, ft = nullptr, gt = nullptr, ct = nullptr, ht = nullptr;
        ComputationNodePtr pastValueXI, pastValueXII, pastValueXIII, pastValueXIV;

        //                input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        input = builder.CreateInputNode(L"features", m_layerSizes[0]);
        featin = input;
        m_net->AddToNodeGroup(L"feature", input);

        if (m_applyMeanVarNorm)
        {
            w = builder.Mean(input);
            b = builder.InvStdDev(input);
            output = builder.PerDimMeanVarNormalization(input, w, b);

            input = output;
        }

        // used for lookuptable node unittest, will delete
        if (m_lookupTableOrder > 0)
        {
            e = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"E%d", 0), m_layerSizes[1], m_layerSizes[0] / m_lookupTableOrder);
            m_net->InitLearnableParameters(e, m_uniformInit, randomSeed++, m_initValueScale);
            output = builder.LookupTable(e, input, L"Lookuptatble");

            if (m_addDropoutNodes)
                input = builder.Dropout(output);
            else
                input = output;
        }

        int recur_idx = 0;
        // unless there is a good algorithm to detect loops, use this explicit setup
        int ik = 1;
        output = input;
        while (ik <= m_maOrder)
        {
            pastValueXI =
                builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], ik, msra::strfun::wstrprintf(L"pastValue%d", ik));
            pastValueXI->SetLearningRateMultiplier(0);
            pastValueXI->AttachInputs({ input });
            // TODO: to figure out sparse matrix size
            Wxi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"DD%d", ik), m_layerSizes[0], m_layerSizes[0]);
            m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

            it = builder.Plus(output, builder.Times(Wxi, pastValueXI));
            output = it;

            ik++;
        }

        if (m_addDropoutNodes)
            input = builder.Dropout(output);
        else
            input = output;

        for (int i = m_lookupTableOrder > 0 ? 1 : 0; i < numHiddenLayers; i++)
        {
            u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i] * (m_lookupTableOrder > 0 ? m_lookupTableOrder : 1));
            m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
            output = builder.Times(u, input);
            input = output;
            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
            {
                w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"R%d", i + 1), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[i + 1], 1);
                output = builder.Plus(builder.Times(w, pastValue), input);

                pastValue->AttachInputs({ output });
                input = output;
                recur_idx++;
            }

            bi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", i), m_layerSizes[i + 1], 1);
            output = builder.Plus(input, bi);

            if (m_addDropoutNodes)
                input = builder.Dropout(output);
            else
                input = output;
        }

        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

        label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);
        AddTrainAndEvalCriterionNodes(input, label, w);

        output = builder.Times(w, input, 1, L"outputs");

        m_net->AddToNodeGroup(L"output", output);

        if (m_needPrior)
        {
            prior = builder.Mean(label);
        }
    }

    return m_net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildDNNLMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
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

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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

            pastValueXI = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], 1);
            pastValueXII = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], 2);
            pastValueXIII = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], 3);
            pastValueXIV = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[0], 4);
            pastValueXI->AttachInputs({ input });
            pastValueXII->AttachInputs({ input });
            pastValueXIII->AttachInputs({ input });
            pastValueXIV->AttachInputs({ input });

            if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == 1)
            {
                // TODO: to figure out sparse matrix size
                Wxi2 = builder.CreateLearnableParameter(L"WXI2", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(Wxi2, m_uniformInit, randomSeed++, m_initValueScale);
                // TODO: to figure out sparse matrix size
                Wxi3 = builder.CreateLearnableParameter(L"WXI3", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(Wxi3, m_uniformInit, randomSeed++, m_initValueScale);
                // TODO: to figure out sparse matrix size
                Wxi4 = builder.CreateLearnableParameter(L"WXI4", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(Wxi4, m_uniformInit, randomSeed++, m_initValueScale);
                // TODO: to figure out sparse matrix size
                Wxi1 = builder.CreateLearnableParameter(L"WXI1", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(Wxi1, m_uniformInit, randomSeed++, m_initValueScale);
                // TODO: to figure out sparse matrix size
                Wxi = builder.CreateLearnableParameter(L"WXI", m_layerSizes[1], m_layerSizes[0]);
                m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);

                // unless there is a good algorithm to detect loops, use this explicit setup
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
                                        builder.Times(Wxi, input)))))),
                    bi);
                output = it;
                pastValueXI->SetLearningRateMultiplier(0);
                pastValueXII->SetLearningRateMultiplier(0);
                pastValueXIII->SetLearningRateMultiplier(0);
                pastValueXIV->SetLearningRateMultiplier(0);
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

            for (int i = 1; i < numHiddenLayers; i++)
            {
                u = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"U%d", i), m_layerSizes[i + 1], m_layerSizes[i]);
                m_net->InitLearnableParameters(u, m_uniformInit, randomSeed++, m_initValueScale);
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                {
                    w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", i), m_layerSizes[i + 1], m_layerSizes[i + 1]);
                    m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
                    std::list<ComputationNodeBasePtr> recurrent_loop;
                    pastValue = builder.PastValue(NULL, m_defaultHiddenActivity, m_layerSizes[i + 1], 1);
                    output = SimpleNetworkBuilder<ElemType>::ApplyNonlinearFunction(builder.Plus(builder.Times(u, input), builder.Times(w, pastValue)), i);
                    pastValue->AttachInputs({ output });
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

        // TODO: to figure out sparse matrix size
        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers + 1], m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
        //                b = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"B%d", numHiddenLayers), m_layerSizes[numHiddenLayers+1], 1);
        label = builder.CreateSparseInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);
        AddTrainAndEvalCriterionNodes(input, label, w);

        output = builder.Times(w, input);

        m_net->AddToNodeGroup(L"output", output);

        if (m_needPrior)
        {
            prior = builder.Mean(label);
        }
    }

    return m_net;
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildDirectConnect(unsigned long& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr input, ComputationNodePtr toNode)
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);

    ComputationNodePtr directOutput, mergedNode;

    for (size_t i = 0; i < m_directConnect.size(); i++)
    {
        if (m_directConnect[i] == iLayer)
        {
            ComputationNodePtr directWIO = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"D%d", i), outputDim, inputDim);
            m_net->InitLearnableParameters(directWIO, m_uniformInit, randomSeed++, m_initValueScale);
            directOutput = ApplyNonlinearFunction(builder.Times(directWIO, input), i);

            ComputationNodePtr scalar = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"SV%d", i), 1, 1);
            scalar->Value().SetValue((ElemType) 0.01);
            ComputationNodePtr scaled = builder.ElementTimes(scalar, directOutput, msra::strfun::wstrprintf(L"S%d", i));

            mergedNode = builder.Plus(toNode, scaled);
        }
    }

    return mergedNode;
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMComponent(unsigned long& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
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

    input = inputObs;
    Wxo = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXO%d", iLayer), outputDim, inputDim);
    Wxi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXI%d", iLayer), outputDim, inputDim);
    Wxf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXF%d", iLayer), outputDim, inputDim);
    Wxc = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WXC%d", iLayer), outputDim, inputDim);

    m_net->InitLearnableParameters(Wxo, m_uniformInit, randomSeed++, m_initValueScale);
    m_net->InitLearnableParameters(Wxi, m_uniformInit, randomSeed++, m_initValueScale);
    m_net->InitLearnableParameters(Wxf, m_uniformInit, randomSeed++, m_initValueScale);
    m_net->InitLearnableParameters(Wxc, m_uniformInit, randomSeed++, m_initValueScale);

    bo = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bo%d", iLayer), outputDim, 1);
    bc = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bc%d", iLayer), outputDim, 1);
    bi = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bi%d", iLayer), outputDim, 1);
    bf = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"bf%d", iLayer), outputDim, 1);
    // if (m_forgetGateInitVal > 0)
    bf->Value().SetValue(m_forgetGateInitVal);
    // if (m_inputGateInitVal > 0)
    bi->Value().SetValue(m_inputGateInitVal);
    // if (m_outputGateInitVal > 0)
    bo->Value().SetValue(m_outputGateInitVal);

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

    pastValueHI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueHF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueHO = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueHC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueCI = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueCF = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);
    pastValueCC = builder.PastValue(NULL, m_defaultHiddenActivity, layer1, 1);

    if (m_constInputGateValue)
    {
        // it = builder.CreateLearnableParameter(msra::strfun::wstrprintf (L"CONSTIT%d", iLayer), outputDim);
        // it->SetLearningRateMultiplier(0);
        // it->Value().SetValue(m_constInputGateValue);
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
                builder.DiagTimes(Wci, pastValueCI)),
            0);

    if (it == nullptr)
    {
        bit = builder.Tanh(
            builder.Plus(
                builder.Times(Wxc, input),
                builder.Plus(
                    builder.Times(Whc, pastValueHC),
                    bc)));
    }
    else
    {
        bit = builder.ElementTimes(it,
                                   builder.Tanh(
                                       builder.Plus(
                                           builder.Times(Wxc, input),
                                           builder.Plus(
                                               builder.Times(Whc, pastValueHC),
                                               bc))));
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
                        builder.Times(Wxf, input),
                        bf),
                    builder.Times(Whf, pastValueHF)),
                builder.DiagTimes(Wcf, pastValueCF)),
            0);

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
                        builder.Times(Wxo, input),
                        bo),
                    builder.Times(Who, pastValueHO)),
                builder.DiagTimes(Wco, ct)),
            0);

    if (ot == nullptr)
    {
        output = builder.Tanh(ct);
    }
    else
    {
        output = builder.ElementTimes(ot, builder.Tanh(ct));
    }

    pastValueHO->AttachInputs({ output });
    pastValueHI->AttachInputs({ output });
    pastValueHF->AttachInputs({ output });
    pastValueHC->AttachInputs({ output });
    pastValueCI->AttachInputs({ ct });
    pastValueCF->AttachInputs({ ct });
    pastValueCC->AttachInputs({ ct });

    if (m_addDropoutNodes)
        input = builder.Dropout(output);
    else
        input = output;
    output = input;

    return output;
}

#ifdef COMING_SOON

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildCRFLSTMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
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
        ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = {nullptr};
        ComputationNodePtr trans;

        input = builder.CreateInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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

        // direct connect from input node to output node

        int recur_idx = 0;
        int offset = m_lookupTableOrder > 0 ? 1 : 0;
        if (numHiddenLayers > 0)
        {
            for (int i = offset; i < numHiddenLayers; i++)
            {
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i + 1)
                {
                    output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, i, m_layerSizes[i] * (offset ? m_lookupTableOrder : 1), m_layerSizes[i + 1], input);
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
        trans->Value().SetValue((ElemType) 1.0 / m_layerSizes[numHiddenLayers + 1]);
        //          m_net->InitLearnableParameters(trans, m_uniformInit, randomSeed++, m_initValueScale);
        trans->SetLearningRateMultiplier(1.0f);
        label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);
        AddTrainAndEvalCriterionNodes(output, label, nullptr, L"CRFTrainCriterion", L"CRFEvalCriterion", nullptr, trans);

        input = output;
        output = builder.SequenceDecoder(label, input, trans, L"outputs");
        m_net->AddToNodeGroup(L"output", output);

        output = builder.Softmax(input, L"PosteriorProb");
    }

    return m_net;
}

#endif

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildClassLSTMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;

        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
        ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
        ComputationNodePtr clslogpostprob;
        ComputationNodePtr clsweight;

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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
            LogicError("BuildClassLSTMNetworkFromDescription: LSTMNode cannot take sparse input. Need to project sparse input to continuous vector using LookupTable. Suggest using setups below\n layerSizes=$VOCABSIZE$:100:$HIDDIM$:$VOCABSIZE$ \nto have 100 dimension projection, and lookupTableOrder=1\n to project to a single window. To use larger context window, set lookupTableOrder=3 for example with width-3 context window.\n ");
        }

        int recur_idx = 0;
        int offset = m_lookupTableOrder > 0 ? 1 : 0;
        if (numHiddenLayers > 0)
        {
            //                output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            // previously used function. now uses LSTMNode which is correct and fast
            input = output;
            for (int i = 1 + offset; i < numHiddenLayers; i++)
            {
                //                    output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);

                if (m_addDropoutNodes)
                    input = builder.Dropout(output);
                else
                    input = output;
            }
        }

        // need to have [input_dim x output_dim] matrix
        // e.g., [200 x 10000], where 10000 is the vocabulary size
        // this is for speed-up issue as per word matrix can be simply obtained using column slice
        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

        // the label is a dense matrix. each element is the word index
        label = builder.CreateInputNode(L"labels", 4);

        clsweight = builder.CreateLearnableParameter(L"WeightForClassPostProb", m_nbrCls, m_layerSizes[numHiddenLayers]);
        m_net->InitLearnableParameters(clsweight, m_uniformInit, randomSeed++, m_initValueScale);
        clslogpostprob = builder.Times(clsweight, input, 1, L"ClassPostProb");

        output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeClassBasedCrossEntropy", L"EvalNodeClassBasedCrossEntrpy",
                                               clslogpostprob);

        output = builder.TransposeTimes(w, input, L"outputs");

        m_net->AddToNodeGroup(L"output", output);

        // add softmax layer (if prob is needed or KL reg adaptation is needed)
        output = builder.Softmax(output, L"PosteriorProb");
    }

    return m_net;
}

#if 1
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMNodeComponent(ULONG&, size_t, size_t, size_t, ComputationNodePtr)
{
    InvalidArgument("BuildLSTMNodeComponent: LSTMNode is no longer available. You should not get here.");
}
#else
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> /*ComputationNodePtr*/ SimpleNetworkBuilder<ElemType>::BuildLSTMNodeComponent(ULONG& randomSeed, size_t iLayer, size_t inputDim, size_t outputDim, ComputationNodePtr inputObs)
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    size_t numHiddenLayers = m_layerSizes.size() - 2;

    ComputationNodePtr input, output;
    ComputationNodePtr wInputGate, wForgetGate, wOutputGate, wMemoryCellMatrix;

    input = inputObs;
    size_t nDim = inputDim + outputDim + 2;
    wInputGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WINPUTGATE%d", iLayer), outputDim, nDim);
    m_net->InitLearnableParameters(wInputGate, m_uniformInit, randomSeed++, m_initValueScale);
    wInputGate->Value().ColumnSlice(0, 1).SetValue(m_inputGateInitVal); // init to input gate bias
    wForgetGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WFORGETGATE%d", iLayer), outputDim, nDim);
    m_net->InitLearnableParameters(wForgetGate, m_uniformInit, randomSeed++, m_initValueScale);
    wForgetGate->Value().ColumnSlice(0, 1).SetValue(m_forgetGateInitVal); // init to forget gate bias
    wOutputGate = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WOUTPUTGATE%d", iLayer), outputDim, nDim);
    m_net->InitLearnableParameters(wOutputGate, m_uniformInit, randomSeed++, m_initValueScale);
    wOutputGate->Value().ColumnSlice(0, 1).SetValue(m_outputGateInitVal); // init to output gate bias
    wMemoryCellMatrix = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"WMEMORYCELLWEIGHT%d", iLayer), outputDim, inputDim + outputDim + 1);
    m_net->InitLearnableParameters(wMemoryCellMatrix, m_uniformInit, randomSeed++, m_initValueScale);
    wMemoryCellMatrix->Value().ColumnSlice(0, 1).SetValue(0); // init to memory cell bias

    output = builder.LSTM(inputObs, wInputGate, wForgetGate, wOutputGate, wMemoryCellMatrix, msra::strfun::wstrprintf(L"LSTM%d", iLayer));

#ifdef DEBUG_DECODER
    wInputGate->Value().SetValue((ElemType) 0.01);
    wForgetGate->Value().SetValue((ElemType) 0.01);
    wOutputGate->Value().SetValue((ElemType) 0.01);
    wMemoryCellMatrix->Value().SetValue((ElemType) 0.01);
#endif

    if (m_addDropoutNodes)
        input = builder.Dropout(output);
    else
        input = output;
    output = input;

    return output;
}
#endif

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildLSTMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
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
        ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = {nullptr};

        if (m_sparse_input)
            input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        else
            input = builder.CreateInputNode(L"features", m_layerSizes[0]);

        m_net->AddToNodeGroup(L"feature", input);

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
            e->Value().SetValue((ElemType) 0.01);
#endif

            if (m_addDropoutNodes)
                input = builder.Dropout(output);
            else
                input = output;

            outputFromEachLayer[1] = input;
        }

        // direct connect from input node to output node

        int recur_idx = 0;
        int offset = m_lookupTableOrder > 0 ? 1 : 0;
        if (numHiddenLayers > 0)
        {

            // output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            // previously used function. now uses LSTMNode which is correct and fast
            input = output;
            outputFromEachLayer[offset + 1] = input;

            for (int i = 1 + offset; i < numHiddenLayers; i++)
            {
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
                {

                    // output = (ComputationNodePtr)BuildLSTMNodeComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
                    output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);
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
        w->Value().SetValue((ElemType) 0.01);
#endif
        label = builder.CreateInputNode(L"labels", m_layerSizes[numHiddenLayers + 1]);
        AddTrainAndEvalCriterionNodes(input, label, w);

        output = builder.Times(w, input, 1, L"outputs");

        if (m_needPrior)
        {
            prior = builder.Mean(label);
            input = builder.Log(prior, L"LogOfPrior");
            ComputationNodePtr
                scaledLogLikelihood = builder.Minus(output, input, L"ScaledLogLikelihood");
            m_net->AddToNodeGroup(L"output", scaledLogLikelihood);
        }
        else
            m_net->AddToNodeGroup(L"output", output);

        // add softmax layer (if prob is needed or KL reg adaptation is needed)
        output = builder.Softmax(output, L"PosteriorProb");
    }

    return m_net;
}

template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNCELSTMNetworkFromDescription()
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);
    if (m_net->GetTotalNumberOfNodes() < 1) // not built yet
    {
        unsigned long randomSeed = 1;

        size_t numHiddenLayers = m_layerSizes.size() - 2;
        size_t numRecurrentLayers = m_recurrentLayers.size();

        ComputationNodePtr input, w, b, u, e, pastValue, output, label, prior;
        ComputationNodePtr Wxo, Who, Wco, bo, Wxi, Whi, Wci, bi;
        ComputationNodePtr clslogpostprob;
        ComputationNodePtr bias;
        ComputationNodePtr outputFromEachLayer[MAX_DEPTH] = {nullptr};

        input = builder.CreateSparseInputNode(L"features", m_layerSizes[0]);
        m_net->AddToNodeGroup(L"feature", input);

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

        // direct connect from input node to output node

        int recur_idx = 0;
        int offset = m_lookupTableOrder > 0 ? 1 : 0;
        if (numHiddenLayers > 0)
        {
            output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, 0, m_layerSizes[offset] * (offset ? m_lookupTableOrder : 1), m_layerSizes[offset + 1], input);
            input = output;
            outputFromEachLayer[offset + 1] = input;

            for (int i = 1 + offset; i < numHiddenLayers; i++)
            {
                if (m_recurrentLayers.size() > 0 && m_recurrentLayers[recur_idx] == i)
                {
                    output = (ComputationNodePtr) BuildLSTMComponent(randomSeed, i, m_layerSizes[i], m_layerSizes[i + 1], input);

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
            // add direct connect from each layers' output to the layer before the output layer
            output = BuildDirectConnect(randomSeed, i, (i > 1) ? m_layerSizes[i] : ((offset == 0) ? m_layerSizes[i] : m_layerSizes[i] * m_lookupTableOrder), m_layerSizes[numHiddenLayers], outputFromEachLayer[i], input);
            if (output != nullptr)
                input = output;
        }

        // need to have [input_dim x output_dim] matrix
        // e.g., [200 x 10000], where 10000 is the vocabulary size
        // this is for speed-up issue as per word matrix can be simply obtained using column slice
        w = builder.CreateLearnableParameter(msra::strfun::wstrprintf(L"W%d", numHiddenLayers), m_layerSizes[numHiddenLayers], m_layerSizes[numHiddenLayers + 1]);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);

        // the label is a dense matrix. each element is the word index
        label = builder.CreateInputNode(L"labels", 2 * (this->nce_noises + 1));

        bias = builder.CreateLearnableParameter(L"BiasVector", 1, m_layerSizes[m_layerSizes.size() - 1]);
        bias->Value().SetValue((ElemType) -std::log(m_layerSizes[m_layerSizes.size() - 1]));
        // m_net->InitLearnableParameters(bias, m_uniformInit, randomSeed++, std::log(m_layerSizes[m_layerSizes.size() - 1])* m_initValueScale);
        // clslogpostprob = builder.Times(clsweight, input, 1, L"ClassPostProb");

        output = AddTrainAndEvalCriterionNodes(input, label, w, L"TrainNodeNCEBasedCrossEntropy", L"EvalNodeNCEBasedCrossEntrpy", bias);

        m_net->AddToNodeGroup(L"output", output);

        if (m_needPrior)
        {
            prior = builder.Mean(label);
        }
    }

    return m_net;
}

// load a model file from Frank Seide's Microsoft-internal legacy tool "DBN.exe"
template <class ElemType>
ComputationNetworkPtr SimpleNetworkBuilder<ElemType>::BuildNetworkFromDbnFile(const std::wstring& dbnModelFileName)
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);

    std::string hdr, comment, name;
    int version;
    int numLayers, i;
    std::string layerType;

    unsigned long randomSeed = 1;

    ComputationNodePtr input, w, b, output, label, prior, scaledLogLikelihood;
    shared_ptr<PreComputedNodeBase<ElemType>> pcNodePtr;

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

    // move to CPU since element-wise operation is expensive and can go wrong in GPU
    int curDevId = globalStdDev.GetDeviceId();
    globalStdDev.TransferFromDeviceToDevice(curDevId, CPUDEVICE, true, false, false);
    for (int i = 0; i < globalStdDev.GetNumRows(); i++)
        globalStdDev(i, 0) = (ElemType) 1.0 / (const ElemType) globalStdDev(i, 0);
    globalStdDev.TransferFromDeviceToDevice(CPUDEVICE, curDevId, true, false, false);

    if (!CheckDbnTag(fstream, "BNET"))
        RuntimeError("Error reading DBN file - did not find expected tag BNET\n");

    for (i = 0; i < numLayers; i++) // 0th index is for input layer,
    {
        fstream >> layerType;

        Matrix<ElemType> wts = ReadMatrixFromDbnFile(fstream, std::string("W"));
        Matrix<ElemType> bias = ReadMatrixFromDbnFile(fstream, std::string("a")); // remnant from pretraining, not needed
        Matrix<ElemType> A = ReadMatrixFromDbnFile(fstream, std::string("b"));
        if (i == 0)
        {
            input = builder.CreateInputNode(L"features", wts.GetNumCols());
            m_net->AddToNodeGroup(L"feature", input);

            size_t frameDim = globalMean.GetNumRows();
            size_t numContextFrames = wts.GetNumCols() / frameDim;
            size_t contextDim = numContextFrames * frameDim;
            Matrix<ElemType> contextMean(contextDim, 1, m_deviceId);
            Matrix<ElemType> contextStdDev(contextDim, 1, m_deviceId);

            // move to CPU since element-wise operation is expensive and can go wrong in GPU
            contextMean.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
            contextStdDev.TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true, false, false);
            for (size_t j = 0; j < frameDim; j++)
            {
                for (size_t k = 0; k < numContextFrames; k++)
                {
                    contextMean(j + k * frameDim, 0) = (const ElemType) globalMean(j, 0);
                    contextStdDev(j + k * frameDim, 0) = (const ElemType) globalStdDev(j, 0);
                }
            }
            contextMean.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);
            contextStdDev.TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true, false, false);

            w = builder.Mean(input, L"MeanOfFeatures");
            static_pointer_cast<PreComputedNodeBase<ElemType>>(w)->SideLoadFromMatrix(contextMean);
            w->SetLearningRateMultiplier(0);

            b = builder.InvStdDev(input, L"InvStdOfFeatures");
            static_pointer_cast<PreComputedNodeBase<ElemType>>(b)->SideLoadFromMatrix(contextStdDev);
            b->SetLearningRateMultiplier(0);

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

        w = builder.CreateLearnableParameter(nameOfW, wts.GetNumRows(), wts.GetNumCols());
        w->Value().SetValue(wts);

        b = builder.CreateLearnableParameter(nameOfB, bias.GetNumRows(), 1);
        b->Value().SetValue(bias);

        if (layerType == "perceptron")
        {
            fprintf(stderr, "DBN: Reading (%lu x %lu) perceptron\n", (unsigned long) wts.GetNumRows(), (unsigned long) wts.GetNumCols());
            output = builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus);
        }
        else if (layerType == "rbmisalinearbernoulli")
        {
            fprintf(stderr, "DBN: Reading (%lu x %lu) linear layer\n", (unsigned long) wts.GetNumRows(), (unsigned long) wts.GetNumCols());
            output = builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus);
        }
        else // assume rbmbernoullibernoulli
        {
            fprintf(stderr, "DBN: Reading (%lu x %lu) non-linear layer\n", (unsigned long) wts.GetNumRows(), (unsigned long) wts.GetNumCols());
            output = ApplyNonlinearFunction(builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus), i, nameOfH);
            if (m_addDropoutNodes)
                input = builder.Dropout(output, L"Drop" + nameOfH);
        }

        input = output;
    }

    if (!CheckDbnTag(fstream, "ENET"))
        RuntimeError("Error reading DBN file - did not find expected tag ENET\n");
    // size_t outputLayerSize =  m_layerSizes[m_layerSizes.size()-1];

    label = builder.CreateInputNode(L"labels", m_outputLayerSize);

    if (layerType == "perceptron") // complete network
    {
        m_net->RenameNode(output, L"HLast");

        Matrix<ElemType> priorVals = ReadMatrixFromDbnFile(fstream, std::string("Pu"));
        assert(priorVals.GetNumCols() == 1 && priorVals.GetNumRows() == m_outputLayerSize);

        prior = builder.Mean(label, L"Prior");
        static_pointer_cast<PreComputedNodeBase<ElemType>>(prior)->SideLoadFromMatrix(priorVals);
        prior->SetLearningRateMultiplier(0);
    }
    else // pretrained network - need to add output layer, initalize
    {
        size_t outputLayerSize = 0;
        if (this->m_outputLayerSize >= 0)
            outputLayerSize = this->m_outputLayerSize;
        else if (m_layerSizes.size() > 0)
            m_layerSizes[m_layerSizes.size() - 1];
        else
            RuntimeError("Output layer size must be specified when converting a pre-trained network, use outputLayerSize=");

        size_t penultimateSize = input->GetSampleMatrixNumRows();

        wstring nameOfW = msra::strfun::wstrprintf(L"W%d", i);
        wstring nameOfB = msra::strfun::wstrprintf(L"B%d", i);
        wstring nameOfPrevH = msra::strfun::wstrprintf(L"H%d", i);
        wstring nameOfTimes = nameOfW + L"*" + nameOfPrevH;
        wstring nameOfPlus = nameOfTimes + L"+" + nameOfB;
        wstring nameOfH = msra::strfun::wstrprintf(L"H%d", i + 1);

        w = builder.CreateLearnableParameter(nameOfW, outputLayerSize, penultimateSize);
        m_net->InitLearnableParameters(w, m_uniformInit, randomSeed++, m_initValueScale);
        b = builder.CreateLearnableParameter(nameOfB, outputLayerSize, 1);
        output = builder.Plus(builder.Times(w, input, 1, nameOfTimes), b, nameOfPlus);
        m_net->RenameNode(output, L"HLast");

        if (m_needPrior)
        {
            Matrix<ElemType> zeros = Matrix<ElemType>::Zeros(outputLayerSize, 1, m_deviceId);
            prior = builder.Mean(label, L"Prior");
            static_pointer_cast<PreComputedNodeBase<ElemType>>(prior)->MarkComputed(false);
            prior->Value().SetValue(zeros);
        }
    }

    AddTrainAndEvalCriterionNodes(output, label);

    if (layerType == "perceptron" || m_needPrior)
    {
        input = builder.Log(prior, L"LogOfPrior");

        // following two lines is needed only if true probability is needed
        // output = builder.Softmax(output);
        // output = builder.Log(output);

        scaledLogLikelihood = builder.CreateComputationNode(OperationNameOf(MinusNode), L"ScaledLogLikelihood");
        scaledLogLikelihood->AttachInputs({ output, input });
        m_net->AddToNodeGroup(L"output", scaledLogLikelihood);
    }
    else
    {
        m_net->AddToNodeGroup(L"output", output);
    }

    if (!CheckDbnTag(fstream, "EDBN"))
        RuntimeError("Error reading DBN file - did not find expected tag ENET\n");

    // perform necessary validation and post-processing
    m_net->CompileNetwork();

    return m_net;
}

// layer is 0 based
template <class ElemType>
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
        output = input; // linear layer
        if (nodeName != L"")
            m_net->RenameNode(output, nodeName);
    }
    else
        LogicError("Unsupported nonlinear function.");

    return output;
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> SimpleNetworkBuilder<ElemType>::AddTrainAndEvalCriterionNodes(ComputationNodePtr input, ComputationNodePtr label, ComputationNodePtr matrix,
                                                                                                    const std::wstring trainNodeName, const std::wstring evalNodeName,
                                                                                                    ComputationNodePtr clspostprob, ComputationNodePtr trans)
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);

    m_net->AddToNodeGroup(L"label", label);

    ComputationNodePtr output;

    // BUGBUG: Use of 'tinput' conflicts with some criteria that expect their top weight matrix transposed, e.g. [200 x 10000] with vocab size of 10000 instead of [10000 x 200].
    //         E.g. ClassCrossEntropyWithSoftmax uses this, but that is incompatible with 'tinput.' Now 'tinput' is computed on demand, but if a criterion node is
    //         used that needs it, we will still have this incompatibility.
    ComputationNodePtr tinput = input;

    switch (m_trainCriterion)
    {
    case TrainingCriterion::CrossEntropyWithSoftmax:
        if (matrix != nullptr)
            tinput = builder.Times(matrix, input);
        output = builder.CrossEntropyWithSoftmax(label, tinput, (trainNodeName == L"") ? L"CrossEntropyWithSoftmax" : trainNodeName);
        break;
    case TrainingCriterion::SquareError:
        if (matrix != nullptr)
            tinput = builder.Times(matrix, input);
        output = builder.SquareError(label, tinput, (trainNodeName == L"") ? L"SquareError" : trainNodeName);
        break;
    case TrainingCriterion::Logistic:
        if (matrix != nullptr)
            tinput = builder.Times(matrix, input);
        output = builder.Logistic(label, tinput, (trainNodeName == L"") ? L"Logistic" : trainNodeName);
        break;
#ifdef COMING_SOON
    case TrainingCriterion::CRF:
        assert(trans != nullptr);
        output = builder.CRF(label, input, trans, (trainNodeName == L"") ? L"CRF" : trainNodeName);
        break;
#endif
    case TrainingCriterion::ClassCrossEntropyWithSoftmax:
        output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"ClassCrossEntropyWithSoftmax" : trainNodeName);
        break;
    case TrainingCriterion::NCECrossEntropyWithSoftmax:
        output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
        // output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (trainNodeName == L"") ? L"NoiseContrastiveEstimationNode" : trainNodeName);
        break;
    default:
        LogicError("Unsupported training criterion.");
    }
    m_net->AddToNodeGroup(L"criterion", output);

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
            if (matrix != nullptr && tinput == input)
                tinput = builder.Times(matrix, input);
            // output = builder.CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"")?L"EvalCrossEntropyWithSoftmax":evalNodeName);
            output = builder.CrossEntropyWithSoftmax(label, tinput, (evalNodeName == L"") ? L"CrossEntropyWithSoftmax" : evalNodeName);
            break;
        case EvalCriterion::ClassCrossEntropyWithSoftmax:
            // output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"EvalClassCrossEntropyWithSoftmax" : evalNodeName);
            output = builder.ClassCrossEntropyWithSoftmax(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"ClassCrossEntropyWithSoftmax" : evalNodeName);
            break;
        case EvalCriterion::NCECrossEntropyWithSoftmax:
            output = builder.NoiseContrastiveEstimation(label, input, matrix, clspostprob, (evalNodeName == L"") ? L"NoiseContrastiveEstimationNode" : evalNodeName);
            break;
        case EvalCriterion::SquareError:
            if (matrix != nullptr && tinput == input)
                tinput = builder.Times(matrix, input);
            // output = builder.SquareError(label, tinput, (evalNodeName == L"")?L"EvalSquareError":evalNodeName);
            output = builder.SquareError(label, tinput, (evalNodeName == L"") ? L"SquareError" : evalNodeName);
            break;
        case EvalCriterion::Logistic:
            if (matrix != nullptr && tinput == input)
                tinput = builder.Times(matrix, input);
            // output = builder.SquareError(label, tinput, (evalNodeName == L"")?L"EvalSquareError":evalNodeName);
            output = builder.Logistic(label, tinput, (evalNodeName == L"") ? L"Logistic" : evalNodeName);
            break;
        case EvalCriterion::ErrorPrediction:
            if (matrix != nullptr && tinput == input)
                tinput = builder.Times(matrix, input);
            output = builder.ErrorPrediction(label, tinput, (evalNodeName == L"") ? L"EvalErrorPrediction" : evalNodeName);
            break;
#ifdef COMING_SOON
        case EvalCriterion::CRF:
            assert(trans != nullptr);
            if (matrix != nullptr && tinput == input)
                tinput = builder.Times(matrix, input);
            output = builder.CRF(label, tinput, trans, (evalNodeName == L"") ? L"EvalCRF" : evalNodeName);
            break;
#endif
        default:
            LogicError("Unsupported training criterion.");
        }
        output->SetLearningRateMultiplier(0);
    }

    m_net->AddToNodeGroup(L"evaluation", output);

    return output;
}

template class SimpleNetworkBuilder<float>;
template class SimpleNetworkBuilder<double>;

// -----------------------------------------------------------------------
// and some helpers
// -----------------------------------------------------------------------

TrainingCriterion ParseTrainingCriterionString(wstring s)
{
    if      (EqualCI(s, L"crossEntropyWithSoftmax"))      return TrainingCriterion::CrossEntropyWithSoftmax;
    else if (EqualCI(s, L"sequenceWithSoftmax"))          return TrainingCriterion::SequenceWithSoftmax;
    else if (EqualCI(s, L"squareError"))                  return TrainingCriterion::SquareError;
    else if (EqualCI(s, L"logistic"))                     return TrainingCriterion::Logistic;
    else if (EqualCI(s, L"noiseContrastiveEstimation"))   return TrainingCriterion::NCECrossEntropyWithSoftmax;
    else if (EqualCI(s, L"classCrossEntropyWithSoftmax")) return TrainingCriterion::ClassCrossEntropyWithSoftmax;
    else LogicError("trainingCriterion: Invalid trainingCriterion value. Valid values are (crossEntropyWithSoftmax | squareError | logistic | classCrossEntropyWithSoftmax| sequenceWithSoftmax)");
}

EvalCriterion ParseEvalCriterionString(wstring s)
{
    if      (EqualCI(s, L"errorPrediction"))              return EvalCriterion::ErrorPrediction;
    else if (EqualCI(s, L"crossEntropyWithSoftmax"))      return EvalCriterion::CrossEntropyWithSoftmax;
    else if (EqualCI(s, L"sequenceWithSoftmax"))          return EvalCriterion::SequenceWithSoftmax; 
    else if (EqualCI(s, L"classCrossEntropyWithSoftmax")) return EvalCriterion::ClassCrossEntropyWithSoftmax;
    else if (EqualCI(s, L"logistic"))                     return EvalCriterion::Logistic;
    else if (EqualCI(s, L"noiseContrastiveEstimation"))   return EvalCriterion::NCECrossEntropyWithSoftmax;
    else if (EqualCI(s, L"squareError"))                  return EvalCriterion::SquareError;
    else LogicError("evalCriterion: Invalid trainingCriterion value. Valid values are (errorPrediction | crossEntropyWithSoftmax | squareError | logistic | sequenceWithSoftmax)");
}

}}}
