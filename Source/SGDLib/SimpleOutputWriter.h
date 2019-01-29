//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "DataReaderHelpers.h"
#include "Helpers.h"
#include "File.h"
#include "fileutil.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <cstdio>
#include "ProgressTracing.h"
#include "ComputationNetworkBuilder.h"
#include <algorithm>

using namespace std;

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <class ElemType>
class SimpleOutputWriter
{
    struct Sequence
    {
        //shared_ptr<Matrix<ElemType>> LabelMatrix;
        std::vector<size_t> labelseq;
        ElemType logP;
        size_t length;
        size_t processlength;
        shared_ptr<Matrix<ElemType>> decodeoutput;
        bool operator<(const Sequence& rhs) const
        {
            return logP < rhs.logP;
        }
    };
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef typename std::vector<Sequence>::iterator iterator;

public:
    SimpleOutputWriter(ComputationNetworkPtr net, int verbosity = 0)
        : m_net(net), m_verbosity(verbosity)
    {
    }

    void WriteOutput(IDataReader& dataReader, size_t mbSize, IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doWriterUnitTest = false)
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        if (outputNodeNames.size() == 0 && m_verbosity > 0)
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes = m_net->InputNodesForOutputs(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        StreamMinibatchInputs inputMatrices = DataReaderHelpers::RetrieveInputMatrices(inputNodes);

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, inputMatrices.GetStreamDescriptions(), numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize, nullptr))
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);
            m_net->ForwardProp(outputNodes);

            for (int i = 0; i < outputNodes.size(); i++)
                outputMatrices[outputNodes[i]->NodeName()] = (void*) (&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());

            if (doWriterUnitTest)
            {
                std::map<std::wstring, void*, nocase_compare> inputMatricesUnitTest;
                for (auto& iter : inputMatrices)
                    inputMatricesUnitTest[iter.first] = (void*) iter.second.matrix.get(); // BUGBUG: void* are evil
                dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
            }
            else
                dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);

            totalEpochSamples += actualMBSize;

            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd();
        }

        if (m_verbosity > 0)
            fprintf(stderr, "Total Samples Evaluated = %lu\n", (unsigned long) totalEpochSamples);

        // clean up
    }

    void WriteOutput_greedy(IDataReader& dataReader, size_t mbSize, IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doWriterUnitTest = false)
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        if (outputNodeNames.size() == 0 && m_verbosity > 0)
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        //get encode input matrix
        std::vector<std::wstring> encodeOutputNodeNames(outputNodeNames.begin(), outputNodeNames.begin() + 1);
        std::vector<ComputationNodeBasePtr> encodeOutputNodes = m_net->OutputNodesByName(encodeOutputNodeNames);
        std::vector<ComputationNodeBasePtr> encodeInputNodes = m_net->InputNodesForOutputs(encodeOutputNodeNames);
        StreamMinibatchInputs encodeInputMatrices = DataReaderHelpers::RetrieveInputMatrices(encodeInputNodes);

        //start encode network
        dataReader.StartMinibatchLoop(mbSize, 0, encodeInputMatrices.GetStreamDescriptions(), numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        m_net->StartEvaluateMinibatchLoop(encodeOutputNodes[0]);

        //get decode input matrix
        std::vector<std::wstring> decodeOutputNodeNames(outputNodeNames.begin() + 1, outputNodeNames.begin() + 2);
        std::vector<ComputationNodeBasePtr> decodeOutputNodes = m_net->OutputNodesByName(decodeOutputNodeNames);
        std::vector<ComputationNodeBasePtr> decodeinputNodes = m_net->InputNodesForOutputs(decodeOutputNodeNames);
        StreamMinibatchInputs decodeinputMatrices = DataReaderHelpers::RetrieveInputMatrices(decodeinputNodes);

        //get merged input
        ComputationNodeBasePtr PlusNode = m_net->GetNodeFromName(outputNodeNames[2]);
        ComputationNodeBasePtr PlusTransNode = m_net->GetNodeFromName(outputNodeNames[3]);
        //StreamMinibatchInputs PlusinputMatrices =
        std::vector<ComputationNodeBasePtr> Plusnodes, Plustransnodes;
        Plusnodes.push_back(PlusNode);
        Plustransnodes.push_back(PlusTransNode);
        //start decode network
        m_net->StartEvaluateMinibatchLoop(decodeOutputNodes[0]);
        auto lminput = decodeinputMatrices.begin();
        //dataReader.StartMinibatchLoop(1, 0, decodeinputMatrices.GetStreamDescriptions(), numOutputSamples);

        //(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeinputNodes[0])->Value()).SetValue();

        size_t deviceid = lminput->second.GetMatrix<ElemType>().GetDeviceId();
        //size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;
        Matrix<ElemType> encodeOutput(deviceid);
        Matrix<ElemType> decodeOutput(deviceid);
        Matrix<ElemType> greedyOutput(deviceid), greedyOutputMax(deviceid);
        Matrix<ElemType> sumofENandDE(deviceid), maxIdx(deviceid), maxVal(deviceid);
        Matrix<ElemType> lmin(deviceid);
        //encodeOutput.GetDeviceId
        const size_t numIterationsBeforePrintingProgress = 100;
        //size_t numItersSinceLastPrintOfProgress = 0;
        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, encodeInputMatrices, actualMBSize, nullptr))
        {
            //encode forward prop for whole utterance
            ComputationNetwork::BumpEvalTimeStamp(encodeInputNodes);
            m_net->ForwardProp(encodeOutputNodes[0]);
            encodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(encodeOutputNodes[0])->Value()));
            //encodeOutput.Print("encodeoutput");
            dataReader.DataEnd();

            //decode forward prop step by step
            size_t vocabSize = PlusTransNode->GetSampleMatrixNumRows();
            size_t blankId = vocabSize - 1;

            /*lmin.Resize(vocabSize, 12);
        lmin.SetValue(0.0);
        std::vector<size_t> labels = {4, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 3};
        for (size_t n = 0; n < labels.size(); n++)
        {
            lmin(labels[n], n) = 1;
        }
        lminput->second.pMBLayout->Init(1, 12);
        std::swap(lminput->second.GetMatrix<ElemType>(), lmin);
        lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 12);
        m_net->ForwardProp(outputNodes[0]);
        decodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[0])->Value()));
        decodeOutput.Print("decode output");*/

            lmin.Resize(vocabSize, 1);
            lmin.SetValue(0.0);
            lmin(blankId, 0) = 1;

            // Resetting layouts.
            lminput->second.pMBLayout->Init(1, 1);
            std::swap(lminput->second.GetMatrix<ElemType>(), lmin);
            lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 200);
            ComputationNetwork::BumpEvalTimeStamp(decodeinputNodes);
            DataReaderHelpers::NotifyChangedNodes<ElemType>(m_net, decodeinputMatrices);
            m_net->ForwardProp(decodeOutputNodes[0]);

            greedyOutputMax.Resize(vocabSize, 200);
            size_t lmt = 0;
            for (size_t t = 0; t < encodeOutput.GetNumCols(); t++)
            {
                decodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeOutputNodes[0])->Value()));
                //decodeOutput.Print("decode output");
                sumofENandDE.AssignSumOf(encodeOutput.ColumnSlice(t, 1), decodeOutput);
                //sumofENandDE.Print("sum");

                (&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusNode)->Value())->SetValue(sumofENandDE);
                //SumMatrix.SetValue(sumofENandDE);
                ComputationNetwork::BumpEvalTimeStamp(Plusnodes);
                //ComputationNetwork::ResetEvalTimeStamps();
                //DataReaderHelpers::NotifyChangedNodes<ElemType>(m_net, Plusnodes);
                auto PlusMBlayout = PlusNode->GetMBLayout();
                PlusMBlayout->Init(1, 1);
                PlusMBlayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 1);
                //m_net->ForwardProp(PlusTransNode);
                m_net->ForwardPropFromTo(Plusnodes, Plustransnodes);
                decodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusTransNode)->Value()));
                decodeOutput.VectorMax(maxIdx, maxVal, true);
                size_t maxId = (size_t)(maxIdx.Get00Element());
                if (maxId != blankId)
                {
                    //fprintf(stderr, "maxid: %d\n", (int) maxId);
                    lmin.Resize(vocabSize, 1);
                    lmin.SetValue(0.0);
                    lmin(maxId, 0) = 1.0;

                    greedyOutputMax.SetColumn(lmin, lmt);

                    std::swap(lminput->second.GetMatrix<ElemType>(), lmin);
                    lminput->second.pMBLayout->Init(1, 1);
                    lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, -1 - lmt, 199 - lmt);
                    ComputationNetwork::BumpEvalTimeStamp(decodeinputNodes);
                    DataReaderHelpers::NotifyChangedNodes<ElemType>(m_net, decodeinputMatrices);
                    m_net->ForwardProp(decodeOutputNodes[0]);

                    //m_net->ForwardPropFromTo(decodeOutputNodes[0], PlusTransNode);
                    lmt++;
                    //fprintf(stderr, "lmt: %d\n", (int) lmt);
                }

                //break;
            }
            greedyOutput.SetValue(greedyOutputMax.ColumnSlice(0, lmt));
            //greedyOutput.Print("greedy output");
            outputMatrices[decodeOutputNodeNames[0]] = (void*) (&greedyOutput);
            if (lmt == 0)
            {
                greedyOutput.Resize(vocabSize, 1);
                lmin.Resize(vocabSize, 1);
                lmin.SetValue(0.0);
                lmin(blankId, 0) = 1;
                greedyOutput.SetColumn(lmin, 0);
                lmt = 1;
            }
            dataWriter.SaveData(0, outputMatrices, lmt, lmt, 0);
            //break;
        }

        //decode

        // clean up
    }

    Sequence newSeq(size_t numRow, size_t numCol, DEVICEID_TYPE deviceId)
    {
        Sequence oneSeq = {std::vector<size_t>(), 0.0, 0, 0, make_shared<Matrix<ElemType>>(numRow, (size_t) 1, deviceId)};
        return oneSeq;
    }
    Sequence newSeq(Sequence a)
    {
        Sequence oneSeq;
        oneSeq.labelseq = a.labelseq;
        oneSeq.logP = a.logP;
        oneSeq.length = a.length;
        oneSeq.processlength = a.processlength;
        oneSeq.decodeoutput = make_shared<Matrix<ElemType>>(a.decodeoutput->GetNumRows(), (size_t) 1, a.decodeoutput->GetDeviceId());
        oneSeq.decodeoutput->SetValue(*(a.decodeoutput));

        return oneSeq;
    }
    void deleteSeq(Sequence oneSeq)
    {
        oneSeq.decodeoutput->ReleaseMemory();
        vector<size_t>().swap(oneSeq.labelseq);
    }
    iterator getMaxSeq(vector<Sequence> seqs)
    {
        double MaxlogP = LOGZERO;
        typename vector<Sequence>::iterator maxIt;
        for (auto it = seqs.begin(); it != seqs.end(); it++)
        {
            if (it->logP > MaxlogP)
                maxIt = it;
        }
        return maxIt;
    }
    void extendSeq(Sequence& insequence, size_t labelId, ElemType logP)
    {
        insequence.labelseq.push_back(labelId);
        insequence.logP = logP;
        insequence.length++;
    }

    void forward_decode(Sequence oneSeq, StreamMinibatchInputs decodeinputMatrices, DEVICEID_TYPE deviceID, std::vector<ComputationNodeBasePtr> decodeOutputNodes,
                        std::vector<ComputationNodeBasePtr> decodeinputNodes, size_t vocabSize, size_t plength)
    {
        //        size_t labelLength = oneSeq.length;
        if (plength > oneSeq.processlength)
        {

            Matrix<ElemType> lmin(deviceID);

            //greedyOutput.SetValue(greedyOutputMax.ColumnSlice(0, lmt));
            lmin.Resize(vocabSize, plength);
            lmin.SetValue(0.0);
            for (size_t n = 0; n < plength; n++)
            {
                lmin(oneSeq.labelseq[n], n) = 1.0;
            }
            auto lminput = decodeinputMatrices.begin();
            lminput->second.pMBLayout->Init(1, plength);
            //std::swap(lminput->second.GetMatrix<ElemType>(), lmin);
            lminput->second.GetMatrix<ElemType>().SetValue(lmin);
            lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, plength);
            ComputationNetwork::BumpEvalTimeStamp(decodeinputNodes);
            DataReaderHelpers::NotifyChangedNodes<ElemType>(m_net, decodeinputMatrices);
            m_net->ForwardProp(decodeOutputNodes[0]);
            //Matrix<ElemType> tempMatrix = *(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeOutputNodes[0])->Value());
            oneSeq.decodeoutput->SetValue((*(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeOutputNodes[0])->Value())).ColumnSlice(plength - 1, 1));
            oneSeq.processlength = plength;
            lmin.ReleaseMemory();
        }
    }
    bool compareseq(Sequence a, Sequence b)
    {
        return a.logP < b.logP;
    }

    vector<pair<size_t, ElemType>> getTopN(Microsoft::MSR::CNTK::Matrix<ElemType>& prob, size_t N)
    {
        vector<pair<size_t, ElemType>> datapair;
        typedef vector<pair<size_t, ElemType>>::value_type ValueType;
        ElemType* probdata = prob.CopyToArray();
        for (size_t n = 0; n < prob.GetNumRows(); n++)
        {
            datapair.push_back(ValueType(n, probdata[n]));
        }
        nth_element(datapair.begin(), datapair.begin() + N, datapair.end(), [](ValueType const& x, ValueType const& y) -> bool {
            return y.second < x.second;
        });
        delete probdata;
        return datapair;
    }
    //check whether a is the prefix of b
    bool isPrefix(Sequence a, Sequence b)
    {
        if (a.labelseq == b.labelseq || a.labelseq.size() >= b.labelseq.size())
            return false;
        for (size_t n = 0; n < a.labelseq.size(); n++)
        {
            if (a.labelseq[n] != b.labelseq[n])
                return false;
        }
        return true;
    }

    void forwardmerged(Sequence a, size_t t, Matrix<ElemType>& sumofENandDE, Matrix<ElemType>& encodeOutput, Matrix<ElemType>& decodeOutput, ComputationNodeBasePtr PlusNode, ComputationNodeBasePtr PlusTransNode, std::vector<ComputationNodeBasePtr> Plusnodes, std::vector<ComputationNodeBasePtr> Plustransnodes)
    {
        sumofENandDE.AssignSumOf(encodeOutput.ColumnSlice(t, 1), *(a.decodeoutput));
        //sumofENandDE.InplaceLogSoftmax(true);

        //plus broadcast
        (&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusNode)->Value())->SetValue(sumofENandDE);
        //SumMatrix.SetValue(sumofENandDE);
        ComputationNetwork::BumpEvalTimeStamp(Plusnodes);
        auto PlusMBlayout = PlusNode->GetMBLayout();
        PlusMBlayout->Init(1, 1);
        PlusMBlayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 1);
        m_net->ForwardPropFromTo(Plusnodes, Plustransnodes);
        decodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusTransNode)->Value()));
        //decodeOutput.VectorMax(maxIdx, maxVal, true);
        decodeOutput.InplaceLogSoftmax(true);
    }
    void WriteOutput_beam(IDataReader& dataReader, size_t mbSize, IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames,
                          size_t numOutputSamples = requestDataSize, bool doWriterUnitTest = false, size_t beamSize = 10, size_t expandBeam = 20)
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        //size_t beamSize = 10;
        if (outputNodeNames.size() == 0 && m_verbosity > 0)
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        //get encode input matrix
        std::vector<std::wstring> encodeOutputNodeNames(outputNodeNames.begin(), outputNodeNames.begin() + 1);
        std::vector<ComputationNodeBasePtr> encodeOutputNodes = m_net->OutputNodesByName(encodeOutputNodeNames);
        std::vector<ComputationNodeBasePtr> encodeInputNodes = m_net->InputNodesForOutputs(encodeOutputNodeNames);
        StreamMinibatchInputs encodeInputMatrices = DataReaderHelpers::RetrieveInputMatrices(encodeInputNodes);

        //start encode network
        dataReader.StartMinibatchLoop(mbSize, 0, encodeInputMatrices.GetStreamDescriptions(), numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        m_net->StartEvaluateMinibatchLoop(encodeOutputNodes[0]);

        //get decode input matrix
        std::vector<std::wstring> decodeOutputNodeNames(outputNodeNames.begin() + 1, outputNodeNames.begin() + 2);
        std::vector<ComputationNodeBasePtr> decodeOutputNodes = m_net->OutputNodesByName(decodeOutputNodeNames);
        std::vector<ComputationNodeBasePtr> decodeinputNodes = m_net->InputNodesForOutputs(decodeOutputNodeNames);
        StreamMinibatchInputs decodeinputMatrices = DataReaderHelpers::RetrieveInputMatrices(decodeinputNodes);

        //get merged input
        ComputationNodeBasePtr PlusNode = m_net->GetNodeFromName(outputNodeNames[2]);
        ComputationNodeBasePtr PlusTransNode = m_net->GetNodeFromName(outputNodeNames[3]);
        //StreamMinibatchInputs PlusinputMatrices =
        std::vector<ComputationNodeBasePtr> Plusnodes, Plustransnodes;
        Plusnodes.push_back(PlusNode);
        Plustransnodes.push_back(PlusTransNode);

        //start decode network
        m_net->StartEvaluateMinibatchLoop(decodeOutputNodes[0]);
        auto lminput = decodeinputMatrices.begin();
        //dataReader.StartMinibatchLoop(1, 0, decodeinputMatrices.GetStreamDescriptions(), numOutputSamples);

        //(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeinputNodes[0])->Value()).SetValue();

        DEVICEID_TYPE deviceid = lminput->second.GetMatrix<ElemType>().GetDeviceId();
        //size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;
        Matrix<ElemType> encodeOutput(deviceid);
        Matrix<ElemType> decodeOutput(deviceid);
        Matrix<ElemType> greedyOutput(deviceid), greedyOutputMax(deviceid);
        Matrix<ElemType> sumofENandDE(deviceid), maxIdx(deviceid), maxVal(deviceid);
        Matrix<ElemType> lmin(deviceid);
        //encodeOutput.GetDeviceId
        const size_t numIterationsBeforePrintingProgress = 100;
        //size_t numItersSinceLastPrintOfProgress = 0;
        size_t actualMBSize;
        vector<Sequence> CurSequences, nextSequences;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, encodeInputMatrices, actualMBSize, nullptr))
        {
            //encode forward prop for whole utterance
            ComputationNetwork::BumpEvalTimeStamp(encodeInputNodes);
            m_net->ForwardProp(encodeOutputNodes[0]);
            encodeOutput.SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(encodeOutputNodes[0])->Value()));
            //encodeOutput.Print("encodeoutput");
            dataReader.DataEnd();

            //decode forward prop step by step
            size_t vocabSize = PlusTransNode->GetSampleMatrixNumRows();
            size_t blankId = vocabSize - 1;

            nextSequences.clear();

            //initialize with blank ID
            Sequence oneSeq = newSeq(vocabSize, (size_t) 50, deviceid);
            extendSeq(oneSeq, blankId, 0.0);
            nextSequences.push_back(oneSeq);

            // loop for each t
            for (size_t t = 0; t < encodeOutput.GetNumCols(); t++)
            {
                for (size_t n = 0; n < CurSequences.size(); n++)
                {
                    deleteSeq(CurSequences[n]);
                }
                vector<Sequence>().swap(CurSequences);
                CurSequences = nextSequences;

                vector<Sequence>().swap(nextSequences);
                //deal with the same prefix
                sort(CurSequences.begin(), CurSequences.end(),
                     [](const Sequence& a, const Sequence& b) -> bool {
                         return a.labelseq.size() > b.labelseq.size();
                     });
                for (size_t n = 0; n < CurSequences.size() - 1; n++)
                {
                    for (size_t h = n + 1; h < CurSequences.size(); h++)
                    {
                        if (isPrefix(CurSequences[h], CurSequences[n]))
                        {
                            //forward_prop the prefix
                            forward_decode(CurSequences[h], decodeinputMatrices, deviceid, decodeOutputNodes, decodeinputNodes, vocabSize, CurSequences[h].labelseq.size());

                            forwardmerged(CurSequences[h], t, sumofENandDE, encodeOutput, decodeOutput, PlusNode, PlusTransNode, Plusnodes, Plustransnodes);
                            
                            size_t idx = CurSequences[h].labelseq.size();
                            ElemType curlogp = CurSequences[h].logP + decodeOutput(CurSequences[n].labelseq[idx], 0);
                            for (size_t k = idx; k < CurSequences[n].labelseq.size() - 1; k++)
                            {
                                forward_decode(CurSequences[n], decodeinputMatrices, deviceid, decodeOutputNodes, decodeinputNodes, vocabSize, k + 1);
                                forwardmerged(CurSequences[n], t, sumofENandDE, encodeOutput, decodeOutput, PlusNode, PlusTransNode, Plusnodes, Plustransnodes);
                                

                                curlogp += decodeOutput(CurSequences[n].labelseq[k + 1], 0);
                            }
                            CurSequences[n].logP = decodeOutput.LogAdd(curlogp, CurSequences[n].logP);
                        }
                    }
                }
                //nextSequences.clear();
                while (true)
                {

                    //auto maxSeq = getMaxSeq(CurSequences);
                    auto maxSeq = std::max_element(CurSequences.begin(), CurSequences.end());
                    //std::max_element()
                    //auto pos = std::find(CurSequences.begin(), CurSequences.end(), maxSeq);
                    Sequence tempSeq = newSeq(*maxSeq);
                    deleteSeq(*maxSeq);
                    CurSequences.erase(maxSeq);
                    forward_decode(tempSeq, decodeinputMatrices, deviceid, decodeOutputNodes, decodeinputNodes, vocabSize, tempSeq.labelseq.size());
                    forwardmerged(tempSeq, t, sumofENandDE, encodeOutput, decodeOutput, PlusNode, PlusTransNode, Plusnodes, Plustransnodes);
                    
                    //sumofENandDE.Print("sum");
                    //sort log posterior and get best N labels
                    vector<pair<size_t, ElemType>> topN = getTopN(decodeOutput, expandBeam);
                    /*ElemType* logP = decodeOutput.CopyToArray();
                    std::priority_queue<std::pair<double, int>> q;
                    int iLabel;
                    for (iLabel = 0; iLabel < vocabSize; iLabel++)
                    {
                        q.push(std::pair<double, int>((double) logP[iLabel], iLabel));
                    }
                    for (iLabel = 0; iLabel < beamSize; iLabel++)
                    {
                        auto Elem = q.top();
                        Sequence seqK = newSeq(tempSeq);
                        double newlogP = Elem.first + tempSeq.logP;
                        seqK.logP = newlogP;

                        if (Elem.second == blankId)
                        {
                            nextSequences.push_back(seqK);
                            q.pop();
                            continue;
                        }
                        extendSeq(seqK, Elem.second, newlogP);
                        CurSequences.push_back(seqK);
                        q.pop();
                    }*/
                    int iLabel;
                    for (iLabel = 0; iLabel < expandBeam; iLabel++)
                    {

                        Sequence seqK = newSeq(tempSeq);
                        ElemType newlogP = topN[iLabel].second + tempSeq.logP;
                        seqK.logP = newlogP;

                        if (topN[iLabel].first == blankId)
                        {
                            nextSequences.push_back(seqK);
                            continue;
                        }
                        extendSeq(seqK, topN[iLabel].first, newlogP);
                        CurSequences.push_back(seqK);
                    }
                    vector<pair<size_t, ElemType>>().swap(topN);
                    //delete topN;
                    deleteSeq(tempSeq);

                    if (CurSequences.size() == 0)
                        break;
                    auto ya = std::max_element(CurSequences.begin(), CurSequences.end());
                    //auto yb = std::max_element(nextSequences.begin(), nextSequences.end());
                    if (nextSequences.size() > beamSize) //                        && yb->logP > ya->logP)
                    {
                        nth_element(nextSequences.begin(), nextSequences.begin() + beamSize, nextSequences.end(),
                            [](const Sequence& a, const Sequence& b)->bool {
                            return a.logP > b.logP;
                        });
                        if (nextSequences[beamSize].logP > ya->logP)
                            break;
                    }
                    //break;
                    //std::nth_element(logP, logP + beamSize, )
                }
                std::sort(nextSequences.begin(), nextSequences.end());
                std::reverse(nextSequences.begin(), nextSequences.end());
                if (nextSequences.size() > beamSize)
                {
                    for (size_t n = beamSize; n < nextSequences.size(); n++)
                    {
                        deleteSeq(nextSequences[n]);
                    }
                }
                for (size_t iseq = nextSequences.size(); iseq > beamSize; iseq--)
                    nextSequences.pop_back();
                //break;
            }

            auto yb = std::max_element(nextSequences.begin(), nextSequences.end());
            size_t lmt = yb->length;
            greedyOutput.Resize(vocabSize, lmt);
            greedyOutput.SetValue(0.0);
            for (size_t n = 0; n < lmt; n++)
            {
                greedyOutput(yb->labelseq[n], n) = 1.0;
            }
            //greedyOutput.SetValue(yb->LabelMatrix->ColumnSlice(0, lmt));
            //greedyOutput.Print("greedy output");
            outputMatrices[decodeOutputNodeNames[0]] = (void*) (&greedyOutput);
            if (lmt == 0)
            {
                greedyOutput.Resize(vocabSize, 1);
                lmin.Resize(vocabSize, 1);
                lmin.SetValue(0.0);
                lmin(blankId, 0) = 1;
                greedyOutput.SetColumn(lmin, 0);
                lmt = 1;
            }
            for (size_t n = 0; n < CurSequences.size(); n++)
            {
                deleteSeq(CurSequences[n]);
            }
            vector<Sequence>().swap(CurSequences);
            for (size_t n = 0; n < nextSequences.size(); n++)
            {
                deleteSeq(nextSequences[n]);
            }
            vector<Sequence>().swap(nextSequences);
            dataWriter.SaveData(0, outputMatrices, lmt, lmt, 0);
            //break;
        }

        //decode

        // clean up
    }

    // Perform a single forward pass to obtain the output values from a network
    void WriteOutput(IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doUnitTest = false)
    {
        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        m_net->ForwardProp(outputNodes);
        for (int i = 0; i < outputNodes.size(); i++)
            outputMatrices[outputNodes[i]->NodeName()] = (void*) (&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());

        // TODO: What should the data size be?
        dataWriter.SaveData(0, outputMatrices, 1, 1, 0);
    }

    void WriteMinibatch(FILE* f, ComputationNodePtr node,
                        const WriteFormattingOptions& formattingOptions, char formatChar, std::string valueFormatString, std::vector<std::string>& labelMapping,
                        size_t numMBsRun, bool gradient, std::function<std::string(size_t)>& idToKeyMapping)
    {
        const auto sequenceSeparator = formattingOptions.Processed(node->NodeName(), formattingOptions.sequenceSeparator, numMBsRun);
        const auto sequencePrologue = formattingOptions.Processed(node->NodeName(), formattingOptions.sequencePrologue, numMBsRun);
        const auto sequenceEpilogue = formattingOptions.Processed(node->NodeName(), formattingOptions.sequenceEpilogue, numMBsRun);
        const auto elementSeparator = formattingOptions.Processed(node->NodeName(), formattingOptions.elementSeparator, numMBsRun);
        const auto sampleSeparator = formattingOptions.Processed(node->NodeName(), formattingOptions.sampleSeparator, numMBsRun);

        node->WriteMinibatchWithFormatting(f, FrameRange(), SIZE_MAX, SIZE_MAX, formattingOptions.transpose, formattingOptions.isCategoryLabel, formattingOptions.isSparse, labelMapping,
                                           sequenceSeparator, sequencePrologue, sequenceEpilogue, elementSeparator, sampleSeparator,
                                           valueFormatString, gradient, false, idToKeyMapping);
    }

    void InsertNode(std::vector<ComputationNodeBasePtr>& allNodes, ComputationNodeBasePtr parent, ComputationNodeBasePtr newNode)
    {
        newNode->SetInput(0, parent);
        for (auto node : allNodes)
        {
            size_t i = 0;
            for (auto n : node->GetInputs())
            {
                if (n == parent)
                    node->SetInput(i, newNode);
                ++i;
            }
        }
    }

    // TODO: Remove code dup with above function by creating a fake Writer object and then calling the other function.
    void WriteOutput(IDataReader& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, const WriteFormattingOptions& formattingOptions, size_t numOutputSamples = requestDataSize, bool nodeUnitTest = false, bool writeSequenceKey = false)
    {
        // In case of unit test, make sure backprop works
        ScopedNetworkOperationMode modeGuard(m_net, nodeUnitTest ? NetworkOperationMode::training : NetworkOperationMode::inferring);

        std::vector<ComputationNodeBasePtr> outputNodes = m_net->OutputNodesByName(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes = m_net->InputNodesForOutputs(outputNodeNames);
        std::vector<ComputationNodePtr> gradientNodes;
        std::vector<ComputationNodeBasePtr> allOutputNodes = outputNodes;

        if (!nodeUnitTest) // regular operation
        {
            m_net->AllocateAllMatrices({}, outputNodes, nullptr); // don't allocate for backward pass
        }
        else // we mis-appropriate this code for unit testing of the back-prop path
        {
            // Unit test only makes sense for one output node.
            if (outputNodes.size() != 1)
                RuntimeError("Expected exactly 1 output node for unit test, got %d.", (int) outputNodes.size());

            // Set up machinery to output gradients alongside forward pass output
            // Gradients are not passed on to inputs. Need to hook an identity function in between.
            ComputationNetworkBuilder<ElemType> builder(*m_net);
            auto allInputs = inputNodes;
            auto allParameters = m_net->LearnableParameterNodes(outputNodes[0]);
            allInputs.insert(allInputs.end(), allParameters.begin(), allParameters.end());
            auto allNodes = m_net->GetAllNodes();

            for (auto inputNode : allInputs)
            {
                auto parent = dynamic_pointer_cast<ComputationNode<ElemType>>(inputNode);
                auto newNode = builder.Pass(parent, inputNode->NodeName() + L".grad");
                newNode->SetLearningRateMultiplier(1.0); // Forces gradient update. Otherwise, backprop might get pruned from this path.
                InsertNode(allNodes, parent, newNode);
                gradientNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(newNode));
                allOutputNodes.push_back(newNode);
            }

            // Update the evaluation order, and other things.
            m_net->CompileNetwork();

            // Allocate memory for forward and backward computation. In case of unit test, treat the output node
            // like a criterion node. Submitting a node as parameter 3 here will allocate the gradients.
            m_net->AllocateAllMatrices({}, outputNodes, outputNodes[0]);
        }

        StreamMinibatchInputs inputMatrices = DataReaderHelpers::RetrieveInputMatrices(inputNodes);

        // load a label mapping if requested
        std::vector<std::string> labelMapping;
        if ((formattingOptions.isCategoryLabel || formattingOptions.isSparse) && !formattingOptions.labelMappingFile.empty())
            File::LoadLabelFile(formattingOptions.labelMappingFile, labelMapping);

        // open output files
        File::MakeIntermediateDirs(outputPath);
        std::map<ComputationNodeBasePtr, shared_ptr<File>> outputStreams; // TODO: why does unique_ptr not work here? Complains about non-existent default_delete()
        for (auto& onode : allOutputNodes)
        {
            std::wstring nodeOutputPath = outputPath;
            if (nodeOutputPath != L"-")
                nodeOutputPath += L"." + onode->NodeName();
            auto f = make_shared<File>(nodeOutputPath, fileOptionsWrite | fileOptionsText);
            outputStreams[onode] = f;
        }

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, inputMatrices.GetStreamDescriptions(), numOutputSamples);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;

        for (auto& onode : outputNodes)
        {
            FILE* f = *outputStreams[onode];
            fprintfOrDie(f, "%s", formattingOptions.prologue.c_str());
        }

        size_t actualMBSize;
        const size_t numIterationsBeforePrintingProgress = 100;
        size_t numItersSinceLastPrintOfProgress = 0;
        char formatChar = !formattingOptions.isCategoryLabel ? 'f' : !formattingOptions.labelMappingFile.empty() ? 's' : 'u';
        std::string valueFormatString = "%" + formattingOptions.precisionFormat + formatChar; // format string used in fprintf() for formatting the values

        for (size_t numMBsRun = 0; DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize, nullptr); numMBsRun++)
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);
            m_net->ForwardProp(outputNodes);

            for (auto& onode : outputNodes)
            {
                // compute the node value
                // Note: Intermediate values are memoized, so in case of multiple output nodes, we only compute what has not been computed already.

                FILE* file = *outputStreams[onode];
                auto getKeyById = writeSequenceKey ? inputMatrices.m_getKeyById : std::function<std::string(size_t)>();
                WriteMinibatch(file, dynamic_pointer_cast<ComputationNode<ElemType>>(onode), formattingOptions, formatChar, valueFormatString, labelMapping, numMBsRun, /* gradient */ false, getKeyById);

                if (nodeUnitTest)
                    m_net->Backprop(onode);
            } // end loop over nodes

            if (nodeUnitTest)
            {
                for (auto& node : gradientNodes)
                {
                    FILE* file = *outputStreams[node];
                    if (!node->GradientPtr())
                    {
                        fprintf(stderr, "Warning: Gradient of node '%s' is empty. Not used in backward pass?", Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(node->NodeName().c_str())).c_str());
                    }
                    else
                    {
                        auto idToKeyMapping = std::function<std::string(size_t)>();
                        WriteMinibatch(file, node, formattingOptions, formatChar, valueFormatString, labelMapping, numMBsRun, /* gradient */ true, idToKeyMapping);
                    }
                }
            }
            totalEpochSamples += actualMBSize;

            fprintf(stderr, "Minibatch[%lu]: ActualMBSize = %lu\n", (unsigned long) numMBsRun, (unsigned long) actualMBSize);
            if (outputPath == L"-") // if we mush all nodes together on stdout, add some visual separator
                fprintf(stdout, "\n");

            numItersSinceLastPrintOfProgress = ProgressTracing::TraceFakeProgress(numIterationsBeforePrintingProgress, numItersSinceLastPrintOfProgress);

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd();
        } // end loop over minibatches

        for (auto& stream : outputStreams)
        {
            FILE* f = *stream.second;
            fprintfOrDie(f, "%s", formattingOptions.epilogue.c_str());
        }

        fprintf(stderr, "Written to %ls*\nTotal Samples Evaluated = %lu\n", outputPath.c_str(), (unsigned long) totalEpochSamples);

        // flush all files (where we can catch errors) so that we can then destruct the handle cleanly without error
        for (auto& iter : outputStreams)
            iter.second->Flush();
    }

private:
    ComputationNetworkPtr m_net;
    int m_verbosity;
    void operator=(const SimpleOutputWriter&); // (not assignable)
};

} // namespace CNTK
} // namespace MSR
} // namespace Microsoft
