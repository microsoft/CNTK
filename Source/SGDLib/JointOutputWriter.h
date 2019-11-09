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
#include "RecurrentNodes.h"
#include <algorithm>
#include "SimpleOutputWriter.h"

using namespace std;

namespace Microsoft
{
namespace MSR
{
namespace CNTK
{

template <class ElemType>
class JointOutputWriter : public SimpleOutputWriter<ElemType>
{
    struct SequenceJoint
    {
        //shared_ptr<Matrix<ElemType>> LabelMatrix;
        std::vector<size_t> labelseq;
        ElemType logP;
        size_t length;
        size_t processlength;
        size_t lengthwithblank;
        vector<shared_ptr<Matrix<ElemType>>> decodeoutput;
        bool operator<(const SequenceJoint& rhs) const
        {
            return logP < rhs.logP;
        }
        vector<unordered_map<wstring, shared_ptr<PastValueNode<ElemType>>>> nameToNodeValues;
    };
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    typedef typename std::vector<SequenceJoint>::iterator iterator;
    vector<unordered_map<wstring, vector<shared_ptr<PastValueNode<ElemType>>>>> m_nameToPastValueNodeCache;

public:
    JointOutputWriter(vector<ComputationNetworkPtr> nets, vector<ElemType> combination_weights, int verbosity = 0, int combination_method = 0)
        : SimpleOutputWriter(NULL, verbosity), combination_method(combination_method), m_nets(nets), combination_weights(combination_weights), m_verbosity(verbosity)
    {
        for (int i = 0; i < m_nets.size(); i++)
        {
            m_nameToPastValueNodeCache.push_back(unordered_map<wstring, vector<shared_ptr<PastValueNode<ElemType>>>>());
        }
    }

    SequenceJoint newSeq(size_t numRow, size_t numCol, DEVICEID_TYPE deviceId, size_t num_models)
    {
        SequenceJoint oneSeq = {std::vector<size_t>(), 0.0, 0, 0, 0, vector<shared_ptr<Matrix<ElemType>>>(num_models, make_shared<Matrix<ElemType>>(numRow, (size_t) 1, deviceId))};
        oneSeq.nameToNodeValues.resize(num_models);
        for (int j = 0; j < num_models; j++)
        {
            for (size_t i = 0; i < m_nodesToCache[j].size(); i++)
            {
                vector<ElemType> v;
                oneSeq.nameToNodeValues[j][m_nodesToCache[j][i]] = make_shared<PastValueNode<ElemType>>(deviceId, L"test");
            }
        }
        return oneSeq;
    }
    SequenceJoint newSeq(SequenceJoint& a, DEVICEID_TYPE deviceId)
    {
        SequenceJoint oneSeq;
        oneSeq.labelseq = a.labelseq;
        oneSeq.logP = a.logP;
        oneSeq.length = a.length;
        oneSeq.lengthwithblank = a.lengthwithblank;
        oneSeq.processlength = a.processlength;
        oneSeq.decodeoutput.resize(a.decodeoutput.size());
        oneSeq.nameToNodeValues.resize(a.decodeoutput.size());
        for (int i = 0; i < oneSeq.decodeoutput.size(); i++)
        {
            oneSeq.decodeoutput[i] = make_shared<Matrix<ElemType>>(a.decodeoutput[i]->GetNumRows(), (size_t) 1, a.decodeoutput[i]->GetDeviceId());
            oneSeq.decodeoutput[i]->SetValue(*(a.decodeoutput[i]));
        }
        unordered_map<wstring, shared_ptr<PastValueNode<ElemType>>>::iterator it;
        for (int i = 0; i < oneSeq.decodeoutput.size(); i++)
        {
            for (it = a.nameToNodeValues[i].begin(); it != a.nameToNodeValues[i].end(); it++)
            {
                auto itin = m_nameToPastValueNodeCache[i].find(it->first);
                if (itin != m_nameToPastValueNodeCache[i].end() && m_nameToPastValueNodeCache[i][it->first].size() > 0)
                {
                    oneSeq.nameToNodeValues[i][it->first] = m_nameToPastValueNodeCache[i][it->first].back();
                    m_nameToPastValueNodeCache[i][it->first].pop_back();
                }
                else
                {
                    oneSeq.nameToNodeValues[i][it->first] = make_shared<PastValueNode<ElemType>>(deviceId, it->first);
                }

                it->second->CopyTo(oneSeq.nameToNodeValues[i][it->first], it->first, CopyNodeFlags::copyNodeAll);
            }
        }
        return oneSeq;
    }
    void deleteSeq(SequenceJoint oneSeq)
    {
        unordered_map<wstring, shared_ptr<PastValueNode<ElemType>>>::iterator it;
        for (int i = 0; i < oneSeq.decodeoutput.size(); i++)
        {
            if (i >= m_nameToPastValueNodeCache.size())
            {
                continue;
            }
            for (it = oneSeq.nameToNodeValues[i].begin(); it != oneSeq.nameToNodeValues[i].end(); it++)
            {
                auto itin = m_nameToPastValueNodeCache[i].find(it->first);
                if (itin == m_nameToPastValueNodeCache[i].end())
                    m_nameToPastValueNodeCache[i][it->first] = vector<shared_ptr<PastValueNode<ElemType>>>();
                m_nameToPastValueNodeCache[i][it->first].push_back(oneSeq.nameToNodeValues[i][it->first]);
            }
            oneSeq.decodeoutput[i]->ReleaseMemory();
        }
        vector<size_t>().swap(oneSeq.labelseq);
    }
    iterator getMaxSeq(const vector<SequenceJoint>& seqs)
    {
        double MaxlogP = LOGZERO;
        typename vector<SequenceJoint>::iterator maxIt;
        for (auto it = seqs.begin(); it != seqs.end(); it++)
        {
            if (it->logP > MaxlogP)
                maxIt = it;
        }
        return maxIt;
    }
    iterator getMatchSeq(const vector<SequenceJoint>& seqs, const vector<size_t>& labelseq)
    {
        iterator it;
        for (it = seqs.begin(); it != seqs.end(); it++)
        {
            if (it->labelseq == labelseq)
                break;
        }
        return it;
    }

    void extendSeq(SequenceJoint& insequence, size_t labelId, ElemType logP)
    {
        insequence.labelseq.push_back(labelId);
        insequence.logP = logP;
        insequence.length++;
        insequence.lengthwithblank++;
    }

    bool compareseq(const SequenceJoint& a, const SequenceJoint& b)
    {
        return a.logP < b.logP;
    }

    void forward_decode(SequenceJoint& oneSeq, std::vector<StreamMinibatchInputs> decodeinputMatrices, DEVICEID_TYPE deviceID, const std::vector<std::vector<ComputationNodeBasePtr>>& decodeOutputNodes,
                        const std::vector<std::vector<ComputationNodeBasePtr>>& decodeinputNodes, size_t vocabSize, size_t plength)
    {
        //        size_t labelLength = oneSeq.length;
        if (oneSeq.processlength + 1 != plength && plength != oneSeq.processlength)
            LogicError("Current implementation assumes 1 step difference");
        if (plength != oneSeq.processlength)
        {

            Matrix<ElemType> lmin(deviceID);

            lmin.Resize(vocabSize, 1);
            lmin.SetValue(0.0);
            lmin(oneSeq.labelseq[plength - 1], 0) = 1.0;
            for (int i = 0; i < m_nets.size(); i++)
            {
                auto lminput = decodeinputMatrices[i].begin();
                lminput->second.pMBLayout->Init(1, 1);
                lminput->second.GetMatrix<ElemType>().SetValue(lmin);
                if (plength == 1)
                {
                    lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 1);
                }
                else
                {
                    lminput->second.pMBLayout->AddSequence(NEW_SEQUENCE_ID, 0, SentinelValueIndicatingUnspecifedSequenceBeginIdx, 1);
                }
                ComputationNetwork::BumpEvalTimeStamp(decodeinputNodes[i]);
            }
            vector<bool> shallowCopy(m_nets.size(), false);
            for (int j = 0; j < m_nets.size(); j++)
            {
                for (size_t i = 0; i < m_nodesToCache[j].size(); i++)
                {
                    auto nodePtr = m_nets[j]->GetNodeFromName(m_nodesToCache[j][i]);
                    if (oneSeq.nameToNodeValues[j][m_nodesToCache[j][i]]->Value().GetNumElements() > 0)
                    {
                        oneSeq.nameToNodeValues[j][m_nodesToCache[j][i]]->CopyTo(nodePtr, m_nodesToCache[j][i], CopyNodeFlags::copyNodeInputLinks);
                        shallowCopy[j] = true;
                    }
                }
                ComputationNetwork::BumpEvalTimeStamp(decodeinputNodes[j]);
            }
            for (int j = 0; j < m_nets.size(); j++)
            {
                DataReaderHelpers::NotifyChangedNodes<ElemType>(m_nets[j], decodeinputMatrices[j]);
                m_nets[j]->ForwardProp(decodeOutputNodes[j][0]);
                //(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeOutputNodes[j][0])->Value())->Print(NULL, 0, 10, 0, 1);
                oneSeq.decodeoutput[j]->SetValue((*(&dynamic_pointer_cast<ComputationNode<ElemType>>(decodeOutputNodes[j][0])->Value())));

                for (size_t i = 0; i < m_nodesToCache[j].size(); i++)
                {
                    auto nodePtr = m_nets[j]->GetNodeFromName(m_nodesToCache[j][i]);
                    if (shallowCopy[j])
                        nodePtr->CopyTo(oneSeq.nameToNodeValues[j][m_nodesToCache[j][i]], m_nodesToCache[j][i], CopyNodeFlags::copyNodeInputLinks);
                    else
                        nodePtr->CopyTo(oneSeq.nameToNodeValues[j][m_nodesToCache[j][i]], m_nodesToCache[j][i], CopyNodeFlags::copyNodeAll);
                }
            }
            oneSeq.processlength = plength;
            lmin.ReleaseMemory();
        }
    }

    void forwardmerged(SequenceJoint a, size_t t, std::vector<Matrix<ElemType>>& sumofENandDE, std::vector<Matrix<ElemType>>& encodeOutput, std::vector<Matrix<ElemType>>& decodeOutput, std::vector<ComputationNodeBasePtr> PlusNode,
                       std::vector<ComputationNodeBasePtr> PlusTransNode, std::vector<std::vector<ComputationNodeBasePtr>> Plusnodes, std::vector<std::vector<ComputationNodeBasePtr>> Plustransnodes, std::vector<Matrix<ElemType>>& Wm, std::vector<Matrix<ElemType>>& bm)
    {
        for (int i = 0; i < m_nets.size(); i++)
        {
            sumofENandDE[i].AssignSumOf(encodeOutput[i].ColumnSlice(t, 1), *(a.decodeoutput[i]));
            Matrix<ElemType> tempMatrix(encodeOutput[i].GetDeviceId());
            //plus broadcast
            (&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusNode[i])->Value())->SetValue(sumofENandDE[i]);
            ComputationNetwork::BumpEvalTimeStamp(Plusnodes[i]);
            auto PlusMBlayout = PlusNode[i]->GetMBLayout();
            PlusMBlayout->Init(1, 1);
            PlusMBlayout->AddSequence(NEW_SEQUENCE_ID, 0, 0, 1);
            m_nets[i]->ForwardPropFromTo(Plusnodes[i], Plustransnodes[i]);
            decodeOutput[i].SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(PlusTransNode[i])->Value()));
            tempMatrix.AssignProductOf(Wm[i], true, decodeOutput[i], false);
            decodeOutput[i].AssignSumOf(tempMatrix, bm[i]);
            decodeOutput[i].InplaceLogSoftmax(true);
        }
    }

    void WriteOutput_beam(IDataReader& dataReader, size_t mbSize, IDataWriter& dataWriter, const std::vector<std::wstring>& outputNodeNames,
                          size_t numOutputSamples = requestDataSize, bool doWriterUnitTest = false, size_t beamSize = 10, size_t expandBeam = 20, string dictfile = L"", ElemType thresh = 0.68)
    {
        for (int i = 0; i < m_nets.size(); i++)
        {
            ScopedNetworkOperationMode modeGuard(m_nets[i], NetworkOperationMode::inferring);
        }

        //size_t beamSize = 10;
        if (outputNodeNames.size() == 0 && m_verbosity > 0)
            fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");

        std::vector<std::vector<ComputationNodeBasePtr>> outputNodes(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            outputNodes[i] = m_nets[i]->OutputNodesByName(outputNodeNames);

            // allocate memory for forward computation
            m_nets[i]->AllocateAllMatrices({}, outputNodes[i], nullptr);
        }

        //vector "hey cortana"
        //vector<size_t> keywords {}
        //get encode input matrix
        std::vector<std::wstring> encodeOutputNodeNames(outputNodeNames.begin(), outputNodeNames.begin() + 1);
        std::vector<std::vector<ComputationNodeBasePtr>> encodeOutputNodes(m_nets.size());
        std::vector<std::vector<ComputationNodeBasePtr>> encodeInputNodes(m_nets.size());
        std::vector<StreamMinibatchInputs> encodeInputMatrices(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            encodeOutputNodes[i] = m_nets[i]->OutputNodesByName(encodeOutputNodeNames);
            encodeInputNodes[i] = m_nets[i]->InputNodesForOutputs(encodeOutputNodeNames);
            encodeInputMatrices[i] = DataReaderHelpers::RetrieveInputMatrices(encodeInputNodes[i]);
        }

        //start encode network
        dataReader.StartMinibatchLoop(mbSize, 0, encodeInputMatrices[0].GetStreamDescriptions(), numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        for (int i = 0; i < m_nets.size(); i++)
        {
            m_nets[i]->StartEvaluateMinibatchLoop(encodeOutputNodes[i][0]);
        }

        //get decode input matrix
        std::vector<std::wstring> decodeOutputNodeNames(outputNodeNames.begin() + 1, outputNodeNames.begin() + 2);
        std::vector<std::vector<ComputationNodeBasePtr>> decodeOutputNodes(m_nets.size());
        std::vector<std::list<ComputationNodeBasePtr>> pastValueNodes(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            decodeOutputNodes[i] = m_nets[i]->OutputNodesByName(decodeOutputNodeNames);
            pastValueNodes[i] = m_nets[i]->PastValueNodesForOutputs(decodeOutputNodes[i]);
        }
        std::list<ComputationNodeBasePtr>::iterator it;
        m_nodesToCache.resize(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            for (it = pastValueNodes[i].begin(); it != pastValueNodes[i].end(); ++it)
            {
                auto pastValueNode = dynamic_pointer_cast<PastValueNode<ElemType>>(*it); //DelayedValueNodeBase
                if (pastValueNode || !(*it)->NodeName().compare(0, 5, L"Loop_"))
                {
                    m_nodesToCache[i].push_back((*it)->NodeName());
                }
            }
        }
        std::vector<std::vector<ComputationNodeBasePtr>> decodeinputNodes(m_nets.size());
        std::vector<StreamMinibatchInputs> decodeinputMatrices(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            decodeinputNodes[i] = m_nets[i]->InputNodesForOutputs(decodeOutputNodeNames);
            decodeinputMatrices[i] = DataReaderHelpers::RetrieveInputMatrices(decodeinputNodes[i]);
        }

        //get merged input
        std::vector<ComputationNodeBasePtr> PlusNode(m_nets.size());
        std::vector<ComputationNodeBasePtr> PlusTransNode(m_nets.size());
        std::vector<ComputationNodeBasePtr> WmNode(m_nets.size());
        std::vector<ComputationNodeBasePtr> bmNode(m_nets.size());
        std::vector<std::vector<ComputationNodeBasePtr>> Plusnodes(m_nets.size());
        std::vector<std::vector<ComputationNodeBasePtr>> Plustransnodes(m_nets.size());
        for (int i = 0; i < m_nets.size(); i++)
        {
            PlusNode[i] = m_nets[i]->GetNodeFromName(outputNodeNames[2]);
            PlusTransNode[i] = m_nets[i]->GetNodeFromName(outputNodeNames[3]);
            WmNode[i] = m_nets[i]->GetNodeFromName(outputNodeNames[4]);
            bmNode[i] = m_nets[i]->GetNodeFromName(outputNodeNames[5]);
            Plusnodes[i].push_back(PlusNode[i]);
            Plustransnodes[i].push_back(PlusTransNode[i]);
        }

        //start decode network
        for (int i = 0; i < m_nets.size(); i++)
        {
            m_nets[i]->StartEvaluateMinibatchLoop(decodeOutputNodes[i][0]);
        }
        auto lminput = decodeinputMatrices[0].begin();

        DEVICEID_TYPE deviceid = lminput->second.GetMatrix<ElemType>().GetDeviceId();
        //size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;
        std::vector<Matrix<ElemType>> encodeOutput;
        std::vector<Matrix<ElemType>> decodeOutput;
        Matrix<ElemType> greedyOutput(deviceid), greedyOutputMax(deviceid);
        std::vector<Matrix<ElemType>> sumofENandDE, maxIdx, maxVal;
        Matrix<ElemType> lmin(deviceid);
        std::vector<Matrix<ElemType>> Wm, bm;
        for (int i = 0; i < m_nets.size(); i++)
        {
            encodeOutput.push_back(Matrix<ElemType>(deviceid));
            decodeOutput.push_back(Matrix<ElemType>(deviceid));
            sumofENandDE.push_back(Matrix<ElemType>(deviceid));
            maxIdx.push_back(Matrix<ElemType>(deviceid));
            maxVal.push_back(Matrix<ElemType>(deviceid));
            Wm.push_back(Matrix<ElemType>(deviceid));
            bm.push_back(Matrix<ElemType>(deviceid));
        }
        for (int i = 0; i < m_nets.size(); i++)
        {
            Wm[i].SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(WmNode[i])->Value()));
            bm[i].SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(bmNode[i])->Value()));
        }
        //encodeOutput.GetDeviceId
        const size_t numIterationsBeforePrintingProgress = 100;
        //size_t numItersSinceLastPrintOfProgress = 0;
        size_t actualMBSize;
        vector<SequenceJoint> CurSequences, nextSequences;

        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_nets[0], nullptr, false, false, encodeInputMatrices[0], actualMBSize, nullptr))
        {
            for (int i = 0; i < m_nets.size(); i++)
            {
                if (i > 0)
                {
                    // Copy encoder input data from first model to other models in the ensemble
                    encodeInputMatrices[i].begin()->second.pMBLayout->CopyFrom(encodeInputMatrices[0].begin()->second.pMBLayout);
                    encodeInputMatrices[i].begin()->second.GetMatrix<ElemType>().SetValue(encodeInputMatrices[0].begin()->second.GetMatrix<ElemType>());
                }

                //encode forward prop for whole utterance
                ComputationNetwork::BumpEvalTimeStamp(encodeInputNodes[i]);

                //forward prop encoder network
                m_nets[i]->ForwardProp(encodeOutputNodes[i][0]);
                encodeOutput[i].SetValue(*(&dynamic_pointer_cast<ComputationNode<ElemType>>(encodeOutputNodes[i][0])->Value()));
                //encodeOutput.Print("encodeoutput");
            }
            dataReader.DataEnd();

            //decode forward prop step by step
            size_t vocabSize = bm[0].GetNumRows();
            size_t blankId = vocabSize - 1;

            nextSequences.clear();

            //initialize with blank ID
            SequenceJoint oneSeq = newSeq(vocabSize, (size_t) 50, deviceid, m_nets.size());
            extendSeq(oneSeq, blankId, 0.0);
            nextSequences.push_back(oneSeq);

            // loop for each frame
            for (size_t t = 0; t < encodeOutput[0].GetNumCols(); t++)
            {
                for (size_t n = 0; n < CurSequences.size(); n++)
                {
                    deleteSeq(CurSequences[n]);
                }
                vector<SequenceJoint>().swap(CurSequences);
                CurSequences = nextSequences;

                vector<SequenceJoint>().swap(nextSequences);
                while (true)
                {
                    //auto maxSeq = getMaxSeq(CurSequences);
                    auto maxSeq = std::max_element(CurSequences.begin(), CurSequences.end());
                    //std::max_element()
                    //auto pos = std::find(CurSequences.begin(), CurSequences.end(), maxSeq);
                    SequenceJoint tempSeq = newSeq(*maxSeq, deviceid);
                    deleteSeq(*maxSeq);
                    CurSequences.erase(maxSeq);
                    forward_decode(tempSeq, decodeinputMatrices, deviceid, decodeOutputNodes, decodeinputNodes, vocabSize, tempSeq.labelseq.size());
                    forwardmerged(tempSeq, t, sumofENandDE, encodeOutput, decodeOutput, PlusNode, PlusTransNode, Plusnodes, Plustransnodes, Wm, bm);

                    // Do system combination
                    Matrix<ElemType> CombinedOutput(deviceid);
                    CombinedOutput.Resize(decodeOutput[0]);
                    switch (combination_method)
                    {
                    case 0: // sum
                        CombinedOutput.SetValue(0.0);
                        for (int i = 0; i < m_nets.size(); i++)
                        {
                            decodeOutput[i].InplaceExp();
                            CombinedOutput += (decodeOutput[i] * combination_weights[i]);
                        }
                        CombinedOutput += (ElemType) 1e-20; // numerical stability
                        CombinedOutput.InplaceLog();
                        CombinedOutput.InplaceLogSoftmax(true); // log-softmax of a log will normalise the posteriors, to compensate for numerical inaccuracies in the sum combination
                        break;
                    case 1: // product
                        CombinedOutput.SetValue(0.0);
                        for (int i = 0; i < m_nets.size(); i++)
                        {
                            //decodeOutput[i].Print(NULL, 0, 10, 0, 1);
                            CombinedOutput += (decodeOutput[i] * combination_weights[i]);
                        }
                        CombinedOutput.InplaceLogSoftmax(true); // log-softmax of a log will normalise the posteriors
                        break;
                    default:
                        stringstream msg;
                        msg << "Unknown combination_method " << combination_method << ". combination_method must be 0 for sum or 1 for product";
                        InvalidArgument(msg.str().c_str());
                        break;
                    }

                    //CombinedOutput.Print(NULL, 0, 10, 0, 1);
                    //if (t == 10)
                    //    exit(1);

                    //sumofENandDE.Print("sum");
                    //sort log posterior and get best N labels
                    vector<pair<size_t, ElemType>> topN = getTopN(CombinedOutput, expandBeam, blankId);

                    //expand blank
                    SequenceJoint seqK = newSeq(tempSeq, deviceid);
                    ElemType newlogP = topN[vocabSize].second + tempSeq.logP;
                    seqK.logP = newlogP;
                    bool existseq = false;
                    for (auto itseq = nextSequences.begin(); itseq != nextSequences.end(); itseq++)
                    //for (Sequence seqP : keyNextSequences)  //does not work
                    {
                        //merge the score with same sequence
                        if (seqK.labelseq == itseq->labelseq)
                        {
                            existseq = true;
                            itseq->logP = decodeOutput[0].LogAdd(seqK.logP, itseq->logP);
                            //itseq->lengthwithblank = (seqK.lengthwithblank + itseq->lengthwithblank) / 2;
                            break;
                        }
                    }
                    if (!existseq)
                    {
                        nextSequences.push_back(seqK);
                    }

                    int iLabel;
                    for (iLabel = 0; iLabel < expandBeam; iLabel++)
                    {

                        seqK = newSeq(tempSeq, deviceid);
                        newlogP = topN[iLabel].second + tempSeq.logP;
                        seqK.logP = newlogP;

                        if (topN[iLabel].first != blankId)
                        {
                            extendSeq(seqK, topN[iLabel].first, newlogP);
                            CurSequences.push_back(seqK);
                        }
                    }
                    vector<pair<size_t, ElemType>>().swap(topN);
                    //delete topN;
                    deleteSeq(tempSeq);

                    if (CurSequences.size() == 0)
                        break;
                    auto ya = std::max_element(CurSequences.begin(), CurSequences.end());
                    auto yb = std::max_element(nextSequences.begin(), nextSequences.end());
                    if (nextSequences.size() > beamSize && yb->logP > ya->logP)
                        break;
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

            //nbest output
            for (size_t n = 0; n < nextSequences.size(); n++)
            {
                nextSequences[n].logP /= nextSequences[n].labelseq.size() - 1;
            }
            auto yb = std::max_element(nextSequences.begin(), nextSequences.end());
            size_t lmt = yb->length - 1;
            greedyOutput.Resize(vocabSize, lmt);
            greedyOutput.SetValue(0.0);
            for (size_t n = 0; n < lmt; n++)
            {
                greedyOutput(yb->labelseq[n + 1], n) = 1.0;
            }
            outputMatrices[decodeOutputNodeNames[0]] = (void*) (&greedyOutput);

            //the first candidates, file no ++
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
            vector<SequenceJoint>().swap(CurSequences);
            for (size_t n = 0; n < nextSequences.size(); n++)
            {
                deleteSeq(nextSequences[n]);
            }
            vector<SequenceJoint>().swap(nextSequences);
            dataWriter.SaveData(0, outputMatrices, lmt, lmt, 0);
            //break;
        }

        //decode

        // clean up
    }

private:
    int combination_method;
    std::vector<ComputationNetworkPtr> m_nets;
    std::vector<ElemType> combination_weights;
    std::vector<std::vector<wstring>> m_nodesToCache;
    int m_verbosity;
    void operator=(const JointOutputWriter&); // (not assignable)
};
} // namespace CNTK
} // namespace MSR
} // namespace Microsoft