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
#include "fileutil.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class SimpleOutputWriter
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

private:
    std::vector<ComputationNodeBasePtr> DetermineOutputNodes(const std::vector<std::wstring>& outputNodeNames)
    {
        std::vector<ComputationNodeBasePtr> outputNodes;

        if (outputNodeNames.size() == 0)
        {
            if (m_verbosity > 0)
                fprintf(stderr, "OutputNodeNames are not specified, using the default outputnodes.\n");
            if (m_net->OutputNodes().size() == 0)
                LogicError("There is no default output node specified in the network.");

            outputNodes = m_net->OutputNodes();
        }
        else
        {
            for (int i = 0; i < outputNodeNames.size(); i++)
                outputNodes.push_back(m_net->GetNodeFromName(outputNodeNames[i]));
        }

        return outputNodes;
    }

    std::vector<ComputationNodeBasePtr> DetermineInputNodes(const std::vector<ComputationNodeBasePtr>& outputNodes)
    {
        //use map to remove duplicated items
        std::map<ComputationNodeBasePtr, int> inputNodesMap;
        for (auto& onode : outputNodes)
        {
            for (auto& inode : m_net->InputNodes(onode))
                inputNodesMap[inode] = 1;
        }

        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto& inode : inputNodesMap)
            inputNodes.push_back(inode.first);

        return inputNodes;
    }

    std::map<std::wstring, Matrix<ElemType>*> RetrieveInputMatrices(const std::vector<ComputationNodeBasePtr>& inputNodes)
    {
        std::map<std::wstring, Matrix<ElemType>*> inputMatrices;

        for (auto& inode : inputNodes)
            inputMatrices[inode->NodeName()] = &dynamic_pointer_cast<ComputationNode<ElemType>>(inode)->Value();

        return inputMatrices;
    }

public:
    SimpleOutputWriter(ComputationNetworkPtr net, int verbosity = 0)
        : m_net(net), m_verbosity(verbosity)
    {
    }

    void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, IDataWriter<ElemType>& dataWriter, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize, bool doUnitTest = false)
    {
        std::vector<ComputationNodeBasePtr> outputNodes = DetermineOutputNodes(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes = DetermineInputNodes(outputNodes);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        std::map<std::wstring, Matrix<ElemType>*> inputMatrices = RetrieveInputMatrices(inputNodes);

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);
        if (!dataWriter.SupportMultiUtterances())
            dataReader.SetNumParallelSequences(1);
        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);

            for (int i = 0; i < outputNodes.size(); i++)
            {
                m_net->ForwardProp(outputNodes[i]);
                outputMatrices[outputNodes[i]->NodeName()] = (void*) (&dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value());
            }

            if (doUnitTest)
            {
                std::map<std::wstring, void*, nocase_compare> inputMatricesUnitTest;
                for (auto iter = inputMatrices.begin(); iter != inputMatrices.end(); iter++)
                    inputMatricesUnitTest[iter->first] = (void*) (iter->second);
                dataWriter.SaveData(0, inputMatricesUnitTest, actualMBSize, actualMBSize, 0);
            }
            else
                dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);

            totalEpochSamples += actualMBSize;

            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd(endDataSentence);
        }

        if (m_verbosity > 0)
            fprintf(stderr, "Total Samples Evaluated = %lu\n", totalEpochSamples);

        // clean up
    }

    // TODO: Remove code dup with above function
    // E.g. create a shared function that takes the actual writing operation as a lambda.
    void WriteOutput(IDataReader<ElemType>& dataReader, size_t mbSize, std::wstring outputPath, const std::vector<std::wstring>& outputNodeNames, size_t numOutputSamples = requestDataSize)
    {
        std::vector<ComputationNodeBasePtr> outputNodes = DetermineOutputNodes(outputNodeNames);
        std::vector<ComputationNodeBasePtr> inputNodes = DetermineInputNodes(outputNodes);

        // allocate memory for forward computation
        m_net->AllocateAllMatrices({}, outputNodes, nullptr);

        std::map<std::wstring, Matrix<ElemType>*> inputMatrices = RetrieveInputMatrices(inputNodes);

        msra::files::make_intermediate_dirs(outputPath);
        std::vector<ofstream*> outputStreams;
        for (int i = 0; i < outputNodes.size(); i++)
#ifdef _MSC_VER
            outputStreams.push_back(new ofstream((outputPath + L"." + outputNodes[i]->NodeName()).c_str()));
#else
            outputStreams.push_back(new ofstream(wtocharpath(outputPath + L"." + outputNodes[i]->NodeName()).c_str()));
#endif

        // evaluate with minibatches
        dataReader.StartMinibatchLoop(mbSize, 0, numOutputSamples);

        m_net->StartEvaluateMinibatchLoop(outputNodes);

        size_t totalEpochSamples = 0;
        size_t numMBsRun = 0;
        size_t tempArraySize = 0;
        ElemType* tempArray = nullptr;

        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork(dataReader, m_net, nullptr, false, false, inputMatrices, actualMBSize))
        {
            ComputationNetwork::BumpEvalTimeStamp(inputNodes);

            for (int i = 0; i < outputNodes.size(); i++)
            {
                m_net->ForwardProp(outputNodes[i]);

                Matrix<ElemType>& outputValues = dynamic_pointer_cast<ComputationNode<ElemType>>(outputNodes[i])->Value();
                ofstream& outputStream = *outputStreams[i];
                outputValues.CopyToArray(tempArray, tempArraySize);
                ElemType* pCurValue = tempArray;
                foreach_column (j, outputValues)
                {
                    foreach_row (k, outputValues)
                    {
                        outputStream << *pCurValue++ << " ";
                    }
                    outputStream << endl;
                }
            }

            totalEpochSamples += actualMBSize;

            fprintf(stderr, "Minibatch[%lu]: ActualMBSize = %lu\n", ++numMBsRun, actualMBSize);
        }

        fprintf(stderr, "Total Samples Evaluated = %lu\n", totalEpochSamples);

        // clean up
        for (int i = 0; i < outputStreams.size(); i++)
        {
            outputStreams[i]->close();
            delete outputStreams[i];
        }

        delete[] tempArray;
    }

private:
    ComputationNetworkPtr m_net;
    int m_verbosity;
    void operator=(const SimpleOutputWriter&); // (not assignable)
};
} } }
