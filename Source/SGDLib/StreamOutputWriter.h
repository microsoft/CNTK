//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

//Stream mode output writer. 
template <class ElemType>
class StreamOutputWriter
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;

public:
    StreamOutputWriter(ComputationNetworkPtr net)
        : m_net(net)
    {
    }

    void PrepareForEvaluation()
    {
        ScopedNetworkOperationMode modeGuard(m_net, NetworkOperationMode::inferring);

        m_outputNodes = m_net->OutputNodes();
        m_inputNodes = m_net->InputNodesForOutputs(m_outputNodes);

        m_inputMatrices = DataReaderHelpers::RetrieveInputMatrices(m_inputNodes);

        m_net->AllocateAllMatrices({}, m_net->OutputNodes(), nullptr);

        m_net->StartEvaluateMinibatchLoop(m_outputNodes);
    }

    void WriteOutput(IDataReader& dataReader, IDataWriter& dataWriter)
    {
        std::map<std::wstring, void*, nocase_compare> outputMatrices;

        size_t actualMBSize;
        while (DataReaderHelpers::GetMinibatchIntoNetwork<ElemType>(dataReader, m_net, nullptr, false, false, m_inputMatrices, actualMBSize, nullptr))
        {
            ComputationNetwork::BumpEvalTimeStamp(m_inputNodes);

            for (int i = 0; i < m_outputNodes.size(); i++)
            {
                m_net->ForwardProp(m_outputNodes[i]);
                outputMatrices[m_outputNodes[i]->NodeName()] = (void*)(&dynamic_pointer_cast<ComputationNode<ElemType>>(m_outputNodes[i])->Value());
            }

            dataWriter.SaveData(0, outputMatrices, actualMBSize, actualMBSize, 0);


            // call DataEnd function in dataReader to do
            // reader specific process if sentence ending is reached
            dataReader.DataEnd();
        }

    }

private:
    ComputationNetworkPtr m_net;
    std::vector<ComputationNodeBasePtr> m_outputNodes;
    std::vector<ComputationNodeBasePtr> m_inputNodes;
    StreamMinibatchInputs m_inputMatrices;
    void operator=(const StreamOutputWriter&); // (not assignable)
};

}}}
