//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// PostStat.h -- CNTK post statistics related actions
//

#pragma once
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include "DataReader.h"
#include "IDistGradAggregator.h"
#include "DistGradHeader.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class IDistGradAggregator;

// Post statistics normally called between training and evaluating, to generate the statistics results used by evaluating
// For now, the application is only with statistics mean and variance of Batch Normalization nodes after training
template <class ElemType>
class PostComputingActions
{
public:
    PostComputingActions(ComputationNetworkPtr net, const MPIWrapperPtr& mpi, bool enableDistributedMBReading = false, const int traceLevel = 0) :
        m_net(net),
        m_traceLevel(traceLevel),
        m_mpi(mpi),
        m_distGradAgg(nullptr),
        m_gradHeader(nullptr),
        m_enableDistributedMBReading(enableDistributedMBReading)
    {
    }

    // This function is used for evaluating the mean and variance of all batch normalization nodes after training. 
    // Details will link to the wiki https://docs.microsoft.com/en-us/cognitive-toolkit/Post-Batch-Normalization-Statistics
    // The reason why put it into evalute is the action take place after trainning and non-backprop processing, which makes me believe 
    // this function is like a kind of evaluate function.
    // In this function,  
    // 1. since all other weights are fix except the un-pbn nodes, I set the networkoperationMode into inferring.
    // 2. The next thing is to load the network model and data source, I follow the Evaluate function to do so, however, I delete something 
    //      seem useless, like error statistics etc.
    // 3. Finding the BN nodes in the network and put them into a vector with evaluate order (This links the nestedNode vector I got in 
    //      ControlFlowNetwork)
    // 4. From node to node in the BN vector to generate the mean and various (This links to the changes of BatchNormalizationNode 
    //      in TrainingNodes.h, since I need to make the nodes "learn" mean and variance in inferring mode)
    // 5. Consider the multi-GPU, we need to sync up the BN results between all the worker and average the value.
    void BatchNormalizationStatistics(IDataReader* dataReader, const vector<wstring>& evalNodeNames, const wstring newModelPath, 
        const size_t mbSize, const int iters = 30);

private:
    ComputationNetworkPtr m_net;
    MPIWrapperPtr m_mpi;
    bool m_enableDistributedMBReading;

    int m_traceLevel;

    std::shared_ptr<IDistGradAggregator<ElemType>> m_distGradAgg;
    std::shared_ptr<struct DistGradHeader> m_gradHeader;
};
}}}
