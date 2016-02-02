//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReaderShim.cpp: implementation for shim wrapping the new reader interface
//

#define _CRT_SECURE_NO_WARNINGS

#ifdef _WIN32
#include <objbase.h>
#endif
#include "Basics.h"

#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
//#include "commandArgUtil.h"
#include "ReaderShim.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
ReaderShim<ElemType>::ReaderShim(ReaderFactory factory)
    : m_layout(make_shared<MBLayout>()), m_factory(factory)
{
}

template <class ElemType>
void ReaderShim<ElemType>::Init(const ConfigParameters& config)
{
    intargvector numberOfuttsPerMinibatchForAllEpochs =
        config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int> { 1 })));

    bool prefetch = config(L"prefetch", true);
    // if prefetch - launching asynchronously,
    // otherwise deferring - synchronous execution during .get() call
    m_launchType = prefetch ? launch::async : launch::deferred;

    auto numSeqsPerMBForAllEpochs = numberOfuttsPerMinibatchForAllEpochs;
    m_layout->Init(numSeqsPerMBForAllEpochs[0], 0);

    m_reader = m_factory(config);
    m_streams = m_reader->GetStreamDescriptions();
    for (auto i : m_streams)
    {
        m_nameToStreamId.insert(std::make_pair(i->m_name, i->m_id));
    }
}

template <class ElemType>
void ReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize)
{
    return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, requestedEpochSamples);
}

template <class ElemType>
void ReaderShim<ElemType>::StartDistributedMinibatchLoop(
    size_t requestedMBSize,
    size_t epoch,
    size_t subsetNum,
    size_t numSubsets,
    size_t requestedEpochSamples /*= requestDataSize*/)
{
    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;

    m_reader->StartEpoch(config);
    m_endOfEpoch = false;

    m_prefetchTask = std::async(m_launchType, [this]()
    {
        return m_reader->ReadMinibatch();
    });
}

template <class ElemType>
bool ReaderShim<ElemType>::GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices)
{
    if (m_endOfEpoch)
    {
        return false;
    }

    // Check that all matrices have the same device id.
    // If not we should inject the IMemoryProvider per stream.
    int deviceId = matrices.begin()->second->GetDeviceId();
    for (auto mx : matrices)
    {
        if (mx.second->GetDeviceId() != deviceId)
        {
            assert(false);
        }
    }

    assert(m_prefetchTask.valid());

    Minibatch minibatch = m_prefetchTask.get();
    if (minibatch.m_endOfEpoch)
    {
        m_endOfEpoch = true;
        if (minibatch.m_data.empty())
        {
            return false;
        }
    }

    if (!minibatch.m_data.empty())
    {
        // Copy returned minibatch to the matrices.
        for (const auto& mx : matrices)
        {
            assert(m_nameToStreamId.find(mx.first) != m_nameToStreamId.end());
            size_t streamId = m_nameToStreamId[mx.first];

            const auto& stream = minibatch.m_data[streamId];
            m_layout = stream->m_layout;

            size_t columnNumber = m_layout->GetNumCols();
            size_t rowNumber = m_streams[streamId]->m_sampleLayout->GetNumElements();

            auto data = reinterpret_cast<const ElemType*>(stream->m_data);
            mx.second->SetValue(rowNumber, columnNumber, mx.second->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
        }
    }

    m_prefetchTask = std::async(m_launchType, [this]()
    {
        return m_reader->ReadMinibatch();
    });

    return !minibatch.m_data.empty();
}

template <class ElemType>
bool ReaderShim<ElemType>::DataEnd(EndDataType /*endDataType*/)
{
    return false;
}

template <class ElemType>
void ReaderShim<ElemType>::CopyMBLayoutTo(MBLayoutPtr layout)
{
    layout->CopyFrom(m_layout);
}

template <class ElemType>
size_t ReaderShim<ElemType>::GetNumParallelSequences()
{
    return m_layout->GetNumParallelSequences();
}

template class ReaderShim<float>;
template class ReaderShim<double>;
} } }
