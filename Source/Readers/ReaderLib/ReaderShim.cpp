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

#include <sstream>
#include "Basics.h"

#define DATAREADER_EXPORTS // creating the exports here
#include "DataReader.h"
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
void ReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples)
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

    // For adaptive minibatch, make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
    {
        m_prefetchTask.wait();
    }

    m_prefetchTask = std::async(m_launchType, [this]()
    {
        return m_reader->ReadMinibatch();
    });
}

string EnumerateInputs(const map<wstring, size_t> &nameToStreamId)
{
    // TODO use boost::algorithm::join, boost::adapters::transformed, make this a generic function
    std::stringstream str;
    bool first = true;

    for (auto s : nameToStreamId)
    {
        str << (first ? "" : ", ");
        auto name = msra::strfun::utf8(s.first);
        str << '\"' << name.c_str() << '\"';
        first = false;
    }

    return str.str();
}

template <class ElemType>
bool ReaderShim<ElemType>::GetMinibatch(StreamMinibatchInputs& matrices)
{
    
    // TODO: verify that the set of matrix names is identical 
    // to the set of reader input names. Warn if it's a subset, throw
    // if it's a superset.

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
        // TODO: Use alternating pinned buffer in the packer, do not copy anything, but pack into the pinned memory.
        // Copy returned minibatch to the matrices.
        for (const auto& mx : matrices)
        {
            if (m_nameToStreamId.find(mx.first) == m_nameToStreamId.end())
            {
                string inputNames = EnumerateInputs(m_nameToStreamId);
                RuntimeError("Could not map input '%ls' to the reader. Reader outputs only [%s].", 
                    mx.first.c_str(), inputNames.c_str());
            }

            size_t streamId = m_nameToStreamId[mx.first];
            
            const auto& stream = minibatch.m_data[streamId];
            m_layout = stream->m_layout;
            size_t sampleSize = m_streams[streamId]->m_sampleLayout->GetNumElements();
            auto& matrix = matrices.GetInputMatrix<ElemType>(mx.first);
            FillMatrixFromStream(m_streams[streamId]->m_storageType, &matrix, sampleSize, stream);
        }
    }

    if (!m_endOfEpoch)
    {
        m_prefetchTask = std::async(m_launchType, [this]()
        {
            return m_reader->ReadMinibatch();
        });
    }

    return !minibatch.m_data.empty();
}

template <class ElemType>
void ReaderShim<ElemType>::FillMatrixFromStream(StorageType type, Matrix<ElemType>* matrix, size_t numRows, const StreamMinibatchPtr& stream)
{
    size_t numCols = stream->m_layout->GetNumCols();

    if (type == StorageType::dense)
    {
        auto data = reinterpret_cast<const ElemType*>(stream->m_data);
        matrix->SetValue(numRows, numCols, matrix->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal);
    }
    else if (type == StorageType::sparse_csc)
    {
        // In the sparse case the m_data layout is identical to CUDA's CSC layout
        // (see http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc).
        size_t* data = reinterpret_cast<size_t*>(stream->m_data);
        size_t nnzCount = *data;
        ElemType* values = reinterpret_cast<ElemType*>(data + 1);
        IndexType* rows = reinterpret_cast<IndexType*>(values + nnzCount);
        IndexType* columns = reinterpret_cast<IndexType*>(rows + nnzCount);
        matrix->SetMatrixFromCSCFormat(columns, rows, values, nnzCount, numRows, numCols);
    }
    else 
    {
        RuntimeError("Storage type %d is not supported.", (int)type);
    }
}

template <class ElemType>
bool ReaderShim<ElemType>::DataEnd() { return false; } // Note: Return value never used.

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
