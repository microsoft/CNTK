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
    : m_factory(factory)
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

    m_numParallelSequences = numberOfuttsPerMinibatchForAllEpochs[0];

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
    int deviceId = matrices.begin()->second.matrix->GetDeviceId();
    for (auto mx : matrices)
        assert(mx.second.matrix->GetDeviceId() == deviceId), UNUSED(deviceId);

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

    // Reset stale mb layouts.
    // BUGBUG: This seems incorrect. (1) layouts should all be updated below, and (2) some of these layouts are the same, we are resetting them twice.
    for (const auto& iter : matrices)
    {
        iter.second.pMBLayout->Init(1, 0);
    }

    // a map to generate error messages when checking layout constraints. 
    map<wstring, wstring> layoutToInputMap;
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

            m_numParallelSequences = stream->m_layout->GetNumParallelSequences();

            // This assert no longer holds - different inputs have different sequence lengths, resulting in different number 
            // of parallel samples.
            // assert(m_numParallelSequences == minibatch.m_data.front()->m_layout->GetNumParallelSequences());

            auto& layout = mx.second.pMBLayout;

            if (layout->GetNumCols() == 0)
            {
                // layout is empty, copy layout info from the reader
                layout->CopyFrom(stream->m_layout, /*keepName*/ true);
                layoutToInputMap[layout->GetAxisName()] = mx.first;
            }
            else if (*layout != *stream->m_layout) // this does a deep value-level comparison
            {
                RuntimeError("Dynamic axis layout '%ls' is shared between inputs '%ls' and '%ls', but layouts generated "
                    "from the input data are incompatible on this axis. Are you using different sequence lengths? "
                    "Did you consider adding a DynamicAxis() to the Input nodes?",
                    layout->GetAxisName(), layoutToInputMap[layout->GetAxisName()].c_str(), mx.first.c_str());
            }

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
    // This method is inherited from IDataReader and should be removed in the near future.
    NOT_IMPLEMENTED;
}

template <class ElemType>
size_t ReaderShim<ElemType>::GetNumParallelSequences()
{
    // BUGBUG This is a property of the stream, of which this reader might produce several, with different nr. of
    // parallel sequences. Thus this property doesn't make sense anymore.
    // This method is called by 
    // * DataReaderHelpers::GetNumSubminibatchesNeeded to estimate mb size
    // * ComputationNetwork::SetBatchNormalizationTimeConstants to compute learning rate per sample
    // * ComputationNetwork::SetBatchNormalizationTimeConstants to compute actual mb size and momentum per sample
    // * SGD::AdaptiveMinibatchSizing  to compute learning rate per sample
    return m_numParallelSequences;
}

template class ReaderShim<float>;
template class ReaderShim<double>;
} } }
