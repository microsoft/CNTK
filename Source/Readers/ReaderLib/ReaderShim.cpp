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
    : m_factory(factory), m_deviceId(CPUDEVICE)
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
void ReaderShim<ElemType>::StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>& inputs, size_t requestedEpochSamples)
{
    return StartDistributedMinibatchLoop(mbSize, epoch, 0, 1, inputs, requestedEpochSamples);
}

template <class ElemType>
void ReaderShim<ElemType>::StartDistributedMinibatchLoop(
    size_t requestedMBSize,
    size_t epoch,
    size_t subsetNum,
    size_t numSubsets,
    const std::unordered_set<InputStreamDescription>& inputs,
    size_t requestedEpochSamples /*= requestDataSize*/)
{
    // For adaptive minibatch, make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
    {
        m_prefetchTask.wait();
    }

    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;

    std::map<std::wstring, int> inputDescriptions;
    auto device = std::find_if(inputs.begin(), inputs.end(), [](const InputStreamDescription& d) { return d.m_deviceId != CPUDEVICE; });

    m_deviceId = device != inputs.end() ? device->m_deviceId : CPUDEVICE;


    for (const auto& i : inputs)
    {
        inputDescriptions[i.m_name] = i.m_deviceId;
        m_prefetchBuffer[i.m_name] = std::make_shared<Matrix<ElemType>>(i.m_deviceId);
    }

    Matrix<ElemType>::EnableConcurrentRead(m_deviceId);

    m_reader->StartEpoch(config, inputDescriptions);
    m_endOfEpoch = false;

    // Starting the prefetch task. There is always a single async read in flight.
    // When the network requests a new minibatch, we wait for the current async to finish,
    // return the result and kick off the new one.
    m_prefetchTask = std::async(m_launchType, [this]() { return PrefetchMinibatch(); });
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

    //TODO: Set proper format on matrices?


    // Check that all matrices have the same device id.
    // If not we should inject the IMemoryProvider per stream.
    int deviceId = matrices.begin()->second.matrix->GetDeviceId();
    for (auto mx : matrices)
        assert(mx.second.matrix->GetDeviceId() == deviceId), UNUSED(deviceId);

    assert(m_prefetchTask.valid());

    // Do sanity checks:
    for (const auto& mx : matrices)
    {
        if (m_nameToStreamId.find(mx.first) == m_nameToStreamId.end())
        {
            string inputNames = EnumerateInputs(m_nameToStreamId);
            RuntimeError("Could not map input '%ls' to the reader. Reader outputs only [%s].",
                mx.first.c_str(), inputNames.c_str());
        }
    }

    auto result = m_prefetchTask.get();

    // Ok, prefetch is done.
    m_endOfEpoch = result.first;
    bool dataNotEmpty = result.second;

    if (m_endOfEpoch && !dataNotEmpty) // No data and end of epoch, simply return.
    {
        return false;
    }

    // We have some data - let's swap it.
    // Safe to swap the matrices now.
    for (auto i = matrices.begin(); i != matrices.end(); ++i)
    {
        std::swap(i->second.GetMatrix<ElemType>(), *m_prefetchBuffer[i->first]);

        // BUGBUG: This seems incorrect. (1) layouts should all be updated below, and (2) some of these layouts are the same, we are resetting them twice.
        i->second.pMBLayout->Init(1, 0);
    }

    // Let's take care of layouts now.

    // a map to generate error messages when checking layout constraints.
    map<wstring, wstring> layoutToInputMap;

    // Let's now check the layouts and throw if the same layout is beeing assigned twice.
    for (auto i = matrices.begin(); i != matrices.end(); ++i)
    {
        auto streamLayout = m_prefetchMbLayouts[i->first];
        auto& layout = i->second.pMBLayout;
        if (layout->GetNumCols() == 0) // just initialized, let's take the layout of the reader.
        {
            // layout is empty, copy layout info from the reader
            layout->CopyFrom(streamLayout, /*keepName*/ true);
            layoutToInputMap[layout->GetAxisName()] = i->first;
        }
        else if (*layout != *streamLayout) // this does a deep value-level comparison
        {
            RuntimeError("Dynamic axis layout '%ls' is shared between inputs '%ls' and '%ls', but layouts generated "
                "from the input data are incompatible on this axis. Are you using different sequence lengths? "
                "Did you consider adding a DynamicAxis() to the Input nodes?",
                layout->GetAxisName(), layoutToInputMap[layout->GetAxisName()].c_str(), i->first);
        }
    }

    // Number of logical sequences should be the same across all streams.
    // So pick up the first one.
    m_numParallelSequences = matrices.begin()->second.pMBLayout->GetNumParallelSequences();

    // It is time to issue the next prefetch.
    if (!m_endOfEpoch)
    {
        // Starting the prefetch task. There is always a single async read in flight.
        // When the network requests a new minibatch, we wait for the current async to finish,
        // return the result and kick off the new one.
        m_prefetchTask = std::async(m_launchType, [this]() { return PrefetchMinibatch(); });
    }

    return dataNotEmpty;
}

template <class ElemType>
std::pair<bool, bool> ReaderShim<ElemType>::PrefetchMinibatch()
{
    Matrix<ElemType>::SetDevice(m_prefetchBuffer.begin()->second->GetDeviceId());

    Minibatch minibatch = m_reader->ReadMinibatch();

    // If there is no data we can simply return.
    if (minibatch.m_data.empty())
    {
        return std::make_pair(minibatch.m_endOfEpoch, false);
    }

    // Ok we have some data. Let's load it to GPU.
    for (const auto& mx : m_prefetchBuffer)
    {
        size_t streamId = m_nameToStreamId[mx.first];
        const auto& stream = minibatch.m_data[streamId];
        m_prefetchMbLayouts[mx.first] = stream->m_layout;

        size_t sampleSize = m_streams[streamId]->m_sampleLayout->GetNumElements();
        FillMatrixFromStream(m_streams[streamId]->m_storageType, mx.second.get(), sampleSize, stream);
    }

    Matrix<ElemType>::SyncPendingRead(m_prefetchBuffer.begin()->second->GetDeviceId());

    return std::make_pair(minibatch.m_endOfEpoch, true);
}


template <class ElemType>
/*static*/ void ReaderShim<ElemType>::FillMatrixFromStream(StorageType type, Matrix<ElemType>* matrix, size_t numRows, const StreamMinibatchPtr& stream)
{
    size_t numCols = stream->m_layout->GetNumCols();

    if (type == StorageType::dense)
    {
        auto data = reinterpret_cast<const ElemType*>(stream->m_data);
        matrix->SetValue(numRows, numCols, matrix->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal, true);
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
        matrix->SetMatrixFromCSCFormat(columns, rows, values, nnzCount, numRows, numCols, true);
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

// TODO: We should return 0 here.
// This forbids the use of learning-rate and momentum per MB if truncation is enabled.
template <class ElemType>
size_t ReaderShim<ElemType>::GetNumParallelSequencesForFixingBPTTMode()
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
