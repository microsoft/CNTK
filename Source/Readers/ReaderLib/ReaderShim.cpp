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
#include "DataTransferer.h"
#include "PerformanceProfiler.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
ReaderShim<ElemType>::ReaderShim() :
    m_deviceId(CPUDEVICE),
    m_dataTransferers(2, DataTransfererPtr()),
    m_currentDataTransferIndex(0),
    m_endOfEpoch(false),
    m_endOfSweep(false),
    m_currentSamplePosition(0),
    m_reader(nullptr),
    m_factory(nullptr)
{
}

template <class ElemType>
ReaderShim<ElemType>::ReaderShim(ReaderFactory factory) :
    ReaderShim()
{
    m_factory = factory;
}

template <class ElemType>
ReaderShim<ElemType>::ReaderShim(ReaderPtr reader) :
    ReaderShim()
{
    m_reader = reader;
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

    if (!m_reader)
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
    EpochConfiguration config;
    config.m_workerRank = subsetNum;
    config.m_numberOfWorkers = numSubsets;
    config.m_minibatchSizeInSamples = requestedMBSize;
    config.m_totalEpochSizeInSamples = requestedEpochSamples;
    config.m_epochIndex = epoch;

    StartEpoch(config, inputs);
}

template <class ElemType>
void ReaderShim<ElemType>::SetCurrentSamplePosition(size_t currentSamplePosition)
{
    // Make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
        m_prefetchTask.wait();

    // Let's check that there is no outstanding copies.
    // Wait on all events if there are any pending copy operations in flight.
    if (m_dataTransferers[m_currentDataTransferIndex])
        m_dataTransferers[m_currentDataTransferIndex]->WaitForCopyCPUToGPU();

    // Set current position.
    m_reader->SetCurrentSamplePosition(currentSamplePosition);
    m_endOfEpoch = false;
    m_currentSamplePosition = m_reader->GetCurrentSamplePosition();
}

template <class ElemType>
void ReaderShim<ElemType>::SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>& inputDescriptions)
{
    // Make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
        m_prefetchTask.wait();

    // Let's check that there is no outstanding copies.
    // Wait on all events if there are any pending copy operations in flight.
    if (m_dataTransferers[m_currentDataTransferIndex])
        m_dataTransferers[m_currentDataTransferIndex]->WaitForCopyCPUToGPU();

    m_reader->SetConfiguration(config, inputDescriptions);
    m_reader->SetCurrentSamplePosition(m_currentSamplePosition);

    // Start prefetch.
    auto localCurrentDataTransferIndex = m_currentDataTransferIndex;
    // Starting the prefetch task. There is always a single async read in flight.
    // When the network requests a new minibatch, we wait for the current async to finish, swap the buffers
    // and kick off the new prefetch.
    m_prefetchTask = std::async(m_launchType,
        [this, localCurrentDataTransferIndex]()
    {
        return PrefetchMinibatch(localCurrentDataTransferIndex);
    });
}

template <class ElemType>
void ReaderShim<ElemType>::StartEpoch(const EpochConfiguration& config, const std::unordered_set<InputStreamDescription>& inputs)
{
    // For adaptive minibatch, make sure there are no outstanding reads.
    if (m_prefetchTask.valid())
    {
        m_prefetchTask.wait();
    }

    // Let's check that there is no outstanding copies.
    // Wait on all events if there are any pending copy operations in flight.
    if (m_dataTransferers[m_currentDataTransferIndex])
        m_dataTransferers[m_currentDataTransferIndex]->WaitForCopyCPUToGPU();

    // Now we can be sure, no prefetch thread is running and there are no outstanding memcopies.
    // Let's check that requested devices are ok and see whether we need to change our data transferers.
    auto device = std::find_if(inputs.begin(), inputs.end(),
        [](const InputStreamDescription& d) { return d.GetDeviceId() != CPUDEVICE; });
    auto deviceId = device != inputs.end() ? device->GetDeviceId() : CPUDEVICE;

    // Check that all devices either the same as m_deviceId or CPU.
    auto secondDevice = std::find_if(inputs.begin(), inputs.end(), 
        [deviceId](const InputStreamDescription& d) { return d.GetDeviceId() != CPUDEVICE && d.GetDeviceId() != deviceId; });
    if (secondDevice != inputs.end())
    {
        LogicError("Readers do not support running on several GPUs in the same process, at least two devices found '%d', '%d'", deviceId, secondDevice->GetDeviceId());
    }

    if (m_deviceId != deviceId)
    {
        // Device changed. Let's change the data transferers.
        m_deviceId = deviceId;
        m_dataTransferers.clear();
        // We need two in order to support two operations in flight.
        m_dataTransferers.push_back(m_deviceId == CPUDEVICE ? nullptr : CreatePrefetchDataTransferer(m_deviceId));
        m_dataTransferers.push_back(m_deviceId == CPUDEVICE ? nullptr : CreatePrefetchDataTransferer(m_deviceId));
    }

    // Let's create the buffers for the prefetch thread.
    std::map<std::wstring, int> inputDescriptions;
    for (const auto& i : inputs)
    {
        inputDescriptions[i.GetStreamName()] = i.GetDeviceId();
        // Creating buffers with the same properties the network expects.
        m_prefetchBuffers[i.GetStreamName()] = StreamPrefetchBuffer
        {
            std::make_shared<Matrix<ElemType>>(0, 0, i.GetDeviceId(), i.GetMatrixType(), i.GetMatrixFormat()),
            std::make_shared<MBLayout>()
        };
    }

    m_endOfEpoch = false;
    m_reader->StartEpoch(config, inputDescriptions);
    m_currentSamplePosition = m_reader->GetCurrentSamplePosition();

    auto localCurrentDataTransferIndex = m_currentDataTransferIndex;
    // Starting the prefetch task. There is always a single async read in flight.
    // When the network requests a new minibatch, we wait for the current async to finish, swap the buffers
    // and kick off the new prefetch.
    m_prefetchTask = std::async(m_launchType,
    [this, localCurrentDataTransferIndex]()
    {
        return PrefetchMinibatch(localCurrentDataTransferIndex);
    });
}

string EnumerateInputs(const unordered_map<wstring, size_t>& nameToStreamId)
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
    // If not we should inject the MemoryProvider per stream.
    int deviceId = matrices.begin()->second.matrix->GetDeviceId();
    for (auto mx : matrices)
        assert(mx.second.matrix->GetDeviceId() == deviceId), UNUSED(deviceId);

    // Do sanity checks: requested streams should be exposed by low level deserializers.
    for (const auto& mx : matrices)
    {
        if (m_nameToStreamId.find(mx.first) == m_nameToStreamId.end())
        {
            string inputNames = EnumerateInputs(m_nameToStreamId);
            RuntimeError("Could not map input '%ls' to the reader. Reader outputs only [%s].",
                mx.first.c_str(), inputNames.c_str());
        }
    }

    // Make sure the prefetch has finished.
    assert(m_prefetchTask.valid());
    auto result = m_prefetchTask.get();

    // Ok, prefetch is done.

    // Let's update our sample position.
    m_currentSamplePosition = m_reader->GetCurrentSamplePosition();

    m_endOfEpoch = result.m_isEndOfEpoch;
    m_endOfSweep = result.m_isEndOfSweep;
    if (m_endOfEpoch && !result.m_isDataAvailable)
    {
        // No data and end of epoch, simply return.
        return false;
    }

    // Remember current data transfer, async memcpy for it already started on the prefetch thread.
    auto currentDataTransferIndex = m_currentDataTransferIndex;

    // Let's update the current data transferer.
    m_currentDataTransferIndex = (m_currentDataTransferIndex + 1) % 2;

    // Record an event that prefetch can wait on to ensure that prior compute has finished.
    if (m_dataTransferers[m_currentDataTransferIndex])
        m_dataTransferers[m_currentDataTransferIndex]->RecordComputeStreamSyncPoint();

    // We have some data - let's swap the matrices.
    // We cannot simply change pointers because it seems they are remembered deeper in the network.
    for (auto i = matrices.begin(); i != matrices.end(); ++i)
    {
        std::swap(i->second.GetMatrix<ElemType>(), *m_prefetchBuffers[i->first].m_matrix);

        // Resetting layouts.
        i->second.pMBLayout->Init(1, 0);
    }

    // a map to generate error messages when checking layout constraints.
    map<wstring, wstring> layoutToInputMap;

    // Let's now check the layouts and throw if the same layout is being assigned twice.
    for (auto i = matrices.begin(); i != matrices.end(); ++i)
    {
        auto streamLayout = m_prefetchBuffers[i->first].m_mbLayout;
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
                layout->GetAxisName(), layoutToInputMap[layout->GetAxisName()].c_str(), i->first.c_str());
        }
    }

    // Number of logical sequences should be the same across all streams.
    // So pick up the first one.
    m_numParallelSequences = matrices.begin()->second.pMBLayout->GetNumParallelSequences();

    // It is time to issue the next prefetch.
    if (!m_endOfEpoch)
    {
        // Starting the prefetch task. There is always a single async read in flight.
        // When the network requests a new minibatch, we wait for the current async to finish, swap the buffers
        // and kick off the new prefetch.
        auto localCurrentDataTransferIndex = m_currentDataTransferIndex;
        m_prefetchTask = std::async(m_launchType, [this, localCurrentDataTransferIndex]() { return PrefetchMinibatch(localCurrentDataTransferIndex); });
    }

    // Let's wait till the previous memcopy has finished.
    if (m_dataTransferers[currentDataTransferIndex])
        m_dataTransferers[currentDataTransferIndex]->WaitForCopyCPUToGPU();

    return result.m_isDataAvailable;
}

template <class ElemType>
typename ReaderShim<ElemType>::PrefetchResult ReaderShim<ElemType>::PrefetchMinibatch(size_t currentDataTransferIndex)
{
    PROFILE_SCOPE(profilerEvtPrefetchMinibatch);

    // Resetting layouts.
    for (auto& mx : m_prefetchBuffers)
        mx.second.m_mbLayout = std::make_shared<MBLayout>();

    Minibatch minibatch = m_reader->ReadMinibatch();

    // If there is no data we can simply return.
    if (minibatch.m_data.empty())
        return PrefetchResult{ minibatch.m_endOfSweep, minibatch.m_endOfEpoch, false };

    // Ok we have some data. Let's load it to GPU.
    // But before we need to make sure that corresponding compute has already finished from the last iteration.

    // We need to make sure that the compute for the current transfer is finished before we start prefetch.
    if (m_dataTransferers[currentDataTransferIndex])
        m_dataTransferers[currentDataTransferIndex]->WaitForSyncPointOnAssignStreamAsync();

    for (auto& mx : m_prefetchBuffers)
    {
        size_t streamId = m_nameToStreamId[mx.first];
        const auto& stream = minibatch.m_data[streamId];
        mx.second.m_mbLayout = stream->m_layout;

        size_t sampleSize = m_streams[streamId]->m_sampleLayout->GetNumElements();
        FillMatrixFromStream(m_streams[streamId]->m_storageType, mx.second.m_matrix.get(), sampleSize, stream, m_dataTransferers[currentDataTransferIndex].get());
    }

    // Let's record that we started the copy, so that the main thread can wait afterwards.
    if (m_dataTransferers[currentDataTransferIndex])
        m_dataTransferers[currentDataTransferIndex]->RecordCPUToGPUCopy();

    return PrefetchResult{ minibatch.m_endOfSweep, minibatch.m_endOfEpoch, true };
}


template <class ElemType>
/*static*/ void ReaderShim<ElemType>::FillMatrixFromStream(StorageType type, Matrix<ElemType>* matrix, size_t numRows, const StreamMinibatchPtr& stream, DataTransferer* transferer)
{
    size_t numCols = stream->m_layout->GetNumCols();

    if (type == StorageType::dense)
    {
        auto data = reinterpret_cast<const ElemType*>(stream->m_data);
        matrix->SetValue(numRows, numCols, matrix->GetDeviceId(), const_cast<ElemType*>(data), matrixFlagNormal, transferer);
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
        matrix->SetMatrixFromCSCFormat(columns, rows, values, nnzCount, numRows, numCols, transferer);
    }
    else
        RuntimeError("Storage type %d is not supported.", (int)type);
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

template <class ElemType>
size_t ReaderShim<ElemType>::GetCurrentSamplePosition()
{
    return m_currentSamplePosition;
}

template class ReaderShim<float>;
template class ReaderShim<double>;
} } }
