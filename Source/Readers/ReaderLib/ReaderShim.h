//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ReaderShim.h: Currently we are preserving the old interface in SGD. So this shim exposes the old interface and calls into the 
// reader implemented with the new interfaces (reader/packer/transforms/serializers)
//

#pragma once

#include <unordered_map>
#include <string>
#include <future>
#include "DataReader.h"
#include "Reader.h"

namespace CNTK
{
    class CompositeMinibatchSource;
}

namespace Microsoft { namespace MSR { namespace CNTK {

typedef ReaderPtr (*ReaderFactory)(const ConfigParameters& parameters);

template <class ElemType>
class ReaderShim : public IDataReader
{
    friend class ::CNTK::CompositeMinibatchSource;
public:
    explicit ReaderShim(ReaderFactory factory);
    explicit ReaderShim(ReaderPtr reader);

    virtual ~ReaderShim() { }

    virtual void Init(const ScriptableObjects::IConfigRecord& /*config*/) override
    {
        assert(false);
    }
    virtual void Init(const ConfigParameters& config) override;

    virtual void Destroy() override
    {
        // Make sure there are no outstanding reads.
        // Future destructor does not wait as of 2013 so probably it is not in VS2013:
        // More info can be found here http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3679.html.
        if (m_prefetchTask.valid())
        {
            // If there are some, give them time to finish.
            m_prefetchTask.wait_for(std::chrono::seconds(5));
        }

        delete this;
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>& inputs, size_t requestedEpochSamples = requestDataSize) override;
    virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets, const std::unordered_set<InputStreamDescription>& inputs, size_t requestedEpochSamples) override;

    void StartEpoch(const EpochConfiguration& epoch, const std::unordered_set<InputStreamDescription>& inputs);

    virtual void StartMinibatchLoop(size_t, size_t, size_t) override
    {
        LogicError("Legacy StartMinibatchLoop is not implemented.");
    }

    virtual void StartDistributedMinibatchLoop(size_t, size_t, size_t, size_t, size_t) override
    {
        LogicError("Legacy StartDistributedMinibatchLoop is not implemented.");
    }

    virtual bool SupportsDistributedMBRead() const override
    {
        return true;
    }

    virtual bool IsLegacyReader() const override
    {
        return false;
    }

    virtual bool GetMinibatch(StreamMinibatchInputs& matrices) override;

    virtual bool DataEnd() override;

    void CopyMBLayoutTo(MBLayoutPtr) override;

    virtual size_t GetNumParallelSequencesForFixingBPTTMode() override;

    virtual size_t GetCurrentSamplePosition() override;

    bool IsEndOfEpoch() const
    {
        return m_endOfEpoch;
    }

private:
    struct PrefetchResult
    {
        bool m_isEndOfEpoch;
        bool m_isDataAvailable;
    };

    PrefetchResult PrefetchMinibatch(size_t currentDataTransferIndex);

    std::future<PrefetchResult> m_prefetchTask;
    ReaderPtr m_reader;
    ReaderFactory m_factory;
    bool m_endOfEpoch;

    size_t m_numParallelSequences;

    std::unordered_map<std::wstring, size_t> m_nameToStreamId;
    std::vector<StreamDescriptionPtr> m_streams;
    launch m_launchType;

    // Data structure required for prefetch.
    struct StreamPrefetchBuffer
    {
        std::shared_ptr<Matrix<ElemType>> m_matrix;
        MBLayoutPtr m_mbLayout;
    };

    // Intermediate buffer where the prefetch thread puts its data to.
    // When the main thread enters GetMinibatch it swaps the matrices from this buffer,
    // triggers the next prefetch and waits if memCpy is still in progress.
    std::unordered_map<std::wstring, StreamPrefetchBuffer> m_prefetchBuffers;

    // Alternating data transfer operations. In the current version these are only two - 
    // currently waiting on the main thread and the one that can be started by the prefetch thread 
    // in the meantime.
    std::vector<DataTransfererPtr> m_dataTransferers;

    // Current data transfer. Flips 0 and 1.
    // Can be changed only from the main thread with no ongoing prefetch.
    size_t m_currentDataTransferIndex; 

    // Device id.
    int m_deviceId;

    // Current sample position of the reader on the global timeline.
    // We have to remember the value locally before starting prefetch.
    // The value is updated only from the main thread (in StartEpoch/GetMinibatch)
    size_t m_currentSamplePosition;

    static void FillMatrixFromStream(
        StorageType type,
        Matrix<ElemType>* matrix,
        size_t numRows,
        const StreamMinibatchPtr& stream,
        DataTransferer* transferer);
};

}}}
