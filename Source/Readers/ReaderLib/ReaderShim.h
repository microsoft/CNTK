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

namespace CNTK {

namespace MSR_CNTK = Microsoft::MSR::CNTK;

class Reader;
typedef std::shared_ptr<Reader> ReaderPtr;

struct EpochConfiguration;
struct ReaderConfiguration;

typedef ReaderPtr (*ReaderFactory)(const MSR_CNTK::ConfigParameters& parameters);

template <class ElemType>
class ReaderShim : public MSR_CNTK::IDataReader
{
    friend class CompositeMinibatchSource;
private:
    ReaderShim();

public:
    explicit ReaderShim(ReaderFactory factory);
    explicit ReaderShim(ReaderPtr reader);

    virtual ~ReaderShim() { }

    virtual void Init(const Microsoft::MSR::ScriptableObjects::IConfigRecord& /*config*/) override
    {
        assert(false);
    }
    virtual void Init(const MSR_CNTK::ConfigParameters& config) override;

    virtual void Destroy() override
    {
        // Make sure there are no outstanding reads.
        // Future destructor does not wait as of 2013 so probably it is not in VS2013:
        // More info can be found here http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3679.html.
        if (m_prefetchTask.valid())
        {
            // If there are some, give them time to finish.
            m_prefetchTask.wait_for(std::chrono::seconds(60));
            // TODO: if the prefetch is still valid, print a warning here!
        }

        delete this;
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch,
        const std::unordered_set<MSR_CNTK::InputStreamDescription>& inputs, size_t requestedEpochSamples = MSR_CNTK::requestDataSize) override;
    virtual void StartDistributedMinibatchLoop(size_t requestedMBSize, size_t epoch, size_t subsetNum, size_t numSubsets,
        const std::unordered_set<MSR_CNTK::InputStreamDescription>& inputs, size_t requestedEpochSamples) override;

    void StartEpoch(const EpochConfiguration& epoch, 
        const std::unordered_set<MSR_CNTK::InputStreamDescription>& inputs);

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

    virtual bool GetMinibatch(MSR_CNTK::StreamMinibatchInputs& matrices) override;

    virtual bool DataEnd() override;

    void CopyMBLayoutTo(MSR_CNTK::MBLayoutPtr) override;

    virtual size_t GetNumParallelSequencesForFixingBPTTMode() override;

    // Legacy v1 API
    virtual size_t GetCurrentSamplePosition() override;
    void SetCurrentSamplePosition(size_t currentSamplePosition);

    const std::map<std::wstring, size_t>& GetState();
    void SetState(const std::map<std::wstring, size_t>& state);
    void SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>& inputDescriptions);

    bool IsEndOfEpoch() const
    {
        return m_endOfEpoch;
    }

    bool IsEndOfSweep() const
    {
        return m_endOfSweep;
    }

private:

    void StartAsyncPrefetching();

    struct PrefetchResult
    {
        bool m_isEndOfSweep;
        bool m_isEndOfEpoch;
        bool m_isDataAvailable;
    };

    PrefetchResult PrefetchMinibatch(size_t currentDataTransferIndex);

    std::future<PrefetchResult> m_prefetchTask;
    ReaderPtr m_reader;
    ReaderFactory m_factory;
    bool m_endOfEpoch;
    bool m_endOfSweep;

    size_t m_numParallelSequences;

    std::unordered_map<std::wstring, size_t> m_nameToStreamId;

    std::vector<StreamInformation> m_streams;
    launch m_launchType;

    // Data structure required for prefetch.
    struct StreamPrefetchBuffer
    {
        std::shared_ptr<MSR_CNTK::Matrix<ElemType>> m_matrix;
        MSR_CNTK::MBLayoutPtr m_mbLayout;
        NDShape m_sampleShape;
    };

    // Intermediate buffer where the prefetch thread puts its data to.
    // When the main thread enters GetMinibatch it swaps the matrices from this buffer,
    // triggers the next prefetch and waits if memCpy is still in progress.
    std::unordered_map<std::wstring, StreamPrefetchBuffer> m_prefetchBuffers;

    // Alternating data transfer operations. In the current version these are only two - 
    // currently waiting on the main thread and the one that can be started by the prefetch thread 
    // in the meantime.
    std::vector<MSR_CNTK::DataTransfererPtr> m_dataTransferers;

    // Id to key mapping.
    std::function<std::string(size_t)> m_getKeyById;

    // Current data transfer. Flips 0 and 1.
    // Can be changed only from the main thread with no ongoing prefetch.
    size_t m_currentDataTransferIndex; 

    // Device id.
    int m_deviceId;

    std::map<std::wstring, size_t> m_currentState;
};

}
