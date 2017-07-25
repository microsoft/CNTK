//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"
#include "ReaderUtil.h"

namespace CNTK {

// Base class for randomizers that perform randomization on a local timeline.
// The inherited class should redefine the following methods:
//   - Prefetch - should prefetch the data if needed
//   - RefillSequenceWindow - to refill the current window with next sequences.
//   - Get/SetState - for checkpointing.
//
// Given a prefetched windows of sequences, this class is responsible for picking 
// a set of sequences for the next minibatch. It also keeps track whether the end 
// of data (as specified in the confguration) is reached.
class LocalTimelineRandomizerBase : public SequenceEnumerator
{
public:
    LocalTimelineRandomizerBase(
        DataDeserializerPtr deserializer,
        bool multithreadedGetNextSequences = false,
        size_t maxNumberOfInvalidSequences= 0); // per worker

    virtual void StartEpoch(const EpochConfiguration& config) override;

    void SetConfiguration(const ReaderConfiguration& config) override
    {
        *((ReaderConfiguration*)&m_config) = config;
    }

    virtual Sequences GetNextSequences(size_t globalSampleCount, size_t localSampleCount) override;

    virtual std::vector<StreamInformation> GetStreamDescriptions() const override
    {
        return m_deserializer->StreamInfos();
    }

    std::map<std::wstring, size_t> GetState() override;
    void SetState(const std::map<std::wstring, size_t>& state) override;

protected:
    // Struct that describes a window of sequences
    // that are currently processed.
    struct SequenceWindow
    {
        SequenceWindow() : m_sequencePosition(0) {}

        std::map<ChunkIdType, ChunkPtr> m_dataChunks;
        std::vector<SequenceInfo> m_sequences;
        size_t m_sequencePosition;
    };

    // Should preserve/retrieve the state in the derived classes.
    // When this function is called, the base class guarantees that prefetch has been finished,
    // so no synchronization between this function and Prefetch is necessary in the child class.
    virtual std::map<std::wstring, size_t> GetInnerState() = 0;
    virtual void SetInnerState(const std::map<std::wstring, size_t>& state) = 0;

    // The function should fill window with new sequences.
    // When this function is called, the base class guarantees that prefetch has been finished,
    // so no synchronization between this function and Prefetch is necessary in the child class.
    virtual void RefillSequenceWindow(SequenceWindow& window) = 0;

    // Peforms prefetch on a different thread,
    // Should not change any state that cannot be recomputed.
    virtual void Prefetch() const = 0;

    // Helper functions.
    // Checks if a sequence descriptor is a special marker for the end of the sweep.
    inline static bool IsEndOfSweep(const SequenceInfo& sequence)
    {
        return sequence.m_indexInChunk == s_endOfSweep.m_indexInChunk &&
            sequence.m_chunkId == s_endOfSweep.m_chunkId &&
            sequence.m_numberOfSamples == s_endOfSweep.m_numberOfSamples;
    }

    inline size_t ValueFrom(const std::map<std::wstring, size_t>& state, const std::wstring& key)
    {
        auto it = state.find(key);
        if (it == state.end())
            RuntimeError("The required key '%ls' was not found in the checkpoint", key.c_str());
        return it->second;
    }

    ~LocalTimelineRandomizerBase()
    {
        if (m_prefetch.valid())
            m_prefetch.wait_for(std::chrono::seconds(60));
    }

    const static SequenceInfo s_endOfSweep; // Marker indicating end of the sweep.

    // Original chunk descriptions.
    const std::vector<ChunkInfo> m_originalChunkDescriptions;

    const DataDeserializerPtr m_deserializer;

    const EpochConfiguration& Config() const
    {
        return m_config;
    }

private:
    // Refills the current window of sequences.
    void Refill();

    // Gets next sequences not exceeding localSampleCount for this worker and globalSampleCount across workers.
    void GetNextSequenceDescriptions(size_t maxSampleCount, Sequences& result);

    // Moves the cursor to the sequence possibly updating the chunk.
    void MoveToNextSequence();

    // Checks if the end of the data has been reached.
    inline bool IsEndReached() const
    {
        if (m_config.m_totalEpochSizeInSweeps != g_infinity)
            return m_config.m_totalEpochSizeInSweeps == m_sweepCount;
        return m_sampleCount >= m_config.m_totalEpochSizeInSamples;
    }

    // Whether to get sequences using multiple thread.
    // Useful in case deserializer performs CPU intensive deserialization (e.g. decompression)
    const bool m_multithreadedGetNextSequences;

    // Epoch configuration
    EpochConfiguration m_config;

    // Minibatch sequences, and minibatch chunks.
    std::vector<SequenceInfo> m_sequenceBuffer;
    std::map<ChunkIdType, ChunkPtr> m_chunkBuffer;

    // Current window of sequence descriptions.
    SequenceWindow m_window;

    // Helper class for removing invalid sequences.
    SequenceCleaner m_cleaner;

    std::map<std::wstring, size_t> m_currentState;
    std::future<void> m_prefetch;

    // Number of sweeps seen from the beginning of data.
    // Incremented when the next minibatch is fetched.
    size_t m_sweepCount;

    // Number of samples seen from the beginning.
    size_t m_sampleCount;
};

}
