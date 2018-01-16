//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#include <functional>
#include "Sequences.h"
#include "ReaderConstants.h"
#include "DataDeserializer.h"

namespace CNTK {

namespace MSR_CNTK = Microsoft::MSR::CNTK;

typedef GPUSPARSE_INDEX_TYPE IndexType;

using MSR_CNTK::MBLayout;
using MSR_CNTK::MBLayoutPtr;


// Configuration for the current epoch.
// Each time the epoch is started CNTK should provide the configuration to the reader using StartEpoch method
// and the below structure.
struct ReaderConfiguration
{
    ReaderConfiguration()
        : m_numberOfWorkers(0), m_workerRank(0), m_minibatchSizeInSamples(0), m_truncationSize(0)
    {}

    size_t m_numberOfWorkers;               // Number of the Open MPI workers for the current epoch
    size_t m_workerRank;                    // Rank of the Open MPI worker, worker rank has to be less than the number of workers
    size_t m_minibatchSizeInSamples;        // Maximum minibatch size for the epoch in samples
    size_t m_truncationSize;                // Truncation size in samples for truncated BPTT mode.

    // This flag indicates whether the minibatches are allowed to overlap the boundary
    // between sweeps (in which case, they can contain data from different sweeps) or
    // if they need to be trimmed at the sweep end.
    bool m_allowMinibatchesToCrossSweepBoundaries{ false };
};

// TODO: Should be deprecated.
struct EpochConfiguration : public ReaderConfiguration
{
    size_t m_totalEpochSizeInSamples;       // Total size of the epoch in samples
    size_t m_totalEpochSizeInSweeps {g_infinity}; // Total size of the epoch in sweeps (default = no limit).
    size_t m_epochIndex;                    // Current epoch index [0 .. max number of epochs)
};

typedef size_t StreamId;

// Represent a minibatch date for a single stream formatted in according to the minibatch layout.
// This data is returned per stream as a part of Minibatch from the ReadMinibatch function.
// All raw non owned pointers are valid till the next call to the ReadMinibatch function.
struct StreamMinibatch
{
    void* m_data;         // Contiguous array of data. Can be encoded in dense or sparse formats depending on the stream description.
                          // The size is (the number of rows * number of columns in the layout) * by the element size of the stream (float/double/etc.).
    MBLayoutPtr m_layout; // Layout of the data
    NDShape m_sampleShape;
};
typedef std::shared_ptr<StreamMinibatch> StreamMinibatchPtr;

// Represents a single minibatch, that contains information about all streams.
struct Minibatch
{
    // Indicates that this minibatch is either adjacent to the data sweep boundary 
    // (-----<minibatch>|---) or crosses the boundary (-----<mini|batch>---).
    bool m_endOfSweep;

    // Indicates that the end of epoch has been reached.
    // It is set to true for the last minibatch, there still
    // can be data in m_data field even if this flag is set.
    bool m_endOfEpoch;

    // Minibatch data
    std::vector<StreamMinibatchPtr> m_data;

    // A function that maps a sequence id from minibatch layout
    // to the string representation of the sequence key.
    std::function<std::string(const size_t)> m_getKeyById;

    Minibatch(bool endOfSweep = false, bool endOfEpoch = false)
        : m_endOfSweep(endOfSweep), m_endOfEpoch(endOfEpoch)
    {
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Main Reader interface. The border interface between the CNTK and reader libraries.
//////////////////////////////////////////////////////////////////////////////////////////////////
class Reader
{
public:
    // Configuration
    // Starts a new epoch with the provided configuration
    // TODO: should be deprecated, SetConfiguration should be used instead.
    virtual void StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& inputDescriptions) = 0;

    // Sets a new configuration for the reader.
    virtual void SetConfiguration(const ReaderConfiguration& config, const std::map<std::wstring, int>& inputDescriptions) = 0;

    // Describes the streams this reader produces.
    virtual std::vector<StreamInformation> GetStreamDescriptions() = 0;

    // Reads a minibatch that contains data across all streams.
    virtual Minibatch ReadMinibatch() = 0;

    // Returns current position in the global timeline. The returned value is in samples.
    // Currently in case of sequence to sequence training, 
    // the logical sequence size in samples = max(constitutuing sequences among all streams)
    // This will probably change in the future.
    virtual std::map<std::wstring, size_t> GetState() = 0;

    // Set current global position
    virtual void SetState(const std::map<std::wstring, size_t>& state) = 0;

    virtual ~Reader() {};
};

typedef std::shared_ptr<Reader> ReaderPtr;
}
