//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <memory>
#include "Sequences.h"
#include "TensorShape.h"

namespace Microsoft { namespace MSR { namespace CNTK {

typedef GPUSPARSE_INDEX_TYPE IndexType;

typedef std::shared_ptr<TensorShape> TensorShapePtr;

struct MBLayout;
typedef std::shared_ptr<MBLayout> MBLayoutPtr;

// Configuration for the current epoch.
// Each time the epoch is started CNTK should provide the configuration to the reader using StartEpoch method
// and the below structure.
struct EpochConfiguration
{
    size_t m_numberOfWorkers;               // Number of the Open MPI workers for the current epoch
    size_t m_workerRank;                    // Rank of the Open MPI worker, worker rank has to be less than the number of workers
    size_t m_minibatchSizeInSamples;        // Maximum minibatch size for the epoch in samples
    size_t m_totalEpochSizeInSamples;       // Total size of the epoch in samples
    size_t m_epochIndex;                    // Current epoch index [0 .. max number of epochs)
};

// Supported primitive element types, will be extended in the future.
enum class ElementType
{
    tfloat,  // single precision
    tdouble, // double precision
    tatom    // sizeof(atom) == 1 constitute of blobs -> sequences of atoms (i.e. used for lattices, hmmm, etc.)
};

// Supported storage types, will be extended in the future.
enum class StorageType
{
    dense,
    sparse_csc,
};

typedef size_t StreamId;

// This class describes a particular stream: its name, element type, storage, etc.
struct StreamDescription
{
    std::wstring m_name;           // Unique name of the stream
    StreamId m_id;                 // Unique identifier of the stream
    StorageType m_storageType;     // Storage type of the stream
    ElementType m_elementType;     // Element type of the stream
    TensorShapePtr m_sampleLayout; // Layout of the sample for the stream
                                   // If not specified - can be specified per sequence
};
typedef std::shared_ptr<StreamDescription> StreamDescriptionPtr;

// Represent a minibatch date for a single stream formatted in according to the minibatch layout.
// This data is returned per stream as a part of Minibatch from the ReadMinibatch function.
// All raw non owned pointers are valid till the next call to the ReadMinibatch function.
struct StreamMinibatch
{
    void* m_data;         // Contiguous array of data. Can be encoded in dense or sparse formats depending on the stream description.
                          // The size is (the number of rows * number of columns in the layout) * by the element size of the stream (float/double/etc.).
    MBLayoutPtr m_layout; // Layout of the data
};
typedef std::shared_ptr<StreamMinibatch> StreamMinibatchPtr;

// Represents a single minibatch, that contains information about all streams.
struct Minibatch
{
    // Indicates that the end of epoch has been reached.
    // It is set to true for the last minibatch, there still
    // can be data in m_data field even if this flag is set.
    bool m_endOfEpoch;

    // Minibatch data
    std::vector<StreamMinibatchPtr> m_data;

    Minibatch() : m_endOfEpoch(false)
    {
    }

    Minibatch(bool endOfEpoch) : m_endOfEpoch(endOfEpoch)
    {
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Main Reader interface. The border interface between the CNTK and reader libraries.
// TODO: Expect to change in a little bit: stream matrices provided by the network as input.
//////////////////////////////////////////////////////////////////////////////////////////////////
class Reader
{
public:
    // Describes the streams this reader produces.
    virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() = 0;

    // Starts a new epoch with the provided configuration
    virtual void StartEpoch(const EpochConfiguration& config) = 0;

    // Reads a minibatch that contains data across all streams.
    virtual Minibatch ReadMinibatch() = 0;

    virtual ~Reader() {};
};

typedef std::shared_ptr<Reader> ReaderPtr;
}}}
