//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Stream (input) metadata. This text-reader specific descriptor adds two 
    // additional fields: stream alias (name prefix in each sample) and expected
    // sample dimension.
    struct StreamDescriptor : StreamDescription 
    {
        std::string m_alias; // sample name prefix used in the input data
        size_t m_sampleDimension; // expected number of elements in a sample
                                  // (can be omitted for sparse input)
    };

    // Sequence metadata. This text-reader specific descriptor adds two additional
    // fields: file offset and size in bytes. Both are required to efficiently 
    // locate and retrieve a sequence from file, given a sequence descriptor.
    struct SequenceDescriptor : SequenceDescription
    {
        SequenceDescriptor() : SequenceDescription({}), m_fileOffsetBytes(0),
            m_byteSize(0)
        {
        }
        // size_t m_numberOfSamples -- number of samples in the sequence (largest count among all inputs)
        // in case of text data this value == number of rows this sequence spans over.
        int64_t m_fileOffsetBytes; // sequence offset in the input file (in bytes)
        size_t m_byteSize; // size in bytes
    };

    // Chunk metadata, similar to the sequence descriptor above, 
    // but used to facilitate indexing and retrieval of blobs of input data of
    // some user-specified size.
    struct ChunkDescriptor : ChunkDescription
    { 
        ChunkDescriptor() : ChunkDescription({}), m_byteSize(0) {}
        // TODO: if we don't want to keep the whole index 
        // (metadata for all sequences in memory), we should not
        // leave this empty when building a chunk index, and only
        // fill it out when the chunk needs to be loaded 
        // (the indexer will have to do a second pass for this chunk).
        std::vector<SequenceDescriptor> m_sequences;
        
        size_t m_byteSize; // size in bytes
    };

    typedef shared_ptr<ChunkDescriptor> ChunkDescriptorPtr;

    // A collection of chunk descriptors, each containing
    // a collection of sequence descriptors for the corresponding
    // chunk of the input data. 
    typedef std::vector<ChunkDescriptor> Index;
}}}
