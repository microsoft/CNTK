//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    typedef size_t SequenceId;
    typedef size_t TimelineOffset;

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
        SequenceDescriptor() 
        {
            m_id = 0;
            m_numberOfSamples = 0;
            m_chunkId = 0;
            m_isValid = false;
            m_fileOffsetBytes = 0;
            m_byteSize = 0;
        }
        // size_t m_numberOfSamples -- number of samples in the sequence (largest count among all inputs)
        // in case of text data this value == number of rows this sequence spans over.
        int64_t m_fileOffsetBytes; // sequence offset in the input file (in bytes)
        size_t m_byteSize; // size in bytes
    };

    // Chunk metadata, similar to the sequence descriptor above, 
    // but used to facilitate indexing and retrieval of blobs of input data of
    // some user-specified size.
    struct ChunkDescriptor
    {
        size_t m_id; 
        size_t m_byteSize; // size in bytes
        size_t m_numSequences; // number of sequences in this chunk
        TimelineOffset m_timelineOffset; // offset into the timeline -- timeline index of
                                         // the very first sequence from this chunk.
    };

    // The index comprises two timelines with different granularities. One is 
    // is a collection of sequences, the other -- of chunks. 
    // TODO: needs to be refactored to support partial timeline.
    struct Index 
    {
        Index(bool hasSequenceIds, 
            std::vector<SequenceDescriptor> timeline,
            std::vector<ChunkDescriptor> chunks) 
            : m_hasSequenceIds(hasSequenceIds), m_timeline(timeline), m_chunks(chunks)
        {
        }

        bool m_hasSequenceIds; // true when input contains sequence id column
        std::vector<SequenceDescriptor> m_timeline;
        std::vector<ChunkDescriptor> m_chunks;
    };

    typedef std::shared_ptr<Index> IndexPtr;
}}}