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

    struct StreamDescriptor : StreamDescription {
        std::string m_alias; // correspoding short name used in the input data (only relevant for input streams)
        size_t m_sampleSize; // number of elements in the sample (same as m_sampleLayout->GetNumElements())
    };

    // Sequence metadata
    struct SequenceDescriptor : SequenceDescription
    {
        SequenceDescriptor() {
            m_id = 0;
            m_numberOfSamples = 0;
            m_chunkId = 0;
            m_isValid = false;
            m_fileOffset = 0;
            m_byteSize = 0;
        }
        // size_t m_numberOfSamples -- number of samples in the sequence (largest count among all inputs)
        // in case of text data this value == number of rows this sequence spans over.
        int64_t m_fileOffset; // sequence offset in the input file (in bytes)
        int64_t m_byteSize; // size in bytes
    };

    // Chunk metadata
    struct ChunkDescriptor
    {
        size_t m_index; // this index is actually redandunt 
        int64_t m_byteSize; // size in bytes
        size_t m_numSequences; // number of sequences in this chunk
        TimelineOffset m_timelineOffset; // offset into the timeline
    };

    struct Index {
        bool hasSequenceIds; // true when the input does not have the sequence id column
        std::vector<SequenceDescriptor> m_timeline;
        std::vector<ChunkDescriptor> m_chunks;
        std::map<SequenceId, TimelineOffset> m_idToOffsetMap; // maps sequence ids to offsets
        // only for sequences where id is different from the position in the timeline.
    };
}}}