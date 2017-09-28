//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS

#include "DataDeserializerBase.h"
#include "Index.h"

namespace CNTK {

    bool DataDeserializerBase::GetSequenceInfoByKey(const Index& index, const SequenceKey& key, SequenceInfo& r)
    {
        if (m_primary)
            LogicError("Matching by sequence key is not supported for primary deserilalizer.");

        auto sequenceLocation = index.GetSequenceByKey(key.m_sequence);
        if (!std::get<0>(sequenceLocation))
            return false;

        r.m_chunkId = std::get<1>(sequenceLocation);
        r.m_indexInChunk = std::get<2>(sequenceLocation);
        r.m_key = key;

        assert(r.m_chunkId < index.Chunks().size());
        const auto& chunk = index.Chunks()[r.m_chunkId];

        assert(r.m_indexInChunk < chunk.Sequences().size());
        const auto& sequence = chunk.Sequences()[r.m_indexInChunk];

        r.m_numberOfSamples = sequence.m_numberOfSamples;
        return true;
    }

}