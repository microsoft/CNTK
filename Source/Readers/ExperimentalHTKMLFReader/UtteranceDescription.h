//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "../HTKMLFReader/htkfeatio.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class represents a descriptor for a single utterance.
// It is only used internally by the HTK deserializer.
class UtteranceDescription
{
    // Archive filename and frame range in that file.
    msra::asr::htkfeatreader::parsedpath m_path;

    // Index of the utterance inside the chunk.
    size_t m_indexInsideChunk;
    // Position of the first sample of the utterance inside the chunk.
    size_t m_startFrameIndexInsideChunk;
    // Chunk id.
    size_t m_chunkId;
    // Utterance id.
    size_t m_id;

public:
    UtteranceDescription(msra::asr::htkfeatreader::parsedpath&& path)
        : m_path(std::move(path)), m_indexInsideChunk(0), m_startFrameIndexInsideChunk(0), m_chunkId(SIZE_MAX)
    {
    }

    const msra::asr::htkfeatreader::parsedpath& GetPath() const
    {
        return m_path;
    }

    size_t GetNumberOfFrames() const
    {
        return m_path.numframes();
    }

    string GetKey() const
    {
        return m_path.GetLogicalPath();
    }

    void AssignToChunk(size_t chunkId, size_t indexInsideChunk, size_t frameInsideChunk)
    {
        m_chunkId = chunkId;
        m_indexInsideChunk = indexInsideChunk;
        m_startFrameIndexInsideChunk = frameInsideChunk;
    }

    size_t GetId() const  { return m_id; }
    void SetId(size_t id) { m_id = id; }

    size_t GetChunkId() const  { return m_chunkId; }
    size_t GetIndexInsideChunk() const { return m_indexInsideChunk;}
    size_t GetStartFrameIndexInsideChunk() const { return m_startFrameIndexInsideChunk; }

    void SetStartFrameInsideChunk(size_t startFrameIndexInsideChunk)
    {
        m_startFrameIndexInsideChunk = startFrameIndexInsideChunk;
    }
};

}}}
