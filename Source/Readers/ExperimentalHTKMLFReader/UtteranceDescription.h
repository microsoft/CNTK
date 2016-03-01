//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "DataDeserializer.h"
#include "../HTKMLFReader/htkfeatio.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class represents a descriptor for a single utterance.
// It is only used internally by the HTK deserializer.
class UtteranceDescription : public SequenceDescription
{
    // Archive filename and frame range in that file.
    msra::asr::htkfeatreader::parsedpath m_path;

    // Index of the utterance inside the chunk.
    size_t m_indexInsideChunk;

public:
    UtteranceDescription(msra::asr::htkfeatreader::parsedpath&& path)
        : m_path(std::move(path)), m_indexInsideChunk(0)
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

    wstring GetKey() const
    {
        std::wstring filename(m_path);
        return filename.substr(0, filename.find_last_of(L"."));
    }

    size_t GetIndexInsideChunk() const
    {
        return m_indexInsideChunk;
    }

    void SetIndexInsideChunk(size_t indexInsideChunk)
    {
        m_indexInsideChunk = indexInsideChunk;
    }
};

}}}
