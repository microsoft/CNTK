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

    // Utterance id.
    size_t m_id;

    // Expansion length in case if utterance should be expanded.
    size_t m_expansionLength;

public:
    UtteranceDescription(msra::asr::htkfeatreader::parsedpath&& path)
        : m_path(std::move(path)), m_expansionLength(0)
    {
    }

    const msra::asr::htkfeatreader::parsedpath& GetPath() const
    {
        return m_path;
    }

    void ClearLogicalPath()
    {
        m_path.ClearLogicalPath();
    }

    size_t GetNumberOfFrames() const
    {
        return m_path.numframes();
    }

    string GetKey() const
    {
        return m_path.GetLogicalPath();
    }

    size_t GetId() const  { return m_id; }
    void SetId(size_t id) { m_id = id; }

    size_t GetExpansionLength() const { return m_expansionLength; }
    void SetExpansionLength(size_t length) { m_expansionLength = length; }
};

}}}
