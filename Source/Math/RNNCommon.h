//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "File.h"
#include <string>
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

struct RnnAttributes
{
    bool m_bidirectional;
    bool m_frameMode;
    size_t m_numLayers;
    size_t m_hiddenSize;
    wstring m_rnnMode;

    // TODO: expose frameMode parameter in constructor, also enable serializing, and bump version of node so that existing models can still be deserialized
    RnnAttributes(bool bidirectional, size_t numLayers, size_t hiddenSize, const wstring& rnnMode/*, bool frameMode = false*/)
        : m_bidirectional(bidirectional), m_numLayers(numLayers), m_hiddenSize(hiddenSize), m_rnnMode(rnnMode), m_frameMode(false)
    {}

    bool operator==(const RnnAttributes& other) const
    {
        return
            m_bidirectional == other.m_bidirectional &&
            m_numLayers == other.m_numLayers &&
            m_hiddenSize == other.m_hiddenSize &&
            m_rnnMode == other.m_rnnMode;
    }

    void Read(File& stream)
    {
        size_t bidirectional;
        stream >> bidirectional; m_bidirectional = !!bidirectional;
        stream >> m_numLayers;
        stream >> m_hiddenSize;
        stream >> m_rnnMode;
#if 0
        stream >> m_frameMode;
#endif
    }
    void Write(File& stream) const
    {
        size_t bidirectional = m_bidirectional ? 1 : 0;
        stream << bidirectional;
        stream << m_numLayers;
        stream << m_hiddenSize;
        stream << m_rnnMode;
#if 0
        stream << m_frameMode;
#endif
    }

private:
    // disallow public default constructor
    RnnAttributes() {}
};

} } }