#pragma once

#include "File.h"
#include <string>
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

struct RnnParameters
{
    bool m_bidirectional;
    size_t m_numLayers;
    size_t m_hiddenSize;
    wstring m_rnnMode;

    RnnParameters(bool bidirectional, size_t numLayers, size_t hiddenSize, const wstring& rnnMode)
        : m_bidirectional(bidirectional), m_numLayers(numLayers), m_hiddenSize(hiddenSize), m_rnnMode(rnnMode)
    {}

    bool operator==(const RnnParameters& other) const
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
    }
    void Write(File& stream) const
    {
        size_t bidirectional = m_bidirectional ? 1 : 0;
        stream << bidirectional;
        stream << m_numLayers;
        stream << m_hiddenSize;
        stream << m_rnnMode;
    }

private:
    // disallow public default constructor
    RnnParameters() {}
};

} } }