//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "File.h"
#include <string>
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

struct RnnAttributes
{
    bool m_bidirectional;
    size_t m_numLayers;
    size_t m_hiddenSize;
    wstring m_rnnMode;
    int m_axis;
    bool IsWindowedRecurrence() const { return m_axis >= 0; }

    RnnAttributes(bool bidirectional, size_t numLayers, size_t hiddenSize, const wstring& rnnMode, int axis) :
        m_bidirectional(bidirectional), m_numLayers(numLayers), m_hiddenSize(hiddenSize), m_rnnMode(rnnMode), m_axis(axis)
    {
        if (m_axis != -1 && m_axis != 2)
            InvalidArgument("OptimizedRNNStack: invalid 'axis' parameter %d, currently supported values are -1 and 2.", m_axis);
    }

    bool operator==(const RnnAttributes& other) const
    {
        return
            m_bidirectional == other.m_bidirectional &&
            m_numLayers   == other.m_numLayers       &&
            m_hiddenSize  == other.m_hiddenSize      &&
            m_rnnMode     == other.m_rnnMode         &&
            m_axis        == other.m_axis;
    }

    void Read(File& stream, bool readAxis)
    {
        size_t bidirectional;
        stream >> bidirectional; m_bidirectional = !!bidirectional;
        stream >> m_numLayers;
        stream >> m_hiddenSize;
        stream >> m_rnnMode;
        if (readAxis)
            stream >> m_axis; // note: back compat for windowed models deliberately dropped
        else
            m_axis = -1;
    }

    void Write(File& stream) const
    {
        size_t bidirectional = m_bidirectional ? 1 : 0;
        stream << bidirectional;
        stream << m_numLayers;
        stream << m_hiddenSize;
        stream << m_rnnMode;
        stream << m_axis;
    }

private:
    // disallow public default constructor
    RnnAttributes() {}
};

}}}
