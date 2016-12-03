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
    wstring m_recurrentOp;
    int m_axis;
    bool IsSpatialRecurrence() const { return m_axis >= 0; }

    RnnAttributes(bool bidirectional, size_t numLayers, size_t hiddenSize, const wstring& recurrentOp, int axis) :
        m_bidirectional(bidirectional), m_numLayers(numLayers), m_hiddenSize(hiddenSize), m_recurrentOp(recurrentOp), m_axis(axis)
    {
        if (m_recurrentOp != wstring(L"lstm")    && m_recurrentOp != wstring(L"gru") &&
            m_recurrentOp != wstring(L"rnnReLU") && m_recurrentOp != wstring(L"rnnTanh"))
        {
            InvalidArgument("Unknown cell type '%ls'. Supported values are 'lstm', 'gru', 'rnnReLU', 'rnnTanh'.", m_recurrentOp.c_str());
        }

        if (m_axis != -1 && m_axis != 2)
            InvalidArgument("OptimizedRNNStack: invalid 'axis' parameter %d, currently supported values are -1 and 2.", m_axis);
    }

    // compute the total number of parameters, for inference of weight matrix size
    pair<size_t, size_t> GetNumParameters(size_t inputDim) const
    {
        const size_t bidirFactor = m_bidirectional ? 2 : 1;
        const size_t numNetworks =
            (m_recurrentOp == L"lstm") ? 4 :
            (m_recurrentOp == L"gru" ) ? 3 :
            /*else*/                     1 ;
        size_t total = 0;
        for (size_t i = 0; i < m_numLayers; i++)
        {
            size_t oneNetTotal =
                numNetworks * m_hiddenSize              // 1, 3, or 4 networks producing hidden-dim output
                            * (inputDim + m_hiddenSize) // each network has these two inputs
              + numNetworks * m_hiddenSize              // biases
                            * 2;                        // for unknown reasons, cudnn5 uses 2 bias terms everywhere
            total += oneNetTotal * bidirFactor;         // 1 or 2 directions
            inputDim = bidirFactor * m_hiddenSize;      // next layer continues with this as input
        }
        return make_pair(m_hiddenSize, total / m_hiddenSize);
    }

    bool operator==(const RnnAttributes& other) const
    {
        return
            m_bidirectional == other.m_bidirectional &&
            m_numLayers    == other.m_numLayers      &&
            m_hiddenSize   == other.m_hiddenSize     &&
            m_recurrentOp  == other.m_recurrentOp    &&
            m_axis         == other.m_axis;
    }

    void Read(File& stream, bool readAxis)
    {
        size_t bidirectional;
        stream >> bidirectional; m_bidirectional = !!bidirectional;
        stream >> m_numLayers;
        stream >> m_hiddenSize;
        stream >> m_recurrentOp;
        if (readAxis)
            stream >> m_axis;
        else // lecagy
        {
            m_axis = -1; // note: back compat for windowed models deliberately dropped
            if      (m_recurrentOp == wstring(L"LSTM"))     m_recurrentOp = L"lstm"; // map names
            else if (m_recurrentOp == wstring(L"GRU"))      m_recurrentOp = L"gru";
            else if (m_recurrentOp == wstring(L"RNN_RELU")) m_recurrentOp = L"rnnReLU";
            else if (m_recurrentOp == wstring(L"RNN_TANH")) m_recurrentOp = L"rnnTanh";
        }
    }

    void Write(File& stream) const
    {
        size_t bidirectional = m_bidirectional ? 1 : 0;
        stream << bidirectional;
        stream << m_numLayers;
        stream << m_hiddenSize;
        stream << m_recurrentOp;
        stream << m_axis;
    }

private:
    // disallow public default constructor
    RnnAttributes() {}
};

}}}
