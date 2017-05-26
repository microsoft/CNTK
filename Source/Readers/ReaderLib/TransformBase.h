//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <unordered_map>

#include "Transformer.h"
#include "Config.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Base class for transforms.
class TransformBase : public Transformer
{
public:
    explicit TransformBase(const ConfigParameters& config)
    {
        m_seed = config(L"seed", 0u);
        std::wstring precision = config(L"precision", L"float");
        if (AreEqualIgnoreCase(precision, L"float"))
            m_precision = ElementType::tfloat;
        else if (AreEqualIgnoreCase(precision, L"double"))
            m_precision = ElementType::tdouble;
        else
            RuntimeError("Unsupported precision type is specified, '%ls'", precision.c_str());
    }

    void StartEpoch(const EpochConfiguration&) override {}

    // The method describes how input stream is transformed to the output stream. Called once per applied stream.
    // Currently we only support transforms of dense streams.
    StreamDescription Transform(const StreamDescription& inputStream) override
    {
        if (inputStream.m_storageType != StorageType::dense)
        {
            LogicError("The class currently only supports transforms on dense input streams.");
        }

        m_inputStream = inputStream;
        m_outputStream = m_inputStream;
        return m_outputStream;
    }

    virtual ~TransformBase() {}

protected:
    // Seed  getter.
    unsigned int GetSeed() const
    {
        return m_seed;
    }

    // Input stream.
    StreamDescription m_inputStream;
    // Output stream.
    StreamDescription m_outputStream;
    // Seed.
    unsigned int m_seed;
    // Required precision.
    ElementType m_precision;
};

}}}
