//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "StringToIdMap.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Currently it is only a skeleton class.
// TODO: For HtkMlf it can be based on the set of sequences from the SCP file.
// TODO: Extract an interface.
class CorpusDescriptor
{
public:
    CorpusDescriptor()
    {}

    // Checks if the specified sequence should be used for reading.
    bool IsIncluded(const std::wstring& sequenceKey)
    {
        UNUSED(sequenceKey);
        return true;
    }

    // Gets string registry
    WStringToIdMap& GetStringRegistry()
    {
        return m_stringRegistry;
    }

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);

    WStringToIdMap m_stringRegistry;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
