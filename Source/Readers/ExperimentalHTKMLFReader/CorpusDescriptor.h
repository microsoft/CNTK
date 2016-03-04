//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Currently it is only a skeleton class.
// TODO: For HtkMlf it will be based on the set of sequences from the SCP file.
// TODO: Extract an interface.
class CorpusDescriptor
{
public:
    CorpusDescriptor(std::vector<std::wstring>&& sequences) : m_sequences(sequences)
    {
    }

    // Checks if the specified sequence should be used for reading.
    bool IsIncluded(const std::wstring& sequenceKey)
    {
        UNUSED(sequenceKey);
        return true;
    }

private:
    std::vector<std::wstring> m_sequences;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
