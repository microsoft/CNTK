//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "StringToIdMap.h"
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Extract an interface.
class CorpusDescriptor
{
    bool m_includeAll;
    std::set<size_t> m_sequenceIds;

public:
    CorpusDescriptor(const std::wstring& file) : m_includeAll(false)
    {
        // Add all sequence ids.
        for (msra::files::textreader r(file); r;)
        {
            m_sequenceIds.insert(m_stringRegistry[r.getline()]);
        }
    }

    // By default include all sequences.
    CorpusDescriptor() : m_includeAll(true)
    {}

    // Checks if the specified sequence should be used for reading.
    bool IsIncluded(const std::string& sequenceKey)
    {
        if (m_includeAll)
        {
            return true;
        }

        size_t id;
        if(!m_stringRegistry.TryGet(sequenceKey, id))
        {
            return false;
        }

        return m_sequenceIds.find(id) != m_sequenceIds.end();
    }

    // Gets the string registry
    StringToIdMap& GetStringRegistry()
    {
        return m_stringRegistry;
    }

    // Gets the string registry
    const StringToIdMap& GetStringRegistry() const
    {
        return m_stringRegistry;
    }

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);

    StringToIdMap m_stringRegistry;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
