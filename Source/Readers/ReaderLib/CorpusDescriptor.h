//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "StringToIdMap.h"
#include <set>
#include <functional>
#include <sstream>

namespace Microsoft { namespace MSR { namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Extract an interface.
class CorpusDescriptor
{
public:
    CorpusDescriptor(const std::wstring& file, bool numericSequenceKeys) : CorpusDescriptor(numericSequenceKeys)
    {
        m_includeAll = false;

        // Add all sequence ids.
        for (msra::files::textreader r(file); r;)
        {
            m_sequenceIds.insert(KeyToId(r.getline()));
        }
    }

    bool IsNumericSequenceKeys() const
    {
        return m_numericSequenceKeys;
    }

    // By default include all sequences.
    CorpusDescriptor(bool numericSequenceKeys) : m_includeAll(true), m_numericSequenceKeys(numericSequenceKeys)
    {
        if (numericSequenceKeys)
        {
            KeyToId = [](const std::string& key)
            {
                size_t id = 0;
                int converted = sscanf_s(key.c_str(), "%" PRIu64, &id);
                if (converted != 1)
                    RuntimeError("Invalid numeric sequence id '%s'", key.c_str());
                return id;
            };

            IdToKey = [](size_t id)
            {
                return std::to_string(id);
            };
        }
        else
        {
            KeyToId = [this](const std::string& key)
            {
                // The function has to provide a size_t unique "hash" for the input key
                // If we see the key for the first time, we add it to the registry.
                // Otherwise we retrieve the hash value for the key from the registry.
                return m_keyToIdMap.AddIfNotExists(key);
            };

            IdToKey = [this](size_t id)
            {
                // This will throw if the id is not present.
                return m_keyToIdMap[id];
            };
        }
    }

    // Checks if the specified sequence key should be used for reading.
    bool IsIncluded(const std::string& sequenceKey)
    {
        if (m_includeAll)
        {
            return true;
        }

        size_t id = 0;
        if (m_numericSequenceKeys)
            id = KeyToId(sequenceKey);
        else
        {
            if (!m_keyToIdMap.TryGet(sequenceKey, id))
                return false;
        }
        return m_sequenceIds.find(id) != m_sequenceIds.end();
    }

    std::function<size_t(const std::string&)> KeyToId;
    std::function<std::string(size_t)> IdToKey;

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);
    bool m_numericSequenceKeys;
    bool m_includeAll;
    std::set<size_t> m_sequenceIds;

    StringToIdMap m_keyToIdMap;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
