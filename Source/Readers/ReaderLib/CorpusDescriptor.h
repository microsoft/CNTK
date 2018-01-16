//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include "StringToIdMap.h"

namespace CNTK {

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Extract an interface.
class CorpusDescriptor
{
    // Defines which sequences should participate in the reading,
    // djb2 algorithm from http://www.cse.yorku.ca/~oz/hash.html
    size_t Hash(const std::string& key)
    {
        size_t result = 5381;
        for (const auto& c : key)
            result = ((result << 5) + result) ^ c;
        return result;
    }

public:
    bool IsNumericSequenceKeys() const
    {
        return m_numericSequenceKeys;
    }

    bool IsHashingEnabled() const
    {
        return m_useHash;
    }

    // Should be incremented each time the Hash() function above is modified.
    static constexpr size_t s_hashVersion = 1;

    // By default include all sequences.
    CorpusDescriptor(bool numericSequenceKeys, bool useHash = false)
        : m_numericSequenceKeys(numericSequenceKeys), m_useHash(useHash)
    {
        if (numericSequenceKeys)
        {
            if (m_useHash)
                RuntimeError("Hashing should not be used with numeric sequence keys.");

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
                if (m_useHash)
                    return Hash(key);

                // The function has to provide a size_t unique "hash" for the input key
                // If we see the key for the first time, we add it to the registry.
                // Otherwise we retrieve the hash value for the key from the registry.
                return m_keyToIdMap.AddIfNotExists(key);
            };

            IdToKey = [this](size_t id)
            {
                if (m_useHash)
                    RuntimeError("Retrieving original sequence key is not supported."
                        " Please disable hashing in configuration.");

                // This will throw if the id is not present.
                return m_keyToIdMap[id];
            };
        }
    }

    std::function<size_t(const std::string&)> KeyToId;
    std::function<std::string(size_t)> IdToKey;

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);
    bool m_numericSequenceKeys;
    bool m_useHash;

    StringToIdMap m_keyToIdMap;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}
