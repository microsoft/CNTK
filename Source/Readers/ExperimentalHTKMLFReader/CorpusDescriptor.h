//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>

namespace Microsoft { namespace MSR { namespace CNTK {

// This class represents a string registry to share strings between different deserializers if needed.
// It associates a unique key for a given string.
// Currently it is implemented in-memory, but can be unloaded to external disk if needed.
class StringRegistry
{
public:
    StringRegistry()
    {}

    // Adds string value to the registry.
    size_t AddValue(const std::wstring& value)
    {
        assert(!Contains(value));
        auto iter = m_values.insert(std::make_pair(value, m_indexedValues.size()));
        m_indexedValues.push_back(&((iter.first)->first));
        return m_indexedValues.size() - 1;
    }

    // Get integer id of the string value.
    size_t GetIdByValue(const std::wstring& value) const
    {
        assert(Contains(value));
        return m_values.find(value)->second;
    }

    // Get string value by its integer id.
    const std::wstring& GetValueById(size_t id) const
    {
        assert(id < m_indexedValues.size());
        return *m_indexedValues[id];
    }

    // Checks whether the value exists.
    bool Contains(const std::wstring& value) const
    {
        return m_values.find(value) != m_values.end();
    }

private:
    DISABLE_COPY_AND_MOVE(StringRegistry);

    std::map<std::wstring, size_t> m_values;
    std::vector<const std::wstring*> m_indexedValues;
};

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
    StringRegistry& GetStringRegistry()
    {
        return m_stringRegistry;
    }

private:
    DISABLE_COPY_AND_MOVE(CorpusDescriptor);

    StringRegistry m_stringRegistry;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
