//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// Currently in-memory, can be externalized.
class StringRegistry
{
public:
    size_t AddValue(const std::wstring& value)
    {
        assert(!Contains(value));
        auto iter = m_values.insert(std::make_pair(value, m_indexedValues.size()));
        m_indexedValues.push_back(&((iter.first)->first));
        return m_indexedValues.size() - 1;
    }

    size_t GetIdByValue(const std::wstring& value)
    {
        assert(Contains(value));
        return m_values[value];
    }

    const std::wstring& GetValueById(size_t id)
    {
        return *m_indexedValues[id];
    }

    bool Contains(const std::wstring& value)
    {
        return m_values.find(value) != m_values.end();
    }

private:
    std::map<std::wstring, size_t> m_values;
    std::vector<const std::wstring*> m_indexedValues;
};

// Represents a full corpus.
// Defines which sequences should participate in the reading.
// TODO: Currently it is only a skeleton class.
// TODO: For HtkMlf it will be based on the set of sequences from the SCP file.
// TODO: Extract an interface.
class CorpusDescriptor
{
public:
    CorpusDescriptor()
    {
    }

    // Checks if the specified sequence should be used for reading.
    bool IsIncluded(const std::wstring& sequenceKey)
    {
        UNUSED(sequenceKey);
        return true;
    }

    StringRegistry& GetStringRegistry()
    {
        return m_stringRegistry;
    }

private:
    StringRegistry m_stringRegistry;
};

typedef std::shared_ptr<CorpusDescriptor> CorpusDescriptorPtr;

}}}
