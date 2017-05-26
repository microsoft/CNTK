//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// This class represents a string registry pattern to share strings between different deserializers if needed.
// It associates a unique key for a given string.
// Currently it is implemented in-memory, but can be unloaded to external disk if needed.
// TODO: Move this class to Basics.h when it is required by more than one reader.
template<class TString>
class TStringToIdMap
{
public:
    TStringToIdMap()
    {}

    // Adds string value to the registry.
    void AddValue(const TString& value)
    {
        auto iter = m_values.insert(std::make_pair(value, m_indexedValues.size()));
        m_indexedValues.push_back(&((iter.first)->first));
    }

    // Tries to get a value by id.
    bool TryGet(const TString& value, size_t& id) const
    {
        const auto& it = m_values.find(value);
        if (it == m_values.end())
        {
            return false;
        }
        else
        {
            id = it->second;
            return true;
        }
    }

    // Get integer id for the string value, adding if not exists.
    size_t AddIfNotExists(const TString& value)
    {
        const auto& it = m_values.find(value);
        if (it == m_values.end())
        {
            AddValue(value);
            return m_values[value];
        }
        return it->second;
    }

    // Get integer id for the string value.
    size_t operator[](const TString& value) const
    {
        const auto& it = m_values.find(value);
        assert(it != m_values.end());
        return it->second;
    }

    // Get string value by its integer id.
    const TString& operator[](size_t id) const
    {
        if (id >= m_indexedValues.size())
            RuntimeError("Unknown id requested");
        return *m_indexedValues[id];
    }

    // Checks whether the value exists.
    bool Contains(const TString& value) const
    {
        return m_values.find(value) != m_values.end();
    }

private:
    // TODO: Move NonCopyable as a separate class to Basics.h
    DISABLE_COPY_AND_MOVE(TStringToIdMap);

    std::map<TString, size_t> m_values;
    std::deque<const TString*> m_indexedValues;
};

typedef TStringToIdMap<std::wstring> WStringToIdMap;
typedef TStringToIdMap<std::string> StringToIdMap;

}}}
