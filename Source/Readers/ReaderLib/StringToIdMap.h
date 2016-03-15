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
template<class TString>
class TStringToIdMap
{
public:
    TStringToIdMap()
    {}

    // Adds string value to the registry.
    size_t AddValue(const TString& value)
    {
        assert(!Contains(value));
        auto iter = m_values.insert(std::make_pair(value, m_indexedValues.size()));
        m_indexedValues.push_back(&((iter.first)->first));
        return m_indexedValues.size() - 1;
    }

    // Get integer id for the string value.
    size_t operator[](const TString& value) const
    {
        assert(Contains(value));
        return m_values.find(value)->second;
    }

    // Get string value by its integer id.
    const TString& operator[](size_t id) const
    {
        assert(id < m_indexedValues.size());
        return *m_indexedValues[id];
    }

    // Checks whether the value exists.
    bool Contains(const TString& value) const
    {
        return m_values.find(value) != m_values.end();
    }

private:
    DISABLE_COPY_AND_MOVE(TStringToIdMap);

    std::map<TString, size_t> m_values;
    std::vector<const TString*> m_indexedValues;
};

typedef TStringToIdMap<std::wstring> WStringToIdMap;
typedef TStringToIdMap<std::string> StringToIdMap;

}}}
