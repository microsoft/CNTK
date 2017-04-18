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
    size_t AddValue(const TString& value)
    {
        m_dirty = true;
        m_values.push_back(std::make_pair(value, m_indexedValues.size()));
        m_indexedValues.push_back(&m_values.back().first);
        return m_indexedValues.size() - 1;
    }

    // Tries to get a value by id.
    bool TryGet(const TString& value, size_t& id) const
    {
        sortIfNeeded();
        auto it = std::lower_bound(m_values.begin(), m_values.end(), std::make_pair(value, 0), [](const TElement& a, const TElement& b) { return a.first < b.first; });
        if (it->first != value)
        {
            return false;
        }
        else
        {
            id = it->second;
            return true;
        }
    }

    // Get integer id for the string value.
    size_t operator[](const TString& value) const
    {
        sortIfNeeded();
        auto it = std::lower_bound(m_values.begin(), m_values.end(), std::make_pair(value, 0), [](const TElement& a, const TElement& b) { return a.first < b.first; });
        if (it->first != value)
            return SIZE_MAX;
        return it->second;
    }

    // Get string value by its integer id.
    const TString& operator[](size_t id) const
    {
        if (id >= m_indexedValues.size())
            RuntimeError("Unknown id requested");
        return *m_indexedValues[id];
    }

private:
    void sortIfNeeded() const
    {
        if (!m_dirty)
            return;

        std::sort(m_values.begin(), m_values.end(), [](const TElement& a, const TElement& b) { return a.first < b.first; });
        m_dirty = false;
    }

    // TODO: Move NonCopyable as a separate class to Basics.h
    DISABLE_COPY_AND_MOVE(TStringToIdMap);

    mutable bool m_dirty;
    typedef std::pair<TString, size_t> TElement;
    mutable std::deque<TElement> m_values;
    std::deque<const TString*> m_indexedValues;
};

typedef TStringToIdMap<std::wstring> WStringToIdMap;
typedef TStringToIdMap<std::string> StringToIdMap;

}}}
