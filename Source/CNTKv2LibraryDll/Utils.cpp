//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"

namespace CNTK
{
    Dictionary::Dictionary()
        : m_dictionaryData(new std::unordered_map < std::wstring, DictionaryValue>)
    {
    }

    Dictionary::~Dictionary()
    {
        delete m_dictionaryData;
    }

    Dictionary::Dictionary(Dictionary&& other)
        : m_dictionaryData(nullptr)
    {
        *this = std::move(other);
    }

    Dictionary& Dictionary::operator=(Dictionary&& other)
    {
        assert(this != &other);

        delete m_dictionaryData;

        m_dictionaryData = other.m_dictionaryData;
        other.m_dictionaryData = nullptr;

        return *this;
    }

    DictionaryValue& Dictionary::operator[](const wchar_t* key)
    {
        return (*m_dictionaryData)[key];
    }

    DictionaryValue Dictionary::operator[](const wchar_t* key) const
    {
        return m_dictionaryData->at(key);
    }

    bool Dictionary::Contains(const wchar_t* key) const
    {
        return (m_dictionaryData->find(key) != m_dictionaryData->end());
    }
}
