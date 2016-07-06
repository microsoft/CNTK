//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"

namespace CNTK
{
    namespace Internal
    {
        ReferenceCount::ReferenceCount()
            : m_rc(new std::atomic<size_t>(0)) 
        {}

        /*virtual*/ ReferenceCount::~ReferenceCount() 
        {
            delete m_rc;
        }

        size_t ReferenceCount::AddReference()
        {
            return ++(*m_rc);
        }

        size_t ReferenceCount::RemoveReference()
        {
            assert(m_rc->load() > 0);
            return --(*m_rc);
        }

        size_t ReferenceCount::GetReferenceCount()
        {
            return m_rc->load();
        }

#pragma region SimpleVector

        template <typename T>
        SimpleVector<T>::SimpleVector()
            : m_vector(new std::vector<T>())
        {
        }

        template <typename T>
        SimpleVector<T>::SimpleVector(size_t numElements, const T& initVal/* = T()*/)
            : m_vector(new std::vector<T>(numElements, initVal))
        {
        }

        template <typename T>
        SimpleVector<T>::~SimpleVector()
        {
            delete m_vector;
        }

        template <typename T>
        SimpleVector<T>::SimpleVector(const SimpleVector<T>& other)
            : m_vector(new std::vector<T>(*other.m_vector))
        {
        }

        template <typename T>
        SimpleVector<T>& SimpleVector<T>::operator=(const SimpleVector<T>& other)
        {
            if (this != &other)
            {
                delete m_vector;
                m_vector = new std::vector<T>(*other.m_vector);
            }

            return *this;
        }

        template <typename T>
        SimpleVector<T>::SimpleVector(SimpleVector<T>&& other)
            : m_vector(nullptr)
        {
            *this = std::move(other);
        }

        template <typename T>
        SimpleVector<T>& SimpleVector<T>::operator=(SimpleVector<T>&& other)
        {
            assert(this != &other);

            delete m_vector;
            m_vector = other.m_vector;

            other.m_vector = nullptr;

            return *this;
        }

        template <typename T>
        T& SimpleVector<T>::operator[](size_t idx)
        {
            assert(idx < Size());
            return (*m_vector)[idx];
        }

        template <typename T>
        const T& SimpleVector<T>::operator[](size_t idx) const
        {
            assert(idx < Size());
            return (*m_vector)[idx];
        }

        template <typename T>
        size_t SimpleVector<T>::Size() const
        {
            return m_vector->size();
        }

        template <typename T>
        T* SimpleVector<T>::Data()
        {
            return m_vector->data();
        }

        template <typename T>
        const T* SimpleVector<T>::Data() const
        {
            return m_vector->data();
        }

        template <typename T>
        void SimpleVector<T>::PushBack(const T& value)
        {
            m_vector->push_back(value);
        }

        template <typename T>
        void SimpleVector<T>::PushBack(T&& value)
        {
            m_vector->push_back(std::move(value));
        }

        template <typename ValueType>
        bool operator==(const SimpleVector<ValueType>& first, const SimpleVector<ValueType>& second)
        {
            return *first.m_vector == *second.m_vector;
        }

        // Explicit template instantiations
        template class SimpleVector<Variable>;
        template class SimpleVector<size_t>;
        template class SimpleVector<Axis>;
        template class SimpleVector<FunctionPtr>;

        template bool operator==(const SimpleVector<size_t>& first, const SimpleVector<size_t>& second);
	
#pragma endregion SimpleVector

#pragma region SimpleSet

        template <typename KeyType>
        SimpleSet<KeyType>::SimpleSet()
            : m_set(new std::unordered_set<KeyType>())
        {
        }

        template <typename KeyType>
        SimpleSet<KeyType>::~SimpleSet()
        {
            delete m_set;
        }

        template <typename KeyType>
        SimpleSet<KeyType>::SimpleSet(const SimpleSet& other)
            : m_set(nullptr)
        {
            *this = other;
        }

        template <typename KeyType>
        SimpleSet<KeyType>& SimpleSet<KeyType>::operator=(const SimpleSet& other)
        {
            if (this != &other)
            {
                delete m_set;
                m_set = new std::unordered_set<KeyType>(*(other.m_set));
            }

            return *this;
        }

        template <typename KeyType>
        SimpleSet<KeyType>::SimpleSet(SimpleSet&& other)
            : m_set(nullptr)
        {
            *this = std::move(other);
        }

        template <typename KeyType>
        SimpleSet<KeyType>& SimpleSet<KeyType>::operator=(SimpleSet&& other)
        {
            assert(this != &other);

            delete m_set;
            m_set = other.m_set;
            other.m_set = nullptr;

            return *this;
        }

        template <typename KeyType>
        bool SimpleSet<KeyType>::Insert(const KeyType& key)
        {
            return m_set->insert(key).second;
        }

        template <typename KeyType>
        bool SimpleSet<KeyType>::Contains(const KeyType& key) const
        {
            return (m_set->find(key) != m_set->end());
        }

        template <typename KeyType>
        size_t SimpleSet<KeyType>::Size() const
        {
            return m_set->size();
        }

        template <typename KeyType>
        SimpleSet<KeyType>::operator SimpleVector<KeyType>() const
        {
            SimpleVector<KeyType> retVector;
            for (auto key : *m_set)
                retVector.PushBack(key);

            return retVector;
        }

        template <typename KeyType>
        bool operator==(const SimpleSet<KeyType>& first, const SimpleSet<KeyType>& second)
        {
            return *first.m_set == *second.m_set;
        }

        // Explicit template instantiations
        template class SimpleSet<FunctionPtr>;
        template class SimpleSet<Variable>;
        template class SimpleSet<Placeholder>;
        template class SimpleSet<const Function*>;

        template bool operator==(const SimpleSet<Variable>& first, const SimpleSet<Variable>& second);
        template bool operator==(const SimpleSet<Placeholder>& first, const SimpleSet<Placeholder>& second);

#pragma endregion SimpleSet

#pragma region SimpleMap

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>::SimpleMap()
            : m_map(new std::unordered_map<KeyType, ValueType>())
        {
        }

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>::~SimpleMap()
        {
            delete m_map;
        }

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>::SimpleMap(const SimpleMap& other)
            : m_map(nullptr)
        {
            *this = other;
        }

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>& SimpleMap<KeyType, ValueType>::operator=(const SimpleMap& other)
        {
            if (this != &other)
            {
                delete m_map;
                m_map = new std::unordered_map<KeyType, ValueType>(*(other.m_map));
            }

            return *this;
        }

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>::SimpleMap(SimpleMap&& other)
            : m_map(nullptr)
        {
            *this = std::move(other);
        }

        template <typename KeyType, typename ValueType>
        SimpleMap<KeyType, ValueType>& SimpleMap<KeyType, ValueType>::operator=(SimpleMap&& other)
        {
            assert(this != &other);

            delete m_map;
            m_map = other.m_map;
            other.m_map = nullptr;

            return *this;
        }

        template <typename KeyType, typename ValueType>
        ValueType& SimpleMap<KeyType, ValueType>::operator[](const KeyType& key)
        {
            return (*m_map)[key];
        }

        template <typename KeyType, typename ValueType>
        const ValueType& SimpleMap<KeyType, ValueType>::operator[](const KeyType& key) const
        {
            return (*m_map)[key];
        }

        template <typename KeyType, typename ValueType>
        bool SimpleMap<KeyType, ValueType>::Insert(const KeyType& key, const ValueType& value)
        {
            return m_map->insert({ key, value }).second;
        }

        template <typename KeyType, typename ValueType>
        bool SimpleMap<KeyType, ValueType>::Contains(const KeyType& key) const
        {
            return (m_map->find(key) != m_map->end());
        }

        template <typename KeyType, typename ValueType>
        size_t SimpleMap<KeyType, ValueType>::Size() const
        {
            return m_map->size();
        }

        template <typename KeyType, typename ValueType>
        SimpleSet<KeyType> SimpleMap<KeyType, ValueType>::Keys() const
        {
            SimpleSet<KeyType> keys;
            for (auto keyValuePair : *m_map)
                keys.Insert(keyValuePair.first);

            return keys;
        }

        // Explicit template instantiations
        template class SimpleMap<Variable, ValuePtr>;
        template class SimpleMap<Variable, const ValuePtr>;
        template class SimpleMap<Placeholder, Variable>;

#pragma endregion SimpleMap

    }

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
