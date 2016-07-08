//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "File.h"

namespace CNTK
{
    namespace _Internal
    {
#pragma region _SimpleVector

        template <typename T>
        _SimpleVector<T>::_SimpleVector()
            : m_vector(new std::vector<T>())
        {
        }

        template <typename T>
        _SimpleVector<T>::_SimpleVector(size_t numElements, const T& initVal/* = T()*/)
            : m_vector(new std::vector<T>(numElements, initVal))
        {
        }

        template <typename T>
        _SimpleVector<T>::~_SimpleVector()
        {
            delete m_vector;
        }

        template <typename T>
        _SimpleVector<T>::_SimpleVector(const _SimpleVector<T>& other)
            : m_vector(new std::vector<T>(*other.m_vector))
        {
        }

        template <typename T>
        _SimpleVector<T>& _SimpleVector<T>::operator=(const _SimpleVector<T>& other)
        {
            if (this != &other)
            {
                delete m_vector;
                m_vector = new std::vector<T>(*other.m_vector);
            }

            return *this;
        }

        template <typename T>
        _SimpleVector<T>::_SimpleVector(_SimpleVector<T>&& other)
            : m_vector(nullptr)
        {
            *this = std::move(other);
        }

        template <typename T>
        _SimpleVector<T>& _SimpleVector<T>::operator=(_SimpleVector<T>&& other)
        {
            assert(this != &other);

            delete m_vector;
            m_vector = other.m_vector;

            other.m_vector = nullptr;

            return *this;
        }

        template <typename T>
        T& _SimpleVector<T>::operator[](size_t idx)
        {
            assert(idx < Size());
            return (*m_vector)[idx];
        }

        template <typename T>
        const T& _SimpleVector<T>::operator[](size_t idx) const
        {
            assert(idx < Size());
            return (*m_vector)[idx];
        }

        template <typename T>
        size_t _SimpleVector<T>::Size() const
        {
            return m_vector->size();
        }

        template <typename T>
        T* _SimpleVector<T>::Data()
        {
            return m_vector->data();
        }

        template <typename T>
        const T* _SimpleVector<T>::Data() const
        {
            return m_vector->data();
        }

        template <typename T>
        void _SimpleVector<T>::PushBack(const T& value)
        {
            m_vector->push_back(value);
        }

        template <typename T>
        void _SimpleVector<T>::PushBack(T&& value)
        {
            m_vector->push_back(std::move(value));
        }

        template <typename ValueType>
        bool operator==(const _SimpleVector<ValueType>& first, const _SimpleVector<ValueType>& second)
        {
            return *first.m_vector == *second.m_vector;
        }

        // Explicit template instantiations
        template class _SimpleVector<Variable>;
        template class _SimpleVector<size_t>;
        template class _SimpleVector<Axis>;
        template class _SimpleVector<FunctionPtr>;
        template class _SimpleVector<ValuePtr>;

        template bool operator==(const _SimpleVector<size_t>& first, const _SimpleVector<size_t>& second);

#pragma endregion _SimpleVector

#pragma region _SimpleSet

        template <typename KeyType>
        _SimpleSet<KeyType>::_SimpleSet()
            : m_set(new std::unordered_set<KeyType>())
        {
        }

        template <typename KeyType>
        _SimpleSet<KeyType>::~_SimpleSet()
        {
            delete m_set;
        }

        template <typename KeyType>
        _SimpleSet<KeyType>::_SimpleSet(const _SimpleSet& other)
            : m_set(nullptr)
        {
            *this = other;
        }

        template <typename KeyType>
        _SimpleSet<KeyType>& _SimpleSet<KeyType>::operator=(const _SimpleSet& other)
        {
            if (this != &other)
            {
                delete m_set;
                m_set = new std::unordered_set<KeyType>(*(other.m_set));
            }

            return *this;
        }

        template <typename KeyType>
        _SimpleSet<KeyType>::_SimpleSet(_SimpleSet&& other)
            : m_set(nullptr)
        {
            *this = std::move(other);
        }

        template <typename KeyType>
        _SimpleSet<KeyType>& _SimpleSet<KeyType>::operator=(_SimpleSet&& other)
        {
            assert(this != &other);

            delete m_set;
            m_set = other.m_set;
            other.m_set = nullptr;

            return *this;
        }

        template <typename KeyType>
        bool _SimpleSet<KeyType>::Insert(const KeyType& key)
        {
            return m_set->insert(key).second;
        }

        template <typename KeyType>
        bool _SimpleSet<KeyType>::Contains(const KeyType& key) const
        {
            return (m_set->find(key) != m_set->end());
        }

        template <typename KeyType>
        size_t _SimpleSet<KeyType>::Size() const
        {
            return m_set->size();
        }

        template <typename KeyType>
        _SimpleSet<KeyType>::operator _SimpleVector<KeyType>() const
        {
            _SimpleVector<KeyType> retVector;
            for (auto iter = m_set->begin(); iter != m_set->end(); ++iter)
                retVector.PushBack(*iter);

            return retVector;
        }

        template <typename KeyType>
        bool operator==(const _SimpleSet<KeyType>& first, const _SimpleSet<KeyType>& second)
        {
            return *first.m_set == *second.m_set;
        }

        // Explicit template instantiations
        template class _SimpleSet<FunctionPtr>;
        template class _SimpleSet<Variable>;
        template class _SimpleSet<Placeholder>;
        template class _SimpleSet<const Function*>;

        template bool operator==(const _SimpleSet<Variable>& first, const _SimpleSet<Variable>& second);
        template bool operator==(const _SimpleSet<Placeholder>& first, const _SimpleSet<Placeholder>& second);

#pragma endregion _SimpleSet

#pragma region _SimpleMap

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>::_SimpleMap()
            : m_map(new std::unordered_map<KeyType, ValueType>())
        {
        }

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>::~_SimpleMap()
        {
            delete m_map;
        }

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>::_SimpleMap(const _SimpleMap& other)
            : m_map(nullptr)
        {
            *this = other;
        }

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>& _SimpleMap<KeyType, ValueType>::operator=(const _SimpleMap& other)
        {
            if (this != &other)
            {
                delete m_map;
                m_map = new std::unordered_map<KeyType, ValueType>(*(other.m_map));
            }

            return *this;
        }

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>::_SimpleMap(_SimpleMap&& other)
            : m_map(nullptr)
        {
            *this = std::move(other);
        }

        template <typename KeyType, typename ValueType>
        _SimpleMap<KeyType, ValueType>& _SimpleMap<KeyType, ValueType>::operator=(_SimpleMap&& other)
        {
            assert(this != &other);

            delete m_map;
            m_map = other.m_map;
            other.m_map = nullptr;

            return *this;
        }

        template <typename KeyType, typename ValueType>
        ValueType& _SimpleMap<KeyType, ValueType>::operator[](const KeyType& key)
        {
            return (*m_map)[key];
        }

        template <typename KeyType, typename ValueType>
        const ValueType& _SimpleMap<KeyType, ValueType>::operator[](const KeyType& key) const
        {
            return (*m_map)[key];
        }

        template <typename KeyType, typename ValueType>
        bool _SimpleMap<KeyType, ValueType>::Insert(const KeyType& key, const ValueType& value)
        {
            return m_map->insert({ key, value }).second;
        }

        template <typename KeyType, typename ValueType>
        bool _SimpleMap<KeyType, ValueType>::Contains(const KeyType& key) const
        {
            return (m_map->find(key) != m_map->end());
        }

        template <typename KeyType, typename ValueType>
        size_t _SimpleMap<KeyType, ValueType>::Size() const
        {
            return m_map->size();
        }

        template <typename KeyType, typename ValueType>
        _SimpleSet<KeyType> _SimpleMap<KeyType, ValueType>::Keys() const
        {
            _SimpleSet<KeyType> keys;
            for (auto iter = m_map->begin(); iter != m_map->end(); ++iter)
                keys.Insert(iter->first);

            return keys;
        }

        // Explicit template instantiations
        template class _SimpleMap<Variable, ValuePtr>;
        template class _SimpleMap<Variable, const ValuePtr>;
        template class _SimpleMap<Placeholder, Variable>;
        template class _SimpleMap<Variable, double>;

#pragma endregion _SimpleMap

    }

    template <typename T>
    void DictionaryValue::AllocateDataPtr(const T& value)
    {
        static_assert(std::is_same<T, NDShape>::value || std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value, "AllocateDataPtr called with invalid type");
        m_data.m_ptr = new T(value);
    }

    template <typename T>
    void DictionaryValue::FreePtrAsType()
    {
        T* typedPtr = reinterpret_cast<T*>(m_data.m_ptr);
        delete typedPtr;

        m_data.m_ptr = nullptr;
    }

    void DictionaryValue::FreeDataPtr()
    {
        if (m_valueType == Type::NDShape)
            FreePtrAsType<NDShape>();
        else if (m_valueType == Type::Vector)
            FreePtrAsType<_Internal::_SimpleVector<DictionaryValue>>();
    }

    Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, DictionaryValue& us)
    {
        size_t version;
        stream >> version;

        stream >> us.m_valueType;

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream >> us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::SizeT:
            stream >> us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream >> us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream >> us.m_data.m_double;
            break;
        case DictionaryValue::Type::NDShape:
        {
            size_t size;
            stream >> size;
            std::vector<size_t> dims(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> dims[i];
            }
            us.AllocateDataPtr(NDShape(dims));
            break;
        }
        case DictionaryValue::Type::Vector:
        {
            size_t size;
            stream >> size;
            _Internal::_SimpleVector<DictionaryValue> values(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> values[i];
            }
            us.AllocateDataPtr(values);
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const DictionaryValue& us)
    {
        stream << us.version;

        stream << us.ValueType();

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream << us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::SizeT:
            stream << us.m_data.m_sizeT;
            break;
        case DictionaryValue::Type::Float:
            stream << us.m_data.m_float;
            break;
        case DictionaryValue::Type::Double:
            stream << us.m_data.m_double;
            break;
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr = reinterpret_cast<NDShape*>(us.m_data.m_ptr);
            auto size = shapePtr->NumAxes();
            stream << size;
            for (auto i = 0; i < size; i++)
            {
                stream << shapePtr->operator[](i);
            }
            break;
        }
        case DictionaryValue::Type::Vector:
        {
            _Internal::_SimpleVector<DictionaryValue>* vectorPtr =
                reinterpret_cast<_Internal::_SimpleVector<DictionaryValue>*>(us.m_data.m_ptr);
            auto size = vectorPtr->Size();
            stream << size;
            for (auto i = 0; i < size; i++)
            {
                stream << vectorPtr->operator[](i);
            }
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    Dictionary::Dictionary()
        : m_dictionaryData(new std::unordered_map <std::wstring, DictionaryValue>)
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

    Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const Dictionary& us)
    {
        stream << us.version;
        stream << us.m_dictionaryData->size();
        for (auto it = us.m_dictionaryData->begin(); it != us.m_dictionaryData->end(); ++it)
        {
            stream << it->first;
            stream << it->second;
        }
        return stream;
    }

    Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, Dictionary& us)
    {
        size_t version;
        stream >> version;
        size_t size;
        stream >> size;
        us.m_dictionaryData->reserve(size);
        for (auto i = 0; i < size; i++)
        {
            std::wstring key;
            stream >> key;
            DictionaryValue value;
            stream >> value;
            us.m_dictionaryData->insert(make_pair(key, value));
        }
        return stream;
    }
    
    namespace _Internal
    {

        template <typename T>
        _Internal::_SimpleVector<DictionaryValue> SerializeToVector(const NDArrayViewPtr& viewPtr)
        {
            if (viewPtr->IsSparse())
            {
                LogicError("Sparse NDArrayView cannot be serialized into a vector.");
            }

            auto numElements = viewPtr->Shape().TotalSize();

            _Internal::_SimpleVector<DictionaryValue> values(numElements);

            NDArrayViewPtr cpuDataViewPtr = viewPtr;
            if ((viewPtr->Device().Type() != DeviceType::CPU))
            {
                cpuDataViewPtr = new NDArrayView(viewPtr->GetDataType(), viewPtr->Shape(), DeviceDescriptor::CPUDevice());
                cpuDataViewPtr->CopyFrom(*viewPtr);
            }

            const T* buffer = cpuDataViewPtr->DataBuffer<T>();
            for (auto i = 0; i < numElements; ++i)
            {
                T v = buffer[i];
                values[i] = DictionaryValue(v);
            }

            return values;
        }

        template <typename T>
        void DeserializeFromVector(const NDArrayViewPtr& viewPtr, const _Internal::_SimpleVector<DictionaryValue>& values)
        {
            if (viewPtr->IsSparse())
            {
                LogicError("Sparse NDArrayView cannot be deserialized from a vector.");
            }

            auto numElements = viewPtr->Shape().TotalSize();

            if (values.Size() != numElements)
            {
                LogicError("Number of elements (%lu) in the deserialized representation does not match the expected value (%lu)",
                           values.Size(), numElements);
            }

            NDArrayViewPtr cpuDataViewPtr = viewPtr;
            if ((viewPtr->Device().Type() != DeviceType::CPU))
            {
                cpuDataViewPtr = new NDArrayView(viewPtr->GetDataType(), viewPtr->Shape(), DeviceDescriptor::CPUDevice());
            }

            T* buffer = cpuDataViewPtr->WritableDataBuffer<T>();
            for (auto i = 0; i < numElements; ++i)
            {
                buffer[i] = values[i].GetValue<T>();
            }

            if ((viewPtr->Device().Type() != DeviceType::CPU))
            {
                viewPtr->CopyFrom(*cpuDataViewPtr);
            }
        }
    }

    // TODO: we store the type info for every element in the vector, which is extremely redundant.
    // Instead, it'd be nice to introduce some sort of DictionaryValueVector.
    _Internal::_SimpleVector<DictionaryValue> SerializeToVector(const NDArrayViewPtr& viewPtr)
    {
        switch (viewPtr->GetDataType())
        {
        case DataType::Float:
            return _Internal::SerializeToVector<float>(viewPtr);
        case DataType::Double:
            return _Internal::SerializeToVector<double>(viewPtr);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
        }
    }

    void DeserializeFromVector(const NDArrayViewPtr& viewPtr, const _Internal::_SimpleVector<DictionaryValue>& values) 
    {
        switch (viewPtr->GetDataType())
        {
        case DataType::Float:
            _Internal::DeserializeFromVector<float>(viewPtr, values);
            break;
        case DataType::Double:
            _Internal::DeserializeFromVector<double>(viewPtr, values);
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
        }
    }
     
    template void DictionaryValue::AllocateDataPtr<NDShape>(const NDShape& value);
    template void DictionaryValue::AllocateDataPtr<_Internal::_SimpleVector<DictionaryValue>>(const _Internal::_SimpleVector<DictionaryValue>& value);
}
