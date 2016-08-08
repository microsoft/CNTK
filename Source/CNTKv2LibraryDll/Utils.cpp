//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "File.h"

using namespace std;

namespace CNTK
{
    template <typename T>
    void DictionaryValue::AllocateDataPtr(const T& value)
    {
        static_assert(is_same<T, NDShape>::value ||
                      is_same<T, wstring>::value ||
                      is_same<T, vector<DictionaryValue>>::value ||
                      is_same<T, Dictionary>::value, "AllocateDataPtr called with invalid type");
        m_data.m_ptr = new T(value);
    }

    template <typename T>
    void DictionaryValue::FreePtrAsType()
    {
        T* typedPtr = reinterpret_cast<T*>(m_data.m_ptr);
        delete typedPtr;

        m_data.m_ptr = nullptr;
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
            vector<size_t> dims(size);
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
            vector<DictionaryValue> values(size);
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
            vector<DictionaryValue>* vectorPtr =
                reinterpret_cast<vector<DictionaryValue>*>(us.m_data.m_ptr);
            auto size = vectorPtr->size();
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
        : m_dictionaryData(new unordered_map <wstring, DictionaryValue>)
    {
    }

    Dictionary::~Dictionary()
    {
    }

    Dictionary::Dictionary(const Dictionary& other)
    {
        *this = other;
    }

    Dictionary& Dictionary::operator=(const Dictionary& other)
    {
        assert(this != &other);
        m_dictionaryData.reset(new std::unordered_map<std::wstring, DictionaryValue>(*(other.m_dictionaryData)));
        return *this;
    }

    Dictionary::Dictionary(Dictionary&& other)
        : m_dictionaryData(nullptr)
    {
        *this = move(other);
    }

    Dictionary& Dictionary::operator=(Dictionary&& other)
    {
        assert(this != &other);

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
            wstring key;
            stream >> key;
            DictionaryValue value;
            stream >> value;
            us.m_dictionaryData->insert(make_pair(key, value));
        }
        return stream;
    }

    template <typename T>
    vector<DictionaryValue> SerializeToVector(const NDArrayViewPtr& viewPtr)
    {
        if (viewPtr->IsSparse())
        {
            LogicError("Sparse NDArrayView cannot be serialized into a vector.");
        }

        auto numElements = viewPtr->Shape().TotalSize();

        vector<DictionaryValue> values(numElements);

        NDArrayViewPtr cpuDataViewPtr = viewPtr;
        if ((viewPtr->Device().Type() != DeviceKind::CPU))
        {
            cpuDataViewPtr = MakeSharedObject<NDArrayView>(viewPtr->GetDataType(), viewPtr->Shape(), DeviceDescriptor::CPUDevice());
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
    void DeserializeFromVector(const NDArrayViewPtr& viewPtr, const vector<DictionaryValue>& values)
    {
        if (viewPtr->IsSparse())
        {
            LogicError("Sparse NDArrayView cannot be deserialized from a vector.");
        }

        auto numElements = viewPtr->Shape().TotalSize();

        if (values.size() != numElements)
        {
            LogicError("Number of elements (%lu) in the deserialized representation does not match the expected value (%lu)",
                        values.size(), numElements);
        }

        NDArrayViewPtr cpuDataViewPtr = viewPtr;
        if ((viewPtr->Device().Type() != DeviceKind::CPU))
        {
            cpuDataViewPtr = MakeSharedObject<NDArrayView>(viewPtr->GetDataType(), viewPtr->Shape(), DeviceDescriptor::CPUDevice());
        }

        T* buffer = cpuDataViewPtr->WritableDataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            buffer[i] = values[i].GetValue<T>();
        }

        if ((viewPtr->Device().Type() != DeviceKind::CPU))
        {
            viewPtr->CopyFrom(*cpuDataViewPtr);
        }
    }

    // TODO: we store the type info for every element in the vector, which is extremely redundant.
    // Instead, it'd be nice to introduce some sort of DictionaryValueVector.
    vector<DictionaryValue> SerializeToVector(const NDArrayViewPtr& viewPtr)
    {
        switch (viewPtr->GetDataType())
        {
        case DataType::Float:
            return SerializeToVector<float>(viewPtr);
        case DataType::Double:
            return SerializeToVector<double>(viewPtr);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
        }
    }

    void DeserializeFromVector(const NDArrayViewPtr& viewPtr, const vector<DictionaryValue>& values) 
    {
        switch (viewPtr->GetDataType())
        {
        case DataType::Float:
            DeserializeFromVector<float>(viewPtr, values);
            break;
        case DataType::Double:
            DeserializeFromVector<double>(viewPtr, values);
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
        }
    }
     
    template void DictionaryValue::AllocateDataPtr<NDShape>(const NDShape& value);
    template void DictionaryValue::AllocateDataPtr<vector<DictionaryValue>>(const vector<DictionaryValue>& value);
    template void DictionaryValue::AllocateDataPtr<wstring>(const wstring& value);
    template void DictionaryValue::AllocateDataPtr<Dictionary>(const Dictionary& value);

    template void DictionaryValue::FreePtrAsType<NDShape>();
    template void DictionaryValue::FreePtrAsType<vector<DictionaryValue>>();
    template void DictionaryValue::FreePtrAsType<wstring>();
    template void DictionaryValue::FreePtrAsType<Dictionary>();
}
