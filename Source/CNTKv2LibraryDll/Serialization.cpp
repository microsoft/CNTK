//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace CNTK
{
    // This wrapper redefines operator<< in terms of unformatted (binary) write operation.
    struct BinaryOStreamWrapper
    {
        BinaryOStreamWrapper(std::ostream& s) : m_stream(s) {}

        template<typename T>
        typename std::enable_if<std::is_pod<T>::value, BinaryOStreamWrapper&>::type
        operator<<(const T& value)
        { 
            m_stream.write(reinterpret_cast<const char*>(&value), sizeof(T)); 
            return *this ; 
        }

        BinaryOStreamWrapper& operator<<(const std::wstring& str)
        {
            size_t length = str.length();
            *this << length;
            m_stream.write(reinterpret_cast<const char*>(str.c_str()), str.length() * sizeof(wchar_t)); 
            return *this; 
        }

        operator std::ostream& () { return m_stream; }

        std::ostream& m_stream;
        BinaryOStreamWrapper(const BinaryOStreamWrapper&) = delete; BinaryOStreamWrapper(BinaryOStreamWrapper&&) = delete; BinaryOStreamWrapper& operator=(const BinaryOStreamWrapper&) = delete; BinaryOStreamWrapper& operator=(BinaryOStreamWrapper&&) = delete;
    };

    // This wrapper redefines operator>> in terms of unformatted (binary) read operation.
    struct BinaryIStreamWrapper
    {
        BinaryIStreamWrapper(std::istream& s) : m_stream(s) {}

        template<typename T>
        typename std::enable_if<std::is_pod<T>::value, BinaryIStreamWrapper&>::type
        operator>>(T& value)
        { 
            static_assert(sizeof(T) <= sizeof(size_t), "size_t is the largest supported type.");
            m_stream.read(buf, sizeof(T)); 
            value = *(reinterpret_cast<T*>(buf));
            return *this ; 
        }

        BinaryIStreamWrapper& operator>>(std::wstring& str)
        { 
            size_t length;
            *this >> length;
            str.reserve(length);
            for (size_t i = 0; i < length; ++i)
            {
                m_stream.read(buf, sizeof(wchar_t)); 
                str.append(reinterpret_cast<wchar_t*>(buf));
            }
            return *this; 
        }

        operator std::istream& () const { return m_stream ;}

        std::istream& m_stream;
        char buf[sizeof(size_t)];
        BinaryIStreamWrapper(const BinaryIStreamWrapper&) = delete; BinaryIStreamWrapper(BinaryIStreamWrapper&&) = delete; BinaryIStreamWrapper& operator=(const BinaryIStreamWrapper&) = delete; BinaryIStreamWrapper& operator=(BinaryIStreamWrapper&&) = delete;
    };

    BinaryOStreamWrapper& operator<<(BinaryOStreamWrapper& stream, const NDShape& us)
    {
        auto size = us.Rank();
        stream << size;
        for (auto i = 0; i < size; i++)
        {
            stream << us[i];
        }
        return stream;
    }

    BinaryOStreamWrapper& operator<<(BinaryOStreamWrapper& stream, const Axis& us)
    {
        stream << us.StaticAxisIndex(false);
        stream << us.Name();
        stream << us.IsOrdered();

        return stream;
    }

    template <typename T>
    void Write(BinaryOStreamWrapper& stream, const NDArrayView& view)
    {
        assert(view.Device().Type() == DeviceKind::CPU);

        auto numElements = view.Shape().TotalSize();
        const T* buffer = view.DataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            stream << buffer[i];
        }
    }

    template <typename T>
    void Read(BinaryIStreamWrapper& stream, NDArrayView& view)
    {
        assert(view.Device().Type() == DeviceKind::CPU);
        
        auto numElements = view.Shape().TotalSize();
        T* buffer = view.WritableDataBuffer<T>();
        for (auto i = 0; i < numElements; ++i)
        {
            stream >> buffer[i];
        }
    }

    std::istream& operator>>(std::istream& stdStream, DictionaryValue& us)
    {
        BinaryIStreamWrapper stream(stdStream);
        size_t version;
        stream >> version;
        
        unsigned int type;
        stream >> type;
        us.m_valueType = static_cast<DictionaryValue::Type>(type);

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream >> us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::Int:
            stream >> us.m_data.m_int;
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
        case DictionaryValue::Type::String:
        {
            std::wstring* strPtr = new std::wstring();
            stream >> *strPtr;
            us.m_data.m_ptr = strPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            size_t size;
            stream >> size;
            NDShape* shapePtr = new NDShape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shapePtr->operator[](i);
            }
            us.m_data.m_ptr = shapePtr;
            break;
        }
        case DictionaryValue::Type::Axis:
        {
            int staticAxisIdx;
            stream >> staticAxisIdx;

            std::wstring axisName;
            stream >> axisName;

            bool isOrderedDynamicAxis;
            stream >> isOrderedDynamicAxis;

            Axis* axisPtr = nullptr;
            if (Axis(staticAxisIdx).IsStaticAxis())
                axisPtr = new Axis(staticAxisIdx);
            else
                axisPtr = new Axis(axisName, isOrderedDynamicAxis);

            us.m_data.m_ptr = axisPtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {   
            size_t size;
            stream >> size;
            std::vector<DictionaryValue>* vectorPtr = new std::vector<DictionaryValue>(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> vectorPtr->at(i);
            }
            us.m_data.m_ptr = vectorPtr;
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = new Dictionary();
            stream >> *dictPtr;
            us.m_data.m_ptr = dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            unsigned int type;
            stream >> type;
            DataType dtype = static_cast<DataType>(type);

            size_t size;
            stream >> size;
            NDShape shape(size);
            for (auto i = 0; i < size; i++)
            {
                stream >> shape[i];
            }

            NDArrayView* viewPtr = new NDArrayView(dtype, shape, DeviceDescriptor::CPUDevice());
            switch (dtype)
            {
            case DataType::Float:
                Read<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Read<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dtype));
            }

            us.m_data.m_ptr = viewPtr;
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream& stdStream, const DictionaryValue& us)
    {
        BinaryOStreamWrapper stream(stdStream);

        stream << us.version;

        stream << static_cast<unsigned int>(us.ValueType());

        switch (us.ValueType())
        {
        case DictionaryValue::Type::Bool:
            stream << us.m_data.m_boolean;
            break;
        case DictionaryValue::Type::Int:
            stream << us.m_data.m_int;
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
        case DictionaryValue::Type::String:
        {
            std::wstring* stringPtr = reinterpret_cast<std::wstring*>(us.m_data.m_ptr);
            stream << *stringPtr;
            break;
        }
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr = reinterpret_cast<NDShape*>(us.m_data.m_ptr);
            stream << *shapePtr;
            break;
        }
        case DictionaryValue::Type::Axis:
        {
            Axis* axisPtr = reinterpret_cast<Axis*>(us.m_data.m_ptr);
            stream << *axisPtr;
            break;
        }
        case DictionaryValue::Type::Vector:
        {
            std::vector<DictionaryValue>* vectorPtr =
                reinterpret_cast<std::vector<DictionaryValue>*>(us.m_data.m_ptr);
            auto size = vectorPtr->size();
            stream << size;
            for (auto i = 0; i < size; i++)
            {
                stream << vectorPtr->at(i);
            }
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr = reinterpret_cast<Dictionary*>(us.m_data.m_ptr);
            stream << *dictPtr;
            break;
        }
        case DictionaryValue::Type::NDArrayView:
        {
            NDArrayView* viewPtr = reinterpret_cast<NDArrayView*>(us.m_data.m_ptr);
            stream << static_cast<unsigned int>(viewPtr->GetDataType());
            stream << viewPtr->Shape();
            switch (viewPtr->GetDataType())
            {
            case DataType::Float:
                Write<float>(stream, *viewPtr);
                break;
            case DataType::Double:
                Write<double>(stream, *viewPtr);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(viewPtr->GetDataType()));
            }
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream& stdStream, const Dictionary& us)
    {
        BinaryOStreamWrapper stream(stdStream);
        stream << us.version;
        stream << us.m_dictionaryData->size();
        for (auto& kv : *(us.m_dictionaryData))
        {
            stream << kv.first;
            stream << kv.second;
        }
        return stream;
    }

    std::istream& operator>>(std::istream& stdStream, Dictionary& us)
    {
        BinaryIStreamWrapper stream(stdStream);
        size_t version;
        stream >> version;
        size_t size;
        stream >> size;
        us.m_dictionaryData->reserve(size);
        for (auto i = 0; i < size; i++)
        {
            std::wstring key;
            stream >> key;
            stream >> us[key];
        }
        return stream;
    }
}