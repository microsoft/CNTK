//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include "CommonMatrix.h"
#include "TensorShape.h"
#include <string>

namespace CNTK
{
    // Forward declarations
    class Dictionary;

    class DictionaryValue
    {
    public:
        enum class Type : unsigned int
        {
            None,
            Bool,
            SizeT,
            Double,
            NDShape,
            Vector
        };

        static const char* TypeName(Type type)
        {
            if (type == Type::None)
                return "None";
            else if (type == Type::Bool)
                return "Bool";
            else if (type == Type::SizeT)
                return "SizeT";
            else if (type == Type::Double)
                return "Double";
            else if (type == Type::NDShape)
                return "NDShape";
            else if (type == Type::Vector)
                return "Vector";
            else
                LogicError("Unknown DictionaryValue::Type");
        }

    public:
        DictionaryValue()
            : m_valueType(Type::None)
        {
        }

        DictionaryValue(bool value)
            : m_valueType(GetValueType<bool>())
        {
            m_data.m_boolean = value;
        }

        DictionaryValue(size_t value)
            : m_valueType(GetValueType<size_t>())
        {
            m_data.m_sizeT = value;
        }

        DictionaryValue(double value)
            : m_valueType(GetValueType<double>())
        {
            m_data.m_double = value;
        }

        template <typename T>
        DictionaryValue(const T& value)
            : m_valueType(GetValueType<T>())
        {
            static_assert(std::is_same<T, NDShape>::value ||
                std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value,
                "Unsupported ValueType");

            AllocateDataPtr(value);
        }

        DictionaryValue(const DictionaryValue& other)
            : m_valueType(Type::Bool)
        {
            *this = other;
        }

        DictionaryValue& operator=(const DictionaryValue& other)
        {
            if (this != &other)
            {
                FreeDataPtr();

                m_valueType = other.m_valueType;
                m_data = other.m_data;

                if (other.m_valueType == Type::NDShape)
                    AllocateDataPtr(other.GetValue<NDShape>());
                else if (other.m_valueType == Type::Vector)
                    AllocateDataPtr(other.GetValue<_Internal::_SimpleVector<DictionaryValue>>());
            }

            return *this;
        }

        ~DictionaryValue()
        {
            FreeDataPtr();
        }

        template <typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_boolean;
        }

        template <typename T, typename std::enable_if<std::is_same<T, size_t>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_sizeT;
        }

        template <typename T, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_double;
        }

        template <typename T, typename std::enable_if<std::is_same<T, NDShape>::value || std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return *(reinterpret_cast<T*>(m_data.m_ptr));
        }

        bool HasValue() const
        {
            return m_valueType != Type::None;
        }

        Type ValueType() const
        {
            return m_valueType;
        }

    private:
        template <typename T>
        static Type GetValueType()
        {
            static_assert(std::is_same<T, bool>::value ||
                std::is_same<T, size_t>::value ||
                std::is_same<T, double>::value ||
                std::is_same<T, NDShape>::value ||
                std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value ||
                std::is_same<T, CNTK::Dictionary>::value,
                "Unsupported ValueType");

            if (std::is_same<T, bool>::value)
                return Type::Bool;
            else if (std::is_same<T, size_t>::value)
                return Type::SizeT;
            else if (std::is_same<T, double>::value)
                return Type::Double;
            else if (std::is_same<T, NDShape>::value)
                return Type::NDShape;
            else if (std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value)
                return Type::Vector;
        }

        template <typename T>
        void VerifyType() const
        {
            if (GetValueType<T>() != m_valueType)
                RuntimeError("Reading a DictionaryValue as the wrong type; Reading as type %s when actual type is %s", typeid(T).name(), DictionaryValue::TypeName(m_valueType));
        }

        template <typename T>
        void AllocateDataPtr(const T& value)
        {
            static_assert(std::is_same<T, NDShape>::value || std::is_same<T, _Internal::_SimpleVector<DictionaryValue>>::value, "AllocateDataPtr called with invalid type");
            m_data.m_ptr = new T(value);
        }

        template <typename T>
        void FreePtrAsType()
        {
            T* typedPtr = reinterpret_cast<T*>(m_data.m_ptr);
            delete typedPtr;

            m_data.m_ptr = nullptr;
        }

        void FreeDataPtr()
        {
            if (m_valueType == Type::NDShape)
                FreePtrAsType<NDShape>();
            else if (m_valueType == Type::Vector)
                FreePtrAsType<_Internal::_SimpleVector<DictionaryValue>>();
        }

    private:
        Type m_valueType;

        union ValueData
        {
            bool m_boolean;
            size_t m_sizeT;
            double m_double;
            void* m_ptr;
        } m_data;
    };

    class Dictionary
    {
    public:
        Dictionary();
        ~Dictionary();

        // Disallow copy contruction and assignment
        Dictionary(const Dictionary&) = delete;
        Dictionary& operator=(const Dictionary&) = delete;

        Dictionary(Dictionary&& other);
        Dictionary& operator=(Dictionary&& other);

        DictionaryValue& operator[](const std::wstring& key)
        {
            return operator[](key.c_str());
        }

        DictionaryValue& operator[](const wchar_t* key);

        DictionaryValue operator[](const std::wstring& key) const
        {
            return operator[](key.c_str());
        }

        DictionaryValue operator[](const wchar_t* key) const;

        bool Contains(const std::wstring& key) const
        {
            return Contains(key.c_str());
        }

        bool Contains(const wchar_t* key) const;

    private:
        std::unordered_map<std::wstring, DictionaryValue>* m_dictionaryData;
    };

    // Helper to get the size of an element of the specified DataType
    inline size_t ElementSize(DataType dataType)
    {
        if (dataType == DataType::Float)
            return sizeof(float);
        else if (dataType == DataType::Double)
            return sizeof(double);
        else
            NOT_IMPLEMENTED;
    }

    inline DEVICEID_TYPE AsCNTKImplDeviceId(const DeviceDescriptor& device)
    {
        if (device.Type() == DeviceType::CPU)
            return -1;
        else if (device.Type() == DeviceType::GPU)
            return device.Id();
        else
            NOT_IMPLEMENTED;
    }

    inline Microsoft::MSR::CNTK::MatrixFormat AsCNTKMatrixFormat(StorageFormat storageFormat)
    {
        if (storageFormat == StorageFormat::Dense)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatDense;
        else if (storageFormat == StorageFormat::SparseCSC)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC;
        else if (storageFormat == StorageFormat::SparseBlockCol)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseBlockCol;
        else
            NOT_IMPLEMENTED;
    }

    inline StorageFormat AsStorageFormat(Microsoft::MSR::CNTK::MatrixFormat matrixFormat)
    {
        if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatDense)
            return StorageFormat::Dense;
        else if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC)
            return StorageFormat::SparseCSC;
        else if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseBlockCol)
            return StorageFormat::SparseBlockCol;
        else
            NOT_IMPLEMENTED;
    }

    inline DeviceDescriptor AsDeviceDescriptor(DEVICEID_TYPE deviceId)
    {
        if (deviceId == CPUDEVICE)
            return DeviceDescriptor::CPUDevice();
        else
            return DeviceDescriptor::GPUDevice(deviceId);
    }

    inline const char* DataTypeName(DataType dataType)
    {
        if (dataType == DataType::Float)
            return "Float";
        else if (dataType == DataType::Double)
            return "Double";
        else
            LogicError("Unknown DataType");
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorShape(const NDShape& viewShape)
    {
        const size_t maxNumAxesSupportedByTensorView = 12;
        if (viewShape.NumAxes() > maxNumAxesSupportedByTensorView)
            LogicError("The number of requested axes exceeds the currently supported limit");

        // TensorShape is required to be at least 2D
        Microsoft::MSR::CNTK::SmallVector<size_t> tensorViewShape(std::max<size_t>(2, viewShape.NumAxes()));
        for (size_t i = 0; i < tensorViewShape.size(); ++i)
            tensorViewShape[i] = (i < viewShape.NumAxes()) ? viewShape[i] : 1;

        return tensorViewShape;
    }

    inline std::string AsString(const NDShape& shape)
    {
        std::string shapeString = "[";
        bool notIsFirst = false;
        for (size_t i = 0; i < shape.NumAxes(); ++i)
        {
            if (notIsFirst)
                shapeString += ", ";

            shapeString += std::to_string(shape[i]);
            notIsFirst = true;
        }

        return shapeString + "]";
    }

    inline std::pair<size_t, size_t> GetMatrixDimensions(const NDShape& viewShape)
    {
        // Ensure none of the shape dimensions are unknown
        if (viewShape.HasInferredDimension())
            InvalidArgument("Cannot create an NDArrayView using a view shape that has unknown dimensions for any of it's axes!");

        size_t matrixRowSize = (viewShape.NumAxes() > 0) ? viewShape[0] : 1;
        size_t matrixColSize = (viewShape.NumAxes() > 0) ? viewShape.SubShape(1).TotalSize() : 1;

        return{ matrixRowSize, matrixColSize };
    }
}
