//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include "CommonMatrix.h"
#include "TensorShape.h"
#include <string>
#include "Config.h"
#include "Reader.h"
#include "ConvolutionEngine.h"

namespace CNTK
{
    // Forward declarations
    class Dictionary;

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
        if (device.Type() == DeviceKind::CPU)
            return -1;
        else if (device.Type() == DeviceKind::GPU)
            return device.Id();
        else
            NOT_IMPLEMENTED;
    }

    inline Microsoft::MSR::CNTK::MatrixFormat AsCNTKImplMatrixFormat(StorageFormat storageFormat)
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

    inline NDShape AsNDShape(const Microsoft::MSR::CNTK::TensorShape& tensorShape)
    {
        // The TensorShape should be flattenable to 1D
        for (size_t i = 1; i < tensorShape.GetRank(); ++i)
        {
            if (!tensorShape.CanFlatten(i))
                InvalidArgument("AsNDShape() can only be called for TensorShapes that can be flattened to 1D");
        }

        return std::vector<size_t>(tensorShape.GetDims().begin(), tensorShape.GetDims().end());
    }

    inline DataType AsDataType(Microsoft::MSR::CNTK::ElementType readerDataType)
    {
        switch (readerDataType)
        {
        case Microsoft::MSR::CNTK::ElementType::tfloat:
            return DataType::Float;
        case Microsoft::MSR::CNTK::ElementType::tdouble:
            return DataType::Double;
        default:
            LogicError("Unsupported ElementType from CNTK Reader");
        }
    }

    inline StorageFormat AsStorageFormat(Microsoft::MSR::CNTK::StorageType readerStorageType)
    {
        switch (readerStorageType)
        {
        case Microsoft::MSR::CNTK::StorageType::dense:
            return StorageFormat::Dense;
        case Microsoft::MSR::CNTK::StorageType::sparse_csc:
            return StorageFormat::SparseCSC;
        default:
            LogicError("Unsupported StorageType from CNTK Reader");
        }
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorShape(const NDShape& viewShape, bool preserveRank = false)
    {
        const size_t maxNumAxesSupportedByTensorView = 12;
        if (viewShape.NumAxes() > maxNumAxesSupportedByTensorView)
            LogicError("The number of requested axes exceeds the currently supported limit");

        // TensorShape is required to be at least 2D
        size_t minRankSize = preserveRank ? viewShape.NumAxes() : 2;
        Microsoft::MSR::CNTK::SmallVector<size_t> tensorViewShape(std::max<size_t>(minRankSize, viewShape.NumAxes()));
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

    inline bool IsSparseInput(const Variable& var)
    {
        return var.IsInput() && var.IsSparse();
    }


    inline void AddIndentation(std::wstringstream& s, size_t numIndentationSpaces)
    {
        for (size_t i = 0; i < numIndentationSpaces; ++i)
            s << L" ";
    }

    static const size_t perLevelIndentSize = 4;
    inline void AddConfigString(std::wstringstream& s, const std::wstring& key, const DictionaryValue& value, size_t numIndentationSpaces);
    inline void AddConfigString(std::wstringstream& s, const DictionaryValue& value, size_t numIndentationSpaces)
    {
        switch (value.ValueType())
        {
        case DictionaryValue::Type::Bool:
            s << value.Value<bool>();
            break;
        case DictionaryValue::Type::Float:
            s << value.Value<float>();
            break;
        case DictionaryValue::Type::Double:
            s << value.Value<double>();
            break;
        case DictionaryValue::Type::String:
            s << value.Value<std::wstring>();
            break;
        case DictionaryValue::Type::SizeT:
            s << value.Value<size_t>();
            break;
        case DictionaryValue::Type::Vector:
        {
            const auto& valueVector = value.Value<std::vector<DictionaryValue>>();
            s << L"(" << std::endl;
            AddIndentation(s, numIndentationSpaces + perLevelIndentSize);
            bool isFirst = true;
            for (const auto& val : valueVector)
            {
                if (!isFirst)
                    s << L":";
                else
                    isFirst = false;

                AddConfigString(s, val, numIndentationSpaces + perLevelIndentSize);
            }
            AddIndentation(s, numIndentationSpaces);
            s << L")";
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            const auto& valueDictionary = value.Value<Dictionary>();
            s << L"[" << std::endl;
            for (const auto& keyValuePair : *(valueDictionary.m_dictionaryData))
            {
                AddConfigString(s, keyValuePair.first, keyValuePair.second, numIndentationSpaces + perLevelIndentSize);
            }
            AddIndentation(s, numIndentationSpaces);
            s << L"]";
            break;
        }
        default:
            LogicError("Unsupported DictionaryValue type");
        }
    }

    inline void AddConfigString(std::wstringstream& s, const std::wstring& key, const DictionaryValue& value, size_t numIndentationSpaces)
    {
        static const size_t perLevelIndentSize = 4;

        AddIndentation(s, numIndentationSpaces);
        s << key << L" = ";
        AddConfigString(s, value, numIndentationSpaces);
        s << std::endl;
    }

    template <typename T>
    inline std::vector<DictionaryValue> AsDictionaryValueVector(const std::vector<T>& basicElementTypeVector)
    {
        static_assert(std::is_same<T, bool>::value ||
                      std::is_same<T, size_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, std::wstring>::value, "Unsupported ValueType");

        std::vector<DictionaryValue> dictionaryValueVector;
        for (auto value : basicElementTypeVector)
            dictionaryValueVector.push_back(value);

        return dictionaryValueVector;
    }

    template <typename T>
    inline std::vector<T> AsBasicElementTypeVector(const std::vector<DictionaryValue>& dictionaryValueVector)
    {
        static_assert(std::is_same<T, bool>::value ||
            std::is_same<T, size_t>::value ||
            std::is_same<T, float>::value ||
            std::is_same<T, double>::value ||
            std::is_same<T, std::wstring>::value, "Unsupported ValueType");

        std::vector<T> basicElementTypeVector;
        for (auto value : dictionaryValueVector)
            basicElementTypeVector.push_back(value.Value<T>());

        return basicElementTypeVector;
    }

    inline PoolingType AsPoolingType(Microsoft::MSR::CNTK::PoolKind cntkPoolingKind)
    {
        switch (cntkPoolingKind)
        {
        case Microsoft::MSR::CNTK::PoolKind::Average:
            return PoolingType::Average;
        case Microsoft::MSR::CNTK::PoolKind::Max:
            return PoolingType::Max;
        default:
            LogicError("Unknown pooling type");
        }
    }

    inline Microsoft::MSR::CNTK::PoolKind AsCNTKPoolKind(PoolingType poolingType)
    {
        switch (poolingType)
        {
        case PoolingType::Average:
            return Microsoft::MSR::CNTK::PoolKind::Average;
        case PoolingType::Max:
            return Microsoft::MSR::CNTK::PoolKind::Max;
        default:
            LogicError("Unknown pooling type");
        }
    }

    inline Axis AsAxis(size_t CNTKInternalAxisIdx)
    {
        if (CNTKInternalAxisIdx == 0)
            LogicError("CNTK internal axis indices must be > 0");

        return Axis(CNTKInternalAxisIdx - 1);
    }

    inline int AsCNTKInternalAxisIdx(const Axis& axis)
    {
        if (!axis.IsStaticAxis())
            LogicError("Only Axis that represent static indices can be converted to a CNTK internal axis index");

        return (int)(axis.StaticAxisIndex() + 1);
    }

    inline std::pair<NDShape, NDShape> GetConvolutionOutputMapCountAndKernelShape(const NDShape& convolutionMapShape, const NDShape& operandShape)
    {
        auto outputMapCount = convolutionMapShape.SubShape(0, convolutionMapShape.NumAxes() - operandShape.NumAxes());
        NDShape paddedOutputMapCount(operandShape.NumAxes(), 1);
        for (size_t i = 0; i < outputMapCount.NumAxes(); ++i)
            paddedOutputMapCount[paddedOutputMapCount.NumAxes() - 1 - i] = outputMapCount[outputMapCount.NumAxes() - 1 - i];
        //for (size_t i = 0; i < outputMapCount.NumAxes(); ++i)
        //    paddedOutputMapCount[i] = outputMapCount[i];

        NDShape kernelShape = convolutionMapShape.SubShape(outputMapCount.NumAxes());

        return{ paddedOutputMapCount, kernelShape };
    }

    inline CNTK::Constant ScalarConstant(CNTK::DataType dataType, float value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
    {
        if (dataType == CNTK::DataType::Float)
            return CNTK::Constant({}, value, device);
        else if (dataType == CNTK::DataType::Double)
            return CNTK::Constant({}, (double)value, device);
        else
            LogicError("CNTK::ScalarConstant: Unsupported DataType %s", DataTypeName(dataType));
    }

    inline double MomentumPerMB(double momentumPerSample, size_t minibatchSize)
    {
        return std::pow(momentumPerSample, minibatchSize);
    }
}
