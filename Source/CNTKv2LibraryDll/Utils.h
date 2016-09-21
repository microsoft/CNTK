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

    inline NDShape AsNDShape(const Microsoft::MSR::CNTK::TensorShape& tensorShape, bool allowNonFlattenableTensorShapes = false)
    {
        if (!allowNonFlattenableTensorShapes)
        {
            // The TensorShape should be flattenable to 1D
            for (size_t i = 1; i < tensorShape.GetRank(); ++i)
            {
                if (!tensorShape.CanFlatten(i))
                    InvalidArgument("AsNDShape() can only be called for TensorShapes that can be flattened to 1D");
            }
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

    inline Microsoft::MSR::CNTK::TensorShape AsTensorShape(const NDShape& viewShape)
    {
        const size_t maxNumAxesSupportedByTensorView = 12;
        if (viewShape.Rank() > maxNumAxesSupportedByTensorView)
            LogicError("The number of requested axes exceeds the currently supported limit");

        // TensorShape is required to be at least 1D
        size_t minRankSize = 1;
        Microsoft::MSR::CNTK::SmallVector<size_t> tensorViewShape(std::max<size_t>(minRankSize, viewShape.Rank()));
        for (size_t i = 0; i < tensorViewShape.size(); ++i)
            tensorViewShape[i] = (i < viewShape.Rank()) ? viewShape[i] : 1;

        return tensorViewShape;
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorViewShape(const Microsoft::MSR::CNTK::TensorShape& viewShape)
    {
        // For TensorView shapes we pad the TensorShape to be at least rank 2
        return viewShape.PadRank(std::max<size_t>(2, viewShape.GetRank()));
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorViewShape(const NDShape& viewShape)
    {
        return AsTensorViewShape(AsTensorShape(viewShape));
    }

    inline std::wstring AsStringForErrorReporting(const NDShape& shape)
    {
        bool invertShape = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
        auto displayShape = shape;
        if (invertShape)
        {
            for (size_t i = 0, j = shape.Rank() - 1; i < shape.Rank(); ++i, --j)
                displayShape[i] = shape[j];
        }

        return displayShape.AsString();
    }

    inline std::pair<size_t, size_t> GetMatrixDimensions(const NDShape& viewShape)
    {
        // Ensure none of the shape dimensions are unknown
        if (viewShape.HasInferredDimension())
            InvalidArgument("Cannot create an NDArrayView using a view shape that has unknown dimensions for any of it's axes!");

        size_t matrixRowSize = (viewShape.Rank() > 0) ? viewShape[0] : 1;
        size_t matrixColSize = (viewShape.Rank() > 0) ? viewShape.SubShape(1).TotalSize() : 1;

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
    inline std::vector<DictionaryValue> AsDictionaryValueVector(const std::vector<T>& elementVector)
    {
        static_assert(std::is_same<T, bool>::value ||
                      std::is_same<T, size_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, Axis>::value ||
                      std::is_same<T, std::wstring>::value,
                      "Unsupported ValueType");

        std::vector<DictionaryValue> dictionaryValueVector;
        for (auto value : elementVector)
            dictionaryValueVector.push_back(value);

        return dictionaryValueVector;
    }

    template <typename T>
    inline std::vector<T> AsVector(const std::vector<DictionaryValue>& dictionaryValueVector)
    {
        static_assert(std::is_same<T, bool>::value ||
                      std::is_same<T, size_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, Axis>::value ||
                      std::is_same<T, std::wstring>::value,
                      "Unsupported ValueType");

        std::vector<T> elementVector;
        for (auto value : dictionaryValueVector)
            elementVector.push_back(value.Value<T>());

        return elementVector;
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
        auto outputMapCount = convolutionMapShape.SubShape(0, convolutionMapShape.Rank() - operandShape.Rank());
        NDShape paddedOutputMapCount(operandShape.Rank(), 1);
        for (size_t i = 0; i < outputMapCount.Rank(); ++i)
            paddedOutputMapCount[paddedOutputMapCount.Rank() - 1 - i] = outputMapCount[outputMapCount.Rank() - 1 - i];
        //for (size_t i = 0; i < outputMapCount.Rank(); ++i)
        //    paddedOutputMapCount[i] = outputMapCount[i];

        NDShape kernelShape = convolutionMapShape.SubShape(outputMapCount.Rank());

        return{ paddedOutputMapCount, kernelShape };
    }

    inline double MomentumPerMB(double momentumPerSample, size_t minibatchSize)
    {
        return std::pow(momentumPerSample, minibatchSize);
    }

    template <typename SourceElementType, typename TargetElementType>
    inline TargetElementType* Copy(const SourceElementType* src, size_t srcSize)
    {
        // Cast to double
        TargetElementType* castValue = new TargetElementType[srcSize];
        for (size_t i = 0; i < srcSize; ++i)
            castValue[i] = (TargetElementType)src[i];

        return castValue;
    }

    inline NDArrayViewPtr CloneAsDataType(const NDArrayViewPtr& source, DataType targetDataType, bool readOnly)
    {
        if (source->Device() != DeviceDescriptor::CPUDevice())
            LogicError("CloneAsDataType currently does not support non-CPU source NDArrayView objects");

        auto sourceDataType = source->GetDataType();
        if (sourceDataType == targetDataType)
            LogicError("CloneAsDataType: Source and target DataTypes are same");

        if (targetDataType != DataType::Double)
            LogicError("CloneAsDataType: Only Double target DataType is supported");

        auto sourceShape = source->Shape();
        auto sourceSize = sourceShape.TotalSize();

        // Cast to double
        double* castValue = Copy<float, double>(source->DataBuffer<float>(), sourceSize);
        return MakeSharedObject<NDArrayView>(sourceShape, castValue, sourceSize, DeviceDescriptor::CPUDevice(), readOnly);
    }
}
