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

    _Internal::_SimpleVector<DictionaryValue> SerializeToVector(const NDArrayViewPtr& viewPtr);

    void DeserializeFromVector(const NDArrayViewPtr& viewPtr, const _Internal::_SimpleVector<DictionaryValue>& values);
}
