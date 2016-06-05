//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "TensorView.h"
#include "Matrix.h"
#include <algorithm>
#include "TensorShape.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    template <typename ElementType>
    static void* AllocateTensorView(const NDShape& viewShape,
                                    const DeviceDescriptor& device,
                                    void* dataBuffer,
                                    size_t bufferSizeInBytes)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<ElementType>> matrix;
        if (dataBuffer == nullptr)
            matrix = std::make_shared<Matrix<ElementType>>(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device));
        else
        {
            if (bufferSizeInBytes < (viewShape.TotalSize() * sizeof(ElementType)))
                InvalidArgument("Size of the specified buffer for creating the NDArrayView is smaller than the specified view shape");

            matrix = std::make_shared<Matrix<ElementType>>(matrixDims.first, matrixDims.second, (ElementType*)dataBuffer, AsCNTKImplDeviceId(device), matrixFlagDontOwnBuffer);
        }

        return new TensorView<ElementType>(matrix, AsTensorShape(viewShape));
    }

    static void* AllocateTensorView(CNTK::DataType dataType,
                                    const NDShape& viewShape,
                                    const DeviceDescriptor& device,
                                    void* dataBuffer,
                                    size_t bufferSizeInBytes)
    {
        switch (dataType)
        {
        case DataType::Float: 
            return AllocateTensorView<float>(viewShape, device, dataBuffer, bufferSizeInBytes);
        case DataType::Double:
            return AllocateTensorView<double>(viewShape, device, dataBuffer, bufferSizeInBytes);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, device, StorageFormat::Dense, viewShape, readOnly, AllocateTensorView(dataType, viewShape, device, dataBuffer, bufferSizeInBytes))
    {
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const DeviceDescriptor& device, CNTK::StorageFormat storageType, const NDShape& viewShape, bool readOnly, void* tensorView)
        : m_dataType(dataType), m_device(device), m_storageFormat(storageType), m_viewShape(viewShape), m_isReadOnly(readOnly), m_tensorView(tensorView)
    {
    }

    void NDArrayView::SetValue(float value)
    {
        GetWritableMatrix<float>()->SetValue(value);
    }

    void NDArrayView::SetValue(double value)
    {
        GetWritableMatrix<double>()->SetValue(value);
    }

    NDArrayView::~NDArrayView()
    {
        switch (m_dataType)
        {
        case DataType::Float:
            delete GetTensorView<float>();
            break;
        case DataType::Double:
            delete GetTensorView<double>();
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
    }

    template <typename ElementType>
    std::shared_ptr<Matrix<ElementType>> NDArrayView::GetMatrixImpl(size_t rowColSplitPoint) const
    {
        const TensorView<ElementType>* tensorView = GetTensorView<ElementType>();
        auto tensorShape = tensorView->GetShape();
        if (tensorShape.GetRank() <= 2)
            return tensorView->AsMatrix();

        size_t splitPoint = rowColSplitPoint;
        if (splitPoint == AutoSelectRowColSplitPoint)
        {
            // Determine the split point
            std::vector<bool> dimsToDrop(tensorShape.GetRank(), false);
            for (size_t k = 1; k < tensorShape.GetRank(); ++k)
                if (tensorShape.CanFlatten(k))
                    dimsToDrop[k - 1] = true;

            // There should be at most 2 dims we cannot drop
            auto numDimsThatCannotBeDropped = std::count_if(dimsToDrop.begin(), dimsToDrop.end(), [](const bool& val) {
                return !val;
            });

            if (numDimsThatCannotBeDropped > 2)
                LogicError("The TensorView underlying this NDArrayView cannot be flattened to a Matrix");

            splitPoint = 0;
            while (dimsToDrop[splitPoint])
                splitPoint++;
        }

        tensorShape.FlattenTo2DInPlace(splitPoint, "NDArrayView::GetMatrix");

        return tensorView->Reshaped(tensorShape).AsMatrix();
    }

    template <typename ElementType>
    std::shared_ptr<const Matrix<ElementType>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        return GetMatrixImpl<ElementType>(rowColSplitPoint);
    }

    template <typename ElementType>
    std::shared_ptr<Matrix<ElementType>> NDArrayView::GetWritableMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        if (IsReadOnly())
            LogicError("NDArrayView::GetWritableMatrix: Cannot get writable Matrix from a read-only NDArrayView");

        return GetMatrixImpl<ElementType>(rowColSplitPoint);
    }

    template <typename ElementType>
    const TensorView<ElementType>* NDArrayView::GetTensorView() const
    {
        if (GetDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetWritableTensorView: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        return (const TensorView<ElementType>*)(m_tensorView);
    }

    template <typename ElementType>
    TensorView<ElementType>* NDArrayView::GetWritableTensorView()
    {
        if (IsReadOnly())
            LogicError("NDArrayView::GetWritableTensorView: Cannot get writable TensorView from a read-only NDArrayView");

        return const_cast<TensorView<ElementType>*>(GetTensorView<ElementType>());
    }

    NDArrayViewPtr NDArrayView::DeepClone(bool readOnly/* = false*/) const
    {
        NDArrayViewPtr newView(new NDArrayView(this->DataType(), this->Shape(), nullptr, 0, this->Device(), readOnly), [](_ReferenceCounter* ptr) { delete ptr; });
        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto newMatrix = newView->GetWritableMatrix<float>();
            auto thisMatrix = GetMatrix<float>();
            newMatrix->AssignValuesOf(*thisMatrix);
            break;
        }
        case DataType::Double:
        {
            auto newMatrix = newView->GetWritableMatrix<double>();
            auto thisMatrix = GetMatrix<double>();
            newMatrix->AssignValuesOf(*thisMatrix);
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return NDArrayViewPtr(newView, [](_ReferenceCounter* ptr) {
            delete ptr;
        });
    }

    void NDArrayView::CopyFrom(const NDArrayView& source)
    {
        if (source.Shape() != Shape())
            InvalidArgument("NDArrayView::CopyFrom: The 'source' view's shape must be same as the shape of this NDArrayView");

        if (IsReadOnly())
            RuntimeError("NDArrayView::CopyFrom: Cannot modify contents of a readonly NDArrayView");

        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto sourceMatrix = source.GetMatrix<float>();
            auto destMatrix = GetWritableMatrix<float>();
            destMatrix->AssignValuesOf(*sourceMatrix);
            break;
        }
        case DataType::Double:
        {
            auto sourceMatrix = source.GetMatrix<double>();
            auto destMatrix = GetWritableMatrix<double>();
            destMatrix->AssignValuesOf(*sourceMatrix);
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
    }

    NDArrayViewPtr NDArrayView::Alias(bool readOnly/* = false*/) const
    {
        void* tensorView = nullptr;
        switch (m_dataType)
        {
        case DataType::Float:
            tensorView = new TensorView<float>(*(GetTensorView<float>()));
            break;
        case DataType::Double:
            tensorView = new TensorView<double>(*(GetTensorView<double>()));
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        auto aliasView = new NDArrayView(DataType(), Device(), StorageFormat(), Shape(), IsReadOnly() || readOnly, tensorView);;
        return NDArrayViewPtr(aliasView, [](_ReferenceCounter* ptr) { delete ptr; });
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    ElementType* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            LogicError("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView");

        return const_cast<ElementType*>(DataBuffer<ElementType>());
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    const ElementType* NDArrayView::DataBuffer() const
    {
        if (GetDataType<ElementType>() != m_dataType)
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        // First make sure that the underlying matrix is on the right device
        auto matrix = GetMatrix<ElementType>();
        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        return matrix->Data();
    }

    template <typename ElementType>
    NDArrayViewPtr NDArrayView::RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::DefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomUniformMatrix = std::make_shared<Matrix<ElementType>>(Matrix<ElementType>::RandomUniform(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (ElementType)rangeStart, (ElementType)rangeEnd, seed));
        auto tensorView = new TensorView<ElementType>(randomUniformMatrix, AsTensorShape(shape));

        auto view = new NDArrayView(GetDataType<ElementType>(), device, StorageFormat::Dense, shape, false, tensorView);
        return NDArrayViewPtr(view, [](_ReferenceCounter* ptr) { delete ptr; });
    }

    // Explicit template instantiations
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float>(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::DefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<double>(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::DefaultDevice()*/);

    template CNTK_API const float* NDArrayView::DataBuffer<float>() const;
    template CNTK_API const double* NDArrayView::DataBuffer<double>() const;

    template CNTK_API float* NDArrayView::WritableDataBuffer<float>();
    template CNTK_API double* NDArrayView::WritableDataBuffer<double>();
}
