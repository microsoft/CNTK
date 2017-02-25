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
    static TensorView<ElementType>* AllocateTensorView(const NDShape& viewShape,
                                                       const DeviceDescriptor& device,
                                                       void* dataBuffer,
                                                       size_t bufferSizeInBytes)
    {
        if (dataBuffer == nullptr)
            InvalidArgument("Cannot create a NDArrayView over a null data buffer");

        if (bufferSizeInBytes < (viewShape.TotalSize() * sizeof(ElementType)))
            InvalidArgument("Size of the specified buffer for creating the NDArrayView is smaller than the specified view shape");

        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<ElementType>> matrix = std::make_shared<Matrix<ElementType>>(matrixDims.first, matrixDims.second, (ElementType*)dataBuffer, AsCNTKImplDeviceId(device), matrixFlagDontOwnBuffer);
        return new TensorView<ElementType>(matrix, AsTensorViewShape(viewShape));
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

    template <typename ElementType>
    static TensorView<ElementType>* AllocateTensorView(const NDShape& viewShape,
                                                       CNTK::StorageFormat storageType,
                                                       const DeviceDescriptor& device)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<ElementType>> matrix = std::make_shared<Matrix<ElementType>>(matrixDims.first,
                                                                                            matrixDims.second,
                                                                                            AsCNTKImplDeviceId(device),
                                                                                            IsSparseStorageFormat(storageType) ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                                            AsCNTKImplMatrixFormat(storageType));
        return new TensorView<ElementType>(matrix, AsTensorViewShape(viewShape));
    }

    static void* AllocateTensorView(CNTK::DataType dataType,
                                    CNTK::StorageFormat storageType,
                                    const NDShape& viewShape,
                                    const DeviceDescriptor& device)
    {
        switch (dataType)
        {
        case DataType::Float:
            return AllocateTensorView<float>(viewShape, storageType, device);
        case DataType::Double:
            return AllocateTensorView<double>(viewShape, storageType, device);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, device, StorageFormat::Dense, viewShape, readOnly, AllocateTensorView(dataType, viewShape, device, dataBuffer, bufferSizeInBytes))
    {
    }

    template <typename ElementType>
    NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(AsDataType<ElementType>(), device, StorageFormat::SparseCSC, viewShape, false, AllocateTensorView<ElementType>(viewShape, StorageFormat::SparseCSC, device))
    {
        if ((colStarts == nullptr) || (rowIndices == nullptr) || (nonZeroValues == nullptr) || (numNonZeroValues == 0) || (numNonZeroValues > viewShape.TotalSize()))
            InvalidArgument("Invalid sparse CSC format initial data specified for NDArrayView construction");

        auto sparseMatrix = GetWritableMatrix<ElementType>(1);
        sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, nonZeroValues, numNonZeroValues, sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
        m_isReadOnly = readOnly;
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const DeviceDescriptor& device, CNTK::StorageFormat storageType, const NDShape& viewShape, bool readOnly, void* tensorView)
        : m_dataType(dataType), m_device(device), m_storageFormat(storageType), m_viewShape(viewShape), m_isReadOnly(readOnly)
    {
        m_tensorView = std::shared_ptr<void>(tensorView, [this](void*) {
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
        });
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device)
        : NDArrayView(dataType, device, storageType, viewShape, false, AllocateTensorView(dataType, storageType, viewShape, device))
    {
    }

    NDArrayView::~NDArrayView()
    {
    }

    void NDArrayView::SetValue(float value)
    {
        if (IsSparse())
            LogicError("Filling a NDArrayView with a scalar is only allowed for NDArrayView objects with dense storage format");

        GetWritableMatrix<float>()->SetValue(value);
    }

    void NDArrayView::SetValue(double value)
    {
        if (IsSparse())
            LogicError("Filling a NDArrayView with a scalar is only allowed for NDArrayView objects with dense storage format");

        GetWritableMatrix<double>()->SetValue(value);
    }

    template <typename ElementType>
    /*static*/ std::shared_ptr<Matrix<ElementType>> NDArrayView::GetMatrixImpl(const TensorView<ElementType>* tensorView, size_t rowColSplitPoint)
    {
        auto tensorShape = tensorView->GetShape();
        if (tensorShape.GetRank() <= 2)
            return tensorView->AsMatrix();

        size_t splitPoint = rowColSplitPoint;
        if (splitPoint == NDArrayView::AutoSelectRowColSplitPoint)
        {
            // Determine the split point by determining which of the axes can be 
            // folded and selecting the non-foldable axis as the split point
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

            // If we can fold the entire tensor down to a vector so any of the axes can be a valid split point,
            // let's pick the split point to be 1
            splitPoint = 1;
            if (numDimsThatCannotBeDropped > 1)
            {
                while (dimsToDrop[splitPoint - 1])
                    splitPoint++;
            }
        }

        tensorShape.FlattenTo2DInPlace(splitPoint, "NDArrayView::GetMatrix");

        return tensorView->Reshaped(tensorShape).AsMatrix();
    }

    template <typename ElementType>
    std::shared_ptr<const Matrix<ElementType>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        return GetMatrixImpl<ElementType>(GetTensorView<ElementType>(), rowColSplitPoint);
    }

    template <typename ElementType>
    std::shared_ptr<Matrix<ElementType>> NDArrayView::GetWritableMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        return GetMatrixImpl<ElementType>(GetWritableTensorView<ElementType>(), rowColSplitPoint);
    }

    template <typename ElementType>
    const TensorView<ElementType>* NDArrayView::GetTensorView() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetTensorView: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        return (const TensorView<ElementType>*)(m_tensorView.get());
    }

    template <typename ElementType>
    TensorView<ElementType>* NDArrayView::GetWritableTensorView()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::GetWritableTensorView: Cannot get writable TensorView from a read-only NDArrayView");

        return const_cast<TensorView<ElementType>*>(GetTensorView<ElementType>());
    }

    NDArrayViewPtr NDArrayView::DeepClone(const DeviceDescriptor& device, bool readOnly/* = false*/) const
    {
        NDArrayViewPtr newView = MakeSharedObject<NDArrayView>(this->GetDataType(), this->GetStorageFormat(), this->Shape(), device);
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

        newView->m_isReadOnly = readOnly;
        return newView;
    }

    void NDArrayView::CopyFrom(const NDArrayView& source)
    {
        if ((source.Shape() != Shape()) && (AsTensorShape(source.Shape()) != AsTensorShape(Shape())))
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

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), Shape(), IsReadOnly() || readOnly, tensorView);
    }

    NDArrayViewPtr NDArrayView::SliceView(const std::vector<size_t>& startOffset, const std::vector<size_t>& extent, bool readOnly) const
    {
        auto rank = Shape().Rank();
        if (startOffset.size() != rank)
            InvalidArgument("NDArrayView::SliceView: Rank of the NDArrayView (%d) does not match the dimensionality of the specified slice offset (%d)", (int)rank, (int)startOffset.size());

        if (extent.size() > rank)
            InvalidArgument("NDArrayView::SliceView: Dimensionality of the specified slice extent (%d) exceeds the rank of this NDArrayView (%d)", (int)extent.size(), (int)rank);

        if (std::find(extent.begin(), extent.end(), 0) != extent.end())
            InvalidArgument("NDArrayView::SliceView: Specified slice extent contains a zero in at least one axes");

        bool anyPrevAxisSliced = false;
        NDShape sliceViewShape(extent);
        std::vector<size_t> endOffset(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            if ((i < sliceViewShape.Rank()) && (sliceViewShape[i] == NDShape::InferredDimension))
                sliceViewShape[i] = Shape()[i] - startOffset[i];

            endOffset[i] = startOffset[i] + ((i < sliceViewShape.Rank()) ? sliceViewShape[i] : 1);

            if (anyPrevAxisSliced && ((endOffset[i] - startOffset[i]) != 1))
                InvalidArgument("NDArrayView::SliceView: Cannot create a slice that is not contiguous in memory");

            bool isCurrentAxisSliced = (startOffset[i] != 0) || (endOffset[i] != Shape()[i]);
            anyPrevAxisSliced = anyPrevAxisSliced || isCurrentAxisSliced;
        }

        auto flatBufferOffset = AsTensorShape(Shape()).Locate(startOffset);
        auto sliceViewMatrixDims = GetMatrixDimensions(sliceViewShape);
        assert((flatBufferOffset % sliceViewMatrixDims.first) == 0);
        auto sliceMatrixColumnOffset = flatBufferOffset / sliceViewMatrixDims.first;
        void* tensorView = nullptr;
        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto currentMatrix = GetMatrix<float>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                LogicError("NDArrayView::SliceView: Currently only slices that can be realized a slice of the Matrix object underlying this NDArrayView, are allowed");

            auto slicedMatrixView = make_shared<Matrix<float>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));
            tensorView = new TensorView<float>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Double:
        {
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                LogicError("NDArrayView::SliceView: Currently only slices that can be realized a slice of the Matrix object underlying this NDArrayView, are allowed");

            auto slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));
            tensorView = new TensorView<double>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), sliceViewShape, IsReadOnly() || readOnly, tensorView);
    }

    NDArrayViewPtr NDArrayView::AsShape(const NDShape& newShape) const
    {
        if (newShape.TotalSize() != Shape().TotalSize())
        {
            InvalidArgument("NDArrayView::AsShape: The size (%d) of 'source' view shape's (%S) must be same as the size (%d) of the newShape (%S)!",
                (int)Shape().TotalSize(), AsStringForErrorReporting(Shape()).c_str(),
                (int)newShape.TotalSize(), AsStringForErrorReporting(newShape).c_str());
        }

        auto newTensorShape = AsTensorShape(newShape);
        void* tensorView = nullptr;
        switch (m_dataType)
        {
        case DataType::Float:
            tensorView = new TensorView<float>(*(GetTensorView<float>()), newTensorShape);
            break;
        case DataType::Double:
            tensorView = new TensorView<double>(*(GetTensorView<double>()), newTensorShape);
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), newShape, IsReadOnly(), tensorView);
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    ElementType* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView");

        return const_cast<ElementType*>(DataBuffer<ElementType>());
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    const ElementType* NDArrayView::DataBuffer() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (IsSparse())
            InvalidArgument("DataBuffer/WritableDataBuffer methods can only be called for NDArrayiew objects with dense storage format");

        // First make sure that the underlying matrix is on the right device
        auto matrix = GetMatrix<ElementType>();
        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        return matrix->Data();
    }

    void NDArrayView::ChangeDevice(const DeviceDescriptor& device)
    {
        if (device == m_device)
            return;

        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto matrix = GetMatrix<float>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        case DataType::Double:
        {
            auto matrix = GetMatrix<double>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        m_device = device;
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr NDArrayView::RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomNormalMatrix = std::make_shared<Matrix<ElementType>>(Matrix<ElementType>::RandomGaussian(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (ElementType)mean, (ElementType)stdDev, seed));
        auto tensorView = new TensorView<ElementType>(randomNormalMatrix, AsTensorViewShape(shape));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), device, StorageFormat::Dense, shape, false, tensorView);
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomUniformMatrix = std::make_shared<Matrix<ElementType>>(Matrix<ElementType>::RandomUniform(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (ElementType)rangeBegin, (ElementType)rangeEnd, seed));
        auto tensorView = new TensorView<ElementType>(randomUniformMatrix, AsTensorViewShape(shape));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), device, StorageFormat::Dense, shape, false, tensorView);
    }

    // Explicit template instantiations
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<double>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<float>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<double>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API const float* NDArrayView::DataBuffer<float>() const;
    template CNTK_API const double* NDArrayView::DataBuffer<double>() const;

    template CNTK_API float* NDArrayView::WritableDataBuffer<float>();
    template CNTK_API double* NDArrayView::WritableDataBuffer<double>();

    template std::shared_ptr<const Matrix<float>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template std::shared_ptr<const Matrix<double>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;

    template std::shared_ptr<Matrix<float>> NDArrayView::GetWritableMatrix<float>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template std::shared_ptr<Matrix<double>> NDArrayView::GetWritableMatrix<double>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template TensorView<float>* NDArrayView::GetWritableTensorView<float>();
    template TensorView<double>* NDArrayView::GetWritableTensorView<double>();

    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);
}
