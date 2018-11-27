//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "TensorView.h"
#include "Matrix.h"
#include "CPUSparseMatrix.h"
#include "GPUSparseMatrix.h"
#include <algorithm>
#include "TensorShape.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    template<typename ElemType>
    inline ElemType quiet_NaN()
    {
        return std::numeric_limits<ElemType>::quiet_NaN();
    }

    template<>
    inline float16 quiet_NaN<float16>()
    {
        return float16(std::numeric_limits<float>::quiet_NaN());
    }

    template <>
    inline char quiet_NaN<char>()
    {
        return char(std::numeric_limits<int8_t>::quiet_NaN());
    }

    template <>
    inline int8_t quiet_NaN<int8_t>()
    {
        return char(std::numeric_limits<int8_t>::quiet_NaN());
    }

    template <>
    inline int16_t quiet_NaN<int16_t>()
    {
        return char(std::numeric_limits<int16_t>::quiet_NaN());
    }

    template <typename V1ElemType>
    static TensorView<V1ElemType>* AllocateTensorView(const NDShape& viewShape,
                                                       const DeviceDescriptor& device,
                                                       void* dataBuffer,
                                                       size_t bufferSizeInBytes)
    {
        if (dataBuffer == nullptr)
            InvalidArgument("Cannot create a NDArrayView over a null data buffer.");

        if (bufferSizeInBytes < (viewShape.TotalSize() * sizeof(V1ElemType)))
            InvalidArgument("Size (%d) of the specified buffer for creating the NDArrayView is smaller than the specified view shape '%S'.",
                            (int)bufferSizeInBytes, viewShape.AsString().c_str());

        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<V1ElemType>> matrix = std::make_shared<Matrix<V1ElemType>>(matrixDims.first, matrixDims.second, (V1ElemType*)dataBuffer, AsCNTKImplDeviceId(device), matrixFlagDontOwnBuffer);
        return new TensorView<V1ElemType>(matrix, AsTensorViewShape(viewShape));
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
        case DataType::Float16:
            return AllocateTensorView<half>(viewShape, device, dataBuffer, bufferSizeInBytes);
        case DataType::Int8:
            return AllocateTensorView<char>(viewShape, device, dataBuffer, bufferSizeInBytes);
        case DataType::Int16:
            return AllocateTensorView<short>(viewShape, device, dataBuffer, bufferSizeInBytes);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    template<typename V1ElemType>
    static TensorView<V1ElemType>* AllocateTensorView(const NDShape& viewShape,
                                                      CNTK::StorageFormat storageType,
                                                      const DeviceDescriptor& device,
                                                      size_t numNonZeroValues = 0)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<V1ElemType>> matrix = std::make_shared<Matrix<V1ElemType>>(matrixDims.first,
                                                                                          matrixDims.second,
                                                                                          AsCNTKImplDeviceId(device),
                                                                                          IsSparseStorageFormat(storageType) ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                                          AsCNTKImplMatrixFormat(storageType),
                                                                                          numNonZeroValues);
        return new TensorView<V1ElemType>(matrix, AsTensorViewShape(viewShape));
    }

    static void* AllocateTensorView(CNTK::DataType dataType,
                                    CNTK::StorageFormat storageType,
                                    const NDShape& viewShape,
                                    const DeviceDescriptor& device,
                                    size_t numNonZeroValues = 0)
    {
        switch (dataType)
        {
        case DataType::Float:
            return AllocateTensorView<float>(viewShape, storageType, device, numNonZeroValues);
        case DataType::Double:
            return AllocateTensorView<double>(viewShape, storageType, device, numNonZeroValues);
        case DataType::Float16:
            return AllocateTensorView<half>(viewShape, storageType, device, numNonZeroValues);
        case DataType::Int8:
            return AllocateTensorView<char>(viewShape, storageType, device, numNonZeroValues);
        case DataType::Int16:
            return AllocateTensorView<short>(viewShape, storageType, device, numNonZeroValues);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, device, StorageFormat::Dense, viewShape, readOnly, AllocateTensorView(dataType, viewShape, device, dataBuffer, bufferSizeInBytes))
    {
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const void* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, device, StorageFormat::SparseCSC, viewShape, false, AllocateTensorView(dataType, StorageFormat::SparseCSC, viewShape, device, numNonZeroValues * DataTypeSize(dataType)))
    {
        if ((colStarts == nullptr) || (rowIndices == nullptr) || (nonZeroValues == nullptr) || (numNonZeroValues == 0) || (numNonZeroValues > viewShape.TotalSize()))
            InvalidArgument("Invalid sparse CSC format data specified for construction of NDArrayView with shape '%S'; "
                            "either one of the specified buffers is null or the count (%d) of non-zero values is invalid.",
                            viewShape.AsString().c_str(), (int)numNonZeroValues);
        switch (dataType)
        {
            case DataType::Float:
            {
                auto sparseMatrix = GetWritableMatrix<float>(1);
                sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, (const float*)nonZeroValues, numNonZeroValues, sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
                break;
            }
            case DataType::Double:
            {
                auto sparseMatrix = GetWritableMatrix<double>(1);
                sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, (const double*)nonZeroValues, numNonZeroValues, sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
                break;
            }
            case DataType::Float16:
            {
                auto sparseMatrix = GetWritableMatrix<half>(1);
                sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, (const half*)nonZeroValues, numNonZeroValues, sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
                break;
            }
            case DataType::Int8:
            {
                auto sparseMatrix = GetWritableMatrix<char>(1);
                sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, (const char*)nonZeroValues, numNonZeroValues,
                    sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
                break;
            }
            case DataType::Int16:
            {
                auto sparseMatrix = GetWritableMatrix<short>(1);
                sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, (const short*)nonZeroValues, numNonZeroValues,
                    sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dataType));
                break;
        }
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
            case DataType::Float16:
                delete GetTensorView<half>();
                break;
            case DataType::Int8:
                delete GetTensorView<char>();
                break;
            case DataType::Int16:
                delete GetTensorView<short>();
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
                break;
            }
        });
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device)
        : NDArrayView(dataType, device, storageType, viewShape, false, AllocateTensorView(dataType, storageType, viewShape, device))
    {}

    NDArrayView::~NDArrayView()
    {}

    void NDArrayView::SetValue(float value)
    {
        if (GetDataType() == DataType::Double)
            SetValue((double)value);
        else if (GetDataType() == DataType::Float16)
            SetValue((float16)value);
        else
        {
            if (IsSparse())
                LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a scalar is only allowed for objects with dense storage format.");

            GetWritableMatrix<float>()->SetValue(value);
        }
    }

    void NDArrayView::SetValue(double value)
    {
        if (IsSparse())
            LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a scalar is only allowed for objects with dense storage format.");

        GetWritableMatrix<double>()->SetValue(value);
    }

    void NDArrayView::SetValue(int8_t value)
    {
        if (IsSparse())
            LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a scalar is only allowed for objects with dense storage format.");

        GetWritableMatrix<char>()->SetValue(value);
    }

    void NDArrayView::SetValue(int16_t value)
    {
        if (IsSparse())
            LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a scalar is only allowed for objects with dense storage format.");

        GetWritableMatrix<short>()->SetValue(value);
    }

    bool NDArrayView::IsSliceView()
    {
        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto currentMatrix = GetMatrix<float>();
            return currentMatrix->IsView();
        }
        case DataType::Double:
        {
            auto currentMatrix = GetMatrix<double>();
            return currentMatrix->IsView();
        }
        case DataType::Float16:
        {
            auto currentMatrix = GetMatrix<half>();
            return currentMatrix->IsView();
        }
        case DataType::Int8:
        {
            auto currentMatrix = GetMatrix<char>();
            return currentMatrix->IsView();
        }
        case DataType::Int16:
        {
            auto currentMatrix = GetMatrix<short>();
            return currentMatrix->IsView();
        }
        }
        return false;
    }

    void NDArrayView::SetValue(float16 value)
    {
        if (IsSparse())
            LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a scalar is only allowed for objects with dense storage format.");

        GetWritableMatrix<half>()->SetValue(*reinterpret_cast<half*>(&value));
    }

    template <typename V1ElemType>
    /*static*/ std::shared_ptr<Matrix<V1ElemType>> NDArrayView::GetMatrixImpl(const TensorView<V1ElemType>* tensorView, size_t rowColSplitPoint)
    {
        auto tensorShape = tensorView->GetShape();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        if (tensorShape.GetRank() <= 1 && rowColSplitPoint != 0)
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
                LogicError("The TensorView (shape = %s) underlying this NDArrayView cannot be flattened to a Matrix.", ((std::string)tensorShape).c_str());

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

    template <typename V1ElemType>
    std::shared_ptr<const Matrix<V1ElemType>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        return GetMatrixImpl<V1ElemType>(GetTensorView<V1ElemType>(), rowColSplitPoint);
    }

    template <typename V1ElemType>
    std::shared_ptr<Matrix<V1ElemType>> NDArrayView::GetWritableMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        return GetMatrixImpl<V1ElemType>(GetWritableTensorView<V1ElemType>(), rowColSplitPoint);
    }

    std::shared_ptr<const MatrixBase> NDArrayView::GetMatrixBase(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        switch (m_dataType)
        {
        case DataType::Float:
            return GetMatrixImpl<float>(GetTensorView<float>(), rowColSplitPoint);
        case DataType::Double:
            return GetMatrixImpl<double>(GetTensorView<double>(), rowColSplitPoint);
        case DataType::Float16:
            return GetMatrixImpl<half>(GetTensorView<half>(), rowColSplitPoint);
        case DataType::Int8:
            return GetMatrixImpl<char>(GetTensorView<char>(), rowColSplitPoint);
        case DataType::Int16:
            return GetMatrixImpl<short>(GetTensorView<short>(), rowColSplitPoint);
        default:
            LogicError("Unknown m_dataType %d", (int)m_dataType);
        }
        return nullptr;
    }

    std::shared_ptr<MatrixBase> NDArrayView::GetWritableMatrixBase(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        switch (m_dataType)
        {
        case DataType::Float:
            return GetMatrixImpl<float>(GetWritableTensorView<float>(), rowColSplitPoint);
        case DataType::Double:
            return GetMatrixImpl<double>(GetWritableTensorView<double>(), rowColSplitPoint);
        case DataType::Float16:
            return GetMatrixImpl<half>(GetWritableTensorView<half>(), rowColSplitPoint);
        case DataType::Int8:
            return GetMatrixImpl<char>(GetWritableTensorView<char>(), rowColSplitPoint);
        case DataType::Int16:
            return GetMatrixImpl<short>(GetWritableTensorView<short>(), rowColSplitPoint);
        default:
            LogicError("Unknown m_dataType %d", (int)m_dataType);
        }
        return nullptr;
    }

    template <typename V1ElemType>
    const TensorView<V1ElemType>* NDArrayView::GetTensorView() const
    {
        if (AsDataType<V1ElemType>() != m_dataType)
            LogicError("NDArrayView::GetTensorView: The specified ElementType %s does not match the DataType %s", typeid(V1ElemType).name(), DataTypeName(m_dataType));

        return (const TensorView<V1ElemType>*)(m_tensorView.get());
    }

    template <typename V1ElemType>
    TensorView<V1ElemType>* NDArrayView::GetWritableTensorView()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::GetWritableTensorView: Cannot get a writable TensorView from a read-only NDArrayView.");

        return const_cast<TensorView<V1ElemType>*>(GetTensorView<V1ElemType>());
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
        case DataType::Float16:
        {
            auto newMatrix = newView->GetWritableMatrix<half>();
            auto thisMatrix = GetMatrix<half>();
            newMatrix->AssignValuesOf(*thisMatrix);
            break;
        }
        case DataType::Int8:
        {
            auto newMatrix = newView->GetWritableMatrix<char>();
            auto thisMatrix = GetMatrix<char>();
            newMatrix->AssignValuesOf(*thisMatrix);
            break;
        }
        case DataType::Int16:
        {
            auto newMatrix = newView->GetWritableMatrix<short>();
            auto thisMatrix = GetMatrix<short>();
            newMatrix->AssignValuesOf(*thisMatrix);
            break;
        }
        default:
            LogicError("NDArrayView::DeepClone: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        newView->m_isReadOnly = readOnly;
        return newView;
    }

    void NDArrayView::CopyFrom(const NDArrayView& source)
    {
        if ((source.Shape() != Shape()) && (AsTensorShape(source.Shape()) != AsTensorShape(Shape())))
            InvalidArgument("NDArrayView::CopyFrom: The source view shape '%S' is not same as the shape '%S' of this NDArrayView.", 
                            source.Shape().AsString().c_str(), Shape().AsString().c_str());

        if (IsReadOnly())
            RuntimeError("NDArrayView::CopyFrom: Cannot modify contents of a readonly NDArrayView.");

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
        case DataType::Float16:
        {
            auto sourceMatrix = source.GetMatrix<half>();
            auto destMatrix = GetWritableMatrix<half>();
            destMatrix->AssignValuesOf(*sourceMatrix);
            break;
        }
        case DataType::Int8:
        {
            auto sourceMatrix = source.GetMatrix<char>();
            auto destMatrix = GetWritableMatrix<char>();
            destMatrix->AssignValuesOf(*sourceMatrix);
            break;
        }
        case DataType::Int16:
        {
            auto sourceMatrix = source.GetMatrix<short>();
            auto destMatrix = GetWritableMatrix<short>();
            destMatrix->AssignValuesOf(*sourceMatrix);
            break;
        }
        default:
            LogicError("NDArrayView::CopyFrom: Unsupported DataType %s", DataTypeName(m_dataType));
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
        case DataType::Float16:
            tensorView = new TensorView<half>(*(GetTensorView<half>()));
            break;
        case DataType::Int8:
            tensorView = new TensorView<char>(*(GetTensorView<char>()));
            break;
        case DataType::Int16:
            tensorView = new TensorView<short>(*(GetTensorView<short>()));
            break;
        default:
            LogicError("NDArrayView::Alias: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), Shape(), IsReadOnly() || readOnly, tensorView);
    }

    NDArrayViewPtr NDArrayView::SliceView(const std::vector<size_t>& startOffset, const std::vector<size_t>& extent, bool readOnly) const
    {
        auto rank = Shape().Rank();
        if (startOffset.size() != rank)
            InvalidArgument("NDArrayView::SliceView: Rank (%d) of the NDArrayView does not match the dimensionality (%d) of the specified slice offset.", (int)rank, (int)startOffset.size());

        if (extent.size() > rank)
            InvalidArgument("NDArrayView::SliceView: Dimensionality (%d) of the specified slice extent exceeds the rank (%d) of this NDArrayView.", (int)extent.size(), (int)rank);

        if (std::find(extent.begin(), extent.end(), 0) != extent.end())
            InvalidArgument("NDArrayView::SliceView: Specified slice extent is zero along at least one of the axes.");

        bool anyPrevAxisSliced = false;
        NDShape sliceViewShape(extent);
        std::vector<size_t> endOffset(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            if ((i < sliceViewShape.Rank()) && (sliceViewShape[i] == NDShape::InferredDimension))
                sliceViewShape[i] = Shape()[i] - startOffset[i];

            endOffset[i] = startOffset[i] + ((i < sliceViewShape.Rank()) ? sliceViewShape[i] : 1);

            if (anyPrevAxisSliced && ((endOffset[i] - startOffset[i]) != 1))
                InvalidArgument("NDArrayView::SliceView: Cannot create a slice which is not contiguous in memory. "
                                "This NDArrayView shape = %S, slice offset = %S, slice extent = %S.",
                                 Shape().AsString().c_str(), NDShape(startOffset).AsString().c_str(), NDShape(extent).AsString().c_str());

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
            std::shared_ptr<Matrix<float>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView = make_shared<Matrix<float>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements()).ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<float>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));

            tensorView = new TensorView<float>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Double:
        {
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<double>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements()).ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));

            tensorView = new TensorView<double>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Float16:
        {
            auto currentMatrix = GetMatrix<half>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<half>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView = make_shared<Matrix<half>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements()).ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<half>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));

            tensorView = new TensorView<half>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Int8:
        {
            auto currentMatrix = GetMatrix<char>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<char>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView =
                make_shared<Matrix<char>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements())
                    .ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<char>>(
                    currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));

            tensorView = new TensorView<char>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Int16:
        {
            auto currentMatrix = GetMatrix<short>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<short>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView =
                make_shared<Matrix<short>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements())
                    .ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<short>>(
                    currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));

            tensorView = new TensorView<short>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        default:
            LogicError("NDArrayView::SliceView: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), sliceViewShape, IsReadOnly() || readOnly, tensorView);
    }

    NDArrayViewPtr NDArrayView::AsShape(const NDShape& newShape) const
    {
        if (newShape.TotalSize() != Shape().TotalSize())
        {
            InvalidArgument("NDArrayView::AsShape: The total size (%d) of this view's shape '%S' must be same as the size (%d) of the newShape '%S'.",
                            (int)Shape().TotalSize(), Shape().AsString().c_str(),
                            (int)newShape.TotalSize(), newShape.AsString().c_str());
        }

        auto newTensorShape = AsTensorViewShape(newShape);
        void* tensorView = nullptr;
        switch (m_dataType)
        {
        case DataType::Float:
            tensorView = new TensorView<float>(*(GetTensorView<float>()), newTensorShape);
            break;
        case DataType::Double:
            tensorView = new TensorView<double>(*(GetTensorView<double>()), newTensorShape);
            break;
        case DataType::Float16:
            tensorView = new TensorView<half>(*(GetTensorView<half>()), newTensorShape);
            break;
        case DataType::Int8:
            tensorView = new TensorView<char>(*(GetTensorView<char>()), newTensorShape);
            break;
        case DataType::Int16:
            tensorView = new TensorView<short>(*(GetTensorView<short>()), newTensorShape);
            break;
        default:
            LogicError("NDArrayView::AsShape: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), newShape, IsReadOnly(), tensorView);
    }

    template <typename ElementType>
    const ElementType* NDArrayView::DataBuffer() const
    {
        return const_cast<ElementType*>(_DataBuffer<ElementType, ElementType>());
    }

    template<>
    const float16* NDArrayView::DataBuffer<float16>() const
    {
        return const_cast<float16*>(_DataBuffer<float16, half>());
    }

    template<>
    const int8_t* NDArrayView::DataBuffer<int8_t>() const
    {
        return const_cast<int8_t*>(_DataBuffer<int8_t, char>());
    }

    template<>
    const int16_t* NDArrayView::DataBuffer<int16_t>() const
    {
        return const_cast<int16_t*>(_DataBuffer<int16_t, short>());
    }

    // TODO: This could actually be strided?
    template <typename ElementType, typename V1ElemType>
    const ElementType* NDArrayView::_DataBuffer() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::DataBuffer: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (IsSparse())
            InvalidArgument("The stroage format of 'this' NDArrayView is sparse. Please use SparseDataBuffers().");

        // First make sure that the underlying matrix is on the right device
        auto matrix = GetMatrix<V1ElemType>();
        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        return reinterpret_cast<ElementType*>(matrix->Data());
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    ElementType* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView.");

        return const_cast<ElementType*>(DataBuffer<ElementType>());
    }

    template <>
    int8_t* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView.");

        return const_cast<int8_t*>(DataBuffer<int8_t>());
    }

    template <>
    int16_t* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView.");

        return const_cast<int16_t*>(DataBuffer<int16_t>());
    }

    template <typename ElementType>
    std::tuple<const ElementType *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers() const
    {
        return _SparseCSCDataBuffers<ElementType, ElementType>();
    }

    template <>
    std::tuple<const float16 *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<float16>() const
    {
        return _SparseCSCDataBuffers<float16, half>();
    }

    template <>
    std::tuple<const int8_t *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<int8_t>() const
    {
        return _SparseCSCDataBuffers<int8_t, char>();
    }

    template <>
    std::tuple<const int16_t *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<int16_t>() const
    {
        return _SparseCSCDataBuffers<int16_t, short>();
    }

    template <typename ElementType, typename V1ElemType>
    std::tuple<const ElementType *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::_SparseCSCDataBuffers() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::SparseDataBuffers: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (!IsSparse())
            RuntimeError("The storage format of 'this' NDArrayView is dense. Please use another DataBuffer().");

        if(GetStorageFormat() != StorageFormat::SparseCSC)
            RuntimeError("The SparseCSCDataBuffers() method only supports CSC sparse format.");

        std::shared_ptr<const Matrix<V1ElemType>> matrix = GetMatrix<V1ElemType>();
        auto matrixDims = GetMatrixDimensions(Shape());
        if (matrix->GetNumRows() != matrixDims.first)
            LogicError("The number of rows of the underlying matrix does not match the shape.");
        if (matrix->GetNumCols() != matrixDims.second)
            LogicError("The number of columns of the underlying matrix does not match the shape.");

        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        if ((matrix->GetMatrixType() != Microsoft::MSR::CNTK::MatrixType::SPARSE) || (matrix->GetFormat() != Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC))
            RuntimeError("NDArrayView::SparseDataBuffers: The underlying matrix of 'this' NDArrayView is not in the CSC sparse format.");

        size_t numNonZeroValues;
        V1ElemType* nonZeroValues;
        SparseIndexType* colStarts;
        SparseIndexType* rowIndices;
        if (m_device.Type() == DeviceKind::CPU)
        {
            if (sizeof(CPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on CPU.");
            std::shared_ptr<Microsoft::MSR::CNTK::CPUSparseMatrix<V1ElemType>> sparseMatrix = matrix->m_CPUSparseMatrix;
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<V1ElemType *>(sparseMatrix->NzValues());
            colStarts = static_cast<SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices = static_cast<SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else if (m_device.Type() == DeviceKind::GPU)
        {
            if (sizeof(GPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on GPU.");
            std::shared_ptr<Microsoft::MSR::CNTK::GPUSparseMatrix<V1ElemType>> sparseMatrix = matrix->m_GPUSparseMatrix;
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<V1ElemType *>(sparseMatrix->NzValues());
            colStarts = static_cast<SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices = static_cast<SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else
        {
            RuntimeError("NDArrayView::SparseDataBuffers: The device %S is currently not supported.",DeviceKindName(m_device.Type()));
        }

        return std::tuple<ElementType *, SparseIndexType *, SparseIndexType *, size_t>(reinterpret_cast<ElementType*>(nonZeroValues), colStarts, rowIndices, numNonZeroValues);
    }

    template <typename ElementType>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers() const
    {
        return _SparseBlockColumnDataBuffers<ElementType, ElementType>();
    }

    template <>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<float16>() const
    {
        return _SparseBlockColumnDataBuffers<float16, half>();
    }

    template <>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<int8_t>() const
    {
        return _SparseBlockColumnDataBuffers<int8_t, char>();
    }

    template <>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<int16_t>() const
    {
        return _SparseBlockColumnDataBuffers<int16_t, short>();
    }

    template <typename ElementType, typename V1ElemType>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::_SparseBlockColumnDataBuffers() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::SparseBlockColumnDataBuffers: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (!IsSparse())
            RuntimeError("The storage format of 'this' NDArrayView is dense. Please use another DataBuffer().");

        if (GetStorageFormat() != StorageFormat::SparseBlockCol)
            RuntimeError("The SparseBlockColumnDataBuffers() method only supports sparse block column format.");

        std::shared_ptr<const Matrix<V1ElemType>> matrix = GetMatrix<V1ElemType>();

        size_t numBlocks;
        size_t numRows;
        size_t numCols;
        V1ElemType* blockValues;
        SparseIndexType* blockId2Col;
        SparseIndexType* col2BlockId;
        if (m_device.Type() == DeviceKind::GPU)
        {
            if (sizeof(GPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on GPU.");
            std::shared_ptr<Microsoft::MSR::CNTK::GPUSparseMatrix<V1ElemType>> sparseMatrix = matrix->m_GPUSparseMatrix;
            numBlocks = sparseMatrix->GetBlockSize();
            numRows = sparseMatrix->GetNumRows();
            numCols = sparseMatrix->GetNumCols();
            blockValues = static_cast<V1ElemType *>(sparseMatrix->NzValues());
            blockId2Col = static_cast<SparseIndexType *>(sparseMatrix->BlockId2ColOrRow());
            col2BlockId = static_cast<SparseIndexType *>(sparseMatrix->ColOrRow2BlockId());
        }
        else
        {
            // CPU sparse block column is not yet supported, as the index format is different from GPU sparse block column
            RuntimeError("NDArrayView::SparseBlockColumnDataBuffers: The device %S is currently not supported.", DeviceKindName(m_device.Type()));
        }

        return std::tuple<ElementType *, SparseIndexType *, SparseIndexType *, size_t, size_t, size_t>(reinterpret_cast<ElementType*>(blockValues), blockId2Col, col2BlockId, numBlocks, numRows, numCols);
    }

    void NDArrayView::AdjustSparseBlockColumn(const SparseIndexType* cpuCol2BlockId, size_t numBlocks, bool useBlockId2Col)
    {
        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto matrix = GetWritableMatrix<float>();
            matrix->AdjustSparseBlockColumn(cpuCol2BlockId, numBlocks, useBlockId2Col);
            break;
        }
        case DataType::Double:
        {
            auto matrix = GetWritableMatrix<double>();
            matrix->AdjustSparseBlockColumn(cpuCol2BlockId, numBlocks, useBlockId2Col);
            break;
        }
        case DataType::Int8:
        {
            auto matrix = GetWritableMatrix<char>();
            matrix->AdjustSparseBlockColumn(cpuCol2BlockId, numBlocks, useBlockId2Col);
            break;
        }
        case DataType::Int16:
        {
            auto matrix = GetWritableMatrix<short>();
            matrix->AdjustSparseBlockColumn(cpuCol2BlockId, numBlocks, useBlockId2Col);
            break;
        }
        default:
            LogicError("NDArrayView::AdjustSparseBlockColumn: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
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
        case DataType::Float16:
        {
            auto matrix = GetMatrix<half>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        case DataType::Int8:
        {
            auto matrix = GetMatrix<char>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        case DataType::Int16:
        {
            auto matrix = GetMatrix<short>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        default:
            LogicError("NDArrayView::ChangeDevice: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        m_device = device;
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr NDArrayView::RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomNormal<ElementType, ElementType>(shape, mean, stdDev, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomNormal<float16>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomNormal<float16, half>(shape, mean, stdDev, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomNormal<int8_t>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomNormal<int8_t, char>(shape, mean, stdDev, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomNormal<int16_t>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomNormal<int16_t, short>(shape, mean, stdDev, seed, device);
    }

    template <typename ElementType, typename V1ElemType>
    /*static*/ NDArrayViewPtr NDArrayView::_RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomNormalMatrix = std::make_shared<Matrix<V1ElemType>>(Matrix<V1ElemType>::RandomGaussian(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (V1ElemType)mean, (V1ElemType)stdDev, seed));
        auto tensorView = new TensorView<V1ElemType>(randomNormalMatrix, AsTensorViewShape(shape));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), device, StorageFormat::Dense, shape, false, tensorView);
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomUniform<ElementType, ElementType>(shape, rangeBegin, rangeEnd, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform<float16>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomUniform<float16, half>(shape, rangeBegin, rangeEnd, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform<int8_t>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomUniform<int8_t, char>(shape, rangeBegin, rangeEnd, seed, device);
    }

    template <>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform<int16_t>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return NDArrayView::_RandomUniform<int16_t, short>(shape, rangeBegin, rangeEnd, seed, device);
    }

    template <typename ElementType, typename V1ElemType>
    /*static*/ NDArrayViewPtr NDArrayView::_RandomUniform(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomUniformMatrix = std::make_shared<Matrix<V1ElemType>>(Matrix<V1ElemType>::RandomUniform(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (V1ElemType)rangeBegin, (V1ElemType)rangeEnd, seed));
        auto tensorView = new TensorView<V1ElemType>(randomUniformMatrix, AsTensorViewShape(shape));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), device, StorageFormat::Dense, shape, false, tensorView);
    }

    template <typename ElementType>
    ElementType NDArrayView::AsScalar() const
    {
        return _AsScalar<ElementType, ElementType>();
    }

    template <>
    float16 NDArrayView::AsScalar<float16>() const
    {
        return _AsScalar<float16, half>();
    }

    template <>
    int8_t NDArrayView::AsScalar<int8_t>() const
    {
        return _AsScalar<int8_t, char>();
    }

    template <>
    int16_t NDArrayView::AsScalar<int16_t>() const
    {
        return _AsScalar<int16_t, short>();
    }

    template <typename ElementType, typename V1ElemType>
    ElementType NDArrayView::_AsScalar() const
    {
        auto scalarData = this->shared_from_this();
        if (scalarData->Shape().TotalSize() != 1)
            LogicError("NDArrayView::AsScalar: The NDArrayView shaped '%S' is not a scalar.", scalarData->Shape().AsString().c_str());

        ElementType scalar = quiet_NaN<ElementType>();
        std::shared_ptr<const NDArrayView> cpuData;
        if (scalarData->Device() == DeviceDescriptor::CPUDevice())
            cpuData = scalarData;
        else
        {
            auto tmpCPUData = std::make_shared<NDArrayView>(scalarData->GetDataType(), scalarData->Shape(), CNTK::DeviceDescriptor::CPUDevice());
            tmpCPUData->CopyFrom(*scalarData);
            cpuData = tmpCPUData;
        }

        if (scalarData->GetDataType() == DataType::Float)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<float>()));
        else if (scalarData->GetDataType() == DataType::Double)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<double>()));
        else if (scalarData->GetDataType() == DataType::Float16)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<float16>()));
        else if (scalarData->GetDataType() == DataType::Int8)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<char>()));
        else if (scalarData->GetDataType() == DataType::Int16)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<short>()));
        else
            LogicError("NDArrayView::AsScalar: Unsupported DataType");

        return scalar;
    }

    std::wstring NDArrayView::AsString() const
    {
        wstringstream wss;
        std::wstring device = DeviceKindName(m_device.Type());
        wss << L"NDArrayView(" << m_viewShape.AsString() << L", " << device << L")";
        return wss.str();
    }

    // Explicit template instantiations
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<double>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float16>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<int8_t>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<int16_t>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<float>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<double>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<float16>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<int8_t>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<int16_t>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API const float* NDArrayView::DataBuffer<float>() const;
    template CNTK_API const double* NDArrayView::DataBuffer<double>() const;
    template CNTK_API const float16* NDArrayView::DataBuffer<float16>() const;
    template CNTK_API const int8_t* NDArrayView::DataBuffer<int8_t>() const;
    template CNTK_API const int16_t* NDArrayView::DataBuffer<int16_t>() const;

    template CNTK_API const TensorView<float>* NDArrayView::GetTensorView<float>() const;
    template CNTK_API const TensorView<double>* NDArrayView::GetTensorView<double>() const;
    template CNTK_API const TensorView<half>* NDArrayView::GetTensorView<half>() const;
    template CNTK_API const TensorView<char>* NDArrayView::GetTensorView<char>() const;
    template CNTK_API const TensorView<short>* NDArrayView::GetTensorView<short>() const;

    template CNTK_API std::tuple<const float*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<float>() const;
    template CNTK_API std::tuple<const double*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<double>() const;
    template CNTK_API std::tuple<const float16*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<float16>() const;
    template CNTK_API std::tuple<const int8_t*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<int8_t>() const;
    template CNTK_API std::tuple<const int16_t*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<int16_t>() const;

    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<float>() const;
    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<double>() const;
    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<float16>() const;
    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<int8_t>() const;
    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<int16_t>() const;

    template CNTK_API float* NDArrayView::WritableDataBuffer<float>();
    template CNTK_API double* NDArrayView::WritableDataBuffer<double>();
    template CNTK_API float16* NDArrayView::WritableDataBuffer<float16>();
    template CNTK_API int8_t* NDArrayView::WritableDataBuffer<int8_t>();
    template CNTK_API int16_t* NDArrayView::WritableDataBuffer<int16_t>();

    template std::shared_ptr<const Matrix<float>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template std::shared_ptr<const Matrix<double>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template std::shared_ptr<const Matrix<half>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template std::shared_ptr<const Matrix<char>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template std::shared_ptr<const Matrix<short>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;

    template std::shared_ptr<Matrix<float>> NDArrayView::GetWritableMatrix<float>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template std::shared_ptr<Matrix<double>> NDArrayView::GetWritableMatrix<double>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template std::shared_ptr<Matrix<half>> NDArrayView::GetWritableMatrix<half>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template std::shared_ptr<Matrix<char>> NDArrayView::GetWritableMatrix<char>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template std::shared_ptr<Matrix<short>> NDArrayView::GetWritableMatrix<short>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template TensorView<float>* NDArrayView::GetWritableTensorView<float>();
    template TensorView<double>* NDArrayView::GetWritableTensorView<double>();
    template TensorView<half>* NDArrayView::GetWritableTensorView<half>();
    template TensorView<char>* NDArrayView::GetWritableTensorView<char>();
    template TensorView<short>* NDArrayView::GetWritableTensorView<short>();

    template float NDArrayView::AsScalar<float>() const;
    template double NDArrayView::AsScalar<double>() const;
    template float16 NDArrayView::AsScalar<float16>() const;
    template int8_t NDArrayView::AsScalar<int8_t>() const;
    template int16_t NDArrayView::AsScalar<int16_t>() const;
}
