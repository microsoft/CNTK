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
            InvalidArgument("Cannot create a NDArrayView over a null data buffer.");

        if (bufferSizeInBytes < (viewShape.TotalSize() * sizeof(ElementType)))
            InvalidArgument("Size (%d) of the specified buffer for creating the NDArrayView is smaller than the specified view shape '%S'.",
                            (int)bufferSizeInBytes, viewShape.AsString().c_str());

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
                                                       const DeviceDescriptor& device,
                                                       size_t numNonZeroValues = 0)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        std::shared_ptr<Matrix<ElementType>> matrix = std::make_shared<Matrix<ElementType>>(matrixDims.first,
                                                                                            matrixDims.second,
                                                                                            AsCNTKImplDeviceId(device),
                                                                                            IsSparseStorageFormat(storageType) ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                                            AsCNTKImplMatrixFormat(storageType),
                                                                                            numNonZeroValues);
        return new TensorView<ElementType>(matrix, AsTensorViewShape(viewShape));
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
        : NDArrayView(AsDataType<ElementType>(), device, StorageFormat::SparseCSC, viewShape, false, AllocateTensorView<ElementType>(viewShape, StorageFormat::SparseCSC, device, numNonZeroValues))
    {
        if ((colStarts == nullptr) || (rowIndices == nullptr) || (nonZeroValues == nullptr) || (numNonZeroValues == 0) || (numNonZeroValues > viewShape.TotalSize()))
            InvalidArgument("Invalid sparse CSC format data specified for construction of NDArrayView with shape '%S'; "
                            "either one of the specified buffers is null or the count (%d) of non-zero values is invalid.",
                            viewShape.AsString().c_str(), (int)numNonZeroValues);

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
    {}

    NDArrayView::~NDArrayView()
    {}

    void NDArrayView::SetValue(float value)
    {
        if (GetDataType() == DataType::Double)
            SetValue((double)value);
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

    template <typename ElementType>
    /*static*/ std::shared_ptr<Matrix<ElementType>> NDArrayView::GetMatrixImpl(const TensorView<ElementType>* tensorView, size_t rowColSplitPoint)
    {
        auto tensorShape = tensorView->GetShape();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        if (tensorShape.GetRank() <= 2 && rowColSplitPoint != 0)
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
            InvalidArgument("NDArrayView::GetWritableTensorView: Cannot get a writable TensorView from a read-only NDArrayView.");

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
        default:
            LogicError("NDArrayView::Alias: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), Shape(), IsReadOnly() || readOnly, tensorView);
    }

    template <typename ElementType>
    const TensorView<ElementType> NDArrayView::NativeTensorView() const
    {
        return TensorView<ElementType>(*GetTensorView<ElementType>(), Microsoft::MSR::CNTK::TensorShape(m_viewShape.Dimensions()));
    }

    template <typename ElementType>
    TensorView<ElementType> NDArrayView::WritableNativeTensorView()
    {
        return TensorView<ElementType>(*GetWritableTensorView<ElementType>(), Microsoft::MSR::CNTK::TensorShape(m_viewShape.Dimensions()));
    }

    NDArrayViewPtr NDArrayView::NumericOperationInPlace(double beta, const std::vector<NDArrayViewPtr>& inputs, double alpha, int opInt, int reductionOpInt)
    {
        const auto          op = (Microsoft::MSR::CNTK::ElementWiseOperator) (opInt);
        const auto reductionOp = (Microsoft::MSR::CNTK::ElementWiseOperator) (reductionOpInt);
        if (inputs.size() < 1 || inputs.size() > 3)
            LogicError("NDArrayView::NumericOperationInPlace: Invalid number of inputs: %d", (int)inputs.size());
        // types must match
        for (const auto& input : inputs)
        {
            if (input->m_dataType != m_dataType)
                LogicError("NDArrayView::NumericOperationInPlace: Input argument's DataType %s differs from result's DataType %s", DataTypeName(input->m_dataType), DataTypeName(m_dataType));
        }
        switch (m_dataType)
        {
        case DataType::Float:
            switch (inputs.size())
            {
            case 1:
                WritableNativeTensorView<float>().DoUnaryOpOf((float)beta, inputs[0]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 2:
                WritableNativeTensorView<float>().DoBinaryOpOf((float)beta, inputs[0]->NativeTensorView<float>(), inputs[1]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 3:
                WritableNativeTensorView<float>().DoTernaryOpOf((float)beta, inputs[0]->NativeTensorView<float>(), inputs[1]->NativeTensorView<float>(), inputs[2]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            }
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            switch (inputs.size())
            {
            case 1:
                WritableNativeTensorView<double>().DoUnaryOpOf((double)beta, inputs[0]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 2:
                WritableNativeTensorView<double>().DoBinaryOpOf((double)beta, inputs[0]->NativeTensorView<double>(), inputs[1]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 3:
                WritableNativeTensorView<double>().DoTernaryOpOf((double)beta, inputs[0]->NativeTensorView<double>(), inputs[1]->NativeTensorView<double>(), inputs[2]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            }
            break;
        default:
            LogicError("NDArrayView::NumericOperationInPlace: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
        return this->shared_from_this(); // return ourselves to allow for chaining
    }

    /*static*/ NDArrayViewPtr NDArrayView::NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, int op, NDArrayViewPtr out, double beta)
    {
        if (!out)
        {
            // for element-wise operations, the output shape is the axis-wise max over all inputs
            // TODO: eventually, this must be reconciled with all the shape-inference code
            size_t rank = 0;
            for (const auto& input : inputs)
                rank = std::max(rank, input->Shape().Rank());
            NDShape shape(rank, 1);
            for (const auto& input : inputs)
            {
                const auto& inputShape = input->Shape();
                for (size_t k = 0; k < inputShape.Rank(); k++)
                    shape[k] = std::max(shape[k], inputShape[k]);
            }
            // create result object; properties besides shape are inherited from input 0 for now
            out = MakeSharedObject<NDArrayView>(inputs[0]->GetDataType(), inputs[0]->GetStorageFormat(), shape, inputs[0]->Device());
            beta = 0; // newly created object is assumed 0
        }
        // perform operation in-place on result object
        return out->NumericOperationInPlace(beta, inputs, alpha, op, (int)Microsoft::MSR::CNTK::ElementWiseOperator::opSum/*not reducing, actually*/);
    }

    NDArrayViewPtr NDArrayView::MatrixProductInPlace(double beta, bool transC, const NDArrayViewPtr& inputA, bool transA, const NDArrayViewPtr& inputB, bool transB, double alpha)
    {
        // types must match
        for (const auto& input : { inputA, inputB })
        {
            if (input->m_dataType != m_dataType)
                LogicError("NDArrayView::MatrixProductInPlace: Input argument's DataType %s differs from result's DataType %s", DataTypeName(input->m_dataType), DataTypeName(m_dataType));
        }
        switch (m_dataType)
        {
        case DataType::Float:
            WritableNativeTensorView<float>().DoMatrixProductOf((float)beta, transC, inputA->NativeTensorView<float>(), transA, inputB->NativeTensorView<float>(), transB, (float)alpha);
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            WritableNativeTensorView<double>().DoMatrixProductOf((double)beta, transC, inputA->NativeTensorView<double>(), transA, inputB->NativeTensorView<double>(), transB, (double)alpha);
            break;
        default:
            LogicError("NDArrayView::MatrixProductInPlace: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
        return this->shared_from_this(); // return ourselves to allow for chaining
    }

    /*static*/ NDArrayViewPtr NDArrayView::MatrixProduct(bool transC, const NDArrayViewPtr& inputA, bool transA, const NDArrayViewPtr& inputB, bool transB, double alpha, size_t outputRank, NDArrayViewPtr out, double beta)
    {
        if (!out)
        {
            // shape inference
            const auto& shapeA = inputA->Shape();
            const auto& shapeB = inputB->Shape();
            if (shapeA.Rank() != 2 && shapeA.Rank() != 1)
                LogicError("NDArrayView::MatrixProductInPlace: For now only vectors and 2D matrices are supported, invalid shape '%S'", shapeA.AsString().c_str());
            if (shapeB.Rank() != 2 && shapeB.Rank() != 1)
                LogicError("NDArrayView::MatrixProductInPlace: For now only vectors and 2D matrices are supported, invalid shape '%S'", shapeB.AsString().c_str());
            const auto innerA = transA ? shapeA[0] : shapeA[shapeA.Rank() - 1]; // inner (dot-product) dimension
            const auto innerB = transB ? shapeB[shapeB.Rank() - 1] : shapeB[0];
            if (innerA != innerB)
                LogicError("NDArrayView::MatrixProductInPlace: Inner dimensions %d and %d don't match", (int)innerA, (int)innerB);
            auto dimsC = std::vector<size_t>();  // TODO: use a different class here to avoid memory allocation?
            // assemble the output shape from the non-inner dimensions. Note that vec^t * vec will end up with a scalar (rank 0)
            if (shapeA.Rank() == 2)
                dimsC.push_back(transA ? shapeA[1] : shapeA[0]);
            if (shapeB.Rank() == 2)
                dimsC.push_back(transB ? shapeB[0] : shapeB[1]);
            if (transC && dimsC.size() == 2)
                std::swap(dimsC[0], dimsC[1]); // reverse
            const auto shapeC = NDShape(dimsC);
            // create result object; properties besides shape are inherited from input 0 for now
            out = MakeSharedObject<NDArrayView>(inputA->GetDataType(), inputA->GetStorageFormat(), shapeC, inputA->Device());
            beta = 0; // newly created object is assumed 0
        }
        // perform operation in-place on result object
        return out->MatrixProductInPlace(beta, transC, inputA, transA, inputB, transB, alpha);
    }

    // TODO: move the Python code down here first, then test. Then optimize.
    /*static*/ NDArrayViewPtr NDArrayView::SpliceFrom(const std::vector<NDArrayViewPtr>& inputs, int axis, NDArrayViewPtr out, double beta)
    {
        size_t numInputs = inputs.size();
        auto dims = inputs[0]->Shape().Dimensions();
        if (axis < dims.size())
            LogicError("Splice: currently only splicing in a new slowest-changing axis is supported");
        if (axis >= dims.size())
            dims.resize(axis + 1, 1);
        dims[axis] *= numInputs;
        NDShape shape(dims);
        if (!out)
        {
            out = MakeSharedObject<NDArrayView>(inputs[0]->GetDataType(), inputs[0]->GetStorageFormat(), shape, inputs[0]->Device());
            beta = 0;
        }
        else if (shape != out->Shape())
            RuntimeError("Splice: output object has wrong shape");
        // for now copy all slices one by one
        // TODO: change into a single CUDA-kernel launch; needs new kernel and to transfer pointers array to GPU
        std::vector<size_t> startOffset(dims.size(), 0);
        auto extent = dims;
        extent[axis] = 1;
        for (auto i = 0; i < numInputs; i++)
        {
            startOffset[axis] = i;
            auto targetSlice = out->SliceView(startOffset, extent, false); // we copy inputs[i] to this slice
            NumericOperation({ inputs[i] }, /*alpha=*/1.0, (int)Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, targetSlice);
        }
        return out;
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
        std::vector<size_t> lastOffset(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            if ((i < sliceViewShape.Rank()) && (sliceViewShape[i] == NDShape::InferredDimension))
                sliceViewShape[i] = Shape()[i] - startOffset[i];

            endOffset[i] = startOffset[i] + ((i < sliceViewShape.Rank()) ? sliceViewShape[i] : 1);
            lastOffset[i] = endOffset[i] - 1;

            if (anyPrevAxisSliced && ((endOffset[i] - startOffset[i]) != 1))
                InvalidArgument("NDArrayView::SliceView: Cannot create a slice which is not contiguous in memory. "
                    "This NDArrayView shape = %S, slice offset = %S, slice extent = %S.",
                    Shape().AsString().c_str(), NDShape(startOffset).AsString().c_str(), NDShape(extent).AsString().c_str());

            bool isCurrentAxisSliced = (startOffset[i] != 0) || (endOffset[i] != Shape()[i]);
            anyPrevAxisSliced = anyPrevAxisSliced || isCurrentAxisSliced;
        }

        auto flatBufferOffset = AsTensorShape(Shape()).Locate(startOffset);  // offset and length into underlying ElementType array...
        auto flatBufferLength = AsTensorShape(Shape()).Locate(lastOffset) + 1 - flatBufferOffset; // ...which is known to be consecutive
        void* tensorView = nullptr;
        // At this point, it is guaranteed that the slice is consecutive in memory. We distinguish two cases:
        // If the slice is expressable a column slice, we will use ColumnSlice(). This will work with sparse data.
        // If, on the other hand, it is a row slice in a single column (such as a single element), we will
        // reshape the matrix into a flat row vector, and then slice the elements.
        // The latter will fail for sparse matrices, as sparse columns can only be slice-viewed as an entire column.
        switch (m_dataType)
        {
        case DataType::Float:
        {
            auto currentMatrix = GetMatrix<float>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            shared_ptr<Matrix<float>> slicedMatrixView;
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                slicedMatrixView = make_shared<Matrix<float>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                slicedMatrixView = make_shared<Matrix<float>>(std::move(sliced));
            }
            tensorView = new TensorView<float>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
            break;
        }
        case DataType::Double:
        {
#if 1
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            shared_ptr<Matrix<double>> slicedMatrixView;
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                slicedMatrixView = make_shared<Matrix<double>>(std::move(sliced));
            }
#else // keeping old version for easier comparison in case something goes wrong--to be deleted soon
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            auto sliceViewMatrixDims = GetMatrixDimensions(sliceViewShape);              // slice if interpreted as Matrix
            assert((flatBufferOffset % sliceViewMatrixDims.first) == 0);
            auto sliceMatrixColumnOffset = flatBufferOffset / sliceViewMatrixDims.first; // Matrix column in which view begins
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                LogicError("NDArrayView::SliceView: Currently only slices that can be realized as a column slice of the underlying Matrix object, are allowed");

            auto slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));
#endif
            tensorView = new TensorView<double>(slicedMatrixView, AsTensorViewShape(sliceViewShape));
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
            LogicError("NDArrayView::AsShape: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), Device(), GetStorageFormat(), newShape, IsReadOnly(), tensorView);
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    ElementType* NDArrayView::WritableDataBuffer()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::WritableDataBuffer: Cannot get writable data buffer from a read-only NDArrayView.");

        return const_cast<ElementType*>(DataBuffer<ElementType>());
    }

    // TODO: This could actually be strided?
    template <typename ElementType>
    const ElementType* NDArrayView::DataBuffer() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::DataBuffer: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (IsSparse())
            InvalidArgument("DataBuffer/WritableDataBuffer methods not supported for sparse NDArrayiew objects.");

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
            LogicError("NDArrayView::ChangeDevice: Unsupported DataType %s", DataTypeName(m_dataType));
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

    template <typename ElementType>
    ElementType NDArrayView::AsScalar() const
    {
        auto scalarData = this->shared_from_this();
        if (scalarData->Shape().TotalSize() != 1)
            LogicError("NDArrayView::AsScalar: The NDArrayView shaped '%S' is not a scalar.", scalarData->Shape().AsString().c_str());

        ElementType scalar = std::numeric_limits<ElementType>::quiet_NaN();
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
            scalar = *(cpuData->DataBuffer<float>());
        else if (scalarData->GetDataType() == DataType::Double)
            scalar = static_cast<ElementType>(*(cpuData->DataBuffer<double>()));
        else
            LogicError("NDArrayView::AsScalar: Unsupported DataType");

        return scalar;
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
    template float NDArrayView::AsScalar<float>() const;
    template double NDArrayView::AsScalar<double>() const;
}
