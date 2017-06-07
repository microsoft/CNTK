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

#define let const auto

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    // matrix from user-provided buffer, template version
    template <typename ElementType>
    static std::shared_ptr<MatrixBase> CreateStorageObject(const NDShape& viewShape,
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
        return std::make_shared<Matrix<ElementType>>(matrixDims.first, matrixDims.second, (ElementType*)dataBuffer, AsCNTKImplDeviceId(device), matrixFlagDontOwnBuffer);
    }

    // matrix from user-provided buffer, dataType version
    static std::shared_ptr<MatrixBase> CreateStorageObject(CNTK::DataType dataType,
                                                    const NDShape& viewShape,
                                                    const DeviceDescriptor& device,
                                                    void* dataBuffer,
                                                    size_t bufferSizeInBytes)
    {
        switch (dataType)
        {
        case DataType::Float:
            return CreateStorageObject<float>(viewShape, device, dataBuffer, bufferSizeInBytes);
        case DataType::Double:
            return CreateStorageObject<double>(viewShape, device, dataBuffer, bufferSizeInBytes);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    // new matrix, template version
    template <typename ElementType>
    static std::shared_ptr<MatrixBase> CreateStorageObject(const NDShape& viewShape,
                                                    CNTK::StorageFormat storageType,
                                                    const DeviceDescriptor& device,
                                                    size_t numNonZeroValues = 0)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        return std::make_shared<Matrix<ElementType>>(matrixDims.first,
                                                     matrixDims.second,
                                                     AsCNTKImplDeviceId(device),
                                                     IsSparseStorageFormat(storageType) ? MatrixType::SPARSE : MatrixType::DENSE,
                                                     AsCNTKImplMatrixFormat(storageType),
                                                     numNonZeroValues);
    }

    // new matrix, dataType version
    static std::shared_ptr<MatrixBase> CreateStorageObject(CNTK::DataType dataType,
                                                    CNTK::StorageFormat storageType,
                                                    const NDShape& viewShape,
                                                    const DeviceDescriptor& device,
                                                    size_t numNonZeroValues = 0)
    {
        switch (dataType)
        {
        case DataType::Float:
            return CreateStorageObject<float>(viewShape, storageType, device, numNonZeroValues);
        case DataType::Double:
            return CreateStorageObject<double>(viewShape, storageType, device, numNonZeroValues);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, viewShape, readOnly, CreateStorageObject(dataType, viewShape, device, dataBuffer, bufferSizeInBytes))
    {
    }

    template <typename ElementType>
    NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(AsDataType<ElementType>(), viewShape, false, CreateStorageObject<ElementType>(viewShape, StorageFormat::SparseCSC, device, numNonZeroValues))
    {
        if ((colStarts == nullptr) || (rowIndices == nullptr) || (nonZeroValues == nullptr) || (numNonZeroValues == 0) || (numNonZeroValues > viewShape.TotalSize()))
            InvalidArgument("Invalid sparse CSC format data specified for construction of NDArrayView with shape '%S'; "
                            "either one of the specified buffers is null or the count (%d) of non-zero values is invalid.",
                            viewShape.AsString().c_str(), (int)numNonZeroValues);

        auto sparseMatrix = GetWritableMatrix<ElementType>(1);
        sparseMatrix->SetMatrixFromCSCFormat(colStarts, rowIndices, nonZeroValues, numNonZeroValues, sparseMatrix->GetNumRows(), sparseMatrix->GetNumCols());
        m_isReadOnly = readOnly;
    }

    // ElementType-erasing version of TensorView(sob, shape), based on dataType
    static void* NewTensorView(CNTK::DataType dataType, const shared_ptr<MatrixBase>& sob, const TensorShape& shape)
    {
        switch (dataType)
        {
        case DataType::Float:
            {
                auto matrix = dynamic_pointer_cast<Matrix<float>>(sob);
                if (matrix)
                    return new TensorView<float>(matrix, shape);
            }
            break;
        case DataType::Double:
            {
                auto matrix = dynamic_pointer_cast<Matrix<double>>(sob);
                if (matrix)
                    return new TensorView<double>(matrix, shape);
            }
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
        LogicError("Storage Object is not of DataType %s", DataTypeName(dataType));
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, bool readOnly, const shared_ptr<MatrixBase>& sob)
        : m_dataType(dataType), m_device(AsDeviceDescriptor(sob->GetDeviceId())), m_storageFormat(AsStorageFormat(sob->GetFormat())), m_viewShape(viewShape), m_isReadOnly(readOnly)
    {
#define LAZY_2D_PADDING // if defined then rank-2 padding of TensorShapes happens upon access, not upon creation
#ifdef LAZY_2D_PADDING
        const auto tensorShape = AsTensorShape(viewShape);
#else
        const auto tensorShape = AsTensorShapeMin2D(viewShape); // not lazy (old version): sdo it here and bake it into teh object
#endif
        void* tensorView = NewTensorView(dataType, sob, tensorShape);
        m_tensorViewPtr = std::shared_ptr<void>(tensorView, [this](void*) {
            switch (m_dataType)
            {
            case DataType::Float:
                delete GetTensorViewPtr<float>();
                break;
            case DataType::Double:
                delete GetTensorViewPtr<double>();
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
                break;
            }
        });
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device)
        : NDArrayView(dataType, viewShape, false, CreateStorageObject(dataType, storageType, viewShape, device))
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
    /*static*/ std::shared_ptr<Matrix<ElementType>> NDArrayView::GetMatrixImpl(const TensorView<ElementType>& tensorView, size_t rowColSplitPoint)
    {
        auto tensorShape = tensorView.GetShape();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        if (tensorShape.GetRank() <= 1 && rowColSplitPoint != 0)
            return tensorView.AsMatrix();

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

        return tensorView.Reshaped(tensorShape).AsMatrix();
    }

    // -ViewMin2D: use if you interop with V1 code that needs shapes of rank 2 or higher
    // These versions are only ever called by GetMatrix() and, from outside, in Accumulator::Update(), which probably could do without.
    // If we get them out from Update(), then we can just inline them here.
    template <typename ElementType>
    std::shared_ptr<const Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::GetTensorViewMin2D() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetTensorViewMin2D: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        auto tensorView = static_pointer_cast<const TensorView<ElementType>>(m_tensorViewPtr);
#ifdef LAZY_2D_PADDING
        const auto& shape = tensorView->GetShape();
        if (shape.size() < 2) // we must pad to at least 2D
        {
            auto paddedShape = AsTensorShapeMin2D(shape); // adds 1-dimensions if rank < 2
            tensorView = make_shared<TensorView<ElementType>>(tensorView->Reshaped(paddedShape));
        }
#endif
        return tensorView;
    }

    template <typename ElementType>
    std::shared_ptr<Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::GetWritableTensorViewMin2D()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::GetWritableTensorViewMin2D: Cannot get a writable TensorView from a read-only NDArrayView.");

        return const_pointer_cast<TensorView<ElementType>>(GetTensorViewMin2D<ElementType>());
    }

    template <typename ElementType>
    std::shared_ptr<const Matrix<ElementType>> NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        return GetMatrixImpl<ElementType>(*GetTensorViewMin2D<ElementType>(), rowColSplitPoint);
    }

    template <typename ElementType>
    std::shared_ptr<Matrix<ElementType>> NDArrayView::GetWritableMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        return GetMatrixImpl<ElementType>(*GetWritableTensorViewMin2D<ElementType>(), rowColSplitPoint);
    }

    // -ViewPtr: use if you don't care about V1-compatible 2D-padded shape
    template <typename ElementType>
    const TensorView<ElementType>* NDArrayView::GetTensorViewPtr() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetTensorViewPtr: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        return (const TensorView<ElementType>*)(m_tensorViewPtr.get());
    }

    template <typename ElementType>
    TensorView<ElementType>* NDArrayView::GetWritableTensorViewPtr()
    {
        if (IsReadOnly())
            InvalidArgument("NDArrayView::GetWritableTensorViewPtr: Cannot get a writable TensorView from a read-only NDArrayView.");

        return const_cast<TensorView<ElementType>*>(GetTensorViewPtr<ElementType>());
    }

    shared_ptr<MatrixBase> NDArrayView::GetStorageObjectPtr() const
    {
        shared_ptr<MatrixBase> matrix;
        switch (m_dataType)
        {
        case DataType::Float:
            return GetTensorViewPtr<float>()->GetSOBPtr();
        case DataType::Double:
            return GetTensorViewPtr<double>()->GetSOBPtr();
        default:
            LogicError("NDArrayView::Alias: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    NDArrayViewPtr NDArrayView::DeepClone(const DeviceDescriptor& device, bool readOnly/* = false*/) const
    {
        NDArrayViewPtr newView = MakeSharedObject<NDArrayView>(this->GetDataType(), this->GetStorageFormat(), this->Shape(), device);
        // TODO: for dense data, this can call TensorView, which will amount to a cudaMemcpy() while bypassing GetMatrix() complexity
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

        // TODO: like DeepClone, for dense data, this can call TensorView, which will amount to a cudaMemcpy() while bypassing GetMatrix() complexity
        //       Maybe we need a shared copy function. Maybe DeepClone can call CopyFrom()?
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
        return MakeSharedObject<NDArrayView>(GetDataType(), Shape(), IsReadOnly() || readOnly, GetStorageObjectPtr());
    }

    // TODO: one day, this just becomes GetTensorViewPtr
    template <typename ElementType>
    std::shared_ptr<const Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::NativeTensorView() const
    {
#ifndef LAZY_2D_PADDING
        if (m_viewShape.Rank() < 2) // m_tensorViewPtr has the wrong shape if rank < 2
            return make_shared<TensorView<ElementType>>(GetTensorViewPtr<ElementType>()->Reshaped(AsTensorShape(m_viewShape)));
#endif
        return static_pointer_cast<const TensorView<ElementType>>(m_tensorViewPtr);
    }

    template <typename ElementType>
    std::shared_ptr<Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::WritableNativeTensorView()
    {
#ifndef LAZY_2D_PADDING
        if (m_viewShape.Rank() < 2) // m_tensorViewPtr has the wrong shape if rank < 2
            return make_shared<TensorView<ElementType>>(GetWritableTensorViewPtr<ElementType>()->Reshaped(AsTensorShape(m_viewShape)));
#endif
        return static_pointer_cast<TensorView<ElementType>>(m_tensorViewPtr);
    }

    /*static*/ NDArrayViewPtr NDArrayView::NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, int opInt, NDArrayViewPtr out, double beta, int reductionOpInt)
    {
        // create result object if not given
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
            beta = 0; // newly created object is already 0
        }
        // perform operation in-place on result object
        const auto          op = (Microsoft::MSR::CNTK::ElementWiseOperator) (opInt);
        const auto reductionOp = reductionOpInt != -1 ? (Microsoft::MSR::CNTK::ElementWiseOperator) (reductionOpInt) : Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
        switch (out->m_dataType)
        {
        case DataType::Float:
            switch (inputs.size())
            {
            case 1:
                out->WritableNativeTensorView<float>()->DoUnaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 2:
                out->WritableNativeTensorView<float>()->DoBinaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), *inputs[1]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 3:
                out->WritableNativeTensorView<float>()->DoTernaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), *inputs[1]->NativeTensorView<float>(), *inputs[2]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            default:
                LogicError("NDArrayView::NumericOperation: Invalid number of inputs: %d", (int)inputs.size());
            }
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            switch (inputs.size())
            {
            case 1:
                out->WritableNativeTensorView<double>()->DoUnaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 2:
                out->WritableNativeTensorView<double>()->DoBinaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), *inputs[1]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 3:
                out->WritableNativeTensorView<double>()->DoTernaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), *inputs[1]->NativeTensorView<double>(), *inputs[2]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            default:
                LogicError("NDArrayView::NumericOperation: Invalid number of inputs: %d", (int)inputs.size());
            }
            break;
        default:
            LogicError("NDArrayView::NumericOperation: Unsupported DataType %s", DataTypeName(out->m_dataType));
            break;
        }
        return out;
    }

    /*static*/ NDArrayViewPtr NDArrayView::MatrixProduct(bool transC, const NDArrayViewPtr& inputA, bool transA, const NDArrayViewPtr& inputB, bool transB, double alpha, size_t outputRank, NDArrayViewPtr out, double beta)
    {
        // create result object if not given
        if (!out)
        {
            // shape inference
            const auto& shapeA = inputA->Shape();
            const auto& shapeB = inputB->Shape();
            if (shapeA.Rank() != 2 && shapeA.Rank() != 1)
                LogicError("NDArrayView::MatrixProduct: For now only vectors and 2D matrices are supported, invalid shape '%S'.", shapeA.AsString().c_str());
            if (shapeB.Rank() != 2 && shapeB.Rank() != 1)
                LogicError("NDArrayView::MatrixProduct: For now only vectors and 2D matrices are supported, invalid shape '%S'.", shapeB.AsString().c_str());
            const auto innerA = transA ? shapeA[0] : shapeA[shapeA.Rank() - 1]; // inner (dot-product) dimension
            const auto innerB = transB ? shapeB[shapeB.Rank() - 1] : shapeB[0];
            if (innerA != innerB)
                LogicError("NDArrayView::MatrixProduct: Inner dimensions %d and %d don't match.", (int)innerA, (int)innerB);
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
            beta = 0; // newly created object is already 0
        }
        switch (out->m_dataType)
        {
        case DataType::Float:
            out->WritableNativeTensorView<float>()->DoMatrixProductOf((float)beta, transC, *inputA->NativeTensorView<float>(), transA, *inputB->NativeTensorView<float>(), transB, (float)alpha);
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            out->WritableNativeTensorView<double>()->DoMatrixProductOf((double)beta, transC, *inputA->NativeTensorView<double>(), transA, *inputB->NativeTensorView<double>(), transB, (double)alpha);
            break;
        default:
            LogicError("NDArrayView::MatrixProduct: Unsupported DataType %s", DataTypeName(out->m_dataType));
            break;
        }
        return out;
    }

    /*static*/ NDArrayViewPtr NDArrayView::GatherBatch(const vector<NDArrayViewPtr>& inputs, int axis, NDArrayViewPtr out)
    {
        size_t numInputs = inputs.size();
        if (!out        || true) // keep this for now for testing this
        {
            // determine output rank
            size_t maxRank = 0;
            for (let& input : inputs)
                if (maxRank < input->Shape().Rank())
                    maxRank = input->Shape().Rank();
            if (axis + 1 < maxRank)
                LogicError("NDArrayView::GatherBatch: Currently only splicing in a new or the slowest-changing axis is supported.");
            let outRank = max(maxRank, (size_t)axis + 1);
            // determine output shape from input0
            vector<size_t> outDims;
            outDims.reserve(outRank);
            const auto& input0 = *inputs[0];
            let& inputDims = input0.Shape().Dimensions();
            outDims.assign(inputDims.begin(), inputDims.end());
            outDims.resize(outRank, 1);   // add batch axis (and pad) if needed
            if (axis >= maxRank) // when batching into a new axis, then new axis = #inputs
                outDims[axis] = numInputs;
            else // if along existing axis, then we must explicitly sum up over all inputs
            {
                size_t sumDim = 0;
                for (let& input : inputs)
                {
                    let& inDims = input->Shape().Dimensions();
                    if (axis >= inDims.size())
                        sumDim += 1;
                    else
                        sumDim += inDims[axis];
                }
                outDims[axis] = sumDim;
            }
            NDShape shape(move(outDims));
            if (out && out->Shape() != shape)
                LogicError("NDArrayView::GatherBatch: bad out dim"); // (this err msg will go away after some testing)
            if (!out)
            out = MakeSharedObject<NDArrayView>(input0.GetDataType(), input0.GetStorageFormat(), shape, input0.Device());
        }
        // perform the operation
        // The underlying TensorView call expects a functor to access the TensorView items.
        // Any error checking will happen inside the TensorView function, so we don't duplicate it here.
        switch (out->m_dataType)
        {
        case DataType::Float:
            out->WritableNativeTensorView<float>()->DoGatherBatchOf(inputs.size(), [&](size_t i) -> const TensorView<float>&
            {
                return *inputs[i]->GetTensorViewPtr<float>();
            });
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            out->WritableNativeTensorView<double>()->DoGatherBatchOf(inputs.size(), [&](size_t i) -> const TensorView<double>&
            {
                return *inputs[i]->GetTensorViewPtr<double>();
            });
            break;
        default:
            LogicError("NDArrayView::GatherBatch: Unsupported DataType %s", DataTypeName(out->m_dataType));
            break;
        }
        return out;
    }

    /*static*/ void NDArrayView::ScatterBatch(const NDArrayViewPtr& input, vector<NDArrayViewPtr>& outputs, double beta)
    {
        // The underlying TensorView call expects a functor to access the TensorView items.
        // Any error checking will happen inside the TensorView function, so we don't duplicate it here.
        switch (input->m_dataType)
        {
        case DataType::Float:
            input->NativeTensorView<float>()->DoScatterBatchOf((float)beta, outputs.size(), [&](size_t i) -> TensorView<float>&
            {
                return *outputs[i]->GetWritableTensorViewPtr<float>();
            });
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            input->NativeTensorView<double>()->DoScatterBatchOf((double)beta, outputs.size(), [&](size_t i) -> TensorView<double>&
            {
                return *outputs[i]->GetWritableTensorViewPtr<double>();
            });
            break;
        default:
            LogicError("NDArrayView::GatherBatch: Unsupported DataType %s", DataTypeName(input->m_dataType));
            break;
        }
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

        bool anyPrevAxisSliced = false; // (only used for error check)
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
        shared_ptr<MatrixBase> matrix;
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
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                matrix = make_shared<Matrix<float>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                matrix = make_shared<Matrix<float>>(std::move(sliced));
            }
            break;
        }
        case DataType::Double:
        {
#if 1
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                matrix = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                matrix = make_shared<Matrix<double>>(std::move(sliced));
            }
#else // keeping old version for easier comparison in case something goes wrong--to be deleted soon
            // TODO: This was changed on master; below is latest. Maybe it does the same as my change above. Test this.
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<double>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements()).ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));
#endif
            break;
        }
        default:
            LogicError("NDArrayView::SliceView: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), sliceViewShape, IsReadOnly() || readOnly, matrix);
    }

    NDArrayViewPtr NDArrayView::IndexLastAxis(size_t index, bool readOnly) const
    {
        auto rank = Shape().Rank();
        if (rank == 0)
            InvalidArgument("NDArrayView::IndexLastAxis: Cannot index a scalar.");

        auto sliceViewShape = m_viewShape.SubShape(0, rank - 1); // final shape drops final axis
        // if last axis already has only 1 element then just reshape it away
        if (m_viewShape[rank - 1] == 1)
            return AsShape(sliceViewShape);

        std::vector<size_t> startOffset(rank, 0);
        std::vector<size_t> endOffset = m_viewShape.Dimensions();
        startOffset[rank - 1] = index;
        endOffset[rank - 1] = index + 1;
        std::vector<size_t> lastOffset(rank);
        for (size_t i = 0; i < rank; ++i)
            lastOffset[i] = endOffset[i] - 1;

        // beyond this point is code duplication from ViewSlice()
        // TODO: simplify further, we can get rid of the vector mallocs altogether

        auto flatBufferOffset = AsTensorShape(Shape()).Locate(startOffset);  // offset and length into underlying ElementType array...
        auto flatBufferLength = AsTensorShape(Shape()).Locate(lastOffset) + 1 - flatBufferOffset; // ...which is known to be consecutive
        shared_ptr<MatrixBase> matrix;
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
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                matrix = make_shared<Matrix<float>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                matrix = make_shared<Matrix<float>>(std::move(sliced));
            }
            break;
        }
        case DataType::Double:
        {
#if 1
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            if (flatBufferOffset % currentMatrixDims.first == 0 && flatBufferLength % currentMatrixDims.first == 0)
            {
                matrix = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(flatBufferOffset / currentMatrixDims.first, flatBufferLength / currentMatrixDims.first));
            }
            else
            {
                auto sliced = currentMatrix->Reshaped(1, currentMatrixDims.first * currentMatrixDims.second).ColumnSlice(flatBufferOffset, flatBufferLength);
                sliced.Reshape(flatBufferLength, 1);
                matrix = make_shared<Matrix<double>>(std::move(sliced));
            }
#else // keeping old version for easier comparison in case something goes wrong--to be deleted soon
            // TODO: This was changed on master; below is latest. Maybe it does the same as my change above. Test this.
            auto currentMatrix = GetMatrix<double>();
            std::pair<size_t, size_t> currentMatrixDims = { currentMatrix->GetNumRows(), currentMatrix->GetNumCols() };
            std::shared_ptr<Matrix<double>> slicedMatrixView;
            if (sliceViewMatrixDims.first != currentMatrixDims.first)
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->Reshaped(1, currentMatrix->GetNumElements()).ColumnSlice(flatBufferOffset, sliceViewShape.TotalSize()));
            else
                slicedMatrixView = make_shared<Matrix<double>>(currentMatrix->ColumnSlice(sliceMatrixColumnOffset, sliceViewMatrixDims.second));
#endif
            break;
        }
        default:
            LogicError("NDArrayView::IndexLastAxis: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), sliceViewShape, IsReadOnly() || readOnly, matrix);
    }

    NDArrayViewPtr NDArrayView::AsShape(const NDShape& newShape) const
    {
        if (newShape.TotalSize() != Shape().TotalSize())
        {
            InvalidArgument("NDArrayView::AsShape: The total size (%d) of this view's shape '%S' must be same as the size (%d) of the newShape '%S'.",
                            (int)Shape().TotalSize(), Shape().AsString().c_str(),
                            (int)newShape.TotalSize(), newShape.AsString().c_str());
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), newShape, IsReadOnly(), GetStorageObjectPtr());
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
        // TODO: Don't we just need the storage object?
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
            // TODO: Don't we just need the storage object?
            auto matrix = GetMatrix<float>();
            matrix->TransferFromDeviceToDevice(matrix->GetDeviceId(), AsCNTKImplDeviceId(device), /*isBeingMoved = */ true, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ true);
            matrix->CollapseDataLocation();
            break;
        }
        case DataType::Double:
        {
            // TODO: Don't we just need the storage object?
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

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), shape, false, randomNormalMatrix);
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr NDArrayView::RandomUniform(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto matrixDims = GetMatrixDimensions(shape);
        auto randomUniformMatrix = std::make_shared<Matrix<ElementType>>(Matrix<ElementType>::RandomUniform(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device), (ElementType)rangeBegin, (ElementType)rangeEnd, seed));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), shape, false, randomUniformMatrix);
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
    template std::shared_ptr<TensorView<float>> NDArrayView::GetWritableTensorViewMin2D<float>();
    template std::shared_ptr<TensorView<double>> NDArrayView::GetWritableTensorViewMin2D<double>();

    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);

    template CNTK_API float NDArrayView::AsScalar<float>() const;
    template CNTK_API double NDArrayView::AsScalar<double>() const;
}
