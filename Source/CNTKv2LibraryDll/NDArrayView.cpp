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

    // ElementType-erasing version of new TensorView(sob, shape), based on dataType.
    static shared_ptr<void> NewTensorView(CNTK::DataType dataType, const shared_ptr<MatrixBase>& sob, const TensorShape& shape)
    {
        switch (dataType)
        {
        case DataType::Float:
            {
                auto matrix = dynamic_pointer_cast<Matrix<float>>(sob);
                if (matrix)
                    return shared_ptr<void>(new TensorView<float>(matrix, shape), [](void* p) { delete (const TensorView<float>*)(p); });
            }
            break;
        case DataType::Double:
            {
                auto matrix = dynamic_pointer_cast<Matrix<double>>(sob);
                if (matrix)
                    return shared_ptr<void>(new TensorView<double>(matrix, shape), [](void* p) { delete (const TensorView<double>*)(p); });
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
        m_tensorViewPtr = NewTensorView(dataType, sob, tensorShape);
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, const TensorShape& tensorShape, bool readOnly, const shared_ptr<MatrixBase>& sob)
        : m_dataType(dataType), m_device(AsDeviceDescriptor(sob->GetDeviceId())), m_storageFormat(AsStorageFormat(sob->GetFormat())),
          m_viewShape(move(tensorShape.GetDimsAsVector())), m_isReadOnly(readOnly),
          m_tensorViewPtr(NewTensorView(dataType, sob, tensorShape))
    {}

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
            if (IsSparse() && value != 0)
                LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a non-zero scalar is only allowed for objects with dense storage format.");

            GetWritableMatrix<float>()->SetValue(value);
        }
    }

    /*static*/ void NDArrayView::Sync(const DeviceDescriptor& device)
    {
        Matrix<float>::SyncDevice(AsCNTKImplDeviceId(device));
    }

    void NDArrayView::SetValue(double value)
    {
        if (GetDataType() == DataType::Float && (float)value == value) // useful for setting stuff to 0 or 1
            SetValue((float)value);
        else
        {
            if (IsSparse() && value != 0)
                LogicError("NDArrayView::SetValue: Setting a NDArrayView contents to a non-zero scalar is only allowed for objects with dense storage format.");

            GetWritableMatrix<double>()->SetValue(value);
        }
    }

    // determine matrix shape from TensorShape
    // Resulting tensorShape has rank 2.
    static void ToMatrixShape(Microsoft::MSR::CNTK::TensorShape& tensorShape, size_t rowColSplitPoint, const size_t AutoSelectRowColSplitPoint)
    {
        size_t splitPoint = rowColSplitPoint;
        if (splitPoint == AutoSelectRowColSplitPoint)
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
            // let's pick the split point to be 1 (that is, the first dim becomes the row, the rest become the column).
            // This is consisten with sparse tensor, where the first dimension is sparse, while the remaining
            // ones are the (dense) index, for which we can fold multiple tensor axes together.
            splitPoint = 1;
            if (numDimsThatCannotBeDropped > 1)
            {
                while (dimsToDrop[splitPoint - 1])
                    splitPoint++;
            }
        }

        tensorShape.FlattenTo2DInPlace(splitPoint, "NDArrayView::ToMatrixShape");
    }

    template <typename ElementType>
    /*static*/ std::shared_ptr<Matrix<ElementType>> NDArrayView::GetMatrixImpl(const TensorView<ElementType>& tensorView, size_t rowColSplitPoint)
    {
        // only contiguous tensors can be processed by the Matrix lib
        tensorView.GetShape().VerifyIsDense();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        if (tensorView.GetShape().GetRank() <= 1 && rowColSplitPoint != 0)
            return tensorView.AsMatrix();

        auto tensorShape = tensorView.GetShape();
        ToMatrixShape(tensorShape, rowColSplitPoint, AutoSelectRowColSplitPoint);

        return tensorView.Reviewed(tensorShape).AsMatrix();
    }

#if 1
    // -ViewMin2D: use if you interop with V1 code that needs shapes of rank 2 or higher
    // These versions are only ever called by GetMatrix(). We could just inline them here.
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
            tensorView = make_shared<TensorView<ElementType>>(tensorView->Reviewed(paddedShape));
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
#endif

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

    // WARNING! The SOBPtr does not necessarily represent the offset field of the TensorShape.
    // So one should never use the SOB for Matrix operations directly. Use AsMatrix() or TensorView operations instead.
    // TODO: Remove this function altogether; and replace with creating an NDArrayView from a TensorShape and an old NDArrayView.
    shared_ptr<MatrixBase> NDArrayView::GetStorageObjectPtr() const
    {
        switch (m_dataType)
        {
        case DataType::Float:
            return GetTensorViewPtr<float>()->GetSOBPtr();
        case DataType::Double:
            return GetTensorViewPtr<double>()->GetSOBPtr();
        default:
            LogicError("NDArrayView::GetStorageObjectPtr: Unsupported DataType %s", DataTypeName(m_dataType));
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
        // BUGBUG: We cannot just use the SOBPtr because with Slice(::View), we may have an offset.
        //         Instead, we should just do this the TensorView way.
        //return MakeSharedObject<NDArrayView>(GetDataType(), Shape(), IsReadOnly() || readOnly, GetStorageObjectPtr());
        return MakeSharedObject<NDArrayView>(GetDataType(), GetTensorShape(), IsReadOnly() || readOnly, GetStorageObjectPtr());
    }

    template <typename ElementType>
    static bool AreAliases(const void* av, const void* bv)
    {
        let* aView = (const TensorView<ElementType>*)av;
        let* bView = (const TensorView<ElementType>*)bv;
        let& aTensorShape = aView->GetShape();
        let& bTensorShape = bView->GetShape();
        if (aTensorShape.GetDims() != bTensorShape.GetDims()) // shape must be the same. This is silly--the only time we call this, we have already compared the shape.
            return false;
        if (aTensorShape.GetStrides() != bTensorShape.GetStrides()) // strides must be the same
            return false;
        if (&aView->GetSOB() == &bView->GetSOB() && aTensorShape.GetOffset() == bTensorShape.GetOffset()) // same SOB and same offset: OK
            return true;
        return aView->GetSOB().Data() + aTensorShape.GetOffset() == bView->GetSOB().Data() + bTensorShape.GetOffset(); // otherwise compute buffer address and compare
    }

    bool NDArrayView::IsAliasOf(const NDArrayViewPtr& other) const
    {
        // note: this should not waste cycles, since it is used inside auto-batching
        if (this == other.get()) // fast path: same object
            return true;
        if (m_dataType != other->m_dataType)
            return false;
        switch (m_dataType)
        {
        case DataType::Float:  return AreAliases<float> (this->m_tensorViewPtr.get(), other->m_tensorViewPtr.get());
        case DataType::Double: return AreAliases<double>(this->m_tensorViewPtr.get(), other->m_tensorViewPtr.get());
        default: LogicError("NDArrayView::CopyFrom: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    static DataType GetType(float)  { return DataType::Float;  }
    static DataType GetType(double) { return DataType::Double; }

    // TODO: merge with GetTensorViewPtr() as GetTensorView() which returns a reference while GetTensorViewPtr returns a shared_ptr. If needed at all.
    template <typename ElementType>
    std::shared_ptr<const Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::NativeTensorView() const
    {
        if (GetType(ElementType(0)) != m_dataType)
            LogicError("NativeTensorView: Called with wrong data type %s; is %s.", DataTypeName(GetType(ElementType(0))), DataTypeName(m_dataType));
#ifndef LAZY_2D_PADDING
        if (m_viewShape.Rank() < 2) // m_tensorViewPtr has the wrong shape if rank < 2
            return make_shared<TensorView<ElementType>>(GetTensorViewPtr<ElementType>()->Reviewed(AsTensorShape(m_viewShape)));
#endif
        return static_pointer_cast<const TensorView<ElementType>>(m_tensorViewPtr);
    }

    template <typename ElementType>
    std::shared_ptr<Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::WritableNativeTensorView()
    {
        if (GetType(ElementType(0)) != m_dataType)
            LogicError("WritableNativeTensorView: Called with wrong data type %s; is %s.", DataTypeName(GetType(ElementType(0))), DataTypeName(m_dataType));
#ifndef LAZY_2D_PADDING
        if (m_viewShape.Rank() < 2) // m_tensorViewPtr has the wrong shape if rank < 2
            return make_shared<TensorView<ElementType>>(GetWritableTensorViewPtr<ElementType>()->Reviewed(AsTensorShape(m_viewShape)));
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
            case 0:
                out->WritableNativeTensorView<float>()->DoNullaryOpOf((float)beta, (float)alpha, op, reductionOp);
                break;
            case 1:
                out->WritableNativeTensorView<float>()->DoUnaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 2:
                out->WritableNativeTensorView<float>()->DoBinaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), *inputs[1]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 3:
                out->WritableNativeTensorView<float>()->DoTernaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), *inputs[1]->NativeTensorView<float>(), *inputs[2]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            case 4:
                out->WritableNativeTensorView<float>()->DoQuaternaryOpOf((float)beta, *inputs[0]->NativeTensorView<float>(), *inputs[1]->NativeTensorView<float>(), *inputs[2]->NativeTensorView<float>(), *inputs[3]->NativeTensorView<float>(), (float)alpha, op, reductionOp);
                break;
            default:
                LogicError("NDArrayView::NumericOperation: Invalid number of inputs: %d", (int)inputs.size());
            }
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            switch (inputs.size())
            {
            case 0:
                out->WritableNativeTensorView<double>()->DoNullaryOpOf((double)beta, (double)alpha, op, reductionOp);
                break;
            case 1:
                out->WritableNativeTensorView<double>()->DoUnaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 2:
                out->WritableNativeTensorView<double>()->DoBinaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), *inputs[1]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 3:
                out->WritableNativeTensorView<double>()->DoTernaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), *inputs[1]->NativeTensorView<double>(), *inputs[2]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
                break;
            case 4:
                out->WritableNativeTensorView<double>()->DoQuaternaryOpOf((double)beta, *inputs[0]->NativeTensorView<double>(), *inputs[1]->NativeTensorView<double>(), *inputs[2]->NativeTensorView<double>(), *inputs[3]->NativeTensorView<double>(), (double)alpha, op, reductionOp);
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
    /*static*/ NDArrayViewPtr NDArrayView::NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, const std::wstring& opStr, NDArrayViewPtr out, double beta, const std::wstring& reductionOpStr)
    {
        return NumericOperation(inputs, alpha, TensorView<float>::OpFromName(opStr), out, beta, TensorView<float>::OpFromName(reductionOpStr));
    }
    /*static*/ NDArrayViewPtr NDArrayView::NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, int opInt, const NDShape& outShape, int reductionOpInt)
    {
        // explicit output shape given (useful when doing reductions)
        let out = MakeSharedObject<NDArrayView>(inputs[0]->GetDataType(), inputs[0]->GetStorageFormat(), outShape, inputs[0]->Device());
        return NumericOperation(inputs, alpha, opInt, out, /*beta=*/0, reductionOpInt);
    }
    /*static*/ NDArrayViewPtr NDArrayView::NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, const std::wstring& opStr, const NDShape& outShape, const std::wstring& reductionOpStr)
    {
        return NumericOperation(inputs, alpha, TensorView<float>::OpFromName(opStr), outShape, TensorView<float>::OpFromName(reductionOpStr));
    }

    NDArrayViewPtr operator+(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        return NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opSum);
    }
    NDArrayViewPtr operator-(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        return NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opDifference);
    }
    NDArrayViewPtr operator*(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        return NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opElementwiseProduct);
    }
    NDArrayViewPtr operator/(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        return NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opElementwiseQuotient);
    }
    NDArrayViewPtr& operator+=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        // using opCopy allows to reduce as we go, e.g. rightOperand can inverse-broadcast into leftOperand
        NDArrayView::NumericOperation({ rightOperand }, /*alpha=*/1.0, ElementWiseOperator::opCopy, leftOperand, /*beta=*/1.0);
        return leftOperand;
    }
    NDArrayViewPtr& operator-=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        // note: to allow reduction, this is implemented the same as operator+= except with a negative alpha
        NDArrayView::NumericOperation({ rightOperand }, /*alpha=*/-1.0, ElementWiseOperator::opCopy, leftOperand, /*beta=*/1.0);
        return leftOperand;
    }
    NDArrayViewPtr& operator*=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        // note: rightOperand must not inverse-broadcast into leftOperand
        NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opElementwiseProduct, leftOperand);
        return leftOperand;
    }
    NDArrayViewPtr& operator/=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand)
    {
        // note: rightOperand must not inverse-broadcast into leftOperand
        NDArrayView::NumericOperation({ leftOperand, rightOperand }, 1.0, ElementWiseOperator::opElementwiseQuotient, leftOperand);
        return leftOperand;
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
            if (shapeB.Rank() == 0 || (shapeB.Rank() > 2 && transB))
                LogicError("NDArrayView::MatrixProduct: For now only vectors and 2D matrices are supported when transposed, invalid shape '%S'.", shapeB.AsString().c_str());
            const auto innerA = transA ? shapeA[0] : shapeA[shapeA.Rank() - 1]; // inner (dot-product) dimension
            const auto innerB = transB ? shapeB[shapeB.Rank() - 1] : shapeB[0];
            if (innerA != innerB)
                LogicError("NDArrayView::MatrixProduct: Inner dimensions %d and %d don't match.", (int)innerA, (int)innerB);
            auto dimsC = std::vector<size_t>();  // TODO: use a different class here to avoid memory allocation?
            // assemble the output shape from the non-inner dimensions. Note that vec^t * vec will end up with a scalar (rank 0)
            if (shapeA.Rank() == 2) // for A we only support 2D matrices for now
                dimsC.push_back(transA ? shapeA[1] : shapeA[0]);
            if (transB)
            {
                if (shapeB.Rank() == 2) // if transB then we only support ...do this right
                    dimsC.push_back(shapeB[0]);
            }
            else
                dimsC.insert(dimsC.end(), shapeB.Dimensions().begin() + 1, shapeB.Dimensions().end());
            if (transC)
                if (dimsC.size() == 2)
                    std::swap(dimsC[0], dimsC[1]); // reverse
                else if (dimsC.size() > 2)
                    LogicError("NDArrayView::MatrixProduct: For now only vectors and 2D matrices are supported when result is transposed, invalid shape '%S'.", shapeB.AsString().c_str());
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

    // helper to get the underlying TensorView's TensorShape
    const TensorShape& NDArrayView::GetTensorShape() const
    {
        switch (m_dataType)
        {
        case DataType::Float:  return NativeTensorView<float> ()->GetShape();
        case DataType::Double: return NativeTensorView<double>()->GetShape();
        default: LogicError("NDArrayView::SlicedTensorView: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    // TODO: remove this function if not needed
    bool NDArrayView::IsContiguous() const
    {
        return GetTensorShape().IsDense();
    }

    NDArrayViewPtr NDArrayView::Slice(const std::vector<size_t>& startOffset, const std::vector<size_t>& extent, const std::vector<size_t>& strides, SliceMode sliceMode, bool readOnly) const
    {
        let rank = Shape().Rank();
        if (startOffset.size() != rank)
            InvalidArgument("NDArrayView::Slice: Rank (%d) of the NDArrayView does not match the dimensionality (%d) of the specified slice offset.", (int)rank, (int)startOffset.size());

        if (extent.size() > rank)
            InvalidArgument("NDArrayView::Slice: Dimensionality (%d) of the specified slice extent exceeds the rank (%d) of this NDArrayView.", (int)extent.size(), (int)rank);

        if (std::find(extent.begin(), extent.end(), 0) != extent.end())
            InvalidArgument("NDArrayView::Slice: Specified slice extent is zero along at least one of the axes.");

        // For sparse input, strided views are not supported in any form.
        // In this case, we just set the sliceMode to ContiguousViews. If we originaly requested a TensorView, this will still
        // do the right thing in case of SliceMode::View, and also for ContiguousViewOrCopy if no copy is needed.
        if (IsSparse())
            sliceMode = SliceMode::ContiguousView;

        if (sliceMode == SliceMode::ContiguousView && any_of(strides.begin(), strides.end(), [](size_t v) { return v != 1; }))
            InvalidArgument("NDArrayView::Slice: Strides != 1 are not allowed for SliceMode::ContiguousView, and presently not supported fo sparse data.");

        // Modes:
        //  - ContiguousView --> create both a TensorView and Matrix-level view; fail if not possible
        //  - View --> create a TensorView view, without updating the underlying matrix
        //  - ContiguousViewOrCopy --> first try TensorView view. If not contiguous, then copy it. If contiguous then fall through to ContiguousView. This way, we always get an updated Matrix view.
        // If sparse then the slice must always be contiguous presently (due to lack of underlying slicing function for sparse matrices/tensors).
        if (sliceMode != SliceMode::ContiguousView)
        {
            // get current actual shape
            auto tensorShape = GetTensorShape();

            // narrow it down
            for (size_t i = 0; i < rank; ++i)
            {
                let beginIndex = startOffset[i];
                let endIndex = i >= extent.size()
                               ? beginIndex + 1 // missing extent at the end means index
                               : extent[i] == NDShape::InferredDimension
                                 ? tensorShape[i] // InferredDimension means end index = dim
                                 : beginIndex + extent[i];
                let stride = i < strides.size() ? strides[i] : 1;
                tensorShape.NarrowTo(i, beginIndex, endIndex, (int)stride);
            }

            // drop trailing singleton dimensions
            tensorShape.TrimRankInPlace(extent.size());

            // now continue based on sliceMode and whether the resulting tensor is contiguous or not
            if (sliceMode == SliceMode::View || !tensorShape.IsDense())
            {
                // create a NDArrayView with this new tensor shape onto the existing storage object
                // Note that this version may create an NDArrayView with strides, which is not accepted by some operations.
                // TODO: Check that at least this will be discovered. AsMatrix() checks it. Does that cover all use cases?
                let view = MakeSharedObject<NDArrayView>(m_dataType, tensorShape, readOnly, GetStorageObjectPtr());

                // View: we are done
                if (sliceMode == SliceMode::View)
                    return view;
                // ContiguousViewOrCopy and not contiguous: return a copy of the view
                else
                {
                    assert(sliceMode == SliceMode::ContiguousViewOrCopy && !tensorShape.IsDense());
                    return view->DeepClone(); // the copy is contiguous
                }
            }
            // we get here for ContiguousViewOrCopy if data is contiguous and does not need to be copied
            assert(sliceMode == SliceMode::ContiguousViewOrCopy && tensorShape.IsDense());
            // fall through SliceMode::ContiguousView
        }

        // create a menmory-contiguous view
        // We get here under these conditions:
        //  - caller requested ContiguousView
        //  - caller requested ContiguousViewOrCopy when no copy is required
        //  - input is sparse
        // This also updates the underlying Matrix view, and is thus a little slower.
        // TODO: This is old code which may be simplified by merging with the TensorShape code above.
        // TODO: We should change NDArrayView::Slice() to never update the Matrix view, but do that on the fly in GetMatrixImpl() (AsMatrix() probably already does most of the work).
        bool anyPrevAxisSliced = false;
        NDShape sliceViewShape(extent);       // note: has same #dims as extent
        std::vector<size_t> endOffset(rank);  // note: these have same #dims as 'this', not extent
        std::vector<size_t> lastOffset(rank);
        for (size_t i = 0; i < rank; ++i)
        {
            if ((i < sliceViewShape.Rank()) && (sliceViewShape[i] == NDShape::InferredDimension))
                sliceViewShape[i] = Shape()[i] - startOffset[i];

            // It is allowed that extent[] is shorter than Shape().
            // In this case, those extend values default to 1, and the dimensions are dropped in the result.
            // This allows to express the common case of indexing the last axis and dropping it.
            endOffset[i] = startOffset[i] + ((i < sliceViewShape.Rank()) ? sliceViewShape[i] : 1);
            lastOffset[i] = endOffset[i] - 1;

            // Only the last non-singleton axis can be a slice proper.
            // Any axes after that must be sliced to a singleton axis.
            // Otherwise, data would be non-contiguous, which the Matrix class does not support.
            if (anyPrevAxisSliced && ((endOffset[i] - startOffset[i]) != 1))
                InvalidArgument("NDArrayView::Slice: Cannot create a slice which is not contiguous in memory. "
                    "This NDArrayView shape = %S, slice offset = %S, slice extent = %S.",
                    Shape().AsString().c_str(), NDShape(startOffset).AsString().c_str(), NDShape(extent).AsString().c_str());

            bool isCurrentAxisSliced = (startOffset[i] != 0) || (endOffset[i] != Shape()[i]);
            anyPrevAxisSliced = anyPrevAxisSliced || isCurrentAxisSliced;
        }

        // determine the canonical matrix shape of our storage object
        if (IsSparse())
        {
            if (rank == 0)
                LogicError("NDArrayView::Slice: Scalars cannot be sparse.");
            if (startOffset.front() != 0 || endOffset.front() != Shape().Dimensions().front())
                InvalidArgument("NDArrayView::Slice: The first axis of a sparse tensor cannot be slice-viewed.");
        }
        auto tensorShape = AsTensorShape(Shape());
        auto flatBufferOffset = tensorShape.Locate(startOffset);  // offset and length into underlying ElementType array...
        auto flatBufferLength = tensorShape.Locate(lastOffset) + 1 - flatBufferOffset; // ...which is known to be consecutive

        ToMatrixShape(tensorShape, NDArrayView::AutoSelectRowColSplitPoint, NDArrayView::AutoSelectRowColSplitPoint);
        let storedRows = tensorShape[0];
        let storedCols = tensorShape[1];

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
            const auto& sob = GetTensorViewPtr<float>()->GetSOBViewPtr();
            if (!anyPrevAxisSliced)
                // Nothing was sliced: current SOB is just fine.
                matrix = sob;
            else if (flatBufferOffset % storedRows == 0 && flatBufferLength % storedRows == 0)
                // SliceView() can be expressed as column slice. This slices the range as columns.
                // In order to not have to care about the actual SOB shape, we tell ColumnSlice()
                // to interpret it as having 'storedCols', w.r.t. which we had found that we can column-slice.
                matrix = make_shared<Matrix<float>>(std::move(sob->ColumnSlice(flatBufferOffset / storedRows,
                                                                               flatBufferLength / storedRows,
                                                                               storedCols)));
            else
                // SliceView() cannot be expressed as column slice. This does the following:
                //  - reinterpret the SOB as a row vector of (storedRows * storedCols)
                //  - column-slice the desired elements (which are really a range of row elements)
                //  - reshape it back to a column
                matrix = make_shared<Matrix<float>>(std::move(sob->ColumnSlice(flatBufferOffset, flatBufferLength, storedRows * storedCols).
                                                                   Reshape(flatBufferLength, 1)));
            break;
        }
        case DataType::Double:
        {
            const auto& sob = GetTensorViewPtr<double>()->GetSOBViewPtr();
            if (!anyPrevAxisSliced)
                // Nothing was sliced: current SOB is just fine.
                matrix = sob;
            else if (flatBufferOffset % storedRows == 0 && flatBufferLength % storedRows == 0)
                // SliceView() can be expressed as column slice. This slices the range as columns.
                // In order to not have to care about the actual SOB shape, we tell ColumnSlice()
                // to interpret it as having 'storedCols', w.r.t. which we had found that we can column-slice.
                matrix = make_shared<Matrix<double>>(std::move(sob->ColumnSlice(flatBufferOffset / storedRows,
                                                                                flatBufferLength / storedRows,
                                                                                storedCols)));
            else
                // SliceView() cannot be expressed as column slice. This does the following:
                //  - reinterpret the SOB as a row vector of (storedRows * storedCols)
                //  - column-slice the desired elements (which are really a range of row elements)
                //  - reshape it back to a column
                matrix = make_shared<Matrix<double>>(std::move(sob->ColumnSlice(flatBufferOffset, flatBufferLength, storedRows * storedCols).
                                                                    Reshape(flatBufferLength, 1)));
            break;
        }
        default:
            LogicError("NDArrayView::Slice: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        return MakeSharedObject<NDArrayView>(GetDataType(), sliceViewShape, IsReadOnly() || readOnly, matrix);
    }

    // TODO: This case is covered by using a shorter extent above; so just implement it with that.
    NDArrayViewPtr NDArrayView::IndexLastAxis(size_t index, bool readOnly) const
    {
        auto rank = Shape().Rank();
        if (rank == 0)
            InvalidArgument("NDArrayView::IndexLastAxis: Cannot index a scalar.");

        auto sliceViewShape = m_viewShape.SubShape(0, rank - 1); // final shape drops final axis
        // if last axis already has only 1 element then just reshape it away
        if (m_viewShape[rank - 1] == 1)
            return AsShape(sliceViewShape);
        // set startOffset to 0 except for the last axis

        std::vector<size_t> startOffset(rank, 0);
        startOffset[rank - 1] = index;
#if 1
        // Slice() keeps as many axes as extent[] has.
        // Any additional axes are sliced to 1 element. So by passing sliceViewShape,
        // which has been striped of the last axis, as extent[], the last axis will be
        // sliced to 1 element and removed.
        //return SliceView(startOffset, /*extent=*/ sliceViewShape.Dimensions(), readOnly);
        return Slice(startOffset, /*extent=*/ sliceViewShape.Dimensions(), vector<size_t>(), SliceMode::View, readOnly);
#else
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
#endif
    }

    NDArrayViewPtr NDArrayView::AsShape(const NDShape& newShape) const
    {
        if (newShape.TotalSize() != Shape().TotalSize())
        {
            InvalidArgument("NDArrayView::AsShape: The total size (%d) of this view's shape '%S' must be same as the size (%d) of the newShape '%S'.",
                            (int)Shape().TotalSize(), Shape().AsString().c_str(),
                            (int)newShape.TotalSize(), newShape.AsString().c_str());
        }

        // BUGBUG: We cannot just use the SOBPtr because with Slice(::View), we may have an offset.
        //         Instead, we should just do this the TensorView way. Reshape requires contiguous data.
        //return MakeSharedObject<NDArrayView>(GetDataType(), newShape, IsReadOnly(), GetStorageObjectPtr());
        auto tensorShape = GetTensorShape();
        tensorShape.ReshapeInPlace(newShape.Dimensions());
        return MakeSharedObject<NDArrayView>(GetDataType(), tensorShape, IsReadOnly(), GetStorageObjectPtr());
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
            InvalidArgument("The stroage format of 'this' NDArrayView is sparse. Please use SparseDataBuffers().");

        // First make sure that the underlying matrix is on the right device
        //auto matrix = GetMatrix<ElementType>();
        //matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        // We transfer the underlying object before taking a view, since objects with views on them cannot be transferred.
        // Note: In Dynamite, most NDArrayViews share one underlying big matrix. Those can never be transferred.
        // TODO: Double-check that. Maybe multiple TensorViews pointing to a single SOB can transfer.
        //       That would be a huge perf hit if the entire arena gets transferred.
        GetTensorViewPtr<ElementType>()->GetSOBPtr()->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        return GetTensorViewPtr<ElementType>()->GetSOBViewPtr()->Data();
    }

    template <typename ElementType>
    std::tuple<const ElementType *, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::SparseDataBuffers: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (!IsSparse())
            RuntimeError("The storage format of 'this' NDArrayView is dense. Please use another DataBuffer().");

        if(GetStorageFormat() != StorageFormat::SparseCSC)
            RuntimeError("The SparseCSCDataBuffers() method only supports CSC sparse format.");

        std::shared_ptr<const Matrix<ElementType>> matrix = GetMatrix<ElementType>();
        auto matrixDims = GetMatrixDimensions(Shape());
        if (matrix->GetNumRows() != matrixDims.first)
            LogicError("The number of rows of the underlying matrix does not match the shape.");
        if (matrix->GetNumCols() != matrixDims.second)
            LogicError("The number of columns of the underlying matrix does not match the shape.");

        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        if ((matrix->GetMatrixType() != Microsoft::MSR::CNTK::MatrixType::SPARSE) || (matrix->GetFormat() != Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC))
            RuntimeError("NDArrayView::SparseDataBuffers: The underlying matrix of 'this' NDArrayView is not in the CSC sparse format.");

        size_t numNonZeroValues;
        ElementType* nonZeroValues;
        SparseIndexType* colStarts;
        SparseIndexType* rowIndices;
        if (m_device.Type() == DeviceKind::CPU)
        {
            if (sizeof(CPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on CPU.");
            std::shared_ptr<Microsoft::MSR::CNTK::CPUSparseMatrix<ElementType>> sparseMatrix = matrix->m_CPUSparseMatrix;
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<ElementType *>(sparseMatrix->NzValues());
            colStarts = static_cast<SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices = static_cast<SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else if (m_device.Type() == DeviceKind::GPU)
        {
            if (sizeof(GPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on GPU.");
            std::shared_ptr<Microsoft::MSR::CNTK::GPUSparseMatrix<ElementType>> sparseMatrix = matrix->m_GPUSparseMatrix;
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<ElementType *>(sparseMatrix->NzValues());
            colStarts = static_cast<SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices = static_cast<SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else
        {
            RuntimeError("NDArrayView::SparseDataBuffers: The device %S is currently not supported.",DeviceKindName(m_device.Type()));
        }

        return std::tuple<ElementType *, SparseIndexType *, SparseIndexType *, size_t>(nonZeroValues, colStarts, rowIndices, numNonZeroValues);
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
#if 1
        if (Shape().TotalSize() != 1)
            LogicError("NDArrayView::AsScalar: The NDArrayView shaped '%S' is not a scalar.", Shape().AsString().c_str());

        if (GetDataType() == DataType::Float)
            return (ElementType)GetTensorViewPtr<float>()->GetSOBViewPtr()->Get00Element();
        else if (GetDataType() == DataType::Double)
            return (ElementType)GetTensorViewPtr<double>()->GetSOBViewPtr()->Get00Element();
        else
            LogicError("NDArrayView::AsScalar: Unsupported DataType");
#else
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
#endif
    }

    std::wstring NDArrayView::AsString() const
    {
        wstringstream wss;
        std::wstring device = DeviceKindName(m_device.Type());
        wss << L"NDArrayView(" << m_viewShape.AsString() << L", " << device << L")";
        return wss.str();
    }

    // log a tensor to a file, e.g. stderr, for debugging purposes
    void NDArrayView::LogToFile(const std::wstring& name, FILE* f, size_t maxItems /*= 6*/, bool columnMajor /*= true*/) const
    {
        std::string asString;
        switch (m_dataType)
        {
        case DataType::Float:
            asString = GetTensorViewPtr<float>()->AsString(maxItems, !Internal::IsReversingTensorShapesInErrorMessagesEnabled());
            break;
        case DataType::Double:
            asString = GetTensorViewPtr<double>()->AsString(maxItems, !Internal::IsReversingTensorShapesInErrorMessagesEnabled());
            break;
        default:
            LogicError("NDArrayView::LogToFile: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
        if (!name.empty())
            fprintf(f, "%S : %s%S =\n", name.c_str(), DataTypeName(GetDataType()), Shape().AsString().c_str());
        fprintf(f, "%s\n", asString.c_str());
    }

    // Explicit template instantiations
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<double>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<float>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<double>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API const float* NDArrayView::DataBuffer<float>() const;
    template CNTK_API const double* NDArrayView::DataBuffer<double>() const;

    // TODO: Was this changed inconsistently between master and fseide/dynamite? Check!
    //template CNTK_API const TensorView<float>* NDArrayView::GetTensorView<float>() const;
    //template CNTK_API const TensorView<double>* NDArrayView::GetTensorView<double>() const;

    template CNTK_API std::tuple<const float*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<float>() const;
    template CNTK_API std::tuple<const double*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<double>() const;

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
