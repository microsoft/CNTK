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

#ifdef _MSC_VER // (causes link errors on gcc)
    __forceinline
#endif
    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly/* = false*/)
        : NDArrayView(dataType, viewShape, readOnly, CreateStorageObject(dataType, viewShape, device, dataBuffer, bufferSizeInBytes))
    {
    }

    template <typename ElementType>
#ifdef _MSC_VER // (causes link errors on gcc)
    __forceinline
#endif
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

    // same but with known ElementType
    template<typename ElementType>
    static __forceinline void* ConstructTensorView(Internal::TensorViewUnion& space, const typename Matrix<ElementType>::MatrixPtr& sob, const TensorShape& shape)
    {
        // placement-construct it inside m_tensorViewUnion, passed to this function as 'space'
        return new ((TensorView<ElementType>*)&space) TensorView<ElementType>(sob, shape);
    }
    // ElementType-erasing version of new TensorView(sob, shape), based on dataType.
    static_assert(sizeof(TensorView<float> ) == sizeof(Internal::TensorViewUnion), "TensorViewUnion has wrong size");
    static_assert(sizeof(TensorView<double>) == sizeof(Internal::TensorViewUnion), "TensorViewUnion has wrong size");
    static void* ConstructTensorView(Internal::TensorViewUnion& space, CNTK::DataType dataType, const shared_ptr<MatrixBase>& sob, const TensorShape& shape)
    {
        // placement-construct it inside m_tensorViewUnion, passed to this function as 'space'
        switch (dataType)
        {
        case DataType::Float:
        {
            auto matrix = dynamic_pointer_cast<Matrix<float>>(sob);
            if (matrix)
                return ConstructTensorView<float>(space, matrix, shape);
        }
        break;
        case DataType::Double:
        {
            auto matrix = dynamic_pointer_cast<Matrix<double>>(sob);
            if (matrix)
                return ConstructTensorView<double>(space, matrix, shape);
        }
        break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
        LogicError("Storage Object is not of DataType %s", DataTypeName(dataType));
    }
    // same but using static cast, without error check  --TODO: remove the code dup
    static __forceinline void* ConstructTensorViewStatic(Internal::TensorViewUnion& space, CNTK::DataType dataType, const shared_ptr<MatrixBase>& sob, const TensorShape& shape)
    {
        // placement-construct it inside m_tensorViewUnion, passed to this function as 'space'
        switch (dataType)
        {
        case DataType::Float:  return ConstructTensorView<float> (space, static_pointer_cast<Matrix<float >>(sob), shape);
        case DataType::Double: return ConstructTensorView<double>(space, static_pointer_cast<Matrix<double>>(sob), shape);
        }
        LogicError("Unsupported DataType %s", DataTypeName(dataType));
    }
    static __forceinline void DestructTensorView(Internal::TensorViewUnion& space, CNTK::DataType dataType)
    {
        // placement-delete
        switch (dataType)
        {
        case DataType::Float:
           ((TensorView<float>*)&space)->~TensorView<float>();
           break;
        case DataType::Double:
            ((TensorView<double>*)&space)->~TensorView<double>();
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(dataType));
            break;
        }
    }
#if 0
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
#endif

#define LAZY_2D_PADDING // if defined then rank-2 padding of TensorShapes happens upon access, not upon creation

    // constructor optimized for shape passed as NDShape (this constructs a dense object)
    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, bool readOnly, const MatrixBasePtr& sob)
        : m_dataType(dataType), m_device(AsDeviceDescriptor(sob->GetDeviceId())), m_storageFormat(AsStorageFormat(sob->GetFormat())), m_viewShape(viewShape), m_isReadOnly(readOnly)
    {
#ifdef LAZY_2D_PADDING
        const auto tensorShape = AsTensorShape(viewShape);
#else
        const auto tensorShape = AsTensorShapeMin2D(viewShape); // not lazy (old version): sdo it here and bake it into teh object
#endif
        ConstructTensorView(m_tensorViewUnion, dataType, sob, tensorShape);
    }

    // constructor optimized for arena allocation (shape passed as NDShape; potential offset into Matrix object)
    NDArrayView::NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, size_t beginOffset, size_t endOffset, const shared_ptr<MatrixBase>& sob)
        : m_dataType(dataType), m_device(AsDeviceDescriptor(sob->GetDeviceId())), m_storageFormat(AsStorageFormat(sob->GetFormat())), m_viewShape(viewShape), m_isReadOnly(false)
    {
#ifdef LAZY_2D_PADDING
        auto tensorShape = AsTensorShape(viewShape);
        // TODO: Can we use a constructor that also sets the offset?
#else
        auto tensorShape = AsTensorShapeMin2D(viewShape); // not lazy (old version): sdo it here and bake it into teh object
#endif
        if (beginOffset)
        {
            // TODO: check the endOffset
            if (sob->GetMatrixType() != MatrixType::DENSE)
                InvalidArgument("This specific NDArayView constructor presently does not support sparse storage objects.");
            //if (sob->GetNumRows() != 1) // seems this is not a virtual function...
            //    InvalidArgument("This specific NDArayView constructor presently assumes a column vector.");
            tensorShape.OverwriteOffsetAs(beginOffset);
            // TODO: this is not nice, conflating two constructors, needing hacks like overwriting m_offset. Separate out the one for arena allocation.
        }
        //m_tensorViewPtr = NewTensorView(dataType, sob, tensorShape);
        ConstructTensorViewStatic(m_tensorViewUnion, dataType, sob, tensorShape); // -Static means that we won't double-check the dynamic type of the sob
    }

    // constructor optimized for shape passed as TensorShape (allowing strides and offset for slicing)
    NDArrayView::NDArrayView(CNTK::DataType dataType, const TensorShape& tensorShape, bool readOnly, const shared_ptr<MatrixBase>& sob, bool sobTypeAlreadyVerified) :
        m_dataType(dataType), m_device(AsDeviceDescriptor(sob->GetDeviceId())), m_storageFormat(AsStorageFormat(sob->GetFormat())),
        m_viewShape(tensorShape.GetDims().begin(), tensorShape.GetDims().end()),
        m_isReadOnly(readOnly)
    {
        //m_tensorViewPtr = NewTensorView(dataType, sob, tensorShape);
        if (sobTypeAlreadyVerified)
            ConstructTensorViewStatic(m_tensorViewUnion, dataType, sob, tensorShape);
        else
            ConstructTensorView(m_tensorViewUnion, dataType, sob, tensorShape);
    }

    // create a new NDArrayView that subplants the tensorShape
    // TensorShape includes offset and stride info, hence this can be used to implement slices as well.
    // All methods that create a new view onto an existing NDArrayView use this.
    NDArrayViewPtr NDArrayView::Reviewed(const Microsoft::MSR::CNTK::TensorShape& tensorShape, bool readOnly) const
    {
        switch (m_dataType)
        {
        case DataType::Float:
            return MakeSharedObject<NDArrayView>(GetDataType(), tensorShape, readOnly, NativeTensorView<float>().GetSOBPtr(), true);
        case DataType::Double:
            return MakeSharedObject<NDArrayView>(GetDataType(), tensorShape, readOnly, NativeTensorView<double>().GetSOBPtr(), true);
        default:
            LogicError("NDArrayView::GetStorageObjectPtr: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    NDArrayView::NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device)
        : NDArrayView(dataType, viewShape, false, CreateStorageObject(dataType, storageType, viewShape, device))
    {}

    NDArrayView::~NDArrayView()
    {
        DestructTensorView(m_tensorViewUnion, m_dataType);
    }

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

    // BUGBUG: This does not honor offsets. Use opConstOne to set it instead.
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

    void NDArrayView::SetValue(const double* data, size_t size)
    {
        switch (m_dataType)
        {
        case DataType::Float:
            WritableNativeTensorView<float>().GetSOBViewPtr()->AssignValues(data, size);
            break;
        case DataType::Double:
            WritableNativeTensorView<float>().GetSOBViewPtr()->AssignValues(data, size);
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
    }

    /*static*/ double NDArrayView::Sync(const DeviceDescriptor& device)
    {
        return Matrix<float>::SyncDevice(AsCNTKImplDeviceId(device));
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

#if 0
    // This function is no longer used (inlined into GetMatrix()).
    // TODO: This processes the TensorShape twice. We should just inline GetMatrixImpl().
    template <typename ElementType>
    /*static*/ Matrix<ElementType>::MatrixPtr NDArrayView::GetMatrixImpl(const TensorView<ElementType>& paddedTensorView, size_t rowColSplitPoint)
    {
        // only contiguous tensors can be processed by the Matrix lib
        paddedTensorView.GetShape().VerifyIsDense();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        if (paddedTensorView.GetShape().GetRank() <= 1 && rowColSplitPoint != 0)
            return paddedTensorView.AsMatrix();

        auto tensorShape = paddedTensorView.GetShape();
        ToMatrixShape(tensorShape, rowColSplitPoint, AutoSelectRowColSplitPoint);

        return paddedTensorView.Reviewed(tensorShape).AsMatrix();
    }

    // -ViewMin2D: use if you interop with V1 code that needs shapes of rank 2 or higher
    // These versions are only ever called by GetMatrix(). We could just inline them here.
    // Especially that we now no longer have a real shared_ptr to return.
    template <typename ElementType>
    std::shared_ptr<const Microsoft::MSR::CNTK::TensorView<ElementType>> NDArrayView::GetTensorViewMin2D() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetTensorViewMin2D: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        let& tensorView = NativeTensorView<ElementType>(); // *static_pointer_cast<const TensorView<ElementType>>(m_tensorViewPtr);
#ifdef LAZY_2D_PADDING
        let& shape = tensorView.GetShape();
        if (shape.size() < 2) // we must pad to at least 2D
            //auto paddedShape = AsTensorShapeMin2D(shape); // adds 1-dimensions if rank < 2
            return make_shared<TensorView<ElementType>>(tensorView.Reviewed(AsTensorShapeMin2D(shape)));
#endif
        return make_shared<TensorView<ElementType>>(tensorView);
        //return tensorView;
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
    typename Matrix<ElementType>::ConstMatrixPtr NDArrayView::GetMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const
    {
        if (AsDataType<ElementType>() != m_dataType)
            LogicError("NDArrayView::GetMatrix: The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(m_dataType));

        let& tensorView = NativeTensorView<ElementType>(); // *static_pointer_cast<const TensorView<ElementType>>(m_tensorViewPtr);
#ifdef LAZY_2D_PADDING
        let& shape = tensorView.GetShape();
        let& paddedTensorView =
            /*if*/ (shape.size() >= 2) ?
                tensorView
            /*else*/:
                tensorView.Reviewed(AsTensorShapeMin2D(shape));
        //if (shape.size() < 2) // we must pad to at least 2D
        //    auto paddedShape = AsTensorShapeMin2D(shape); // adds 1-dimensions if rank < 2
        //    return GetMatrixImpl<ElementType>(tensorView.Reviewed(paddedShape), rowColSplitPoint);
#else
        let& paddedTensorView = tensorView;
#endif
        //return GetMatrixImpl<ElementType>(paddedTensorView, rowColSplitPoint);

        // only contiguous tensors can be processed by the Matrix lib
        paddedTensorView.GetShape().VerifyIsDense();

        // we should always reshape for rank-0, so that batch and sequence axis goes to columns
        //if (paddedTensorView.GetShape().GetRank() <= 1 && rowColSplitPoint != 0) // BUGBUG: This is never possibl since we pad to rank-2, or is it? Or can we remove the padding above?
        //    return paddedTensorView.AsMatrix();

        auto tensorShape = paddedTensorView.GetShape();
        ToMatrixShape(tensorShape, rowColSplitPoint, AutoSelectRowColSplitPoint); // TODO: do the IsDense() check in here

        return paddedTensorView.Reviewed(tensorShape).AsMatrix();
    }

    template <typename ElementType>
    typename Matrix<ElementType>::MatrixPtr NDArrayView::GetWritableMatrix(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/)
    {
        let* thisc = const_cast<NDArrayView*>(this);
        let resc = thisc->GetMatrix<ElementType>(rowColSplitPoint);
        return const_pointer_cast<Matrix<ElementType>>(resc);
        //return GetMatrixImpl<ElementType>(*GetWritableTensorViewMin2D<ElementType>(), rowColSplitPoint);
    }

    // WARNING! The SOBPtr does not necessarily represent the offset field of the TensorShape.
    // So one should never use the SOB for Matrix operations directly. Use AsMatrix() or TensorView operations instead.
    // TODO: Remove this function altogether; and replace with creating an NDArrayView from a TensorShape and an old NDArrayView.
    //shared_ptr<MatrixBase> NDArrayView::GetStorageObjectPtr() const
    //{
    //    switch (m_dataType)
    //    {
    //    case DataType::Float:
    //        return NativeTensorView<float>().GetSOBPtr();
    //    case DataType::Double:
    //        return NativeTensorView<double>().GetSOBPtr();
    //    default:
    //        LogicError("NDArrayView::GetStorageObjectPtr: Unsupported DataType %s", DataTypeName(m_dataType));
    //    }
    //}

    NDArrayViewPtr NDArrayView::DeepClone(const DeviceDescriptor& device, bool readOnly/* = false*/) const
    {
        NDArrayViewPtr newView = MakeSharedObject<NDArrayView>(this->GetDataType(), this->GetStorageFormat(), this->Shape(), device);
        // For dense data on the same device, we use TensorView, which will amount to a cudaMemcpy() for contiguous data
        // (bypassing GetMatrix() complexity) and handle non-contiguous tensors.
        bool useTensorView = !IsSparse() && Device() == device;
        auto* us = const_cast<NDArrayView*>(this); // (need to go through a writable ref below)
        switch (m_dataType)
        {
        case DataType::Float:
            if (useTensorView)
                TensorView<float>::template Do<2>(1, { std::ref(us->WritableNativeTensorView<float>()), std::ref(newView->WritableNativeTensorView<float>()) }, ElementWiseOperator::opCopy, ElementWiseOperator::opSum, /*alpha=*/1, /*beta=*/0);
            else
                newView->GetWritableMatrix<float>()->AssignValuesOf(*GetMatrix<float>());
            break;
        case DataType::Double:
            if (useTensorView)
                TensorView<double>::template Do<2>(1, { std::ref(us->WritableNativeTensorView<double>()), std::ref(newView->WritableNativeTensorView<double>()) }, ElementWiseOperator::opCopy, ElementWiseOperator::opSum, /*alpha=*/1, /*beta=*/0);
            else
                newView->GetWritableMatrix<double>()->AssignValuesOf(*GetMatrix<double>());
            break;
        default:
            LogicError("NDArrayView::DeepClone: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }

        newView->m_isReadOnly = readOnly;
        return newView;
    }

    // BUGBUG: Does not work for sliced tensors. We can use TensorView assignment, except for sparse. Test with doSplice in Tests.cpp.
    void NDArrayView::CopyFrom(const NDArrayView& source)
    {
        if ((source.Shape() != Shape()) && (AsTensorShape(source.Shape()) != AsTensorShape(Shape())))
            InvalidArgument("NDArrayView::CopyFrom: The source view shape '%S' is not same as the shape '%S' of this NDArrayView.", 
                            source.Shape().AsString().c_str(), Shape().AsString().c_str());

        if (IsReadOnly())
            RuntimeError("NDArrayView::CopyFrom: Cannot modify contents of a readonly NDArrayView.");

        // TODO: like DeepClone, for dense data, this can call TensorView, which will amount to a cudaMemcpy() while bypassing GetMatrix() complexity
        //       Maybe we need a shared copy function. Maybe DeepClone can call CopyFrom()?
#if 0   // #if 1 to allow loading 'float' model as 'double'
        if (m_dataType == DataType::Double && source.m_dataType == DataType::Float)
        {
            auto sourceMatrix = source.GetMatrix<float>();
            auto destMatrix = GetWritableMatrix<double>();
            std::vector<float> fBuffer;
            source.CopyDataTo(fBuffer);
            std::vector<double> dBuffer(fBuffer.begin(), fBuffer.end());
            SetValue(dBuffer.data(), dBuffer.size());
        }
        else
#endif
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
        //return MakeSharedObject<NDArrayView>(GetDataType(), GetTensorShape(), IsReadOnly() || readOnly, GetStorageObjectPtr());
        return Reviewed(GetTensorShape(), IsReadOnly() || readOnly);
        // MakeSharedObject<NDArrayView>(GetDataType(), tensorShape, readOnly, GetStorageObjectPtr());
    }

    template <typename ElementType>
    static bool AreAliases(const TensorView<ElementType>& aView, const TensorView<ElementType>& bView)
    {
        let& aTensorShape = aView.GetShape();
        let& bTensorShape = bView.GetShape();
        if (aTensorShape.GetDims() != bTensorShape.GetDims()) // shape must be the same. This is silly--the only time we call this, we have already compared the shape.
            return false;
        if (aTensorShape.GetStrides() != bTensorShape.GetStrides()) // strides must be the same
            return false;
        if (&aView.GetSOB() == &bView.GetSOB() && aTensorShape.GetOffset() == bTensorShape.GetOffset()) // same SOB and same offset: OK
            return true;
#if 1   // BUGBUG: The test below does not work for sparse. For now, let's pretend shifted sparse matrices are not the same, which will break CSE.
        if (aView.GetSOB().GetMatrixType() != MatrixType::DENSE)
            return false;
#endif
        return aView.GetSOB().Data() + aTensorShape.GetOffset() == bView.GetSOB().Data() + bTensorShape.GetOffset(); // otherwise compute buffer address and compare
    }

    bool NDArrayView::IsAliasOf(const NDArrayViewPtr& other) const
    {
        // note: this should not waste cycles, since it is used inside auto-batching
        if (this == other.get()) // fast path: same object
            return true;
        if (m_dataType != other->m_dataType)
            return false;
        if (IsSparse() != other->IsSparse())
            return false;
        switch (m_dataType)
        {
        case DataType::Float:  return AreAliases(NativeTensorView<float> (), other->NativeTensorView<float> ());
        case DataType::Double: return AreAliases(NativeTensorView<double>(), other->NativeTensorView<double>());
        default: LogicError("NDArrayView::CopyFrom: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    //static DataType GetType(float)  { return DataType::Float;  }
    //static DataType GetType(double) { return DataType::Double; }

    // these are strictly for internal use, so we don't need additional error checks
    template <typename ElementType>
    const Microsoft::MSR::CNTK::TensorView<ElementType>& NDArrayView::NativeTensorView() const
    {
        //return *(const TensorView<ElementType>*)(m_tensorViewPtr.get());
        return *(const TensorView<ElementType>*)(&m_tensorViewUnion);
    }

    template <typename ElementType>
    Microsoft::MSR::CNTK::TensorView<ElementType>& NDArrayView::WritableNativeTensorView()
    {
        let* thisc = const_cast<NDArrayView*>(this);
        let& vc = thisc->NativeTensorView<ElementType>();
        return const_cast<TensorView<ElementType>&>(vc);
        //return *(TensorView<ElementType>*)(m_tensorViewPtr.get());
    }

    // templated helper to execute TensorView operations
    template <typename ElementType, size_t N>
    /*static*/ inline void NativeNumericOperation(const std::array<NDArrayView*, N>& args, int opInt, int reductionOpInt, double alpha, double beta)
    {
        // note: ^^ static does not work because gcc does not accept "static" in "friend" decl, but MSVC takes the non-static friend decl as a declaration
        const auto          op = (Microsoft::MSR::CNTK::ElementWiseOperator)opInt;
        const auto reductionOp = reductionOpInt != -1 ? (Microsoft::MSR::CNTK::ElementWiseOperator)reductionOpInt : Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
        TensorView<ElementType>::template Do<N>(N-1, MapArray(args, [](NDArrayView* view) { return std::ref(view->WritableNativeTensorView<ElementType>()); }), op, reductionOp, (ElementType)alpha, (ElementType)beta);
        // Note: Only the last element of args[] is written to, but for regularity of interface, we pass all as writable.
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
        switch (out->m_dataType)
        {
        case DataType::Float:
            switch (inputs.size())
            {
            case 0: NativeNumericOperation<float, 1>({                                                                     out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 1: NativeNumericOperation<float, 2>({ inputs[0].get(),                                                    out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 2: NativeNumericOperation<float, 3>({ inputs[0].get(), inputs[1].get(),                                   out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 3: NativeNumericOperation<float, 4>({ inputs[0].get(), inputs[1].get(), inputs[2].get(),                  out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 4: NativeNumericOperation<float, 5>({ inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get(), out.get() }, opInt, reductionOpInt, alpha, beta); break;
            default: LogicError("NDArrayView::NumericOperation: Invalid number of inputs: %d", (int)inputs.size());
            }
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            switch (inputs.size())
            {
            case 0: NativeNumericOperation<double, 1>({                                                                     out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 1: NativeNumericOperation<double, 2>({ inputs[0].get(),                                                    out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 2: NativeNumericOperation<double, 3>({ inputs[0].get(), inputs[1].get(),                                   out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 3: NativeNumericOperation<double, 4>({ inputs[0].get(), inputs[1].get(), inputs[2].get(),                  out.get() }, opInt, reductionOpInt, alpha, beta); break;
            case 4: NativeNumericOperation<double, 5>({ inputs[0].get(), inputs[1].get(), inputs[2].get(), inputs[3].get(), out.get() }, opInt, reductionOpInt, alpha, beta); break;
            default: LogicError("NDArrayView::NumericOperation: Invalid number of inputs: %d", (int)inputs.size());
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
            out->WritableNativeTensorView<float>().DoMatrixProductOf((float)beta, transC, inputA->NativeTensorView<float>(), transA, inputB->NativeTensorView<float>(), transB, (float)alpha);
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            out->WritableNativeTensorView<double>().DoMatrixProductOf((double)beta, transC, inputA->NativeTensorView<double>(), transA, inputB->NativeTensorView<double>(), transB, (double)alpha);
            break;
        default:
            LogicError("NDArrayView::MatrixProduct: Unsupported DataType %s", DataTypeName(out->m_dataType));
            break;
        }
        return out;
    }

    // debugging helper
    // Cut this into return statements to look at the result.
    static inline NDArrayViewPtr LogHelper(const wstring& name, NDArrayViewPtr view)
    {
        let clone = view->IsSparse() ? view : view->NumericOperation({ view }, 1.0, opCopy); // make a copy since we can only Log contigous ones for now
        clone->LogToFile(name);
        return view;
    }

    // output is sparse or dense depending on 'out'
    /*static*/ NDArrayViewPtr NDArrayView::AsOneHot(NDArrayViewPtr arg, size_t axis, NDArrayViewPtr out)
    {
        // create result object if not given
        if (!out)
            InvalidArgument("NDArrayView::AsOneHot: output matrix must presently be supplied"); // can be fixed later if we care
        if (out->IsSparse() && axis != 0)
            InvalidArgument("NDArrayView::AsOneHot: sparse OneHot is only supported for axis 0"); // can be fixed later if we care

        // PERF BUGBUG: Don't use vector<> for the shape!
        // TODO: do this more nicely, e.g. move into TensorView as an op
        const auto& outDims = out->Shape().Dimensions();
        std::vector<size_t> outShape(outDims.begin(), outDims.end());
        switch (arg->m_dataType)
        {
        case DataType::Float:
            out->WritableNativeTensorView<float>().GetSOBViewPtr()->AssignOneHot(*arg->NativeTensorView<float>().GetSOBViewPtr(), outShape, axis, out->IsSparse());
            break;
        case DataType::Double:
            out->WritableNativeTensorView<double>().GetSOBViewPtr()->AssignOneHot(*arg->NativeTensorView<double>().GetSOBViewPtr(), outShape, axis, out->IsSparse());
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(arg->m_dataType));
            break;
        }
        return out;
    }

    template<typename ElementType>
    class TensorViewPtrArrayRef : public TensorView<ElementType>::template IArrayRef<TensorView<ElementType> const*>
    {
        const vector<NDArrayViewPtr>& m_inputs;
    public:
        TensorViewPtrArrayRef(const vector<NDArrayViewPtr>& inputs) : m_inputs(inputs) { }
        virtual size_t size() const { return m_inputs.size(); }
        virtual TensorView<ElementType> const** data() const { NOT_IMPLEMENTED; }
        // TODO: do we need a check here?
        virtual TensorView<ElementType> const* /*const&*/ operator[](size_t i) const { return &m_inputs[i]->NativeTensorView<ElementType>(); }
        virtual TensorView<ElementType> const* & operator[](size_t i) { NOT_IMPLEMENTED; }
        virtual TensorView<ElementType> const* const* begin() const { NOT_IMPLEMENTED; }; // TODO: get the const-ness thingy right
        virtual TensorView<ElementType> const* const * end() const { NOT_IMPLEMENTED; }
    };

    template<typename ElementType> // TODO: unify these two
    class WritableTensorViewPtrArrayRef : public TensorView<ElementType>::template IArrayRef<TensorView<ElementType>*>
    {
        vector<NDArrayViewPtr>& m_inputs;
    public:
        WritableTensorViewPtrArrayRef(vector<NDArrayViewPtr>& inputs) : m_inputs(inputs) { }
        virtual size_t size() const { return m_inputs.size(); }
        virtual TensorView<ElementType>** data() const { NOT_IMPLEMENTED; }
        // TODO: do we need a check here?
        virtual TensorView<ElementType>* /*const&*/ operator[](size_t i) const { return &m_inputs[i]->WritableNativeTensorView<ElementType>(); }
        virtual TensorView<ElementType>* & operator[](size_t i) { NOT_IMPLEMENTED; }
        virtual TensorView<ElementType>* const* begin() const { NOT_IMPLEMENTED; }; // TODO: get the const-ness thingy right
        virtual TensorView<ElementType>* const * end() const { NOT_IMPLEMENTED; }
    };

    // unroll a sequence of (sparse) inputs along their last axis, for subsequent Gather or Scatter
    static vector<NDArrayViewPtr> Unroll(const vector<NDArrayViewPtr>& inputs, size_t axisForChecking, size_t sizeHint)
    {
        vector<NDArrayViewPtr> unrolledInputs;
        unrolledInputs.reserve(sizeHint);
        for (let& input : inputs)
        {
            let dims = input->Shape().Dimensions();
            if (dims.size() != axisForChecking + 1)
                InvalidArgument("Gather/Scatter for stacking sparse objects: an input has an incorrect rank");
            let len = dims.back();
            for (size_t i = 0; i < len; i++)
                unrolledInputs.push_back(input->IndexLastAxis(i));
        }
        if (unrolledInputs.size() != sizeHint)
            LogicError("Gather/Scatter for stacking sparse objects has received in incorrect size hint");
        return unrolledInputs;
    }

    /*static*/ NDArrayViewPtr NDArrayView::GatherBatch(const vector<NDArrayViewPtr>& inputs, size_t axis, NDArrayViewPtr out)
    {
        if (!out) //        || true) // keep this for now for testing this
        {
            vector<size_t> totalShape(axis+1, 1); // total shape
            // first check all dims and determine the shared shape
            size_t splicedDim = 0;
            for (let& val : inputs)
            {
                let& shape = val->Shape();
                if (shape.Rank() > totalShape.size())
                    totalShape.resize(shape.Rank(), 1);
                for (size_t k = 0; k < shape.Rank(); k++)
                {
                    if (k != axis && totalShape[k] != shape[k] && totalShape[k] != 1 && shape[k] != 1) // shapes must match, considering broadcasting
                        InvalidArgument("GatherBatch: incompatible shapes");
                    if (shape[k] != 1)
                        totalShape[k] = shape[k]; // collect the axis
                }
                splicedDim += axis < shape.Rank() ? shape[axis] : 1; // accumulate the total dimension for the spliced axis
            }
            // now implant the spliced dimension into totalShape
            totalShape[axis] = splicedDim;
            NDShape shape(move(totalShape));
            if (out && out->Shape() != shape)
                LogicError("NDArrayView::GatherBatch: bad out dim"); // (this err msg will go away after some testing)
            if (!out)
            {
                let& input0 = *inputs[0];
                out = MakeSharedObject<NDArrayView>(input0.GetDataType(), input0.GetStorageFormat(), shape, input0.Device());
            }
        }

        // a special case is sparse. TensorView can presently not stack variable-width sparse tensors.
        // We will unroll all columns. Not very efficient. --TODO: fix TensorView's Gather function
        if (out->IsSparse() && axis == out->Shape().Rank() - 1 && axis == inputs.front()->Shape().Rank() - 1)
            return GatherBatch(Unroll(inputs, axis, out->Shape().Dimensions().back()), axis, out);

        // perform the operation
        // The underlying TensorView call expects a functor to access the TensorView items.
        // Any error checking will happen inside the TensorView function, so we don't duplicate it here.
        switch (out->m_dataType)
        {
        case DataType::Float:
            out->WritableNativeTensorView<float>().DoGatherBatchOf(TensorViewPtrArrayRef<float>(inputs), axis);
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            out->WritableNativeTensorView<double>().DoGatherBatchOf(TensorViewPtrArrayRef<double>(inputs), axis);
            break;
        default:
            LogicError("NDArrayView::GatherBatch: Unsupported DataType %s", DataTypeName(out->m_dataType));
            break;
        }
        return out;
    }

    /*static*/ void NDArrayView::ScatterBatch(const NDArrayViewPtr& input, vector<NDArrayViewPtr>& outputs, size_t axis, double beta)
    {
        // a special case is sparse. TensorView can presently not stack variable-width sparse tensors.
        // We will unroll all columns. Not very efficient. --TODO: fix TensorView's Gather function
        if (input->IsSparse() && axis == input->Shape().Rank() - 1 && axis == outputs.front()->Shape().Rank() - 1)
        {
            auto unrolledOutputs = Unroll(outputs, axis, input->Shape().Dimensions().back());
            return ScatterBatch(input, unrolledOutputs, axis, beta);
        }

        // Unlike GatherBatch(), the target must already have been fully shaped and allocated.
        // Any error checking will happen inside the TensorView function, so we don't duplicate it here.
        switch (input->m_dataType)
        {
        case DataType::Float:
            input->NativeTensorView<float>().DoScatterBatchOf((float)beta, WritableTensorViewPtrArrayRef<float>(outputs), axis);
            break;
        case DataType::Double: // note: keep this block a 100% copy of above, replacing float with double
            input->NativeTensorView<double>().DoScatterBatchOf((double)beta, WritableTensorViewPtrArrayRef<double>(outputs), axis);
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
        case DataType::Float:  return NativeTensorView<float> ().GetShape();
        case DataType::Double: return NativeTensorView<double>().GetShape();
        default: LogicError("NDArrayView::SlicedTensorView: Unsupported DataType %s", DataTypeName(m_dataType));
        }
    }

    // TODO: remove this function if not needed
    bool NDArrayView::IsContiguous() const
    {
        return GetTensorShape().IsDense();
    }

    NDArrayViewPtr NDArrayView::Slice(const NDShapeDimensionsSpan& startOffset, const NDShapeDimensionsSpan& extent, const NDShapeDimensionsSpan& strides, SliceMode sliceMode, bool readOnly) const
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
                //let view = MakeSharedObject<NDArrayView>(m_dataType, tensorShape, readOnly, GetStorageObjectPtr());
                let view = Reviewed(tensorShape, readOnly);

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
        NDShape sliceViewShape(extent);        // note: has same rank as extent
        NDShapeDimensions endOffset(rank, 0);  // note: these have same rank as 'this', not extent
        NDShapeDimensions lastOffset(rank, 0);
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
            const auto& sob = NativeTensorView<float>().GetSOBViewPtr();
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
            const auto& sob = NativeTensorView<double>().GetSOBViewPtr();
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

    NDArrayViewPtr NDArrayView::SliceViewAsShape(size_t beginIndex, size_t endIndex, const NDShape& shape, bool readOnly) const
    {
        if (endIndex <= beginIndex)
            InvalidArgument("NDArrayView::SliceViewAsShape: Invalid index range [%d,%d).", (int)beginIndex, (int)endIndex);
        let rank = Shape().Rank();
        if (rank == 0)
            InvalidArgument("NDArrayView::SliceViewAsShape: Cannot index a scalar.");

        // For sparse input, strided views are not supported in any form.
        // In this case, we just call Slice() with sliceMode ContiguousView.
        // TODO: This is defensive, as there is no test for this presently. But is this necessary? Wouldn't the resulting slice still be contiguous?
        if (IsSparse())
        {
            // if the index range is the full axis, then we only need the reshape
            let& viewDims = m_viewShape.Dimensions();
            if (beginIndex == 0 && endIndex == viewDims.back())
                return AsShape(shape);

            NDShapeDimensions startOffset(rank, 0);
            NDShapeDimensions extent(viewDims);
            startOffset.back() = beginIndex;
            extent     .back() = endIndex - beginIndex;
            return Slice(startOffset, extent, NDShapeDimensions(), SliceMode::ContiguousView, readOnly)->AsShape(shape);
        }

        // determine an updated TensorShape object
        auto tensorShape = GetTensorShape();                  // get current actual TensorShape
        tensorShape.NarrowTo(rank - 1, beginIndex, endIndex); // narrow it down
        tensorShape.ReshapeInPlace(shape.Dimensions());       // and implant the shape

        // create a NDArrayView with this new tensor shape onto the existing storage object
        //return MakeSharedObject<NDArrayView>(m_dataType, tensorShape, readOnly, GetStorageObjectPtr());
        return Reviewed(tensorShape, readOnly);
    }

    NDArrayViewPtr NDArrayView::IndexLastAxis(size_t index, bool readOnly) const
    {
        let rank = Shape().Rank();
        if (rank == 0)
            InvalidArgument("NDArrayView::IndexLastAxis: Cannot index a scalar.");
        let axis = rank - 1;

        // For sparse input, strided views are not supported in any form.
        // In this case, we just call Slice() with sliceMode ContiguousView.
        // TODO: This is defensive, as there is no test for this presently. But is this necessary? Wouldn't the resulting slice still be contiguous?
        if (IsSparse())
        {
            auto sliceViewShape = m_viewShape.SubShape(0, axis); // final shape drops final axis
            // if last axis already has only 1 element then just reshape it away
            if (m_viewShape[axis] == 1)
                return AsShape(sliceViewShape);
            // set startOffset to 0 except for the last axis

            NDShapeDimensions startOffset(rank, 0);
            startOffset[axis] = index;

            return Slice(startOffset, /*extent=*/ sliceViewShape.Dimensions(), NDShapeDimensions(), SliceMode::ContiguousView, readOnly);
        }

        auto tensorShape = GetTensorShape();          // get current actual shape
        tensorShape.NarrowTo(axis, index, index + 1); // narrow it down
        tensorShape.TrimRankInPlace(axis);            // drop trailing singleton dimensions

        // create a NDArrayView with this new tensor shape onto the existing storage object
        // Note that this version may create an NDArrayView with strides, which is not accepted by some operations.
        // TODO: Check that at least this will be discovered. AsMatrix() checks it. Does that cover all use cases?
        //return MakeSharedObject<NDArrayView>(m_dataType, tensorShape, readOnly, GetStorageObjectPtr());
        return Reviewed(tensorShape, readOnly);
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
        //return MakeSharedObject<NDArrayView>(m_dataType, tensorShape, m_isReadOnly, GetStorageObjectPtr());
        return Reviewed(tensorShape, m_isReadOnly);
    }

    NDArrayViewPtr NDArrayView::AsTransposed(const NDShapePermutation& permutation, bool inverted) const
    {
        // invert = false: permutation[i] denotes which original axis will become axis i: newShape[i] <- shape[permutation[i]]
        // invert = true: opposite
        auto tensorShape = GetTensorShape();
        tensorShape.PermuteDimsInPlace(permutation, inverted);
        // result is no longer memory-consecutive
        return Reviewed(tensorShape, IsReadOnly());
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
    // TODO: towards allowing non-dense tensors: Define the DataBuffers as contiguous, and enforce it with a check here.
    template <typename ElementType>
    const ElementType* NDArrayView::DataBuffer() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::DataBuffer: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (IsSparse())
            InvalidArgument("The storage format of 'this' NDArrayView is sparse. Please use SparseDataBuffers().");

        // First make sure that the underlying matrix is on the right device
        //auto matrix = GetMatrix<ElementType>();
        //matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        // We transfer the underlying object before taking a view, since objects with views on them cannot be transferred.
        // Note: In Dynamite, most NDArrayViews share one underlying big matrix. Those can never be transferred.
        // TODO: Double-check that. Maybe multiple TensorViews pointing to a single SOB can transfer.
        //       That would be a huge perf hit if the entire arena gets transferred.
        let& tensorView = NativeTensorView<ElementType>();
        let& sob = tensorView.GetSOB();
        sob.TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        let* dataBuffer = sob.Data() + tensorView.GetShape().GetOffset();
#if 1   // sanity check against old implementation
        //NativeTensorView<ElementType>().GetSOBPtr()->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        // the above version is faster; make sure it is correct
        let* dataBuffer1 = NativeTensorView<ElementType>().GetSOBViewPtr()->Data();
        if (dataBuffer1 != dataBuffer)
            LogicError("NDArrayView::DataBuffer: fast and slow way of getting the data buffer give different answers?");
#endif
        return dataBuffer;
    }

    // copy dense data to a CPU-side data buffer, possibly with type casting
    template <typename ElementType, typename ResultType>
    static void DoCopyDataTo(const typename Matrix<ElementType>::MatrixPtr& from, std::vector<ResultType>& to)
    {
        if (from->GetMatrixType() != MatrixType::DENSE)
            InvalidArgument("NDArrayView::CopyDataTo is presently only implemented for dense data.");
        size_t numElements = from->GetNumElements();
        to.resize(numElements);
        // get a CPU-side pointer to the source data
        // If it lives on the GPU, then we must make a copy; otherwise we point to the source buffer directly.
        ElementType* data;
        std::vector<ElementType> intermediateCPUCopy;
        if (from->GetCurrentMatrixLocation() != CurrentDataLocation::CPU) // must transfer
        {
            if (std::is_same<ElementType, ResultType>::value) // if the type is right, transfer directly into target buffer
                data = (ElementType*)to.data();
            else
            {
                intermediateCPUCopy.resize(numElements); // otherwise use the intermediate buffer
                data = intermediateCPUCopy.data();
            }
            from->CopySection(from->GetNumRows(), from->GetNumCols(), data, /*colStride=*/from->GetNumRows());
        }
        else
            data = from->Data(); // lives on CPU: no need to copy
        // copy and type-cast, unless we had to transfer it, and were able to transfer directly to target
        if (data != (ElementType*)to.data())
        {
            for (size_t i = 0; i < numElements; i++)
                to[i] = (ResultType)data[i];
        }
    }

    // helper to copy dense data from NDArrayView object to a CPU-side data buffer
    template <typename ResultType>
    void NDArrayView::CopyDataTo(std::vector<ResultType>& outputBuffer) const
    {
        switch (m_dataType)
        {
        case DataType::Float:
            return DoCopyDataTo<float>(NativeTensorView<float>().GetSOBViewPtr(), outputBuffer);
        case DataType::Double:
            return DoCopyDataTo<double>(NativeTensorView<double>().GetSOBViewPtr(), outputBuffer);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
        }
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

        typename Matrix<ElementType>::ConstMatrixPtr matrix = GetMatrix<ElementType>();
        auto matrixDims = GetMatrixDimensions(Shape());
        if (matrix->GetNumRows() != matrixDims.first)
            LogicError("The number of rows of the underlying matrix does not match the shape.");
        if (matrix->GetNumCols() != matrixDims.second)
            LogicError("The number of columns of the underlying matrix does not match the shape.");

        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        if ((matrix->GetMatrixType() != Microsoft::MSR::CNTK::MatrixType::SPARSE) || (matrix->GetFormat() != Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC))
            RuntimeError("NDArrayView::SparseDataBuffers: The underlying matrix of 'this' NDArrayView is not in the CSC sparse format.");

        size_t numNonZeroValues;
        const ElementType* nonZeroValues;
        const SparseIndexType* colStarts;
        const SparseIndexType* rowIndices;
        if (m_device.Type() == DeviceKind::CPU)
        {
            if (sizeof(CPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on CPU.");
            const auto* /*std::shared_ptr<Microsoft::MSR::CNTK::CPUSparseMatrix<ElementType>>*/ sparseMatrix = matrix->m_CPUSparseMatrix.get();
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<const ElementType *>(sparseMatrix->NzValues());
            colStarts     = static_cast<const SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices    = static_cast<const SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else if (m_device.Type() == DeviceKind::GPU)
        {
            if (sizeof(GPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on GPU.");
            const auto* /*std::shared_ptr<Microsoft::MSR::CNTK::GPUSparseMatrix<ElementType>>*/ sparseMatrix = matrix->m_GPUSparseMatrix.get();
            numNonZeroValues = sparseMatrix->NzCount();
            nonZeroValues = static_cast<const ElementType *>(sparseMatrix->NzValues());
            colStarts     = static_cast<const SparseIndexType *>(sparseMatrix->ColLocation());
            rowIndices    = static_cast<const SparseIndexType *>(sparseMatrix->RowLocation());
        }
        else
        {
            RuntimeError("NDArrayView::SparseDataBuffers: The device %S is currently not supported.",DeviceKindName(m_device.Type()));
        }

        return std::tuple<const ElementType *, const SparseIndexType *, const SparseIndexType *, size_t>(nonZeroValues, colStarts, rowIndices, numNonZeroValues);
    }

    template <typename ElementType>
    std::tuple<const void *, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers() const
    {
        if (AsDataType<ElementType>() != m_dataType)
            InvalidArgument("NDArrayView::SparseBlockColumnDataBuffers: The specified ElementType '%s' does not match this NDArrayView's DataType '%s'.", typeid(ElementType).name(), DataTypeName(m_dataType));

        if (!IsSparse())
            RuntimeError("The storage format of 'this' NDArrayView is dense. Please use another DataBuffer().");

        if (GetStorageFormat() != StorageFormat::SparseBlockCol)
            RuntimeError("The SparseBlockColumnDataBuffers() method only supports sparse block column format.");

        typename Matrix<ElementType>::ConstMatrixPtr matrix = GetMatrix<ElementType>();

        size_t numBlocks;
        size_t numRows;
        size_t numCols;
        const ElementType* blockValues;
        const SparseIndexType* blockId2Col;
        const SparseIndexType* col2BlockId;
        if (m_device.Type() == DeviceKind::GPU)
        {
            if (sizeof(GPUSPARSE_INDEX_TYPE) != sizeof(SparseIndexType))
                LogicError("Inconsistent data type for sparse index in 'this' Value and the underlying matrix on GPU.");
            const auto* /*std::shared_ptr<Microsoft::MSR::CNTK::GPUSparseMatrix<ElementType>>*/ sparseMatrix = matrix->m_GPUSparseMatrix.get();
            numBlocks = sparseMatrix->GetBlockSize();
            numRows = sparseMatrix->GetNumRows();
            numCols = sparseMatrix->GetNumCols();
            blockValues = static_cast<const ElementType *>(sparseMatrix->NzValues());
            blockId2Col = static_cast<const SparseIndexType *>(sparseMatrix->BlockId2ColOrRow());
            col2BlockId = static_cast<const SparseIndexType *>(sparseMatrix->ColOrRow2BlockId());
        }
        else
        {
            // CPU sparse block column is not yet supported, as the index format is different from GPU sparse block column
            RuntimeError("NDArrayView::SparseBlockColumnDataBuffers: The device %S is currently not supported.", DeviceKindName(m_device.Type()));
        }

        return std::tuple<const ElementType *, const SparseIndexType *, const SparseIndexType *, size_t, size_t, size_t>(blockValues, blockId2Col, col2BlockId, numBlocks, numRows, numCols);
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
            return (ElementType)NativeTensorView<float>().GetSOBViewPtr()->Get00Element();
        else if (GetDataType() == DataType::Double)
            return (ElementType)NativeTensorView<double>().GetSOBViewPtr()->Get00Element();
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
            asString = NativeTensorView<float>().AsString(maxItems, !Internal::IsReversingTensorShapesInErrorMessagesEnabled());
            break;
        case DataType::Double:
            asString = NativeTensorView<double>().AsString(maxItems, !Internal::IsReversingTensorShapesInErrorMessagesEnabled());
            break;
        default:
            LogicError("NDArrayView::LogToFile: Unsupported DataType %s", DataTypeName(m_dataType));
            break;
        }
        fprintf(f, "%S : %s%S = ", name.empty() ? L"_" : name.c_str(), DataTypeName(GetDataType()), Shape().AsString().c_str());
        fprintf(f, "%s\n", asString.c_str());
        fflush(f); // flush right away since most likely users are debugging
    }

    // Explicit template instantiations
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<float>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomUniform<double>(const NDShape& shape, double rangeBegin, double rangeEnd, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<float>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);
    template CNTK_API NDArrayViewPtr NDArrayView::RandomNormal<double>(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device/* = DeviceDescriptor::UseDefaultDevice()*/);

    template CNTK_API const float* NDArrayView::DataBuffer<float>() const;
    template CNTK_API const double* NDArrayView::DataBuffer<double>() const;

    template CNTK_API void NDArrayView::CopyDataTo<float>(std::vector<float>& outputBuffer) const;
    template CNTK_API void NDArrayView::CopyDataTo<double>(std::vector<double>& outputBuffer) const;
    template CNTK_API void NDArrayView::CopyDataTo<size_t>(std::vector<size_t>& outputBuffer) const;
    template CNTK_API void NDArrayView::CopyDataTo<int>(std::vector<int>& outputBuffer) const;
    template CNTK_API void NDArrayView::CopyDataTo<unsigned int>(std::vector<unsigned int>& outputBuffer) const;

    // TODO: Was this changed inconsistently between master and fseide/dynamite? Check!
    //template CNTK_API const TensorView<float>* NDArrayView::GetTensorView<float>() const;
    //template CNTK_API const TensorView<double>* NDArrayView::GetTensorView<double>() const;

    template CNTK_API std::tuple<const float*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<float>() const;
    template CNTK_API std::tuple<const double*, const SparseIndexType*, const SparseIndexType*, size_t> NDArrayView::SparseCSCDataBuffers<double>() const;

    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<float>() const;
    template CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> NDArrayView::SparseBlockColumnDataBuffers<double>() const;

    template CNTK_API float* NDArrayView::WritableDataBuffer<float>();
    template CNTK_API double* NDArrayView::WritableDataBuffer<double>();

    template Matrix<float>::ConstMatrixPtr NDArrayView::GetMatrix<float>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;
    template Matrix<double>::ConstMatrixPtr NDArrayView::GetMatrix<double>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/) const;

    template Matrix<float>::MatrixPtr NDArrayView::GetWritableMatrix<float>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    template Matrix<double>::MatrixPtr NDArrayView::GetWritableMatrix<double>(size_t rowColSplitPoint/* = AutoSelectRowColSplitPoint*/);
    //template std::shared_ptr<TensorView<float>> NDArrayView::GetWritableTensorViewMin2D<float>();
    //template std::shared_ptr<TensorView<double>> NDArrayView::GetWritableTensorViewMin2D<double>();

    template const Microsoft::MSR::CNTK::TensorView<float>& NDArrayView::NativeTensorView() const;
    template const Microsoft::MSR::CNTK::TensorView<double>& NDArrayView::NativeTensorView() const;

    template Microsoft::MSR::CNTK::TensorView<float>& NDArrayView::WritableNativeTensorView();
    template Microsoft::MSR::CNTK::TensorView<double>& NDArrayView::WritableNativeTensorView();

    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template CNTK_API NDArrayView::NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly/* = false*/);

    template CNTK_API float NDArrayView::AsScalar<float>() const;
    template CNTK_API double NDArrayView::AsScalar<double>() const;
}
