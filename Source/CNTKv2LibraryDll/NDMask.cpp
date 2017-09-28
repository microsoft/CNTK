//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Matrix.h"
#include <algorithm>
#include "TensorShape.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    static Matrix<char>* AllocateMatrix(const NDShape& viewShape, const DeviceDescriptor& device)
    {
        auto matrixDims = GetMatrixDimensions(viewShape);
        return new Matrix<char>(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device));
    }

    NDMask::NDMask(const NDShape& shape, Matrix<char>* matrix)
        : m_device(AsDeviceDescriptor(matrix->GetDeviceId())), m_maskShape(shape)
    {
        m_matrixView = std::shared_ptr<Matrix<char>>(matrix, [](Matrix<char>* ptr) { delete ptr; });
    }

    NDMask::NDMask(const NDShape& shape, const DeviceDescriptor& device)
        : NDMask(shape, AllocateMatrix(shape, device))
    {
        if (shape.Rank() > 2)
            LogicError("NDMask instance shaped '%S' with more than 2 axes is currently unsupported.", shape.AsString().c_str());

        Clear();
    }

    NDMask::~NDMask()
    {}

    void NDMask::MarkSectionAs(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape, MaskKind maskKind)
    {
        // TODO: Implement batching of masking operation for masks residing on GPUs to avoid making
        // GPU invocations for each MaskSection call.

        if (sectionOffset.size() > m_maskShape.Rank())
            LogicError("NDMask::MaskSection: The sectionOffset dimensionality (%d) must be <= rank (%d) of 'this' mask.", (int)sectionOffset.size(), (int)m_maskShape.Rank());

        if (sectionShape.Rank() > m_maskShape.Rank())
            LogicError("NDMask::MaskSection: The section shape '%S' rank (%d) must be <= rank (%d) of 'this' mask.", sectionShape.AsString().c_str(), (int)sectionShape.Rank(), (int)m_maskShape.Rank());

        std::vector<size_t> offset(m_maskShape.Rank(), 0);
        for (size_t i = 0; i < sectionOffset.size(); ++i)
            offset[i] = sectionOffset[i];

        NDShape shape = sectionShape.AppendShape(NDShape(m_maskShape.Rank() - sectionShape.Rank(), NDShape::InferredDimension));

        auto maskMatrix = GetMatrix();
        size_t rowOffset = offset[0];
        size_t colOffset = offset[1];
        size_t sliceRowLength = (shape[0] != NDShape::InferredDimension) ? shape[0] : (maskMatrix->GetNumRows() - rowOffset);
        size_t sliceColLength = (shape[1] != NDShape::InferredDimension) ? shape[1] : (maskMatrix->GetNumCols() - colOffset);
        if ((rowOffset == 0) && (sliceRowLength == maskMatrix->GetNumRows()))
            maskMatrix->ColumnSlice(colOffset, sliceColLength).SetValue((char)maskKind);
        else
        {
            // Since Matrix does not support strides in the row dimension, we will need to create separate slices for each column
            for (size_t i = colOffset; i < (colOffset + sliceColLength); ++i)
            {
                auto column = maskMatrix->ColumnSlice(i, 1);
                column.Reshape(1, maskMatrix->GetNumRows());
                column.ColumnSlice(rowOffset, sliceRowLength).SetValue((char)maskKind);
            }
        }
    }

    void NDMask::Clear()
    {
        // Clear the mask by marking all samples as Valid
        GetMatrix()->SetValue((char)MaskKind::Valid);
    }

    size_t NDMask::MaskedCount() const
    {
        auto maskMatrix = GetMatrix();
        std::unique_ptr<char[]> maskData(maskMatrix->CopyToArray());
        return std::count_if(maskData.get(), maskData.get() + maskMatrix->GetNumElements(), [](const char& val) {
            return val == (char)MaskKind::Invalid;
        });
    }

    // TODO: This could actually be strided?
    const MaskKind* NDMask::DataBuffer() const
    {
        // First make sure that the underlying matrix is on the right device
        auto matrix = GetMatrix();
        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(m_device), true);
        return (const MaskKind*)(matrix->Data());
    }

    Matrix<char>* NDMask::GetMatrix() const
    {
        return m_matrixView.get();
    }

    void NDMask::CopyFrom(const NDMask& source)
    {
        if (source.Shape() != Shape())
            InvalidArgument("NDMask::CopyFrom: The source mask's shape '%S' must be same as this NDMask's shape '%S'.", source.Shape().AsString().c_str(), Shape().AsString().c_str());

        GetMatrix()->AssignValuesOf(*source.GetMatrix());
    }

    NDMaskPtr NDMask::DeepClone(const DeviceDescriptor& device) const
    {
        NDMaskPtr newMask = MakeSharedObject<NDMask>(this->Shape(), device);
        newMask->CopyFrom(*this);

        return newMask;
    }

    NDMaskPtr NDMask::Alias() const
    {
        return MakeSharedObject<NDMask>(this->Shape(), new Matrix<char>(GetMatrix()->AsReference()));
    }
}
