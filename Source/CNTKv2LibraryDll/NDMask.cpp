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
        auto maskMatrix = new Matrix<char>(matrixDims.first, matrixDims.second, AsCNTKImplDeviceId(device));
        maskMatrix->SetValue(1);

        return maskMatrix;
    }

    NDMask::NDMask(const NDShape& shape, Matrix<char>* matrix)
        : m_device(AsDeviceDescriptor(matrix->GetDeviceId())), m_maskShape(shape), m_matrixView(matrix)
    {
    }

    NDMask::NDMask(const NDShape& shape, const DeviceDescriptor& device/* = DeviceDescriptor::DefaultDevice()*/)
        : NDMask(shape, AllocateMatrix(shape, device))
    {
        if (shape.NumAxes() > 2)
            LogicError("NDMask instances with more than 2 axes are currently unsupported");
    }

    NDMask::~NDMask()
    {
        delete m_matrixView;
    }

    void NDMask::MaskSection(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape)
    {
        // TODO: Implement batching of masking operation for masks residing on GPUs to avoid making 
        // GPU invocations for each MaskSection call.

        if (sectionOffset.size() > m_maskShape.NumAxes())
            LogicError("NDMask::MaskSection: The sectionOffset cannot have dimensionality higher than the number of axes of 'this' mask");

        if (sectionShape.NumAxes() > m_maskShape.NumAxes())
            LogicError("NDMask::MaskSection: The section shape cannot have an axes count higher than the number of axes of 'this' mask");

        std::vector<size_t> offset(m_maskShape.NumAxes(), 0);
        for (size_t i = 0; i < sectionOffset.size(); ++i)
            offset[i] = sectionOffset[i];

        NDShape shape = sectionShape.AppendShape(NDShape(m_maskShape.NumAxes() - sectionShape.NumAxes(), NDShape::InferredDimension));

        auto maskMatrix = GetMatrix();
        size_t rowOffset = offset[0];
        size_t colOffset = offset[1];
        size_t sliceRowLength = (shape[0] != NDShape::InferredDimension) ? shape[0] : (maskMatrix->GetNumRows() - rowOffset);
        size_t sliceColLength = (shape[1] != NDShape::InferredDimension) ? shape[1] : (maskMatrix->GetNumCols() - colOffset);
        if ((rowOffset == 0) && (sliceRowLength == maskMatrix->GetNumRows()))
            maskMatrix->ColumnSlice(colOffset, sliceColLength).SetValue(0);
        else
        {
            // Since Matrix does not support strides in the row dimension, we will need to create separate slices for each column
            for (size_t i = colOffset; i < (colOffset + sliceColLength); ++i)
            {
                auto column = maskMatrix->ColumnSlice(i, 1);
                column.Reshape(1, maskMatrix->GetNumRows());
                column.ColumnSlice(rowOffset, sliceRowLength).SetValue(0);
            }
        }
    }

    void NDMask::Clear()
    {
        GetMatrix()->SetValue(1);
    }

    Matrix<char>* NDMask::GetMatrix() const
    {
        return m_matrixView;
    }

    void NDMask::CopyFrom(const NDMask& source)
    {
        if (source.Shape() != Shape())
            InvalidArgument("NDMask::CopyFrom: The 'source' mask's shape must be same as the shape of this NDMask");

        GetMatrix()->AssignValuesOf(*source.GetMatrix());
    }

    NDMaskPtr NDMask::DeepClone() const
    {
        NDMaskPtr newMask = new NDMask(this->Shape(), this->Device());
        newMask->CopyFrom(*this);

        return NDMaskPtr(newMask, [](_ReferenceCounter* ptr) { delete ptr; });
    }

    NDMaskPtr NDMask::Alias() const
    {
        return NDMaskPtr(new NDMask(this->Shape(), new Matrix<char>(GetMatrix()->AsReference())), [](_ReferenceCounter* ptr) { delete ptr; });
    }
}
