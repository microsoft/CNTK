//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Sequences.h"
#include "Utils.h"

namespace CNTK
{
    class PackedValue final : public Value
    {
    public:
        template <typename ElementType>
        PackedValue(const NDShape& sampleShape, const std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>& packedDataMatrix, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout, bool isReadOnly)
            : Value(nullptr), m_isPacked(true), m_sampleShape(sampleShape), m_packedData(nullptr), m_packedDataLayout(packedDataLayout), m_isReadOnly(isReadOnly)
        {
            NDShape packedMatrixShape({ packedDataMatrix->GetNumRows(), packedDataMatrix->GetNumCols() });
            auto tensorView = new TensorView<ElementType>(packedDataMatrix, AsTensorViewShape(packedMatrixShape));
            m_packedData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(packedDataMatrix->GetDeviceId()), AsStorageFormat(packedDataMatrix->GetFormat()), packedMatrixShape, m_isReadOnly, tensorView);

            // Determine unpacked shape
            m_unpackedShape = sampleShape;
            if (packedDataLayout)
                m_unpackedShape = m_unpackedShape.AppendShape({ packedDataLayout->GetNumTimeSteps(), packedDataLayout->GetNumSequences() });
        }

        void Unpack() const;

        const NDShape& Shape() const override { return m_unpackedShape; }
        DeviceDescriptor Device() const override { return m_isPacked ? m_packedData->Device() : Value::Device(); }
        DataType GetDataType() const override { return m_isPacked ? m_packedData->GetDataType() : Value::GetDataType(); }
        StorageFormat GetStorageFormat() const override { return m_isPacked? m_packedData->GetStorageFormat() : Value::GetStorageFormat(); }
        bool IsReadOnly() const override { return m_isPacked ? m_packedData->IsReadOnly() : Value::IsReadOnly(); }

        NDArrayViewPtr Data() const override
        {
            Unpack();
            return Value::Data();
        }

        NDMaskPtr Mask() const override
        {
            Unpack();
            return Value::Mask();
        }

        ValuePtr DeepClone(bool /*readOnly = false*/) const override
        {
            LogicError("DeepClone is currently unsupported for PackedValue objects");
        }

        ValuePtr Alias(bool /*readOnly = false*/) const override
        {
            LogicError("Alias is currently unsupported for PackedValue objects");
        }

        void CopyFrom(const Value& /*source*/) override
        {
            LogicError("CopyFrom is currently unsupported for PackedValue objects");
        }

        template <typename ElementType>
        std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>> PackedData()
        {
            if (!m_isPacked)
                InvalidArgument("PackedValue::PackedData called on a Value object that has already been unpacked");

            return { m_packedData->GetMatrix<ElementType>(), m_packedDataLayout };
        }

    private:
        bool m_isReadOnly;
        NDShape m_sampleShape;
        NDShape m_unpackedShape;

        mutable bool m_isPacked;
        mutable NDArrayViewPtr m_packedData;
        mutable std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> m_packedDataLayout;
    };
}
