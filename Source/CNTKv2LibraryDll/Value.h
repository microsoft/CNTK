//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Sequences.h"
#include "TensorView.h"
#include "Utils.h"

namespace CNTK
{
    class PackedValue final : public Value
    {
        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

    public:
#if 0   // I run into problems with template resolution, it does not match the packedDataMatrix argument, in VS 2015.
        template <typename ElementType>
        PackedValue(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const typename Microsoft::MSR::CNTK::Matrix<ElementType>::MatrixPtr& packedDataMatrix, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout, bool isReadOnly)
            : Value(nullptr), m_isPacked(true), m_sampleShape(sampleShape), m_sampleDynamicAxes(sampleDynamicAxes), m_packedData(nullptr), m_packedDataLayout(packedDataLayout), m_isReadOnly(isReadOnly)
        {
            NDShape packedMatrixShape({ packedDataMatrix->GetNumRows(), packedDataMatrix->GetNumCols() });
#if 1
            m_packedData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), packedMatrixShape, m_isReadOnly, packedDataMatrix);
#else
            auto tensorView = new Microsoft::MSR::CNTK::TensorView<ElementType>(packedDataMatrix, AsTensorShapeMin2D(packedMatrixShape));
            m_packedData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(packedDataMatrix->GetDeviceId()), AsStorageFormat(packedDataMatrix->GetFormat()), packedMatrixShape, m_isReadOnly, tensorView);
#endif

            // Determine unpacked shape
            m_unpackedShape = GetUnpackedShape(sampleShape, sampleDynamicAxes, packedDataLayout);
        }
#else   // temporary solution: manually unrolling it
        PackedValue(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const typename Microsoft::MSR::CNTK::Matrix<float>::MatrixPtr& packedDataMatrix, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout, bool isReadOnly)
            : Value(nullptr), m_isPacked(true), m_sampleShape(sampleShape), m_sampleDynamicAxes(sampleDynamicAxes), m_packedData(nullptr), m_packedDataLayout(packedDataLayout), m_isReadOnly(isReadOnly)
        {
            NDShape packedMatrixShape({ packedDataMatrix->GetNumRows(), packedDataMatrix->GetNumCols() });
            m_packedData = MakeSharedObject<NDArrayView>(AsDataType<float>(), packedMatrixShape, m_isReadOnly, packedDataMatrix);
            m_unpackedShape = GetUnpackedShape(sampleShape, sampleDynamicAxes, packedDataLayout);
        }
        PackedValue(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const typename Microsoft::MSR::CNTK::Matrix<double>::MatrixPtr& packedDataMatrix, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout, bool isReadOnly)
            : Value(nullptr), m_isPacked(true), m_sampleShape(sampleShape), m_sampleDynamicAxes(sampleDynamicAxes), m_packedData(nullptr), m_packedDataLayout(packedDataLayout), m_isReadOnly(isReadOnly)
        {
            NDShape packedMatrixShape({ packedDataMatrix->GetNumRows(), packedDataMatrix->GetNumCols() });
            m_packedData = MakeSharedObject<NDArrayView>(AsDataType<double>(), packedMatrixShape, m_isReadOnly, packedDataMatrix);
            m_unpackedShape = GetUnpackedShape(sampleShape, sampleDynamicAxes, packedDataLayout);
        }
#endif

        bool IsPacked() const { return m_isPacked; }

        void Unpack() const;

        void Erase() override
        {
            if (IsPacked())
            {
                m_packedData = nullptr;
                m_packedDataLayout = nullptr;
                m_isPacked = false;
            }
            else
                Value::Erase();
        }

        /* virtual */ bool IsValid() const
        {
            if (IsPacked())
                return m_packedData != nullptr;
            else
                return Value::IsValid();
        }

        ///
        /// Returns the dynamic axes associated with this packed value
        ///
        const std::vector<Axis>& DynamicAxes() const { return m_sampleDynamicAxes; }

        const NDShape& Shape() const override { return m_unpackedShape; }
        DeviceDescriptor Device() const override { return m_isPacked ? m_packedData->Device() : Value::Device(); }
        DataType GetDataType() const override { return m_isPacked ? m_packedData->GetDataType() : Value::GetDataType(); }
        StorageFormat GetStorageFormat() const override { return m_isPacked? m_packedData->GetStorageFormat() : Value::GetStorageFormat(); }
        bool IsReadOnly() const override { return m_isPacked ? m_packedData->IsReadOnly() : Value::IsReadOnly(); }

        size_t MaskedCount() const override
        {
            if (m_isPacked)
                // Compute the number of masked samples after the data will be unpacked
                return m_packedDataLayout ? ((m_packedDataLayout->GetNumTimeSteps() * m_packedDataLayout->GetNumSequences()) - m_packedDataLayout->GetActualNumSamples()) : 0;
            else
                return Value::MaskedCount();
        }

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

        ValuePtr DeepClone(bool readOnly) const override
        {
            if (m_isPacked)
            {
                std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> packedLayoutCopy;
                if (m_packedDataLayout)
                {
                    packedLayoutCopy = std::make_shared<Microsoft::MSR::CNTK::MBLayout>();
                    packedLayoutCopy->CopyFrom(m_packedDataLayout);
                }
                return MakeSharedObject<PackedValue>(m_sampleShape, m_sampleDynamicAxes, m_packedData->DeepClone(readOnly), packedLayoutCopy, readOnly);
            }
            else
                return Value::DeepClone(readOnly);
        }

        ValuePtr Alias(bool /*readOnly = false*/) const override
        {
            LogicError("Value::Alias is currently unsupported for PackedValue objects.");
        }

        void CopyFrom(const Value& /*source*/) override
        {
            LogicError("Value::CopyFrom is currently unsupported for PackedValue objects");
        }

        template <typename ElementType>
        std::pair<typename Microsoft::MSR::CNTK::Matrix<ElementType>::ConstMatrixPtr, std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>> PackedData()
        {
            if (!m_isPacked)
                InvalidArgument("PackedValue::PackedData called on a Value object that has already been unpacked.");

            return { m_packedData->GetMatrix<ElementType>(), m_packedDataLayout };
        }

        static NDShape GetUnpackedShape(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout)
        {
            // Determine unpacked shape
            auto unpackedShape = sampleShape;
            if (packedDataLayout)
            {
                if (sampleDynamicAxes.empty())
                    LogicError("A PackedValue object that has a layout must have at least one dynamic axis.");

                // Sequence dynamic axes are "Ordered" as opposed to the batch axis which is unordered.
                bool hasSequenceAxis = (std::find_if(sampleDynamicAxes.begin(), sampleDynamicAxes.end(), [](const Axis& axis) {return axis.IsOrdered(); }) != sampleDynamicAxes.end());
                if (hasSequenceAxis)
                    unpackedShape = unpackedShape.AppendShape({ packedDataLayout->GetNumTimeSteps() });
                else if ((packedDataLayout->GetNumTimeSteps() != 1) || packedDataLayout->HasSequenceBeyondBegin())
                    LogicError("A PackedValue object with no sequence dynamic axis, must have a layout with exactly one time step and no sequences beginning in the past.");

                unpackedShape = unpackedShape.AppendShape({ packedDataLayout->GetNumSequences() });
            }
            else if (!sampleDynamicAxes.empty())
                LogicError("A PackedValue object that does not have a layout cannot have any dynamic axes.");

            return unpackedShape;
        }

    private:
    public: // for MakeSharedObject() only. TODO: Remove once we know how to do that right.
        PackedValue(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const NDArrayViewPtr& packedData, const std::shared_ptr<Microsoft::MSR::CNTK::MBLayout>& packedDataLayout, bool isReadOnly)
            : Value(nullptr), m_isPacked(true), m_sampleShape(sampleShape), m_sampleDynamicAxes(sampleDynamicAxes), m_packedData(packedData), m_packedDataLayout(packedDataLayout), m_isReadOnly(isReadOnly)
        {
            // Determine unpacked shape
            m_unpackedShape = GetUnpackedShape(sampleShape, sampleDynamicAxes, packedDataLayout);
        }

    private:
        bool m_isReadOnly;
        NDShape m_sampleShape;
        std::vector<Axis> m_sampleDynamicAxes;
        NDShape m_unpackedShape;

        mutable bool m_isPacked;
        mutable NDArrayViewPtr m_packedData;
        mutable std::shared_ptr<Microsoft::MSR::CNTK::MBLayout> m_packedDataLayout;
    };
}
