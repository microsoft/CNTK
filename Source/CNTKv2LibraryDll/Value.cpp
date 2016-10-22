//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#ifdef _WIN32
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "CNTKLibrary.h"
#include "Utils.h"
#include "Value.h"
#include "Function.h"

namespace CNTK
{
    Value::Value(const NDArrayViewPtr& data)
        : Value(data, nullptr)
    {
    }

    Value::Value(const NDArrayViewPtr& data, const NDMaskPtr& mask)
        : m_data(data), m_mask(mask)
    {
        if (mask != nullptr)
        {
            auto dataShape = data->Shape();
            auto maskShape = mask->Shape();

            if (maskShape.Rank() > dataShape.Rank())
                InvalidArgument("The rank (%d) of the mask of a Value object cannot exceed the rank (%d) of the data NDArrayView object", (int)maskShape.Rank(), (int)dataShape.Rank());

            if (dataShape.SubShape(dataShape.Rank() - maskShape.Rank()) != maskShape)
                InvalidArgument("Invalid Value object; the data and mask are incompatible. The trailing dimensions of the data with shape %S do not match the dimensions of the mask with shape %S", AsStringForErrorReporting(dataShape).c_str(), AsStringForErrorReporting(maskShape).c_str());
        }
    }

    template <typename T>
    static NDMaskPtr CreateMask(size_t numElementsPerSample, const std::vector<std::vector<T>>& sequences, const DeviceDescriptor& device)
    {
        size_t numSequences = sequences.size();
        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxSequenceLength = 0;
        bool needsMask = false;
        for (size_t i = 0; i < numSequences; ++i)
        {
            sequenceLengths[i] = sequences[i].size() / numElementsPerSample;

            if (maxSequenceLength < sequenceLengths[i])
                maxSequenceLength = sequenceLengths[i];

            if ((i > 0) && (sequenceLengths[i - 1] != sequenceLengths[i]))
                needsMask = true;
        }

        // If needed, create a mask to account for variability in lengths of specified sequences
        NDMaskPtr deviceValueMask;
        if (needsMask)
        {
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = MakeSharedObject<NDMask>(valueMaskShape, device);
            for (size_t i = 0; i < numSequences; ++i)
            {
                deviceValueMask->MarkSequenceBegin({0, i});
                deviceValueMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
            }
        }

        return deviceValueMask;
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        NDMaskPtr deviceValueMask = CreateMask(1, oneHotSequences, DeviceDescriptor::CPUDevice());
        size_t maxSequenceLength = (deviceValueMask == nullptr) ? oneHotSequences[0].size() : deviceValueMask->Shape()[0];

        size_t numSequences = oneHotSequences.size();
        NDShape sampleShape = { vocabularySize };
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        size_t numCSCCols = valueDataShape.SubShape(1).TotalSize() + 1;
        std::vector<SparseIndexType> colStarts(numCSCCols);
        std::vector<ElementType> nonZeroValues;
        std::vector<SparseIndexType> rowIndices;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceLength = oneHotSequences[i].size();
            size_t j = 0;
            for (; j < currentSequenceLength; ++j)
            {
                colStarts[(i * maxSequenceLength) + j] = (SparseIndexType)nonZeroValues.size();
                nonZeroValues.push_back(1);
                if (oneHotSequences[i][j] >= vocabularySize)
                    InvalidArgument("Value::Create: one-hot data exceeds vocabulary size");
                rowIndices.push_back((SparseIndexType)(oneHotSequences[i][j]));
            }

            for (; j < maxSequenceLength; ++j)
                colStarts[(i * maxSequenceLength) + j] = (SparseIndexType)(nonZeroValues.size());
        }

        colStarts[numSequences * maxSequenceLength] = (SparseIndexType)(nonZeroValues.size());
        NDArrayViewPtr deviceValueData = MakeSharedObject<NDArrayView>(valueDataShape, colStarts.data(), rowIndices.data(), nonZeroValues.data(), nonZeroValues.size(), device, readOnly);
        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        size_t numElementsPerSample = sampleShape.TotalSize();
        NDMaskPtr deviceValueMask = CreateMask(numElementsPerSample, sequences, DeviceDescriptor::CPUDevice());
        size_t maxSequenceLength = (deviceValueMask == nullptr) ? sequences[0].size() : deviceValueMask->Shape()[0];

        size_t numSequences = sequences.size();
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        NDArrayViewPtr valueData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), valueDataShape, DeviceDescriptor::CPUDevice());
        ElementType* dataBuffer = valueData->WritableDataBuffer<ElementType>();
        for (size_t i = 0; i < numSequences; ++i)
            std::copy(sequences[i].data(), sequences[i].data() + sequences[i].size(), dataBuffer + (maxSequenceLength * i * numElementsPerSample));

        NDArrayViewPtr deviceValueData;
        if (device == DeviceDescriptor::CPUDevice())
        {
            if (readOnly)
                deviceValueData = valueData->Alias(true);
            else
                deviceValueData = valueData;
        }
        else
        {
            deviceValueData = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), valueDataShape, device);
            deviceValueData->CopyFrom(*valueData);
            if (readOnly)
                deviceValueData = deviceValueData->Alias(true);
        }

        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    /*virtual*/ Value::~Value()
    {
    }

    /*virtual*/ NDArrayViewPtr Value::Data() const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return m_data;
    }

    /*virtual*/ NDMaskPtr Value::Mask() const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return m_mask;
    }

    /*virtual*/ ValuePtr Value::DeepClone(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return MakeSharedObject<Value>(Data()->DeepClone(readOnly), (Mask() != nullptr) ? Mask()->DeepClone() : nullptr);
    }

    /*virtual*/ ValuePtr Value::Alias(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return MakeSharedObject<Value>(Data()->Alias(readOnly), (Mask() != nullptr) ? Mask()->Alias() : nullptr);
    }

    /*virtual*/ void Value::CopyFrom(const Value& source)
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        Data()->CopyFrom(*source.Data());
        if ((Mask() == nullptr) && (source.Mask() != nullptr))
            InvalidArgument("Value::CopyFrom: Invalid source object; Cannot copy a Value with a mask into 'this' Value that does not have a mask.");

        if (source.Mask() != nullptr)
            Mask()->CopyFrom(*source.Mask());
        else
        {
            if (Mask() != nullptr)
            {
                // Clear the mask
                Mask()->Clear();
            }
        }
    }

    void PackedValue::Unpack() const
    {
        if (m_packedDataLayout && (m_packedDataLayout->GetNumTimeSteps() != 1) && (m_packedDataLayout->GetNumSequences() != 1) && Internal::IsAutomaticUnpackingOfPackedValuesDisabled())
            LogicError("PackedValue::Unpack: Automatic unpacking of PackedValue objects is disabled");

        if (m_isPacked)
        {
            ValuePtr valueObject;
            auto dataType = m_packedData->GetDataType();
            switch (dataType)
            {
            case DataType::Float:
                valueObject = CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, *(m_packedData->GetMatrix<float>()), m_packedDataLayout, m_isReadOnly);
                break;
            case DataType::Double:
                valueObject = CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, *(m_packedData->GetMatrix<double>()), m_packedDataLayout, m_isReadOnly);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(dataType));
            }

            m_data = valueObject->Data();
            m_mask = valueObject->Mask();

            m_packedData = nullptr;
            m_packedDataLayout = nullptr;
            m_isPacked = false;

            if (m_unpackedShape != m_data->Shape())
                LogicError("The computed unpacked shape of the PackedValue object does not match the actual Data NDArrayView's shape after unpacking");
        }
    }

    // Explicit template instantiations
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<std::vector<double>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
}
