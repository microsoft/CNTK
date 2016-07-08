//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

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

            if (maskShape.NumAxes() > dataShape.NumAxes())
                InvalidArgument("The number of axes of the mask of a Value object cannot exceed the number of axes of the data NDArrayView object");

            if (dataShape.SubShape(dataShape.NumAxes() - maskShape.NumAxes()) != maskShape)
                InvalidArgument("Invalid Value object; the data and mask are incompatible. The trailing dimensions of the data do not match the dimensions of the mask");
        }
    }

    template <typename T>
    static NDMaskPtr CreateMask(size_t sampleSize, const std::vector<std::vector<T>>& sequences, const DeviceDescriptor& device)
    {
        size_t numSequences = sequences.size();
        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxSequenceLength = 0;
        bool needsMask = false;
        for (size_t i = 0; i < numSequences; ++i)
        {
            sequenceLengths[i] = sequences[i].size() / sampleSize;

            if (maxSequenceLength < sequenceLengths[i])
                maxSequenceLength = sequenceLengths[i];

            if ((i > 0) && (sequenceLengths[i - 1] != sequenceLengths[i]))
                needsMask = true;
        }

        NDMaskPtr deviceValueMask;
        if (needsMask)
        {
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = NDMaskPtr(new NDMask(valueMaskShape, device), [](Internal::ReferenceCount* ptr) {delete ptr; });
            for (size_t i = 0; i < numSequences; ++i)
                deviceValueMask->MaskSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
        }

        return deviceValueMask;
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        NDMaskPtr deviceValueMask = CreateMask(1, oneHotSequences, device);
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
                rowIndices.push_back((SparseIndexType)(oneHotSequences[i][j]));
            }

            for (; j < maxSequenceLength; ++j)
                colStarts[(i * maxSequenceLength) + j] = (SparseIndexType)(nonZeroValues.size());
        }

        colStarts[numSequences * maxSequenceLength] = (SparseIndexType)(nonZeroValues.size());
        NDArrayViewPtr deviceValueData(new NDArrayView(valueDataShape, colStarts.data(), rowIndices.data(), nonZeroValues.data(), nonZeroValues.size(), device, readOnly), [](ReferenceCount* ptr) { delete ptr; });
        return ValuePtr(new Value(deviceValueData, deviceValueMask), [](ReferenceCount* ptr) { delete ptr; });
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        size_t sampleSize = sampleShape.TotalSize();
        NDMaskPtr deviceValueMask = CreateMask(sampleSize, sequences, device);
        size_t maxSequenceLength = (deviceValueMask == nullptr) ? sequences[0].size() : deviceValueMask->Shape()[0];

        size_t numSequences = sequences.size();
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        NDArrayViewPtr valueData(new NDArrayView(AsDataType<ElementType>(), valueDataShape, DeviceDescriptor::CPUDevice()), [](ReferenceCount* ptr) { delete ptr; });
        ElementType* dataBuffer = valueData->WritableDataBuffer<ElementType>();
        for (size_t i = 0; i < numSequences; ++i)
            std::copy(sequences[i].data(), sequences[i].data() + sequences[i].size(), dataBuffer + (maxSequenceLength * i * sampleSize));

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
            deviceValueData = NDArrayViewPtr(new NDArrayView(AsDataType<ElementType>(), valueDataShape, device), [](ReferenceCount* ptr) { delete ptr; });
            deviceValueData->CopyFrom(*valueData);
            if (readOnly)
                deviceValueData = deviceValueData->Alias(true);
        }

        return ValuePtr(new Value(deviceValueData, deviceValueMask), [](ReferenceCount* ptr) { delete ptr; });
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
        return ValuePtr(new Value(Data()->DeepClone(readOnly), (Mask() != nullptr) ? Mask()->DeepClone() : nullptr), [](ReferenceCount* ptr) { delete ptr; });
    }

    /*virtual*/ ValuePtr Value::Alias(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return ValuePtr(new Value(Data()->Alias(readOnly), (Mask() != nullptr) ? Mask()->Alias() : nullptr), [](ReferenceCount* ptr) { delete ptr; });
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

    // Explicit template instantiations
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<std::vector<double>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(size_t vocabSize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
}
