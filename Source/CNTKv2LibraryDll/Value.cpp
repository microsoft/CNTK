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
        if ((mask != nullptr) && (mask->Shape().NumAxes() > data->Shape().NumAxes()))
            InvalidArgument("The number of axes of the mask of a Value object cannot exceed the number of axes of the data NDArrayView object");

        if (mask != nullptr)
        {
            auto dataShape = data->Shape();
            auto maskShape = mask->Shape();
            if (dataShape.SubShape(dataShape.NumAxes() - maskShape.NumAxes()) != maskShape)
                InvalidArgument("Invalid Value object; the data and mask are incompatible. The trailing dimensions of the data do not match the dimensions of the mask");
        }
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<const std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        size_t numSequences = sequences.size();
        std::vector<size_t> sequenceLengths(numSequences);
        size_t sampleSize = sampleShape.TotalSize();
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

        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        NDArrayViewPtr valueData(new NDArrayView(GetDataType<ElementType>(), valueDataShape, DeviceDescriptor::CPUDevice()), [](_ReferenceCounter* ptr) { delete ptr; });
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
            deviceValueData = NDArrayViewPtr(new NDArrayView(GetDataType<ElementType>(), valueDataShape, device), [](_ReferenceCounter* ptr) { delete ptr; });
            deviceValueData->CopyFrom(*valueData);
            if (readOnly)
                deviceValueData = deviceValueData->Alias(true);
        }

        NDMaskPtr deviceValueMask;
        if (needsMask)
        { 
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = NDMaskPtr(new NDMask(valueMaskShape, device), [](_ReferenceCounter* ptr) {delete ptr; });
            for (size_t i = 0; i < numSequences; ++i)
                deviceValueMask->MaskSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
        }

        return ValuePtr(new Value(deviceValueData, deviceValueMask), [](_ReferenceCounter* ptr) { delete ptr; });
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
        return ValuePtr(new Value(Data()->DeepClone(readOnly), (Mask() != nullptr) ? Mask()->DeepClone() : nullptr), [](_ReferenceCounter* ptr) { delete ptr; });
    }

    /*virtual*/ ValuePtr Value::Alias(bool readOnly/* = false*/) const
    {
        // TODO: Check if this is a derived type and throw an exception in that case
        return ValuePtr(new Value(Data()->Alias(readOnly), (Mask() != nullptr) ? Mask()->Alias() : nullptr), [](_ReferenceCounter* ptr) { delete ptr; });
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
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<const std::vector<float>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<const std::vector<double>>& sequences, const DeviceDescriptor& device, bool readOnly/* = false*/);
}
