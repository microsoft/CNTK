//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#ifdef _WIN32
#define _SCL_SECURE_NO_WARNINGS
#endif

#include "CNTKLibrary.h"
#include "CompositeFunction.h"
#include "Utils.h"
#include "Value.h"
#include "Matrix.h"
#include "CPUSparseMatrix.h"
#include "RecurrentNodes.h"

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
                InvalidArgument("The rank (%zu) of the mask of a Value object cannot exceed the rank (%zu) of the data NDArrayView object", maskShape.Rank(), dataShape.Rank());

            if (dataShape.SubShape(dataShape.Rank() - maskShape.Rank()) != maskShape)
                InvalidArgument("Invalid Value object: data and mask are incompatible. The %s dimensions of the data with shape '%S' "
                                "do not match the dimensions of the mask with shape '%S'", 
                                Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "leading" : "trailing",
                                dataShape.AsString().c_str(), maskShape.AsString().c_str());
        }
    }

    static NDMaskPtr CreateMask(const std::vector<size_t>& sequenceLengths, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device)
    {
        size_t numSequences = sequenceLengths.size();

        if (!sequenceStartFlags.empty() && (sequenceStartFlags.size() != numSequences))
            InvalidArgument("Value::Create:: The number (%zu) of sequence start flags does not match the number (%zu) of sequences,",
                            sequenceStartFlags.size(), numSequences);

        std::vector<bool> actualStarts = sequenceStartFlags;
        if (actualStarts.empty())
            actualStarts.resize(numSequences, true);

        size_t maxSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
            maxSequenceLength = std::max(maxSequenceLength, sequenceLengths[i]);

        bool needsMask = (std::find(actualStarts.begin(), actualStarts.end(), false) != actualStarts.end());
        needsMask = needsMask || (std::find_if(sequenceLengths.begin(), sequenceLengths.end(), [maxSequenceLength](const size_t& currentSequenceLength) {
            return (currentSequenceLength != maxSequenceLength);
        }) != sequenceLengths.end());

        // If needed, create a mask to account for variability in lengths of specified sequences
        NDMaskPtr deviceValueMask;
        if (needsMask)
        {
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = MakeSharedObject<NDMask>(valueMaskShape, device);
            for (size_t i = 0; i < numSequences; ++i)
            {
                if (actualStarts[i])
                    deviceValueMask->MarkSequenceBegin({ 0, i });
                deviceValueMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
            }
        }

        return deviceValueMask;
    }

    //
    // Create NDMask for the 'sequences' if the 'sequences' do not have the same length.
    // It returns null if all the 'sequences' have the same length.
    //
    template <typename T>
    static NDMaskPtr CreateMask(size_t numElementsPerSample, const std::vector<std::vector<T>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device)
    {
        size_t numSequences = sequences.size();
        std::vector<size_t> sequenceLengths(numSequences);
        for (size_t i = 0; i < numSequences; ++i)
            sequenceLengths[i] = sequences[i].size() / numElementsPerSample;

        return CreateMask(sequenceLengths, sequenceStartFlags, device);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        if (oneHotSequences.size() == 0)
            InvalidArgument("Value::Create:: The number of sequences must be > 0");

        if (sampleShape.Rank() < 1)
            InvalidArgument("Value::Create:: The sample rank must be > 0");

        auto dimension = sampleShape[0];
        auto numElementsPerSample = sampleShape.SubShape(1).TotalSize();
        NDMaskPtr deviceValueMask = CreateMask(numElementsPerSample, oneHotSequences, sequenceStartFlags, DeviceDescriptor::CPUDevice());
        // If deviceValueMask is null, all the sequences have the same length.
        size_t maxSequenceLength = (deviceValueMask == nullptr) ? (oneHotSequences[0].size() / numElementsPerSample) : deviceValueMask->Shape()[0];
        size_t maxSequenceNumCols = maxSequenceLength * numElementsPerSample;

        size_t numSequences = oneHotSequences.size();
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        size_t numCSCCols = valueDataShape.SubShape(1).TotalSize() + 1;
        std::vector<SparseIndexType> colStarts(numCSCCols);
        std::vector<ElementType> nonZeroValues;
        std::vector<SparseIndexType> rowIndices;
        for (size_t i = 0; i < numSequences; ++i)
        {
            size_t currentSequenceNumCols = oneHotSequences[i].size();
            size_t j = 0;
            for (; j < currentSequenceNumCols; ++j)
            {
                colStarts[(i * maxSequenceNumCols) + j] = (SparseIndexType)nonZeroValues.size();
                nonZeroValues.push_back(1);
                if (oneHotSequences[i][j] >= dimension)
                    InvalidArgument("Value::Create: one-hot index value (%zu) exceeds vocabulary size (%zu).", oneHotSequences[i][j], dimension);

                rowIndices.push_back((SparseIndexType)(oneHotSequences[i][j]));
            }

            for (; j < maxSequenceNumCols; ++j)
                colStarts[(i * maxSequenceNumCols) + j] = (SparseIndexType)(nonZeroValues.size());
        }

        colStarts[numCSCCols - 1] = (SparseIndexType)(nonZeroValues.size());
        NDArrayViewPtr deviceValueData = MakeSharedObject<NDArrayView>(valueDataShape, colStarts.data(), rowIndices.data(), nonZeroValues.data(), nonZeroValues.size(), device, readOnly);
        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    template <typename ElementType>
    /*static*/ void Value::AppendSparseSequenceData(const NDArrayViewPtr& sequenceData, std::vector<SparseIndexType>& colStarts, std::vector<SparseIndexType>& rowIndices, std::vector<char>& nonZeroValues, size_t maxSequenceLength)
    {
        size_t existingNumNonZeroValues = nonZeroValues.size() / sizeof(ElementType);
        std::vector<SparseIndexType> currentSequencePaddedColStarts(maxSequenceLength);

        auto matrix = sequenceData->GetMatrix<ElementType>();
        matrix->TransferToDeviceIfNotThere(AsCNTKImplDeviceId(DeviceDescriptor::CPUDevice()), true);
        auto cpuSparseMatrix = matrix->m_CPUSparseMatrix;
        auto currentSequenceNumCols = matrix->GetNumCols();
        auto currentSequenceColStarts = cpuSparseMatrix->SecondaryIndexLocation();
        auto currentSequenceNumNonZeroValues = currentSequenceColStarts[currentSequenceNumCols] - currentSequenceColStarts[0];
        std::copy(cpuSparseMatrix->MajorIndexLocation(), cpuSparseMatrix->MajorIndexLocation() + currentSequenceNumNonZeroValues, std::back_inserter(rowIndices));
        std::copy((char*)(cpuSparseMatrix->Data()), (char*)(cpuSparseMatrix->Data() + currentSequenceNumNonZeroValues), std::back_inserter(nonZeroValues));

        for (size_t j = 0; j < currentSequenceNumCols; ++j)
            currentSequencePaddedColStarts[j] = existingNumNonZeroValues + (currentSequenceColStarts[j] - currentSequenceColStarts[0]);

        for (size_t j = currentSequenceNumCols; j < maxSequenceLength; ++j)
            currentSequencePaddedColStarts[j] = existingNumNonZeroValues + currentSequenceNumNonZeroValues;

        std::copy(currentSequencePaddedColStarts.begin(), currentSequencePaddedColStarts.end(), std::back_inserter(colStarts));
    }

    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<NDArrayViewPtr>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly, bool createNewCopy)
    {
        auto numSequences = sequences.size();
        if (numSequences == 0)
            InvalidArgument("Value::Create:: The number of sequences must be > 0");

        std::vector<size_t> sequenceLengths(numSequences);
        size_t maxSequenceLength = 0;
        auto dataType = sequences[0]->GetDataType();
        auto storageFormat = sequences[0]->GetStorageFormat();
        for (size_t i = 0; i < numSequences; ++i)
        {
            auto currentSequenceData = sequences[i];
            if (currentSequenceData->GetDataType() != dataType)
                InvalidArgument("Value::Create: The data for all sequences/samples must have the same data type");

            if (currentSequenceData->GetStorageFormat() != storageFormat)
                InvalidArgument("Value::Create: All NDArrayView objects must have the same storage format");

            if ((numSequences > 1) && (currentSequenceData->Device() != DeviceDescriptor::CPUDevice()))
                InvalidArgument("Value::Create: All NDArrayView objects must be located on the CPU");

            auto currentSequenceDataShape = currentSequenceData->Shape();

            // Since scalar samples can be rank=1 with dim=1, we automatically pad the sequence data shape with a leading axis 
            // of dim=1 if the sequence data shape's leading axis's dimensionality is not 1
            if ((sampleShape.Rank() == 1) && (sampleShape.TotalSize() == 1) && (currentSequenceDataShape.Rank() > 0) && (currentSequenceDataShape[0] != 1))
                currentSequenceDataShape = NDShape(1, 1).AppendShape(currentSequenceDataShape);

            if ((currentSequenceDataShape.Rank() < sampleShape.Rank()) || (currentSequenceDataShape.Rank() > (sampleShape.Rank() + 1)) || (currentSequenceDataShape.SubShape(0, sampleShape.Rank()) != sampleShape))
                InvalidArgument("Value::Create: The shape '%S' of the sequence %zu is not compatible with the sample shape '%S'", currentSequenceData->Shape().AsString().c_str(), i, sampleShape.AsString().c_str());

            sequenceLengths[i] = currentSequenceDataShape.SubShape(sampleShape.Rank()).TotalSize();
            maxSequenceLength = std::max(maxSequenceLength, sequenceLengths[i]);
        }

        bool isDataSparse = sequences[0]->IsSparse();
        if (isDataSparse && (sampleShape[0] != sampleShape.TotalSize()))
            InvalidArgument("Value::Create: For sparse data, the sample shape '%S' %s axis dimensionality must equal the total size (%zu) of the sample.",
                            sampleShape.AsString().c_str(), Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading", sampleShape.TotalSize());

        NDMaskPtr deviceValueMask = CreateMask(sequenceLengths, sequenceStartFlags, DeviceDescriptor::CPUDevice());

        NDArrayViewPtr valueData;
        NDShape valueDataShape = sampleShape.AppendShape({ maxSequenceLength, numSequences });
        if (numSequences == 1)
        {
            if (createNewCopy)
                valueData = sequences[0]->DeepClone();
            else
                valueData = sequences[0];

            // We can use the original buffer directly but need to reshape to the valueDataShape
            valueData = valueData->AsShape(valueDataShape);
        }
        else
        {
            if (isDataSparse)
            {
                if (storageFormat != StorageFormat::SparseCSC)
                    LogicError("Value::Create currently only SparseCSC format sparse data is supported");

                std::vector<SparseIndexType> colStarts;
                std::vector<SparseIndexType> rowIndices;
                std::vector<char> nonZeroValues;
                for (size_t i = 0; i < numSequences; ++i)
                {
                    switch (dataType)
                    {
                    case DataType::Float:
                        AppendSparseSequenceData<float>(sequences[i], colStarts, rowIndices, nonZeroValues, maxSequenceLength);
                        break;
                    case DataType::Double:
                        AppendSparseSequenceData<double>(sequences[i], colStarts, rowIndices, nonZeroValues, maxSequenceLength);
                        break;
                    default:
                        NOT_IMPLEMENTED;
                    }
                }

                auto totalNumNonZeroValues = nonZeroValues.size() / DataTypeSize(dataType);
                colStarts.push_back(totalNumNonZeroValues);

                switch (dataType)
                {
                case DataType::Float:
                    // TODO: In case of sparse we can directly create on target device
                    valueData = MakeSharedObject<NDArrayView>(valueDataShape, colStarts.data(), rowIndices.data(), (float*)nonZeroValues.data(), totalNumNonZeroValues, device, readOnly);
                    break;
                case DataType::Double:
                    valueData = MakeSharedObject<NDArrayView>(valueDataShape, colStarts.data(), rowIndices.data(), (double*)nonZeroValues.data(), totalNumNonZeroValues, device, readOnly);
                    break;
                default:
                    NOT_IMPLEMENTED;
                }
            }
            else
            {
                valueData = MakeSharedObject<NDArrayView>(dataType, valueDataShape, DeviceDescriptor::CPUDevice());
                auto maxSequenceSizeInElements = sampleShape.TotalSize() * maxSequenceLength;
                switch (dataType)
                {
                case DataType::Float:
                {
                    float* dataBuffer = valueData->WritableDataBuffer<float>();
                    for (size_t i = 0; i < numSequences; ++i)
                    {
                        const float* currentSequenceBuffer = sequences[i]->DataBuffer<float>();
                        auto currentSequenceSizeInElements = sequences[i]->Shape().TotalSize();
                        std::copy(currentSequenceBuffer, currentSequenceBuffer + currentSequenceSizeInElements, dataBuffer + (maxSequenceSizeInElements * i));
                    }
                    break;
                }
                case DataType::Double:
                {
                    double* dataBuffer = valueData->WritableDataBuffer<double>();
                    for (size_t i = 0; i < numSequences; ++i)
                    {
                        const double* currentSequenceBuffer = sequences[i]->DataBuffer<double>();
                        auto currentSequenceSizeInElements = sequences[i]->Shape().TotalSize();
                        std::copy(currentSequenceBuffer, currentSequenceBuffer + currentSequenceSizeInElements, dataBuffer + (maxSequenceSizeInElements * i));
                    }
                    break;
                }
                default:
                    NOT_IMPLEMENTED;
                }
            }
        }

        NDArrayViewPtr deviceValueData;
        if (device == valueData->Device())
        {
            if (readOnly)
                deviceValueData = valueData->Alias(readOnly);
            else
                deviceValueData = valueData;
        }
        else
            deviceValueData = valueData->DeepClone(device, readOnly);

        return MakeSharedObject<Value>(deviceValueData, deviceValueMask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly)
    {
        // Create a NDArrayView object wrapping each of the vectors representing a sequence 
        size_t numElementsPerSample = sampleShape.TotalSize();
        size_t numSequences = sequences.size();
        std::vector<NDArrayViewPtr> sequencesData;
        for (size_t i = 0; i < numSequences; ++i)
        {
            auto& currentSequence = sequences[i];
            if ((currentSequence.size() % numElementsPerSample) != 0)
                InvalidArgument("Value::Create: The number of elements (%zu) in the vector containing sequence data must be a multiple of the size (%zu) of specified sample shape '%S'",
                                currentSequence.size(), numElementsPerSample, sampleShape.AsString().c_str());

            auto sequenceLength = currentSequence.size() / numElementsPerSample;
            auto sequenceDataShape = sampleShape.AppendShape({ sequenceLength });
            sequencesData.push_back(MakeSharedObject<NDArrayView>(sequenceDataShape, currentSequence));
        }

        return Create(sampleShape, sequencesData, sequenceStartFlags, device, readOnly, /*createNewCopy =*/ true);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::CreateBatch(const NDShape& sampleShape, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly /*= false */)
    {
        auto shapeSize = sampleShape.TotalSize();
        if (batchData.size() % shapeSize != 0)
            InvalidArgument("The number of elements (%zu) in the vector containing batch data must be a multiple of the size (%zu) of the sample shape '%S'.",
                            batchData.size(), shapeSize, sampleShape.AsString().c_str());

        auto numOfSequences = batchData.size() / shapeSize;
        std::vector<NDArrayViewPtr> sequencesView(numOfSequences);
        for (size_t i = 0; i < numOfSequences; i++)
        {
            // Sequence lenght is 1.
            auto sequenceDataShape = sampleShape.AppendShape({ 1 });
            sequencesView[i] = MakeSharedObject<NDArrayView>(sequenceDataShape, batchData.data() + i * shapeSize, shapeSize, DeviceDescriptor::CPUDevice());
        }
        // Pass the empty seqStartFlags means all sequences have the start flag with true.
        return Create(sampleShape, sequencesView, {}, device, readOnly, /*createNewCopy =*/ true);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::CreateSequence(const NDShape& sampleShape, const std::vector<ElementType>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly /*= false */)
    {
        auto shapeSize = sampleShape.TotalSize();
        if (sequenceData.size() % shapeSize != 0)
            InvalidArgument("The number of elements (%zu) in the sequence data must be a multiple of the size (%zu) of the sample shape '%S'", 
                            sequenceData.size(), shapeSize, sampleShape.AsString().c_str());

        auto sequenceLength = sequenceData.size() / shapeSize;
        std::vector<NDArrayViewPtr> sequencesView(1);
        auto sequenceDataShape = sampleShape.AppendShape({ sequenceLength });
        sequencesView[0] = MakeSharedObject<NDArrayView>(sequenceDataShape, sequenceData);
        return Create(sampleShape, sequencesView, { sequenceStartFlag }, device, readOnly, /*createNewCopy =*/ true);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::CreateBatch(size_t dimension, const std::vector<size_t>& batchData, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        //TODO: avoid data copy.
        std::vector<std::vector<size_t>> input(batchData.size());
        for (size_t i = 0; i < batchData.size(); i++)
        {
            input[i] = {batchData[i]};
        }
        // Pass the empty seqStartFlags means all sequences have the start flag with true.
        return Create<ElementType>(dimension, input, {}, device, readOnly);
    }

    template <typename ElementType>
    /*static*/ ValuePtr Value::CreateSequence(size_t dimension, const std::vector<size_t>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        //TODO: avoid data copy.
        std::vector<std::vector<size_t>> input = { sequenceData };
        return Create<ElementType>(dimension, input, {sequenceStartFlag}, device, readOnly);
    }

    template <typename ElementType>
    /*static*/  ValuePtr Value::CreateSequence(const NDShape& sampleShape, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/)
    {
        auto sequenceShape = sampleShape.AppendShape({sequenceLength});
        auto sequenceData = MakeSharedObject<NDArrayView>(sequenceShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
        return Create(sampleShape, {sequenceData}, {sequenceStartFlag}, device, readOnly, false);
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
            InvalidArgument("Value::CopyFrom: Invalid source object; Cannot copy a Value with a mask into 'this' Value which does not have a mask.");

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

    void Value::GetSequenceStartsAndLengths(const NDMaskPtr& mask, std::vector<ptrdiff_t>& sequenceBeginIndices, std::vector<size_t>& sequenceLengths, size_t numDynamicAxes)
    {
        if (!mask)
            return;

        auto cpuMask = mask;
        if (mask->Device() != DeviceDescriptor::CPUDevice())
            cpuMask = mask->DeepClone(DeviceDescriptor::CPUDevice());

        const MaskKind* maskBuffer = cpuMask->DataBuffer();
        size_t maxNumTimeSteps, numSequences;
        std::tie(maxNumTimeSteps, numSequences) = GetNumTimeStepsAndSequences(mask->Shape(), numDynamicAxes);

        assert(sequenceLengths.size() == numSequences);
        assert(sequenceBeginIndices.size() == numSequences);

        for (size_t i = 0; i < numSequences; ++i)
        {
            MaskKind firstMaskEntry = maskBuffer[i * maxNumTimeSteps];
            if (firstMaskEntry == MaskKind::SequenceBegin)
                sequenceBeginIndices[i] = 0;
            else if (firstMaskEntry == MaskKind::Valid)
                sequenceBeginIndices[i] = Microsoft::MSR::CNTK::SentinelValueIndicatingUnspecifedSequenceBeginIdx;
            else
                LogicError("The first entry of a Value mask must be Valid or SequenceBegin");

            size_t currentSequenceLength = 1;
            bool currentSequenceEndAlreadyFound = false;
            for (size_t j = 1; j < maxNumTimeSteps; ++j)
            {
                if (maskBuffer[(i * maxNumTimeSteps) + j] == MaskKind::Invalid)
                    currentSequenceEndAlreadyFound = true;
                else
                {
                    if (currentSequenceEndAlreadyFound)
                        InvalidArgument("Invalid Value object; only trailing steps of a sequence can be masked.");

                    currentSequenceLength++;
                }
            }

            sequenceLengths[i] = currentSequenceLength;
        }
    }

    template <typename ElementType, typename DestType>
    void DirectCopy(const ElementType *source, size_t elementCount, std::vector<DestType>& dest);

    template <typename ElementType, typename DestType>
    void CopyDenseToOneHot(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest);

    template <typename ElementType>
    void Value::CopyVariableValueToVector(const Variable& outputVariable, std::vector<std::vector<ElementType>>& sequences)
    { 
        // Check the data type matches
        if (AsDataType<ElementType>() != GetDataType())
            InvalidArgument("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(GetDataType()));

        CopyVariableValueToImpl<ElementType, ElementType>(outputVariable, sequences);
    }

    template <typename ElementType>
    void Value::CopyVariableValueToVector(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences)
    {
        if (outputVariable.Shape()[0] != outputVariable.Shape().TotalSize())
            InvalidArgument("For sparse data, the outputVariable's leading axis dimensionality (%zu) must equal the total size (%zu) of the Variable '%S'.",
                            outputVariable.Shape()[0], outputVariable.Shape().TotalSize(), outputVariable.AsString().c_str());

        CopyVariableValueToImpl<ElementType, size_t>(outputVariable, sequences);
    }

    template <typename ValueType, typename DestType>
    void Value::CopyVariableValueToImpl(const Variable& outputVariable, std::vector<std::vector<DestType>>& sequences)
    {
        // PackedValue should be automatically unpacked when accessing Data() and Mask().
        size_t numOfSequences;
        size_t maxSequenceLen;
        std::tie(maxSequenceLen, numOfSequences) = GetSequenceAndBatchLength(outputVariable);

        if (sequences.size() < numOfSequences)
            RuntimeError("The size of output buffer (%zu) is smaller than the number (%zu) of sequences.", sequences.size(), numOfSequences);

        // Copy data to the CPU device if required.
        const ValueType *valueData;
        NDArrayViewPtr cpuArrayView;
        if (Device().Type() == DeviceKind::GPU)
        {
            // TODO: leverage sparse if the original NDArrayView is in spase.
            cpuArrayView = MakeSharedObject<NDArrayView>(GetDataType(), Shape(), DeviceDescriptor::CPUDevice());
            cpuArrayView->CopyFrom(*Data());
        }
        else if (Device().Type() == DeviceKind::CPU)
        {
            // TODO: direct process sparse data without copy
            if (GetStorageFormat() != StorageFormat::Dense)
            {
                cpuArrayView = MakeSharedObject<NDArrayView>(GetDataType(), Shape(), DeviceDescriptor::CPUDevice());
                cpuArrayView->CopyFrom(*Data());
            }
            else
            {
                cpuArrayView = Data();
            }
        } 
        else
        {
            LogicError("Invalid device type (%u).", (unsigned int)Device().Type());
        }

        valueData = cpuArrayView->DataBuffer<ValueType>();

        auto sampleSize = outputVariable.Shape().TotalSize();
        for (auto seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
        {
            size_t seqStart = seqIndex * maxSequenceLen;

            // The assumption here is that a sequence always start at 0 (no invaid mark at the beginning),
            // and ends at the first invalid mask. 
            // Therefore, no need to check NDMask again.
            // And the sequences has been resized to match the number of sequences and the length of each sequence in the Value object.

            // TODO: if function pointer or lambda could support template, switch to use them.
            if (std::is_same<DestType, size_t>::value)
            {
                // If the output is of the one-hot vector format, each value in sequences[seqIndex] is an index which represents a sample of sampleSize elements.
                CopyDenseToOneHot<ValueType, DestType>(valueData + seqStart * sampleSize, sequences[seqIndex].size(), sampleSize, sequences[seqIndex]);
            }
            else
            {
                // If the output is of the dense format, each value in sequences[seqIndex] represents an element of a sample.
                DirectCopy<ValueType, DestType>(valueData + seqStart * sampleSize, sequences[seqIndex].size(), sequences[seqIndex]);
            }
        }
    }

    std::pair<size_t, size_t> Value::GetSequenceAndBatchLength(const Variable& outputVariable)
    {
        Utils::VerifyVariableValueCompatibility(outputVariable, shared_from_this());

        size_t varRank = outputVariable.Shape().Rank();
        size_t maxSequenceLength = 1;
        size_t numSequences = 1;
        std::tie(maxSequenceLength, numSequences) = GetNumTimeStepsAndSequences(Shape().SubShape(varRank), outputVariable.DynamicAxes().size());

        return std::pair<size_t, size_t>(maxSequenceLength, numSequences);
    }

    template <typename ElementType>
    ElementType Value::AsScalar() const
    {
        if (Mask())
            LogicError("Value::AsScalar: Scalar Value object must not have an associated mask");

        return Data()->AsScalar<ElementType>();
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
                valueObject = Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, m_sampleDynamicAxes, *(m_packedData->GetMatrix<float>()), m_packedDataLayout, m_isReadOnly);
                break;
            case DataType::Double:
                valueObject = Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(m_sampleShape, m_sampleDynamicAxes, *(m_packedData->GetMatrix<double>()), m_packedDataLayout, m_isReadOnly);
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
                LogicError("The computed unpacked shape '%S' of the PackedValue object does not match the actual Data NDArrayView's shape '%S' after unpacking.",
                           m_unpackedShape.AsString().c_str(), m_data->Shape().AsString().c_str());
        }
    }

    template <typename ElementType, typename DestType>
    void DirectCopy(const ElementType *source, const size_t elementCount, std::vector<DestType>& dest)
    {
        if (!std::is_same<ElementType, DestType>::value)
            RuntimeError("Copy: Source and destination must be the same data type.");

        DestType *destData = dest.data();
        if (elementCount > dest.size())
            RuntimeError("Copy: The output buffer size (%zu) is smaller than the number (%zu) of source elements to copy.", dest.size(), elementCount);

        std::copy(source, source + elementCount, reinterpret_cast<ElementType *>(destData));
    }

    template <typename ElementType, typename DestType>
    void CopyDenseToOneHot(const ElementType *source, const size_t sampleCount, const size_t sampleSize, std::vector<DestType>& dest)
    {
        if (!std::is_same<DestType, size_t>::value)
            RuntimeError("Copy: The destination data type must be size_t.");

        const ElementType *currentp = source;
        const ElementType *lastp = source + sampleCount * sampleSize;
        size_t destIndex = 0;
        while (currentp < lastp)
        {
            size_t index = sampleSize;
            bool found = false;
            for (size_t i = 0; i < sampleSize; i++)
            {
                if (*currentp == 1)
                {
                    if (found)
                        RuntimeError("CopyDenseToOneHot: Cannot convert to onehot vector; more than one non-zero value in the sample.");

                    index = i;
                    found = true;
                }
                else if (*currentp != 0)
                    RuntimeError("CopyDenseToOneHot: Cannot convert to onehot vector; contains value other than 0/1.");

                currentp++;
            }
            if (!found)
                RuntimeError("CopyDenseToOneHot: Cannot convert to onehot vector; the sample does not have any non-zero value.");

            assert(index != sampleSize);
            dest[destIndex++] = static_cast<DestType>(index);
        }
        assert(currentp == lastp);
    }

    // Explicit template instantiations
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<std::vector<float>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<std::vector<double>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<float>(const NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::Create<double>(const NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateBatch<float>(const NDShape& sampleShape, const std::vector<float>& batchData, const DeviceDescriptor& device, bool readOnly /*= false */);
    template /*static*/ CNTK_API ValuePtr Value::CreateBatch<double>(const NDShape& sampleShape, const std::vector<double>& batchData, const DeviceDescriptor& device, bool readOnly /*= false */);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<float>(const NDShape& sampleShape, const std::vector<float>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly /*= false */);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<double>(const NDShape& sampleShape, const std::vector<double>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly /*= false */);
    template /*static*/ CNTK_API ValuePtr Value::CreateBatch<float>(size_t dimension, const std::vector<size_t>& batchData, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateBatch<double>(size_t dimension, const std::vector<size_t>& batchData, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<float>(size_t dimension, const std::vector<size_t>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<double>(size_t dimension, const std::vector<size_t>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<float>(const NDShape& sampleShape, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template /*static*/ CNTK_API ValuePtr Value::CreateSequence<double>(const NDShape& sampleShape, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly/* = false*/);
    template CNTK_API void Value::CopyVariableValueToVector<float>(const Variable& outputVariable, std::vector<std::vector<float>>& sequences);
    template CNTK_API void Value::CopyVariableValueToVector<double>(const Variable& outputVariable, std::vector<std::vector<double>>& sequences);
    template CNTK_API void Value::CopyVariableValueToVector<float>(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences);
    template CNTK_API void Value::CopyVariableValueToVector<double>(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences);
    template float Value::AsScalar<float>() const;
    template double Value::AsScalar<double>() const;
}
