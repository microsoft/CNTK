//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTK_ValueExtend.i -- Common interface to extend the Value class.
//

// This file contains methods extending the Value class in Python, C# and Java.
//
// Value
//
%extend CNTK::Value {
    // Instantiation template functions: dense input.
    static CNTK::ValuePtr CNTK::Value::CreateDenseFloat(const CNTK::NDShape& sampleShape, const std::vector<std::vector<float>>& sequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(sampleShape, sequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateDenseDouble(const CNTK::NDShape& sampleShape, const std::vector<std::vector<double>>& sequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(sampleShape, sequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateDenseFloat(const CNTK::NDShape& sampleShape, const std::vector<std::vector<float>>& sequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(sampleShape, sequences, sequenceStartFlags, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateDenseDouble(const CNTK::NDShape& sampleShape, const std::vector<std::vector<double>>& sequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(sampleShape, sequences, sequenceStartFlags, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateBatchFloat(const NDShape& sampleShape, const float *dataBuffer, int dataStart, int dataSize,
        const DeviceDescriptor& device, bool readOnly = false) {
        std::vector<float> batchData(dataBuffer + dataStart, dataBuffer + dataStart + dataSize);
        return CNTK::Value::CreateBatch<float>(sampleShape, batchData, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateBatchDouble(const NDShape& sampleShape, const double *dataBuffer, int dataStart, int dataSize,
        const DeviceDescriptor& device, bool readOnly = false) {
        std::vector<double> batchData(dataBuffer + dataStart, dataBuffer + dataStart + dataSize);
        return CNTK::Value::CreateBatch<double>(sampleShape, batchData, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceFloat(const NDShape& sampleShape, const float *dataBuffer, int dataSize,
        bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false) {
        std::vector<float> sequenceData(dataBuffer, dataBuffer + dataSize);
        return CNTK::Value::CreateSequence<float>(sampleShape, sequenceData, sequenceStartFlag, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceDouble(const NDShape& sampleShape, const double *dataBuffer, int dataSize,
        bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false) {
        std::vector<double> sequenceData(dataBuffer, dataBuffer + dataSize);
        return CNTK::Value::CreateSequence<double>(sampleShape, sequenceData, sequenceStartFlag, device, readOnly);
    }

    // Instantiation template functions: ND onehot vector input.
    static CNTK::ValuePtr CNTK::Value::CreateOneHotFloat(const CNTK::NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(sampleShape, oneHotSequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotDouble(const CNTK::NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(sampleShape, oneHotSequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotFloat(const CNTK::NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(sampleShape, oneHotSequences, sequenceStartFlags, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotDouble(const CNTK::NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(sampleShape, oneHotSequences, sequenceStartFlags, device, readOnly);
    }

    // Instantiation template functions: 1D onehot vector input.
    static CNTK::ValuePtr CNTK::Value::CreateOneHotFloat(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(CNTK::NDShape({dimension}), oneHotSequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotDouble(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences,
        const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(CNTK::NDShape({dimension}), oneHotSequences, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotFloat(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<float>(CNTK::NDShape({dimension}), oneHotSequences, sequenceStartFlags, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateOneHotDouble(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences,
        const std::vector<bool>& sequenceStartFlags, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::Create<double>(CNTK::NDShape({dimension}), oneHotSequences, sequenceStartFlags, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateBatchFloat(size_t dimension, const std::vector<size_t>& batchData,
        const DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateBatch<float>(dimension, batchData, device, false);
    }

    static CNTK::ValuePtr CNTK::Value::CreateBatchDouble(size_t dimension, const std::vector<size_t>& batchData,
        const DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateBatch<double>(dimension, batchData, device, false);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceFloat(size_t dimension, const std::vector<size_t>& sequenceData,
        bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<float>(dimension, sequenceData, sequenceStartFlag, device, false);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceDouble(size_t dimension, const std::vector<size_t>& sequenceData,
        bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<double>(dimension, sequenceData, sequenceStartFlag, device, false);
    }

    // Instantiation template functions: ND sparse input.
    static CNTK::ValuePtr CNTK::Value::CreateSequenceFloat(const CNTK::NDShape& sampleShape, size_t sequenceLength,
        const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues,
        bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<float>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceDouble(const CNTK::NDShape& sampleShape, size_t sequenceLength,
        const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues,
        bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<double>(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    // Instantiation template functions: 1D sparse input.
    static CNTK::ValuePtr CNTK::Value::CreateSequenceFloat(size_t dimension, size_t sequenceLength,
        const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues,
        bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<float>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    static CNTK::ValuePtr CNTK::Value::CreateSequenceDouble(size_t dimension, size_t sequenceLength,
        const CNTK::SparseIndexType* colStarts, const CNTK::SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues,
        bool sequenceStartFlag, const CNTK::DeviceDescriptor& device, bool readOnly = false) {
        return CNTK::Value::CreateSequence<double>(dimension, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
    }

    // Instantiation template functions: copy value
    void CNTK::Value::CopyVariableValueToFloat(const CNTK::Variable& outputVariable, std::vector<std::vector<float>>& sequences)
    {
        return self->CopyVariableValueTo<float>(outputVariable, sequences);
    }

    void CNTK::Value::CopyVariableValueToDouble(const CNTK::Variable& outputVariable, std::vector<std::vector<double>>& sequences)
    {
        return self->CopyVariableValueTo<double>(outputVariable, sequences);
    }

    void CNTK::Value::CopyVariableValueToFloat(const Variable& outputVariable, int* sequenceLength, std::vector<SparseIndexType>& colStarts, 
        std::vector<SparseIndexType>& rowIndices, std::vector<float>& nonZeroValues, int* numNonZeroValues)
    {
        size_t sequenceLengthSizeT, numNonZeroValuesSizeT;
        self->CopyVariableValueTo<float>(outputVariable, sequenceLengthSizeT, colStarts, rowIndices, nonZeroValues, numNonZeroValuesSizeT);
        *sequenceLength = (int)sequenceLengthSizeT;
        *numNonZeroValues = (int)numNonZeroValuesSizeT;
    }

    void CNTK::Value::CopyVariableValueToDouble(const Variable& outputVariable, int* sequenceLength, std::vector<SparseIndexType>& colStarts, 
        std::vector<SparseIndexType>& rowIndices, std::vector<double>& nonZeroValues, int* numNonZeroValues)
    {
        size_t sequenceLengthSizeT, numNonZeroValuesSizeT;
        self->CopyVariableValueTo<double>(outputVariable, sequenceLengthSizeT, colStarts, rowIndices, nonZeroValues, numNonZeroValuesSizeT);
        *sequenceLength = (int)sequenceLengthSizeT;
        *numNonZeroValues = (int)numNonZeroValuesSizeT;
    }
}