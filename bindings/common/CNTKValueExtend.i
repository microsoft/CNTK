// This file contains methods extending the Value class. 

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
}