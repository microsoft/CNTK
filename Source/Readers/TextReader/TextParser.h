//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include "DataDeserializerBase.h"
#include "Descriptors.h"

namespace Microsoft { namespace MSR { namespace CNTK {


struct Data {};

struct DenseData : Data {

    // capacity = expected number of samples * sample size
    DenseData(size_t capacity) : m_numberOfSamples(0)
    {
        m_buffer.reserve(capacity);
    }

    size_t m_numberOfSamples = 0;
    std::vector<float> m_buffer;
};

struct SparseData : Data {
    size_t m_numberOfSamples = 0;
    std::vector<float> m_buffer;
    std::vector<std::vector<size_t>> m_indices;
};

typedef std::vector<Data*> Sequence;

// TODO: implement/extend DataDeserializerBase
class TextParser {
private:

    enum TraceLevel {
        Error = 0,
        Warning = 1,
        Info = 2
    };

    enum State {
        Init = 0,
        Sign,
        IntegralPart,
        Period,
        FractionalPart,
        TheLetterE,
        ExponentSign,
        Exponent
    };

    static const auto BUFFER_SIZE = 256 * 1024;

    FILE* m_file = NULL;

    const std::vector<StreamDescriptor>& m_streams;

    Index* m_index = NULL;

    std::map<std::string, size_t> m_aliasToIdMap;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    char* m_bufferStart = NULL; //
    char* m_bufferEnd = NULL;
    char* m_pos = NULL; // buffer index

    char* m_scratch; // local buffer for string parsing

    unsigned int m_traceLevel = 0;
    unsigned int m_numErrors = 0;

    size_t m_maxAliasLength;

    // throw runtime exception when num errors gt some threshold
    void IncrementNumberOfErrors();

    void SetFileOffset(int64_t position);

    void SkipToNextValue(uint64_t& bytesToRead);
    void SkipToNextInput(uint64_t& bytesToRead);

    bool Fill();

    uint64_t GetFileOffset() { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    bool ReadSequenceId(size_t& id, uint64_t& bytesToRead);

    // reads an alias/name and converts it to an index into the m_streams vector.
    bool GetInputId(size_t& id, uint64_t& bytesToRead);

    //TODO: use a template here
    bool ReadValue(float& value, uint64_t& bytesToRead);

    bool ReadIndex(size_t& index, uint64_t& bytesToRead);

    bool ReadDenseSample(std::vector<float>& values, size_t sampleSize, uint64_t& bytesToRead);

    bool ReadSparseSample(std::vector<float>& values, std::vector<size_t>& indices, uint64_t& bytesToRead);

    // read one whole row (terminated by a row delimeter) of samples
    bool ReadRow(Sequence& sequence, uint64_t& bytesToRead);

    void Initialize();

    bool inline Available() { return m_pos != m_bufferEnd || Fill(); }

    TextParser(const TextParser&) = delete;
    TextParser& operator=(const TextParser&) = delete;

public:
    TextParser(const std::string& filename, const std::vector<StreamDescriptor>& streams);

    ~TextParser();

    const std::vector<SequenceDescriptor>& GetTimeline();

    std::vector<Sequence> LoadChunk(const ChunkDescriptor& chunk);

    Sequence LoadSequence(bool verifyId, const SequenceDescriptor& descriptor);

    void SetTraceLevel(unsigned int traceLevel);
};

}}}
