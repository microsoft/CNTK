//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <array>
#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {


struct Data 
{
    virtual ~Data() {};
};

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

// should these be shared pointers instead?
typedef std::vector<std::unique_ptr<Data>> Sequence;

class TextParser : public DataDeserializerBase {
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

    struct StreamInfo {
        StorageType m_type;
        size_t m_sampleSize;
    };

    static const auto BUFFER_SIZE = 256 * 1024;

    const std::string m_filename;
    FILE* m_file = NULL;

    const size_t m_numberOfStreams;
    StreamInfo * m_streamInfos;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    Index* m_index = NULL;

    int64_t m_fileOffsetStart;
    int64_t m_fileOffsetEnd;

    char* m_bufferStart = NULL; //
    char* m_bufferEnd = NULL;
    char* m_pos = NULL; // buffer index

    char* m_scratch; // local buffer for string parsing

    unsigned int m_traceLevel = 0;
    unsigned int m_numAllowedErrors = 0;
    bool m_skipSequenceIds;
    
    // All streams this reader provides.
    std::vector<StreamDescriptionPtr> m_streams;

    // Buffer to store feature data.
    std::vector<Sequence> m_loadedSequences;

    // throw runtime exception when num errors gt some threshold
    void IncrementNumberOfErrors();

    void SetFileOffset(int64_t position);

    void SkipToNextValue(int64_t& bytesToRead);
    void SkipToNextInput(int64_t& bytesToRead);

    bool Fill();

    int64_t GetFileOffset() { return m_fileOffsetStart + (m_pos - m_bufferStart); }

    bool ReadSequenceId(size_t& id, int64_t& bytesToRead);

    // reads an alias/name and converts it to an internal stream id (= stream index).
    bool GetInputId(size_t& id, int64_t& bytesToRead);

    //TODO: use a template here
    bool ReadValue(float& value, int64_t& bytesToRead);

    bool ReadIndex(size_t& index, int64_t& bytesToRead);

    bool ReadDenseSample(std::vector<float>& values, size_t sampleSize, int64_t& bytesToRead);

    bool ReadSparseSample(std::vector<float>& values, std::vector<size_t>& indices, int64_t& bytesToRead);

    // read one whole row (terminated by a row delimeter) of samples
    bool ReadRow(Sequence& sequence, int64_t& bytesToRead);

    void Initialize();

    bool inline Available() { return m_pos != m_bufferEnd || Fill(); }

    TextParser(const TextParser&) = delete;
    TextParser& operator=(const TextParser&) = delete;

protected:
    void FillSequenceDescriptions(SequenceDescriptions& timeline) const override;

public:
    TextParser(const std::string& filename, const vector<StreamDescriptor>& streams);

    TextParser(const TextConfigHelper& configHelper);

    ~TextParser();

    // Description of streams that this data deserializer provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // TODO: why doesn't this take a vector of SequenceDescription instead?
    // Get sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override;

    std::vector<Sequence> LoadChunk(const ChunkDescriptor& chunk);

    Sequence LoadSequence(bool verifyId, const SequenceDescriptor& descriptor);

    void SetTraceLevel(unsigned int traceLevel);
};

typedef std::shared_ptr<TextParser> TextParserPtr;
}}}
