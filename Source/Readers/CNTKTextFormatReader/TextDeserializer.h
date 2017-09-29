//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Descriptors.h"
#include "TextConfigHelper.h"
#include "Index.h"
#include "CorpusDescriptor.h"

namespace CNTK {

template <class ElemType>
class CNTKTextFormatReaderTestRunner;

class FileWrapper;

template <class ElemType>
class TextParser;


template <class ElemType>
class TextDataChunk;

struct TextParserInfo;

// TODO: more details when tracing warnings
// (e.g., buffer content around the char that triggered the warning)
template <class ElemType>
class TextDeserializer : public DataDeserializerBase {
public:
    TextDeserializer(CorpusDescriptorPtr corpus, const TextConfigHelper& helper, bool primary);
    ~TextDeserializer();

    // Retrieves a chunk of data.
    ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Get information about chunks.
    std::vector<ChunkInfo> ChunkInfos() override;

    // Get information about particular chunk.
    void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override;

    bool GetSequenceInfoByKey(const SequenceKey&, SequenceInfo&) override;

private:
    TextDeserializer(CorpusDescriptorPtr corpus, const std::wstring& filename, const vector<StreamDescriptor>& streams, bool primary = true);

    std::shared_ptr<TextParserInfo> m_parserInfo;

    const std::wstring m_filename;
    std::shared_ptr<FileWrapper> m_file;
    std::shared_ptr<Index> m_index;

    size_t m_maxAliasLength;
    std::map<std::string, size_t> m_aliasToIdMap;

    size_t m_chunkSizeBytes;
    bool m_cacheIndex;
    unsigned int m_numRetries; // specifies the number of times an unsuccessful
                               // file operation should be repeated (default value is 5).

    unsigned int m_traceLevel;
    unsigned int m_numAllowedErrors;
    bool m_skipSequenceIds;

    // Indicates if the sequence length is computed as the maximum 
    // of number of samples across all streams (inputs).
    bool m_useMaximumAsSequenceLength;

    std::vector<StreamInformation> m_streamInfos;
    std::vector<StreamDescriptor> m_streamDescriptors;

    // Corpus descriptor.
    CorpusDescriptorPtr m_corpus;

    typedef std::shared_ptr<TextDataChunk<ElemType>> TextChunkPtr;

    // Builds an index of the input data.
    void Initialize();

    // Given a descriptor, retrieves the data for the corresponding chunk from the file.
    void LoadChunk(TextChunkPtr& chunk, const ChunkDescriptor& descriptor);

    void SetNumRetries(unsigned int numRetries);

    void SetTraceLevel(unsigned int traceLevel);

    void SetMaxAllowedErrors(unsigned int maxErrors);

    void SetSkipSequenceIds(bool skip);

    void SetChunkSize(size_t size);

    void SetCacheIndex(bool value);

    friend class CNTKTextFormatReaderTestRunner<ElemType>;

    DISABLE_COPY_AND_MOVE(TextDeserializer);
};
}