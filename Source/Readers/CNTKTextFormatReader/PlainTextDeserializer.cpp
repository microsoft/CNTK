//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements a plain-text deserializer, for lines of space-separated word tokens.
//

#include "stdafx.h"
#include "Basics.h"
#include "DataDeserializer.h"
#include "ReaderConstants.h"
#include "File.h"

#include <vector>
#include <memory>
#include <unordered_map>

#define let const auto

namespace CNTK {

using namespace std;
using namespace Microsoft::MSR::CNTK;
using namespace msra::strfun;

// helpers
static void LoadTextFileAsCharArray(vector<char>& buf, const wstring& path, uint64_t begin = 0, uint64_t end = SIZE_MAX)
{
    File f(path, fileOptionsRead | fileOptionsBinary);
    if (end == SIZE_MAX)
        end = f.Size();
    f.SetPosition(begin);
    buf.reserve(end - begin + 1); // add 1 byte as courtesy to the caller, which needs it for the 0-terminator
    freadOrDie(buf, end - begin, f);
}

static inline bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\r'; }

// ===========================================================================
// This class implements a plain-text deserializer, for lines of space-separate tokens
// as needed for many language-modeling and translation tasks.
// ===========================================================================

class PlainTextDeserializerImpl : public DataDeserializer
{
public:
    // -----------------------------------------------------------------------
    // per-stream configuration
    // -----------------------------------------------------------------------

    struct PlainTextVocabularyConfiguration
    {
        const std::wstring fileName;
        const std::wstring insertAtStart;
        const std::wstring insertAtEnd;
        const std::wstring substituteForUnknown;
    };
    struct PlainTextStreamConfiguration : public StreamInformation
    {
        PlainTextStreamConfiguration(StreamInformation&& streamInfo, vector<wstring>&& fileNames, PlainTextVocabularyConfiguration&& vocabularyConfig) :
            StreamInformation(move(streamInfo)), m_fileNames(move(fileNames)), m_vocabularyConfig(vocabularyConfig)
        { }
        const vector<wstring> m_fileNames;
        const PlainTextVocabularyConfiguration m_vocabularyConfig;
    };

    // -----------------------------------------------------------------------
    // vocabulary (maps grapheme to index)
    // -----------------------------------------------------------------------

    class Vocabulary
    {
    public:
        Vocabulary(const PlainTextVocabularyConfiguration& config)
        {
            // load vocab file
            vector<char> buf;
            LoadTextFileAsCharArray(buf, config.fileName);
            // tokenize it
            m_words = split(string(buf.begin(), buf.end()), "\r\n");
            // now put them all into an STL hash map
            m_wordMap.reserve(m_words.size() + 3);
            for (size_t i = 0; i < m_words.size(); i++)
            {
                let res = m_wordMap.insert(make_pair(m_words[i], i));
                if (!res.second)
                    InvalidArgument("Vocabulary: Duplicate word '%s' in vocabulary file %S.", m_words[i].c_str(), config.fileName.c_str());
            }
            // add the special tokens
            // These are supposed to be in the vocabulary, but they are missing in some configs.
            // They are added to the end of the vocab in the order sent-start, sent-end, unknonw-substitute.
            // Note that sent-start and sent-end are only special if they are specified to be added on the fly.
            // Special tokens must only use characters in ASCII range, since we don't know the encoding of the vocab file, which we match as 8-bit strings.
            for (let& specialToken : vector<string>{ AsAsciiString(config.insertAtStart), AsAsciiString(config.insertAtEnd), AsAsciiString(config.substituteForUnknown) })
            {
                if (!specialToken.empty())
                {
                    let res = m_wordMap.insert(make_pair(specialToken, m_words.size()));
                    if (res.second)
                    {
                        fprintf(stderr, "Vocabulary: Special token '%s' missing, adding as index %d. %S.\n", specialToken.c_str(), (int)m_words.size(), config.fileName.c_str());
                        m_words.push_back(specialToken);
                    }
                }
            }
            // precompute the word index of the unknown-substitute token if given
            m_unknownId = TryGetWordId(config.substituteForUnknown);
        }

        // look up a word in the vocabulary
        // If the word is not found, and an unknown-substitute was specified, that is returned instead.
        // Otherwise it fails.
        size_t operator[](const char* word) const
        {
            let res = m_wordMap.find(word);
            if (res != m_wordMap.end())
                return res->second;
            else if (m_unknownId != SIZE_MAX)
                return m_unknownId;
            else
                InvalidArgument("Vocabulary: Encountered unknown word '%s' when no subsitute was specified.", word);
        }

        // special look-up function for special tokens which can be "" (it returns the sentinel value SIZE_MAX for those)
        size_t TryGetWordId(const wstring& word) const
        {
            if (word.empty())
                return SIZE_MAX;
            else
                return operator[](AsAsciiString(word).c_str());
        }

        size_t size() const { return m_words.size(); }
    private:
        vector<string> m_words;                  // [wordId] the word list
        unordered_map<string, size_t> m_wordMap; // [word] -> wordId map
        size_t m_unknownId;                      // wordId of unknown-substitute, or SIZE_MAX if not specified
    };

    // -----------------------------------------------------------------------
    // per-stream data
    // This holds all information for one stream.
    // -----------------------------------------------------------------------

    struct PlainTextStream : public PlainTextStreamConfiguration
    {
        // constructor
        PlainTextStream(const PlainTextStreamConfiguration& streamConfig) :
            PlainTextStreamConfiguration(streamConfig),
            m_vocabulary(streamConfig.m_vocabularyConfig)
        {
            if (NDShape{ m_vocabulary.size() } != m_sampleLayout)
                // BUGBUG: AsString() cannot be called from outside--how to do that?
                InvalidArgument("PlainTextDeserializer: Sample layout %S does not match vocabulary size %d for %S.",
                                m_sampleLayout.AsString().c_str(), (int)m_vocabulary.size(), streamConfig.m_vocabularyConfig.fileName.c_str());

            // make sure we don't overflow. Word ids must fit into SparseIndexType (==int).
            let maxId = m_vocabulary.size() - 1;
            if ((size_t)(SparseIndexType)maxId != maxId)
                InvalidArgument("PlainTextDeserializer: Vocabulary size too large for SparseIndexType for %S.", streamConfig.m_vocabularyConfig.fileName.c_str());
        }

        // initialization helpers
        // This is not a fast operation, so call this after the quick construction steps have completed and error-checked.
        void IndexAllFiles(int traceLevel)
        {
            // index all files
            // That is, determine all line offsets and word counts and remember them in m_textLineRefs.
            // TODO: Later this can be cached.
            vector<char> fileAsCString;
            let numWords0 = !m_vocabularyConfig.insertAtStart.empty() + !m_vocabularyConfig.insertAtEnd.empty(); // initial count
            m_textLineRefs.resize(m_fileNames.size());
            m_totalNumWords = 0;
            for (size_t i = 0; i < m_fileNames.size(); i++)
            {
                let& fileName = m_fileNames[i];
                auto& thisFileTextLineRefs = m_textLineRefs[i];
                // load file
                // TODO: This should use smaller memory size; no need to allocate a GB of RAM for this indexing process.
                LoadTextFileAsCharArray(fileAsCString, fileName);
                if (fileAsCString.back() != '\n') // this is required for correctness of the subsequent loop
                {
                    fprintf(stderr, "PlainTextStream: Missing newline character at end of file %S.\n", fileName.c_str());
                    fileAsCString.push_back('\n');
                }
                // determine line offsets and per-line word counts
                // This is meant to run over files of GB size and thus be fast.
                size_t numWordsInFile = 0;
                const char* pdata = fileAsCString.data(); // start of buffer
                const char* pend = pdata + fileAsCString.size();
                for (const char* p0 = pdata; p0 < pend; p0++) // loop over lines
                {
                    const char* p; // running pointer
                    for (p = p0; IsSpace(*p); p++) // skip initial spaces
                        ;
                    size_t numWords = numWords0;
                    bool prevIsSpace = true;
                    for (p = p0; *p != '\n'; p++) // count all space/non-space transitions in numWords
                    {
                        let thisIsSpace = IsSpace(*p);
                        numWords += prevIsSpace && !thisIsSpace;
                        prevIsSpace = thisIsSpace;
                    }
                    thisFileTextLineRefs.push_back(TextLineRef{ (uint64_t)(p0 - pdata), numWords });
                    numWordsInFile += numWords;
                    p0 = p; // now points to '\n'; will be increased above
                }
                thisFileTextLineRefs.push_back(TextLineRef{ (uint64_t)(pend - pdata), 0 }); // add one extra token
                m_totalNumWords += numWordsInFile;
                // log
                if (traceLevel >= 1)
                    fprintf(stderr, "PlainTextDeserializer: %d words, %d lines. %S\n", (int)numWordsInFile, (int)thisFileTextLineRefs.size() - 1, fileName.c_str());
            }
        }

        // information organized by files
        struct TextLineRef
        {
            const uint64_t beginOffset; // byte offset of this line in the file
            const size_t numWords;      // number of final word tokens in the line. This includes <s> and </s> if specified to be added.
        };
        vector<vector<TextLineRef>> m_textLineRefs; // [fileIndex][lineIndex] -> (byte offset of line, #words in this line); has one extra line beyond last line to simplify length calculation
        size_t m_totalNumWords;  // total word count for this stream
        Vocabulary m_vocabulary; // mapping from word strings to neuron indices
    };

    // -----------------------------------------------------------------------
    // implementation of PlainTextDeserializer itself
    // -----------------------------------------------------------------------

    // constructor indexes all files already (determines chunks and line offsets in files)
    PlainTextDeserializerImpl(const vector<PlainTextDeserializerImpl::PlainTextStreamConfiguration>& streamConfigs, size_t chunkSizeBytes, bool cacheIndex,
        DataType elementType, bool primary, int traceLevel) :
        m_streams(streamConfigs.begin(), streamConfigs.end()), m_cacheIndex(cacheIndex), m_elementType(elementType), m_primary(primary), m_traceLevel(traceLevel)
    {
        if (m_streams.empty())
            InvalidArgument("PlainTextDeserializer: At least one stream must be specified.");

        // index all files
        for (auto& stream : m_streams)
        {
            stream.IndexAllFiles(m_traceLevel);
            if (stream.m_totalNumWords == 0)
                InvalidArgument("PlainTextDeserializer: Corpus must not be empty.");
        }

        // statistics and verify that all files have the same #lines
        m_totalNumLines = 0;
        size_t totalNumSrcWords = 0;
        let& firstFileTextLineRefs = m_streams[0].m_textLineRefs; // [fileIndex]
        for (size_t j = 0; j < firstFileTextLineRefs.size(); j++) // compute statistics from first stream
        {
            m_totalNumLines += firstFileTextLineRefs[j].size() - 1; // -1 because m_textLineRefs[fileIndex][*] has one extra entry at end
            totalNumSrcWords += m_streams[0].m_totalNumWords;
        }
        for (size_t i = 1; i < m_streams.size(); i++) // all additional streams must match the first stream's statistics
        {
            let& thisFileTextLineRefs = m_streams[i].m_textLineRefs;
            if (firstFileTextLineRefs.size() != thisFileTextLineRefs.size())
                InvalidArgument("PlainTextDeserializer: All streams must have the same number of files.");
            for (size_t j = 0; j < firstFileTextLineRefs.size(); j++)
                if (firstFileTextLineRefs[j].size() != thisFileTextLineRefs[j].size())
                    InvalidArgument("PlainTextDeserializer: Files across streams must have the same number of lines.");
        }

        if (m_traceLevel >= 1)
            fprintf(stderr, "PlainTextDeserializer: %d files with %d lines and %d words in the first stream out of %d\n",
            (int)m_streams[0].m_textLineRefs.size(), (int)m_totalNumLines, (int)totalNumSrcWords, (int)m_streams.size());

        // form chunks
        // Chunks are formed by byte size of the first stream. The entire stream (which may
        // span multiple files) is divided up into chunks of approximately chunkSizeBytes bytes.
        // The m_chunkRefs array then records the end position (file/line) of each chunk.
        // The start position is computed on the fly later when loading a chunk.
        let& definingTextLineRefs = m_streams[m_definingStream].m_textLineRefs;
        size_t totalDataSize = 0;
        for (let& lineRefs : definingTextLineRefs)
            totalDataSize += lineRefs.back().beginOffset;
        let numTargetChunks = max(totalDataSize / chunkSizeBytes, (size_t)1); // (tend towards less chunks)
        let roundedChunkSize = (totalDataSize + numTargetChunks - 1) / numTargetChunks;
        ChunkRef chunkRef;
        for (chunkRef.m_endFileIndex = 0; chunkRef.m_endFileIndex < definingTextLineRefs.size(); chunkRef.m_endFileIndex++)
        {
            let& fileLineRefs = definingTextLineRefs[chunkRef.m_endFileIndex];
            for (chunkRef.m_endLineNo = 1; chunkRef.m_endLineNo < fileLineRefs.size(); chunkRef.m_endLineNo++)
            {
                // add line to chunk
                chunkRef.m_numberOfSequences++;
                chunkRef.m_numberOfSamples += fileLineRefs[chunkRef.m_endLineNo - 1].numWords;
                chunkRef.m_size += fileLineRefs[chunkRef.m_endLineNo].beginOffset - fileLineRefs[chunkRef.m_endLineNo - 1].beginOffset;
                chunkRef.m_endSequenceId++;
                // if chunk is large enough, or if we hit the end, then flush the chunk
                if (chunkRef.m_size >= roundedChunkSize ||
                    (chunkRef.m_endLineNo + 1 == fileLineRefs.size() && chunkRef.m_endFileIndex + 1 ==  definingTextLineRefs.size()))
                {
                    assert(chunkRef.m_id == m_chunkRefs.size());
                    m_chunkRefs.push_back(chunkRef);
                    // and reset for forming next chunk
                    chunkRef.m_id++;
                    chunkRef.m_numberOfSamples = 0;
                    chunkRef.m_numberOfSequences = 0;
                    chunkRef.m_size = 0;
                }
            }
        }
        assert(chunkRef.m_size == 0); // we must have flushed it out inside the loop already
    }

    ///
    /// Gets stream information for all streams this deserializer exposes.
    ///
    virtual std::vector<StreamInformation> StreamInfos() override
    {
        return vector<StreamInformation>(m_streams.begin(), m_streams.end());
    }

    ///
    /// Gets metadata for chunks this deserializer exposes.
    ///
    virtual std::vector<ChunkInfo> ChunkInfos() override
    {
        return std::vector<ChunkInfo>(m_chunkRefs.begin(), m_chunkRefs.end());
    }

    ///
    /// Gets sequence infos for a given a chunk.
    ///
    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override
    {
        let& chunkRef = m_chunkRefs[chunkId];
        let& definingTextLineRefs = m_streams[m_definingStream].m_textLineRefs;
        result.clear();
        // iterate over all lines in the chunk and gather their SequenceInfo records
        size_t fileIndex  = chunkId == 0 ? 0 : m_chunkRefs[chunkId - 1].m_endFileIndex;
        size_t lineNo     = chunkId == 0 ? 0 : m_chunkRefs[chunkId - 1].m_endLineNo;
        size_t sequenceId = chunkId == 0 ? 0 : m_chunkRefs[chunkId - 1].m_endSequenceId;
        while (lineNo != chunkRef.m_endLineNo || fileIndex != chunkRef.m_endFileIndex)
        {
            let& fileLineRefs = definingTextLineRefs[fileIndex];
            // if we are pointing to the last entry of a file, advance to the next file
            if (lineNo == fileLineRefs.size() - 1)
            {
                lineNo = 0;
                fileIndex++;
                continue;
            }
            // emit this line's info
            let& lineRef = fileLineRefs[lineNo];
            result.emplace_back(SequenceInfo
            {
                result.size(),                  // m_indexInChunk
                (unsigned int)lineRef.numWords, // m_numberOfSamples
                ChunkIdType(chunkId),           // m_chunkId
                SequenceKey(sequenceId, 0)      // m_key
            });
            sequenceId++;
            lineNo++;
        }
        if (result.size() != chunkRef.m_numberOfSequences || sequenceId != chunkRef.m_endSequenceId)
            LogicError("PlainTextDeserializer: SequenceInfosForChunk ran into a discrepancy on sequence counts.");
    }

    ///
    /// Gets sequence information given the one of the primary deserializer.
    /// Used for non-primary deserializers.
    /// Returns false if the corresponding secondary sequence is not valid.
    ///
    virtual bool GetSequenceInfo(const SequenceInfo& primary, SequenceInfo& result) override
    {
        primary; result;
        NOT_IMPLEMENTED; // TODO: What should this do? Just return false?
    }

private:
    // -----------------------------------------------------------------------
    // data structures returned by GetChunk()
    // -----------------------------------------------------------------------

    // ChunkRef is a reference to the data of a chunk. It stores the end position (one
    // line beyond end) as file index/line number. The start position is determined when
    // needed as the previous chunk's end position (or 0/0 for the first chunk).
    struct ChunkRef : public ChunkInfo
    {
        size_t m_size = 0;          // size of this chunk in bytes
        size_t m_endFileIndex = 0;  // file and line number of end position (and one after last line)
        size_t m_endLineNo = 0;     // (start position is computed from the previous chunk end when loading the data)
        size_t m_endSequenceId = 0; // sequence identifier (=global line number) of end entry in this chunk; that is, the one after the last line

        ChunkRef() : ChunkInfo{ 0, 0, 0 } { }
    };

    // PlainTextChunk loads and holds the data for one data chunk when requested.
    // It carries the GetSequence() method, which returns individual sequences from 
    // the loaded chunk data.
    class PlainTextChunk : public Chunk
    {
    public:
        // constructor loads the chunk data and parses it into binary id sequences
        PlainTextChunk(const ChunkRef& chunkRef, const PlainTextDeserializerImpl& owningDeserializer) :
            m_chunkRef(chunkRef), m_owningDeserializer(owningDeserializer)
        {
            let chunkId = chunkRef.m_id;
            let numStreams = owningDeserializer.m_streams.size();
            m_data.resize(numStreams);
            m_endOffsets.resize(numStreams);
            // first sequence id
            m_firstSequenceId = chunkId == 0 ? 0 : owningDeserializer.m_chunkRefs[chunkId - 1].m_endSequenceId;
            // load all streams' data
            size_t maxSequenceLength = 0;
            vector<char> buf;
            vector<SparseIndexType> data; // we build the sequence of word indices here
            for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
            {
                auto& endOffsets = m_endOffsets[streamIndex];
                data.clear();
                endOffsets.clear();
                endOffsets.reserve(chunkRef.m_numberOfSequences);
                let& stream = owningDeserializer.m_streams[streamIndex];
                let& textLineRefs = stream.m_textLineRefs;
                let& vocabulary = stream.m_vocabulary;
                // fetch special symbol ids if given (will be SIZE_MAX if not given)
                let insertAtStartId = vocabulary.TryGetWordId(stream.m_vocabularyConfig.insertAtStart);
                let insertAtEndId   = vocabulary.TryGetWordId(stream.m_vocabularyConfig.insertAtEnd);
                let numWords0 = !(insertAtStartId == SIZE_MAX) + !(insertAtEndId == SIZE_MAX); // number of words we generate on the fly
                // generate all sequences for this chunk, for the current stream
                size_t fileIndex  = chunkId == 0 ? 0 : owningDeserializer.m_chunkRefs[chunkId - 1].m_endFileIndex;
                size_t lineNo     = chunkId == 0 ? 0 : owningDeserializer.m_chunkRefs[chunkId - 1].m_endLineNo;
                while (fileIndex < chunkRef.m_endFileIndex ||
                       (fileIndex == chunkRef.m_endFileIndex && lineNo < chunkRef.m_endLineNo))
                {
                    // fetch lines until endLineNo if in same file or end of file
                    let& fileLineRefs = textLineRefs[fileIndex];
                    let endLineNo = (fileIndex < chunkRef.m_endFileIndex) ? fileLineRefs.size() - 1 : chunkRef.m_endLineNo;
                    let bufBeginOffset = fileLineRefs[lineNo].beginOffset; // offset of first line in buffer
                    let& fileName = stream.m_fileNames[fileIndex];
                    LoadTextFileAsCharArray(buf, fileName, bufBeginOffset, fileLineRefs[endLineNo].beginOffset);
                    buf.push_back('\n'); // ensure line-end marker for consistency checking
                    for (; lineNo < endLineNo; lineNo++)
                    {
                        // insert <s> if requested
                        if (insertAtStartId != SIZE_MAX)
                            data.push_back((SparseIndexType)insertAtStartId);
                        // word tokens
                        auto* p    = buf.data() + fileLineRefs[lineNo    ].beginOffset - bufBeginOffset;
                        auto* pend = buf.data() + fileLineRefs[lineNo + 1].beginOffset - bufBeginOffset; // (for consistency check only)
                        let numWords = fileLineRefs[lineNo].numWords;
                        for (size_t i = numWords0; i < numWords; i++)
                        {
                            // locate next word
                            // For simplicity, We do not distinguish space and newline here, since we already counted everything.
                            while (IsSpace(*p)) // skip to next word beginning
                                p++;
                            if (p >= pend)
                                LogicError("PlainTextDeserializer: Unexpectedly run over end of line while consuming input words of line %d in %S.", (int)lineNo, fileName.c_str());
                            const char* word = p;
                            while (!IsSpace(*p) && *p != '\n')
                                p++;
                            *p++ = 0; // 'word' is now a 0-terminated string
                            // get word id
                            data.push_back((SparseIndexType)vocabulary[word]);
                        }
                        // insert </s> if requested
                        if (insertAtEndId != SIZE_MAX)
                            data.push_back((SparseIndexType)insertAtEndId);
                        endOffsets.push_back(data.size());
                        // keep track of max length
                        if (numWords > maxSequenceLength)
                            maxSequenceLength = numWords;
                    }
                    // advance to beginning of next file
                    fileIndex++;
                    lineNo = 0;
                }
                // consistency check
                if (endOffsets.size() != chunkRef.m_numberOfSequences)
                    LogicError("PlainTextDeserializer: Chunk %d's actual sequence count inconsistent with underlying files.", (int)chunkId);
                // move the data array into a shared_ptr
                m_data[streamIndex] = shared_ptr<SparseIndexType>(new SparseIndexType[data.size()], default_delete<SparseIndexType[]>());
                copy(data.begin(), data.end(), m_data[streamIndex].get());
            }
            m_onesFloatBuffer .reset(new float [maxSequenceLength], default_delete<float []>()); // dense data arrays for sparse data; constant 1 across all sequences and streams, hence has length of longest sequence
            m_onesDoubleBuffer.reset(new double[maxSequenceLength], default_delete<double[]>());
            for (size_t i = 0; i < maxSequenceLength; i++)
                m_onesDoubleBuffer.get()[i] = m_onesFloatBuffer.get()[i] = 1;
        }

        // PlainTextSequenceData is the data structure returned by GetSequence().
        // It holds references to the data of one sequence in one stream.
        class PlainTextSequenceData : public SparseSequenceData
        {
        public:
            PlainTextSequenceData(const shared_ptr<SparseIndexType>& indicesForWholeChunk, size_t beginOffset, size_t numWords, SequenceKey key,
                                  const NDShape& sampleShape,
                                  const shared_ptr<float>& onesFloatBuffer, const shared_ptr<double>& onesDoubleBuffer, DataType elementType) :
                SparseSequenceData((unsigned int)numWords), m_sampleShape(sampleShape), m_indicesForWholeChunk(indicesForWholeChunk)
            {
                // BUGBUG (API): SequenceDataBase should accept numberOfSamples as a size_t and check the range
                m_elementType = elementType;
                if (elementType == DataType::Float) // indicesForWholeChunk must point to an array of 1.0 values with at least numWords elements
                    m_ones = static_pointer_cast<void>(onesFloatBuffer);
                else if (elementType == DataType::Double)
                    m_ones = onesDoubleBuffer;
                else
                    LogicError("PlainTextDeserializer: Unsupported DataType.");
                m_key = key;
                m_indices = indicesForWholeChunk.get() + beginOffset; // BUGBUG (API): This should be a const pointer
                m_nnzCounts.resize(numWords, 1);                      // BUGBUG (API): This should be a pointer as well
                m_totalNnzCount = (SparseIndexType)numWords;
            }
            ~PlainTextSequenceData() { }
            virtual const NDShape& /*SparseSequenceData::*/GetSampleShape() override final { return m_sampleShape; } // BUGBUG (API): Shouldn't these be const?
            virtual const void*    /*SparseSequenceData::*/GetDataBuffer()  override final { return m_ones.get(); }
        private:
            const NDShape& m_sampleShape;
            shared_ptr<const void> m_ones;                      // ref-count to the constant array of ones (float or double)
            shared_ptr<SparseIndexType> m_indicesForWholeChunk; // ref-count of the index data for the whole chunk
        };

        ///
        /// Gets data for the sequence with the given index.
        /// result contains a SequenceDataPtr for every input stream declared by the
        /// deserializer that produced this chunk.
        ///
        virtual void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override final
        {
            if (sequenceIndex > m_chunkRef.m_numberOfSequences)
                LogicError("PlainTextDeserializer: GetSequence sequenceIndex parameter (%d) out of bounds.", (int)sequenceIndex);
            let numStreams = m_data.size();
            result.resize(numStreams);
            for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
            {
                let& data = m_data[streamIndex];
                let& endOffsets = m_endOffsets[streamIndex];
                let& stream = m_owningDeserializer.m_streams[streamIndex];

                let beginOffset = sequenceIndex == 0 ? 0 : endOffsets[sequenceIndex - 1];
                let numWords = endOffsets[sequenceIndex] - beginOffset;
                let sequenceId = m_firstSequenceId + sequenceIndex;
                result[streamIndex] = make_shared<PlainTextSequenceData>(data, beginOffset, numWords, SequenceKey(sequenceId, 0),
                                                                         stream.m_sampleLayout,
                                                                         m_onesFloatBuffer, m_onesDoubleBuffer, stream.m_elementType);
            }
        }
    private:
        const ChunkRef& m_chunkRef;             // reference to the chunk descriptor that this Chunk represents
        const PlainTextDeserializerImpl& m_owningDeserializer; // reference to the owning PlainTextDeserializer instance
        vector<shared_ptr<SparseIndexType>> m_data; // [streamIndex][n] concatenated data for all sequences for all streams
        vector<vector<size_t>> m_endOffsets;    // [streamIndex][lineNo] end offset of line (the begin offset is determined from lineNo-1)
        size_t m_firstSequenceId;               // sequence id of first sequence in this chunk
        shared_ptr<float>  m_onesFloatBuffer;   // dense buffer of ones for use as values for the sparse data; used across all sequences and streams, hence has length of longest sequence
        shared_ptr<double> m_onesDoubleBuffer;  // Note: It appears that PlainTextSequenceData lives longer than Chunk at the end of the data sweep; hence these are shared_ptrs
    };
    friend class PlainTextChunk;
public:

    ///
    /// Gets chunk data given its id.
    ///
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override
    {
        return make_shared<PlainTextChunk>(m_chunkRefs[chunkId], *this);
    }

private:
    vector<PlainTextStream> m_streams; // information of all streams, including configuration
    const bool m_cacheIndex;           // (not implemented) if true then persist the index to disk
    const DataType m_elementType;
    const bool m_primary;              // whether this deserializer is the primary one (determines chunking) or not
    const int m_traceLevel;
    const size_t m_definingStream = 0; // chunks are defined by this stream  --TODO: we may change this, e.g. select the first that has definesMBSize set? (currently none has)

    // working data
    size_t m_totalNumLines;       // total line count for the stream (same for all streams)
    vector<ChunkRef> m_chunkRefs; // descriptors of all chunks (but not the actual data, which is loaded on-demand)
};

// ---------------------------------------------------------------------------
// factory function to create a PlainTextDeserializer from a V1 config object
// ---------------------------------------------------------------------------

shared_ptr<DataDeserializer> CreatePlainTextDeserializer(const ConfigParameters& configParameters, bool primary)
{
    // convert V1 configParameters back to a C++ data structure
    const ConfigParameters& input = configParameters(L"input");
    if (input.empty())
        InvalidArgument("PlainTextDeserializer configuration contains an empty \"input\" section.");

    string precision = configParameters.Find("precision", "float");
    DataType elementType;
    if (precision == "double")
        elementType = DataType::Double;
    else if (precision == "float")
        elementType = DataType::Float;
    else
        RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());

    // enumerate all streams
    vector<PlainTextDeserializerImpl::PlainTextStreamConfiguration> streamConfigs; streamConfigs.reserve(input.size());
    for (const pair<string, ConfigParameters>& section : input)
    {
        ConfigParameters streamConfigParameters = section.second;
        // basic stream info
        StreamInformation streamInfo;
        streamInfo.m_name          = utf16(section.first);
        streamInfo.m_id            = streamConfigs.size(); // id = index of stream config in stream-config array
        streamInfo.m_storageFormat = StorageFormat::SparseCSC;
        streamInfo.m_elementType   = elementType;
        streamInfo.m_sampleLayout  = NDShape{ streamConfigParameters(L"dim") };
        streamInfo.m_definesMbSize = streamConfigParameters(L"definesMBSize", false);
        // list of files. Wildcard expansion is happening here.
        wstring fileNamesArg = streamConfigParameters(L"dataFiles");
#if 1
        // BUGBUG: We pass these pathnames currently as a single string, since passing a vector<wstring> through the V1 Configs gets tripped up by * and :
        fileNamesArg.erase(0, 2); fileNamesArg.pop_back(); fileNamesArg.pop_back(); // strip (\n and )\n
        let fileNames = split(fileNamesArg, L"\n");
#else
        vector<wstring> fileNames = streamConfigParameters(L"dataFiles", ConfigParameters::Array(Microsoft::MSR::CNTK::stringargvector()));
#endif
        vector<wstring> expandedFileNames;
        for (let& fileName : fileNames)
            expand_wildcards(fileName, expandedFileNames);
        // vocabularyConfig
        streamConfigs.emplace_back(PlainTextDeserializerImpl::PlainTextStreamConfiguration
        (
            move(streamInfo),
            move(expandedFileNames),
            {
                streamConfigParameters(L"vocabularyFile"),
                streamConfigParameters(L"insertAtStart", L""),
                streamConfigParameters(L"insertAtEnd", L""),
                streamConfigParameters(L"substituteForUnknown", L"")
            }
        ));
    }

    bool cacheIndex = configParameters(L"cacheIndex");
    let traceLevel = configParameters(L"traceLevel", 1);
    let chunkSizeBytes = 10000; // configParameters(L"chunkSizeInBytes", g_32MB); // 32 MB by default   --TODO: currently not passed in from owningDeserializer factory function

    // construct the deserializer from that
    return make_shared<PlainTextDeserializerImpl>(move(streamConfigs), chunkSizeBytes, cacheIndex, elementType, primary, traceLevel);
}

}; // namespace
