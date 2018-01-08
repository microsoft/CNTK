//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "LatticeDeserializer.h"
#include "LatticeIndexBuilder.h"
#include "ConfigHelper.h"
#include "Basics.h"
#include "MLFUtils.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

using namespace std;

// This class stores sequence data for HTK for floats.
struct LatticeFloatSequenceData : DenseSequenceData
{
    LatticeFloatSequenceData(void* data, unsigned int bufferSize, const NDShape& frameShape) :DenseSequenceData(bufferSize,true),
         m_frameShape(frameShape)
    {
        //Ensure the sequence owns the data, since the chunk can be released before the sequence is released
        size_t byteBufferSize = bufferSize * sizeof(float);
        m_buffer = new char[byteBufferSize];
        memcpy(m_buffer, data, byteBufferSize);
    }

    ~LatticeFloatSequenceData() 
    {
        delete[] m_buffer;
    }
    const void* GetDataBuffer() override
    {
        return m_buffer;
    }

    const NDShape& GetSampleShape() override
    {
        return m_frameShape;
    }

private:
    const NDShape& m_frameShape;
    char* m_buffer;
};

// Base class for chunks in frame and sequence mode.
// The lifetime is always less than the lifetime of the parent deserializer.
class LatticeDeserializer::ChunkBase : public Chunk
{
protected:
    vector<char> m_buffer;   // Buffer for the whole chunk
    vector<bool> m_valid;    // Bit mask whether the parsed sequence is valid.

    const LatticeDeserializer& m_deserializer;
    const ChunkDescriptor& m_descriptor;     // Current chunk descriptor.

    ChunkBase(const LatticeDeserializer& deserializer, const ChunkDescriptor& descriptor, const wstring& fileName):
        m_descriptor(descriptor),
        m_deserializer(deserializer)
    {
        if (descriptor.NumberOfSequences() == 0 || descriptor.SizeInBytes() == 0)
            LogicError("Empty chunks are not supported.");

        auto f = FileWrapper::OpenOrDie(fileName, L"rbS");
        size_t sizeInBytes = descriptor.SizeInBytes();

        // Make sure we always have 3 at the end for buffer overrun, i.e. 4 byte alignment
        m_buffer.resize(sizeInBytes + sizeof(float) - 1);
        for (int fl = 0; fl < sizeof(float)-1; fl++) {
            m_buffer[sizeInBytes + fl] = 0;
        }

        // Seek and read chunk into memory.
        f.SeekOrDie(descriptor.StartOffset(), SEEK_SET);

        f.ReadOrDie(m_buffer.data(), sizeInBytes, 1);

        // all sequences are valid by default.
        m_valid.resize(m_descriptor.NumberOfSequences(), true);
    }

    string KeyOf(const SequenceDescriptor& s)
    {
        return m_deserializer.m_corpus->IdToKey(s.m_key);
    }
};

// MLF chunk when operating in sequence mode.
class LatticeDeserializer::SequenceChunk : public LatticeDeserializer::ChunkBase
{

public:
    SequenceChunk(const LatticeDeserializer& parent, const ChunkDescriptor& descriptor, const wstring& fileName)
        : ChunkBase(parent, descriptor, fileName), m_ndShape({ 1 })
    {
    }

    void GetSequence(size_t sequenceIndex, vector<SequenceDataPtr>& result) override
    {
        return GetSequence<float>(sequenceIndex, result);
    }

    template<class ElementType>
    void GetSequence(size_t sequenceIndex, vector<SequenceDataPtr>& result)
    {
        const auto& sequence = m_descriptor.Sequences()[sequenceIndex];
        
        // Deserialize the binary lattice graph and serialize it into a vector
        SequenceDataPtr s = make_shared<LatticeFloatSequenceData>(m_buffer.data() + sequence.OffsetInChunk(), sequence.NumberOfSamples(), m_ndShape);

        result.push_back(s);
    }

private:
    const NDShape m_ndShape; 
};

LatticeDeserializer::LatticeDeserializer(
    CorpusDescriptorPtr corpus,
    const ConfigParameters& cfg,
    bool primary)
    : DataDeserializerBase(primary),
      m_verbosity(0),
      m_corpus(corpus)
{
    if (primary)
        LogicError("Lattice deserializer does not support primary mode, it cannot control chunking. "
            "Please specify HTK deserializer as the first deserializer in your config file.");

    m_verbosity = cfg(L"verbosity", 0);
    m_chunkSizeBytes = cfg(L"chunkSizeInBytes", g_64MB);

    ConfigParameters input = cfg(L"input");
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);

    ConfigHelper config(streamConfig);
    InitializeStreams(inputName);
    InitializeChunkInfos(corpus, config);
}

size_t LatticeDeserializer::RecordChunk(const string& latticePath, const vector<string>& tocLines, CorpusDescriptorPtr corpus, bool enableCaching)
{
    size_t totalNumSequences = 0;
    wstring latticePathW;
    latticePathW.assign(latticePath.begin(), latticePath.end());
    attempt(5, [this, latticePathW, tocLines, enableCaching, corpus]()
    {
        LatticeIndexBuilder builder(FileWrapper(latticePathW, L"rbS"), tocLines, corpus);
        builder.SetChunkSize(m_chunkSizeBytes).SetCachingEnabled(enableCaching);
        m_indices.emplace_back(builder.Build());
    });

    m_latticeFiles.push_back(latticePathW);

    auto& index = m_indices.back();
    // Build auxiliary for GetSequenceByKey.
    for (const auto& chunk : index->Chunks())
    {
        // Preparing chunk info that will be exposed to the outside.
        auto chunkId = static_cast<ChunkIdType>(m_chunks.size());
        for (uint32_t i = 0; i < chunk.NumberOfSequences(); ++i)
        {
            const auto& sequence = chunk[i];
            auto sequenceIndex = i;
            m_keyToChunkLocation.push_back(std::make_tuple(sequence.m_key, chunkId, sequenceIndex));
        }

        totalNumSequences += chunk.NumberOfSequences();
        m_chunkToFileIndex.insert(make_pair(&chunk, m_latticeFiles.size() - 1));
        m_chunks.push_back(&chunk);
        if (m_chunks.size() >= numeric_limits<ChunkIdType>::max())
            RuntimeError("Number of chunks exceeded overflow limit.");
    }

    return totalNumSequences;
}

static inline bool LessByFirstItem(const std::tuple<size_t, size_t, size_t>& a, const std::tuple<size_t, size_t, size_t>& b)
{
    return std::get<0>(a) < std::get<0>(b);
}

// Initializes chunks based on the configuration and utterance descriptions.
void LatticeDeserializer::InitializeChunkInfos(CorpusDescriptorPtr corpus, ConfigHelper& config)
{
    std::string latticeIndexPath = config.GetLatticeIndexFilePath();

    fprintf(stderr, "Reading lattice index file '%s' ...\n", latticeIndexPath.c_str());
    ifstream latticeIndexStream(latticeIndexPath.c_str());
    if (!(latticeIndexStream && latticeIndexStream.good()))
        RuntimeError("Failed to open input file: '%s'", latticeIndexPath.c_str());

    bool enableCaching = corpus->IsHashingEnabled() && config.GetCacheIndex();
    size_t totalNumSequences = 0;
    vector<string> tocLines;
    string tocPath;
    while (getline(latticeIndexStream, tocPath))
    {
        tocPath.erase(tocPath.find_last_not_of(" \n\r\t") + 1);
        std::ifstream tocFileStream(tocPath);
        if (!(tocFileStream && tocFileStream.good())) 
            fprintf(stderr, "Failed to open input file: %s", tocPath.c_str());

        std::string tocLine;
        tocLines.clear();
        bool firstIndex = true;
        string prevLatticePath;
        while (std::getline(tocFileStream, tocLine))
        {
            size_t start = tocLine.find("=") + 1;
            size_t end = tocLine.find("[");
            string latticePath = tocLine.substr(start, end - start);
            if (latticePath.size() > 0) {
                if (firstIndex)
                    firstIndex = false;
                else {
                    totalNumSequences += RecordChunk(prevLatticePath, tocLines, corpus, enableCaching);
                    tocLines.clear();
                }

                prevLatticePath = latticePath;
            }
                
            tocLines.push_back(tocLine);
        }
        totalNumSequences += RecordChunk(prevLatticePath, tocLines, corpus, enableCaching);
    }
    latticeIndexStream.close();

    std::sort(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), LessByFirstItem);

    fprintf(stderr, "LatticeDeserializer: '%zu' sequences\n", totalNumSequences);
}

// Describes exposed stream - a single stream of htk features.
void LatticeDeserializer::InitializeStreams(const wstring& featureName)
{
    StreamInformation stream;
    stream.m_id = 0;
    stream.m_name = featureName;
    stream.m_sampleLayout = NDShape({ 1 });
    stream.m_storageFormat = StorageFormat::Dense;
    stream.m_elementType = DataType::Float;
    stream.m_isBinary = true;
    m_streams.push_back(stream);
}

// Gets information about available chunks.
std::vector<ChunkInfo> LatticeDeserializer::ChunkInfos()
{
    std::vector<ChunkInfo> chunks;
    chunks.reserve(m_chunks.size());
    for (size_t i = 0; i < m_chunks.size(); ++i)
    {
        ChunkInfo cd;
        cd.m_id = static_cast<ChunkIdType>(i);
        if (cd.m_id != i)
            RuntimeError("ChunkIdType overflow during creation of a chunk description.");

        cd.m_numberOfSequences = m_chunks[i]->NumberOfSequences();
        cd.m_numberOfSamples = m_chunks[i]->NumberOfSamples();
        chunks.push_back(cd);
    }
    return chunks;
}

// Gets sequences for a particular chunk.
// This information is used by the randomizer to fill in current windows of sequences.
void LatticeDeserializer::SequenceInfosForChunk(ChunkIdType, vector<SequenceInfo>& result)
{
    UNUSED(result);
    LogicError("Lattice deserializer does not support primary mode, it cannot control chunking. "
        "Please specify HTK deserializer as the first deserializer in your config file.");

}

ChunkPtr LatticeDeserializer::GetChunk(ChunkIdType chunkId)
{
    ChunkPtr result;
    attempt(5, [this, &result, chunkId]()
    {
        auto chunk = m_chunks[chunkId];
        auto& fileName = m_latticeFiles[m_chunkToFileIndex[chunk]];

        result = make_shared<SequenceChunk>(*this, *chunk, fileName);
    });

    return result;
};

bool LatticeDeserializer::GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& result)
{
    auto found = std::lower_bound(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(), std::make_tuple(key.m_sequence, 0, 0),
        LessByFirstItem);

    if (found == m_keyToChunkLocation.end() || std::get<0>(*found) != key.m_sequence)
    {
        return false;
    }

    auto chunkId = std::get<1>(*found);
    auto sequenceIndexInChunk = std::get<2>(*found);

    result.m_chunkId = std::get<1>(*found);
    result.m_key = key;

    assert(result.m_key.m_sample == 0);

    const auto* chunk = m_chunks[chunkId];
    const auto& sequence = chunk->Sequences()[sequenceIndexInChunk];
    result.m_indexInChunk = sequenceIndexInChunk;
    result.m_numberOfSamples = sequence.m_numberOfSamples;

    return true;
}

}
