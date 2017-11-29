//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "LatticeDeserializer.h"
#include "LatticeIndexBuilder.h"
#include "ConfigHelper.h"
#include "Basics.h"
#include "StringUtil.h"
#include <unordered_set>

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

using namespace std;

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
    wstring phoneFile = cfg(L"phoneFile");
    wstring transpFile = cfg(L"transpFile");
    wstring labelMappingFile = cfg(L"labelMappingFile");

    m_hset.loadfromfile(phoneFile, labelMappingFile, transpFile);
    const std::unordered_map<std::string, size_t>& modelsymmap = m_hset.getsymmap();

    InitializeStreams(inputName);
    InitializeChunkInfos(corpus, config);
}

// Initializes chunks based on the configuration and utterance descriptions.
void LatticeDeserializer::InitializeChunkInfos(CorpusDescriptorPtr corpus, ConfigHelper& config)
{
    std::string symListPath = config.GetSymListFilePath();
    std::string latticeIndexPath = config.GetLatticeIndexFilePath();
    
    fprintf(stderr, "Reading lattice index file %s ...", latticeIndexPath.c_str());

    wifstream latticeIndexStream(latticeIndexPath.c_str());
    if (!latticeIndexStream)
        RuntimeError("Failed to open input file: %s", latticeIndexPath.c_str());

    bool enableCaching = corpus->IsHashingEnabled() && config.GetCacheIndex();

    std::wstring path;
    while (!latticeIndexStream.eof())
    {
        std::getline(latticeIndexStream, path);

        attempt(5, [this, path, enableCaching, corpus]()
        {
            LatticeIndexBuilder builder(FileWrapper(path, L"rbS"), corpus);
            builder.SetChunkSize(m_chunkSizeBytes).SetCachingEnabled(enableCaching);
            m_indices.emplace_back(builder.Build());
        });

        m_latticeFiles.push_back(path);

    }
    latticeIndexStream.close();


    deque<UtteranceDescription> utterances;
    size_t totalNumberOfBytes = 0;
    size_t totalNumSequences = 0;
    
    std::unordered_map<size_t, std::vector<string>> duplicates;
    std::unordered_set<size_t> uniqueIds;
    while (getline(latticeIndexStream, path))
    {
        // Read lattice file
        //open(tocpaths[i])

        

        ifstream latticeFileStream(latticeFile.c_str(), ios::binary | ios::ate);
        if (!latticeFileStream)
            RuntimeError("Failed to open input lattice file: %s", latticeFile.c_str());

        string line, key;
        while (getline(latticeFileStream, line))
        {
            key.clear();
            
            //TODO: check how this method works for consequtive lines
            UtteranceDescription description(htkfeatreader::parsedpath::Parse(line, key));

            size_t id = m_corpus->KeyToId(key);
            description.SetId(id);
            if (uniqueIds.find(id) == uniqueIds.end())
            {
                utterances.push_back(std::move(description));
                uniqueIds.insert(id);
            }
            else
            {
                duplicates[id].push_back(key);
            }
        }

        if (latticeFileStream.bad())
            RuntimeError("An error occurred while reading input file: %s", line.c_str());

        totalNumberOfBytes += latticeIndexStream.tellg();
    }
   
    if (latticeIndexStream.bad())
        RuntimeError("An error occurred while reading input file: %s", latticeIndexPath.c_str());

    fprintf(stderr, " %zu entries\n", utterances.size());

    m_chunks.reserve(totalNumberOfBytes / chunkSizeBytes + 1);

    ChunkIdType chunkId = 0;
    foreach_index(i, utterances)
    {
        // Skip duplicates.
        if (duplicates.find(utterances[i].GetId()) != duplicates.end())
        {
            continue;
        }

        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().GetTotalFrames() > ChunkFrames)
        {
            m_chunks.push_back(HTKChunkInfo(chunkId++));
        }

        // append utterance to last chunk
        HTKChunkInfo& currentChunk = m_chunks.back();
        if (!m_primary)
        {
            // Have to store key <-> utterance mapping for non primary deserializers.
            m_keyToChunkLocation.push_back(std::make_tuple(utterances[i].GetId(), currentChunk.GetChunkId(), currentChunk.GetNumberOfUtterances()));
        }

        currentChunk.Add(move(utterances[i]));
    }

    std::sort(m_keyToChunkLocation.begin(), m_keyToChunkLocation.end(),
        [](const std::tuple<size_t, size_t, size_t>& a, const std::tuple<size_t, size_t, size_t>& b)
    {
        return std::get<0>(a) < std::get<0>(b);
    });

    // Report duplicates.
    size_t numberOfDuplicates = 0;
    for (const auto& u : duplicates)
    {
        if (m_verbosity)
        {
            fprintf(stderr, "ID '%zu':\n", u.first);
            for (const auto& k : u.second)
                fprintf(stderr, "Key '%s'\n", k.c_str());
        }

        numberOfDuplicates += (u.second.size() + 1);
    }

    if (numberOfDuplicates)
        fprintf(stderr, "WARNING: Number of duplicates is '%zu'. All duplicates will be dropped. Consider switching to numeric sequence ids.\n", numberOfDuplicates);

    fprintf(stderr,
        "HTKDeserializer: selected '%zu' utterances grouped into '%zu' chunks, "
        "average chunk size: %.1f utterances, %.1f frames "
        "(for I/O: %.1f utterances, %.1f frames)\n",
        utterances.size(),
        m_chunks.size(),
        utterances.size() / (double)m_chunks.size(),
        totalNumberOfFrames / (double)m_chunks.size(),
        utterances.size() / (double)m_chunks.size(),
        totalNumberOfFrames / (double)m_chunks.size());

    if (utterances.empty())
    {
        RuntimeError("HTKDeserializer: No utterances to process.");
    }
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
void LatticeDeserializer::SequenceInfosForChunk(ChunkIdType chunkId, vector<SequenceInfo>& result)
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
        auto& fileName = m_mlfFiles[m_chunkToFileIndex[chunk]];

        result = make_shared<SequenceChunk>(*this, *chunk, fileName, m_stateTable);
    });

    return result;
};

}
