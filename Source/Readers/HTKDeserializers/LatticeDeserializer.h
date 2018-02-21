//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializerBase.h"
#include "Config.h"
#include "CorpusDescriptor.h"
#include "ConfigHelper.h"
#include "Index.h"
#include <boost/noncopyable.hpp>

namespace CNTK {

// Class represents a speech lattice.
// Provides a set of chunks/sequences to the upper layers.
class LatticeDeserializer : public DataDeserializerBase, private boost::noncopyable
{
public:
    // Expects new configuration.
    LatticeDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

    // Retrieves sequence description by its key. Used for deserializers that are not in "primary"/"driving" mode.
    bool GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& s) override;

    // Get information about chunks.
    virtual std::vector<ChunkInfo> ChunkInfos() override;

    // Get information about particular chunk.
    virtual void SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result) override;

    // Retrieves data for a chunk.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

private:
    class LatticeChunk;
    class ChunkBase;
    class SequenceChunk;

    // Initialization functions.
    void InitializeChunkInfos(CorpusDescriptorPtr corpus, ConfigHelper& config);
    void InitializeStreams(const std::wstring& featureName);
    size_t RecordChunk(const string& latticePath, const vector<string>& tocLines, CorpusDescriptorPtr corpus, bool enableCaching, bool lastChunkInTOC);

    CorpusDescriptorPtr m_corpus;

    // General configuration
    int m_verbosity;

    // Used to correlate a sequence key with the sequence inside the chunk when deserializer is running not in primary mode.
    // <key, chunkid, offset inside chunk>, sorted by key to be able to retrieve by binary search.
    std::vector<std::tuple<size_t, ChunkIdType, uint32_t>> m_keyToChunkLocation;

    std::vector<const ChunkDescriptor*> m_chunks;
    std::map<const ChunkDescriptor*, size_t> m_chunkToFileIndex;
    size_t m_chunkSizeBytes;
    std::vector<std::shared_ptr<Index>> m_indices;
    std::vector<std::wstring> m_latticeFiles;
};

}
