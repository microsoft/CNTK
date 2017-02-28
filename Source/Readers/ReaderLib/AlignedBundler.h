//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "DataDeserializer.h"
#include "DataDeserializerBase.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents an bundler of several deserializers.
// It assumes the inputs between different deserializers are aligned, even though IO chunks can be different.
//    Deserializer1              Deserializer2
//    C1  |-s1-                  C1 | -s1-
//        |-s2-                     | -s2-
//        |-s3-                     | -s3-
//        |-s4-                     | -s4-
//                                  |
//    C2  |-s5-                     | -s5-
//        |-s6-                     | -s6-
//        |
//        |-s7-                  C2 | -s7-
//        |-s8-                     | -s8-
//        |-s9-                     | -s9-
class AlignedBundler : public DataDeserializerBase
{
public:
    AlignedBundler(const ConfigParameters& readerConfig, IDataDeserializerPtr driver, std::vector<IDataDeserializerPtr> deserializers);

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Gets sequence descriptions for a particular chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result) override;

    // Gets a chunk with data.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

private:
    DISABLE_COPY_AND_MOVE(AlignedBundler);

    class BundlingChunk;
    struct BundlerChunkDescription;
    typedef std::shared_ptr<BundlerChunkDescription> BundlerChunkDescriptionPtr;

    // Creates chunk descriptions based on chunks of underlying deserializers.
    void CreateChunkDescriptions();

    // Underlying deserializers.
    std::vector<IDataDeserializerPtr> m_deserializers;

    // Driving deserializer that defines chunks.
    IDataDeserializerPtr m_primaryDeserializer;

    // Chunk descriptions.
    std::vector<BundlerChunkDescriptionPtr> m_chunks;

    // If flag is set to true the sequence length is counted by the primary deserializer only.
    // Used for optimization when sequences between different deserializers are of the same length
    // (i.e. often in speech)
    bool m_takePrimarySequenceLength;

    // A table of loaded chunks to make sure we do not load same chunk twice.
    // Inner vector is the table of chunk id into weak pointer, the outer vector has an element per deserializer.
    std::vector<std::vector<std::weak_ptr<Chunk>>> m_weakChunkTable;

    // Index of chunks.
    struct ChunkIndex
    {
        size_t chunkIndex;
        size_t sequenceIndex;
    };
    std::vector<std::vector<std::pair<ChunkIndex, ChunkIndex>>> m_index;

    // General configuration
    int m_verbosity;
};

}}}
