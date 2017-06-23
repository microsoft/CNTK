//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ImageDeserializerBase.h"
#include "Config.h"
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Base 64 Image deserializer.
    class Base64ImageDeserializer : public ImageDeserializerBase
    {
    public:
        Base64ImageDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

        // Get a chunk by id.
        ChunkPtr GetChunk(ChunkIdType chunkId) override;

        // Get chunk descriptions.
        ChunkDescriptions GetChunkDescriptions() override;

        // Gets sequence descriptions for the chunk.
        void GetSequencesForChunk(ChunkIdType, std::vector<SequenceDescription>&) override;

        // Gets sequence description by key.
        bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override;

    private:
        // Creates a set of sequence descriptions.
        void CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath);

        class ImageChunk;

        std::shared_ptr<Index> m_index;
        std::shared_ptr<FILE> m_dataFile;
        std::wstring m_fileName;
        bool m_hasSequenceIds;
    };

}}}
