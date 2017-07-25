//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ImageDeserializerBase.h"
#include "Config.h"
#include "CorpusDescriptor.h"

namespace CNTK {

    // Base 64 Image deserializer.
    class Base64ImageDeserializerImpl : public ImageDeserializerBase
    {
    public:
        Base64ImageDeserializerImpl(CorpusDescriptorPtr corpus, const Microsoft::MSR::CNTK::ConfigParameters& config, bool primary);

        // Get a chunk by id.
        ChunkPtr GetChunk(ChunkIdType chunkId) override;

        // Get chunk descriptions.
        std::vector<ChunkInfo> ChunkInfos() override;

        // Gets sequence descriptions for the chunk.
        void SequenceInfosForChunk(ChunkIdType, std::vector<SequenceInfo>&) override;

        // Gets sequence description by key.
        bool GetSequenceInfoByKey(const SequenceKey&, SequenceInfo&) override;

    private:
        class ImageChunk;

        std::shared_ptr<Index> m_index;
        std::shared_ptr<FILE> m_dataFile;
        std::wstring m_fileName;
        bool m_hasSequenceIds;
    };

}
