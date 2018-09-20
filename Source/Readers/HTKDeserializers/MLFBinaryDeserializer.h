//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <boost/noncopyable.hpp>
#include "MLFDeserializer.h"
#include "Index.h"

namespace CNTK {

    // Class represents an MLF deserializer.
    // Provides a set of chunks/sequences to the upper layers.
    class MLFBinaryDeserializer : public MLFDeserializer
    {
    public:
        // Expects new configuration.
        MLFBinaryDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary);

        // TODO: Should be removed, when all readers go away, expects configuration in a legacy mode.
        MLFBinaryDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, const std::wstring& streamName);

        // Retrieves a chunk with data.
        virtual ChunkPtr GetChunk(ChunkIdType) override;

    private:
        class BinarySequenceChunk;
    };
}
