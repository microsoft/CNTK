//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "ImageDeserializerBase.h"
#include "Config.h"
#include "ByteReader.h"
#include <unordered_map>
#include "CorpusDescriptor.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Image data deserializer based on the OpenCV library.
// The deserializer currently supports two output streams only: a feature and a label stream.
// All sequences consist only of a single sample (image/label).
// For features it uses dense storage format with different layout (dimensions) per sequence.
// For labels it uses the csc sparse storage format.
class ImageDataDeserializer : public ImageDeserializerBase
{
public:
    // A new constructor to support new compositional configuration,
    // that allows composition of deserializers and transforms on inputs.
    ImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config);

    // TODO: This constructor should be deprecated in the future. Compositional config should be used instead.
    explicit ImageDataDeserializer(const ConfigParameters& config);

    // Gets sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override;

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Gets sequence descriptions for the chunk.
    virtual void GetSequencesForChunk(ChunkIdType, std::vector<SequenceDescription>&) override;

    // Gets sequence description by key.
    bool GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override;

private:
    // Creates a set of sequence descriptions.
    void CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath, size_t labelDimension, bool isMultiCrop);

    // Image sequence descriptions. Currently, a sequence contains a single sample only.
    struct ImageSequenceDescription : public SequenceDescription
    {
        std::string m_path;
        size_t m_classId;
    };

    class ImageChunk;

    // Sequence descriptions for all input data.
    std::vector<ImageSequenceDescription> m_imageSequences;

    // Not using nocase_compare here as it's not correct on Linux.
    using PathReaderMap = std::unordered_map<std::string, std::shared_ptr<ByteReader>>;
    using ReaderSequenceMap = std::map<std::string, std::map<std::string, std::vector<size_t>>>;
    void RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders, ReaderSequenceMap& readerSequences, const std::string& expandDirectory);
    cv::Mat ReadImage(size_t seqId, const std::string& path, bool grayscale);

    // REVIEW alexeyk: can potentially use vector instead of map. Need to handle default reader and resizing though.
    using SeqReaderMap = std::unordered_map<size_t, std::shared_ptr<ByteReader>>;
    SeqReaderMap m_readers;

    std::unique_ptr<FileByteReader> m_defaultReader;
};

}}}
