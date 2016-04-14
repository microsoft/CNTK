//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "DataDeserializerBase.h"
#include "Config.h"
#include "ByteReader.h"
#include "ImageConfigHelper.h"
#include <unordered_map>

namespace Microsoft { namespace MSR { namespace CNTK {

// Image data deserializer based on the OpenCV library.
// The deserializer currently supports two output streams only: a feature and a label stream.
// All sequences consist only of a single sample (image/label).
// For features it uses dense storage format with different layout (dimensions) per sequence.
// For labels it uses the csc sparse storage format.
class ImageDataDeserializer : public DataDeserializerBase
{
public:
    explicit ImageDataDeserializer(const ConfigParameters& config);

    // Gets sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    virtual ChunkPtr GetChunk(size_t chunkId) override;

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override;

    // Gets sequence descriptions for the chunk.
    virtual void GetSequencesForChunk(size_t, std::vector<SequenceDescription>&) override;

private:
    // Creates a set of sequence descriptions.
    void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension, const ImageConfigHelper& config);

    // Image sequence descriptions. Currently, a sequence contains a single sample only.
    struct ImageSequenceDescription : public SequenceDescription
    {
        std::string m_path;
        size_t m_classId;
    };

    class ImageChunk;

    // A helper class for generation of type specific labels (currently float/double only).
    class LabelGenerator;
    typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;
    LabelGeneratorPtr m_labelGenerator;

    // Sequence descriptions for all input data.
    std::vector<ImageSequenceDescription> m_imageSequences;

    // Element type of the feature/label stream (currently float/double only).
    ElementType m_featureElementType;

    // Not using nocase_compare here as it's not correct on Linux.
    using PathReaderMap = std::unordered_map<std::string, std::shared_ptr<ByteReader>>;
    void RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders);
    cv::Mat ReadImage(size_t seqId, const std::string& path);

    // REVIEW alexeyk: can potentially use vector instead of map. Need to handle default reader and resizing though.
    using SeqReaderMap = std::unordered_map<size_t, std::shared_ptr<ByteReader>>;
    SeqReaderMap m_readers;

    FileByteReader m_defaultReader;
};

}}}
