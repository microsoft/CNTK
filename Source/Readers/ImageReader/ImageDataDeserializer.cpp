//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class ImageDataDeserializer::LabelGenerator
{
public:
    virtual void CreateLabelFor(size_t classId, SparseSequenceData& data) = 0;
    virtual ~LabelGenerator() { }
};

// A helper class to generate a typed label in a sparse format.
// A label is just a category/class the image belongs to.
// It is represented as a array indexed by the category with zero values for all categories the image does not belong to, 
// and a single one for a category it belongs to: [ 0, .. 0.. 1 .. 0 ]
// The class is parameterized because the representation of 1 is type specific.
template <class TElement>
class TypedLabelGenerator : public ImageDataDeserializer::LabelGenerator
{
public:
    TypedLabelGenerator() : m_value(1)
    {
    }

    virtual void CreateLabelFor(size_t classId, SparseSequenceData& data) override
    {
        data.m_indices.resize(1);
        data.m_indices[0] = std::vector<size_t>{ classId };
        data.m_data = &m_value;
    }

private:
    TElement m_value;
};

// Used to keep track of the image. Accessed only using DenseSequenceData interface.
struct DeserializedImage : DenseSequenceData
{
    cv::Mat m_image;
};

// For image, chunks correspond to a single image.
class ImageDataDeserializer::ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk>
{
    ImageSequenceDescription m_description;
    ImageDataDeserializer& m_parent;

public:
    ImageChunk(ImageSequenceDescription& description, ImageDataDeserializer& parent)
        : m_description(description), m_parent(parent)
    {
    }

    virtual std::vector<SequenceDataPtr> GetSequence(const size_t& sequenceId) override
    {
        assert(sequenceId == m_description.m_id);
        UNUSED(sequenceId);
        const auto& imageSequence = m_description;

        auto image = std::make_shared<DeserializedImage>();
        image->m_image = std::move(ReadImage(m_description.m_id, imageSequence.m_path));
        auto& cvImage = image->m_image;

        if (!cvImage.data)
        {
            RuntimeError("Cannot open file '%s'", imageSequence.m_path.c_str());
        }

        // Convert element type.
        int dataType = m_parent.m_featureElementType == ElementType::tfloat ? CV_32F : CV_64F;
        if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
        {
            cvImage.convertTo(cvImage, dataType);
        }

        if (!cvImage.isContinuous())
        {
            cvImage = cvImage.clone();
        }
        assert(cvImage.isContinuous());

        image->m_data = image->m_image.data;
        ImageDimensions dimensions(cvImage.cols, cvImage.rows, cvImage.channels());
        image->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
        image->m_numberOfSamples = 1;
        image->m_chunk = shared_from_this();

        SparseSequenceDataPtr label = std::make_shared<SparseSequenceData>();
        label->m_chunk = shared_from_this();
        m_parent.m_labelGenerator->CreateLabelFor(imageSequence.m_classId, *label);
        return std::vector<SequenceDataPtr> { image, label };
    }
};

ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
{
    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);
    const auto& label = m_streams[configHelper.GetLabelStreamId()];
    const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

    // Expect data in HWC.
    ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
    feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));

    label->m_storageType = StorageType::sparse_csc;
    feature->m_storageType = StorageType::dense;

    m_featureElementType = feature->m_elementType;
    size_t labelDimension = label->m_sampleLayout->GetDim(0);

    if (label->m_elementType == ElementType::tfloat)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<float>>();
    }
    else if (label->m_elementType == ElementType::tdouble)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<double>>();
    }
    else
    {
        RuntimeError("Unsupported label element type '%d'.", (int)label->m_elementType);
    }

    CreateSequenceDescriptions(configHelper.GetMapPath(), labelDimension);
}

void ImageDataDeserializer::CreateSequenceDescriptions(std::string mapPath, size_t labelDimension)
{
    UNUSED(labelDimension);

    std::ifstream mapFile(mapPath);
    if (!mapFile)
    {
        RuntimeError("Could not open %s for reading.", mapPath.c_str());
    }

    PathReaderMap knownReaders;

    std::string line;
    ImageSequenceDescription description;
    description.m_numberOfSamples = 1;
    description.m_isValid = true;
    for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
    {
        std::stringstream ss(line);
        std::string imagePath;
        std::string classId;
        if (!std::getline(ss, imagePath, '\t') || !std::getline(ss, classId, '\t'))
        {
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.",
                         mapPath.c_str(),
                         static_cast<int>(lineIndex));
        }

        description.m_id = lineIndex;
        description.m_chunkId = lineIndex;
        description.m_path = imagePath;
        description.m_classId = std::stoi(classId);

        if (description.m_classId >= labelDimension)
        {
            RuntimeError(
                "Image '%s' has invalid class id '%d'. Expected label dimension is '%d'.",
                mapPath.c_str(),
                static_cast<int>(description.m_classId),
                static_cast<int>(labelDimension));
        }
        m_imageSequences.push_back(description);
        RegisterByteReader(description.m_id, description.m_path, knownReaders);
    }
}

std::vector<StreamDescriptionPtr> ImageDataDeserializer::GetStreamDescriptions() const
{
    return m_streams;
}

void ImageDataDeserializer::FillSequenceDescriptions(SequenceDescriptions& timeline) const
{
    timeline.resize(m_imageSequences.size());
    std::transform(
        m_imageSequences.begin(),
        m_imageSequences.end(),
        timeline.begin(),
        [](const ImageSequenceDescription& desc)
        {
            return &desc;
        });
}

ChunkPtr ImageDataDeserializer::GetChunk(size_t chunkId)
{
    auto sequenceDescription = m_imageSequences[chunkId];
    return std::make_shared<ImageChunk>(sequenceDescription, *this);
}

void ImageDataDeserializer::RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders)
{
    assert(!path.empty());

    auto atPos = path.find_first_of('@');
    // Is it container or plain image file?
    if (atPos == std::string::npos)
        return;
    assert(atPos > 0);
    assert(atPos + 1 < path.length());
    // REVIEW alexeyk: only .zip container support for now.
    auto containerPath = path.substr(0, atPos);
    // skip @ symbol and path separator (/ or \)
    auto itemPath = path.substr(atPos + 2);
    // zlib only supports / as path separator.
    std::replace(begin(itemPath), end(itemPath), '\\', '/');
    std::shared_ptr<ByteReader> reader;
    auto r = knownReaders.find(containerPath);
    if (r == knownReaders.end())
    {
        reader = std::make_shared<ZipByteReader>(containerPath);
        knownReaders[containerPath] = reader;
    }
    else
    {
        reader = (*r).second;
    }
    reader->Register(seqId, itemPath);
    m_readers[seqId] = reader;
}

cv::Mat ImageDataDeserializer::ReadImage(size_t seqId, const std::string& path)
{
    assert(!path.empty());

    ImageDataDeserializer::SeqReaderMap::const_iterator r;
    if (m_readers.empty() || (r = m_readers.find(seqId)) == m_readers.end())
        return m_defaultReader.Read(seqId, path);
    return (*r).second->Read(seqId, path);
}

cv::Mat FileByteReader::Read(size_t, const std::string& path)
{
    assert(!path.empty());

    return cv::imread(path, cv::IMREAD_COLOR);
}
}}}
