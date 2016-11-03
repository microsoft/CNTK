//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"
#include "StringUtil.h"
#include "ConfigUtil.h"
#include "TimerUtility.h"
#include "ImageTransformers.h"
#include "ImageUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// For image, chunks correspond to a single image.
class ImageDataDeserializer::ImageChunk : public Chunk
{
    ImageSequenceDescription m_description;
    ImageDataDeserializer& m_deserializer;

public:
    ImageChunk(ImageSequenceDescription& description, ImageDataDeserializer& parent)
        : m_description(description), m_deserializer(parent)
    {
    }

    virtual void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
    {
        assert(sequenceIndex == 0 && sequenceIndex == m_description.m_indexInChunk);
        UNUSED(sequenceIndex);

        auto cvImage = m_deserializer.ReadImage(m_description.m_key.m_sequence, m_description.m_path, m_deserializer.m_grayscale);
        if (!cvImage.data)
            RuntimeError("Cannot open file '%s'", m_description.m_path.c_str());

        m_deserializer.PopulateSequenceData(cvImage, m_description.m_classId, m_description.m_copyId, m_description.m_key, result);
    }

private:
    ElementType ConvertImageToSupportedDataType(cv::Mat& image)
    {
        ElementType resultType;
        if (!IdentifyElementTypeFromOpenCVType(image.depth(), resultType))
        {
            // Could not identify element type.
            // Natively unsupported image type. Let's convert it to required precision.
            int requiredType = m_deserializer.m_precision == ElementType::tfloat ? CV_32F : CV_64F;
            image.convertTo(image, requiredType);
            resultType = m_deserializer.m_precision;
        }
        return resultType;
    }
};

// A new constructor to support new compositional configuration,
// that allows composition of deserializers and transforms on inputs.
ImageDataDeserializer::ImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary) : ImageDeserializerBase(corpus, config, primary)
{
    CreateSequenceDescriptions(corpus, config(L"file"), m_labelGenerator->LabelDimension(), m_multiViewCrop);
}

// TODO: Should be removed at some point.
// Supports old type of ImageReader configuration.
ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
{
    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);
    m_grayscale = configHelper.UseGrayscale();
    const auto& label = m_streams[configHelper.GetLabelStreamId()];
    const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

    m_verbosity = config(L"verbosity", 0);

    string precision = (ConfigValue)config("precision", "float");
    m_precision = AreEqualIgnoreCase(precision, "float") ? ElementType::tfloat : ElementType::tdouble;

    // Expect data in HWC.
    ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
    feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));

    label->m_storageType = StorageType::sparse_csc;
    feature->m_storageType = StorageType::dense;

    // Due to performance, now we support images of different types.
    feature->m_elementType = ElementType::tvariant;

    size_t labelDimension = label->m_sampleLayout->GetDim(0);

    if (label->m_elementType == ElementType::tfloat)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<float>>(labelDimension);
    }
    else if (label->m_elementType == ElementType::tdouble)
    {
        m_labelGenerator = std::make_shared<TypedLabelGenerator<double>>(labelDimension);
    }
    else
    {
        RuntimeError("Unsupported label element type '%d'.", (int)label->m_elementType);
    }

    CreateSequenceDescriptions(std::make_shared<CorpusDescriptor>(false), configHelper.GetMapPath(), labelDimension, configHelper.IsMultiViewCrop());
}

// Descriptions of chunks exposed by the image reader.
ChunkDescriptions ImageDataDeserializer::GetChunkDescriptions()
{
    ChunkDescriptions result;
    result.reserve(m_imageSequences.size());
    for (auto const& s : m_imageSequences)
    {
        auto chunk = std::make_shared<ChunkDescription>();
        chunk->m_id = s.m_chunkId;
        chunk->m_numberOfSamples = 1;
        chunk->m_numberOfSequences = 1;
        result.push_back(chunk);
    }

    return result;
}

void ImageDataDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
{
    // Currently a single sequence per chunk.
    result.push_back(m_imageSequences[chunkId]);
}

void ImageDataDeserializer::CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath, size_t labelDimension, bool isMultiCrop)
{
    std::ifstream mapFile(mapPath);
    if (!mapFile)
    {
        RuntimeError("Could not open %s for reading.", mapPath.c_str());
    }

    // Creating the default reader with expanded directory to the map file.
    auto mapFileDirectory = ExtractDirectory(mapPath);
    m_defaultReader = make_unique<FileByteReader>(mapFileDirectory);

    size_t numberOfCopies = isMultiCrop ? ImageDeserializerBase::NumMultiViewCopies : 1;
    static_assert(ImageDeserializerBase::NumMultiViewCopies < std::numeric_limits<uint8_t>::max(), "Do not support more than 256 copies.");

    size_t curId = 0;
    std::string line;
    PathReaderMap knownReaders;
    ReaderSequenceMap readerSequences;
    ImageSequenceDescription description;
    description.m_numberOfSamples = 1;

    Timer timer;
    timer.Start();

    for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
    {
        std::stringstream ss(line);
        std::string imagePath, classId, sequenceKey;
        // Try to parse sequence id, file path and label.
        if (!std::getline(ss, sequenceKey, '\t') || !std::getline(ss, imagePath, '\t') || !std::getline(ss, classId, '\t'))
        {
            // In case when the sequence key is not specified we set it to the line number inside the mapping file.
            // Assume that only image path and class label is given (old format).
            classId = imagePath;
            imagePath = sequenceKey;
            sequenceKey = std::to_string(lineIndex);

            if (classId.empty() || imagePath.empty())
                RuntimeError("Invalid map file format, must contain 2 or 3 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
        }

        // Skipping sequences that are not included in corpus.
        if (!corpus->IsIncluded(sequenceKey))
        {
            continue;
        }

        char* eptr;
        errno = 0;
        size_t cid = strtoull(classId.c_str(), &eptr, 10);
        if (classId.c_str() == eptr || errno == ERANGE)
            RuntimeError("Cannot parse label value on line %" PRIu64 ", second column, in file %s.", lineIndex, mapPath.c_str());

        if (cid >= labelDimension)
        {
            RuntimeError(
                "Image '%s' has invalid class id '%" PRIu64 "'. It is exceeding the label dimension of '%" PRIu64 "'. Line %" PRIu64 " in file %s.",
                imagePath.c_str(), cid, labelDimension, lineIndex, mapPath.c_str());
        }

        if (CHUNKID_MAX < curId + numberOfCopies)
        {
            RuntimeError("Maximum number of chunks exceeded.");
        }

        // Fill in original sequence.
        description.m_indexInChunk = 0;
        description.m_path = imagePath;
        description.m_classId = cid;
        description.m_key.m_sequence = corpus->KeyToId(sequenceKey);
        description.m_key.m_sample = 0;

        if (!m_primary)
        {
            m_keyToSequence[description.m_key.m_sequence] = m_imageSequences.size();
        }

        RegisterByteReader(description.m_key.m_sequence, description.m_path, knownReaders, readerSequences, mapFileDirectory);

        // Fill in copies.
        for (uint8_t index = 0; index < numberOfCopies; index++)
        {
            description.m_chunkId = (ChunkIdType)curId;
            description.m_copyId = index;

            m_imageSequences.push_back(description);
            curId++;
        }
    }

    for (auto& reader : knownReaders)
    {
        reader.second->Register(readerSequences[reader.first]);
    }

    timer.Stop();
    if (m_verbosity > 1)
    {
        fprintf(stderr, "ImageDeserializer: Read information about %d images in %.6g seconds\n", (int)m_imageSequences.size(), timer.ElapsedSeconds());
    }
}

ChunkPtr ImageDataDeserializer::GetChunk(ChunkIdType chunkId)
{
    auto sequenceDescription = m_imageSequences[chunkId];
    return std::make_shared<ImageChunk>(sequenceDescription, *this);
}

void ImageDataDeserializer::RegisterByteReader(size_t seqId, const std::string& seqPath, PathReaderMap& knownReaders, ReaderSequenceMap& readerSequences, const std::string& expandDirectory)
{
    assert(!seqPath.empty());

    auto path = Expand3Dots(seqPath, expandDirectory);

    auto atPos = path.find_first_of('@');
    // Is it container or plain image file?
    if (atPos == std::string::npos)
        return;
    // REVIEW alexeyk: only .zip container support for now.
#ifdef USE_ZIP
    assert(atPos > 0);
    assert(atPos + 1 < path.length());
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
        readerSequences[containerPath] = MultiMap();
    }
    else
    {
        reader = (*r).second;
    }

    readerSequences[containerPath][itemPath].push_back(seqId);
    m_readers[seqId] = reader;
#else
    UNUSED(seqId);
    UNUSED(knownReaders);
    UNUSED(readerSequences);
    RuntimeError("The code is built without zip container support. Only plain image files are supported.");
#endif
}

cv::Mat ImageDataDeserializer::ReadImage(size_t seqId, const std::string& path, bool grayscale)
{
    assert(!path.empty());

    ImageDataDeserializer::SeqReaderMap::const_iterator r;
    if (m_readers.empty() || (r = m_readers.find(seqId)) == m_readers.end())
        return m_defaultReader->Read(seqId, path, grayscale);
    return (*r).second->Read(seqId, path, grayscale);
}

cv::Mat FileByteReader::Read(size_t, const std::string& seqPath, bool grayscale)
{
    assert(!seqPath.empty());
    auto path = Expand3Dots(seqPath, m_expandDirectory);

    return cv::imread(path, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
}

bool ImageDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
{
    auto index = m_keyToSequence.find(key.m_sequence);
    // Checks whether it is a known sequence for us.
    if (key.m_sample != 0 || index == m_keyToSequence.end())
    {
        return false;
    }

    result = m_imageSequences[index->second];
    return true;
}

}}}
