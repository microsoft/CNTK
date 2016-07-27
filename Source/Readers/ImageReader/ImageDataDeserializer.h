//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "DataDeserializerBase.h"
#include "Config.h"
#include "ByteReader.h"
#include <unordered_map>
#include <numeric>
#include <inttypes.h>
#include "CorpusDescriptor.h"
#include "StringUtil.h"
#include "ImageConfigHelper.h"
#include "ConfigUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Used to keep track of the image. Accessed only using DenseSequenceData interface.
struct DeserializedImage : DenseSequenceData
{
    cv::Mat m_image;
};

// Image data deserializer based on the OpenCV library.
// The deserializer currently supports two output streams only: a feature and a label stream.
// All sequences consist only of a single sample (image/label).
// For features it uses dense storage format with different layout (dimensions) per sequence.
// For labels it either uses the csc sparse (classification) or dense (regression) storage format.
template<LabelType labelType, class PrecisionType>
class ImageDataDeserializer : public DataDeserializerBase
{
public:
    // A new constructor to support new compositional configuration,
    // that allows composition of deserializers and transforms on inputs.
    ImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config)
    {
        // TODO: Avoid code repetition, cf. constructor of ImageConfigHelper
        ConfigParameters inputs = config("input");
        std::vector<std::string> featureNames = GetSectionsWithParameter("ImageDataDeserializer", inputs, "transforms");
        std::vector<std::string> labelNames = GetSectionsWithParameter("ImageDataDeserializer", inputs, "labelDim");

        // TODO: currently support only one feature and label section.
        if (featureNames.size() != 1 || labelNames.size() != 1)
        {
            RuntimeError(
                "ImageReader currently supports a single feature and label stream. '%d' features , '%d' labels found.",
                static_cast<int>(featureNames.size()),
                static_cast<int>(labelNames.size()));
        }

        // Feature stream.
        ConfigParameters featureSection = inputs(featureNames[0]);
        auto features = std::make_shared<StreamDescription>();
        features->m_id = 0;
        features->m_name = msra::strfun::utf16(featureSection.ConfigName());
        // TODO: How to best infer m_sampleLayout? cf. constructor of ImageConfigHelper
        features->m_storageType = StorageType::dense;
        features->m_elementType = std::is_same<PrecisionType, float>::value ? ElementType::tfloat : ElementType::tdouble;
        m_streams.push_back(features);

        // Label stream.
        ConfigParameters labelSection = inputs(labelNames[0]);
        m_labelDimension = labelSection("labelDim");
        auto labels = std::make_shared<StreamDescription>();
        labels->m_id = 1;
        labels->m_name = msra::strfun::utf16(labelSection.ConfigName());
        labels->m_sampleLayout = std::make_shared<TensorShape>(m_labelDimension);
        labels->m_storageType = labelType == LabelType::Classification ? StorageType::sparse_csc : StorageType::dense;
        labels->m_elementType = std::is_same<PrecisionType, float>::value ? ElementType::tfloat : ElementType::tdouble;
        m_streams.push_back(labels);

        m_grayscale = config(L"grayscale", false);

        // TODO: multiview should be done on the level of randomizer/transformers - it is responsiblity of the
        // TODO: randomizer to collect how many copies each transform needs and request same sequence several times.
        CreateSequenceDescriptions(corpus, config(L"file"), config(L"multiViewCrop", false));
    }

    // TODO: This constructor should be deprecated in the future. Compositional config should be used instead.
    explicit ImageDataDeserializer(const ConfigParameters& config)
    {
        ImageConfigHelper configHelper(config);
        m_streams = configHelper.GetStreams();
        assert(m_streams.size() == 2);
        m_grayscale = configHelper.UseGrayscale();
        const auto& label = m_streams[configHelper.GetLabelStreamId()];
        const auto& feature = m_streams[configHelper.GetFeatureStreamId()];

        // Expect data in HWC.
        ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
        feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));

        // could be done statically via a template, but not as simple and readable
        label->m_storageType = labelType == LabelType::Classification ? StorageType::sparse_csc : StorageType::dense;
        feature->m_storageType = StorageType::dense;

        m_featureElementType = feature->m_elementType;
        m_labelDimension = label->m_sampleLayout->GetDim(0);

        CreateSequenceDescriptions(std::make_shared<CorpusDescriptor>(), configHelper.GetMapPath(), configHelper.IsMultiViewCrop());
    }

    // Gets sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override
    {
        auto sequenceDescription = m_imageSequences[chunkId];
        return std::make_shared<ImageChunk<labelType, PrecisionType>>(sequenceDescription, *this);
    }

    // Gets chunk descriptions.
    virtual ChunkDescriptions GetChunkDescriptions() override
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

    // Gets sequence descriptions for the chunk.
    virtual void GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result) override
    {
        // Currently a single sequence per chunk.
        result.push_back(m_imageSequences[chunkId]);
    }

    // Gets sequence description by key.
    bool GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result) override
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

private:
    // Creates a set of sequence descriptions.
    void CreateSequenceDescriptions(CorpusDescriptorPtr corpus, std::string mapPath, bool isMultiCrop)
    {
        std::ifstream mapFile(mapPath);
        if (!mapFile)
        {
            RuntimeError("Could not open %s for reading.", mapPath.c_str());
        }

        size_t itemsPerLine = isMultiCrop ? 10 : 1;
        size_t curId = 0;
        std::string line;
        std::string sequenceKey;
        PathReaderMap knownReaders;
        ImageSequenceDescription<labelType, PrecisionType> description;
        description.m_numberOfSamples = 1;

        auto& stringRegistry = corpus->GetStringRegistry();
        for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
        {
            if (CHUNKID_MAX < curId + itemsPerLine)
                RuntimeError("Maximum number of chunks exceeded.");

            parseLine(/* in */ line, lineIndex, mapPath, /* out */ description, sequenceKey);

            // Skipping sequences that are not included in corpus.
            if (!corpus->IsIncluded(sequenceKey))
                continue;
        
            for (size_t start = curId; curId < start + itemsPerLine; curId++)
            {
                description.m_id = curId;
                description.m_chunkId = (ChunkIdType)curId;
                description.m_path = description.m_path;
                description.m_key.m_sequence = stringRegistry[sequenceKey];
                description.m_key.m_sample = 0;

                m_keyToSequence[description.m_key.m_sequence] = m_imageSequences.size();
                m_imageSequences.push_back(description);
                RegisterByteReader(description.m_id, description.m_path, knownReaders);
            }
        }
    }

    // Image sequence descriptions. Currently, a sequence contains a single sample only.
    // Class not defined to prevent instantiation of non-specialized versions
    template<LabelType labelType, class PrecisionType>
    struct ImageSequenceDescription; // : public SequenceDescription

    // Specialized template for classification
    template<class PrecisionType>
    struct ImageSequenceDescription<LabelType::Classification, PrecisionType> : public SequenceDescription
    {
        std::string m_path;
        size_t m_classId;
    };

    // Specialized template for regression
    template<class PrecisionType>
    struct ImageSequenceDescription<LabelType::Regression, PrecisionType> : public SequenceDescription
    {
        std::string m_path;
        std::vector<PrecisionType> m_label;
    };

    // Function to create label vector out of the data given to it, replaces the former LabelGenerator class
    template<LabelType labelType, class SequenceType>
    void CreateLabelFor(ImageSequenceDescription<labelType, PrecisionType>& desc, SequenceType& data);

    // Specialization for classification
    template<>
    void ImageDataDeserializer::CreateLabelFor(ImageSequenceDescription<LabelType::Classification, PrecisionType>& desc, SparseSequenceData& data)
    {
        auto zero_upto_n_minus_one = [](size_t n)
        {
            std::vector<IndexType> x(n);
            std::iota(x.begin(), x.end(), 0);
            return x;
        };
        static PrecisionType one(1);
        static std::vector<IndexType> indices = zero_upto_n_minus_one(m_labelDimension);
        data.m_nnzCounts.resize(1);
        data.m_nnzCounts[0] = 1;
        data.m_totalNnzCount = 1;
        data.m_data = &one;
        data.m_indices = &(indices[desc.m_classId]);
    }

    // Specialization for regression
    template<>
    void ImageDataDeserializer::CreateLabelFor(ImageSequenceDescription<LabelType::Regression, PrecisionType>& desc, DenseSequenceData& data)
    {
        data.m_data = static_cast<void*>(desc.m_label.data());
    }

    template<LabelType labelType>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<labelType, PrecisionType>& description, std::string& sequenceKey) const;

    // Specialization for classification
    template<>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<LabelType::Classification, PrecisionType>& description, std::string& sequenceKey) const
    {
        std::stringstream ss(line);
        std::string label;

        // Try to parse sequence id, file path and label.
        if (!std::getline(ss, sequenceKey, '\t') || !std::getline(ss, description.m_path, '\t') || !std::getline(ss, label, '\t'))
        {
            // In case when the sequence key is not specified we set it to the line number inside the mapping file.
            // Assume that only image path and class label is given (old format).
            label = description.m_path;
            description.m_path = sequenceKey;
            sequenceKey = std::to_string(lineIndex);

            if (label.empty() || description.m_path.empty())
                RuntimeError("Invalid map file format, must contain 2 or 3 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
        }

        char* eptr;
        errno = 0;
        description.m_classId = strtoull(label.c_str(), &eptr, 10);
        if (label.c_str() == eptr || errno == ERANGE)
            RuntimeError("Invalid map file format, must contain 2 or 3 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());

        if (description.m_classId >= m_labelDimension)
        {
            RuntimeError(
                "Image '%s' has invalid class id '%" PRIu64 "'. Expected label dimension is '%" PRIu64 "'. Line %" PRIu64 " in file %s.",
                description.m_path.c_str(), description.m_classId, m_labelDimension, lineIndex, mapPath.c_str());
        }
    }

    // Specialization for regression
    template<>
    void parseLine(const std::string& line, const size_t lineIndex, const std::string& mapPath, ImageSequenceDescription<LabelType::Regression, PrecisionType>& description, std::string& sequenceKey) const
    {
        std::stringstream ss(line);
        std::string label;
        PrecisionType value;
        std::vector<PrecisionType> result;

        if (!std::getline(ss, sequenceKey, '\t'))
            RuntimeError("Could not read map file, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
        
        char* eptr;
        errno = 0;
        // Test whether first entry is indeed an integer type
        UNUSED(strtoll(sequenceKey.c_str(), &eptr, 0));
        if (sequenceKey.c_str() == eptr || errno == ERANGE)
        {
            // In case when the sequence key is not specified we set it to the line number inside the mapping file.
            // Assume that only image path and regression label(s) are given (old format).
            description.m_path = sequenceKey;
            sequenceKey = std::to_string(lineIndex);

            if (description.m_path.empty())
                RuntimeError("Invalid map file format, must contain at least 2 tab-delimited columns, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
        }
        else
        {
            if (!std::getline(ss, description.m_path, '\t'))
                RuntimeError("Could not read map file, line %" PRIu64 " in file %s.", lineIndex, mapPath.c_str());
        }
        
        for (size_t i = 0; i < m_labelDimension; ++i)
        {
            auto invoke_error = [&]()
            {
                RuntimeError("Could not parse label value on line %" PRIu64 ", column %d, in file %s.", lineIndex, i, mapPath.c_str());
            };

            if (!std::getline(ss, label, '\t'))
                invoke_error();
            errno = 0;
            value = static_cast<PrecisionType>(strtod(label.c_str(), &eptr));
            if (label.c_str() == eptr || errno == ERANGE)
            {
                // try to recover to std::nan
                if (AreEqualIgnoreCase(label, "nan"))
                    value = std::numeric_limits<PrecisionType>::lowest();
                else
                    invoke_error();
            }

            result.push_back(value);
        }

        description.m_label = result;
    }

    // For image, chunks correspond to a single image.
    template<LabelType labelType, class PrecisionType>
    class ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk<labelType, PrecisionType>>
    {
        ImageSequenceDescription<labelType, PrecisionType> m_description;
        ImageDataDeserializer<labelType, PrecisionType>& m_parent;

        template<LabelType labelType>
        SequenceDataPtr createLabeledSequence();

        // Specialization for classification
        template<>
        SequenceDataPtr createLabeledSequence<LabelType::Classification>()
        {
            auto result = std::make_shared<SparseSequenceData>();
            m_parent.CreateLabelFor(m_description, *result);
            return result;
        }

        // Specialization for regression
        template<>
        SequenceDataPtr createLabeledSequence<LabelType::Regression>()
        {
            auto result = std::make_shared<DenseSequenceData>();
            m_parent.CreateLabelFor(m_description, *result);
            return result;
        }

    public:
        ImageChunk(ImageSequenceDescription<labelType, PrecisionType>& description, ImageDataDeserializer<labelType, PrecisionType>& parent)
            : m_description(description), m_parent(parent)
        {
        }

        virtual void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
        {
            assert(sequenceId == m_description.m_id);

            auto image = std::make_shared<DeserializedImage>();
            image->m_image = std::move(m_parent.ReadImage(m_description.m_id, m_description.m_path, m_parent.m_grayscale));
            auto& cvImage = image->m_image;

            if (!cvImage.data)
            {
                RuntimeError("Cannot open file '%s'", m_description.m_path.c_str());
            }

            // Convert element type.
            const int dataType = std::is_same<PrecisionType, float>::value ? CV_32F : CV_64F;
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
            image->m_id = sequenceId;
            image->m_numberOfSamples = 1;
            image->m_chunk = shared_from_this();

            SequenceDataPtr label = createLabeledSequence<labelType>();
            label->m_chunk = shared_from_this();
            label->m_numberOfSamples = 1;

            result.push_back(image);
            result.push_back(label);
        }
    };

    // Sequence descriptions for all input data.
    std::vector<ImageSequenceDescription<labelType, PrecisionType>> m_imageSequences;

    // Mapping of logical sequence key into sequence description.
    std::map<size_t, size_t> m_keyToSequence;

    // Element type of the feature/label stream (currently float/double only).
    ElementType m_featureElementType;

    // whether images shall be loaded in grayscale 
    bool m_grayscale;

    // Dimension of the label vector
    size_t m_labelDimension;

    // Not using nocase_compare here as it's not correct on Linux.
    using PathReaderMap = std::unordered_map<std::string, std::shared_ptr<ByteReader>>;

    void RegisterByteReader(size_t seqId, const std::string& path, PathReaderMap& knownReaders)
    {
        assert(!path.empty());

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
        }
        else
        {
            reader = (*r).second;
        }
        reader->Register(seqId, itemPath);
        m_readers[seqId] = reader;
#else
        UNUSED(seqId);
        UNUSED(knownReaders);
        RuntimeError("The code is built without zip container support. Only plain image files are supported.");
#endif
    }

    cv::Mat ReadImage(size_t seqId, const std::string& path, bool grayscale)
    {
        assert(!path.empty());

        ImageDataDeserializer::SeqReaderMap::const_iterator r;
        if (m_readers.empty() || (r = m_readers.find(seqId)) == m_readers.end())
            return m_defaultReader.Read(seqId, path, grayscale);
        return (*r).second->Read(seqId, path, grayscale);
    }

    // REVIEW alexeyk: can potentially use vector instead of map. Need to handle default reader and resizing though.
    using SeqReaderMap = std::unordered_map<size_t, std::shared_ptr<ByteReader>>;
    SeqReaderMap m_readers;

    FileByteReader m_defaultReader;
};

// Non-template factory functions for C ABI and for convenience
// LabelType and ElementType will be automatically deduced from the config
std::unique_ptr<IDataDeserializer> createImageDataDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config);
std::unique_ptr<IDataDeserializer> createImageDataDeserializer(const ConfigParameters& config); // will be deprecated

}}}
