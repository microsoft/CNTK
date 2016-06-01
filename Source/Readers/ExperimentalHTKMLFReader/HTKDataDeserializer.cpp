//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "HTKDataDeserializer.h"
#include "ConfigHelper.h"
#include "Basics.h"
#include <numeric>

// TODO: This will be removed when dependency on old code is eliminated.
// Currently this fixes the linking.
namespace msra { namespace asr {

std::unordered_map<std::wstring, unsigned int> htkfeatreader::parsedpath::archivePathStringMap;
std::vector<std::wstring> htkfeatreader::parsedpath::archivePathStringVector;

}}

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

HTKDataDeserializer::HTKDataDeserializer(
    CorpusDescriptorPtr corpus,
    const ConfigParameters& cfg,
    bool primary)
    : m_verbosity(0),
      m_corpus(corpus),
      m_primary(primary)
{
    // TODO: This should be read in one place, potentially given by SGD.
    m_frameMode = (ConfigValue)cfg("frameMode", "true");

    m_verbosity = cfg(L"verbosity", 0);

    argvector<ConfigValue> inputs = cfg("input");
    if (inputs.size() != 1)
    {
        InvalidArgument("HTKDataDeserializer supports a single input stream only.");
    }

    ConfigParameters input = inputs.front();
    auto inputName = input.GetMemberIds().front();

    ConfigParameters streamConfig = input(inputName);

    ConfigHelper config(streamConfig);
    auto context = config.GetContextWindow();

    m_elementType = config.GetElementType();
    m_dimension = config.GetFeatureDimension();
    m_dimension = m_dimension * (1 + context.first + context.second);

    InitializeChunkDescriptions(config);
    InitializeStreams(inputName);
    InitializeFeatureInformation();
    InitializeAugmentationWindow(config);
}

HTKDataDeserializer::HTKDataDeserializer(
    CorpusDescriptorPtr corpus,
    const ConfigParameters& feature,
    const wstring& featureName,
    bool primary)
    : m_corpus(corpus),
      m_primary(primary)
{
    // The frame mode is currently specified once per configuration,
    // not in the configuration of a particular deserializer, but on a higher level in the configuration.
    // Because of that we are using find method below.
    m_frameMode = feature.Find("frameMode", "true");

    ConfigHelper config(feature);
    config.CheckFeatureType();

    m_verbosity = feature(L"verbosity", 0);

    auto context = config.GetContextWindow();
    m_elementType = config.GetElementType();

    m_dimension = config.GetFeatureDimension();
    m_dimension = m_dimension * (1 + context.first + context.second);

    InitializeChunkDescriptions(config);
    InitializeStreams(featureName);
    InitializeFeatureInformation();
    InitializeAugmentationWindow(config);
}

void HTKDataDeserializer::InitializeAugmentationWindow(ConfigHelper& config)
{
    m_augmentationWindow = config.GetContextWindow();

    // If not given explicitly, we need to identify the required augmentation range from the expected dimension
    // and the number of dimensions in the file.
    if (m_augmentationWindow.first == 0 && m_augmentationWindow.second == 0)
    {
        m_augmentationWindow.first = m_augmentationWindow.second = msra::dbn::augmentationextent(m_ioFeatureDimension, m_dimension);
    }
}

// Initializes chunks based on the configuration and utterance descriptions.
void HTKDataDeserializer::InitializeChunkDescriptions(ConfigHelper& config)
{
    // Read utterance descriptions.
    vector<wstring> paths = config.GetSequencePaths();
    vector<UtteranceDescription> utterances;
    utterances.reserve(paths.size());
    auto& stringRegistry = m_corpus->GetStringRegistry();
    size_t allUtterances = 0, allFrames = 0;

    for (const auto& u : paths)
    {
        UtteranceDescription description(move(msra::asr::htkfeatreader::parsedpath(u)));
        size_t numberOfFrames = description.GetNumberOfFrames();

        // For logging, also account for utterances and frames that we skip
        allUtterances++;
        allFrames += numberOfFrames;

        string key = description.GetKey();
        if (!m_corpus->IsIncluded(key))
        {
            continue;
        }

        // No need to store key, releasing it.
        description.ClearLogicalPath();

        size_t id = stringRegistry[key];
        description.SetId(id);
        utterances.push_back(description);
        m_totalNumberOfFrames += numberOfFrames;
    }

    // TODO: We should be able to configure IO chunks based on size.
    // distribute utterances over chunks
    // We simply count off frames until we reach the chunk size.
    // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.

    // We have 100 frames in a second.
    const size_t FramesPerSec = 100;

    // A chunk constitutes of 15 minutes
    const size_t ChunkFrames = 15 * 60 * FramesPerSec; // number of frames to target for each chunk

    m_chunks.resize(0);
    m_chunks.reserve(m_totalNumberOfFrames / ChunkFrames);

    ChunkIdType chunkId = 0;
    foreach_index(i, utterances)
    {
        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().GetTotalFrames() > ChunkFrames)
        {
            m_chunks.push_back(HTKChunkDescription(chunkId++));
        }

        // append utterance to last chunk
        HTKChunkDescription& currentChunk = m_chunks.back();
        if (!m_primary)
        {
            // Have to store key <-> utterance mapping for non primary deserializers.
            m_keyToChunkLocation[utterances[i].GetId()] = make_pair(currentChunk.GetChunkId(), currentChunk.GetNumberOfUtterances());
        }

        currentChunk.Add(move(utterances[i]));
    }

    fprintf(stderr,
        "HTKDataDeserializer::HTKDataDeserializer: "
        "selected %" PRIu64 " utterances grouped into %" PRIu64 " chunks, "
        "average chunk size: %.1f utterances, %.1f frames "
        "(for I/O: %.1f utterances, %.1f frames)\n",
        utterances.size(),
        m_chunks.size(),
        utterances.size() / (double)m_chunks.size(),
        m_totalNumberOfFrames / (double)m_chunks.size(),
        allUtterances / (double)m_chunks.size(),
        allFrames / (double)m_chunks.size());
}

// Describes exposed stream - a single stream of htk features.
void HTKDataDeserializer::InitializeStreams(const wstring& featureName)
{
    StreamDescriptionPtr stream = make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = featureName;
    stream->m_sampleLayout = make_shared<TensorShape>(m_dimension);
    stream->m_elementType = m_elementType;
    stream->m_storageType = StorageType::dense;
    m_streams.push_back(stream);
}

// Reading information about the features from the first file.
// This information is used later to check that all features among all files have the same properties.
void HTKDataDeserializer::InitializeFeatureInformation()
{
    msra::util::attempt(5, [&]()
    {
        msra::asr::htkfeatreader reader;
        reader.getinfo(m_chunks.front().GetUtterance(0)->GetPath(), m_featureKind, m_ioFeatureDimension, m_samplePeriod);
        fprintf(stderr, "HTKDataDeserializer::HTKDataDeserializer: determined feature kind as %d-dimensional '%s' with frame shift %.1f ms\n",
            (int)m_ioFeatureDimension, m_featureKind.c_str(), m_samplePeriod / 1e4);
    });
}

// Gets information about available chunks.
ChunkDescriptions HTKDataDeserializer::GetChunkDescriptions()
{
    ChunkDescriptions chunks;
    chunks.reserve(m_chunks.size());

    for (ChunkIdType i = 0; i < m_chunks.size(); ++i)
    {
        auto cd = make_shared<ChunkDescription>();
        cd->m_id = i;
        cd->m_numberOfSamples = m_chunks[i].GetTotalFrames();
        // In frame mode, each frame is represented as sequence.
        // The augmentation is still done for frames in the same sequence only, please see GetSequenceById method.
        cd->m_numberOfSequences = m_frameMode ? m_chunks[i].GetTotalFrames() : m_chunks[i].GetNumberOfUtterances();
        chunks.push_back(cd);
    }
    return chunks;
}

// Gets sequences for a particular chunk.
// This information is used by the randomizer to fill in current windows of sequences.
void HTKDataDeserializer::GetSequencesForChunk(ChunkIdType chunkId, vector<SequenceDescription>& result)
{
    const HTKChunkDescription& chunk = m_chunks[chunkId];
    result.reserve(m_frameMode ? chunk.GetTotalFrames() : chunk.GetNumberOfUtterances());
    size_t offsetInChunk = 0;
    for (size_t i = 0; i < chunk.GetNumberOfUtterances(); ++i)
    {
        auto utterance = chunk.GetUtterance(i);
        // Currently we do not support common prefix, so simply assign the minor to the key.
        size_t sequence = utterance->GetId();

        if (m_frameMode)
        {
            // Because it is a frame mode, creating a sequence for each frame.
            for (size_t k = 0; k < utterance->GetNumberOfFrames(); ++k)
            {
                SequenceDescription f;
                f.m_chunkId = chunkId;
                f.m_key.m_sequence = sequence;
                f.m_key.m_sample = k;
                f.m_id = offsetInChunk++;
                f.m_numberOfSamples = 1;
                result.push_back(f);
            }
        }
        else
        {
            // Creating sequence description per utterance.
            SequenceDescription f;
            f.m_chunkId = chunkId;
            f.m_key.m_sequence = sequence;
            f.m_key.m_sample = 0;
            f.m_id = offsetInChunk++;
            if (SEQUENCELEN_MAX < utterance->GetNumberOfFrames())
            {
                RuntimeError("Maximum number of samples per sequence exceeded");
            }

            f.m_numberOfSamples = (uint32_t) utterance->GetNumberOfFrames();
            result.push_back(f);
        }
    }
}

// A wrapper around a matrix that views it as a vector of column vectors.
// Does not have any memory associated.
class MatrixAsVectorOfVectors
{
public:
    MatrixAsVectorOfVectors(msra::dbn::matrixbase& m)
        : m_matrix(m)
    {
    }

    size_t size() const
    {
        return m_matrix.cols();
    }

    const_array_ref<float> operator[](size_t j) const
    {
        return array_ref<float>(&m_matrix(0, j), m_matrix.rows());
    }

private:
    DISABLE_COPY_AND_MOVE(MatrixAsVectorOfVectors);
    msra::dbn::matrixbase& m_matrix;
};


// Represents a chunk data in memory. Given up to the randomizer.
// It is up to the randomizer to decide when to release a particular chunk.
class HTKDataDeserializer::HTKChunk : public Chunk
{
public:
    HTKChunk(HTKDataDeserializer* parent, ChunkIdType chunkId) : m_parent(parent), m_chunkId(chunkId)
    {
        auto& chunkDescription = m_parent->m_chunks[chunkId];

        // possibly distributed read
        // making several attempts
        msra::util::attempt(5, [&]()
        {
            chunkDescription.RequireData(m_parent->m_featureKind, m_parent->m_ioFeatureDimension, m_parent->m_samplePeriod, m_parent->m_verbosity);
        });
    }

    // Gets data for the sequence.
    virtual void GetSequence(size_t sequenceId, vector<SequenceDataPtr>& result) override
    {
        m_parent->GetSequenceById(m_chunkId, sequenceId, result);
    }

    // Unloads the data from memory.
    ~HTKChunk()
    {
        auto& chunkDescription = m_parent->m_chunks[m_chunkId];
        chunkDescription.ReleaseData(m_parent->m_verbosity);
    }

private:
    DISABLE_COPY_AND_MOVE(HTKChunk);
    HTKDataDeserializer* m_parent;
    ChunkIdType m_chunkId;
};

// Gets a data chunk with the specified chunk id.
ChunkPtr HTKDataDeserializer::GetChunk(ChunkIdType chunkId)
{
    return make_shared<HTKChunk>(this, chunkId);
};

// A matrix that stores all samples of a sequence without padding (differently from ssematrix).
// The number of columns equals the number of samples in the sequence.
// The number of rows equals the size of the feature vector of a sample (= dimensions).
class FeatureMatrix
{
public:
    FeatureMatrix(size_t numRows, size_t numColumns) : m_numRows(numRows), m_numColumns(numColumns)
    {
        m_data.resize(m_numRows * m_numColumns);
    }

    // Returns a reference to the column.
    inline array_ref<float> col(size_t column)
    {
        return array_ref<float>(m_data.data() + m_numRows * column, m_numRows);
    }

    // Gets pointer to the data.
    inline float* GetData()
    {
        return m_data.data();
    }

    // Gets the number of columns. It equals the number of samples in the sequence/utterance.
    inline size_t GetNumberOfColumns() const
    {
        return m_numColumns;
    }

    // Gets total size in elements of stored features.
    inline size_t GetTotalSize() const
    {
        return m_data.size();
    }

private:
    // Features
    std::vector<float> m_data;
    // Number of rows = dimension of the feature
    size_t m_numRows;
    // Number of columns = number of samples in utterance.
    size_t m_numColumns;
};

// This class stores sequence data for HTK for floats.
struct HTKFloatSequenceData : DenseSequenceData
{
    HTKFloatSequenceData(FeatureMatrix&& data) : m_buffer(data)
    {
        m_numberOfSamples = (uint32_t)data.GetNumberOfColumns();
        if (m_numberOfSamples != data.GetNumberOfColumns())
        {
            RuntimeError("Maximum number of samples per sequence exceeded.");
        }
        m_data = m_buffer.GetData();
    }

private:
    FeatureMatrix m_buffer;
};

// This class stores sequence data for HTK for doubles.
struct HTKDoubleSequenceData : DenseSequenceData
{
    HTKDoubleSequenceData(FeatureMatrix& data) : m_buffer(data.GetData(), data.GetData() + data.GetTotalSize())
    {
        m_numberOfSamples = (uint32_t)data.GetNumberOfColumns();
        if (m_numberOfSamples != data.GetNumberOfColumns())
        {
            RuntimeError("Maximum number of samples per sequence exceeded.");
        }
        m_data = m_buffer.data();
    }

private:
    std::vector<double> m_buffer;
};

// Get a sequence by its chunk id and sequence id.
// Sequence ids are guaranteed to be unique inside a chunk.
void HTKDataDeserializer::GetSequenceById(ChunkIdType chunkId, size_t id, vector<SequenceDataPtr>& r)
{
    const auto& chunkDescription = m_chunks[chunkId];
    size_t utteranceIndex = m_frameMode ? chunkDescription.GetUtteranceForChunkFrameIndex(id) : id;
    const UtteranceDescription* utterance = chunkDescription.GetUtterance(utteranceIndex);
    auto utteranceFrames = chunkDescription.GetUtteranceFrames(utteranceIndex);

    // wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors()
    MatrixAsVectorOfVectors utteranceFramesWrapper(utteranceFrames);
    FeatureMatrix features(m_dimension, m_frameMode ? 1 : utterance->GetNumberOfFrames());

    if (m_frameMode)
    {
        // For frame mode augment a single frame.
        size_t frameIndex = id - chunkDescription.GetStartFrameIndexInsideChunk(utteranceIndex);
        msra::dbn::augmentneighbors(utteranceFramesWrapper, vector<char>(), frameIndex, m_augmentationWindow.first, m_augmentationWindow.second, features, 0);
    }
    else
    {
        // Augment complete utterance.
        for (size_t frameIndex = 0; frameIndex < utterance->GetNumberOfFrames(); ++frameIndex)
        {
            msra::dbn::augmentneighbors(utteranceFramesWrapper, vector<char>(), frameIndex, m_augmentationWindow.first, m_augmentationWindow.second, features, frameIndex);
        }
    }

    // Copy features to the sequence depending on the type.
    DenseSequenceDataPtr result;
    if (m_elementType == ElementType::tdouble)
    {
        result = make_shared<HTKDoubleSequenceData>(features);
    }
    else if (m_elementType == ElementType::tfloat)
    {
        result = make_shared<HTKFloatSequenceData>(std::move(features));
    }
    else
    {
        LogicError("Currently, HTK Deserializer supports only double and float types.");
    }

    r.push_back(result);
}

// Gets sequence description by its key.
bool HTKDataDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& d)
{
    assert(!m_primary);
    auto iter = m_keyToChunkLocation.find(key.m_sequence);
    if (iter == m_keyToChunkLocation.end())
    {
        return false;
    }

    auto chunkId = iter->second.first;
    auto utteranceIndexInsideChunk = iter->second.second;
    const auto& chunk = m_chunks[chunkId];
    const auto& sequence = chunk.GetUtterance(utteranceIndexInsideChunk);
    d.m_chunkId = (ChunkIdType)chunkId;
    d.m_id = m_frameMode ? chunk.GetStartFrameIndexInsideChunk(utteranceIndexInsideChunk) + key.m_sample : utteranceIndexInsideChunk;
    d.m_numberOfSamples = m_frameMode ? 1 : (uint32_t)sequence->GetNumberOfFrames();
    return true;
}

}}}
