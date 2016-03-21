//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
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
    const ConfigParameters& feature,
    const wstring& featureName)
    : m_ioFeatureDimension(0),
      m_samplePeriod(0),
      m_verbosity(0),
      m_corpus(corpus),
      m_totalNumberOfFrames(0)
{
    // Currently we only support frame mode.
    // TODO: Support of full sequences.
    bool frameMode = feature.Find("frameMode", "true");
    if (!frameMode)
    {
        LogicError("Currently only reader only supports frame mode. Please check your configuration.");
    }

    ConfigHelper config(feature);
    config.CheckFeatureType();

    auto context = config.GetContextWindow();
    m_elementType = config.GetElementType();

    m_dimension = config.GetFeatureDimension();
    m_dimension = m_dimension * (1 + context.first + context.second);

    m_augmentationWindow = config.GetContextWindow();

    InitializeChunkDescriptions(config);
    InitializeStreams(featureName);
    InitializeFeatureInformation();
}

// Initializes chunks based on the configuration and utterance descriptions.
void HTKDataDeserializer::InitializeChunkDescriptions(ConfigHelper& config)
{
    // Read utterance descriptions.
    vector<wstring> paths = config.GetSequencePaths();
    vector<UtteranceDescription> utterances;
    utterances.reserve(paths.size());
    auto& stringRegistry = m_corpus->GetStringRegistry();
    for (const auto& u : paths)
    {
        UtteranceDescription description(move(msra::asr::htkfeatreader::parsedpath(u)));
        size_t numberOfFrames = description.GetNumberOfFrames();

        // TODO: we need at least 2 frames for boundary markers to work
        // TODO: this should be removed when MLF deserializer is rewritten.
        if (numberOfFrames < 2)
        {
            fprintf(stderr, "HTKDataDeserializer::HTKDataDeserializer: skipping utterance with %d frames because it has less than 2 frames: %ls\n",
                (int)numberOfFrames, description.GetKey().c_str());
            continue;
        }

        size_t id = stringRegistry.AddValue(description.GetKey());
        description.SetId(id);
        utterances.push_back(description);
        m_totalNumberOfFrames += numberOfFrames;
    }

    const size_t MaxUtterancesPerChunk = 65535;
    // distribute them over chunks
    // We simply count off frames until we reach the chunk size.
    // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.

    // We have 100 frames in a second.
    const size_t FramesPerSec = 100;

    // A chunk consitutes 15 minutes
    const size_t ChunkFrames = 15 * 60 * FramesPerSec; // number of frames to target for each chunk

    // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
    // When paging chunk by chunk, chunk size ~14 MB.

    m_chunks.resize(0);
    m_chunks.reserve(m_totalNumberOfFrames / ChunkFrames);

    int chunkId = -1;
    size_t startFrameInsideChunk = 0;
    foreach_index(i, utterances)
    {
        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().GetTotalFrames() > ChunkFrames || m_chunks.back().GetNumberOfUtterances() >= MaxUtterancesPerChunk)
        {
            m_chunks.push_back(HTKChunkDescription());
            chunkId++;
            startFrameInsideChunk = 0;
        }

        // append utterance to last chunk
        HTKChunkDescription& currentChunk = m_chunks.back();
        utterances[i].AssignToChunk(chunkId, currentChunk.GetNumberOfUtterances(), startFrameInsideChunk);
        startFrameInsideChunk += utterances[i].GetNumberOfFrames();
        currentChunk.Add(move(utterances[i]));
    }

    // Creating a table of weak pointers to chunks,
    // so that if randomizer asks the same chunk twice 
    // we do not need to recreated the chunk if we already uploaded in memory.
    m_weakChunks.resize(m_chunks.size());

    fprintf(stderr,
        "HTKDataDeserializer::HTKDataDeserializer: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
        (int)utterances.size(),
        (int)m_chunks.size(),
        utterances.size() / (double)m_chunks.size(),
        m_totalNumberOfFrames / (double)m_chunks.size());
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
            (int)m_dimension, m_featureKind.c_str(), m_samplePeriod / 1e4);
    });
}

// Gets information about available chunks.
ChunkDescriptions HTKDataDeserializer::GetChunkDescriptions()
{
    ChunkDescriptions chunks;
    chunks.reserve(m_chunks.size());

    for (size_t i = 0; i < m_chunks.size(); ++i)
    {
        auto cd = make_shared<ChunkDescription>();
        cd->m_id = i;
        cd->m_numberOfSamples = m_chunks[i].GetTotalFrames();
        cd->m_numberOfSequences = m_chunks[i].GetTotalFrames();
        chunks.push_back(cd);
    }
    return chunks;
}

// Gets sequences for a particular chunk.
// This information is used by the randomizer to fill in current windows of sequences.
void HTKDataDeserializer::GetSequencesForChunk(size_t chunkId, vector<SequenceDescription>& result)
{
    const HTKChunkDescription& chunk = m_chunks[chunkId];
    result.reserve(chunk.GetTotalFrames());
    size_t offsetInChunk = 0;
    for (size_t i = 0; i < chunk.GetNumberOfUtterances(); ++i)
    {
        auto utterance = chunk.GetUtterance(i);
        size_t major = utterance->GetId();
        // Because it is a frame mode, creating sequences for each frame.
        for (size_t k = 0; k < utterance->GetNumberOfFrames(); ++k)
        {
            SequenceDescription f;
            f.m_chunkId = chunkId;
            f.m_key.m_major = major;
            f.m_key.m_minor = k;
            f.m_id = offsetInChunk++;
            f.m_isValid = true;
            f.m_numberOfSamples = 1;
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


// Represets a chunk data in memory. Given up to the randomizer.
// It is up to the randomizer to decide when to release a particular chunk.
class HTKDataDeserializer::HTKChunk : public Chunk
{
public:
    HTKChunk(HTKDataDeserializer* parent, size_t chunkId) : m_parent(parent), m_chunkId(chunkId)
    {
        auto& chunkDescription = m_parent->m_chunks[chunkId];

        // possibly distributed read
        // making several attempts
        msra::util::attempt(5, [&]()
        {
            chunkDescription.RequireData(m_parent->m_featureKind, m_parent->m_ioFeatureDimension, m_parent->m_samplePeriod, m_parent->m_verbosity);
        });
    }

    // Gets data for the sequnce.
    virtual void GetSequence(size_t sequenceId, vector<SequenceDataPtr>& result) override
    {
        m_parent->GetSequenceById(m_chunkId, sequenceId, result);
    }

    // Unloads the data from memory.
    ~HTKChunk()
    {
        auto& chunkDescription = m_parent->m_chunks[m_chunkId];
        chunkDescription.ReleaseData();
    }

private:
    DISABLE_COPY_AND_MOVE(HTKChunk);
    HTKDataDeserializer* m_parent;
    size_t m_chunkId;
};

// Gets a data chunk with the specified chunk id.
ChunkPtr HTKDataDeserializer::GetChunk(size_t chunkId)
{
    if (!m_weakChunks[chunkId].expired())
    {
        return m_weakChunks[chunkId].lock();
    }

    auto chunk = make_shared<HTKChunk>(this, chunkId);
    m_weakChunks[chunkId] = chunk;
    return chunk;
};

// This class stores sequence data for HTK,
//     - for floats: a simple pointer to the chunk data
//     - for doubles: allocated array of doubles which is freed when the sequence is no longer used.
struct HTKSequenceData : DenseSequenceData
{
    msra::dbn::matrix m_buffer;

    ~HTKSequenceData()
    {
        msra::dbn::matrixstripe frame(m_buffer, 0, m_buffer.cols());

        // Checking if m_data just a pointer in to the 
        if (m_data != &frame(0, 0))
        {
            delete[] reinterpret_cast<double*>(m_data);
            m_data = nullptr;
        }
    }
};

typedef shared_ptr<HTKSequenceData> HTKSequenceDataPtr;

// Get a sequence by its chunk id and id.
void HTKDataDeserializer::GetSequenceById(size_t chunkId, size_t id, vector<SequenceDataPtr>& r)
{
    const auto& chunkDescription = m_chunks[chunkId];
    size_t utteranceIndex = chunkDescription.GetUtteranceForChunkFrameIndex(id);
    const UtteranceDescription* utterance = chunkDescription.GetUtterance(utteranceIndex);
    auto utteranceFrames = chunkDescription.GetUtteranceFrames(utteranceIndex);
    size_t frameIndex = id - utterance->GetStartFrameIndexInsideChunk();

    // wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors()
    MatrixAsVectorOfVectors utteranceFramesWrapper(utteranceFrames);

    size_t leftExtent = m_augmentationWindow.first;
    size_t rightExtent = m_augmentationWindow.second;

    // page in the needed range of frames
    if (leftExtent == 0 && rightExtent == 0)
    {
        leftExtent = rightExtent = msra::dbn::augmentationextent(utteranceFramesWrapper[0].size(), m_dimension);
    }

    HTKSequenceDataPtr result = make_shared<HTKSequenceData>();
    result->m_buffer.resize(m_dimension, 1);
    const vector<char> noBoundaryFlags; // TODO: dummy, currently to boundaries supported.
    msra::dbn::augmentneighbors(utteranceFramesWrapper, noBoundaryFlags, frameIndex, leftExtent, rightExtent, result->m_buffer, 0);

    result->m_numberOfSamples = 1;
    msra::dbn::matrixstripe stripe(result->m_buffer, 0, result->m_buffer.cols());
    if (m_elementType == ElementType::tfloat)
    {
        result->m_data = &stripe(0, 0);
    }
    else
    {
        assert(m_elementType == ElementType::tdouble);
        const size_t dimensions = stripe.rows();
        double *doubleBuffer = new double[dimensions];
        const float *floatBuffer = &stripe(0, 0);

        for (size_t i = 0; i < dimensions; i++)
        {
            doubleBuffer[i] = floatBuffer[i];
        }

        result->m_data = doubleBuffer;
    }

    r.push_back(result);
}

}}}
