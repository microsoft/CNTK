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

HTKDataDeserializer::HTKDataDeserializer(
    CorpusDescriptorPtr corpus,
    const ConfigParameters& feature,
    const std::wstring& featureName)
    : m_ioFeatureDimension(0),
      m_samplePeriod(0),
      m_verbosity(0)
{
    bool frameMode = feature.Find("frameMode", "true");
    if (!frameMode)
    {
        LogicError("Currently only reader only supports frame mode. Please check your configuration.");
    }

    ConfigHelper config(feature);

    config.CheckFeatureType();

    std::vector<std::wstring> featureFiles = config.GetFeaturePaths();

    auto context = config.GetContextWindow();
    m_elementType = config.GetElementType();
    m_dimension = config.GetFeatureDimension();
    m_dimension = m_dimension * (1 + context.first + context.second);

    size_t numSequences = featureFiles.size();

    m_augmentationWindow = config.GetContextWindow();

    m_utterances.reserve(numSequences);
    size_t totalFrames = 0;
    foreach_index (i, featureFiles)
    {
        UtteranceDescription description(std::move(msra::asr::htkfeatreader::parsedpath(featureFiles[i])));
        size_t numberOfFrames = description.GetNumberOfFrames();
        description.m_id = i;

        // we need at least 2 frames for boundary markers to work
        if (numberOfFrames < 2)
        {
            fprintf(stderr, "HTKDataDeserializer::HTKDataDeserializer: skipping utterance with %d frames because it has less than 2 frames: %ls\n",
                (int)numberOfFrames, description.GetKey().c_str());
            description.m_isValid = false;
            description.m_numberOfSamples = 0;
        }
        else
        {
            description.m_isValid = true;
            description.m_numberOfSamples = numberOfFrames;
        }

        m_utterances.push_back(description);
        totalFrames += description.m_numberOfSamples;
    }

    size_t totalSize = std::accumulate(
        m_utterances.begin(),
        m_utterances.end(),
        static_cast<size_t>(0),
        [](size_t sum, const UtteranceDescription& s)
        {
            return s.m_numberOfSamples + sum;
        });

    const size_t MaxUtterancesPerChunk = 65535;
    // distribute them over chunks
    // We simply count off frames until we reach the chunk size.
    // Note that we first randomize the chunks, i.e. when used, chunks are non-consecutive and thus cause the disk head to seek for each chunk.
    const size_t framespersec = 100;                   // we just assume this; our efficiency calculation is based on this
    const size_t chunkframes = 15 * 60 * framespersec; // number of frames to target for each chunk

    // Loading an initial 24-hour range will involve 96 disk seeks, acceptable.
    // When paging chunk by chunk, chunk size ~14 MB.

    m_chunks.resize(0);
    m_chunks.reserve(totalSize / chunkframes);

    int chunkId = -1;
    foreach_index(i, m_utterances)
    {
        // if exceeding current entry--create a new one
        // I.e. our chunks are a little larger than wanted (on av. half the av. utterance length).
        if (m_chunks.empty() || m_chunks.back().GetTotalFrames() > chunkframes || m_chunks.back().GetNumberOfUtterances() >= MaxUtterancesPerChunk)
        {
            m_chunks.push_back(ChunkDescription());
            chunkId++;
        }

        // append utterance to last chunk
        ChunkDescription& currentchunk = m_chunks.back();
        m_utterances[i].SetIndexInsideChunk(currentchunk.GetNumberOfUtterances());
        currentchunk.Add(&m_utterances[i]); // move it out from our temp array into the chunk
        m_utterances[i].m_chunkId = chunkId;
    }

    fprintf(stderr,
        "HTKDataDeserializer::HTKDataDeserializer: %d utterances grouped into %d chunks, av. chunk size: %.1f utterances, %.1f frames\n",
        (int)m_utterances.size(),
        (int)m_chunks.size(),
        m_utterances.size() / (double)m_chunks.size(),
        totalSize / (double)m_chunks.size());

    // TODO: Currently we have global sequence id.
    // After changing the timeline interface they must never referred to by a sequential id, only by chunk/within-chunk index
    // because they are asked on the chunk anyway.

    m_frames.reserve(totalFrames);
    foreach_index(i, m_utterances)
    {
        if (!m_utterances[i].m_isValid)
        {
            continue;
        }

        std::wstring key = m_utterances[i].GetKey();
        for (size_t k = 0; k < m_utterances[i].m_numberOfSamples; ++k)
        {
            Frame f(&m_utterances[i]);
            f.m_key.major = key;
            f.m_key.minor = k;
            f.m_id = m_frames.size();
            f.m_chunkId = m_utterances[i].m_chunkId;
            f.m_numberOfSamples = 1;
            f.m_frameIndex = k;
            f.m_isValid = true;
            m_frames.push_back(f);

            m_sequences.push_back(&m_frames[f.m_id]);
        }
    }

    m_weakChunks.resize(m_chunks.size());

    StreamDescriptionPtr stream = std::make_shared<StreamDescription>();
    stream->m_id = 0;
    stream->m_name = featureName;
    stream->m_sampleLayout = std::make_shared<TensorShape>(m_dimension);
    stream->m_elementType = m_elementType;
    stream->m_storageType = StorageType::dense;
    m_streams.push_back(stream);

    msra::util::attempt(5, [&]()
    {
        msra::asr::htkfeatreader reader;
        reader.getinfo(m_utterances[0].GetPath(), m_featureKind, m_ioFeatureDimension, m_samplePeriod);
        fprintf(stderr, "HTKDataDeserializer::HTKDataDeserializer: determined feature kind as %d-dimensional '%s' with frame shift %.1f ms\n",
            (int)m_dimension, m_featureKind.c_str(), m_samplePeriod / 1e4);
    });
}

const SequenceDescriptions& HTKDataDeserializer::GetSequenceDescriptions() const
{
    return m_sequences;
}

std::vector<StreamDescriptionPtr> HTKDataDeserializer::GetStreamDescriptions() const
{
    return m_streams;
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
    HTKDataDeserializer* m_parent;
    size_t m_chunkId;
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

    virtual std::vector<SequenceDataPtr> GetSequence(size_t sequenceId) override
    {
        return m_parent->GetSequenceById(sequenceId);
    }

    ~HTKChunk()
    {
        auto& chunkDescription = m_parent->m_chunks[m_chunkId];
        chunkDescription.ReleaseData();
    }
};

ChunkPtr HTKDataDeserializer::GetChunk(size_t chunkId)
{
    if (!m_weakChunks[chunkId].expired())
    {
        return m_weakChunks[chunkId].lock();
    }

    auto chunk = std::make_shared<HTKChunk>(this, chunkId);
    m_weakChunks[chunkId] = chunk;
    return chunk;
};

struct HTKSequenceData : DenseSequenceData
{
    msra::dbn::matrix m_buffer;

    ~HTKSequenceData()
    {
        msra::dbn::matrixstripe frame(m_buffer, 0, m_buffer.cols());
        if (m_data != &frame(0, 0))
        {
            delete[] reinterpret_cast<double*>(m_data);
            m_data = nullptr;
        }
    }
};

typedef std::shared_ptr<HTKSequenceData> HTKSequenceDataPtr;

std::vector<SequenceDataPtr> HTKDataDeserializer::GetSequenceById(size_t id)
{
    const auto& frame = m_frames[id];
    UtteranceDescription* utterance = frame.m_utterence;

    const auto& chunkDescription = m_chunks[utterance->m_chunkId];
    auto utteranceFrames = chunkDescription.GetUtteranceFrames(utterance->GetIndexInsideChunk());

    // wrapper that allows m[j].size() and m[j][i] as required by augmentneighbors()
    MatrixAsVectorOfVectors utteranceFramesWrapper(utteranceFrames);

    size_t leftExtent = m_augmentationWindow.first;
    size_t rightExtent = m_augmentationWindow.second;

    // page in the needed range of frames
    if (leftExtent == 0 && rightExtent == 0)
    {
        leftExtent = rightExtent = msra::dbn::augmentationextent(utteranceFramesWrapper[0].size(), m_dimension);
    }

    HTKSequenceDataPtr result = std::make_shared<HTKSequenceData>();
    result->m_buffer.resize(m_dimension, 1);
    const std::vector<char> noBoundaryFlags; // dummy
    msra::dbn::augmentneighbors(utteranceFramesWrapper, noBoundaryFlags, frame.m_frameIndex, leftExtent, rightExtent, result->m_buffer, 0);

    result->m_numberOfSamples = frame.m_numberOfSamples;
    msra::dbn::matrixstripe stripe(result->m_buffer, 0, result->m_buffer.cols());
    const size_t dimensions = stripe.rows();

    if (m_elementType == ElementType::tfloat)
    {
        result->m_data = &stripe(0, 0);
    }
    else
    {
        assert(m_elementType == ElementType::tdouble);
        double *doubleBuffer = new double[dimensions];
        const float *floatBuffer = &stripe(0, 0);

        for (size_t i = 0; i < dimensions; i++)
        {
            doubleBuffer[i] = floatBuffer[i];
        }

        result->m_data = doubleBuffer;
    }

    return std::vector<SequenceDataPtr>(1, result);
}

const SequenceDescription* HTKDataDeserializer::GetSequenceDescriptionByKey(const KeyType&)
{
    LogicError("HTKDataDeserializer::GetSequenceDescriptionByKey: currently not implemented. Supported only as a primary deserializer.");
}

size_t HTKDataDeserializer::GetTotalNumberOfChunks()
{
    return m_chunks.size();
}

} } }
