//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include "DataDeserializer.h"
#include "HTKFeaturesIO.h"
#include "UtteranceDescription.h"
#include "ssematrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents a description of an HTK chunk.
// It is only used internally by the HTK deserializer.
// Can exist without associated data and provides methods for requiring/releasing chunk data.
// TODO: We should consider splitting data load from the description in the future versions.
class HTKChunkDescription
{
    // All utterances in the chunk.
    std::vector<UtteranceDescription> m_utterances;

    // Stores all frames of the chunk consecutively (mutable since this is a cache).
    mutable msra::dbn::matrix m_frames;

    // First frames of all utterances. m_firstFrames[utteranceIndex] == index of the first frame of the utterance.
    // Size of m_firstFrames should be equal to the number of utterances.
    std::vector<size_t> m_firstFrames;

    // Total number of frames in this chunk
    size_t m_totalFrames = 0;

    // Chunk id.
    ChunkIdType m_chunkId;

public:

    HTKChunkDescription() : m_chunkId(CHUNKID_MAX) { };

    HTKChunkDescription(ChunkIdType chunkId) : m_chunkId(chunkId) { };

    // Gets number of utterances in the chunk.
    size_t GetNumberOfUtterances() const
    {
        return m_utterances.size();
    }

    ChunkIdType GetChunkId() const
    {
        return m_chunkId;
    }

    // Adds an utterance to the chunk.
    void Add(UtteranceDescription&& utterance)
    {
        if (IsInRam())
        {
            LogicError("Frames already paged into RAM -- too late to add data.");
        }

        m_firstFrames.push_back(m_totalFrames);
        m_totalFrames += utterance.GetNumberOfFrames();
        m_utterances.push_back(std::move(utterance));
    }

    // Gets total number of frames in the chunk.
    size_t GetTotalFrames() const
    {
        return m_totalFrames;
    }

    // Get utterance description by its index.
    const UtteranceDescription* GetUtterance(size_t index) const
    {
        return &m_utterances[index];
    }

    // Get utterance description by its index.
    UtteranceDescription* GetUtterance(size_t index)
    {
        return &m_utterances[index];
    }

    // Get start frame index inside chunk.
    size_t GetStartFrameIndexInsideChunk(size_t index) const
    {
        return m_firstFrames[index];
    }

    // Get utterance by the absolute frame index in chunk.
    // Uses the upper bound to do the binary search among sequences of the chunk.
    size_t GetUtteranceForChunkFrameIndex(size_t frameIndex) const
    {
        auto result = std::upper_bound(
            m_firstFrames.begin(),
            m_firstFrames.end(),
            frameIndex,
            [](size_t fi, const size_t& a) { return fi < a; });
        return result - 1 - m_firstFrames.begin();
    }

    // Returns all frames of a given utterance.
    msra::dbn::matrixstripe GetUtteranceFrames(size_t index) const
    {
        if (!IsInRam())
        {
            LogicError("GetUtteranceFrames was called when data have not yet been paged in.");
        }

        const size_t ts = m_firstFrames[index];
        const size_t n = m_utterances[index].GetNumberOfFrames();
        return msra::dbn::matrixstripe(m_frames, ts, n);
    }

    // Pages-in the data for this chunk.
    // this function supports retrying since we read from the unreliable network, i.e. do not return in a broken state
    // We pass in the feature info variables to check that that data being read has expected properties.
    void RequireData(const string& featureKind, size_t featureDimension, unsigned int samplePeriod, int verbosity = 0) const
    {
        if (GetNumberOfUtterances() == 0)
        {
            LogicError("Cannot page-in empty chunk.");
        }

        if (IsInRam())
        {
            LogicError("Cannot page-in data that is already in memory.");
        }

        try
        {
            // feature reader (we reinstantiate it for each block, i.e. we reopen the file actually)
            // if this is the first feature read ever, we explicitly open the first file to get the information such as feature dimension
            msra::asr::htkfeatreader reader;

            // read all utterances; if they are in the same archive, htkfeatreader will be efficient in not closing the file
            m_frames.resize(featureDimension, m_totalFrames);
            foreach_index(i, m_utterances)
            {
                // read features for this file
                auto framesWrapper = GetUtteranceFrames(i);
                reader.read(m_utterances[i].GetPath(), featureKind, samplePeriod, framesWrapper);
            }

            if (verbosity)
            {
                fprintf(stderr, "HTKChunkDescription::RequireData: read physical chunk %u (%" PRIu64 " utterances, %" PRIu64 " frames, %" PRIu64 " bytes)\n",
                        m_chunkId,
                        m_utterances.size(),
                        m_totalFrames,
                        sizeof(float) * m_frames.rows() * m_frames.cols());
            }
        }
        catch (...)
        {
            // Releasing all data
            m_frames.resize(0, 0);
            throw;
        }
    }

    // Pages-out data for this chunk.
    void ReleaseData(int verbosity = 0) const
    {
        if (GetNumberOfUtterances() == 0)
        {
            LogicError("Cannot page-out empty block.");
        }

        if (!IsInRam())
        {
            LogicError("Cannot page-out data that is not memory.");
        }

        if (verbosity)
        {
            fprintf(stderr, "HTKChunkDescription::ReleaseData: release physical chunk %u (%" PRIu64 " utterances, %" PRIu64 " frames, %" PRIu64 " bytes)\n",
                    m_chunkId,
                    m_utterances.size(),
                    m_totalFrames,
                    sizeof(float) * m_frames.rows() * m_frames.cols());
        }

        // release frames
        m_frames.resize(0, 0);
    }

    private:
        // test if data is in memory at the moment
        bool IsInRam() const
        {
            return !m_frames.empty();
        }
};

}}}
