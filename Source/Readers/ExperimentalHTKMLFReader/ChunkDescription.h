//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "DataDeserializer.h"
#include "../HTKMLFReader/htkfeatio.h"
#include "ssematrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Class represents a description of an HTK chunk.
// It is only used internally by the HTK deserializer.
// Can exist without associated data and provides methods for requiring/releasing chunk data.
class ChunkDescription
{
    // All utterances in the chunk.
    std::vector<UtteranceDescription*> m_utteranceSet;

    // Stores all frames of the chunk consecutively (mutable since this is a cache).
    mutable msra::dbn::matrix m_frames;

    // First frames of all utterances. m_firstFrames[utteranceIndex] == index of the first frame of the utterance.
    // Size of m_firstFrames should be equal to the number of utterances.
    std::vector<size_t> m_firstFrames;

    // Total number of frames in this chunk
    size_t m_totalFrames;

public:
    ChunkDescription() : m_totalFrames(0)
    {
    }

    // Gets number of utterances in the chunk.
    size_t GetNumberOfUtterances() const
    {
        return m_utteranceSet.size();
    }

    // Adds an utterance to the chunk.
    void Add(UtteranceDescription* utterance)
    {
        if (IsInRam())
        {
            LogicError("Frames already paged into RAM -- too late to add data.");
        }

        m_firstFrames.push_back(m_totalFrames);
        m_totalFrames += utterance->GetNumberOfFrames();
        m_utteranceSet.push_back(utterance);
    }

    // Gets total number of frames in the chunk.
    size_t GetTotalFrames() const
    {
        return m_totalFrames;
    }

    // Get number of frames in a sequences identified by the index.
    size_t GetUtteranceNumberOfFrames(size_t index) const
    {
        return m_utteranceSet[index]->GetNumberOfFrames();
    }

    // Returns frames of a given utterance.
    msra::dbn::matrixstripe GetUtteranceFrames(size_t index) const
    {
        if (!IsInRam())
        {
            LogicError("GetUtteranceFrames was called when data have not yet been paged in.");
        }

        const size_t ts = m_firstFrames[index];
        const size_t n = GetUtteranceNumberOfFrames(index);
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
            foreach_index(i, m_utteranceSet)
            {
                // read features for this file
                auto framesWrapper = GetUtteranceFrames(i);
                reader.read(m_utteranceSet[i]->GetPath(), featureKind, samplePeriod, framesWrapper);
            }

            if (verbosity)
            {
                fprintf(stderr, "RequireData: %d utterances read\n", (int)m_utteranceSet.size());
            }
        }
        catch (...)
        {
            ReleaseData();
            throw;
        }
    }

    // Pages-out data for this chunk.
    void ReleaseData() const
    {
        if (GetNumberOfUtterances() == 0)
        {
            LogicError("Cannot page-out empty block.");
        }

        if (!IsInRam())
        {
            LogicError("Cannot page-out data that is not memory.");
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
