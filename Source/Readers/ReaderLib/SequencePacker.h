//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "PackerBase.h"

namespace CNTK {

// This packer generates minibatches containing full sequences packed for 
// efficient (concurrent) consumption on a GPU.
class SequencePacker : public PackerBase
{
public:
    SequencePacker(
        SequenceEnumeratorPtr sequenceEnumerator,
        const std::vector<StreamInformation>& streams,
        size_t numberOfBuffers = 2,
        bool useLocalTimeline = false,
        CorpusDescriptorPtr corpus = nullptr) :
        PackerBase(corpus, sequenceEnumerator, streams, numberOfBuffers),
        m_useLocalTimeline(useLocalTimeline),
        m_globalMinibatchSizeInSamples(0),
        m_localMinibatchSizeInSamples(0)
    {}

    virtual Minibatch ReadMinibatch() override;

    void SetConfiguration(const ReaderConfiguration& config, const std::vector<MemoryProviderPtr>& memoryProviders) override;

protected:
    virtual MBLayoutPtr PackDenseStream(const StreamBatch& batch, size_t streamIndex);
    virtual MBLayoutPtr PackSparseStream(const StreamBatch& batch, size_t streamIndex);
    virtual MBLayoutPtr PackBinaryStream(const StreamBatch& batch, size_t streamIndex);

    // Given a number of sequences, creates an MB layout that is used to guide
    // the actual packing.
    virtual MBLayoutPtr CreateMBLayout(const StreamBatch& batch);
    virtual MBLayoutPtr CreateBinaryMBLayout(const StreamBatch& batch);

    // Helper function to check and refresh the sample shape of input samples.
    void RefreshSampleShape(const std::vector<SequenceDataPtr>& minibatch, StreamInformation& outputStream);

    std::pair<vector<MBLayout::SequenceInfo>,size_t> CreateSequenceInfos(const StreamBatch& batch);

    // A flag indicating whether to use local timeline for data.
    bool m_useLocalTimeline;

    // A minibatch size for this worker in local samples.
    size_t m_localMinibatchSizeInSamples;

    // A minibatch size for this worker in global samples.
    size_t m_globalMinibatchSizeInSamples;

};

typedef std::shared_ptr<SequencePacker> SequencePackerPtr;

}
