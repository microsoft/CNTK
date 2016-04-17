//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "NoRandomizer.h"
#include "DataDeserializer.h"
#include "BlockRandomizer.h"
#include <numeric>
#include <random>

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(ReaderLibTests)

class MockChunk : public Chunk
{
private:
    size_t m_chunkBegin;
    size_t m_chunkEnd;
    std::vector<float>& m_data;
    TensorShapePtr m_sampleLayout;

public:
    MockChunk(size_t chunkBegin, size_t chunkEnd, std::vector<float>& data)
        : m_chunkBegin(chunkBegin),
          m_chunkEnd(chunkEnd),
          m_data(data),
          m_sampleLayout(std::make_shared<TensorShape>(1))
    {
        assert(chunkBegin <= chunkEnd);
        assert(chunkEnd <= data.size());
    }

    void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
    {
        assert(m_chunkBegin <= sequenceId);
        assert(sequenceId < m_chunkEnd);

        auto data = std::make_shared<DenseSequenceData>();
        data->m_data = &m_data[sequenceId];
        data->m_numberOfSamples = 1;
        data->m_sampleLayout = m_sampleLayout;
        result.push_back(data);
    }

    ~MockChunk() override {};
};

class MockDeserializer : public IDataDeserializer
{
private:

    size_t m_numChunks;
    size_t m_numSequencesPerChunk;
    std::vector<SequenceDescription> m_descriptions;
    std::vector<float>& m_data;
    std::vector<StreamDescriptionPtr> m_streams;
    TensorShapePtr m_sampleLayout;
    std::vector<ChunkDescriptionPtr> m_chunkDescriptions;

public:
    MockDeserializer(size_t numChunks, size_t numSequencesPerChunks, std::vector<float>& data)
        : m_numChunks(numChunks),
          m_numSequencesPerChunk(numSequencesPerChunks),
          m_sampleLayout(std::make_shared<TensorShape>(1)),
          m_data(data)
    {
        size_t numSequences = numChunks * numSequencesPerChunks;
        m_descriptions.reserve(numSequences);
        assert(data.size() == numSequences);

        for (size_t i = 0; i < numSequences; i++)
        {
            m_descriptions.push_back(SequenceDescription {
                i,
                1,
                i / numSequencesPerChunks,
                true,
                { 0, i }
            });
        }

        for (size_t i = 0; i < numChunks; i++)
        {
            m_chunkDescriptions.push_back(std::make_shared<ChunkDescription>(ChunkDescription {
                i,
                numSequencesPerChunks,
                numSequencesPerChunks
            }));
        }

        m_streams.push_back(std::make_shared<StreamDescription>(StreamDescription{
            L"input",
            0,
            StorageType::dense,
            ElementType::tfloat,
            m_sampleLayout
        }));


    };

    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_streams;
    }

    virtual ChunkPtr GetChunk(size_t chunkId) override
    {
        assert(chunkId < m_numChunks);
        size_t chunkBegin = chunkId * m_numSequencesPerChunk;
        size_t chunkEnd = chunkBegin + m_numSequencesPerChunk;
        std::shared_ptr<Chunk> chunk = std::make_shared<MockChunk>(chunkBegin, chunkEnd, m_data);
        return chunk;
    }

    virtual void GetSequenceDescriptionByKey(const KeyType&, SequenceDescription&) override
    {
        throw std::logic_error("Not implemented");
    }

    virtual ChunkDescriptions GetChunkDescriptions() override
    {
        return m_chunkDescriptions;
    }

    virtual void GetSequencesForChunk(size_t chunkId, std::vector<SequenceDescription>& descriptions) override
    {
        for (size_t i = chunkId * m_numSequencesPerChunk; i < (chunkId + 1) * m_numSequencesPerChunk; i++)
        {
            descriptions.push_back(SequenceDescription{
                i,
                1,
                chunkId,
                true,
                { 0, i }
            });
        }
    }

    MockDeserializer(const MockDeserializer&) = delete;
    MockDeserializer& operator=(const MockDeserializer&) = delete;
};

BOOST_AUTO_TEST_CASE(BlockRandomizerInstantiate)
{
    std::vector<float> data;
    auto mockDeserializer = std::make_shared<MockDeserializer>(0, 0, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer, BlockRandomizer::DecimationMode::chunk, false);
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpoch)
{
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = std::make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    std::vector<float> expected { 3, 4, 1, 8, 0, 5, 9, 6, 7, 2 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    std::vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochWithChunks1)
{
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = std::make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0, 4, mockDeserializer, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    std::vector<float> expected{ 9, 8, 6, 7, 3, 2, 1, 0, 4, 5 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    std::vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
        actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochWithChunks2)
{
    std::vector<float> data(20);
    std::iota(data.begin(), data.end(), 0.0f);

    auto mockDeserializer = std::make_shared<MockDeserializer>(10, 2, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0, 18, mockDeserializer, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    std::vector<float> expected {
        16, 14, 15, 8, 13, 6, 17, 4, 12, 9,
        3, 18, 0, 5, 2, 11, 19, 7, 1, 10
    };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    std::vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
        actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerChaosMonkey)
{
    const int seed = 42;
    const int numChunks = 100;
    const int numSequencesPerChunk = 10;
    const int windowSize = 18;
    std::vector<float> data(numChunks * numSequencesPerChunk);
    std::iota(data.begin(), data.end(), 0.0f);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distr(1, 10);

    auto mockDeserializer = std::make_shared<MockDeserializer>(numChunks, numSequencesPerChunk, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0, windowSize, mockDeserializer, BlockRandomizer::DecimationMode::chunk, false);

    for (int t = 0; t < 100; t++)
    {
        EpochConfiguration epochConfiguration;
        epochConfiguration.m_numberOfWorkers = distr(rng);
        do
        {
            epochConfiguration.m_workerRank = distr(rng) - 1;
        }
        while (epochConfiguration.m_numberOfWorkers <= epochConfiguration.m_workerRank);

        epochConfiguration.m_minibatchSizeInSamples = 0; // don't care
        epochConfiguration.m_totalEpochSizeInSamples = data.size() / distr(rng);
        epochConfiguration.m_epochIndex = distr(rng);
        randomizer->StartEpoch(epochConfiguration);

        int samplesToGet = 0;
        for (int i = 0; i < epochConfiguration.m_totalEpochSizeInSamples + 1; i += samplesToGet)
        {
            samplesToGet = distr(rng);
            Sequences sequences = randomizer->GetNextSequences(samplesToGet);
        }
    }
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochLegacyRandomization)
{
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = std::make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = std::make_shared<BlockRandomizer>(0,
        SIZE_MAX,
        mockDeserializer,
        BlockRandomizer::DecimationMode::sequence,
        true);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    std::vector<float> expected { 9, 4, 1, 2, 0, 5, 3, 6, 7, 8 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    std::vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < 10)
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1);
            actual.push_back(*((float*)data.m_data));

        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(NoRandomizerOneEpoch)
{
    std::vector<float> data(10);
    std::iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = std::make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = std::make_shared<NoRandomizer>(mockDeserializer);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    // Note: for NoRandomizer, end-of-epoch is only returned if there's no data.

    std::vector<float> actual;
    for (int i = 0; i < data.size() + 2; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1);
            actual.push_back(*((float*)data.m_data));
        }

        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
