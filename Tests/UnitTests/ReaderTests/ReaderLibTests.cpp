//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"

#include "NoRandomizer.h"
#include "DataDeserializer.h"
#include "BlockRandomizer.h"
#include "CorpusDescriptor.h"

#include <numeric>
#include <random>

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(ReaderLibTests)

class MockChunk : public Chunk
{
private:
    size_t m_chunkBegin;
    size_t m_chunkEnd;
    TensorShapePtr m_sampleLayout;
    uint32_t m_sequenceLength;
    vector<vector<float>>& m_sequenceData;

public:
    MockChunk(size_t chunkBegin, size_t chunkEnd, vector<vector<float>>& sequenceData, uint32_t sequenceLength)
        : m_chunkBegin(chunkBegin),
          m_chunkEnd(chunkEnd),
          m_sampleLayout(make_shared<TensorShape>(1)),
          m_sequenceLength(sequenceLength),
          m_sequenceData(sequenceData)
    {
        assert(chunkBegin <= chunkEnd);
        assert(chunkEnd <= sequenceData.size());
    }

    void GetSequence(size_t sequenceId, vector<SequenceDataPtr>& result) override
    {
        assert(m_chunkBegin <= sequenceId);
        assert(sequenceId < m_chunkEnd);

        auto data = make_shared<DenseSequenceData>();
        data->m_data = &m_sequenceData[sequenceId][0];
        data->m_numberOfSamples = m_sequenceLength;
        data->m_sampleLayout = m_sampleLayout;
        result.push_back(data);
    }

    ~MockChunk() override {};
};

class MockDeserializer : public IDataDeserializer
{
private:
    uint32_t m_sequenceLength;
    size_t m_numChunks;
    size_t m_numSequencesPerChunk;
    vector<SequenceDescription> m_descriptions;
    vector<StreamDescriptionPtr> m_streams;
    TensorShapePtr m_sampleLayout;
    vector<ChunkDescriptionPtr> m_chunkDescriptions;
    vector<vector<float>> m_sequenceData;

public:
    MockDeserializer(size_t numChunks, size_t numSequencesPerChunks, vector<float>& data, uint32_t sequenceLength = 1)
        : m_numChunks(numChunks),
          m_numSequencesPerChunk(numSequencesPerChunks),
          m_sampleLayout(make_shared<TensorShape>(1)),
          m_sequenceLength(sequenceLength)
    {
        m_sequenceData.reserve(data.size());
        for (float d : data)
        {
            m_sequenceData.push_back(vector<float>(m_sequenceLength, d));
        }

        size_t numSequences = numChunks * numSequencesPerChunks;
        m_descriptions.reserve(numSequences);
        assert(data.size() == numSequences);

        for (size_t i = 0; i < numSequences; i++)
        {
            m_descriptions.push_back(SequenceDescription {
                i,
                m_sequenceLength,
                (ChunkIdType) (i / m_numSequencesPerChunk),
                { 0, i }
            });
        }

        for (ChunkIdType i = 0; i < numChunks; i++)
        {
            m_chunkDescriptions.push_back(make_shared<ChunkDescription>(ChunkDescription {
                i,
                m_numSequencesPerChunk * m_sequenceLength,
                m_numSequencesPerChunk
            }));
        }

        m_streams.push_back(make_shared<StreamDescription>(StreamDescription{
            L"input",
            0,
            StorageType::dense,
            ElementType::tfloat,
            m_sampleLayout
        }));


    };

    vector<StreamDescriptionPtr> GetStreamDescriptions() const override
    {
        return m_streams;
    }

    virtual ChunkPtr GetChunk(ChunkIdType chunkId) override
    {
        assert(chunkId < m_numChunks);
        size_t chunkBegin = chunkId * m_numSequencesPerChunk;
        size_t chunkEnd = chunkBegin + m_numSequencesPerChunk;
        shared_ptr<Chunk> chunk = make_shared<MockChunk>(chunkBegin, chunkEnd, m_sequenceData, m_sequenceLength);
        return chunk;
    }

    virtual bool GetSequenceDescription(const SequenceDescription&, SequenceDescription&) override
    {
        throw logic_error("Not implemented");
    }

    virtual ChunkDescriptions GetChunkDescriptions() override
    {
        return m_chunkDescriptions;
    }

    virtual void GetSequencesForChunk(ChunkIdType chunkId, vector<SequenceDescription>& descriptions) override
    {
        for (size_t i = chunkId * m_numSequencesPerChunk; i < (chunkId + 1) * m_numSequencesPerChunk; i++)
        {
            descriptions.push_back(SequenceDescription{
                i,
                m_sequenceLength,
                chunkId,
                { 0, i }
            });
        }
    }

    MockDeserializer(const MockDeserializer&) = delete;
    MockDeserializer& operator=(const MockDeserializer&) = delete;
};

void BlockRandomizerInstantiateTest(bool prefetch)
{
    vector<float> data;
    auto mockDeserializer = make_shared<MockDeserializer>(0, 0, data);
    auto randomizer = make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer, prefetch, BlockRandomizer::DecimationMode::chunk, false);
}

BOOST_AUTO_TEST_CASE(BlockRandomizerInstantiate)
{
    BlockRandomizerInstantiateTest(false);
    BlockRandomizerInstantiateTest(true);
}

void BlockRandomizerOneEpochTest(bool prefetch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer, prefetch, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected { 3, 4, 1, 8, 0, 5, 9, 6, 7, 2 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpoch)
{
    BlockRandomizerOneEpochTest(false);
    BlockRandomizerOneEpochTest(true);
}

void BlockRandomizerOneEpochWithChunks1Test(bool prefetch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = make_shared<BlockRandomizer>(0, 4, mockDeserializer, prefetch, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected{ 9, 8, 6, 7, 3, 2, 1, 0, 4, 5 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
        actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochWithChunks1)
{
    BlockRandomizerOneEpochWithChunks1Test(false);
    BlockRandomizerOneEpochWithChunks1Test(true);
}

void BlockRandomizerOneEpochWithChunks2Test(bool prefetch)
{
    vector<float> data(20);
    iota(data.begin(), data.end(), 0.0f);

    auto mockDeserializer = make_shared<MockDeserializer>(10, 2, data);

    auto randomizer = make_shared<BlockRandomizer>(0, 18, mockDeserializer, prefetch, BlockRandomizer::DecimationMode::chunk, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected {
        16, 14, 15, 8, 13, 6, 17, 4, 12, 9,
        3, 18, 0, 5, 2, 11, 19, 7, 1, 10
    };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data.m_data));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
        actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochWithChunks2)
{
    BlockRandomizerOneEpochWithChunks2Test(false);
    BlockRandomizerOneEpochWithChunks2Test(true);
}

void BlockRandomizerChaosMonkeyTest(bool prefetch)
{
    const int sequenceLength = 3;
    const int seed = 42;
    const int numChunks = 100;
    const int numSequencesPerChunk = 10;
    const int windowSize = 18;
    vector<float> data(numChunks * numSequencesPerChunk);
    iota(data.begin(), data.end(), 0.0f);
    mt19937 rng(seed);
    uniform_int_distribution<int> distr(1, 10);

    auto mockDeserializer = make_shared<MockDeserializer>(numChunks, numSequencesPerChunk, data, sequenceLength);

    auto randomizer = make_shared<BlockRandomizer>(0, windowSize, mockDeserializer, prefetch, BlockRandomizer::DecimationMode::chunk, false);

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

            // In case end of epoch/decimation/single sequence -> skip the mbSize check.
            if (sequences.m_endOfEpoch || sequences.m_data.empty() || sequences.m_data.front().size() < 2)
            {
                continue;
            }

            // Check that we do not exceed the minibatch size.
            size_t count = 0;
            for (const auto& sequence : sequences.m_data.front())
            {
                count += sequence->m_numberOfSamples;
            }
            BOOST_CHECK_LE(count, samplesToGet);
        }
    }
}

BOOST_AUTO_TEST_CASE(BlockRandomizerChaosMonkey)
{
    BlockRandomizerChaosMonkeyTest(false);
    BlockRandomizerChaosMonkeyTest(true);
}

void BlockRandomizerOneEpochLegacyRandomizationTest(bool prefetch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = make_shared<BlockRandomizer>(0,
        SIZE_MAX,
        mockDeserializer,
        prefetch,
        BlockRandomizer::DecimationMode::sequence,
        true);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected { 9, 4, 1, 2, 0, 5, 3, 6, 7, 8 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < 10)
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data.m_data));

        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochLegacyRandomization)
{
    BlockRandomizerOneEpochLegacyRandomizationTest(false);
    BlockRandomizerOneEpochLegacyRandomizationTest(true);
}

BOOST_AUTO_TEST_CASE(NoRandomizerOneEpoch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = make_shared<NoRandomizer>(mockDeserializer);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    // Note: for NoRandomizer, end-of-epoch is only returned if there's no data.

    vector<float> actual;
    for (int i = 0; i < data.size() + 2; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto data = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data.m_data));
        }

        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i));
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(DefaultCorpusDescriptor)
{
    const int seed = 13;
    mt19937 rng(seed);
    uniform_int_distribution<int> distr(50, 60);

    string randomKey(10, (char)distr(rng));

    CorpusDescriptor corpus;
    BOOST_CHECK_EQUAL(true, corpus.IsIncluded(randomKey));
    BOOST_CHECK_EQUAL(true, corpus.IsIncluded(""));
}

BOOST_AUTO_TEST_CASE(CorpusDescriptorFromFile)
{
    FILE* test = fopen("test.tmp", "w+");
    fwrite("1\n", sizeof(char), 2, test);
    fwrite("2\n", sizeof(char), 2, test);
    fwrite("4\n", sizeof(char), 2, test);
    fclose(test);

    CorpusDescriptor corpus(L"test.tmp");
    BOOST_CHECK_EQUAL(false, corpus.IsIncluded("0"));
    BOOST_CHECK_EQUAL(true, corpus.IsIncluded("1"));
    BOOST_CHECK_EQUAL(true, corpus.IsIncluded("2"));
    BOOST_CHECK_EQUAL(false, corpus.IsIncluded("3"));
    BOOST_CHECK_EQUAL(true, corpus.IsIncluded("4"));
    BOOST_CHECK_EQUAL(false, corpus.IsIncluded("5"));

    remove("test.tmp");
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
