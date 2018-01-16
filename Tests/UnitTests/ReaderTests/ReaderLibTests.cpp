//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <numeric>
#include <random>
#include <set>
#include "NoRandomizer.h"
#include "DataDeserializer.h"
#include "BlockRandomizer.h"
#include "CorpusDescriptor.h"
#include "FramePacker.h"
#include "SequencePacker.h"
#include "TruncatedBpttPacker.h"
#include "CudaMemoryProvider.h"
#include "HeapMemoryProvider.h"
#include "BufferedFileReader.h"

#pragma warning(push)
// disable warning about possible mod 0 operation in uniform_int_distribution
#pragma warning(disable:4724)
#include <boost/random/uniform_int_distribution.hpp>
#pragma warning(pop)

#include "SequentialDeserializer.h"

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

using namespace ::CNTK;

BOOST_AUTO_TEST_SUITE(ReaderLibTests)

class MockChunk : public Chunk
{
private:
    size_t m_chunkBegin;
    size_t m_chunkEnd;
    NDShape m_sampleShape;
    uint32_t m_sequenceLength;
    vector<vector<float>>& m_sequenceData;

public:
    MockChunk(size_t chunkBegin, size_t chunkEnd, vector<vector<float>>& sequenceData, uint32_t sequenceLength)
        : m_chunkBegin(chunkBegin),
          m_chunkEnd(chunkEnd),
          m_sampleShape(NDShape({ 1 })),
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

        auto data = make_shared<MockDenseSequenceData>();
        data->m_data = &m_sequenceData[sequenceId][0];
        data->m_numberOfSamples = m_sequenceLength;
        data->m_sampleShape = m_sampleShape;
        result.push_back(data);
    }

    ~MockChunk() override {};
};

class MockDeserializer : public DataDeserializer
{
private:
    uint32_t m_sequenceLength;
    size_t m_numChunks;
    size_t m_numSequencesPerChunk;
    vector<SequenceInfo> m_descriptions;
    vector<StreamInformation> m_streams;
    NDShape m_sampleShape;
    vector<ChunkInfo> m_chunkDescriptions;
    vector<vector<float>> m_sequenceData;

public:
    MockDeserializer(size_t numChunks, size_t numSequencesPerChunks, const vector<float>& data, uint32_t sequenceLength = 1)
        : m_numChunks(numChunks),
          m_numSequencesPerChunk(numSequencesPerChunks),
          m_sampleShape(NDShape(1)),
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
            m_descriptions.push_back(SequenceInfo {
                i,
                m_sequenceLength,
                (ChunkIdType) (i / m_numSequencesPerChunk),
                SequenceKey(0, static_cast<uint32_t>(i))
            });
        }

        for (ChunkIdType i = 0; i < numChunks; i++)
        {
            m_chunkDescriptions.push_back(ChunkInfo {
                i,
                m_numSequencesPerChunk * m_sequenceLength,
                m_numSequencesPerChunk
            });
        }

        StreamInformation si;
        si.m_name = L"input";
        si.m_id = 0;
        si.m_storageFormat = StorageFormat::Dense;
        si.m_elementType = DataType::Float;
        si.m_sampleLayout = m_sampleShape;
        m_streams.push_back(si);
    };

    vector<StreamInformation> StreamInfos() override
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

    virtual bool GetSequenceInfo(const SequenceInfo&, SequenceInfo&) override
    {
        throw logic_error("Not implemented");
    }

    virtual std::vector<ChunkInfo> ChunkInfos() override
    {
        return m_chunkDescriptions;
    }

    virtual void SequenceInfosForChunk(ChunkIdType chunkId, vector<SequenceInfo>& descriptions) override
    {
        for (size_t i = chunkId * m_numSequencesPerChunk; i < (chunkId + 1) * m_numSequencesPerChunk; i++)
        {
            descriptions.push_back(SequenceInfo{
                i,
                m_sequenceLength,
                chunkId,
                { 0, static_cast<uint32_t>(i) }
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
    auto randomizer = make_shared<BlockRandomizer>(0, SIZE_MAX, mockDeserializer, prefetch, false);
}

BOOST_AUTO_TEST_CASE(CheckSetCurrentCursorForRandomizers)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 5;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    auto expectedBlock = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);
    auto expectedNo = make_shared<NoRandomizer>(deserializer, false);

    auto underTestBlock = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);
    auto unterTestNo = make_shared<NoRandomizer>(deserializer, false);

    auto test = [](SequenceEnumeratorPtr expected, SequenceEnumeratorPtr underTest, size_t epochSize)
    {
        auto firstEpoch = ReadFullEpoch(expected, epochSize, 0);
        auto secondEpoch = ReadFullEpoch(expected, epochSize, 1);
        auto thirdEpoch = ReadFullEpoch(expected, epochSize, 2);

        // First setup the enumerator to ead unbounded amount of data
        EpochConfiguration config;
        config.m_numberOfWorkers = 1;
        config.m_workerRank = 0;
        config.m_minibatchSizeInSamples = 1;
        config.m_totalEpochSizeInSamples = std::numeric_limits<size_t>().max() / 2;
        config.m_epochIndex = 0;
        underTest->StartEpoch(config);

        // Rereading second epoch
        std::map<std::wstring, size_t> state;
        state[g_minibatchSourcePosition] = firstEpoch.size();
        underTest->SetState(state);
        auto anotherSecond = ReadNextSamples(underTest, secondEpoch.size());
        BOOST_CHECK_EQUAL_COLLECTIONS(
            secondEpoch.begin(),
            secondEpoch.end(),
            anotherSecond.begin(),
            anotherSecond.end());

        // Rereading first epoch
        state[g_minibatchSourcePosition] = 0;
        underTest->SetState(state);
        auto anotherFirst = ReadNextSamples(underTest, firstEpoch.size());
        BOOST_CHECK_EQUAL_COLLECTIONS(
            firstEpoch.begin(),
            firstEpoch.end(),
            anotherFirst.begin(),
            anotherFirst.end());

        // Rereading third epoch
        state[g_minibatchSourcePosition] = firstEpoch.size() + secondEpoch.size();
        underTest->SetState(state);
        auto anotherThird = ReadNextSamples(underTest, thirdEpoch.size());
        BOOST_CHECK_EQUAL_COLLECTIONS(
            thirdEpoch.begin(),
            thirdEpoch.end(),
            anotherThird.begin(),
            anotherThird.end());
    };

    // Inside sweep
    size_t epochSize = 50000;
    test(expectedBlock, underTestBlock, epochSize);
    test(expectedNo, unterTestNo, epochSize);

    // Between sweeps
    epochSize = (size_t)(sweepNumberOfSamples / 1.5);
    test(expectedBlock, underTestBlock, epochSize);
    test(expectedNo, unterTestNo, epochSize);
}

BOOST_AUTO_TEST_CASE(RandRollbackToEarlierEpochBetweenSweeps)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 5;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    // Let's randomize complete sweep, so that we have a baseline.
    auto randomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);

    // Let's read all sequences from the first three sweeps in the randomized order.
    auto firstSweep = ReadFullSweep(randomizer, 0, sweepNumberOfSamples);
    auto secondSweep = ReadFullSweep(randomizer, 1, sweepNumberOfSamples);
    auto thirdSweep = ReadFullSweep(randomizer, 2, sweepNumberOfSamples);

    // Now let's merge the global timeline of these three sweeps.
    std::vector<float> threeSweeps = Concat(std::vector<vector<float>>{ firstSweep, secondSweep, thirdSweep });

    // Ok, now let's run smaller epochs and check whether they are the same as full sweeps.
    size_t epochSize = threeSweeps.size() / 5;
    auto firstEpoch = ReadFullEpoch(randomizer, epochSize, 0);
    auto secondEpoch = ReadFullEpoch(randomizer, epochSize, 1);
    auto thirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);
    auto fourthEpoch = ReadFullEpoch(randomizer, epochSize, 3);
    auto fifthEpoch = ReadFullEpoch(randomizer, epochSize, 4);
    std::vector<float> anotherThreeSweeps = Concat(std::vector<vector<float>>{ firstEpoch, secondEpoch, thirdEpoch, fourthEpoch, fifthEpoch });

    // Check that data is the same.
    BOOST_CHECK_EQUAL_COLLECTIONS(threeSweeps.begin(), threeSweeps.end(), anotherThreeSweeps.begin(), anotherThreeSweeps.end());

    // Now roll back to the third one.
    auto anotherThirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);

    // Check that it is the same.
    BOOST_CHECK_EQUAL_COLLECTIONS(thirdEpoch.begin(), thirdEpoch.end(), anotherThirdEpoch.begin(), anotherThirdEpoch.end());
}

BOOST_AUTO_TEST_CASE(RandRollbackToEarlierEpochInTheSweep)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 3;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    // Let's randomize complete sweep, so that we have a baseline.
    auto randomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);

    // Let's read all sequences from the first three sweeps in the randomized order.
    auto firstSweep = ReadFullSweep(randomizer, 0, sweepNumberOfSamples);

    // Ok, now let's run smaller epochs and check whether they are the same as full sweeps.
    size_t epochSize = firstSweep.size() / 3;
    auto firstEpoch = ReadFullEpoch(randomizer, epochSize, 0);
    auto secondEpoch = ReadFullEpoch(randomizer, epochSize, 1);
    auto thirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);
    std::vector<float> anotherThreeSweeps = Concat(std::vector<vector<float>>{ firstEpoch, secondEpoch, thirdEpoch });

    // Check that data is the same.
    BOOST_CHECK_EQUAL_COLLECTIONS(firstSweep.begin(), firstSweep.end(), anotherThreeSweeps.begin(), anotherThreeSweeps.end());

    // Now roll back to the second one.
    auto anotherSecondEpoch = ReadFullEpoch(randomizer, epochSize, 1);

    // Check that it is the same.
    BOOST_CHECK_EQUAL_COLLECTIONS(secondEpoch.begin(), secondEpoch.end(), anotherSecondEpoch.begin(), anotherSecondEpoch.end());
}

BOOST_AUTO_TEST_CASE(RandRollbackToSameEpochInTheSweep)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 3;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    // Let's randomize complete sweep, so that we have a baseline.
    auto randomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);

    // Let's read all sequences from the first three sweeps in the randomized order.
    auto firstSweep = ReadFullSweep(randomizer, 0, sweepNumberOfSamples);

    // Ok, now let's run smaller epochs and check whether they are the same as full sweeps.
    size_t epochSize = firstSweep.size() / 4;
    auto firstEpoch = ReadFullEpoch(randomizer, epochSize, 0);
    auto secondEpoch = ReadFullEpoch(randomizer, epochSize, 1);
    auto thirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);

    // Now roll back to the third one.
    auto anotherThirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);

    // Check that it is the same.
    BOOST_CHECK_EQUAL_COLLECTIONS(thirdEpoch.begin(), thirdEpoch.end(), anotherThirdEpoch.begin(), anotherThirdEpoch.end());
}

BOOST_AUTO_TEST_CASE(RandRollbackToSameEpochInBigRandomizationWindow)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = sweepNumberOfSamples / 2;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    // Let's randomize complete sweep, so that we have a baseline.
    auto randomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);

    // Let's read all sequences from the first three sweeps in the randomized order.
    auto firstSweep = ReadFullSweep(randomizer, 0, sweepNumberOfSamples);

    // Ok, now let's run smaller epochs and check whether they are the same as full sweeps.
    size_t epochSize = firstSweep.size() / 5;
    auto firstEpoch = ReadFullEpoch(randomizer, epochSize, 0);
    auto secondEpoch = ReadFullEpoch(randomizer, epochSize, 1);
    auto thirdEpoch = ReadFullEpoch(randomizer, epochSize, 2);
    auto fourthEpoch = ReadFullEpoch(randomizer, epochSize, 3);

    // Now roll back to the third one.
    auto current = ReadFullEpoch(randomizer, epochSize, 1);
    BOOST_CHECK_EQUAL_COLLECTIONS(secondEpoch.begin(), secondEpoch.end(), current.begin(), current.end());

    current = ReadFullEpoch(randomizer, epochSize, 3);
    BOOST_CHECK_EQUAL_COLLECTIONS(fourthEpoch.begin(), fourthEpoch.end(), current.begin(), current.end());

    current = ReadFullEpoch(randomizer, epochSize, 2);
    BOOST_CHECK_EQUAL_COLLECTIONS(thirdEpoch.begin(), thirdEpoch.end(), current.begin(), current.end());

    current = ReadFullEpoch(randomizer, epochSize, 2);
    BOOST_CHECK_EQUAL_COLLECTIONS(thirdEpoch.begin(), thirdEpoch.end(), current.begin(), current.end());
}


BOOST_AUTO_TEST_CASE(BlockRandomizerInstantiate)
{
    BlockRandomizerInstantiateTest(false);
    BlockRandomizerInstantiateTest(true);
}

void OneEpochRandomizationTest(SequenceEnumerator& randomizer, size_t sweepSize, const EpochConfiguration& epochConfig, const vector<float>& expectedOutput, size_t sequenceLength = 1)
{
    auto epochSize = epochConfig.m_totalEpochSizeInSamples;
    auto mbSize = epochConfig.m_minibatchSizeInSamples;

    BOOST_ASSERT(epochSize == expectedOutput.size());

    randomizer.StartEpoch(epochConfig);

    vector<float> actual;
    for (int totalSamplesRead = 0; totalSamplesRead < epochSize;)
    {
        Sequences sequences = randomizer.GetNextSequences(mbSize, mbSize);
        BOOST_ASSERT(sequences.m_data.size() == 1); // only one input stream
        auto& stream = sequences.m_data[0];
        auto numSampleRead = 0;
        for (auto& sequence : stream) 
        {
            auto numSamples = sequence->m_numberOfSamples;
            numSampleRead += numSamples;
            auto& data = reinterpret_cast<DenseSequenceData&>(*sequence);
            actual.reserve(actual.size() + numSamples);
            std::copy_n(((float*)data.GetDataBuffer()), numSamples, std::back_inserter(actual));
        }
        
        auto expectedSize = std::min(epochSize - totalSamplesRead, mbSize);
        if (!epochConfig.m_allowMinibatchesToCrossSweepBoundaries) 
        {
            expectedSize = std::min(sweepSize - totalSamplesRead % sweepSize, expectedSize);
        }
       
        // at least one sequence is returned in case when mbSize < sequenceLength
        expectedSize = std::max(expectedSize, sequenceLength);
        BOOST_REQUIRE(numSampleRead <= std::max(mbSize, sequenceLength));
        if (sequenceLength == 1) 
            BOOST_REQUIRE(numSampleRead == expectedSize);
        else 
            BOOST_REQUIRE(expectedSize - numSampleRead < sequenceLength);
        
        BOOST_REQUIRE(sequences.m_endOfEpoch == (totalSamplesRead + numSampleRead == epochSize));
        BOOST_REQUIRE(sequences.m_endOfSweep == (totalSamplesRead / sweepSize != (totalSamplesRead + numSampleRead) / sweepSize));

        totalSamplesRead += numSampleRead;
    }

    for (int i = 0; i < 3; i++)
    {
        auto numSamples = i + 1;
        Sequences sequences = randomizer.GetNextSequences(numSamples, numSamples);
        BOOST_REQUIRE(sequences.m_data.size() == 0);
        BOOST_REQUIRE(sequences.m_endOfEpoch == true);
        BOOST_REQUIRE(sequences.m_endOfSweep == (epochSize % sweepSize == 0));
    }

    BOOST_REQUIRE_EQUAL_COLLECTIONS(expectedOutput.begin(), expectedOutput.end(),
                                  actual.begin(), actual.end());
}

void TestRandomization(EpochConfiguration& epochConfiguration, DataDeserializerPtr deserializer, size_t sweepSize, const vector<float>& expectedRandomized, const vector<float>& expectedNotRandomized, size_t sequenceLength = 1)
{
    BlockRandomizer randomizer1(0, SIZE_MAX, deserializer, /*prefetch =*/ false);
    BlockRandomizer randomizer2(0, SIZE_MAX, deserializer, /*prefetch =*/ true);
    NoRandomizer randomizer3(deserializer);

    BlockRandomizer randomizer4(0, SIZE_MAX, deserializer, /*prefetch =*/ false, false, /*multithreadedGetNextSequences =*/ true);
    BlockRandomizer randomizer5(0, SIZE_MAX, deserializer, /*prefetch =*/ true, false, /*multithreadedGetNextSequences =*/ true);
    NoRandomizer randomizer6(deserializer, /*multithreadedGetNextSequences =*/ true);
    
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_totalEpochSizeInSamples = expectedRandomized.size();

    for (int i = 1; i <= epochConfiguration.m_totalEpochSizeInSamples + 1; i++)
    {
        epochConfiguration.m_minibatchSizeInSamples = i;
        OneEpochRandomizationTest(randomizer1, sweepSize, epochConfiguration, expectedRandomized, sequenceLength);
        OneEpochRandomizationTest(randomizer2, sweepSize, epochConfiguration, expectedRandomized, sequenceLength);
        OneEpochRandomizationTest(randomizer3, sweepSize, epochConfiguration, expectedNotRandomized, sequenceLength);

        OneEpochRandomizationTest(randomizer4, sweepSize, epochConfiguration, expectedRandomized, sequenceLength);
        OneEpochRandomizationTest(randomizer5, sweepSize, epochConfiguration, expectedRandomized, sequenceLength);
        OneEpochRandomizationTest(randomizer6, sweepSize, epochConfiguration, expectedNotRandomized, sequenceLength);
    }
}

BOOST_AUTO_TEST_CASE(TestChunkBasedRandomization)
{
    auto num_chunks = 10;
    auto num_sequences = 100;
    vector<float> input(num_sequences * num_chunks);
    iota(input.begin(), input.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(num_chunks, num_sequences, input);

    for (int k = 0; k <= 20; k++) 
    {
        ChunkRandomizer randomizer1(mockDeserializer, k * num_sequences, true);
        ChunkRandomizer randomizer2(mockDeserializer, k, false);

        randomizer1.Randomize(k);
        randomizer2.Randomize(k);

        auto& randomizedChunks1 = randomizer1.GetRandomizedChunks();
        auto& randomizedChunks2 = randomizer2.GetRandomizedChunks();

        BOOST_ASSERT(randomizedChunks1.size() == randomizedChunks2.size());

        for (int i = 0; i < randomizedChunks1.size(); i++)
        {
            auto& a = randomizedChunks1[i];
            auto& b = randomizedChunks2[i];
            BOOST_CHECK(a.m_chunkId == b.m_chunkId);
            BOOST_CHECK(a.m_original->m_id == b.m_original->m_id);
            BOOST_CHECK(a.m_samplePositionStart == b.m_samplePositionStart);
            BOOST_CHECK(a.m_sequencePositionStart == b.m_sequencePositionStart);

            BOOST_CHECK(b.m_randomizationWindow.m_end > b.m_randomizationWindow.m_begin);

            BOOST_CHECK(a.m_randomizationWindow.m_begin >= b.m_randomizationWindow.m_begin);
            BOOST_CHECK(a.m_randomizationWindow.m_end <= b.m_randomizationWindow.m_end);

            auto window = size_t(std::min(num_chunks, std::max(k, 1)));
            BOOST_CHECK(b.m_randomizationWindow.m_end - b.m_randomizationWindow.m_begin == window);
        }
    }
}


BOOST_AUTO_TEST_CASE(TestChunkBasedRandomizationQuality)
{
    auto num_chunks = 10;
    auto num_sequences = 100;
    vector<float> input(num_sequences * num_chunks);
    iota(input.begin(), input.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(num_chunks, num_sequences, input);

    auto randomizationRange = 3;

    BlockRandomizer randomizer(0, randomizationRange, 
        mockDeserializer, /*prefetch =*/ false,
        /*multithreadedGetNextSequences =*/ false,
        /*maxNumberOfInvalidSequences =*/ 0,
        /*sampleBasedRandomizationWindow =*/ false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = num_sequences;
    epochConfiguration.m_totalEpochSizeInSamples = input.size();
    epochConfiguration.m_epochIndex = 0;

    randomizer.StartEpoch(epochConfiguration);

    for (int i = 0; i < num_chunks; i++)
    {
        Sequences sequences = randomizer.GetNextSequences(num_sequences, num_sequences);
        BOOST_ASSERT(sequences.m_data.size() == 1); // 1 stream
        BOOST_ASSERT(sequences.m_data[0].size() == num_sequences); // 100 sequences, each containing 1 sample
        
        std::set<int> chunkIds;

        for (int j = 0; j < num_sequences; j++)
        {
            // make sure that not all 100 consecutive samples belong to the same chunk
            auto& sample = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][j]);
            float value = *((float*)sample.GetDataBuffer());
            chunkIds.insert(int(value / num_sequences));
        }
        
        // TODO: actually, each chunk-worth of sequences should contain data from 
        // randomizationRange different chunks!
        BOOST_CHECK(chunkIds.size() > 1);
    }
}

BOOST_AUTO_TEST_CASE(TestRandomization_FirstEpoch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 6, 3, 1, 5, 9, 0, 4, 2, 7, 8 };

    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_epochIndex = 0;

    TestRandomization(epochConfiguration, mockDeserializer, data.size(), expected, data);
}

BOOST_AUTO_TEST_CASE(TestRandomization_SecondEpoch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 3, 0, 8, 4, 7, 5, 2, 9, 1, 6 };

    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);
    
    EpochConfiguration epochConfiguration;
    epochConfiguration.m_epochIndex = 1;

    TestRandomization(epochConfiguration, mockDeserializer, data.size(), expected, data);
}


BOOST_AUTO_TEST_CASE(TestRandomization_TwoSweeps)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 6, 3, 1, 5, 9, 0, 4, 2, 7, 8, 3, 0, 8, 4, 7, 5, 2, 9, 1, 6 };

    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto sweepSize = data.size();
    data.reserve(2 * sweepSize);
    std::copy_n(data.begin(), sweepSize, std::back_inserter(data));

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_epochIndex = 0;

    TestRandomization(epochConfiguration, mockDeserializer, sweepSize, expected, data);
}

BOOST_AUTO_TEST_CASE(TestRandomization_TwoSweeps_WithSequences)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 6, 3, 1, 5, 9, 0, 4, 2, 7, 8, 3, 0, 8, 4, 7, 5, 2, 9, 1, 6 };

    for (int seqLength = 2; seqLength <= 10; seqLength++)
    {
        vector<float> expectedRandomized;
        vector<float> expectedNotRandomized;
        for (auto f : expected) {
            std::fill_n(back_inserter(expectedRandomized), seqLength, f);
        }

        for (int i = 0; i < 2 * data.size(); i++) {
            std::fill_n(back_inserter(expectedNotRandomized), seqLength, data[i % data.size()]);
        }

        auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data, seqLength);

        auto sweepSize = data.size() * seqLength;

        EpochConfiguration epochConfiguration;
        epochConfiguration.m_epochIndex = 0;

        TestRandomization(epochConfiguration, mockDeserializer, sweepSize, expectedRandomized, expectedNotRandomized, seqLength);
    }
}

BOOST_AUTO_TEST_CASE(TestRandomization_TwoSweeps_AllowToCrossSweepBoundary)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 6, 3, 1, 5, 9, 0, 4, 2, 7, 8, 3, 0, 8, 4, 7, 5, 2, 9, 1, 6 };

    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto sweepSize = data.size();
    data.reserve(2 * sweepSize);
    std::copy_n(data.begin(), sweepSize, std::back_inserter(data));

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_epochIndex = 0;
    epochConfiguration.m_allowMinibatchesToCrossSweepBoundaries = true;

    TestRandomization(epochConfiguration, mockDeserializer, sweepSize, expected, data);
}


BOOST_AUTO_TEST_CASE(TestRandomization_TwoSweeps_AllowToCrossSweepBoundary_WithSequences)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);

    vector<float> expected{ 6, 3, 1, 5, 9, 0, 4, 2, 7, 8, 3, 0, 8, 4, 7, 5, 2, 9, 1, 6 };

    for (int seqLength = 2; seqLength <= 10; seqLength++)
    {
        vector<float> expectedRandomized;
        vector<float> expectedNotRandomized;
        for (auto f : expected) {
            std::fill_n(back_inserter(expectedRandomized), seqLength, f);
        }

        for (int i = 0; i < 2 * data.size(); i++) {
            std::fill_n(back_inserter(expectedNotRandomized), seqLength, data[i % data.size()]);
        }

        auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data, seqLength);

        auto sweepSize = data.size() * seqLength;

        EpochConfiguration epochConfiguration;
        epochConfiguration.m_epochIndex = 0;
        epochConfiguration.m_allowMinibatchesToCrossSweepBoundaries = true;

        TestRandomization(epochConfiguration, mockDeserializer, sweepSize, expectedRandomized, expectedNotRandomized, seqLength);
    }
}

void BlockRandomizerOneEpochWithChunks1Test(bool prefetch)
{
    vector<float> data(10);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(5, 2, data);

    auto randomizer = make_shared<BlockRandomizer>(0, 4, mockDeserializer, prefetch, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected{ 8, 9, 1, 0, 6, 7, 2, 3, 4, 5 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1, 1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto& data2 = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data2.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data2.GetDataBuffer()));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i + 1));
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

    auto randomizer = make_shared<BlockRandomizer>(0, 18, mockDeserializer, prefetch, false);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected {
        18, 19, 7, 14, 6, 9, 8, 15, 5, 2,
        10, 13, 16, 17, 1, 4, 3, 12, 11, 0
    };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1, 1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto& data2 = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data2.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data2.GetDataBuffer()));
        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i + 1));
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(expected.begin(), expected.end(),
        actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(BlockRandomizerOneEpochWithChunks2)
{
    BlockRandomizerOneEpochWithChunks2Test(false);
    BlockRandomizerOneEpochWithChunks2Test(true);
}

void RandomizerChaosMonkeyTest(SequenceEnumerator& randomizer, size_t sweepSize, int seed)
{
    std::mt19937 rng(seed);
    boost::random::uniform_int_distribution<int> distr(1, 100);

    for (int t = 0; t < 100; t++)
    {
        EpochConfiguration epochConfiguration;
        epochConfiguration.m_numberOfWorkers = distr(rng);
        epochConfiguration.m_workerRank = distr(rng) % epochConfiguration.m_numberOfWorkers;

        epochConfiguration.m_minibatchSizeInSamples = 0; // don't care
        epochConfiguration.m_totalEpochSizeInSamples = sweepSize * distr(rng) / distr(rng);
        epochConfiguration.m_epochIndex = distr(rng);
        epochConfiguration.m_allowMinibatchesToCrossSweepBoundaries = (distr(rng) % 2 == 0);
        randomizer.StartEpoch(epochConfiguration);

        auto epochStart = epochConfiguration.m_epochIndex * epochConfiguration.m_totalEpochSizeInSamples;
        auto epochEnd = epochStart + epochConfiguration.m_totalEpochSizeInSamples;
        auto numSweeps = epochEnd / sweepSize - epochStart / sweepSize;

        auto sweepCount = 0;
        int samplesToGet = 0;
        for (;;)
        {
            samplesToGet = distr(rng);
            Sequences sequences = randomizer.GetNextSequences(samplesToGet, samplesToGet);

            if (sequences.m_endOfSweep)
                sweepCount++;

            // In case end of epoch/decimation/single sequence -> skip the mbSize check.
            if (!(sequences.m_data.empty() || sequences.m_data.size() == 1))
            {
                // Check that we do not exceed the minibatch size.
                size_t count = 0;
                for (const auto& sequence : sequences.m_data.front())
                {
                    count += sequence->m_numberOfSamples;
                }
                BOOST_REQUIRE_LE(count, samplesToGet);
            }

            if (sequences.m_endOfEpoch)
                break;
            
        }
        BOOST_REQUIRE(sweepCount == numSweeps);
    }
}

BOOST_AUTO_TEST_CASE(RandomizerChaosMonkey)
{
    const int sequenceLength = 3;
    const int numChunks = 100;
    const int numSequencesPerChunk = 10;
    const int windowSize = 18;
    vector<float> data(numChunks * numSequencesPerChunk);
    iota(data.begin(), data.end(), 0.0f);
    auto mockDeserializer = make_shared<MockDeserializer>(numChunks, numSequencesPerChunk, data, sequenceLength);
    BlockRandomizer blockRandomizerNoPrefetch(0, windowSize, mockDeserializer, false, false);
    BlockRandomizer blockRandomizerWithPrefetch(0, windowSize, mockDeserializer, true, false);
    NoRandomizer norandomizer(mockDeserializer);

    auto sweepSize = data.size() * sequenceLength;

    RandomizerChaosMonkeyTest(blockRandomizerNoPrefetch, sweepSize, 42);
    RandomizerChaosMonkeyTest(blockRandomizerWithPrefetch, sweepSize, 43);
    RandomizerChaosMonkeyTest(norandomizer, sweepSize, 44);
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
        true);

    EpochConfiguration epochConfiguration;
    epochConfiguration.m_numberOfWorkers = 1;
    epochConfiguration.m_workerRank = 0;
    epochConfiguration.m_minibatchSizeInSamples = 0;
    epochConfiguration.m_totalEpochSizeInSamples = data.size();
    epochConfiguration.m_epochIndex = 0;
    randomizer->StartEpoch(epochConfiguration);

    vector<float> expected { 6, 3, 1, 5, 9, 0, 4, 2, 7, 8 };
    BOOST_CHECK_EQUAL(data.size(), expected.size());
    vector<float> actual;
    for (int i = 0; i < data.size() + 1; i++)
    {
        Sequences sequences = randomizer->GetNextSequences(1, 1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < 10)
        {
            auto& data2 = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data2.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data2.GetDataBuffer()));

        }
        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i + 1));
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
        Sequences sequences = randomizer->GetNextSequences(1, 1);
        BOOST_CHECK_EQUAL(sequences.m_data.size(), 1 - (i / data.size()));
        if (i < data.size())
        {
            auto& data2 = reinterpret_cast<DenseSequenceData&>(*sequences.m_data[0][0]);
            BOOST_CHECK_EQUAL(data2.m_numberOfSamples, 1u);
            actual.push_back(*((float*)data2.GetDataBuffer()));
        }

        BOOST_CHECK_EQUAL(sequences.m_endOfEpoch, (data.size() <= i + 1));
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(data.begin(), data.end(),
                                  actual.begin(), actual.end());
}

BOOST_AUTO_TEST_CASE(CheckGetCurrentCursorForRandomizers)
{
    size_t chunkSizeInSamples = 10000;
    size_t sweepNumberOfSamples = 500000;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 5;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true, false);
    auto noRandomizer = make_shared<NoRandomizer>(deserializer, false);

    auto test = [](SequenceEnumeratorPtr r, size_t epochSize)
    {
        auto firstEpoch = ReadFullEpoch(r, epochSize, 0);
        auto firstCursor = r->GetState();
        BOOST_CHECK_EQUAL(firstCursor[g_minibatchSourcePosition], firstEpoch.size());

        auto secondEpoch = ReadFullEpoch(r, epochSize, 1);
        auto secondCursor = r->GetState();
        BOOST_CHECK_EQUAL(secondCursor[g_minibatchSourcePosition] - firstCursor[g_minibatchSourcePosition], secondEpoch.size());

        auto thirdEpoch = ReadFullEpoch(r, epochSize, 2);
        auto thirdCursor = r->GetState();
        BOOST_CHECK_EQUAL(thirdCursor[g_minibatchSourcePosition] - secondCursor[g_minibatchSourcePosition], thirdEpoch.size());

        auto anotherSecondEpoch = ReadFullEpoch(r, epochSize, 1);
        auto anotherSecondCursor = r->GetState();

        BOOST_CHECK_EQUAL(anotherSecondCursor[g_minibatchSourcePosition], secondCursor[g_minibatchSourcePosition]);
    };

    // Inside sweep
    size_t epochSize = 50000;
    test(blockRandomizer, epochSize);
    test(noRandomizer, epochSize);

    // Between sweeps
    epochSize = (size_t)(sweepNumberOfSamples / 1.5);
    test(blockRandomizer, epochSize);
    test(noRandomizer, epochSize);
}

BOOST_AUTO_TEST_CASE(DefaultCorpusDescriptor)
{
    const int seed = 13;
    std::mt19937 rng(seed);
    boost::random::uniform_int_distribution<int> distr(50, 60);

    string randomKey(10, (char)distr(rng));

    CorpusDescriptor corpus(false);
    BOOST_CHECK_EQUAL(false, corpus.IsHashingEnabled());
    BOOST_CHECK_EQUAL(false, corpus.IsNumericSequenceKeys());

    BOOST_CHECK_EQUAL(0, corpus.KeyToId(randomKey));
    BOOST_CHECK_EQUAL(1, corpus.KeyToId(""));
}

BOOST_AUTO_TEST_CASE(CorpusDescriptorHashing)
{
    auto hashVersion = CorpusDescriptor::s_hashVersion;
    BOOST_CHECK_EQUAL(1, hashVersion);
    BOOST_CHECK_EQUAL(1661589163364855789u, CorpusDescriptor(false, true).KeyToId("abcDEF_+123!890x.Y.Z@"));
}

BOOST_AUTO_TEST_CASE(NumericCorpusDescriptor)
{
    const int seed = 13;
    std::mt19937 rng(seed);
    boost::random::uniform_int_distribution<size_t> distr;

    CorpusDescriptor corpus(true);
    for (int i = 0; i < 10; ++i)
    {
        auto value = distr(rng);
        BOOST_CHECK_EQUAL(value, corpus.KeyToId(std::to_string(value)));
    }
    BOOST_CHECK_EXCEPTION(
        corpus.KeyToId("not a number"),
        std::exception, 
        [](const std::exception& e) { return e.what() == std::string("Invalid numeric sequence id 'not a number'"); });
}

BOOST_AUTO_TEST_CASE(LiteralCorpusDescriptor)
{
    const int seed = 13;
    std::mt19937 rng(seed);
    boost::random::uniform_int_distribution<int> distr(50, 60);

    string randomKey(10, (char)distr(rng));

    CorpusDescriptor corpus(false);
    BOOST_CHECK(100 != corpus.KeyToId("100"));
    BOOST_CHECK_NO_THROW(corpus.KeyToId("not a number"));
}

BOOST_AUTO_TEST_CASE(LiteralCorpusDescriptorWithHash)
{
    CorpusDescriptor corpus(false, true);

    // The constants are offline calculated hash values according to CorpusDescriptor::Hash.
    BOOST_CHECK_EQUAL(corpus.KeyToId("100"), 193358996);
    BOOST_CHECK_EQUAL(corpus.KeyToId("not"), 193419184);
}

BOOST_AUTO_TEST_CASE(NumericCorpusDescriptorWithHash)
{
    BOOST_REQUIRE_EXCEPTION(
        CorpusDescriptor corpus(true, true),
        runtime_error,
        [](const runtime_error& e)
        { return string("Hashing should not be used with numeric sequence keys.") == e.what(); });
}

BOOST_AUTO_TEST_CASE(CheckEpochBoundarySingleWorker)
{
    size_t chunkSizeInSamples = 1000;
    size_t sweepNumberOfSamples = 15000;
    uint32_t maxSequenceLength = 1;
    size_t randomizationWindow = chunkSizeInSamples * 5;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    auto test = [](SequenceEnumeratorPtr underTest)
    {
        size_t epochSize = 128 * 3 + 63;

        EpochConfiguration config;
        config.m_numberOfWorkers = 1;
        config.m_workerRank = 0;
        config.m_minibatchSizeInSamples = 128;
        config.m_totalEpochSizeInSamples = epochSize;
        config.m_epochIndex = 0;
        underTest->StartEpoch(config);

        Sequences s;
        size_t numberOfSamples = 0;
        bool globalsMoreThanLocals = false;
        do
        {
            s = underTest->GetNextSequences(globalsMoreThanLocals ? 256 : 128, globalsMoreThanLocals ? 128 : 256);
            globalsMoreThanLocals = !globalsMoreThanLocals;
            for (const auto& seq : s.m_data.front())
                numberOfSamples += seq->m_numberOfSamples;
        }
        while (!s.m_endOfEpoch);

        // Check the last minibatch is 63.
        BOOST_CHECK_EQUAL(s.m_data.front().size(), 63);

        // Check total number.
        BOOST_CHECK_EQUAL(numberOfSamples, epochSize);
    };

    auto underTestBlock = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
    auto underTestNo = make_shared<NoRandomizer>(deserializer);

    test(underTestBlock);
    test(underTestNo);
}

// Make sure we do not cut the minibatches at the end of the epoch such that they
// contain only a single sequence. For example, with an input data consisting of 3-sample
// sequences, minibatch size set to 90 and the epoch size to 100, the source should return
// two minibatches 90 and 12 samples in each and not three minibatches (as it used to) 
// with 90, 9 and 3 samples. In other words, the maximum number of minibatches in an epoch
// should be <= ceil(epoch size / expected minibatch size)
BOOST_AUTO_TEST_CASE(CheckNoDegenerateMinibatches)
{
    struct Parameters
    {
        size_t numSequences;
        size_t sequenceLength;
        size_t epochSize;
        size_t minibatchSize;
        size_t epochIndex;
    };

    vector<Parameters> params = { {50, 3, 100, 90, 0} };

    std::mt19937 rng(77);
    while (params.size() < 100)
    {
        Parameters p;
        p.numSequences = rng() % 100 + 1;
        p.sequenceLength = rng() % 100 + 1;
        p.minibatchSize = (rng() % 10) * p.sequenceLength + 1;
        p.epochSize = (rng() % 20) * p.sequenceLength + 1;
        p.epochIndex = rng() % 10;
        params.push_back(p);
    }


    for (const auto& p : params)
    {
        vector<float> data(p.numSequences);
        iota(data.begin(), data.end(), 0.0f);

        auto mockDeserializer = make_shared<MockDeserializer>(1, p.numSequences, data, uint32_t(p.sequenceLength));

        auto test = [&p](SequenceEnumeratorPtr underTest)
        {
            size_t epochSize = p.epochSize;


            EpochConfiguration config;
            config.m_numberOfWorkers = 1;
            config.m_workerRank = 0;
            config.m_minibatchSizeInSamples = p.minibatchSize;
            config.m_totalEpochSizeInSamples = epochSize;
            config.m_epochIndex = p.epochIndex;
            config.m_allowMinibatchesToCrossSweepBoundaries = true;
            underTest->StartEpoch(config);

            // if max expected minibatch size must be a multiple of sequence size 
            auto maxMBSize = (p.minibatchSize / p.sequenceLength) * p.sequenceLength;
            if (maxMBSize == 0)
                maxMBSize = p.sequenceLength;

            Sequences s;
            size_t numberOfSamples = 0;
            size_t numberOfMinibatches = 0;
            do
            {
                s = underTest->GetNextSequences(p.minibatchSize, p.minibatchSize);
                if (!s.m_data.empty())
                    for (const auto& seq : s.m_data.front())
                        numberOfSamples += seq->m_numberOfSamples;

                numberOfMinibatches++;
            } while (!s.m_endOfEpoch);


            auto epochStart = p.epochIndex * epochSize;
            auto epochEnd = epochStart + epochSize;
            auto startingOffset = (size_t)ceil(epochStart * 1.0 / p.sequenceLength)* p.sequenceLength;

            if (startingOffset >= epochEnd)
            {
                BOOST_TEST((s.m_data.empty() && numberOfMinibatches == 1 && numberOfSamples == 0));
                return;
            }

            auto actualEpochSize = epochEnd - startingOffset;
            // make sure that the last minibatch contains more than a single sequence:
            auto lastMBSize = actualEpochSize % maxMBSize;

            auto numSequencesInTheLastMB = s.m_data[0].size();

            if (lastMBSize == 0)
                // epoch size is a multiple of maxMBSize => it's a multiple of sequence length
                BOOST_TEST(((maxMBSize / p.sequenceLength) == numSequencesInTheLastMB));
            else
            {
                // last sequence overlaps the epoch boundary. 
                BOOST_TEST((ceil(lastMBSize*1.0 / p.sequenceLength) == numSequencesInTheLastMB));
            }

            BOOST_TEST((numberOfSamples <= epochSize || (numberOfSamples - epochSize) < p.sequenceLength));
            BOOST_TEST((numberOfMinibatches <= ceil(epochSize* 1.0 / maxMBSize)));
        };

        auto underTestBlock = make_shared<BlockRandomizer>(0, size_t(-1), mockDeserializer, true);
        auto underTestNo = make_shared<NoRandomizer>(mockDeserializer);

        test(underTestBlock);
        test(underTestNo);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(PackerTests)

typedef std::multimap<size_t, SequentialDeserializer::MockSequenceInfo> CorpusSubset;

// Runs single worker till reach end of epoch.
void RunSingleWorker(
    PackerPtr packerUnderTest,
    const std::map<size_t, SequentialDeserializer::MockSequenceInfo>& corpus,
    std::multimap<size_t, SequentialDeserializer::MockSequenceInfo>& subset,
    size_t expectedMinibatchSize,
    bool strict)
{
    bool shouldContinue = true;
    size_t counter = 0; // for debugging purposes.
    while (shouldContinue)
    {
        auto minibatch = packerUnderTest->ReadMinibatch();

        if (minibatch.m_endOfEpoch)
            shouldContinue = false;

        if (minibatch.m_data.size() == 0)
            continue;

        auto layout = minibatch.m_data.front()->m_layout;
        size_t numParallelSequences = layout->GetNumParallelSequences();
        if (numParallelSequences == 0)
        {
            continue;
        }

        if (strict)
        {
            BOOST_REQUIRE_EQUAL(layout->GetActualNumSamples() <= expectedMinibatchSize, true);
        }
        else
        {
            if (layout->GetActualNumSamples() > expectedMinibatchSize)
                BOOST_REQUIRE_EQUAL(layout->GetAllSequences().size(), 1);
        }

        auto data = (float*)minibatch.m_data.front()->m_data;
        auto sequences = layout->GetAllSequences();
        for (const auto& s : sequences)
        {
            if (s.seqId == GAP_SEQUENCE_ID)
                continue;

            size_t sequenceFirstValueIndex = numParallelSequences * s.tBegin + s.s;

            float sequenceValue = data[sequenceFirstValueIndex];
            size_t sequenceLength = s.GetNumTimeSteps();

            auto key = (size_t)sequenceValue;
            auto correspondingS = corpus.find(key);
            BOOST_REQUIRE_EQUAL(correspondingS != corpus.end(), true);
            BOOST_REQUIRE_EQUAL(correspondingS->second.startingValue, sequenceValue);
            BOOST_REQUIRE_EQUAL(correspondingS->second.size, sequenceLength);

            subset.insert(make_pair(key, correspondingS->second));
        }

        counter++; // For debugging if something goes wrong...
    }
    UNUSED(counter);
}

void RunAllWorkers(
    size_t numWorkers,
    const std::map<size_t, SequentialDeserializer::MockSequenceInfo>& corpus,
    std::map<pair<size_t, size_t>, CorpusSubset>& result,
    PackerPtr packer,
    SequenceEnumeratorPtr randomizer,
    size_t numEpochs,
    size_t epochSize,
    size_t minibatchSize,
    bool strictMinibatchSizeCheck)
{
    for (size_t rank = 0; rank < numWorkers; ++rank)
    {
        for (size_t epoch = 0; epoch < numEpochs; ++epoch)
        {
            CorpusSubset subset;
            EpochConfiguration config;
            config.m_minibatchSizeInSamples = minibatchSize;
            config.m_truncationSize = 0;
            config.m_epochIndex = epoch;
            config.m_totalEpochSizeInSamples = epochSize;
            config.m_numberOfWorkers = numWorkers;
            config.m_workerRank = rank;

            packer->SetConfiguration(config, std::vector<MemoryProviderPtr> { std::make_shared<HeapMemoryProvider>() });
            randomizer->StartEpoch(config);

            bool shouldAddOneMinibatchSample = config.m_minibatchSizeInSamples % numWorkers > rank;
            RunSingleWorker(packer, corpus, subset,
                config.m_minibatchSizeInSamples / numWorkers + (shouldAddOneMinibatchSample ? 1 : 0),
                strictMinibatchSizeCheck);

            result.insert(make_pair(make_pair(rank, epoch), std::move(subset)));
        }
    }
}

// Helper functions

template<class K, class V>
std::multiset<V> ToSet(const multimap<K, V>& multiMap)
{
    std::multiset<V> result;
    for (const auto& v : multiMap)
    {
        result.insert(v.second);
    }

    return result;
}

std::set<size_t> GetWorkerChunks(const std::map<pair<size_t, size_t>, CorpusSubset>& corpus, size_t rank)
{
    std::set<size_t> result;
    for (const auto& c : corpus)
    {
        if (c.first.first != rank)
            continue;

        for (auto const& s : c.second)
        {
            result.insert(s.second.chunkId);
        }
    }

    return result;
}

size_t GetEpochSamples(const std::map<pair<size_t, size_t>, CorpusSubset>& corpus, size_t epoch)
{
    size_t sampleCount = 0;
    for (const auto& c : corpus)
    {
        if (c.first.second != epoch)
            continue;

        for (auto const& s : c.second)
        {
            sampleCount += s.second.size;
        }
    }

    return sampleCount;
}

CorpusSubset GetCorpus(const std::map<pair<size_t, size_t>, CorpusSubset>& corpus)
{
    CorpusSubset result;
    for (const auto& c : corpus)
    {
        for (auto const& s : c.second)
        {
            result.insert(make_pair(s.first, s.second));
        }
    }

    return result;
}


// Runs a packer on a data set for different number of workers.
void CheckPackerOnDataSet(
    PackerPtr packer,
    SequenceEnumeratorPtr randomizer,
    SequentialDeserializerPtr deserializer,
    size_t numEpochs,
    size_t epochSize,
    size_t numSweeps,
    size_t sweepSize,
    size_t minibatchSize,
    bool strictMinibatchSizeCheck)
{
    BOOST_REQUIRE_EQUAL(numEpochs * epochSize, sweepSize * numSweeps);

    std::vector<size_t> numberOfSamplesInEpoch;
    numberOfSamplesInEpoch.resize(numEpochs);

    for (auto numWorkers : { 1, 8, 16 })
    {
        // numWorkers, rank, epoch -> subset of data.
        std::map<pair<size_t, size_t>, CorpusSubset> allData;

        RunAllWorkers(
            numWorkers,
            deserializer->Corpus(),
            allData,
            packer,
            randomizer,
            numEpochs,
            epochSize,
            minibatchSize,
            strictMinibatchSizeCheck);

        auto actual = ToSet(GetCorpus(allData));
        auto singleSweep = ToSet(CorpusSubset(deserializer->Corpus().begin(), deserializer->Corpus().end()));
        auto expected = singleSweep;
        for (size_t i = 0; i < numSweeps - 1; ++i)
        {
            expected.insert(singleSweep.begin(), singleSweep.end());
        }

        // Check that the epoch size matches no matter how many workers we run.
        for (size_t e = 0; e < numEpochs; e++)
        {
            size_t sampleCount = GetEpochSamples(allData, e);
            if (numWorkers == 1)
                numberOfSamplesInEpoch[e] = sampleCount;
            else
                BOOST_REQUIRE_EQUAL(numberOfSamplesInEpoch[e], sampleCount);
        }

        BOOST_REQUIRE_EQUAL_COLLECTIONS(
            expected.begin(), expected.end(),
            actual.begin(), actual.end());
    }
}


// Runs a packer on a single sweep for different number of workers.
void CheckPackerOnSweep(
    PackerPtr packer,
    SequenceEnumeratorPtr randomizer,
    SequentialDeserializerPtr deserializer,
    size_t numEpochs,
    size_t minibatchSize,
    bool strictMinibatchSizeCheck,
    bool performWorkerChunkCheck)
{
    std::vector<size_t> numberOfSamplesInEpoch;
    numberOfSamplesInEpoch.resize(numEpochs);

    for (auto numWorkers : { 1, 8, 16 })
    {
        // numWorkers, rank, epoch -> subset of data.
        std::map<pair<size_t, size_t>, CorpusSubset> allData;

        RunAllWorkers(
            numWorkers,
            deserializer->Corpus(),
            allData,
            packer,
            randomizer,
            numEpochs,
            deserializer->TotalSize() / numEpochs,
            minibatchSize,
            strictMinibatchSizeCheck);

        // Let's check that chunks are not shared between all workers 
        // and that total number of chunks among all workers in all the epochs 
        // == number of chunks in a sweep.
        std::set<size_t> totalChunks;
        for (size_t rank = 0; rank < numWorkers; ++rank)
        {
            auto workerChunks = GetWorkerChunks(allData, rank);

            if (performWorkerChunkCheck) // We know expected number of chunks.
            {
                bool shouldAddOne = deserializer->ChunkInfos().size() % numWorkers > rank;
                size_t expectedNumberOfChunks = deserializer->ChunkInfos().size() / numWorkers + (shouldAddOne ? 1 : 0);
                BOOST_REQUIRE_EQUAL(workerChunks.size(), expectedNumberOfChunks);

                std::set<size_t> intersect;
                set_intersection(totalChunks.begin(), totalChunks.end(), workerChunks.begin(), workerChunks.end(),
                    std::inserter(intersect, intersect.begin()));
                BOOST_REQUIRE_EQUAL(intersect.empty(), true);
            }

            totalChunks.insert(workerChunks.begin(), workerChunks.end());
        }

        // Check that the epoch size matches no matter how many workers we run.
        for (size_t e = 0; e < numEpochs; e++)
        {
            size_t sampleCount = GetEpochSamples(allData, e);
            if (numWorkers == 1)
                numberOfSamplesInEpoch[e] = sampleCount;
            else
                BOOST_REQUIRE_EQUAL(numberOfSamplesInEpoch[e], sampleCount);
        }

        BOOST_REQUIRE_EQUAL(totalChunks.size(), deserializer->ChunkInfos().size());

        auto actual = ToSet(GetCorpus(allData));
        auto expected = ToSet(CorpusSubset(deserializer->Corpus().begin(), deserializer->Corpus().end()));

        BOOST_REQUIRE_EQUAL_COLLECTIONS(
            expected.begin(), expected.end(),
            actual.begin(), actual.end());
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerBigChunksWithFrames1Sweep)
{
    size_t chunkSizeInSamples = 998;
    size_t sweepNumberOfSamples = 21335;
    uint32_t maxSequenceLength = 1;
    size_t randomizationWindow = chunkSizeInSamples * 5;

    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 64, true, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 5, 64, true, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 33, true, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 5, 31, true, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 64, true, false);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 5, 64, true, false);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 33, true, false);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 5, 31, true, false);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerSmallChunksWithFrames1Sweep)
{
    size_t chunkSizeInSamples = 1;
    size_t sweepNumberOfSamples = 1332;
    uint32_t maxSequenceLength = 1;
    size_t randomizationWindow = 1;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 64, true, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 3, 64, true, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 33, true, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 3, 31, true, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 64, true, true);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 3, 64, true, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 33, true, true);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 3, 31, true, true);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerBigChunksWithSequences1Sweep)
{
    size_t chunkSizeInSamples = 998;
    size_t sweepNumberOfSamples = 21335;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 5;

    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 64, false, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 5, 64, false, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 33, false, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 5, 31, false, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 64, false, false);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 5, 64, false, false);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 33, false, false);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 5, 31, false, false);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerSmallChunksWithSequences1Sweep)
{
    size_t chunkSizeInSamples = 1;
    size_t sweepNumberOfSamples = 1332;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = 1;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 64, false, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 3, 64, false, true);

        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 1, 33, false, true);
        CheckPackerOnSweep(packer, blockRandomizer, deserializer, 3, 31, false, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 64, false, true);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 3, 64, false, true);

        CheckPackerOnSweep(packer, noRandomizer, deserializer, 1, 33, false, true);
        CheckPackerOnSweep(packer, noRandomizer, deserializer, 3, 31, false, true);
    }
}

////
////
//// On two sweeps
////
////

BOOST_AUTO_TEST_CASE(SequencePackerBigChunksWithFrames)
{
    size_t chunkSizeInSamples = 998;
    size_t sweepNumberOfSamples = 21335;
    uint32_t maxSequenceLength = 1;
    size_t randomizationWindow = chunkSizeInSamples * 5;

    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, true);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 64, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, true);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 31, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, true);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 64, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, true);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 31, true);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerSmallChunksWithFrames)
{
    size_t chunkSizeInSamples = 1;
    size_t sweepNumberOfSamples = 1332;
    uint32_t maxSequenceLength = 1;
    size_t randomizationWindow = 1;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, true);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 64, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, true);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 31, true);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, true);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 64, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, true);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 31, true);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerBigChunksWithSequences)
{
    size_t chunkSizeInSamples = 998;
    size_t sweepNumberOfSamples = 21335;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = chunkSizeInSamples * 5;

    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, false);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 64, false);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, false);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 31, false);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, false);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 64, false);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, false);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 5, sweepNumberOfSamples * 2 / 5, 2, sweepNumberOfSamples, 31, false);
    }
}

BOOST_AUTO_TEST_CASE(SequencePackerSmallChunksWithSequences)
{
    size_t chunkSizeInSamples = 1;
    size_t sweepNumberOfSamples = 1332;
    uint32_t maxSequenceLength = 300;
    size_t randomizationWindow = 1;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);

    {
        auto blockRandomizer = make_shared<BlockRandomizer>(0, randomizationWindow, deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(blockRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, false);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 64, false);

        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, false);
        CheckPackerOnDataSet(packer, blockRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 31, false);
    }

    {
        auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
        PackerPtr packer = std::make_shared<SequencePacker>(noRandomizer, deserializer->StreamInfos(), 1, true);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 64, false);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 64, false);

        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 1, sweepNumberOfSamples * 2, 2, sweepNumberOfSamples, 33, false);
        CheckPackerOnDataSet(packer, noRandomizer, deserializer, 3, sweepNumberOfSamples * 2 / 3, 2, sweepNumberOfSamples, 31, false);
    }
}

BOOST_AUTO_TEST_CASE(TestTruncatedBpttPacker)
{
    size_t chunkSizeInSamples = 100;
    size_t sweepNumberOfSamples = 100;
    uint32_t maxSequenceLength = 10;
    auto deserializer = make_shared<SequentialDeserializer>(0, chunkSizeInSamples, sweepNumberOfSamples, maxSequenceLength);
    auto noRandomizer = make_shared<NoRandomizer>(deserializer, true);
    auto packer = std::make_shared<TruncatedBPTTPacker>(noRandomizer, deserializer->StreamInfos());
    
    EpochConfiguration config;
    config.m_allowMinibatchesToCrossSweepBoundaries = true;
    config.m_numberOfWorkers = 1;
    config.m_minibatchSizeInSamples = 30;
    config.m_truncationSize = 3;
    config.m_totalEpochSizeInSweeps = 3;
    config.m_epochIndex = 0;

    noRandomizer->StartEpoch(config);
    packer->SetConfiguration(config, 
        std::vector<MemoryProviderPtr> { std::make_shared<HeapMemoryProvider>() });

    size_t sampleCount = 0;
    while (true) {
        auto mb = packer->ReadMinibatch();
        
        if (mb.m_endOfSweep)
            break;

        BOOST_ASSERT(sampleCount < sweepNumberOfSamples * 10);

        sampleCount += mb.m_data[0]->m_layout->GetActualNumSamples();
    }

    std::map<std::wstring, size_t> state;
    state[g_minibatchSourcePosition] = sweepNumberOfSamples;
    noRandomizer->SetState(state);
    packer->Reset();

    auto mb = packer->ReadMinibatch();

    BOOST_TEST(!mb.m_endOfSweep);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
