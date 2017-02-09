//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

// Mock communicator to simulate MPI run
class MockCommunicator : public DistributedCommunicator
{
private:
    std::unordered_set<DistributedWorkerDescriptor> m_workers;
    DistributedWorkerDescriptor m_self;

public:
    virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const override
    {
        return m_workers;
    }

    virtual const DistributedWorkerDescriptor& CurrentWorker() const override
    {
        return m_self;
    }

    virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>&) const override
    {
        return nullptr;
    }

    virtual void Concatenate(
        const std::vector<ValuePtr>&,
        std::vector<ValuePtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Concatenate(
        const std::vector<NDArrayViewPtr>&,
        std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Gather(
        const Dictionary&,
        std::vector<DictionaryPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void AggregateInPlace(
        const std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}

    virtual void Aggregate(
        const std::vector<NDArrayViewPtr>&,
        std::vector<NDArrayViewPtr>&,
        const std::unordered_set<DistributedWorkerDescriptor>&) override
    {}
    
    virtual void Barrier() override
    {}

    MockCommunicator(size_t numWorkers)
    {
        for (size_t i = 0; i < numWorkers; i++)
        {
            DistributedWorkerDescriptor desc;
            desc.m_hostId = L"MockCommunicator";
            desc.m_globalRank = i;

            m_workers.insert(desc);
        }
        MockRank(0);
    }

    void MockRank(size_t rank)
    {
        m_self.m_hostId = L"MockCommunicator";
        m_self.m_globalRank = rank;
    }
};

MinibatchSourcePtr CreateTextFormatMinibatchSource(const std::wstring& dataFilePath, const std::vector<StreamConfiguration>& streamConfigs, size_t epochSize, bool randomize, size_t chunkSizeInBytes)
{
    ::CNTK::Dictionary minibatchSourceConfiguration;
    minibatchSourceConfiguration[L"epochSize"] = epochSize;

    if (randomize)
        minibatchSourceConfiguration[L"randomize"] = true;
    else
        minibatchSourceConfiguration[L"randomize"] = false;

    ::CNTK::Dictionary deserializerConfiguration;
    deserializerConfiguration[L"type"] = L"CNTKTextFormatDeserializer";
    deserializerConfiguration[L"file"] = dataFilePath;

    ::CNTK::Dictionary inputStreamsConfig;
    for (auto streamConfig : streamConfigs)
    {
        std::wstring streamName = streamConfig.m_streamName;
        size_t streamDim = streamConfig.m_dim;
        bool isSparse = streamConfig.m_isSparse;
        std::wstring streamAlias = streamConfig.m_streamAlias;

        ::CNTK::Dictionary inputStreamConfig;
        inputStreamConfig[L"dim"] = streamDim;
        inputStreamConfig[L"format"] = isSparse ? L"sparse" : L"dense";
        if (!streamAlias.empty())
            inputStreamConfig[L"alias"] = streamAlias;

        inputStreamsConfig[streamName] = inputStreamConfig;
    }

    deserializerConfiguration[L"input"] = inputStreamsConfig;
    deserializerConfiguration[L"chunkSizeInBytes"] = chunkSizeInBytes;
    minibatchSourceConfiguration[L"deserializers"] = std::vector<::CNTK::DictionaryValue>({ deserializerConfiguration });
    return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
}

void TestMinibatchSourceWarmStart(size_t minibatchSize, size_t warmStartSamples, bool randomize, size_t chunkSizeInBytes, bool expectNoData = false)
{
    // TODO: Currently this test is based on the number of samples.
    // We should switch to the real data instead.

    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    auto featureStreamName = L"features";
    auto labelsStreamName = L"labels";

    const size_t numberOfSamplesInSweep = 10000;

    // Let's create two workers.
    auto minibatchSource = CreateTextFormatMinibatchSource(
        L"SimpleDataTrain_cntk_text.txt",
        { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },
        numberOfSamplesInSweep,
        randomize,
        chunkSizeInBytes);

    auto featureStreamInfo = minibatchSource->StreamInfo(featureStreamName);
    auto labelStreamInfo = minibatchSource->StreamInfo(labelsStreamName);

    auto minibatchSource2 = CreateTextFormatMinibatchSource(
        L"SimpleDataTrain_cntk_text.txt",
        { { featureStreamName, inputDim }, { labelsStreamName, numOutputClasses } },
        numberOfSamplesInSweep,
        randomize,
        chunkSizeInBytes);

    size_t totalSamples = 0;
    bool hasData = true;
    while (hasData)
    {
        if (totalSamples < warmStartSamples)
        {
            auto minibatchData = minibatchSource->GetNextMinibatch(0, minibatchSize, 1, 0);
            auto minibatchData2 = minibatchSource2->GetNextMinibatch(0, minibatchSize, 1, 0);

            if (minibatchData[featureStreamInfo].numberOfSamples != minibatchData2[featureStreamInfo].numberOfSamples)
                ReportFailure("Data does not match, reads are not deterministic!!!");

            // Because they are supposed to read the same data - adding it only once.
            totalSamples += minibatchData[featureStreamInfo].numberOfSamples;
        }
        else
        {
            // We are in distributed mode, the sum should be equal to the minibatch size
            // or less at the end of the sweep.
            auto minibatchData = minibatchSource->GetNextMinibatch(0, minibatchSize, 2, 0);
            auto minibatchData2 = minibatchSource2->GetNextMinibatch(0, minibatchSize, 2, 1);

            hasData = !minibatchData.empty() || !minibatchData2.empty();
            if (!hasData)
                break;

            // Update the counter
            size_t accumulative = 0;
            if (!minibatchData.empty())
                accumulative += minibatchData[featureStreamInfo].numberOfSamples;
            if (!minibatchData2.empty())
                accumulative += minibatchData2[featureStreamInfo].numberOfSamples;

            totalSamples += accumulative;

            if (expectNoData) // second worker does not have any data.
            {
                if (minibatchData[featureStreamInfo].numberOfSamples != minibatchSize/2 && totalSamples != numberOfSamplesInSweep)
                    ReportFailure("TestMinibatchSourceWarmStart failed because data did not match."
                                  "Expected minibatch size '%d', acutal '%d'. Total number of sample '%d', sweep '%d'.",
                                  (int)minibatchSize,
                                  (int)minibatchData[featureStreamInfo].numberOfSamples,
                                  (int)totalSamples,
                                  (int)numberOfSamplesInSweep);
            }
            else
            {
                if (accumulative != minibatchSize &&
                    minibatchData[featureStreamInfo].numberOfSamples != minibatchSize / 2 &&
                    minibatchData2[featureStreamInfo].numberOfSamples != minibatchSize / 2 &&
                    totalSamples != numberOfSamplesInSweep)
                    ReportFailure("TestMinibatchSourceWarmStart failed because data did not match."
                        "Expected minibatch size '%d', acutal '%d'. Total number of sample '%d', sweep '%d'.",
                        (int)minibatchSize,
                        (int)accumulative,
                        (int)totalSamples,
                        (int)numberOfSamplesInSweep);
            }
        }
    }

    if (totalSamples != numberOfSamplesInSweep)
        ReportFailure("Expected sweep number '%d' did not match the actual '%d'.",
                      (int)numberOfSamplesInSweep,
                      (int)totalSamples);
}

void TestEndOfSweepFlag(size_t maxSamples, size_t mbSize, bool randomize)
{
    const size_t sweepSize = 603;
    auto ctfInput = L"SimpleDataTest_cntk_text.txt";
    std::vector<StreamConfiguration> streamConfig{ { L"features", 2 } };
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    auto src = TextFormatMinibatchSource(ctfInput, streamConfig, maxSamples, randomize);

    maxSamples = (maxSamples == MinibatchSource::FullDataSweep) ? sweepSize : maxSamples;

    bool reachedEndOfEpoch = false;
    size_t sampleCount = 0;

    while (sampleCount < maxSamples)
    {
        auto& dataMap = src->GetNextMinibatch(mbSize, cpuDevice);

        if (dataMap.size() != streamConfig.size())
        {
            ReportFailure("TestThatEndOfSweepFlagIsSetCorrectly failed: "
                          "unexpected number of streams in the minibatch (%zu).", dataMap.size());
        }

        for (auto& streamData : dataMap)
        {
            auto numSamplesInMinibatch = streamData.second.numberOfSamples;
            bool expectedEndOfSweep = ((sampleCount + numSamplesInMinibatch) % sweepSize) == 0;
            expectedEndOfSweep |= ((sampleCount) / sweepSize) < ((sampleCount + numSamplesInMinibatch) / sweepSize);

            reachedEndOfEpoch = (sampleCount + mbSize >= maxSamples);
            size_t expectedNumSamples = reachedEndOfEpoch ? (maxSamples - sampleCount) : mbSize;

            if (streamData.second.sweepEnd != expectedEndOfSweep)
            {
                ReportFailure("TestThatEndOfSweepFlagIsSetCorrectly failed: end of sweep flag is not set.");
            }
            if (streamData.second.numberOfSamples != expectedNumSamples)
            {
                ReportFailure("TestThatEndOfSweepFlagIsSetCorrectly failed: "
                              "unexpected number of samples in the minibatch (%zu).", streamData.second.numberOfSamples);
            }
            if (streamData.second.numberOfSequences != expectedNumSamples)
            {
                ReportFailure("TestThatEndOfSweepFlagIsSetCorrectly failed: "
                              "unexpected number of sequences in the minibatch (%zu).", streamData.second.numberOfSequences);
            }
        }

        sampleCount += mbSize;
    }

    auto& emptyDataMap = src->GetNextMinibatch(mbSize, cpuDevice);
    assert(emptyDataMap.empty());
}

void TestThatEndOfSweepFlagIsSetCorrectly()
{
    for (auto randomize : { false, true })
    {
        TestEndOfSweepFlag(MinibatchSource::FullDataSweep, 603, randomize);
        TestEndOfSweepFlag(MinibatchSource::FullDataSweep, 1000, randomize);
        TestEndOfSweepFlag(MinibatchSource::FullDataSweep, 100, randomize);

        TestEndOfSweepFlag(100, 30, randomize);
        TestEndOfSweepFlag(2000, 500, randomize);
        TestEndOfSweepFlag(2412, 301, randomize);
    }
}

BOOST_AUTO_TEST_SUITE(MinibatchSourceSuite)

BOOST_AUTO_TEST_CASE(EndOfSweepFlagIsSetCorrectly)
{
    TestThatEndOfSweepFlagIsSetCorrectly();
}

BOOST_AUTO_TEST_CASE(NoRandomizedMinibatchSourceWarmStart)
{
    TestMinibatchSourceWarmStart(64, 128, false, 1024);
    TestMinibatchSourceWarmStart(64, 0, false, 1024);
    TestMinibatchSourceWarmStart(64, 100, false, 1024);
}

BOOST_AUTO_TEST_CASE(NoRandomizedMinibatchSourceWithSingleChunk)
{
    size_t chunk32MB = 1024 * 1024 * 32;
    TestMinibatchSourceWarmStart(64, 128, false, chunk32MB);
    TestMinibatchSourceWarmStart(64, 0, false, chunk32MB);
    TestMinibatchSourceWarmStart(64, 100, false, chunk32MB);
}

BOOST_AUTO_TEST_CASE(RandomizedMinibatchSourceWithSmallChunks)
{
    TestMinibatchSourceWarmStart(64, 0, true, 1024);
    TestMinibatchSourceWarmStart(64, 128, true, 1024);
}

BOOST_AUTO_TEST_CASE(RandomizedMinibatchSourceWithNoData)
{
    size_t chunk32MB = 1024 * 1024 * 32;
    bool expectNoData = true;
    TestMinibatchSourceWarmStart(64, 0, true, chunk32MB, expectNoData);
    TestMinibatchSourceWarmStart(64, 128, true, chunk32MB, expectNoData);
}

BOOST_AUTO_TEST_SUITE_END()

}}
