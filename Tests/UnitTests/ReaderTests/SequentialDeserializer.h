//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <random>
#include <boost/random/uniform_int_distribution.hpp>
#include "NoRandomizer.h"
#include "DataDeserializer.h"
#include "BlockRandomizer.h"
#include <thread>
#include <chrono>

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

    struct MockDenseSequenceData : DenseSequenceData
    {
        const void* GetDataBuffer() override
        {
            return m_data;
        }

        void* m_data;
    };

    // A mock deserializer that produces N sequential samples
    // with value from 0 .. N-1

    class SequentialDeserializer : public IDataDeserializer
    {
    public:
        struct SequenceInfo
        {
            size_t id;
            size_t size;
            size_t chunkId;
            float startingValue;
        };

        struct SequentialChunk : Chunk
        {
            std::vector<std::vector<float>> m_data;
            size_t m_sizeInSamples;
            size_t m_sizeInSequences;
            const TensorShapePtr m_sampleLayout;

            SequentialChunk(size_t approxSize) : m_sizeInSamples{ 0 }, m_sizeInSequences{ 0 }, m_sampleLayout(std::make_shared<TensorShape>(1))
            {
                m_data.reserve(approxSize);
            }

            SequentialChunk(const SequentialChunk& other)
            {
                m_data = other.m_data;
                m_sizeInSamples = other.m_sizeInSamples;
                m_sizeInSequences = other.m_sizeInSequences;
            }

            void AddSequence(const std::vector<float>&& data)
            {
                m_sizeInSamples += data.size();
                m_sizeInSequences++;
                m_data.push_back(std::move(data));
            }

            size_t SizeInSamples() const
            {
                return m_sizeInSamples;
            }

            size_t SizeInSequences() const
            {
                return m_sizeInSequences;
            }

            void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
            {
                const auto& data = m_data[sequenceId];

                auto s = make_shared<MockDenseSequenceData>();
                s->m_data = (void*)&data[0];
                s->m_numberOfSamples = (uint32_t)data.size();
                s->m_sampleLayout = m_sampleLayout;
                result.push_back(s);
            }
        };
        typedef std::shared_ptr<SequentialChunk> SequentialChunkPtr;

        SequentialDeserializer(
            size_t seed,
            size_t chunkSizeInSamples,
            size_t sweepNumberOfSamples,
            uint32_t maxSequenceLength)
            : m_sampleLayout(make_shared<TensorShape>(1))
        {
            std::mt19937_64 engine(seed);
            boost::random::uniform_int_distribution<int> length(1, maxSequenceLength);

            // Let's generate our data.
            float currentValue = 0;
            size_t currentNumberOfSamples = 0;
            SequentialChunkPtr currentChunk = std::make_shared<SequentialChunk>(chunkSizeInSamples);
            m_chunks.reserve(sweepNumberOfSamples / chunkSizeInSamples + 1);
            while (currentNumberOfSamples < sweepNumberOfSamples)
            {
                size_t sequenceLength = 0;
                if (sweepNumberOfSamples - currentNumberOfSamples < maxSequenceLength) // last one?
                {
                    sequenceLength = sweepNumberOfSamples - currentNumberOfSamples;
                }
                else
                {
                    // From time to time we want small sequences.
                    sequenceLength = currentNumberOfSamples % 1000 == 0 ? 1 : length(engine);
                }

                std::vector<float> sequenceData;
                for (size_t i = 0; i < sequenceLength; ++i)
                {
                    sequenceData.push_back(currentValue++);
                }

                if (currentChunk->SizeInSamples() >= chunkSizeInSamples)
                {
                    m_chunks.push_back(currentChunk);
                    currentChunk = std::make_shared<SequentialChunk>(chunkSizeInSamples);
                }

                // Let's record information about the sequence.
                SequenceInfo info
                {
                    currentChunk->m_data.size(),
                    sequenceData.size(),
                    m_chunks.size(),
                    sequenceData.front()
                };
                m_sequenceInfos[(size_t)info.startingValue] = info;

                currentChunk->AddSequence(std::move(sequenceData));
                currentNumberOfSamples += sequenceLength;
            }

            if (currentChunk->SizeInSamples() != 0)
            {
                m_chunks.push_back(currentChunk);
            }

            size_t sum = 0;
            std::for_each(m_chunks.begin(), m_chunks.end(), [&](const SequentialChunkPtr p) { sum += p->SizeInSamples(); });
            assert(sweepNumberOfSamples == sum);
        };

        const std::vector<SequentialChunkPtr>& Chunks() const
        {
            return m_chunks;
        }

        vector<StreamDescriptionPtr> GetStreamDescriptions() const override
        {
            return std::vector<StreamDescriptionPtr>
            {
                make_shared<StreamDescription>(StreamDescription{
                    L"input",
                    0,
                    StorageType::dense,
                    ElementType::tfloat,
                    m_sampleLayout
                })
            };
        }

        virtual ChunkPtr GetChunk(ChunkIdType chunkId) override
        {
            // We cannot simply give a chunk, otherwise we do not test the case when the chunk gets released.
            // Let's create a new one.
            return std::make_shared<SequentialChunk>(*m_chunks[chunkId]);
        }

        virtual bool GetSequenceDescription(const SequenceDescription&, SequenceDescription&) override
        {
            throw logic_error("Not implemented");
        }

        virtual ChunkDescriptions GetChunkDescriptions() override
        {
            ChunkDescriptions result;
            for (size_t i = 0; i < m_chunks.size(); ++i)
            {
                result.push_back(std::make_shared<ChunkDescription>(
                    ChunkDescription{ (ChunkIdType)i, m_chunks[i]->SizeInSamples(), m_chunks[i]->SizeInSequences() }));
            }
            return result;
        }

        virtual void GetSequencesForChunk(ChunkIdType chunkId, vector<SequenceDescription>& descriptions) override
        {
            const auto& chunk = m_chunks[chunkId];
            for (size_t i = 0; i < chunk->SizeInSequences(); ++i)
            {
                KeyType key;
                key.m_sample = 0;
                key.m_sequence = (uint32_t)chunk->m_data[i][0];
                descriptions.push_back(SequenceDescription{
                    i,
                    (uint32_t)(chunk->m_data[i].size()),
                    chunkId,
                    key
                });
            }
        }

        size_t TotalSize() const
        {
            size_t size = 0;
            for (auto& c : m_chunks)
                size += c->m_sizeInSamples;
            return size;
        }

        const std::map<size_t, SequenceInfo>& Corpus() const
        {
            return m_sequenceInfos;
        }

    private:
        std::vector<SequentialChunkPtr> m_chunks;
        std::map<size_t, SequenceInfo> m_sequenceInfos;
        TensorShapePtr m_sampleLayout;

        DISABLE_COPY_AND_MOVE(SequentialDeserializer);
    };

    bool operator == (const SequentialDeserializer::SequenceInfo& a, const SequentialDeserializer::SequenceInfo& b)
    {
        return a.id == b.id && a.size == b.size && a.chunkId == b.chunkId && a.startingValue == b.startingValue;
    }

    bool operator != (const SequentialDeserializer::SequenceInfo& a, const SequentialDeserializer::SequenceInfo& b)
    {
        return !(a == b);
    }

    std::ostream& operator << (std::ostream& ostr, const SequentialDeserializer::SequenceInfo& a)
    {
        ostr << a.startingValue;
        return ostr;
    }

    bool operator < (const SequentialDeserializer::SequenceInfo& a, const SequentialDeserializer::SequenceInfo& b)
    {
        return a.startingValue < b.startingValue;
    }


    typedef std::shared_ptr<SequentialDeserializer> SequentialDeserializerPtr;

    inline std::vector<float> ReadNextSamples(SequenceEnumeratorPtr sequenceEnumerator, size_t numSamples)
    {
        ReaderConfiguration config;
        config.m_numberOfWorkers = 1;
        config.m_workerRank = 0;
        config.m_minibatchSizeInSamples = 1;
        config.m_truncationSize = 0;
        sequenceEnumerator->SetConfiguration(config);

        size_t mbSize = 1;
        std::vector<float> result;
        while (result.size() < numSamples)
        {
            auto sequences = sequenceEnumerator->GetNextSequences(mbSize, mbSize);
            assert(!sequences.m_endOfEpoch);
            assert(sequences.m_data.size() == 1 || sequences.m_data.size() == 0);
            if (sequences.m_data.empty())
                continue;

            for (auto& s : sequences.m_data[0])
            {
                float* casted = (float*)s->GetDataBuffer();
                for (size_t i = 0; i < s->m_numberOfSamples; ++i)
                {
                    result.push_back(casted[i]);
                }
            }
        }
        return result;
    }

    // A set of helper functions
    inline std::vector<float> ReadFullEpoch(SequenceEnumeratorPtr sequenceEnumerator, size_t epochSize, size_t epochIndex)
    {
        EpochConfiguration config;
        config.m_numberOfWorkers = 1;
        config.m_workerRank = 0;
        config.m_minibatchSizeInSamples = 1;
        config.m_totalEpochSizeInSamples = epochSize;
        config.m_epochIndex = epochIndex;
        sequenceEnumerator->StartEpoch(config);

        bool shouldBreak = false;
        size_t mbSize = 1;
        std::vector<float> epoch;
        while (!shouldBreak)
        {
            auto sequences = sequenceEnumerator->GetNextSequences(mbSize, mbSize);
            shouldBreak = sequences.m_endOfEpoch;
            assert(sequences.m_data.size() == 1 || sequences.m_data.size() == 0);
            if (sequences.m_data.size() == 0)
            {
                // last sequence
                break;
            }

            for (auto& s : sequences.m_data[0])
            {
                float* casted = (float*)s->GetDataBuffer();
                for (size_t i = 0; i < s->m_numberOfSamples; ++i)
                {
                    epoch.push_back(casted[i]);
                }
            }
        }
        return epoch;
    }

    inline bool CheckFullSweep(size_t sweepNumberOfSamples, const std::vector<float> values)
    {
        long long sum = 0;
        for (auto v : values)
        {
            sum += (long)v;
        }

        long long expected = (long long)((sweepNumberOfSamples - 1) * sweepNumberOfSamples / 2);
        return sum == expected;
    }

    inline std::vector<float> ReadFullSweep(SequenceEnumeratorPtr sequenceEnumerator, size_t sweep, size_t sweepNumberOfSamples)
    {
        const size_t randomizeAll = ((size_t)-1) >> 2;
        auto data = ReadFullEpoch(sequenceEnumerator, randomizeAll, sweep);
        bool valid = CheckFullSweep(sweepNumberOfSamples, data);
        if (!valid)
            RuntimeError("Invalid sweep data.");
        return data;
    }

    template <class TElement>
    std::vector<TElement> Concat(const std::vector<vector<TElement>>& args)
    {
        std::vector<TElement> result;
        for (const auto& i : args)
        {
            result.insert(result.end(), i.begin(), i.end());
        }
        return result;
    }

}}}}
