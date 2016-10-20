//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <random>
#include <process.h>
#include <iostream>

#include "CategoryBasedRandomizer.h"
#include "DataReader.h"
#include "ExceptionCapture.h"

namespace Microsoft {
    namespace MSR {
        namespace CNTK {
            CategoryBasedRandomizer::CategoryBasedRandomizer(IDataDeserializerPtr deserializer, size_t samplePerCategory, std::wstring categoryInfoName, bool multithreadedGetNextSequences)
                : m_deserializer(deserializer),
                m_samplePositionInEpoch(0),
                m_currentChunkPosition(CHUNKID_MAX),
                m_globalSamplePosition(0),
                m_totalNumberOfSamples(0),
                m_currentSequencePositionInChunk(0),
                m_multithreadedGetNextSequences(multithreadedGetNextSequences),
                m_samplePerCategory(samplePerCategory),
                m_categoryInfoName(categoryInfoName)
            {
                assert(deserializer != nullptr);
                m_streams = m_deserializer->GetStreamDescriptions();
                m_chunkDescriptions = m_deserializer->GetChunkDescriptions();

                size_t sampleCount = 0;
                for (const auto& chunk : m_chunkDescriptions)
                {
                    // Check that position corresponds to chunk id.
                    assert(m_chunkSampleOffset.size() == chunk->m_id);

                    m_chunkSampleOffset.push_back(sampleCount);
                    sampleCount += chunk->m_numberOfSamples;
                }

                if (sampleCount == 0)
                {
                    RuntimeError("NoRandomizer: Expected input to contain samples, but the number of successfully read samples was 0.");
                }

                m_totalNumberOfSamples = sampleCount;

                m_labels.resize(m_totalNumberOfSamples);

                // Get the stream index for the labels
                for (const auto& stream : m_streams) {
                    // TODO: Currently hard code the label name
                    if (stream->m_name == L"Labels") {
                        m_labelStreamIdx = stream->m_id;
                    }
                }
            }

            ChunkIdType CategoryBasedRandomizer::GetChunkIndexOf(size_t samplePosition)
            {
                auto result = std::upper_bound(m_chunkSampleOffset.begin(), m_chunkSampleOffset.end(), samplePosition);
                return (ChunkIdType)(result - 1 - m_chunkSampleOffset.begin());
            }

            void CategoryBasedRandomizer::StartEpoch(const EpochConfiguration& config)
            {
                m_config = config;

                if (m_config.m_totalEpochSizeInSamples == requestDataSize)
                {
                    m_config.m_totalEpochSizeInSamples = m_totalNumberOfSamples;
                }

                // Read all sequences for sampling
                m_allSequences.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(m_totalNumberOfSamples));
                // load the sequencedescription for the first chunk
                ChunkIdType chunkIndex = GetChunkIndexOf(0);

                if (chunkIndex != m_currentChunkPosition)
                {
                    // unloading everything.
                    m_currentChunkId = CHUNKID_MAX;
                    m_currentChunk = nullptr;

                    m_currentChunkPosition = chunkIndex;
                    m_currentSequencePositionInChunk = 0;
                    m_sequenceWindow.clear();
                    m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
                }
                std::vector<SequenceDescription> descriptions = GetNextSequenceDescriptions(m_totalNumberOfSamples);
                
                if (m_currentChunk != nullptr)
                {
                    m_chunks[m_currentChunkId] = m_currentChunk;
                }
                for (int i = 0; i < m_totalNumberOfSamples; ++i)
                {
                    const auto& sequenceDescription = descriptions[i];
                    auto it = m_chunks.find(sequenceDescription.m_chunkId);
                    if (it == m_chunks.end())
                    {
                        m_chunks[sequenceDescription.m_chunkId] = m_deserializer->GetChunk(sequenceDescription.m_chunkId);
                    }
                }


                auto process = [&](int i) -> void {
                    std::vector<SequenceDataPtr> sequence;
                    const auto& sequenceDescription = descriptions[i];

                    auto it = m_chunks.find(sequenceDescription.m_chunkId);
                    if (it == m_chunks.end())
                    {
                        LogicError("Invalid chunk requested.");
                    }

                    it->second->GetSequence(sequenceDescription.m_id, sequence);
                    for (int j = 0; j < m_streams.size(); ++j)
                    {
                        m_allSequences.m_data[j][i] = sequence[j];
                        // Get Label
                        if (j == m_labelStreamIdx) {
                            // Hacky here, need refactoring
                            if (m_streams[j]->m_elementType == ElementType::tfloat)
                                memcpy(&m_labels[i], (const char*)(sequence[j]->m_data), sizeof(float));
                            else
                                memcpy(&m_labels[i], (const char*)(sequence[j]->m_data), sizeof(double));
                            //std::cout << j << " " << i << " " << m_labels[i] << std::endl;
                        }
                    }
                };

                // TODO: This will be changed, when we move transformers under the (no-) randomizer, should not deal with multithreading here.

                /*
                if (m_multithreadedGetNextSequences)
                {
                    ExceptionCapture capture;
#pragma omp parallel for schedule(dynamic)
                    for (int i = 0; i < m_totalNumberOfSamples; ++i)
                        capture.SafeRun(process, i);
                    capture.RethrowIfHappened();
                }
                else
                */
                {
                    for (int i = 0; i < m_totalNumberOfSamples; ++i)
                        process(i);
                }
                
                // Prepare for category sampling 
                class_sample_list_.clear();
                sample_idx_in_ori_list_.clear();
                idx_to_class_.clear();
                int label_idx = 0;
                for (size_t i = 0; i < m_totalNumberOfSamples; ++i) {
                    // Bug, currently assume only one sample in a sequence
                    auto it = class_sample_list_.find(size_t(m_labels[i]));
                    if (it != class_sample_list_.end()) it->second.push_back(i);
                    else class_sample_list_.insert(std::make_pair(size_t(m_labels[i]), std::vector<size_t>(1, size_t(i))));
                }
                for (auto it = class_sample_list_.begin(); it != class_sample_list_.end();)
                {
                    if (m_samplePerCategory > 0 && it->second.size() < m_samplePerCategory)
                    {
                        erased_class_.push_back(it->first);
                        it = class_sample_list_.erase(it);
                    }
                    else
                    {
                        for (int i = 0; i < it->second.size(); ++i)
                        {
                            size_t id = sample_idx_in_ori_list_.size();
                            sample_idx_in_ori_list_.push_back(it->second[i]);
                            it->second[i] = id;
                        }
                        idx_to_class_[label_idx++] = it->first;
                        class_sample_list_len_.push_back(it->second.size());
                        assert(label_idx == class_sample_list_len_.size());
                        ++it;
                    }
                }
            };

            // Moving the cursor to the next sequence. Possibly updating the chunk information if needed.
            void CategoryBasedRandomizer::MoveToNextSequence()
            {
                SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
                m_samplePositionInEpoch += sequence.m_numberOfSamples;
                m_globalSamplePosition += sequence.m_numberOfSamples;

                if (m_currentSequencePositionInChunk + 1 >= m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences)
                {
                    // Moving to the next chunk.
                    m_currentChunkPosition = (m_currentChunkPosition + 1) % m_chunkDescriptions.size();
                    m_currentSequencePositionInChunk = 0;
                    m_sequenceWindow.clear();
                    m_deserializer->GetSequencesForChunk(m_currentChunkPosition, m_sequenceWindow);
                }
                else
                {
                    m_currentSequencePositionInChunk++;
                }
            }

            // Gets next sequence descriptions with total size less than sampleCount.
            std::vector<SequenceDescription> CategoryBasedRandomizer::GetNextSequenceDescriptions(size_t sampleCount)
            {
                assert(m_sequenceWindow.size() != 0);
                assert(m_chunkDescriptions[m_currentChunkPosition]->m_numberOfSequences > m_currentSequencePositionInChunk);
                int samples = (int)sampleCount;

                std::vector<SequenceDescription> result;

                do
                {
                    const SequenceDescription& sequence = m_sequenceWindow[m_currentSequencePositionInChunk];
                    result.push_back(sequence);
                    samples -= (int)sequence.m_numberOfSamples;
                    MoveToNextSequence();
                }
                // Check whether the next sequence fits into the sample count, if not, exit.
                while (samples - (int)m_sequenceWindow[m_currentSequencePositionInChunk].m_numberOfSamples >= 0);
                return result;
            }

            Sequences CategoryBasedRandomizer::GetNextSequences(size_t sampleCount)
            {
                Sequences result;
                int samples = (int)sampleCount;
                result.m_data.resize(m_streams.size(), std::vector<SequenceDataPtr>(samples));

                RandomPermutation class_perm(class_sample_list_len_);

                // Sample a batch
                for (int num = 0; num < samples;) {
                    // first random get next class
                    size_t classid = idx_to_class_[class_perm.GetNext()];
                    auto& vec = class_sample_list_[classid];
                    // second permulate with the samples of a class
                    RandomPermutation sample_in_class_perm((int)(vec.size() - 1));
                    for (int numc = 0; numc < m_samplePerCategory && num < samples; ++numc){
                        size_t sampleid = vec[sample_in_class_perm.GetNext()];
                        //std::cout << numc << " " << num << " " << classid << " " << sampleid << " " << sample_idx_in_ori_list_[sampleid] << std::endl;
                        for (int j = 0; j < m_streams.size(); ++j)
                        {
                            result.m_data[j][num] = m_allSequences.m_data[j][sample_idx_in_ori_list_[sampleid]];
                        }
                        num++;
                    }
                }
                return result;
            }

        }
    }
}
