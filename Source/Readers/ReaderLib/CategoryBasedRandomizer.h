//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include "SequenceEnumerator.h"
#include "DataDeserializer.h"

namespace Microsoft {
    namespace MSR {
        namespace CNTK {
            // return a random permutation of a sequence. If not weighted, each element is equally occured at different places; If weighted, the more weighted index have the chance to occur earlier;
            class RandomPermutation {
            private:
                int sample_range_;
                std::unordered_map<int, int> reorder_;
                vector<int> sample_index_;
                std::unordered_set<int> used_;
                std::random_device rd_;
                std::mt19937 gen_;
                std::uniform_int_distribution<> dis_;
                inline int ReorderedWithMap(int id)
                {
                    auto it = reorder_.find(id);
                    return (it != reorder_.end()) ? it->second : id;
                }
            public:
                RandomPermutation(int range) : sample_range_(range), gen_(rd_()){
                    for (int i = 0; i <= range; i++)
                        sample_index_.push_back(i);
                    used_.clear();
                    dis_ = std::uniform_int_distribution<>(0, sample_range_);
                }
                RandomPermutation(const vector<size_t>& index_weight) {
                    for (int i = 0; i < index_weight.size(); i++) {
                        for (int j = 0; j < index_weight[i]; j++) {
                            sample_index_.push_back(i);
                        }
                    }
                    sample_range_ = int(sample_index_.size()) - 1;
                    used_.clear();
                    dis_ = std::uniform_int_distribution<>(0, sample_range_);
                }
                inline void Reset() { reorder_.clear(); used_.clear(); }
                int GetNext() {
                    int rid = -1, id = -1;
                    do {
                        rid = dis_(gen_);
                        id = ReorderedWithMap(rid);
                        reorder_[rid] = ReorderedWithMap(sample_range_--);
                    } while (!used_.insert(sample_index_[id]).second);
                    assert(id != -1);
                    return sample_index_[id];
                }
            };

            // The class represents a randomizer that does category based sampling (For a minibatch of size N
            // , sample K categories, in each category, sample N/K instance).
            // Used for siamesenet training.
            class CategoryBasedRandomizer : public SequenceEnumerator
            {
            public:
                CategoryBasedRandomizer(IDataDeserializerPtr deserializer, size_t samplePerCategory, std::wstring categoryInfoName, bool multithreadedGetNextSequences = false);

                virtual void StartEpoch(const EpochConfiguration& config) override;
                virtual Sequences GetNextSequences(size_t sampleCount) override;
                virtual std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override
                {
                    return m_deserializer->GetStreamDescriptions();
                }

            private:
                // Gets next sequence descriptions with total size less than sampleCount.
                std::vector<SequenceDescription> GetNextSequenceDescriptions(size_t sampleCount);

                // Get chunk index for the sample offset from the beginning of the sweep.
                ChunkIdType GetChunkIndexOf(size_t samplePosition);

                // Moves the cursor to the sequence possibly updating the chunk.
                void MoveToNextSequence();

                IDataDeserializerPtr m_deserializer;

                // Whether to get sequences using multiple thread.
                // TODO temporary; should go away when transformers are moved closer to the deserializer
                bool m_multithreadedGetNextSequences;

                // Stream descriptions
                std::vector<StreamDescriptionPtr> m_streams;

                // Epoch configuration
                EpochConfiguration m_config;

                // Chunk descriptions.
                ChunkDescriptions m_chunkDescriptions;

                // m_chunkDescription defines the complete sweep of samples: [0 .. N]
                // m_chunkSampleOffset for each chunk contains the sample offset in the sweep where the chunk begins.
                std::vector<size_t> m_chunkSampleOffset;

                // Current chunk data.
                ChunkPtr m_currentChunk;
                // Current chunk data id.
                ChunkIdType m_currentChunkId;

                // Current window of sequence descriptions.
                std::vector<SequenceDescription> m_sequenceWindow;

                // Current sequence position the randomizer works with.
                size_t m_currentSequencePositionInChunk;

                // Current chunk position that the randomizer works with.
                // An index inside the m_chunkDescriptions.
                ChunkIdType m_currentChunkPosition;

                // Global sample position on the timeline.
                // TODO: possible recalculate it base on samplePositionInEpoch.
                size_t m_globalSamplePosition;

                // Current sample position in the epoch.
                size_t m_samplePositionInEpoch;

                // Total number of samples in the sweep.
                size_t m_totalNumberOfSamples;

                // Label Stream Index
                size_t m_labelStreamIdx;

                // Collect all the chunks for sampling
                std::map<ChunkIdType, ChunkPtr> m_chunks;

                // Collect all sequences for sampling
                Sequences m_allSequences;

                // Category Based Sampling
                // maintain the samples in each class
                std::unordered_map<size_t, vector<size_t>> class_sample_list_;
                // number of samples in each class, this is for weighted sampling
                vector<size_t> class_sample_list_len_;
                // log the erased classes to evaluate class
                vector<size_t> erased_class_;
                // sample_idx_in_ori_list_[i] is the indexes of that sample in the list before erasing
                vector<size_t> sample_idx_in_ori_list_;
                // give each label an index to handle float or non-continous label
                std::unordered_map<size_t, size_t> idx_to_class_;
                // map from category id and index_in_category to sample_i
                std::map<std::pair<size_t, size_t>, size_t> m_categoryinfo_to_sampleid;
                // store all the labels
                std::vector<float> m_labels;
                // how many instances to sample from each category
                size_t m_samplePerCategory;
                // name of the label stream
                std::wstring m_categoryInfoName;
            };
        }
    }
}
