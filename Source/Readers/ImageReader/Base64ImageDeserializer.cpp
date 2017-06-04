//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include "Base64ImageDeserializer.h"
#include "ImageTransformers.h"
#include "ReaderUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    class Base64ImageDeserializer::ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk>
    {
        ChunkDescriptor m_descriptor;
        size_t m_chunkOffset;
        Base64ImageDeserializer& m_deserializer;
        // TODO: Could probably be a memory mapped region.
        std::vector<char> m_buffer;

    public:
        ImageChunk(const ChunkDescriptor& descriptor, Base64ImageDeserializer& parent)
            : m_descriptor(descriptor), m_deserializer(parent)
        {
            // Let's see if the open descriptor has problems.
            if (ferror(m_deserializer.m_dataFile.get()) != 0)
                m_deserializer.m_dataFile.reset(fopenOrDie(m_deserializer.m_fileName.c_str(), L"rbS"), [](FILE* f) { if (f) fclose(f); });

            if (descriptor.Sequences().empty() || !descriptor.SizeInBytes())
                LogicError("Empty chunks are not supported.");

            m_buffer.resize(descriptor.SizeInBytes() + 1);

            // Make sure we always have 0 at the end for buffer overrun.
            m_buffer[descriptor.SizeInBytes()] = 0;
            m_chunkOffset = descriptor.m_offset;

            // Read chunk into memory.
            int rc = _fseeki64(m_deserializer.m_dataFile.get(), m_chunkOffset, SEEK_SET);
            if (rc)
                RuntimeError("Error seeking to position '%" PRId64 "' in the input file '%ls', error code '%d'", m_chunkOffset, m_deserializer.m_fileName.c_str(), rc);

            freadOrDie(m_buffer.data(), descriptor.SizeInBytes(), 1, m_deserializer.m_dataFile.get());
        }

        std::string KeyOf(const SequenceDescriptor& s) const
        {
            return m_deserializer.m_corpus->IdToKey(s.m_key);
        }

        void GetSequence(size_t sequenceIndex, std::vector<SequenceDataPtr>& result) override
        {
            const size_t innerSequenceIndex = m_deserializer.m_multiViewCrop ? sequenceIndex / ImageDeserializerBase::NumMultiViewCopies : sequenceIndex;
            const size_t copyId = m_deserializer.m_multiViewCrop ? sequenceIndex % ImageDeserializerBase::NumMultiViewCopies : 0;

            const auto& sequence = m_descriptor.Sequences()[innerSequenceIndex];
            const size_t offset = sequence.OffsetInChunk();

            // m_buffer always end on 0, so no overrun can happen.
            const char* currentSequence = &m_buffer[0] + offset;

            bool hasSequenceKey = m_deserializer.m_indexer->HasSequenceIds();
            if (hasSequenceKey) // Skip sequence key.
            {
                currentSequence = strchr(currentSequence, '\t');

                // Let's check the sequence id.
                if (!currentSequence)
                    RuntimeError("Empty label value for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

                currentSequence++;
            }

            char* eptr = nullptr;
            errno = 0;
            size_t classId = strtoull(currentSequence, &eptr, 10);
            if (currentSequence == eptr || errno == ERANGE)
                RuntimeError("Cannot parse label value for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

            size_t labelDimension = m_deserializer.m_labelGenerator->LabelDimension();
            if (classId >= labelDimension)
                RuntimeError(
                    "Image with id '%s' has invalid class id '%zu'. It is exceeding the label dimension of '%zu'",
                    KeyOf(sequence).c_str(), classId, labelDimension);

            // Let's find the end of the label, we still expect to find the data afterwards.
            currentSequence = strstr(currentSequence, "\t");
            if (!currentSequence)
                RuntimeError("No data found for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

            currentSequence++;

            // Let's get the image.
            const char* imageStart = currentSequence;
            currentSequence = strstr(currentSequence, "\n");
            if (!currentSequence)
                RuntimeError("Empty image for sequence '%s'", KeyOf(sequence).c_str());

            // Remove non base64 characters at the end of the string (tabs/spaces)
            while (currentSequence > imageStart &&  !IsBase64Char(*(currentSequence - 1)))
                currentSequence--;

            std::vector<char> decodedImage;
            cv::Mat image;
            if (!DecodeBase64(imageStart, currentSequence, decodedImage))
            {
                fprintf(stderr, "WARNING: Cannot decode sequence with id %zu in the input file '%ls'\n", sequence.m_key, m_deserializer.m_fileName.c_str());
            }
            else
            {
                image = cv::imdecode(decodedImage, m_deserializer.m_grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
            }

            m_deserializer.PopulateSequenceData(image, classId, copyId, { sequence.m_key, 0 }, result);
        }
    };

    static bool HasSequenceKeys(const std::string& mapPath)
    {
        std::ifstream mapFile(mapPath);
        if (!mapFile)
            RuntimeError("Could not open '%s' for reading.", mapPath.c_str());

        string line;
        if (!std::getline(mapFile, line))
            RuntimeError("Could not read the file '%s'.", mapPath.c_str());

        // Try to parse sequence id, file path and label.
        std::string image, classId, sequenceKey;
        std::stringstream ss(line);
        if (!std::getline(ss, sequenceKey, '\t') || !std::getline(ss, classId, '\t') || !std::getline(ss, image, '\t'))
        {
            return false;
        }
        return true;
    }

    Base64ImageDeserializer::Base64ImageDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary) : ImageDeserializerBase(corpus, config, primary)
    {
        auto mapFile = config(L"file");
        bool hasSequenceKeys = HasSequenceKeys(mapFile);
        m_fileName.assign(mapFile.begin(), mapFile.end());

        attempt(5, [this, hasSequenceKeys, corpus]()
        {
            if (!m_dataFile || ferror(m_dataFile.get()) != 0)
                m_dataFile.reset(fopenOrDie(m_fileName, L"rbS"), [](FILE* f) { if (f) fclose(f); });

            m_indexer = make_unique<Indexer>(m_dataFile.get(), m_primary, !hasSequenceKeys);
            m_indexer->Build(corpus);
        });
    }

    ChunkDescriptions Base64ImageDeserializer::GetChunkDescriptions()
    {
        const auto& index = m_indexer->GetIndex();
        // In case of multi crop the deserializer provides the same sequence NumMultiViewCopies times.
        size_t sequencesPerInitialSequence = m_multiViewCrop ? ImageDeserializerBase::NumMultiViewCopies : 1;
        ChunkDescriptions result;
        result.reserve(index.Chunks().size() * sequencesPerInitialSequence);
        for(uint32_t i = 0; i < index.Chunks().size(); ++i)
        {
            const auto& chunk = index.Chunks()[i];
            auto c = std::make_shared<ChunkDescription>();
            c->m_id = i;
            assert(chunk.NumSamples() == chunk.Sequences().size());
            c->m_numberOfSamples = c->m_numberOfSequences = chunk.Sequences().size() * sequencesPerInitialSequence;
            result.push_back(c);
        }
        return result;
    }

    void Base64ImageDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
    {
        const auto& index = m_indexer->GetIndex();
        const auto& chunk = index.Chunks()[chunkId];
        size_t sequenceCopies = m_multiViewCrop ? NumMultiViewCopies : 1;
        result.reserve(sequenceCopies * chunk.Sequences().size());
        size_t currentId = 0;
        for (uint32_t indexInChunk = 0; indexInChunk < chunk.Sequences().size(); ++indexInChunk)
        {
            auto const& s = chunk.Sequences()[indexInChunk];
            assert(currentId / sequenceCopies == indexInChunk);
            for (size_t i = 0; i < sequenceCopies; ++i)
            {
                result.push_back(
                {
                    currentId,
                    s.m_numberOfSamples,
                    chunkId,
                    { s.m_key, 0 }
                });

                currentId++;
            }
        }
    }

    ChunkPtr Base64ImageDeserializer::GetChunk(ChunkIdType chunkId)
    {
        const auto& chunkDescriptor = m_indexer->GetIndex().Chunks()[chunkId];
        return make_shared<ImageChunk>(chunkDescriptor, *this);
    }

    bool Base64ImageDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& r)
    {
        return DataDeserializerBase::GetSequenceDescriptionByKey(m_indexer->GetIndex(), key, r);
    }
}}}
