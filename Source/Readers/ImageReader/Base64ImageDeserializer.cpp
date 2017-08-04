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
#include "Index.h"
#include "IndexBuilder.h"

namespace CNTK {
    using namespace Microsoft::MSR::CNTK;

    class Base64ImageDeserializerImpl::ImageChunk : public Chunk, public std::enable_shared_from_this<ImageChunk>
    {
        ChunkDescriptor m_descriptor;
        size_t m_chunkOffset;
        Base64ImageDeserializerImpl& m_deserializer;
        // TODO: Could probably be a memory mapped region.
        std::vector<char> m_buffer;

    public:
        ImageChunk(const ChunkDescriptor& descriptor, Base64ImageDeserializerImpl& parent)
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
            m_chunkOffset = descriptor.StartOffset();

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

            if (m_deserializer.m_hasSequenceIds) // Skip sequence key.
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

    Base64ImageDeserializerImpl::Base64ImageDeserializerImpl(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool primary) : ImageDeserializerBase(corpus, config, primary)
    {
        auto mapFile = config(L"file");
        m_hasSequenceIds = HasSequenceKeys(mapFile);
        m_fileName.assign(mapFile.begin(), mapFile.end());

        bool cacheIndex = config(L"cacheIndex", false);

        attempt(5, [this, cacheIndex, corpus]()
        {
            if (!m_dataFile || ferror(m_dataFile.get()) != 0)
                m_dataFile.reset(fopenOrDie(m_fileName, L"rbS"), [](FILE* f) { if (f) fclose(f); });

            m_index = TextInputIndexBuilder(FileWrapper(m_fileName, m_dataFile.get()))
                .SetSkipSequenceIds(!m_hasSequenceIds)
                .SetPrimary(m_primary)
                .SetCorpus(corpus)
                .SetCachingEnabled(cacheIndex)
                .Build();
        });
    }

    std::vector<ChunkInfo> Base64ImageDeserializerImpl::ChunkInfos()
    {
        // In case of multi crop the deserializer provides the same sequence NumMultiViewCopies times.
        size_t sequencesPerInitialSequence = m_multiViewCrop ? ImageDeserializerBase::NumMultiViewCopies : 1;
        std::vector<ChunkInfo> result;
        result.reserve(m_index->NumberOfChunks() * sequencesPerInitialSequence);
        for(uint32_t i = 0; i < m_index->NumberOfChunks(); ++i)
        {
            const auto& chunk = m_index->Chunks()[i];
            ChunkInfo c;
            c.m_id = i;
            assert(chunk.NumberOfSamples() == chunk.NumberOfSequences());
            c.m_numberOfSamples = c.m_numberOfSequences = chunk.NumberOfSequences() * sequencesPerInitialSequence;
            result.push_back(c);
        }
        return result;
    }

    void Base64ImageDeserializerImpl::SequenceInfosForChunk(ChunkIdType chunkId, std::vector<SequenceInfo>& result)
    {
        const auto& chunk = m_index->Chunks()[chunkId];
        size_t sequenceCopies = m_multiViewCrop ? NumMultiViewCopies : 1;
        result.reserve(sequenceCopies * chunk.NumberOfSequences());
        size_t currentId = 0;
        for (uint32_t indexInChunk = 0; indexInChunk < chunk.NumberOfSequences(); ++indexInChunk)
        {
            auto const& s = chunk[indexInChunk];
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

    ChunkPtr Base64ImageDeserializerImpl::GetChunk(ChunkIdType chunkId)
    {
        const auto& chunkDescriptor = m_index->Chunks()[chunkId];
        return make_shared<ImageChunk>(chunkDescriptor, *this);
    }

    bool Base64ImageDeserializerImpl::GetSequenceInfoByKey(const SequenceKey& key, SequenceInfo& r)
    {
        return DataDeserializerBase::GetSequenceInfoByKey(*m_index, key, r);
    }
}
