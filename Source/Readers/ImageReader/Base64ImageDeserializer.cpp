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

            if (descriptor.m_sequences.empty() || !descriptor.m_byteSize)
                LogicError("Empty chunks are not supported.");

            m_buffer.resize(descriptor.m_byteSize + 1);

            // Make sure we always have 0 at the end for buffer overrun.
            m_buffer[descriptor.m_byteSize] = 0;
            m_chunkOffset = descriptor.m_sequences.front().m_fileOffsetBytes;

            // Read chunk into memory.
            int rc = _fseeki64(m_deserializer.m_dataFile.get(), m_chunkOffset, SEEK_SET);
            if (rc)
                RuntimeError("Error seeking to position '%" PRId64 "' in the input file '%ls', error code '%d'", m_chunkOffset, m_deserializer.m_fileName.c_str(), rc);

            freadOrDie(m_buffer.data(), descriptor.m_byteSize, 1, m_deserializer.m_dataFile.get());
        }

        std::string KeyOf(const SequenceDescriptor& s) const
        {
            return m_deserializer.m_corpus->IdToKey(s.m_key.m_sequence);
        }

        void GetSequence(size_t sequenceId, std::vector<SequenceDataPtr>& result) override
        {
            size_t innerSequenceId = m_deserializer.m_multiViewCrop ? sequenceId / ImageDeserializerBase::NumMultiViewCopies : sequenceId;
            const auto& sequence = m_descriptor.m_sequences[innerSequenceId];
            size_t offset = sequence.m_fileOffsetBytes - m_chunkOffset;

            // Let's parse the string
            char* next_token = nullptr;
            char* token = strtok_s(&m_buffer[0] + offset, "\t", &next_token);
            bool hasSequenceKey = m_deserializer.m_indexer->HasSequenceIds();
            if (hasSequenceKey) // Skip sequence key.
            {
                token = strtok_s(nullptr, "\t", &next_token);
                assert(!std::string(token).empty());
            }

            // Let's get the label.
            if (!token)
                RuntimeError("Empty label value for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

            char* eptr = nullptr;
            errno = 0;
            size_t classId = strtoull(token, &eptr, 10);
            if (token == eptr || errno == ERANGE)
                RuntimeError("Cannot parse label value for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

            size_t labelDimension = m_deserializer.m_labelGenerator->LabelDimension();
            if (classId >= labelDimension)
                RuntimeError(
                    "Image with id '%s' has invalid class id '%" PRIu64 "'. It is exceeding the label dimension of '%" PRIu64,
                    KeyOf(sequence).c_str(), classId, labelDimension);

            // Let's get the image.
            token = strtok_s(nullptr, "\n", &next_token);
            if (!token)
                RuntimeError("Empty image for sequence '%s'", KeyOf(sequence).c_str());

            // Find line end or end of buffer.
            char* endToken = strchr(token, 0);
            if (!endToken)
                RuntimeError("Cannot find the end of the image for sequence '%s' in the input file '%ls'", KeyOf(sequence).c_str(), m_deserializer.m_fileName.c_str());

            // Remove non base64 characters at the end of the string (tabs/spaces)
            while (endToken > token &&  !IsBase64Char(*(endToken - 1)))
                endToken--;

            std::vector<char> decodedImage;
            cv::Mat image;
            if (!DecodeBase64(token, endToken, decodedImage))
            {
                fprintf(stderr, "WARNING: Cannot decode sequence with id %" PRIu64 " in the input file '%ls'\n", sequence.m_key.m_sequence, m_deserializer.m_fileName.c_str());
            }
            else
            {
                image = cv::imdecode(decodedImage, m_deserializer.m_grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
            }

            m_deserializer.PopulateSequenceData(image, classId, sequenceId, result);
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

    Base64ImageDeserializer::Base64ImageDeserializer(CorpusDescriptorPtr corpus, const ConfigParameters& config, bool isPrimary) : ImageDeserializerBase(corpus, config)
    {
        auto mapFile = config(L"file");
        bool hasSequenceKeys = HasSequenceKeys(mapFile);
        m_fileName.assign(mapFile.begin(), mapFile.end());

        attempt(5, [this, hasSequenceKeys, corpus, isPrimary]()
        {
            if (!m_dataFile || ferror(m_dataFile.get()) != 0)
                m_dataFile.reset(fopenOrDie(m_fileName, L"rbS"), [](FILE* f) { if (f) fclose(f); });

            m_indexer = make_unique<Indexer>(m_dataFile.get(), isPrimary, !hasSequenceKeys);
            m_indexer->Build(corpus);
        });
    }

    ChunkDescriptions Base64ImageDeserializer::GetChunkDescriptions()
    {
        const auto& index = m_indexer->GetIndex();
        // In case of multi crop the deserializer provides the same sequence NumMultiViewCopies times.
        size_t sequencesPerInitialSequence = m_multiViewCrop ? ImageDeserializerBase::NumMultiViewCopies : 1;
        ChunkDescriptions result;
        result.reserve(index.m_chunks.size() * sequencesPerInitialSequence);
        for (auto const& chunk : index.m_chunks)
        {
            auto c = std::make_shared<ChunkDescription>();
            c->m_id = chunk.m_id;
            assert(chunk.m_numberOfSamples == chunk.m_numberOfSequences);
            c->m_numberOfSamples = c->m_numberOfSequences = chunk.m_numberOfSequences * sequencesPerInitialSequence;
            result.push_back(c);
        }
        return result;
    }

    void Base64ImageDeserializer::GetSequencesForChunk(ChunkIdType chunkId, std::vector<SequenceDescription>& result)
    {
        const auto& index = m_indexer->GetIndex();
        const auto& chunk = index.m_chunks[chunkId];
        size_t sequencesPerInitialSequence = m_multiViewCrop ? 10 : 1;
        result.reserve(sequencesPerInitialSequence * chunk.m_sequences.size());
        size_t currentId = 0;
        for (auto const& s : chunk.m_sequences)
        {
            assert(currentId / sequencesPerInitialSequence == s.m_id);
            for (size_t i = 0; i < sequencesPerInitialSequence; ++i)
            {
                result.push_back(
                {
                    currentId,
                    s.m_numberOfSamples,
                    s.m_chunkId,
                    s.m_key
                });
                currentId++;
            }
        }
    }

    ChunkPtr Base64ImageDeserializer::GetChunk(ChunkIdType chunkId)
    {
        const auto& chunkDescriptor = m_indexer->GetIndex().m_chunks[chunkId];
        return make_shared<ImageChunk>(chunkDescriptor, *this);
    }

    bool Base64ImageDeserializer::GetSequenceDescriptionByKey(const KeyType& key, SequenceDescription& result)
    {
        const auto& index = m_indexer->GetIndex();

        const auto& keys = index.m_keyToSequenceInChunk;
        auto sequenceLocation = keys.find(key.m_sequence);
        if (sequenceLocation == keys.end())
            return false;

        const auto& chunks = index.m_chunks;
        result = chunks[sequenceLocation->second.first].m_sequences[sequenceLocation->second.second];
        return true;
    }

}}}
