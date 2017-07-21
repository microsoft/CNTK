//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS
#include <inttypes.h>
#include "Indexer.h"
#include <boost/utility/string_ref.hpp>
#include <boost/algorithm/string.hpp>

using std::string;

namespace CNTK {

Indexer::Indexer(FILE* file, bool primary, bool skipSequenceIds, char streamPrefix, size_t chunkSize, const std::string& mainStream, size_t bufferSize) :
    m_streamPrefix(streamPrefix),
    m_buffer(bufferSize, !mainStream.empty()),
    m_file(file),
    m_hasSequenceIds(!skipSequenceIds),
    m_index(chunkSize, primary),
    m_mainStream(mainStream)
{
    if (m_file == nullptr)
        RuntimeError("Input file not open for reading");
    m_fileSize = filesize(file);
}

void Indexer::BuildFromLines()
{
    m_hasSequenceIds = false;
    int64_t offset = m_buffer.GetFileOffset();
    while (!m_buffer.Eof())
    {
        auto pos = m_buffer.MoveToNextLine();
        if (pos)
        {
            auto sequenceOffset = offset;
            offset = m_buffer.GetFileOffset();
            m_index.AddSequence(SequenceDescriptor{ m_buffer.CurrentLine() - 1, 1 }, sequenceOffset, offset);
        }
        else
            m_buffer.RefillFrom(m_file);
    }

    if (offset < m_fileSize)
    {
        // There's a number of characters, not terminated by a newline,
        // add a sequence to the index, parser will have to deal with it.
        m_index.AddSequence(SequenceDescriptor{ m_buffer.CurrentLine(), 1 }, offset, m_fileSize);
    }
}

void Indexer::Build(CorpusDescriptorPtr corpus)
{
    if (!m_index.IsEmpty())
    {
        return;
    }

    // Create a lambda to read symbolic or numeric sequence ids,
    // depending on what the corpus expects.
    std::function<bool(size_t&)> tryGetSequenceId;
    if (corpus->IsNumericSequenceKeys())
        tryGetSequenceId = [this](size_t& id) { return TryGetNumericSequenceId(id); };
    else
        tryGetSequenceId = [this, corpus](size_t& id) { return TryGetSymbolicSequenceId(id, corpus->KeyToId); };

    m_index.Reserve(m_fileSize);

    m_buffer.RefillFrom(m_file);
    if (m_buffer.Eof())
        RuntimeError("Input file is empty");

    m_buffer.SkipBOMIfPresent();

    // check the first byte and decide what to do next
    if (!m_hasSequenceIds || *m_buffer.m_current == m_streamPrefix)
    {
        // Skip sequence id parsing, treat lines as individual sequences
        // In this case the sequences do not have ids, they are assigned a line number.
        // If corpus expects to have sequence ids as symbolic names we throw.
        if (!corpus->IsNumericSequenceKeys())
            RuntimeError("Corpus expects non-numeric sequence keys present but the input file does not have them."
                "Please use the configuration to enable numeric keys instead.");

        BuildFromLines();
        m_index.MapSequenceKeyToLocation();
        return;
    }

    size_t id = 0;
    int64_t offset = m_buffer.GetFileOffset();
    // read the very first sequence id
    if (!tryGetSequenceId(id))
    {
        RuntimeError("Expected a sequence id at the offset %" PRIi64 ", none was found.", offset);
    }

    auto sequenceOffset = offset;
    size_t previousId = id;
    uint32_t numberOfSamples = 0;
    while (!m_buffer.Eof())
    {
        if (!m_mainStream.empty())
        {
            if(SkipLineWithCheck())
                numberOfSamples++;
        }
        else
        {
            SkipLine(); // ignore whatever is left on this line.
            numberOfSamples++;
        }

        offset = m_buffer.GetFileOffset(); // a new line starts at this offset;
        if (!m_buffer.Eof() && tryGetSequenceId(id) && id != previousId)
        {
            // found a new sequence, which starts at the [offset] bytes into the file
            // adding the previous one to the index.
            m_index.AddSequence(SequenceDescriptor{ previousId, numberOfSamples }, sequenceOffset, offset);

            sequenceOffset = offset;
            previousId = id;
            numberOfSamples = 0;
        }
    }

    m_index.AddSequence(SequenceDescriptor{ previousId, numberOfSamples }, sequenceOffset, m_fileSize);
    m_index.MapSequenceKeyToLocation();
}

void Indexer::SkipLine()
{
    while (!m_buffer.Eof())
    {
        auto pos = m_buffer.MoveToNextLine();
        if (pos)
        {
            //found a new-line character
            if (pos == m_buffer.End())
                m_buffer.RefillFrom(m_file);
            return;
        }

        m_buffer.RefillFrom(m_file);
    }
}

bool Indexer::SkipLineWithCheck()
{
    auto currentLine = m_buffer.m_current;
    auto pos = m_buffer.MoveToNextLine();
    // In this function we always expect the buffer to contain full lines only.
    // The only exception is at the end of the file \n is missing. Let's check this situation.
    if (!pos && currentLine != m_buffer.End() &&
        filesize(m_file) == m_buffer.GetFileOffset() + m_buffer.Left())
        pos = m_buffer.End();

    if (pos)
    {
        boost::string_ref s(currentLine, pos - currentLine);
        bool found = s.find(m_mainStream) != boost::string_ref::npos;
        if (pos == m_buffer.End())
            m_buffer.RefillFrom(m_file);

        return found;
    }

    if (currentLine != m_buffer.End())
        RuntimeError("Unexpected end of line");

    m_buffer.RefillFrom(m_file);
    return false;
}


bool Indexer::TryGetNumericSequenceId(size_t& id)
{
    bool found = false;
    id = 0;
    while (!m_buffer.Eof())
    {
        char c = *m_buffer.m_current;
        if (!isdigit(c))
        {
            // Stop as soon as there's a non-digit character
            return found;
        }

        size_t temp = id;
        id = id * 10 + (c - '0');
        if (temp > id)
        {
            RuntimeError("Overflow while reading a numeric sequence id (%zu-bit value).", sizeof(id));
        }
        
        found = true;
        ++m_buffer.m_current;

        if (m_buffer.m_current == m_buffer.End())
            m_buffer.RefillFrom(m_file);
    }

    // reached EOF without hitting the pipe character,
    // ignore it for not, parser will have to deal with it.
    return false;
}

bool Indexer::TryGetSymbolicSequenceId(size_t& id, std::function<size_t(const std::string&)> keyToId)
{
    bool found = false;
    id = 0;
    std::string key;
    key.reserve(256);
    while (!m_buffer.Eof())
    {
        char c = *m_buffer.m_current;
        if (isspace(c))
        {
            if (found)
                id = keyToId(key);
            return found;
        }

        key += c;
        found = true;
        ++m_buffer.m_current;

        if(m_buffer.m_current == m_buffer.End())
            m_buffer.RefillFrom(m_file);
    }

    // reached EOF without hitting the pipe character,
    // ignore it for not, parser will have to deal with it.
    return false;
}

void Index::AddSequence(SequenceDescriptor&& sd, size_t startOffsetInFile, size_t endOffsetInFile)
{
    sd.SetSize(endOffsetInFile - startOffsetInFile);

    if (m_chunks.empty() || !m_chunks.back().HasSpaceFor(sd))
    {
        m_chunks.push_back({ m_maxChunkSize, startOffsetInFile });
        if (std::numeric_limits<ChunkIdType>::max() < m_chunks.size())
            RuntimeError("Maximum number of chunks exceeded.");
    }

    ChunkDescriptor* chunk = &m_chunks.back();
    sd.SetOffsetInChunk(startOffsetInFile - chunk->m_offset);
    chunk->AddSequence(std::move(sd), m_trackFirstSamples);
}

std::tuple<bool, uint32_t, uint32_t> Index::GetSequenceByKey(size_t key) const
{
    auto found = std::lower_bound(m_keyToSequenceInChunk.begin(), m_keyToSequenceInChunk.end(), key,
        [](const std::tuple<size_t, size_t, size_t>& a, size_t b)
        {
            return std::get<0>(a) < b;
        });

    if (found == m_keyToSequenceInChunk.end() || std::get<0>(*found) != key)
    {
        return std::make_tuple(false, 0, 0);
    }

    return std::make_tuple(true, std::get<1>(*found), std::get<2>(*found));
}

void Index::MapSequenceKeyToLocation()
{
    if (m_primary)
        return;

    // Precalculate size of the mapping.
    size_t numSequences = 0;
    for (const auto& c : m_chunks)
        numSequences += c.Sequences().size();

    m_keyToSequenceInChunk.reserve(numSequences);

    for (uint32_t i = 0; i < m_chunks.size(); i++)
        for (uint32_t j = 0; j < m_chunks[i].Sequences().size(); j++)
            m_keyToSequenceInChunk.emplace_back(m_chunks[i].Sequences()[j].m_key, i, j);

    // Sort for fast retrieval afterwards
    std::sort(m_keyToSequenceInChunk.begin(), m_keyToSequenceInChunk.end(),
        [](const std::tuple<size_t, uint32_t, uint32_t>& a, const std::tuple<size_t, uint32_t, uint32_t>& b)
    {
        return std::get<0>(a) < std::get<0>(b);
    });
}

}
