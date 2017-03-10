//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include <inttypes.h>
#include "MLFIndexer.h"
#include "MLFUtils.h"

using std::string;

const static char ROW_DELIMITER = '\n';

namespace Microsoft { namespace MSR { namespace CNTK {

    MLFIndexer::MLFIndexer(FILE* file, bool frameMode, size_t chunkSize, size_t bufferSize) :
        m_bufferSize(bufferSize),
        m_file(file),
        m_fileOffsetStart(0),
        m_fileOffsetEnd(0),
        m_buffer(bufferSize),
        m_done(false),
        m_index(chunkSize, true, frameMode)
    {
        if (!m_file)
            RuntimeError("Input file not open for reading");
    }

    void MLFIndexer::RefillBuffer()
    {
        if (!m_done)
        {
            // Make sure we always have m_bufferSize elements in side the buffer.
            m_buffer.resize(m_bufferSize);

            memcpy(&m_buffer[0], m_lastLineInBuffer.data(), m_lastLineInBuffer.size());
            size_t bytesRead = fread(&m_buffer[0] + m_lastLineInBuffer.size(), 1, m_buffer.size() - m_lastLineInBuffer.size(), m_file);

            if (bytesRead == (size_t)-1)
                RuntimeError("Could not read from the input file.");

            if (bytesRead == 0)
            {
                m_buffer.clear();
                m_lastLineInBuffer.clear();
                m_fileOffsetStart = m_fileOffsetEnd;
                m_done = true;
                return;
            }

            int lastLF = 0;
            {
                // Let's find the latest \n if exists.
                for (lastLF = (int)m_lastLineInBuffer.size() + (int)bytesRead - 1; lastLF >= 0; lastLF--)
                {
                    if (m_buffer[lastLF] == '\n')
                        break;
                }

                if (lastLF < 0)
                    RuntimeError("Length of MLF sequence cannot exceed %d bytes.", (int)m_bufferSize);
            }

            auto logicalBufferEnd = lastLF + 1;

            m_fileOffsetStart = m_fileOffsetEnd;
            m_fileOffsetEnd = m_fileOffsetStart + logicalBufferEnd;

            auto lastParialLineSize = m_buffer.size() - logicalBufferEnd;

            // Remember the last possibly parital line.
            m_lastLineInBuffer.resize(lastParialLineSize);
            memcpy(&m_lastLineInBuffer[0], m_buffer.data() + logicalBufferEnd, lastParialLineSize);

            m_buffer.resize(logicalBufferEnd);
        }
    }

    void MLFIndexer::Build(CorpusDescriptorPtr corpus)
    {
        if (!m_index.IsEmpty())
            return;

        m_index.Reserve(filesize(m_file));

        RefillBuffer(); // read the first block of data
        if (m_done)
            RuntimeError("Input file is empty");

        size_t id = 0;
        SequenceDescriptor sd = {};
        int currentState = 0;
        vector<boost::iterator_range<char*>> lines;
        vector<boost::iterator_range<char*>> tokens;
        bool isValid = true;
        size_t lastNonEmptyString = 0;
        while (!m_done)
        {
            lines.clear();
            ReadLines(m_buffer, lines);

            lastNonEmptyString = SIZE_MAX;
            for (size_t i = 0; i < lines.size(); i++)
            {
                if (lines[i].begin() == lines[i].end()) // Skip all empty lines.
                    continue;

                switch (currentState)
                {
                case 0:
                {
                    if (std::string(lines[i].begin(), lines[i].end()) != "#!MLF!#")
                        RuntimeError("Expected MLF header was not found.");
                    currentState = 1;
                }
                break;
                case 1:
                {
                    // When several files are appended to a big mlf, usually the can be 
                    // an MLF header between the utterances.
                    if (std::string(lines[i].begin(), lines[i].end()) == "#!MLF!#")
                        continue;

                    sd = {};
                    sd.m_fileOffsetBytes = m_fileOffsetStart + lines[i].begin() - m_buffer.data();
                    isValid = TryParseSequenceId(lines[i], id, corpus->KeyToId);
                    sd.m_key.m_sequence = id;
                    currentState = 2;
                }
                break;

                case 2:
                {
                    if (std::distance(lines[i].begin(), lines[i].end()) == 1 && *lines[i].begin() == '.')
                    {
                        sd.m_byteSize = m_fileOffsetStart + lines[i].end() - m_buffer.data() - sd.m_fileOffsetBytes;
                        currentState = 1;

                        // Let's find last non empty string and parse information about frames out of it.
                        // Here we assume that the sequence is correct, if not - it will be invalidated later 
                        // when the actual data is read.
                        if (lastNonEmptyString != SIZE_MAX)
                            m_lastNonEmptyLine = std::string(lines[lastNonEmptyString].begin(), lines[lastNonEmptyString].end());

                        if (m_lastNonEmptyLine.empty())
                            isValid = false;
                        else
                        {
                            tokens.clear();
                            auto container = boost::make_iterator_range(&m_lastNonEmptyLine[0], &m_lastNonEmptyLine[0] + m_lastNonEmptyLine.size());
                            boost::split(tokens, container, boost::is_any_of(" "));
                            auto range = MLFFrameRange::ParseFrameRange(tokens);
                            sd.m_numberOfSamples = static_cast<uint32_t>(range.second);
                        }

                        if (isValid)
                            m_index.AddSequence(sd);
                        else
                            fprintf(stderr, "WARNING: Cannot parse the utterance %s at offset (%" PRIu64 ")\n", corpus->IdToKey(sd.m_key.m_sequence).c_str(), sd.m_fileOffsetBytes);
                    }
                }
                break;
                default:
                    LogicError("Unexpected MLF state.");
                }

                lastNonEmptyString = i;
            }

            // Remembering last non empty string to be able to retrieve time frame information 
            // when dot is encoutered on the border of two buffers.
            if (lastNonEmptyString != SIZE_MAX)
                m_lastNonEmptyLine = std::string(lines[lastNonEmptyString].begin(), lines[lastNonEmptyString].end());
            else
                m_lastNonEmptyLine.clear();

            RefillBuffer();
        }
    }

    void MLFIndexer::ReadLines(vector<char>& buffer, vector<boost::iterator_range<char*>>& lines)
    {
        lines.clear();
        auto range = boost::make_iterator_range(buffer.data(), buffer.data() + buffer.size());
        boost::split(lines, range, boost::is_any_of("\r\n"));
    }

    bool MLFIndexer::TryParseSequenceId(const boost::iterator_range<char*>& line, size_t& id, std::function<size_t(const std::string&)> keyToId)
    {
        id = 0;

        std::string key(line.begin(), line.end());
        boost::trim_right(key);

        if (key.size() > 2 && key.front() == '"' && key.back() == '"')
        {
            key = key.substr(1, key.size() - 2);
            if (key.size() > 2 && key[0] == '*' && key[1] == '/')
                key = key.substr(2);

            // Remove extension if specified.
            key = key.substr(0, key.find_last_of("."));

            id = keyToId(key);
            return true;
        }

        return false;
    }
}}}
