//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include "MLFIndexBuilder.h"
#include "MLFUtils.h"
#include "ReaderUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    static bool SingleDot(const boost::iterator_range<char*>& line)
    {
        return distance(line.begin(), line.end()) == 1 && *line.begin() == '.';
    }

    MLFIndexBuilder::MLFIndexBuilder(const std::wstring& filename, FILE* file, CorpusDescriptorPtr corpus)
        : IndexBuilder(filename, file),
        m_fileOffsetStart(0),
        m_done(false)
    {
        if (!m_file)
            RuntimeError("Input file not open for reading");
        // MLF index builder does not need to map sequence keys to locations,
        // deserializer will do that instead, set primary to true to skip that step.
        m_primary = true; 
        IndexBuilder::SetCorpus(corpus);
    }

    /*virtual*/ std::wstring MLFIndexBuilder::GetCacheFilename() /*override*/
    {
        std::wstringstream  wss;
        wss << m_filename << "."
            << (m_frameMode ? "1" : "0") << "."
            << ((m_corpus && !m_corpus->IsNumericSequenceKeys()) ? "1" : "0") << "."
            << ((m_corpus && m_corpus->IsHashingEnabled()) ? "1" : "0") << "."
            << L"v" << IndexBuilder::s_version << "."
            << L"cache";

        return wss.str();
    }

    // Building an index of the MLF file:
    //     MLF file -> MLF Header [MLF Utterance]+
    //     MLF Utterance -> Key EOL [Frame Range EOL]+ "." EOL
    // MLF file should start with the MLF header (State::Header -> State:UtteranceKey).
    // Each utterance starts with an utterance key (State::UtteranceKey -> State::UtteranceFrames).
    // End of utterance is indicated by a single dot on a line (State::UtteranceFrames -> State::UtteranceKey)

    /*virtual*/ void MLFIndexBuilder::Populate(shared_ptr<Index>& index) /*override*/
    {
        index->Reserve(filesize(m_file));

        RefillBuffer(); // read the first block of data
        if (m_done)
            RuntimeError("Input file is empty");
   
        if (!m_corpus)
            RuntimeError("Corpus descriptor was not specified");


        size_t id = 0;
        State currentState = State::Header;
        vector<boost::iterator_range<char*>> tokens;
        bool isValid = true;                    // Flag indicating whether the current sequence is valid.
        size_t lastNonEmptyString = 0;          // Needed to parse information about last frame
        size_t sequenceStartOffset = 0;         // Offset in file where current sequence starts.
        IndexedSequence sequence;
        while (!m_done)
        {
            ReadLines();

            lastNonEmptyString = SIZE_MAX;
            for (size_t i = 0; i < m_lines.size(); i++)
            {
                const auto& line = m_lines[i];

                if (line.begin() == line.end()) // Skip all empty lines.
                    continue;

                switch (currentState)
                {
                case State::Header:
                {
                    if (string(line.begin(), line.end()) != "#!MLF!#")
                        RuntimeError("Expected MLF header was not found.");
                    currentState = State::UtteranceKey;
                }
                break;
                case State::UtteranceKey:
                {
                    // When several files are appended to a big mlf, there can be
                    // an MLF header between the utterances.
                    if (string(line.begin(), line.end()) == "#!MLF!#")
                        continue;

                    sequenceStartOffset = m_fileOffsetStart + line.begin() - m_buffer.data();
                    isValid = TryParseSequenceKey(line, id, m_corpus->KeyToId);
                    currentState = State::UtteranceFrames;
                }
                break;

                case State::UtteranceFrames:
                {
                    if (!SingleDot(line))
                        break; // Still current utterance.

                               // Ok, a single . on a line means we found the end of the utterance.
                    auto sequenceEndOffset = m_fileOffsetStart + line.end() - m_buffer.data(); 

                    // Let's find last non empty string and parse information about frames out of it.
                    // Here we assume that the sequence is correct, if not - it will be invalidated later
                    // when the actual data is read.
                    if (lastNonEmptyString != SIZE_MAX)
                        m_lastNonEmptyLine = string(m_lines[lastNonEmptyString].begin(), m_lines[lastNonEmptyString].end());

                    uint32_t numberOfSamples = 0;
                    if (m_lastNonEmptyLine.empty())
                        isValid = false;
                    else
                    {
                        tokens.clear();
                        auto container = boost::make_iterator_range(&m_lastNonEmptyLine[0], &m_lastNonEmptyLine[0] + m_lastNonEmptyLine.size());

                        const static std::vector<bool> delim = DelimiterHash({ ' ' });
                        Split(container.begin(), container.end(), delim, tokens);

                        auto range = MLFFrameRange::ParseFrameRange(tokens, sequenceEndOffset);
                        numberOfSamples = static_cast<uint32_t>(range.second);
                    }

                    if (isValid)
                    {
                        
                        sequence.SetKey(id)
                            .SetNumberOfSamples(numberOfSamples)
                            .SetOffset(sequenceStartOffset)
                            .SetSize(sequenceEndOffset - sequenceStartOffset + 2);
                        index->AddSequence(sequence);
                    }
                    else
                        fprintf(stderr, "WARNING: Cannot parse the utterance '%s' at offset (%" PRIu64 ")\n", m_corpus->IdToKey(id).c_str(), sequenceStartOffset);
                    currentState = State::UtteranceKey; // Let's try the next one.
                }
                break;
                default:
                    LogicError("Unexpected MLF state.");
                }

                lastNonEmptyString = i;
            }

            // Remembering last non empty string to be able to retrieve time frame information 
            // when the dot is just at the beginning of the next buffer.
            if (lastNonEmptyString != SIZE_MAX)
                m_lastNonEmptyLine = string(m_lines[lastNonEmptyString].begin(), m_lines[lastNonEmptyString].end());
            else
                m_lastNonEmptyLine.clear();

            RefillBuffer();
        }

        // Clean the buffer.
        std::vector<char> tmp;
        m_buffer.swap(tmp);
    }

    void MLFIndexBuilder::RefillBuffer()
    {
        if (m_done)
            return;

        // Move start forward according to the already read buffer.
        m_fileOffsetStart += m_buffer.size();

        // Read the new portion of data into the buffer.
        m_buffer.resize(m_bufferSize);

        // Copy last partial line if it was left during the last read.
        memcpy(&m_buffer[0], m_lastPartialLineInBuffer.data(), m_lastPartialLineInBuffer.size());

        size_t bytesRead = fread(&m_buffer[0] + m_lastPartialLineInBuffer.size(), 1, m_buffer.size() - m_lastPartialLineInBuffer.size(), m_file);
        if (bytesRead == (size_t)-1)
            RuntimeError("Could not read from the input file.");

        if (bytesRead == 0) // End of file reached.
        {
            boost::trim(m_lastPartialLineInBuffer);
            if (!m_lastPartialLineInBuffer.empty()) // It seems like a corrupted file at the end.
                RuntimeError("Unexpected line at the end of the file '%s'", m_lastPartialLineInBuffer.c_str());

            m_buffer.clear();
            m_lastPartialLineInBuffer.clear();
            m_done = true;
            return;
        }

        size_t readBufferSize = m_lastPartialLineInBuffer.size() + bytesRead;

        // Let's find the last LF.
        int lastLF = 0;
        {
            // Let's find the latest \n if exists.
            for (lastLF = (int)readBufferSize - 1; lastLF >= 0; lastLF--)
            {
                if (m_buffer[lastLF] == '\n')
                    break;
            }

            if (lastLF < 0)
                RuntimeError("Length of MLF sequence cannot exceed '%zu' bytes.", readBufferSize);
        }

        // Let's cut the buffer at the last EOL and save partial string
        // in m_lastPartialLineInBuffer.
        auto logicalBufferSize = lastLF + 1;
        auto lastPartialLineSize = readBufferSize - logicalBufferSize;

        // Remember the last parital line.
        m_lastPartialLineInBuffer.resize(lastPartialLineSize);
        if (lastPartialLineSize)
            memcpy(&m_lastPartialLineInBuffer[0], m_buffer.data() + logicalBufferSize, lastPartialLineSize);
        m_buffer.resize(logicalBufferSize);
    }

    void MLFIndexBuilder::ReadLines()
    {
        m_lines.clear();

        // TODO: '\n' should be enough as the delimiter EOL. Having both might create extra empty lines.
        const static std::vector<bool> delim = DelimiterHash({ '\r', '\n' }); 
        Split(m_buffer.data(), m_buffer.data() + m_buffer.size(), delim, m_lines);
    }

    // Tries to parse sequence key
    // In MLF a sequence key should be in quotes. During parsing the extension should be removed.
    bool MLFIndexBuilder::TryParseSequenceKey(const boost::iterator_range<char*>& line, size_t& id, function<size_t(const string&)> keyToId)
    {
        id = 0;

        string key(line.begin(), line.end());
        boost::trim_right(key);

        if (key.size() <= 2 || key.front() != '"' || key.back() != '"')
            return false;

        key = key.substr(1, key.size() - 2);
        if (key.size() > 2 && key[0] == '*' && key[1] == '/') // Preserving the old behavior
            key = key.substr(2);

        // Remove extension if specified.
        key = key.substr(0, key.find_last_of("."));

        id = keyToId(key);
        return true;
    }
}}}
