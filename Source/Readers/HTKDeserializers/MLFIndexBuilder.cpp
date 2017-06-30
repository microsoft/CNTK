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

    inline bool SingleDot(const std::string& line)
    {
        return line.size() == 1 && line[0] == '.';
    }

    MLFIndexBuilder::MLFIndexBuilder(const std::wstring& filename, FILE* file, CorpusDescriptorPtr corpus)
        : IndexBuilder(filename, file)
    {
        // MLF index builder does not need to map sequence keys to locations,
        // deserializer will do that instead, set primary to true to skip that step.
        m_primary = true; 
        IndexBuilder::SetCorpus(corpus);
        IndexBuilder::SetChunkSize(g_64MB);
    }

    /*virtual*/ std::wstring MLFIndexBuilder::GetCacheFilename() /*override*/
    {
        std::wstringstream  wss;
        wss << m_filename << "."
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
        if (!m_file)
            RuntimeError("Input file not open for reading");

        index->Reserve(filesize(m_file));

        BufferedFileReader reader(m_bufferSize, m_file);

        if (reader.Empty())
            RuntimeError("Input file is empty");
   
        if (!m_corpus)
            RuntimeError("Corpus descriptor was not specified");


        size_t id = 0;
        State currentState = State::Header;
        vector<boost::iterator_range<char*>> tokens;
        bool isValid = true; // Flag indicating whether the current sequence is valid.
        size_t sequenceStartOffset = 0; // Offset in file where current sequence starts.
        std::string lastNonEmptyLine; // Needed to parse information about last frame
        IndexedSequence sequence;
        string line;
        while (true)
        {
            auto offset = reader.GetFileOffset();

            if (!reader.TryReadLine(line))
                break;

            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            switch (currentState)
            {
            case State::Header:
            {
                if (line != "#!MLF!#")
                    RuntimeError("Expected MLF header was not found.");
                currentState = State::UtteranceKey;
            }
            break;
            case State::UtteranceKey:
            {
                // When several files are appended to a big mlf, there can be
                // an MLF header between the utterances.
                if (line == "#!MLF!#")
                    continue;

                lastNonEmptyLine.clear();

                sequenceStartOffset = offset;
                isValid = TryParseSequenceKey(line, id, m_corpus->KeyToId);
                currentState = State::UtteranceFrames;
            }
            break;
            case State::UtteranceFrames:
            {
                if (!SingleDot(line))
                {
                    // Remembering last non empty string to be able to retrieve time frame information 
                    // when the dot is just at the beginning of the next buffer.
                    lastNonEmptyLine = line;
                    continue; // Still current utterance.
                }   

                // Ok, a single . on a line means we found the end of the utterance.
                auto sequenceEndOffset = reader.GetFileOffset();

                uint32_t numberOfSamples = 0;
                if (lastNonEmptyLine.empty())
                    isValid = false;
                else
                {
                    tokens.clear();
                    auto container = boost::make_iterator_range(&lastNonEmptyLine[0], &lastNonEmptyLine[0] + lastNonEmptyLine.size());

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
                        .SetSize(sequenceEndOffset - sequenceStartOffset);
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
        }
    }


    // Tries to parse sequence key
    // In MLF a sequence key should be in quotes. During parsing the extension should be removed.
    bool MLFIndexBuilder::TryParseSequenceKey(const std::string& line, size_t& id, function<size_t(const string&)> keyToId)
    {
        id = 0;

        string key(line);

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
