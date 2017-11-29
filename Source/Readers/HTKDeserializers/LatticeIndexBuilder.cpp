//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include "LatticeIndexBuilder.h"
#include "MLFUtils.h"
#include "ReaderUtil.h"

namespace CNTK {

    using namespace std;

    LatticeIndexBuilder::LatticeIndexBuilder(const FileWrapper& input, CorpusDescriptorPtr corpus)
        : IndexBuilder(input)
    {
        IndexBuilder::SetCorpus(corpus);
        IndexBuilder::SetChunkSize(g_64MB);

        if (m_corpus == nullptr)
            InvalidArgument("LatticeIndexBuilder: corpus descriptor was not specified.");
        // Lattice index builder does not need to map sequence keys to locations,
        // deserializer will do that instead, set primary to true to skip that step.
        m_primary = true;
    }

    /*virtual*/ wstring LatticeIndexBuilder::GetCacheFilename() /*override*/
    {
        if (m_isCacheEnabled && !m_corpus->IsNumericSequenceKeys() && !m_corpus->IsHashingEnabled())
            InvalidArgument("Index caching is not supported for non-numeric sequence keys "
                "using in a corpus with disabled hashing.");

        wstringstream  wss;
        wss << m_input.Filename() << "."
            << (m_corpus->IsNumericSequenceKeys() ? "1" : "0") << "."
            << (m_corpus->IsHashingEnabled() ? std::to_wstring(CorpusDescriptor::s_hashVersion) : L"0") << "."
            << L"v" << IndexBuilder::s_version << "."
            << L"cache";

        return wss.str();
    }

    // Building an index of the Lattice file
    /*virtual*/ void LatticeIndexBuilder::Populate(shared_ptr<Index>& index) /*override*/
    {
        m_input.CheckIsOpenOrDie();

        index->Reserve(filesize(m_input.File()));

        BufferedFileReader reader(m_bufferSize, m_input);

        if (reader.Empty())
            RuntimeError("Input file is empty");

        if (!m_corpus)
            RuntimeError("LatticeIndexBuilder: corpus descriptor was not specified.");


        size_t id = 0;
        vector<boost::iterator_range<char*>> tokens;
        bool isValid = true; // Flag indicating whether the current sequence is valid.

        string line;
        string latticeFile = "";
        while (true)
        {
            if (!reader.TryReadLine(line))
                break;

            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            if (line.empty())
                continue;
            size_t spLoc = line.find('=');
            size_t openBracketLoc = line.find('[');
            size_t closeBracketLoc = line.find(']');
            
            string seqKey = line.substr(0, spLoc);
            
            size_t byteOffset = -1;
            sscanf(line.substr(openBracketLoc + 1, closeBracketLoc).c_str(), "%zu", &byteOffset);
            
            string curLatticeFile = line.substr(spLoc + 1, openBracketLoc);
            if (curLatticeFile.empty()) {
                if (latticeFile.empty()) 
                    RuntimeError("The lattice TOC file is malformed. Reference to the binary lattice file is missing.");
            }
            else {
                latticeFile = curLatticeFile;
            }
            
                //This line should contain pointer to the lattice file
                
                latticeFile
            }


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
                if (line != ".")
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

                    const static vector<bool> delim = DelimiterHash({ ' ' });
                    Split(&lastNonEmptyLine[0], &lastNonEmptyLine[0] + lastNonEmptyLine.size(), delim, tokens);

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

        size_t sequenceStartOffset = 0; // Offset in file where current sequence starts.
        IndexedSequence sequence;
        sequence.SetKey(id)
            .SetNumberOfSamples(0)
            .SetOffset(sequenceStartOffset)
            .SetSize(0 - sequenceStartOffset);
        index->AddSequence(sequence);
    }

}
