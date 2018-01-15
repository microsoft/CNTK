//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include "LatticeIndexBuilder.h"
#include "ReaderUtil.h"

namespace CNTK {

    using namespace std;

    LatticeIndexBuilder::LatticeIndexBuilder(const FileWrapper& latticeFile, const std::vector<std::string>& latticeToc, CorpusDescriptorPtr corpus, bool lastChunkInTOC)
        : IndexBuilder(latticeFile), m_latticeToc(latticeToc), m_lastChunkInTOC(lastChunkInTOC)
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

        if (m_latticeToc.size() == 0)
            RuntimeError("Lattice TOC is empty");

        if (!m_corpus)
            RuntimeError("LatticeIndexBuilder: corpus descriptor was not specified.");


        IndexedSequence sequence;

        size_t prevId{ 0 };
        bool firstLine = true;
        size_t prevSequenceStartOffset{ 0 };
        string prevSeqKey = "";
        for (string const& line : m_latticeToc)
        {
            fprintf(stderr, "Reading line '%s'\n", line.c_str());
            if (line.empty())
                continue;

            size_t eqLoc = line.find('=');
            size_t openBracketLoc = line.find('[');
            size_t closeBracketLoc = line.find(']');
            
            if (eqLoc == string::npos || openBracketLoc == string::npos || closeBracketLoc == string::npos)
                RuntimeError("The lattice TOC line is malformed: %s", line.c_str());

            string seqKey = line.substr(0, eqLoc);
            
            size_t byteOffset;
            sscanf(line.substr(openBracketLoc + 1, closeBracketLoc).c_str(), "%zu", &byteOffset);

            if (firstLine)
            {
                firstLine = false;
            }
            else
            {
                auto seqSize = (uint32_t) (byteOffset - prevSequenceStartOffset);
                fprintf(stderr, "SeqKey '%s', byteOffset '%zu', prevSequenceStartOffset '%zu' \n", seqKey.c_str(), byteOffset, prevSequenceStartOffset);
                if (byteOffset - prevSequenceStartOffset < 1073741824)
                {
                    if (seqSize % sizeof(float) == 0)
                        seqSize = seqSize / sizeof(float);
                    else
                        seqSize = seqSize / sizeof(float) + 1;

                    sequence.SetKey(prevId)
                        .SetNumberOfSamples(seqSize)
                        .SetOffset(prevSequenceStartOffset)
                        .SetSize(byteOffset - prevSequenceStartOffset);
                    index->AddSequence(sequence);
                }
                else
                {
                    fprintf(stderr, "WARNING: Lattice with the key '%s' inside the TOC file '%ls' more than 1GB. Skipping it...\n", prevSeqKey.c_str(), m_input.Filename().c_str());
                    LogicError("Breaking...");
                }
            }
            prevId = m_corpus->KeyToId(seqKey);
            prevSequenceStartOffset = byteOffset;
        }
        if (m_lastChunkInTOC) {
            size_t fileSize = filesize(m_input.File());
            auto seqSize = (uint32_t)(fileSize - prevSequenceStartOffset);
            fprintf(stderr, "Last line: byteOffset '%zu', prevSequenceStartOffset '%zu' \n", fileSize, prevSequenceStartOffset);
            if (fileSize - prevSequenceStartOffset < 1073741824)
            {
                if (seqSize % sizeof(float) == 0)
                    seqSize = seqSize / sizeof(float);
                else
                    seqSize = seqSize / sizeof(float) + 1;
                sequence.SetKey(prevId)
                    .SetNumberOfSamples(seqSize)
                    .SetOffset(prevSequenceStartOffset)
                    .SetSize(fileSize - prevSequenceStartOffset);
                index->AddSequence(sequence);
            }
            else
            {
                fprintf(stderr, "WARNING: Lattice with the key '%s' inside the TOC file '%ls' more than 1GB. Skipping it...\n", prevSeqKey.c_str(), m_input.Filename().c_str());
                LogicError("Breaking2...");
            }
        }
    }

}
