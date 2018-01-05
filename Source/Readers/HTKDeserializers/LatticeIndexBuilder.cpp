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

    LatticeIndexBuilder::LatticeIndexBuilder(const FileWrapper& latticeFile, const std::vector<std::string>& latticeToc, CorpusDescriptorPtr corpus)
        : IndexBuilder(latticeFile), m_latticeToc(latticeToc)
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


        size_t id = 0;
        IndexedSequence sequence;
        string latticeFile = "";

        size_t prevId{ 0 };
        bool firstLine = true;
        size_t prevSequenceStartOffset{ 0 };
        for (string const& line : m_latticeToc)
        {
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
            size_t fileStart = eqLoc + 1;
            string curLatticeFile = line.substr(fileStart, openBracketLoc - fileStart);
            if (curLatticeFile.empty()) 
            {
                //This line should contain pointer to the lattice file
                if (latticeFile.empty()) 
                    RuntimeError("The lattice TOC file is malformed. Reference to the binary lattice file is missing.");
            }
            else
            {
                latticeFile = curLatticeFile;
            }
            id = m_corpus->KeyToId(seqKey);

            if (firstLine)
            {
                firstLine = false;
            }
            else
            {
                auto seqSize = (uint32_t) (byteOffset - prevSequenceStartOffset);
                if (seqSize % sizeof(float) == 0)
                    seqSize = seqSize / sizeof(float);
                else
                    seqSize = seqSize / sizeof(float)+1;

                sequence.SetKey(prevId)
                    .SetNumberOfSamples(seqSize)
                    .SetOffset(prevSequenceStartOffset)
                    .SetSize(byteOffset - prevSequenceStartOffset);
                index->AddSequence(sequence);
            }
            prevId = id;
            prevSequenceStartOffset = byteOffset;
        }
        size_t fileSize = filesize(m_input.File());
        auto seqSize = (uint32_t)(fileSize - prevSequenceStartOffset);
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

}
