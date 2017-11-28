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
        size_t sequenceStartOffset = 0; // Offset in file where current sequence starts.
        IndexedSequence sequence;
        sequence.SetKey(id)
            .SetNumberOfSamples(0)
            .SetOffset(sequenceStartOffset)
            .SetSize(0 - sequenceStartOffset);
        index->AddSequence(sequence);
    }

}
