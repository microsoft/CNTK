//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include "MLFBinaryIndexBuilder.h"
#include "MLFUtils.h"
#include "ReaderUtil.h"
#include "assert.h"

namespace CNTK {

    using namespace std;

    MLFBinaryIndexBuilder::MLFBinaryIndexBuilder(const FileWrapper& input, CorpusDescriptorPtr corpus)
        : IndexBuilder(input)
    {
        IndexBuilder::SetCorpus(corpus);
        IndexBuilder::SetChunkSize(g_64MB);

        if (m_corpus == nullptr)
            InvalidArgument("MLFBinaryIndexBuilder: corpus descriptor was not specified.");
        // MLF index builder does not need to map sequence keys to locations,
        // deserializer will do that instead, set primary to true to skip that step.
        m_primary = true;
    }

    /*virtual*/ wstring MLFBinaryIndexBuilder::GetCacheFilename() /*override*/
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

    // Building an index of the MLF file:
    //     MLF file -> MLF Header [MLF Utterance]+
    //     MLF Utterance -> Key EOL [Frame Range EOL]+ "." EOL
    // MLF file should start with the MLF header (State::Header -> State:UtteranceKey).
    // Each utterance starts with an utterance key (State::UtteranceKey -> State::UtteranceFrames).
    // End of utterance is indicated by a single dot on a line (State::UtteranceFrames -> State::UtteranceKey)

    /*virtual*/ void MLFBinaryIndexBuilder::Populate(shared_ptr<Index>& index) /*override*/
    {
        m_input.CheckIsOpenOrDie();

        index->Reserve(filesize(m_input.File()));

        BufferedFileReader reader(m_bufferSize, m_input);

        if (reader.Empty())
            RuntimeError("Input file is empty");
   
        if (!m_corpus)
            RuntimeError("MLFBinaryIndexBuilder: corpus descriptor was not specified.");
        vector<char> buffer(4);

        // Validate file label
        assert(reader.TryReadBinarySegment(3, buffer.data()));
        std::string mlfLabel(buffer.data(),3);
        assert(mlfLabel == MLF_BIN_LABEL);

        //Validate MLF format version
        assert(reader.TryReadBinarySegment(sizeof(short), buffer.data()));
        short* pModelVersion = (short*)buffer.data();
        assert(*pModelVersion == MODEL_VERSION);

        auto sequenceStartOffset = reader.GetFileOffset();

        // Iterate over the bin MLF
        while (reader.TryReadBinarySegment(sizeof(uint), buffer.data()))
        {
            
            uint uttrKey = *(uint*)buffer.data();
            auto uttrId = m_corpus->KeyToId(std::to_string(uttrKey));
            fprintf(stderr, "Reading: %zu\n", uttrId);

            uint32_t uttrFrameCount = 0;
            // Read size of this uttrs
            assert(reader.TryReadBinarySegment(sizeof(ushort), buffer.data()));
            ushort uttrSamplesCount = *(ushort*)buffer.data();

            //fprintf(stderr, "uttrSamplesCount: %zu\n", uttrSamplesCount);

            for (size_t k = 0;k < uttrSamplesCount;k++) {
                assert(reader.TryReadBinarySegment(sizeof(ushort), buffer.data()));
                assert(reader.TryReadBinarySegment(sizeof(ushort), buffer.data()));
                ushort stateCount = *(ushort*)buffer.data();
                uttrFrameCount += stateCount;
            }
            fprintf(stderr, "uttrFrameCount: %lu\n", uttrFrameCount);

            // uttrid, sample count, senone/count pairs
            size_t uttrSize = sizeof(uint) + sizeof(ushort) + uttrSamplesCount * 2 * sizeof(ushort);

            IndexedSequence sequence;
            sequence.SetKey(uttrId)
                        .SetNumberOfSamples(uttrFrameCount)
                        .SetOffset(sequenceStartOffset)
                        .SetSize(uttrSize);
            index->AddSequence(sequence);
             
            sequenceStartOffset = reader.GetFileOffset();
            fprintf(stderr, "sequenceStartOffset: %zu\n", sequenceStartOffset);
                    
           // reader.SetFileOffset(sequenceStartOffset);
        }
    }

}
