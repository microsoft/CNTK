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

    bool readUtteranceLabel(short modelVersion, BufferedFileReader& reader, vector<char>& buffer, std::string& out)
    {
        bool result = false;
        if (modelVersion == 1)
        {
            result = reader.TryReadBinarySegment(sizeof(uint), buffer.data());
            uint uttrKey = *(uint*)buffer.data();
            out = std::to_string(uttrKey);
        }
        else if (modelVersion == 2)
        {
            result = reader.TryReadBinarySegment(sizeof(ushort), buffer.data());
            if (!result)
                return false;
            ushort uttLabelLength = *(ushort*)buffer.data();

            result = result && reader.TryReadBinarySegment(sizeof(char) * uttLabelLength, buffer.data());
            if (!result)
                return false;

            out = std::string(buffer.data()).substr(0, uttLabelLength);

            if (uttLabelLength > MAX_UTTERANCE_LABEL_LENGTH)
                RuntimeError("Utterance label length is greater than limit %hu: %s", uttLabelLength, out.c_str());
        }
        else 
        {
            RuntimeError("Not supported MLF model version.");
        }

        return result;
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
        vector<char> buffer(MAX_UTTERANCE_LABEL_LENGTH);

        // Validate file label
        reader.TryReadBinarySegment(3, buffer.data());
        std::string mlfLabel(buffer.data(),3);
        if (mlfLabel != MLF_BIN_LABEL)
            RuntimeError("MLFBinaryIndexBuilder: MLF binary file is malformed.");

        //Validate MLF format version
        reader.TryReadBinarySegment(sizeof(short), buffer.data());
        short modelVersion = *(short*)buffer.data();

        // Iterate over the bin MLF
        string uttrKey;
        while (readUtteranceLabel(modelVersion, reader, buffer, uttrKey))
        {
            auto uttrId = m_corpus->KeyToId(uttrKey);

            reader.TryReadBinarySegment(sizeof(uint), buffer.data());
            uint uttrFrameCount = *(uint*)buffer.data();

            auto sequenceStartOffset = reader.GetFileOffset();

            // Read size of this uttrs
            reader.TryReadBinarySegment(sizeof(ushort), buffer.data());
            ushort uttrSamplesCount = *(ushort*)buffer.data();

            // sample count, senone/count pairs
            size_t uttrSize =sizeof(ushort) + uttrSamplesCount * 2 * sizeof(ushort);

            IndexedSequence sequence;
            sequence.SetKey(uttrId)
                        .SetNumberOfSamples(uttrFrameCount)
                        .SetOffset(sequenceStartOffset)
                        .SetSize(uttrSize);
            index->AddSequence(sequence);
            reader.SetFileOffset(reader.GetFileOffset() + uttrSamplesCount * 2 * sizeof(ushort));
        }
    }

}
