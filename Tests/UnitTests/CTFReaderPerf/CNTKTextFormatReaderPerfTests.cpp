//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx2.h"
#include <algorithm>
#ifdef _WIN32
#include <io.h>
#else // On Linux
#define _dup2 dup2
#define _dup dup
#define _close close
#define _fileno fileno
#endif
#include <cstdio>
#include "BufferedFileReader.h"
#include "FileWrapper.h"
#include "Index.h"
#include "Platform.h"
#include "IndexBuilder.h"
#include "ReaderUtil.h"
// #include "Common/ReaderTestHelper.h"
#include "TextParser.h"
#include <iostream>

using namespace Microsoft::MSR::CNTK;

// #pragma warning(disable: 4459) // declaration of 'boost_scope_exit_aux_args' hides global declaration

namespace CNTK {

    // A thin wrapper around CNTK text format reader
    template <class ElemType>
    class CNTKTextFormatReaderTestRunner
    {
        TextParser<ElemType> m_parser;

    public:
        ChunkPtr m_chunk;

        CNTKTextFormatReaderTestRunner(const string& filename,
            const vector<StreamDescriptor>& streams, unsigned int maxErrors) :
            m_parser(std::make_shared<CorpusDescriptor>(true), wstring(filename.begin(), filename.end()), streams, true)
        {
            m_parser.SetMaxAllowedErrors(maxErrors);
            m_parser.SetTraceLevel(TextParser<ElemType>::TraceLevel::Info);
            m_parser.SetChunkSize(SIZE_MAX);
            m_parser.SetNumRetries(0);
            m_parser.Initialize();
        }
        // Retrieves a chunk of data.
        void LoadChunk()
        {
            m_chunk = m_parser.GetChunk(0);
        }
    };
}

int main (int argc, char* argv[]) {
#if 0
    std::vector<::CNTK::StreamDescriptor> streams(3);
    streams[0].m_alias = "F0";
    streams[0].m_name = L"F0";
    streams[0].m_storageFormat = ::CNTK::StorageFormat::Dense;
    streams[0].m_sampleDimension = 10;

    streams[1].m_alias = "F1";
    streams[1].m_name = L"F1";
    streams[1].m_storageFormat = ::CNTK::StorageFormat::Dense;
    streams[1].m_sampleDimension = 50;

    streams[2].m_alias = "F2";
    streams[2].m_name = L"F2";
    streams[2].m_storageFormat = ::CNTK::StorageFormat::Dense;
    streams[2].m_sampleDimension = 100;

    ::CNTK::CNTKTextFormatReaderTestRunner<float> testRunner("100x100x3_jagged_sequences_dense.txt", streams, 99999);
#endif

    std::vector<::CNTK::StreamDescriptor> streams(2);
    streams[0].m_alias = "M";
    streams[0].m_name = L"M";
    streams[0].m_storageFormat = ::CNTK::StorageFormat::SparseCSC;
    // streams[0].m_sampleDimension = 165393;

    streams[1].m_alias = "R";
    streams[1].m_name = L"R";
    streams[1].m_storageFormat = ::CNTK::StorageFormat::SparseCSC;
    // streams[1].m_sampleDimension = 165393;
    ::CNTK::CNTKTextFormatReaderTestRunner<float> testRunner("/HOST/home/thiagofc/train.2GB.ctf", streams, 999999);


    testRunner.LoadChunk();
    std::cout << "CTF was loaded!" << std::endl;


    return 0;
}