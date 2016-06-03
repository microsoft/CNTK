//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <algorithm>
#include <io.h>
#include <cstdio>
#include <boost/scope_exit.hpp>
#include "Common/ReaderTestHelper.h"
#include "BinaryChunkDeserializer.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK {

// A thin wrapper around CNTK text format reader
class CNTKBinaryReaderTestRunner
{
    BinaryChunkDeserializer m_deserializer;

public:
    ChunkPtr m_chunk;

    CNTKBinaryReaderTestRunner(const string& filename) :
        m_deserializer(wstring(filename.begin(), filename.end()))
    {
    }
    // Retrieves a chunk of data.
    void LoadChunk()
    {
        m_chunk = m_deserializer.GetChunk(0);
    }
};

namespace Test {

struct CNTKBinaryReaderFixture : ReaderFixture
{
    CNTKBinaryReaderFixture()
        : ReaderFixture("/Data/CNTKBinaryReader/")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, CNTKBinaryReaderFixture)

// Note that the output for this is the same as the output for sparse seq with two matrices removed.
// I believe this is actually incorrect, as it means that the packer is packing less data than it 
// should/could, as it packs the largest matrix with only 150 samples, not 250. Is this a bug?
BOOST_AUTO_TEST_CASE(CNTKBinaryReader_sparse_seq_part)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse_seq_part.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse_seq_part_Output.txt",
        "SparseSeqPart",
        "reader",
        1600, // epoch size
        250,  // mb size
        1,   // num epochs 
        1,
        1,
        0,
        1, true, false, false);
};

BOOST_AUTO_TEST_CASE(CNTKBinaryReader_sparse_seq)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse_seq.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse_seq_Output.txt",
        "SparseSeq",
        "reader",
        1600, // epoch size
        250,  // mb size
        1,   // num epochs 
        2,
        2,
        0,
        1, true, false, false);
};

BOOST_AUTO_TEST_CASE(CNTKBinaryReader_Simple_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_sparse_Output.txt",
        "Sparse",
        "reader",
        1600, // epoch size
        250,  // mb size
        1,   // num epochs 
        2,
        2,
        0,
        1, true, false, false);
};

BOOST_AUTO_TEST_CASE(CNTKBinaryReader_Simple_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_dense.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_dense_Output.txt",
        "Simple",
        "reader",
        1600, // epoch size
        250,  // mb size
        1,   // num epochs 
        4,
        0,
        0,
        1, false, false, false);
};

BOOST_AUTO_TEST_CASE(CNTKBinaryReader_Simple_dense2)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_dense_312.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_dense_312_Output.txt",
        "Simple",
        "reader",
        1600, // epoch size
        312,  // mb size
        1,   // num epochs 
        4,
        0,
        0,
        1, false, false, false);
};


BOOST_AUTO_TEST_SUITE_END()

} } } }
