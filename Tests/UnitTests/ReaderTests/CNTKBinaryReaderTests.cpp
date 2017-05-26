//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <algorithm>
#include <boost/scope_exit.hpp>
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct CNTKBinaryReaderFixture : ReaderFixture
{
    CNTKBinaryReaderFixture()
        : ReaderFixture("/Data/CNTKBinaryReader/")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, CNTKBinaryReaderFixture)


BOOST_AUTO_TEST_CASE(CNTKBinaryReader_Simple_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/Simple_dense.txt",
        testDataPath() + "/Control/CNTKBinaryReader/Simple_dense_Output.txt",
        "Simple",
        "reader",
        1000, // epoch size
        250,  // mb size
        10,   // num epochs 
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKBinaryReader_MNIST_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/MNIST_dense.txt",
        testDataPath() + "/Control/CNTKBinaryReader/MNIST_dense_Output.txt",
        "MNIST",
        "reader",
        1000, // epoch size
        1000,  // mb size
        1,   // num epochs
        1,
        1,
        0,
        1);
};

// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKBinaryReader_10x10_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_dense.txt",
        testDataPath() + "/Control/CNTKBinaryReader/10x10_dense_Output.txt",
        "10x10_dense",
        "reader",
        100, // epoch size
        100,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};


// 50 sequences with up to 20 samples each (508 samples in total)
BOOST_AUTO_TEST_CASE(CNTKBinaryReader_50x20_jagged_sequences_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_dense.txt",
        testDataPath() + "/Control/CNTKBinaryReader/50x20_jagged_sequences_dense_Output.txt",
        "50x20_jagged_sequences_dense",
        "reader",
        508,  // epoch size
        508,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1);
};

// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKBinaryReader_10x10_sparse)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_sparse.txt",
        testDataPath() + "/Control/CNTKBinaryReader/10x10_sparse_Output.txt",
        "10x10_sparse",
        "reader",
        100, // epoch size
        100, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 50 sequences with up to 20 samples each (536 samples in total)
BOOST_AUTO_TEST_CASE(CNTKBinaryReader_50x20_jagged_sequences_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKBinaryReader/test.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_sparse.txt",
        testDataPath() + "/Control/CNTKBinaryReader/50x20_jagged_sequences_sparse_Output.txt",
        "50x20_jagged_sequences_sparse",
        "reader",
        564,  // epoch size
        564,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1,
        true);
};

BOOST_AUTO_TEST_SUITE_END()

} } } }
