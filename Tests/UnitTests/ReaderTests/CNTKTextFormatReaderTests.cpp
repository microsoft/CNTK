//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct CNTKTextFormatReaderFixture : ReaderFixture
{
    CNTKTextFormatReaderFixture()
        : ReaderFixture("/Data")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, CNTKTextFormatReaderFixture)

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_Simple_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/Simple_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/Simple_dense_Output.txt",
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


BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_MNIST_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/MNIST_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/MNIST_dense_Output.txt",
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

// 1 single sample sequence
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_1_dense_Output.txt",
        "1x1",
        "reader",
        1, // epoch size
        1,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 1 sequence with 2 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_2_dense_Output.txt",
        "1x1",
        "reader",
        2, // epoch size
        1,  // mb size
        3,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 1 sequence with 10 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_dense_Output.txt",
        "1x10",
        "reader",
        10, // epoch size
        10,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 10 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x1_MI_2_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_2_dense_Output.txt",
        "10x1_MI",
        "reader",
        7, // epoch size
        3, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

// 10 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_MI_1_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_1_dense_Output.txt",
        "10x1_MI",
        "reader",
        10, // epoch size
        1, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x10_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_dense_Output.txt",
        "10x10",
        "reader",
        100, // epoch size
        100,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 100 identical single sample sequences 
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x1_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_1_dense_Output.txt",
        "100x1",
        "reader",
        10, // epoch size
        1,  // mb size
        10,  // num epochs
        1,
        1,
        0,
        1);
};

// 100 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x1_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_2_dense_Output.txt",
        "100x1",
        "reader",
        5,  // epoch size
        3,  // mb size
        4,  // num epochs
        1,
        1,
        0,
        1);
};

// 50 sequences with up to 20 samples each (508 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_50x20_jagged_sequences_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_dense_Output.txt",
        "50x20_jagged_sequences",
        "reader",
        508,  // epoch size
        508,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1);
};

// 1 single sample sequence
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_sparse_Output.txt",
        "1x1",
        "reader",
        1, // epoch size
        1, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 1 sequence with 2 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x2_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x2_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x2_sparse_Output.txt",
        "1x2",
        "reader",
        2, // epoch size
        2, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 1 sequence with 10 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_sparse)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_sparse_Output.txt",
        "1x10",
        "reader",
        10, // epoch size
        10, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};


// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x10_sparse)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_sparse_Output.txt",
        "10x10",
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

// 3 sequences with 5 samples for each of 3 input stream (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_3x5_MI_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/3x5_MI_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/3x5_MI_sparse_Output.txt",
        "3x5_MI",
        "reader",
        15, // epoch size
        15, // mb size
        1, // num epochs
        3,
        0, // no labels
        0,
        1,
        true);
};

// 20 sequences with 10 samples for each of 3 input stream with
// random number of values in each sample (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_20x10_MI_jagged_samples_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/20x10_MI_jagged_samples_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/20x10_MI_jagged_samples_sparse_Output.txt",
        "20x10_MI_jagged_samples",
        "reader",
        200, // epoch size
        200, // mb size
        1, // num epochs
        3,
        0, // no labels
        0,
        1,
        true);
};

// 50 sequences with up to 20 samples each (536 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_50x20_jagged_sequences_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_sparse_Output.txt",
        "50x20_jagged_sequences",
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

// 50 sequences with up to 20 samples each and up to 20 values in each sample 
// (4887 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x100_jagged_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x100_jagged_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x100_jagged_sparse_Output.txt",
        "100x100_jagged",
        "reader",
        4887,  // epoch size
        4887,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1,
        true);
};


BOOST_AUTO_TEST_SUITE_END()

} } } }
