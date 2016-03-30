//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"

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

BOOST_AUTO_TEST_CASE(CNTKTextFormatReaderSimple_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
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


BOOST_AUTO_TEST_CASE(CNTKTextFormatReaderMNIST_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
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

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x1_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
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

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x1_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
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

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x10_MI_2_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_MI_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_MI_2_dense_Output.txt",
        "1x10_MI",
        "reader",
        7, // epoch size
        3, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x10_MI_1_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_MI_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_MI_1_dense_Output.txt",
        "1x10_MI",
        "reader",
        10, // epoch size
        1, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x100_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x100_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x100_1_dense_Output.txt",
        "1x100",
        "reader",
        10, // epoch size
        1,  // mb size
        10,  // num epochs
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x100_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader_dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x100_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x100_2_dense_Output.txt",
        "1x100",
        "reader",
        5, // epoch size
        3,  // mb size
        4,  // num epochs
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_SUITE_END()

} } } }
