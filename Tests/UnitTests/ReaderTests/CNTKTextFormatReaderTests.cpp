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

BOOST_AUTO_TEST_CASE(CNTKTextFormatReaderSimple)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReaderSimple_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReaderSimple_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReaderSimple_Output.txt",
        "Simple_Test",
        "reader",
        1000, // epoch size
        250,  // mb size
        10,   // num epochs
        1,
        1,
        0,
        1);
};


BOOST_AUTO_TEST_CASE(CNTKTextFormatReaderMNIST)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReaderMNIST_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReaderMNIST_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReaderMNIST_Output.txt",
        "MNIST_Test",
        "reader",
        1000, // epoch size
        1000,  // mb size
        1,   // num epochs
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x1_1)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader1x1_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x1_1_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x1_1_Output.txt",
        "1x1_Test",
        "reader",
        1, // epoch size
        1,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x1_2)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader1x1_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x1_2_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x1_2_Output.txt",
        "1x1_Test",
        "reader",
        2, // epoch size
        1,  // mb size
        3,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x100_1)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader1x100_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x100_1_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x100_1_Output.txt",
        "1x100_Test",
        "reader",
        10, // epoch size
        1,  // mb size
        10,  // num epochs
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x10_MI_2)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader1x10_MI_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x10_MI_2_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x10_MI_2_Output.txt",
        "1x10_MI_Test",
        "reader",
        7, // epoch size
        3, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x100_2)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader1x100_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x100_2_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x100_2_Output.txt",
        "1x100_Test",
        "reader",
        5, // epoch size
        3,  // mb size
        4,  // num epochs
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader1x10_MI_1)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader1x10_MI_Config.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader1x10_MI_1_Control.txt",
        testDataPath() + "/Control/CNTKTextFormatReader1x10_MI_1_Output.txt",
        "1x10_MI_Test",
        "reader",
        10, // epoch size
        1, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

BOOST_AUTO_TEST_SUITE_END()

} } } }
