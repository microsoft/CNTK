//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// Fixture specific to the AN4 data
struct AN4ReaderFixture : ReaderFixture
{
    AN4ReaderFixture()
        : ReaderFixture(
              "%CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY%/Speech/AN4Corpus/v0",
              "This test uses external data that is not part of the CNTK repository. Environment variable CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY must be set to point to the external test data location. \n Refer to the 'Setting up CNTK on Windows' documentation.)")
    {
    }
};

struct iVectorFixture : ReaderFixture
{
    iVectorFixture()
        : ReaderFixture(
        "%CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY%/iVector",
        "This test uses external data that is not part of the CNTK repository. Environment variable CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY must be set to point to the external test data location. \n Refer to the 'Setting up CNTK on Windows' documentation.)")
    {
    }
};


// Use SpeechReaderFixture for most tests
// Some of them (e.g. 10, will use different data, thus a different fixture)
BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, AN4ReaderFixture)

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop1)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop1_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop2)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop2_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_12_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop3)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop3_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_13_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_Output.txt",
        "Simple_Test",
        "reader",
        randomizeAuto, // epoch size - all available
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop4)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop4_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop5)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop5_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop5_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop7)
{
    HelperRunReaderTestWithException<float, std::invalid_argument>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop7_Config.cntk",
        "Simple_Test",
        "reader");
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop8)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop8_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop8_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop10)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop10_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_20_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        2,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop11)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop11_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop11_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop12)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop12_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_12_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop12_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop13)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop13_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_13_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop13_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop14)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop14_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop14_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop16)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop16_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop6_16_17_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop16_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop19)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop19_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop19_Output.txt",
        "Simple_Test",
        "reader",
        2000,
        250,
        1,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop20)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop20_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_20_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop20_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        2,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop21_0)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop21_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_0_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_0_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        2);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop21_1)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop21_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_1_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_1_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        1,
        2,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop22)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop22_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop22_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop22_Output.txt",
        "Simple_Test",
        "reader",
        5000,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop1)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop1_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop5)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop5_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop5_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop11)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop11_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop1_5_11_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop11_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop21_0)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop21_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_0_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_0_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        2);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop21_1)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop21_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_1_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop21_1_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        1,
        2,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop4)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop4_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKDeserializersSimpleDataLoop4_Output.txt",
        "Simple_Test",
        "reader",
        500,
        140,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop8)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop8_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop8_Output.txt",
        "Simple_Test",
        "reader",
        500,
        140,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop14)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop14_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop4_8_14_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop14_Output.txt",
        "Simple_Test",
        "reader",
        500,
        140,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop9)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop9_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
        testDataPath() + "/Control/HTKDeserializersSimpleDataLoop9_Output.txt",
        "Simple_Test",
        "reader",
        2000,
        250,
        1,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop9)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop9_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_Output.txt",
        "Simple_Test",
        "reader",
        2000,
        250,
        1,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop19)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop19_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
        testDataPath() + "/Control/HTKDeserializersSimpleDataLoop19_Output.txt",
        "Simple_Test",
        "reader",
        2000,
        250,
        1,
        1,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop10)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop10_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_20_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        2,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop20)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop20_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop10_20_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop20_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        2,
        1,
        0,
        1,
        false,
        false,
        true,
        {},
        true);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop3)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop3_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_13_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_Output.txt",
        "Simple_Test",
        "reader",
        randomizeAuto, // epoch size - all available
        1,
        2,
        1,
        0,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKDeserializers_NonExistingScpFile)
{
    HelperRunReaderTestWithException<float, std::runtime_error>(
        testDataPath() + "/Config/HTKDeserializers_NonExistingScpFile.cntk",
        "Simple_Test",
        "reader");
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop2)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop2_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_12_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(HTKDeserializersSimpleDataLoop12)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/HTKDeserializersSimpleDataLoop12_Config.cntk",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop2_12_Control.txt",
        testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop12_Output.txt",
        "Simple_Test",
        "reader",
        500,
        250,
        2,
        1,
        1,
        0,
        1);
};

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(ReaderIVectorTestSuite, iVectorFixture)

BOOST_AUTO_TEST_CASE(HTKNoIVector)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<double>(
            testDataPath() + "/Config/HTKMLFReaderNoIVectorSimple_Config.cntk",
            testDataPath() + "/Control/HTKMLFReaderNoIVectorSimple_Control.txt",
            testDataPath() + "/Control/HTKMLFReaderNoIVectorSimple_Output.txt",
            "Simple_Test",
            "reader",
            400,
            30,
            1,
            1,
            1,
            0,
            1,
            false,
            false,
            true,
            additionalParameters);
    };

    test({});
    test({ L"Simple_Test=[reader=[readerType=HTKDeserializers]]" });
};

BOOST_AUTO_TEST_CASE(HTKIVectorFrame)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<double>(
            testDataPath() + "/Config/HTKMLFReaderIVectorSimple_Config.cntk",
            testDataPath() + "/Control/HTKMLFReaderIVectorSimple_Control.txt",
            testDataPath() + "/Control/HTKMLFReaderIVectorSimple_Output.txt",
            "Simple_Test",
            "reader",
            400,
            30,
            1,
            2,
            1,
            0,
            1,
            false,
            false,
            true,
            additionalParameters);
    };

    test({});
    test({ L"Simple_Test=[reader=[readerType=HTKDeserializers]]" });
};

BOOST_AUTO_TEST_CASE(HTKIVectorSequence)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/HTKMLFReaderIVectorSimple_Config.cntk",
            testDataPath() + "/Control/HTKMLFReaderIVectorSequenceSimple_Control.txt",
            testDataPath() + "/Control/HTKMLFReaderIVectorSequenceSimple_Output.txt",
            "Simple_Test",
            "reader",
            200,
            30,
            1,
            2,
            1,
            0,
            1,
            false,
            false,
            true,
            additionalParameters);
    };

    test({ L"frameMode=false", L"precision=float" });
    test({ L"frameMode=false", L"precision=float", L"Simple_Test=[reader=[readerType=HTKDeserializers]]" });
};

BOOST_AUTO_TEST_CASE(HTKIVectorBptt)
{
    auto test = [this](std::vector<std::wstring> additionalParameters)
    {
        HelperRunReaderTest<double>(
            testDataPath() + "/Config/HTKMLFReaderIVectorSimple_Config.cntk",
            testDataPath() + "/Control/HTKMLFReaderIVectorBpttSimple_Control.txt",
            testDataPath() + "/Control/HTKMLFReaderIVectorBpttSimple_Output.txt",
            "Simple_Test",
            "reader",
            400,
            30,
            1,
            2,
            1,
            0,
            1,
            false,
            false,
            true,
            additionalParameters);
    };
    test({ L"frameMode=false", L"truncated=true" });
    test({ L"frameMode=false", L"truncated=true", L"Simple_Test=[reader=[readerType=HTKDeserializers]]" });
};

BOOST_AUTO_TEST_SUITE_END()

}

}}}
