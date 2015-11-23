//
// <copyright file="HTKMLFReaderTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft {  namespace MSR  {  namespace CNTK
{
    namespace Test
    {
        // Fixture specific to the AN4 data
        struct AN4ReaderFixture : ReaderFixture
        {
            AN4ReaderFixture() : ReaderFixture("%CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY%/Speech/AN4Corpus/v0")
            {}
        };

        // Fixture specific for the TIMIT data
        struct TIMITReaderFixture : ReaderFixture
        {
            //TIMITReaderFixture() : ReaderFixture("%CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY%/Speech/ASR")
            //TIMITReaderFixture() : ReaderFixture("\\\\speech-data\\CNTKExampleSetups\\ASR")
            TIMITReaderFixture() : ReaderFixture("/Data")
            {}
        };

        // Use SpeechReaderFixture for most tests
        // Some of them (e.g. 10, will use different data, thus a different fixture)
        BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, AN4ReaderFixture)

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop1)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop1_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop2)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop2_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop3_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_13_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop3_Output.txt",
                "Simple_Test",
                "reader",
                5,
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop4_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop5_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop6)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop6_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop6_16_17_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop6_Output.txt",
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

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop7)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop7_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop7_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop7_Output.txt",
                "Simple_Test",
                "reader",
                500,
                200,
                2,
                1,
                1,
                0,
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop8)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop8_Config.txt",
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

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop9)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop9_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_Output.txt",
                "Simple_Test",
                "reader",
                5000,
                250,
                2,
                1,
                1,
                0,
                1);
        };

        BOOST_FIXTURE_TEST_CASE(HTKMLFReaderSimpleDataLoop10, TIMITReaderFixture)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop10_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop11)
        {
            HelperRunReaderTest<double>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop11_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop12)
        {
            HelperRunReaderTest<double>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop12_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop13_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop14_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop16_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop17)
        {
            HelperRunReaderTest<double>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop17_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop6_16_17_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop17_Output.txt",
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

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop19)
        {

            HelperRunReaderTest<double>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop19_Config.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop9_19_Control.txt",
                testDataPath() + "/Control/HTKMLFReaderSimpleDataLoop19_Output.txt",
                "Simple_Test",
                "reader",
                5000,
                250,
                2,
                1,
                1,
                0,
                1);
        };

        BOOST_FIXTURE_TEST_CASE(HTKMLFReaderSimpleDataLoop20, TIMITReaderFixture)
        {
            HelperRunReaderTest<double>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop20_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop21_0)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop21_Config.txt",
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
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop21_Config.txt",
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
                2);
        };

        BOOST_AUTO_TEST_CASE(HTKMLFReaderSimpleDataLoop22)
        {
            HelperRunReaderTest<float>(
                testDataPath() + "/Config/HTKMLFReaderSimpleDataLoop22_Config.txt",
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
                1);
        };

        BOOST_AUTO_TEST_SUITE_END()
    }
}}}
