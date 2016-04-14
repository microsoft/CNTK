//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct ImageReaderFixture : ReaderFixture
{
    ImageReaderFixture()
        : ReaderFixture("/Data")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, ImageReaderFixture)

BOOST_AUTO_TEST_CASE(ImageReaderSimple)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderSimple_Config.cntk",
        testDataPath() + "/Control/ImageReaderSimple_Control.txt",
        testDataPath() + "/Control/ImageReaderSimple_Output.txt",
        "Simple_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderBadMap)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderBadMap_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Invalid map file format, must contain 2 tab-delimited columns, line 2 in file ./ImageReaderBadMap_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderBadLabel)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderBadLabel_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Cannot parse label value on line 1, second column, in file ./ImageReaderBadLabel_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderLabelOutOfRange)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderLabelOutOfRange_Config.cntk",
            testDataPath() + "/Control/ImageReaderSimple_Control.txt",
            testDataPath() + "/Control/ImageReaderSimple_Output.txt",
            "Simple_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Image 'images\\red.jpg' has invalid class id '10'. Expected label dimension is '4'. Line 3 in file ./ImageReaderLabelOutOfRange_map.txt.") == ex.what(); });
}

BOOST_AUTO_TEST_CASE(ImageReaderZip)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/ImageReaderZip_Config.cntk",
        testDataPath() + "/Control/ImageReaderZip_Control.txt",
        testDataPath() + "/Control/ImageReaderZip_Output.txt",
        "Zip_Test",
        "reader",
        4,
        4,
        1,
        1,
        0,
        0,
        1);
}

BOOST_AUTO_TEST_CASE(ImageReaderZipMissingFile)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<float>(
            testDataPath() + "/Config/ImageReaderZipMissing_Config.cntk",
            testDataPath() + "/Control/ImageReaderZip_Control.txt",
            testDataPath() + "/Control/ImageReaderZip_Output.txt",
            "ZipMissing_Test",
            "reader",
            4,
            4,
            1,
            1,
            0,
            0,
            1),
            std::runtime_error,
            [](std::runtime_error const& ex) { return string("Failed to get file info of missing.jpg, zip library error: Unknown error -1") == ex.what(); });
}

BOOST_AUTO_TEST_SUITE_END()
} } } }
