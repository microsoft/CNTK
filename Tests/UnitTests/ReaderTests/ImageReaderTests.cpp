//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"

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
};

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
};

BOOST_AUTO_TEST_SUITE_END()
} } } }
