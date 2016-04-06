//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/ReaderTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct UCIReaderFixture : ReaderFixture
{
    UCIReaderFixture()
        : ReaderFixture("/Data")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, UCIReaderFixture)

BOOST_AUTO_TEST_CASE(UCIFastReaderSimpleDataLoop)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/UCIFastReaderSimpleDataLoop_Config.cntk",
        testDataPath() + "/Control/UCIFastReaderSimpleDataLoop_Control.txt",
        testDataPath() + "/Control/UCIFastReaderSimpleDataLoop_Output.txt",
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

} } } }
