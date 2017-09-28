//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/NetworkTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

// Fixture specific to the operators
struct OperatorFixture : DataFixture
{
    OperatorFixture()
        : DataFixture("/Data")
    {
    }
};

// Use SpeechReaderFixture for most tests
BOOST_FIXTURE_TEST_SUITE(NetworkTestSuite, OperatorFixture)

BOOST_AUTO_TEST_CASE(NetworkOperatorPlus)
{
    HelperRunNetworkTest<float>(
        L"../Config/Network_Operator_Plus.cntk" /*config*/,
        "../Control/Network_Operator_Plus_Control.txt" /*control*/,
        "../Output/out.txt.v2" /*output*/);
};


BOOST_AUTO_TEST_SUITE_END()

}}}}