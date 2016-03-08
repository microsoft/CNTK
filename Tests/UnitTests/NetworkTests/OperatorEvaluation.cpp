//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/NetworkTestHelper.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft {
namespace MSR {
namespace CNTK {
namespace Test {

// Fixture specific to the operators
struct OperatorFixture : DataFixture
{
    OperatorFixture()
        : DataFixture("/Data")
    {
    }
};

// Use SpeechReaderFixture for most tests
// Some of them (e.g. 10, will use different data, thus a different fixture)
BOOST_FIXTURE_TEST_SUITE(NetworkTestSuite, OperatorFixture)

BOOST_AUTO_TEST_CASE(NetworkOperatorPlus)
{
    wstring configFileName(L"../config/Network_Operator_Plus.cntk");
    string baseFileName("../Control/Network_Operator_Plus_Control.txt");
    string outputFileName("../output/out.txt.v2");

    HelperRunNetworkTest<float>(configFileName, baseFileName, outputFileName);
};

}
}
}
}
}