//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "BrainScriptParser.h"
#include "ParserTestsData.h"

#include <utility>

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {


BOOST_AUTO_TEST_SUITE(ParserTests)

void parseLine(const std::pair<wstring, wstring> & testLine)
{        
    let expr = BS::ParseConfigDictFromString(testLine.first, L"Test", vector<wstring>());

    wstringstream actualStream;
    expr->DumpToStream(actualStream);

    BOOST_TEST(actualStream.str() == testLine.second, boost::test_tools::per_element());

}

BOOST_AUTO_TEST_CASE(ParseExpressionsAndCompareTree)
{
    int testCount = 0;
    for (auto & testPair : parserTestVector)
    {
        fprintf(stderr, "Test %d:\n", testCount);
        parseLine(testPair);
        testCount++;
    }
}

BOOST_AUTO_TEST_SUITE_END()

} } } }