//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "BrainScriptParser.h"

#include <utility>

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {


BOOST_AUTO_TEST_SUITE(ParserTests)

const std::vector<std::pair<wstring, wstring>> testVector{
    std::make_pair(L"do = Parameter(13,42) * Input(42) + Parameter(13,1)", 
    L" []\n do =\n  +\n   *\n    (\n     Parameter\n     ()\n      13\n      42\n\n\n    (\n     Input\n"
    L"     ()\n      42\n\n\n\n   (\n    Parameter\n    ()\n     13\n     1\n\n\n\n\n")
};

void parseLine(const std::pair<wstring, wstring> & testLine)
{        
    let expr = BS::ParseConfigDictFromString(testLine.first, L"Test", vector<wstring>());

    wstringstream actualStream;
    expr->DumpToStream(actualStream);
    //fprintf(stderr, "Actual: %ls", myStream.str().c_str());
    //fprintf(stderr, "Expected: %ls\n", testLine.second.c_str());
    BOOST_TEST(actualStream.str() == testLine.second);
}

BOOST_AUTO_TEST_CASE(ParseExpressionsAndCompareTree)
{
    std::for_each(testVector.begin(), testVector.end(), &parseLine);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }