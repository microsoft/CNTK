//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "BrainScriptParser.h"
#include "BrainScriptTestsHelper.h"
#include "boost/filesystem.hpp"
#include <boost/algorithm/string.hpp>

#include <utility>

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_FIXTURE_TEST_SUITE(ParserSuite, BSFixture)

// normalize strings:
//  - remove CR characters (so that we can create the reference on Windows)
//  - trailing spaces (which are impossible to copy-paste from screen output)
static void Normalize(wstring& s)
{
    boost::replace_all(s, L"\r", L"");
    //boost::trim_right_if(s, boost::is_any_of(L" \n"));
    // ^^ fails with 'std::_Copy_impl': Function call with parameters that may be unsafe
    // this ugly version compiles:
    while (!s.empty() && (s.back() == ' ' || s.back() == '\n'))
        s.pop_back();
}

void parseLine(wstring input, wstring expectedOutput)
{
    let expr = BS::ParseConfigDictFromString(input, L"Test", vector<wstring>());

    wstringstream actualStream;
    expr->DumpToStream(actualStream);

    wstring actualOutput = actualStream.str();

    // we normalize for newlines and trailing spaces
    Normalize(expectedOutput);
    Normalize(actualOutput);
    //printf("%ls\n", wstring (actualStream.str()).c_str());

    BOOST_TEST(actualOutput == expectedOutput, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ParseExpressionsAndCompareTree)
{
    wstring inputPrefix(L"Input");
    wstring outputPrefix(L"ExpectedOutput");

    wstring dataPath = getDataPath() + L"/Data/Parser/";

    bool filesAvailable = true;
    int testCount = 1;
    while (filesAvailable)
    {
        wstring inputPath(dataPath + inputPrefix + to_wstring(testCount) + L".txt");
        wstring outputPath(dataPath + outputPrefix + to_wstring(testCount) + L".txt");

        wifstream inputFile;
        wifstream outputFile;

#ifdef _WIN32
        inputFile.open(inputPath.c_str(), wifstream::in);
        outputFile.open(outputPath.c_str(), wifstream::in);
#else
        inputFile.open(wtocharpath(inputPath.c_str()).c_str(), wifstream::in);
        outputFile.open(wtocharpath(outputPath.c_str()).c_str(), wifstream::in);
#endif

        if (!inputFile.is_open())
        {
            filesAvailable = false;
            continue;
        }

        fprintf(stderr, "Test %d...\n", testCount);

        std::wostringstream inputStream;
        inputStream << inputFile.rdbuf();

        std::wostringstream expectedStream;
        expectedStream << outputFile.rdbuf();

        parseLine(inputStream.str().c_str(), expectedStream.str().c_str());
        testCount++;
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}}}
