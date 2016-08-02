//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "BrainScriptParser.h"
#include "boost/filesystem.hpp"

#include <utility>

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

struct ParserTestsFixture
{
public:
    ParserTestsFixture(){
        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
        wstring parentPath = boost::filesystem::canonical(path.parent_path()).generic_wstring();

#ifdef _WIN32
        // The executable path on Windows is e.g. <cntk>/x64/Debug/Unittests/
        m_testDataPath = parentPath + L"/../../../Tests/UnitTests/BrainScriptTests";
#else
        // The executable path on Linux is e.g. <cntk>/build/cpu/release/bin/
        m_testDataPath = parentPath + L"/../../../../Tests/UnitTests/BrainScriptTests";
#endif

        boost::filesystem::path absTestPath(m_testDataPath);
        absTestPath = boost::filesystem::canonical(absTestPath);
        m_testDataPath = absTestPath.generic_wstring();
    }

    const wstring getDataPath(){
        return m_testDataPath;
    }
private:
    wstring m_testDataPath;
};


namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_FIXTURE_TEST_SUITE(ParserTests, ParserTestsFixture)

void parseLine(const wstring & input, const wstring & expectedOutput)
{
    let expr = BS::ParseConfigDictFromString(input, L"Test", vector<wstring>());

    wstringstream actualStream;
    expr->DumpToStream(actualStream);

    BOOST_TEST(actualStream.str() == expectedOutput, boost::test_tools::per_element());
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

        if (!inputFile.is_open()){
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

} } } }