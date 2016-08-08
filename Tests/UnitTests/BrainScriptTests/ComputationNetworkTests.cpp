//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "BrainScriptParser.h"
#include "BrainScriptTestsHelper.h"
#include "ComputationNetwork.h"
#include "CommonMatrix.h"
#include "boost/filesystem.hpp"

#include <utility>
#include <vector>
#include <istream>

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_FIXTURE_TEST_SUITE(ComputationNetworkSuite, BSFixture)

void parseLine(const wstring & input, const wstring & expectedOutput)
{
    let expr = BS::ParseConfigDictFromString(input, L"Test", vector<wstring>());

    wstringstream actualStream;
    expr->DumpToStream(actualStream);

    BOOST_TEST(actualStream.str() == expectedOutput, boost::test_tools::per_element());
}

std::vector<wstring> inputModelNames{
    L"LR_reg.dnn"
};

BOOST_AUTO_TEST_CASE(CompareNetworkStructureFromModel)
{
    wstring inputPrefix(L"Input");
    wstring outputPrefix(L"ExpectedOutput");

    wstring computationData = getDataPath() + L"/Data/ComputationNetwork/";

    for (auto & modelName : inputModelNames)
    {
        wstring modelPath = computationData + modelName;
        ComputationNetworkPtr net = ComputationNetwork::CreateFromFile<float>(CPUDEVICE, modelPath);
        net->DumpNodeInfoToFile(L"", true, true, modelPath + L"_Actual.txt", L"");
        
        wstring actualNetworkPath(modelPath + L"_Actual.txt");
        wstring expectedNetworkPath(modelPath + L"_Expected.txt");
        
        wifstream actualNetworkStream;
        wifstream expectedNetworkStream;

#ifdef _WIN32
        actualNetworkStream.open(actualNetworkPath.c_str(), wifstream::in);
        expectedNetworkStream.open(expectedNetworkPath.c_str(), wifstream::in);
#else
        actualNetworkStream.open(wtocharpath(actualNetworkPath.c_str()).c_str(), wifstream::in);
        expectedNetworkStream.open(wtocharpath(expectedNetworkPath.c_str()).c_str(), wifstream::in);
#endif
        wstring actualNetwork;
        wstring expectedNetwork;

        actualNetworkStream >> actualNetwork;
        expectedNetworkStream >> expectedNetwork;

        BOOST_TEST(actualNetwork == expectedNetwork, boost::test_tools::per_element());

    }
}

BOOST_AUTO_TEST_SUITE_END()

} } } }