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

void compareNetworks(const wstring & modelPath)
{
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

    actualNetworkStream.close();
    remove(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(actualNetworkPath)).c_str());
}

BOOST_AUTO_TEST_CASE(CompareNetworkStructureFromModel)
{
    wstring computationData = getDataPath() + L"/Data/ComputationNetwork/";

    std::vector<wstring> inputModelPaths = getListOfFilesByExtension(L".dnn", computationData);

    for (auto & modelPath : inputModelPaths)
    {
        fprintf(stderr, "Model path: %ls\n", modelPath.c_str());
        ComputationNetworkPtr net = ComputationNetwork::CreateFromFile<float>(CPUDEVICE, modelPath);
        net->DumpNodeInfoToFile(L"", true, true, modelPath + L"_Actual.txt", L"");

        compareNetworks(modelPath);
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}}}
