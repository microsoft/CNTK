//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common/NetworkTestHelper.h"
#include "Actions.h"
#include "NDLNetworkBuilder.h"


using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct Fixture : DataFixture
{
    Fixture() : DataFixture("/Data")
    { }
};


BOOST_FIXTURE_TEST_SUITE(BatchNormTestSuite, Fixture)


BOOST_AUTO_TEST_CASE(TestLoadingNetworkFromLegacyNDLConfig)
{
    NDLScript<float> ndlScript;
    ndlScript.ClearGlobal(); // clear global macros between tests

    ConfigParameters config;
    config.LoadConfigFile(L"../Config/BatchNorm_NDL_Builder_BN5.cntk");
    vector<wstring> ignored;
    ComputationNetworkPtr net;
    BOOST_CHECK_NO_THROW((net = GetModelFromConfig<ConfigParameters, float>(config, L"", ignored)));
    BOOST_CHECK(net != nullptr);
};

BOOST_AUTO_TEST_CASE(TestLoadingNetworkFromUpdatedNDLConfig)
{
    NDLScript<float> ndlScript;
    ndlScript.ClearGlobal(); // clear global macros between tests

    ConfigParameters config;
    config.LoadConfigFile(L"../Config/BatchNorm_NDL_Builder_BN6.cntk");
    vector<wstring> ignored;
    ComputationNetworkPtr net;
    BOOST_CHECK_NO_THROW((net = GetModelFromConfig<ConfigParameters, float>(config, L"", ignored)));
    BOOST_CHECK(net != nullptr);
};


BOOST_AUTO_TEST_CASE(TestLoadingNetworkFromLegacyLegacyNDLModel)
{
    ConfigParameters config;
    config.LoadConfigFile(L"../Config/BatchNorm_NDL_Model.cntk");
    vector<wstring> ignored;
    ComputationNetworkPtr net;
    BOOST_CHECK_NO_THROW((net = GetModelFromConfig<ConfigParameters, float>(config, L"", ignored)));
    BOOST_CHECK(net != nullptr);
};

BOOST_AUTO_TEST_CASE(TestLoadingNetworkFromLegacyBSConfig)
{
    ConfigParameters config;
    config.LoadConfigFile(L"../Config/BatchNorm_BS_Builder.cntk");
    vector<wstring> ignored;
    ComputationNetworkPtr net;
    BOOST_CHECK_NO_THROW((net = GetModelFromConfig<ConfigParameters, float>(config, L"", ignored)));
    BOOST_CHECK(net != nullptr);
};

BOOST_AUTO_TEST_CASE(TestLoadingNetworkFromLegacyLegacyBSModel)
{
    ConfigParameters config;
    config.LoadConfigFile(L"../Config/BatchNorm_BS_Model.cntk");
    vector<wstring> ignored;
    ComputationNetworkPtr net;
    BOOST_CHECK_NO_THROW((net = GetModelFromConfig<ConfigParameters, float>(config, L"", ignored)));
    BOOST_CHECK(net != nullptr);
};

BOOST_AUTO_TEST_SUITE_END()

}}}}