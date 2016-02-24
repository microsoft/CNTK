//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EsotericActions.cpp -- CNTK actions that are deprecated
//

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "stdafx.h"
#include "Basics.h"
#include "Actions.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "SimpleNetworkBuilder.h"
#include "Config.h"
#include "ScriptableObjects.h"

#include <string>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include <memory>

#ifndef let
#define let const auto
#endif

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

// ===========================================================================
// DoConvertFromDbn() - implements CNTK "convertdbn" command
// ===========================================================================

template <typename ElemType>
void DoConvertFromDbn(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring dbnModelPath = config(L"dbnModelPath");

    auto netBuilder = make_shared<SimpleNetworkBuilder<ElemType>>(config);
    ComputationNetworkPtr net = netBuilder->BuildNetworkFromDbnFile(dbnModelPath);
    net->Save(modelPath);
}

// ===========================================================================
// DoExportToDbn() - implements CNTK "exportdbn" command
// ===========================================================================

template <typename ElemType>
void DoExportToDbn(const ConfigParameters& config)
{
    DEVICEID_TYPE deviceID = DeviceFromConfig(config);

    const wstring modelPath = config("modelPath");
    wstring dbnModelPath = config("dbnModelPath");

    ComputationNetworkPtr net = make_shared<ComputationNetwork>(deviceID);
    net->Load<ElemType>(modelPath);

    // write dbn file
    net->SaveToDbnFile<ElemType>(net, dbnModelPath);
}

template void DoConvertFromDbn<float>(const ConfigParameters& config);
template void DoConvertFromDbn<double>(const ConfigParameters& config);
template void DoExportToDbn<float>(const ConfigParameters& config);
template void DoExportToDbn<double>(const ConfigParameters& config);
