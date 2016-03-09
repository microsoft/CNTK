//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NetworkCreator.h -- CNTK network creation related functions
//
#pragma once

#include "ComputationNetwork.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"

#ifndef let
#define let const auto
#endif

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

// TODO: decide where these should go. Also, do we need three variables?
extern wstring standardFunctions;
extern wstring commonMacros;
extern wstring computationNodes;

// helper that returns 'float' or 'double' depending on ElemType
template <typename ElemType> /*static*/ const wchar_t* ElemTypeName();

function<ComputationNetworkPtr(DEVICEID_TYPE)> GetCreateNetworkFn(const ScriptableObjects::IConfigRecord& config);

template <class ConfigRecordType, typename ElemType>
function<ComputationNetworkPtr(DEVICEID_TYPE)> GetNetworkFactory(const ConfigRecordType& config);
