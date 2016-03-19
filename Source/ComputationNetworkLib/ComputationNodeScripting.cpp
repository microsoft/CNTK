//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "ComputationNetworkBuilder.h" // TODO: We should only pull in NewComputationNodeFromConfig(). Nodes should not know about network at large.
#include "TensorShape.h"

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// ===================================================================
// behave like a config
// This allows to access nodes inside a network as if it was an IConfigRecord.
// This is meant to be used by whatever we will replace MEL.
// TODO: implement this
// ===================================================================

#if 0
const ScriptableObjects::ConfigValuePtr& /*IConfigRecord::*/ ComputationNodeBase::operator[](const wstring& id) const // e.g. confRec[L"message"]
{
    id;
    RuntimeError("unknown class parameter"); // (for now)
}
const ScriptableObjects::ConfigValuePtr* /*IConfigRecord::*/ ComputationNodeBase::Find(const wstring& id) const // returns nullptr if not found
{
    id;
    return nullptr; // (for now)
}
vector<wstring> /*IConfigRecord::*/ ComputationNodeBase::GetMemberIds() const
{
    return vector<wstring>{ L"name", L"operation", L"inputs" };
}
#endif

}}}
