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
using namespace ScriptableObjects;

// ===================================================================
// behave like a config
// This allows to access nodes inside a network as if it was an IConfigRecord.
// This is meant to be used by whatever we will replace MEL.
// ===================================================================

// not in the cache yet: create it (or not if no such member)
void /*CustomConfigRecord::*/ ComputationNodeBase::LazyCreateConfigMember(const wstring& id) const /*override*/
{
    if (id == L"name")
    {
        InsertConfigMember(id, ConfigValuePtr(make_shared<String>(NodeName()), [](const std::wstring &) { LogicError("should not get here"); }, L""));
    }
    else if (id == L"operation")
    {
        InsertConfigMember(id, ConfigValuePtr(make_shared<String>(OperationName()), [](const std::wstring &) { LogicError("should not get here"); }, L""));
    }
    // TODO: Think through what tags mean. Do we allow user-named tags? Is it a set or a single string? If set, then how to compare?
    //else if (id == L"tag")
    //{
    //}
    else if (id == L"inputs")
    {
        std::vector<ConfigValuePtr> inputsAsValues;
        for (let& input : GetInputs())
            inputsAsValues.push_back(ConfigValuePtr(input, [](const std::wstring &) { LogicError("should not get here"); }, L""));
        let& arr = make_shared<ScriptableObjects::ConfigArray>(0, move(inputsAsValues));
        InsertConfigMember(id, ConfigValuePtr(arr, [](const std::wstring &) { LogicError("should not get here"); }, L""));
    }
    // any other id does not exist, don't create any entry for it
}

vector<wstring> /*IConfigRecord::*/ ComputationNodeBase::GetMemberIds() const
{
    return vector<wstring>{ L"name", L"operation", /*L"tag", */L"inputs" };
}

}}}
