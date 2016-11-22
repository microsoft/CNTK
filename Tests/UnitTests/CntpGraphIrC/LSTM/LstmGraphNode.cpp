//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalMultithreads.cpp : Sample application shows how to evaluate a model in multiple threading environment. 
//
#include <functional>
#include <thread>
#include <iostream>
#include <vector>
#include <random>


#include <CNTKLibrary.h>

#include <BrainSliceClient.h>
#include "LSTMClient.h"
#include "LstmGraphNode.h"

using namespace CNTK;
using namespace BrainSlice;
using namespace LSTM;
using namespace std;


static LSTMClient * lstmClient = nullptr;


LstmGraphNode::LstmGraphNode(
        std::vector<Variable>& inputs,
        Dictionary&& functionConfig,
        const std::wstring& name,
        const std::wstring& uid)
        : _inputs(inputs), _functionConfig(functionConfig), _name(name), _uid(uid)
{
}

LstmGraphNode::~LstmGraphNode()
{
}

/*BackPropStatePtr*/void LstmGraphNode::ForwardFloat(
        std::vector<float>& out,
        const std::vector<float>& left,
        const std::vector<float>& right)
{
    fprintf(stderr, "LstmGraphNode::Forward(out %u, left %u, right %u) called\n", out.size(), left.size(), right.size());

    for (auto n = 0; n < out.size(); n++)
    {
        out[n] = n;
    }
}

void LstmGraphNode::Backward(
        ////const BackPropStatePtr& /*state*/,
        ////const std::unordered_map<Variable, ValuePtr>& /*rootGradientValues*/,
        ////std::unordered_map<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/
    )
{
    NOT_IMPLEMENTED;
}

FunctionPtr LstmGraphNodeFactory(
    const std::wstring& op,
    std::vector<Variable>& inputs,
    Dictionary&& functionConfig,
    const std::wstring& functionName,
    const std::wstring& uid)
{
    fprintf(stderr, "%-32S%S\n", uid.c_str(), functionName.c_str());

	if (lstmClient == nullptr)
	{
		//auto bsClient = BrainSliceClient::Create(false);
		lstmClient = LSTMClient::Create(false);
	}

	if (op == L"Times")
    {
        fprintf(stderr, "    OVERRIDING as fpga node.\n");

        auto functionConfigCopy = functionConfig;
        auto interceptTarget = std::make_shared<LstmGraphNode>(inputs, std::move(functionConfigCopy), functionName, uid);

        return UserDefinedFuntion(
                inputs,
                std::move(functionConfig),
                functionName,
                uid,
                interceptTarget);
    }

    return nullptr;
}
