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
#include "FixedPoint.hpp"
#include "LSTMClient.h"

#include "LstmGraphNode.h"

using namespace CNTK;
using namespace BrainSlice;
using namespace LSTM;
using namespace std;


static LSTMClient * lstmClient = nullptr;

typedef FixedPoint<LSTM::dword_t, 12> FixedWord;

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
	// TODO: these are hardcoded in the lstm client.
	const size_t inVectorLen = 200,
		outVectorLen = 500;

    fprintf(stderr, "LstmGraphNode::Forward(out %lu, left %lu, right %lu) called\n", (unsigned long)out.size(), (unsigned long)left.size(), (unsigned long)right.size());

	assert(lstmClient != nullptr);

	vector<vector<dword_t>> inputs;

	// inputs
	for (auto leftright : { left, right })
	{
		vector<dword_t> input(inVectorLen);
		for (size_t n = 0; n < min(inVectorLen, leftright.size()); n++)
		{
			FixedWord number(leftright[n]);
			input[n] = number.m_value;
		}

		inputs.push_back(input);
	}

	vector<dword_t> output(outVectorLen);

	lstmClient->Evaluate(0, inputs, output);

    for (auto n = 0; n < out.size(); n++)
    {
		if (n < output.size())
		{
			FixedWord number(output[n]);
			out[n] = number.toFloat();
		}
		else
		{
			out[n] = (float)n;
		}
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
