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


//typedef FixedPoint<LSTM::dword_t, 12> FixedWord;

static void InitRandom(vector<LSTM::dword_t> &p_vector, int count)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 2.0);

	p_vector.resize(count);
	for (int i = 0; i < count; ++i)
	{
		double number = distribution(generator);
		//FixedWord fixed = number; // TODO ENABLE ME
		p_vector[i] = (short)(number * 1000); // fixed.m_value;
	}

	return;
}

static bool IsEqual(std::vector<LSTM::dword_t> &p_vectorA, std::vector<LSTM::dword_t> &p_vectorB)
{
	assert(p_vectorA.size() == p_vectorB.size());
	for (int i = 0; i < p_vectorA.size(); ++i)
	{
		if (p_vectorA[i] != p_vectorB[i])
		{
			std::cerr << "Error, mismatch in vector @ element " << i << " got:" << p_vectorA[i] << " expected:" << p_vectorB[i] << std::endl;
			return false;
		}
	}
	return true;
}

double ElapsedSeconds(LARGE_INTEGER StartingTime, LARGE_INTEGER EndingTime)
{
	LARGE_INTEGER Frequency;
	QueryPerformanceFrequency(&Frequency);  // cycles per second
	return double(EndingTime.QuadPart - StartingTime.QuadPart) / double(Frequency.QuadPart);
}

static void RunLoopbackTest(LSTMClient *pLstmClient, const uint32_t pDstIp, const int pSamples)
{
	LSTMClient::LSTMInfo lstmInfo = pLstmClient->Info(pDstIp);

	vector<LSTM::dword_t> inputVector(lstmInfo.nativeDim);
	vector<LSTM::dword_t> outputVector(lstmInfo.nativeDim);

	InitRandom(inputVector, lstmInfo.nativeDim);
	InitRandom(outputVector, lstmInfo.nativeDim);

	LARGE_INTEGER t1, t2;
	QueryPerformanceCounter(&t1);
	for (int i = 0; i < pSamples; ++i)
	{
		pLstmClient->TestLoopback(pDstIp, inputVector, outputVector);
	}
	QueryPerformanceCounter(&t2);

	if (!IsEqual(inputVector, outputVector))
	{
		throw new std::runtime_error("Mismatch in loop back test");
	}

	double elapsedUs = (double)ElapsedSeconds(t1, t2) * 1.0e6f;
	double bwMB = (double)(pSamples * lstmInfo.nativeDim * sizeof(LSTM::dword_t)) / (double)ElapsedSeconds(t1, t2) * 1.0e-6f;

	printf_s("\nVector Loopback Test\n");
	printf_s("    Round-trip microseconds: %.2f\n", elapsedUs / (double)pSamples);
	printf_s("    I/O Bandwidth (MB/s):    %.2f\n", bwMB);
}


FunctionPtr LstmGraphNodeFactory(
    const std::wstring& op,
    std::vector<Variable>& inputs,
    Dictionary&& functionConfig,
    const std::wstring& functionName,
    const std::wstring& uid)
{
    fprintf(stderr, "Inspecting %-32S%S\n", uid.c_str(), functionName.c_str());

	if (lstmClient == nullptr)
	{
		//auto bsClient = BrainSliceClient::Create(false);
		lstmClient = LSTMClient::Create(false);

		RunLoopbackTest(lstmClient, 0x0 /* localhost */, 1000 /* samples */);
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
