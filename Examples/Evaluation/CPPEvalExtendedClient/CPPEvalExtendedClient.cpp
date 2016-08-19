//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//
#include "Eval.h"
#ifdef _WIN32
#include "Windows.h"
#endif

#include <inttypes.h>
#include <algorithm>


using namespace std;
using namespace Microsoft::MSR::CNTK;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModelExtended<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> Variable;
typedef std::map<std::wstring, std::vector<float>*> Variables;

IEvaluateModelExtended<float>* SetupNetworkAndGetLayouts(std::string modelDefinition, VariableSchema& inputLayouts, VariableSchema& outputLayouts)
{
	// Native model evaluation instance
	IEvaluateModelExtended<float> *eval;

	GetEvalExtendedF(&eval);

	try
	{
		eval->CreateNetwork(modelDefinition);
	}
	catch (std::exception& ex)
	{
		fprintf(stderr, "%s\n", ex.what());
		throw;
	}
	fflush(stderr);

	// Get the model's layers dimensions
	outputLayouts = eval->GetOutputSchema();

	for (auto vl : outputLayouts)
	{
		fprintf(stderr, "Output dimension: %" PRIu64 "\n", vl.m_numElements);
		fprintf(stderr, "Output name: %ls\n", vl.m_name.c_str());
	}

	eval->StartForwardEvaluation({ outputLayouts[0].m_name });
	inputLayouts = eval->GetInputSchema();
	outputLayouts = eval->GetOutputSchema();

	return eval;
}


/// <summary>
/// Program for demonstrating how to run model evaluations using the native extended evaluation interface, also show
/// how to input sequence vectors to LSTM(RNN) network.
/// </summary>
/// <description>
/// This program is a native C++ client using the native extended evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation library (EvalDLL.dll on Windows, and LibEval.so on Linux), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/Text/ATIS. Once the model file ATIS.slot.lstm is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.
int main(int argc, char* argv[])
{
	// Get the binary path (current working directory)
	argc = 0;
	std::string app = argv[0];
	std::string path;
	size_t pos;

#ifdef _WIN32
	pos = app.rfind("\\");
	path = (pos == std::string::npos) ? "." : app.substr(0, pos);

	// This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
	const std::string modelWorkingDirectory = path + "/../../Examples/Text/ATIS/work/";
#else // on Linux
	pos = app.rfind("/");
	path = (pos == std::string::npos) ? "." : app.substr(0, pos);

	// This relative path assumes launching from CNTK's binary folder, e.g. build/release/bin/
	const std::string modelWorkingDirectory = path + "/../../../Examples/Text/ATIS/work/";
#endif

	const std::string modelFilePath = modelWorkingDirectory + "ATIS.slot.lstm";
	std::string networkConfiguration;
	networkConfiguration += "modelPath=\"" + modelFilePath + "\"";

	VariableSchema inputLayouts;
	VariableSchema outputLayouts;
	IEvaluateModelExtended<float> *eval;
	eval = SetupNetworkAndGetLayouts(networkConfiguration, inputLayouts, outputLayouts);

	vector<size_t> inputBufferSize;
	for (size_t i = 0; i < inputLayouts.size(); i++)
	{
		fprintf(stderr, "Input node name: %ls\n", inputLayouts[i].m_name.c_str());
		fprintf(stdout, "Input feature dimension: %" PRIu64 "\n", inputLayouts[i].m_numElements);
		inputBufferSize.push_back(inputLayouts[i].m_numElements);
	}

	vector<size_t> outputBufferSize;
	for (size_t i = 0; i < outputLayouts.size(); i++)
	{
		fprintf(stderr, "Output node name: %ls\n", outputLayouts[i].m_name.c_str());
		fprintf(stdout, "Output feature dimension: %" PRIu64 "\n", outputLayouts[i].m_numElements);
		outputBufferSize.push_back(outputLayouts[i].m_numElements);
	}

	// assume sequence length is 20 (step size)
	size_t seqLen = 20;

	Values<float> inputBuffers = inputLayouts.CreateBuffers<float>(inputBufferSize);
	Values<float> outputBuffers = outputLayouts.CreateBuffers<float>(outputBufferSize);

	// This is fake input, you need to change input vectors arrording your data 
	for (size_t k = 0; k < inputLayouts.size(); k++)
	{
		// prepare input for each input node
		size_t inputDim = inputLayouts[k].m_numElements;

		for (size_t i = 0; i < seqLen; i++)
		{
			size_t onehotIdx = i;
			for (size_t j = 0; j < inputDim; j++)
			{
				if (j == onehotIdx)
				{
					inputBuffers[k].m_buffer.push_back(1);
				}
				else
				{
					inputBuffers[k].m_buffer.push_back(0);
				}
			}
		}
	}

	// forward propagation
	eval->ForwardPass(inputBuffers, outputBuffers);

	// get output from output layer
	auto buf = outputBuffers[0].m_buffer;
	auto iter = buf.begin();

	std::vector<int> outputs;
	size_t outputDim = outputLayouts[0].m_numElements;

	for (int i = 0; i < seqLen; i++)
	{
		auto max_iter = std::max_element(iter, iter + outputDim);
		auto index = max_iter - iter;
		outputs.push_back(static_cast<int>(index));
	}

	fprintf(stdout, "Indices of max value in output layer for each step:\n");
	for (std::vector<int>::iterator iter = outputs.begin(); iter != outputs.end(); iter++)
	{
		fprintf(stdout, "%" PRIu64 "\n", *iter);
	}

	eval->Destroy();
	system("pause");
	return 0;
}

