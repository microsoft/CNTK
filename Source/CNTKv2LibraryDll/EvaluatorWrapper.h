//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{

	class EvaluatorWrapper
	{

	public:

		EvaluatorWrapper(const wchar_t* modelFilePath);

		~EvaluatorWrapper();

		int GetModelArgumentsInfo(wchar_t*** inputNames, unsigned int** inputSizes);

		int GetModelOutputsInfo(wchar_t*** outputNames, unsigned int** outputSizes);

		int EvaluateSequence(
			unsigned int inputCount, wchar_t** inputNames,
			unsigned int outputCount, wchar_t** outputNames,
			unsigned int sequenceLength,
			unsigned int* inputBufferLengths, float** stackedInputs,
			bool** sequenceResetFlags,
			unsigned int* outputBufferLengths, float** stackedOutputs);

		int EvaluateBatch(
			unsigned int inputCount, wchar_t** inputNames,
			unsigned int outputCount, wchar_t** outputNames,
			unsigned int batchSize,
			unsigned int* inputBufferLengths, float** stackedInputs,
			unsigned int* outputBufferLengths, float** stackedOutputs);

	private:

		int GetVariableInfo(std::vector<Variable> vars, wchar_t*** names, unsigned int** sizes);

		Variable GetVariableByName(const std::vector<Variable>& vars, const std::wstring& varName);

		FunctionPtr m_func;
	};

}
