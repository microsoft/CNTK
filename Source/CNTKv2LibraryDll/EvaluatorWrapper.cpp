//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "EvaluatorWrapper.h"
#include "CNTKLibraryCAPI.h"

namespace CNTK
{
	EvaluatorWrapper::EvaluatorWrapper(const wchar_t* modelFilePath)
	{
		m_func = Function::Load(modelFilePath);
	}

	EvaluatorWrapper::~EvaluatorWrapper()
	{
		// TODO: How do we release m_func?
	}

	int EvaluatorWrapper::GetModelArgumentsInfo(wchar_t*** inputNames, unsigned int** inputSizes)
	{
		return GetVariableInfo(m_func->Arguments(), inputNames, inputSizes);
	}

	int EvaluatorWrapper::GetModelOutputsInfo(wchar_t*** outputNames, unsigned int** outputSizes)
	{
		return GetVariableInfo(m_func->Outputs(), outputNames, outputSizes);
	}

	int EvaluatorWrapper::EvaluateSequence(
		unsigned int inputCount, wchar_t** inputNames,
		unsigned int outputCount, wchar_t** outputNames,
		unsigned int sequenceLength,
		unsigned int* inputBufferLengths, float** stackedInputs,
		bool** sequenceResetFlags,
		unsigned int* outputBufferLengths, float** stackedOutputs)
	{
		// TODO
	}

	int EvaluatorWrapper::EvaluateBatch(
		unsigned int inputCount, wchar_t** inputNames,
		unsigned int outputCount, wchar_t** outputNames,
		unsigned int batchSize,
		unsigned int* inputBufferLengths, float** stackedInputs,
		unsigned int* outputBufferLengths, float** stackedOutputs)
	{
		if (inputCount != 1)
		{
			throw std::exception("EvaluateBatch only supports inputCount = 1 for now.");
		}
		
		if (outputCount != 1)
		{
			throw std::exception("EvaluateBatch only supports outputCount = 1 for now.");
		}

		Variable inputVar = GetVariableByName(m_func->Arguments(), inputNames[0]);
		Variable outputVar = GetVariableByName(m_func->Outputs(), outputNames[0]);

		// Crate the input buffer.
		// TODO: Does this end up being a shallow or deep copy of the input?
		unsigned int inputVectorLength = inputBufferLengths[0]; // in elements
		float* inputVector = stackedInputs[0];
		std::vector<float> inputBuffer(inputVector, inputVector + inputVectorLength);

		// If the input is a batch of 5 frames that are each 80-elements, then the shape needs to be: { 80, 5 }

		// TODO: I think some adjustments to the example shape (first argument) are needed in order to get CNTK to understand that inputBuffer
		// has one or more input vectors concatenated in the contiguous block of memory.
		ValuePtr inputVal = Value::CreateBatch<float>(inputVar.Shape(), inputBuffer, DeviceDescriptor::UseDefaultDevice());
		std::unordered_map<Variable, ValuePtr> inputMap = { { inputVar, inputVal } };

		// TODO: create the output
		// .. I need to actually learn what these values, ndarrayviews, etc. are so that I do this correctly!

	}

	int EvaluatorWrapper::GetVariableInfo(std::vector<Variable> vars, wchar_t*** names, unsigned int** sizes)
	{
		if ((nullptr == names) || (nullptr == sizes))
		{
			return CNTK_ERROR_NULL_POINTER;
		}

		int varCount = vars.size();

		// Now that we know the input count, allocate the arrays.
		wchar_t** namesBuffer = new wchar_t*[varCount];
		unsigned int* sizesBuffer = new unsigned int[varCount];

		for (int i = 0; i < varCount; ++i)
		{
			// assume variable is 1-dimensional
			sizesBuffer[i] = vars[i].Shape().TotalSize();

			// copy variable name
			const std::wstring& name = vars[i].Name();
			unsigned int nameBufferSize = name.size() + 1;
			wchar_t* nameBuffer = new wchar_t[nameBufferSize]; // +1 for the null-terminator
			wcsncpy(nameBuffer, name.c_str(), nameBufferSize);
			namesBuffer[i] = nameBuffer;
		}

		*names = namesBuffer;
		*sizes = sizesBuffer;

		return varCount;
	}

	Variable EvaluatorWrapper::GetVariableByName(const std::vector<Variable>& vars, const std::wstring& varName)
	{
		auto itr = std::find_if(vars.cbegin(), vars.cend(), [&varName](const Variable& v) { return v.Name().compare(varName) == 0; });
		if (itr == vars.cend())
		{
			throw std::exception("GetVariableByName failed.");
		}

		return *itr;
	}
}
