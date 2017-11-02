//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "CNTKLibraryCAPI.h"
#include "EvaluatorWrapper.h"

using namespace CNTK;

CNTK_API CNTK_ModelHandle CNTK_LoadModel(const wchar_t* modelFilePath)
{
	EvaluatorWrapper* evaluator = nullptr;

	try
	{
		evaluator = new EvaluatorWrapper(modelFilePath);
	}
	catch (...)
	{
		delete evaluator;
		evaluator = nullptr;
	}

	return (CNTK_ModelHandle)evaluator;
}

CNTK_API void CNTK_ReleaseModel(CNTK_ModelHandle model)
{
	if (model != CNTK_INVALID_MODEL_HANDLE)
	{
		EvaluatorWrapper* evaluator = (EvaluatorWrapper*)model;
		delete evaluator;
	}
}

CNTK_API int CNTK_GetModelArgumentsInfo(CNTK_ModelHandle model, wchar_t*** inputNames, unsigned int** inputSizes)
{
	if (CNTK_INVALID_MODEL_HANDLE == model)
	{
		return CNTK_ERROR_INVALID_HANDLE;
	}

	return ((EvaluatorWrapper*)model)->GetModelArgumentsInfo(inputNames, inputSizes);
}

CNTK_API int CNTK_GetModelOutputsInfo(CNTK_ModelHandle model, wchar_t*** outputNames, unsigned int** outputSizes)
{
	if (CNTK_INVALID_MODEL_HANDLE == model)
	{
		return CNTK_ERROR_INVALID_HANDLE;
	}

	return ((EvaluatorWrapper*)model)->GetModelOutputsInfo(outputNames, outputSizes);
}

CNTK_API void CNTK_ReleaseArrayBuffer(void* arrayBuffer)
{
	delete[] arrayBuffer;
}

CNTK_API int CNTK_EvaluateSequence(CNTK_ModelHandle model,
	unsigned int inputCount, wchar_t** inputNames,
	unsigned int outputCount, wchar_t** outputNames,
	unsigned int sequenceLength,
	unsigned int* inputBufferLengths, float** stackedInputs,
	bool** sequenceResetFlags,
	unsigned int* outputBufferLengths, float** stackedOutputs)
{
	if (CNTK_INVALID_MODEL_HANDLE == model)
	{
		return CNTK_ERROR_INVALID_HANDLE;
	}

	return ((EvaluatorWrapper*)model)->EvaluateSequence(
		inputCount, inputNames,
		outputCount, outputNames,
		sequenceLength,
		inputBufferLengths, stackedInputs,
		sequenceResetFlags,
		outputBufferLengths, stackedOutputs);
}

CNTK_API int CNTK_EvaluateBatch(CNTK_ModelHandle model,
	unsigned int inputCount, wchar_t** inputNames,
	unsigned int outputCount, wchar_t** outputNames,
	unsigned int batchSize,
	unsigned int* inputBufferLengths, float** stackedInputs,
	unsigned int* outputBufferLengths, float** stackedOutputs)
{
	if (CNTK_INVALID_MODEL_HANDLE == model)
	{
		return CNTK_ERROR_INVALID_HANDLE;
	}

	return ((EvaluatorWrapper*)model)->EvaluateBatch(
		inputCount, inputNames,
		outputCount, outputNames,
		batchSize,
		inputBufferLengths, stackedInputs,
		outputBufferLengths, stackedOutputs);
}

CNTK_API int CNTK_SetDefaultDevice(wchar_t* deviceName)
{

}
