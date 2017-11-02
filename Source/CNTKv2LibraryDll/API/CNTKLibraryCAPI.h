//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This contains a plain C API for a subset of the model evaluation functionality.
//

#pragma once

typedef void* CNTK_ModelHandle;

#define CNTK_INVALID_MODEL_HANDLE nullptr

#define CNTK_ERROR_INVALID_HANDLE -1 // null or invalid model handle was passed to API call
#define CNTK_ERROR_NULL_POINTER -2 // null pointer was passed as argument to API call where non-null pointer is required
#define CNTK_ERROR_EXCEPTION_THROWN -3 // an exception was thrown during the API call

// We can't have name-mangling for the plain C API, thus the extern "C" is required.
// The exported functions have "CNTK_" prefixes because we can't use a namespace here.
#ifdef __cplusplus
extern "C"
#endif
{

// Returns an opaque handle to the model that will be passed to future operations.
// modelFilePath is the null-terminated path to the CNTK model file.
// If model-loading failed, returns CNTK_INVALID_MODEL_HANDLE
CNTK_API CNTK_ModelHandle CNTK_LoadModel(const wchar_t* modelFilePath);

// Releases the allocated memory and other resources of model.
// This always results in deleting writable internal buffers since each clone of the model has a separate copy of them.
// This only results in deleting the read-only buffers if their reference count just went back to 0 as a result of this operation.
CNTK_API void CNTK_ReleaseModel(CNTK_ModelHandle model);

// On success, returns the number of inputs.
// On failure, returns <= 0.
// example output:
//		(*inputNames)[0] = L"InputName1"
//		(*inputNames)[1] = L"InputName2"
//		(*inputSizes)[0] = 50
//		(*inputSizes)[1] = 100
CNTK_API int CNTK_GetModelArgumentsInfo(CNTK_ModelHandle model, wchar_t*** inputNames, unsigned int** inputSizes);

CNTK_API int CNTK_GetModelOutputsInfo(CNTK_ModelHandle model, wchar_t*** outputNames, unsigned int** outputSizes);

CNTK_API void CNTK_ReleaseArrayBuffer(void* arrayBuffer);

CNTK_API int CNTK_EvaluateSequence(CNTK_ModelHandle model,
	unsigned int inputCount, wchar_t** inputNames,
	unsigned int outputCount, wchar_t** outputNames,
	unsigned int sequenceLength,
	unsigned int* inputBufferLengths, float** stackedInputs,
	bool** sequenceResetFlags,
	unsigned int* outputBufferLengths, float** stackedOutputs);

CNTK_API int CNTK_EvaluateBatch(CNTK_ModelHandle model,
	unsigned int inputCount, wchar_t** inputNames,
	unsigned int outputCount, wchar_t** outputNames,
	unsigned int batchSize,
	unsigned int* inputBufferLengths, float** stackedInputs,
	unsigned int* outputBufferLengths, float** stackedOutputs);

CNTK_API int CNTK_SetDefaultDevice(wchar_t* deviceName);

#ifdef __cplusplus
}
#endif
