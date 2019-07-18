//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This file contains a plain C API for a subset of the CNTK model evaluation functionality.
//

#pragma once

#include <stdint.h>

#ifdef _WIN32
#ifdef CNTKV2LIBRARYDLL
#define CNTK_API __declspec(dllexport)
#else
#define CNTK_API __declspec(dllimport)
#endif
#else // no DLLs on Linux
#define CNTK_API
#endif

//
// We can't have name-mangling for the plain C API, thus the extern "C" is required.
// The exported functions have "CNTK_" prefixes because we can't use a namespace here.
//
#ifdef __cplusplus
extern "C"
{
#endif

typedef void* CNTK_ModelHandle;

#define CNTK_STATUSCODE_DescriptionSize 4096u

typedef struct CNTK_StatusCode
{
    int32_t value;
    char description[CNTK_STATUSCODE_DescriptionSize];
} CNTK_StatusCode;

//
// Status codes
//

// Function was called successfuly.
#define CNTK_SUCCESS 0

// Null or invalid model handle was passed to API call.
#define CNTK_ERROR_INVALID_HANDLE -1

// Null pointer was passed as argument to API call where non-null pointer is required.
#define CNTK_ERROR_NULL_POINTER -2

// An CNTK internal error.
#define CNTK_ERROR_INTERNAL_ERROR -3

// An exception was thrown during the API call.
#define CNTK_ERROR_INVALID_INPUT -4

// Invalid model handle.
#define CNTK_INVALID_MODEL_HANDLE 0

//
// Device kind.
//
typedef enum CNTK_DeviceKind
{
    CNTK_DeviceKind_CPU,
    CNTK_DeviceKind_GPU,
} CNTK_DeviceKind;

//
// Device desciptor.
//
typedef struct CNTK_DeviceDescriptor
{
    CNTK_DeviceKind kind;
    uint32_t id; // CUDA device id, in case of GPU, always 0 for CPU.
} CNTK_DeviceDescriptor;

//
// Returns all available devices.
//
// Parameters:
//     devices [in/out]: list of devices.
//     size[out]: size of the list.
//
CNTK_API CNTK_StatusCode CNTK_AllDevices(
    /*[out]*/ CNTK_DeviceDescriptor** devices,
    /*[out]*/ uint32_t* size);

//
// Returns default device.
//
CNTK_API CNTK_StatusCode CNTK_DefaultDevice(
    /*[in/out]*/ CNTK_DeviceDescriptor* device);

//
// Loads a model from the specified file and returns an opaque handle to the model
// that should be passed to further operations.
//
// Parameters:
//     modelFilePath [in]: a null-terminated path to a CNTK model file
//     device [in]: device descriptor.
//     model [out]: the resulting loaded model
//
CNTK_API CNTK_StatusCode CNTK_LoadModel(
    /*[in]*/ const char* modelFilePath,
    /*[in]*/ const CNTK_DeviceDescriptor* device,
    /*[out]*/ CNTK_ModelHandle* model);

//
// Loads a model from the specified buffer and returns an opaque handle to the model
// that should be passed to further operations.
//
// Parameters:
//     modelData [in]: a buffer that holds the CNTK model
//     modelDataLen [in]: the length of the buffer
//     device [in]: device descriptor
//     model [out]: the resulting loaded model
//
CNTK_StatusCode CNTK_LoadModel_FromArray(
    /*[in]*/ const void* modelData,
    /*[in]*/ int modelDataLen,
    /*[in]*/ const CNTK_DeviceDescriptor* device,
    /*[out]*/ CNTK_ModelHandle* model);

enum CNTK_ParameterCloningMethod
{
    ///
    /// Parameters are shared between the Function being cloned and the new clone
    ///
    CNTK_ModelParameterShare,

    ///
    /// New learnable Parameters are created and initialized with the current values of the
    /// corresponding Parameters of the Function being cloned
    ///
    CNTK_ModelParameterClone,

    ///
    /// Parameters are cloned and made immutable; i.e. Constants in the new clone
    /// (e.g. for use as a fixed feature extractor)
    ///
    CNTK_ModelParameterFreeze
};

//
// Clones the specified model using the provided method.
//
// Parameters:
//    model [in]: model to clone
//    method [in]: method of cloning
//    flattened [in]: whether flatten all block functions in the model
//    cloned [out]: the resulting cloned model
//
CNTK_API CNTK_StatusCode CNTK_CloneModel(
    /*[in]*/ CNTK_ModelHandle model,
    /*[in]*/ CNTK_ParameterCloningMethod method,
    /*[in]*/ bool flattened,
    /*[out]*/ CNTK_ModelHandle* cloned);

//
// Releases all resources associated with the model.
//
// Parameters:
//    model [in]: model to release
//
CNTK_API void CNTK_ReleaseModel(
    /*[in]*/ CNTK_ModelHandle model);

//
// Represents a shape of multi dimensional array. Counterpart of CNTK::NDShape.
//
typedef struct CNTK_Shape
{
    uint32_t* value;   // Buffer containing the shape values for all dimensions
    uint32_t size;     // Buffer size
} CNTK_Shape;

//
// Represents a variable. Counterpart of CNTK::Variable.
//
typedef struct CNTK_Variable
{
    char* name;     // A null terminating name of the variable
    CNTK_Shape shape;  // Variable shape
} CNTK_Variable;

//
// Gets all arguments of the model.
//
// Parameters:
//    model [in]: model from where the argument info will be fetched
//    arguments [out]: model arguments
//    numArguments [out]: number of fetched arguments
//
CNTK_API CNTK_StatusCode CNTK_GetModelArgumentsInfo(
    /*[in]*/  CNTK_ModelHandle model,
    /*[out]*/ CNTK_Variable** arguments,
    /*[out]*/ uint32_t* numArguments);

//
// Gets all outputs of the model.
//
// Parameters:
//    model [in]: model from where the output info will be fetched
//    outputs [out]: model outputs
//    numOutputs [out]: number of fetched outputs
//
CNTK_API CNTK_StatusCode CNTK_GetModelOutputsInfo(
    /*[int]*/ CNTK_ModelHandle model,
    /*[out]*/ CNTK_Variable** outputs,
    /*[out]*/ uint32_t* numOutputs);

//
// Represents a value. Counterpart of CNTK::Value and CNTK::NDArrayView.
// All data should be column major.
//
typedef struct CNTK_Value
{
    CNTK_Shape shape;   // Shape of the value
    float* data;        // Actual buffer, size is equal to total size of the elements in the shape.
} CNTK_Value;

//
// Represents a value.
//
CNTK_API CNTK_StatusCode CNTK_EvaluateSequence(CNTK_ModelHandle model,
    /*[in]*/const CNTK_Variable* inputs,
    /*[in]*/const CNTK_Value* inputValues,
    /*[in]*/const bool* inputResetFlags,
    /*[in]*/uint32_t numInputs,
    /*[in]*/const CNTK_Variable* outputs,
    /*[in]*/uint32_t numOutputs,
    /*[in/out]*/CNTK_Value** outputValues);

//
// Auxiliary functions.
//

//
// Releases previously allocated POD type array.
//
CNTK_API void CNTK_ReleaseArray(/*[in]*/ void* array);

//
// Releases all resources associated with the variable,
// all internal members will be properly cleaned.
//
CNTK_API void CNTK_CleanVariable(/*[in]*/ CNTK_Variable* variable);

//
// Releases all resources associated with the value.
// all internal members will be properly cleaned.
//
CNTK_API void CNTK_CleanValue(/*[in]*/ CNTK_Value* value);

//
// Releases all resources associated with the shape.
//
CNTK_API void CNTK_CleanShape(/*[in]*/ CNTK_Shape* shape);

#ifdef __cplusplus
}
#endif
