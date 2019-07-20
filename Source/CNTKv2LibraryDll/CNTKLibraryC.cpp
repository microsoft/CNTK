//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Redirector from C to C++ for public methods.
// This file does not contain any business logic, so if something is returned from C++ land,
// it should pass the result to the calling side to avoid any resource leaks.
//

#define _SCL_SECURE_NO_WARNINGS

#include "stdafx.h"
#include <string>
#include <algorithm>
#include <boost/noncopyable.hpp>
#include "ExceptionWithCallStack.h"
#include "EvaluatorWrapper.h"

using namespace Microsoft::MSR::CNTK;
using namespace CNTK;
using namespace std;

namespace
{
    static CNTK_StatusCode StatusCode(int32_t code, const string& message)
    {
        CNTK_StatusCode result{ code, {0} };
        auto size = min((uint32_t)(message.size() + 1), CNTK_STATUSCODE_DescriptionSize - 1);
        copy(message.c_str(), message.c_str() + size, result.description);
        return result;
    }

    class ExceptionCatcher
    {
    public:
        static CNTK_StatusCode Call(function<void()> action)
        {
            try
            {
                action();
                return CNTK_StatusCode{ CNTK_SUCCESS };
            }
            catch (const IExceptionWithCallStackBase& er)
            {
                string message = "Exception occurred: '";
                message += dynamic_cast<const exception&>(er).what();
                message += "'\n, CallStack: ";
                message += er.CallStack();
                return StatusCode(CNTK_ERROR_INTERNAL_ERROR, message);
            }
            catch (const exception& e)
            {
                return StatusCode(CNTK_ERROR_INTERNAL_ERROR, e.what());
            }
            catch (...)
            {
                return StatusCode(CNTK_ERROR_INTERNAL_ERROR, "Unknown exception.");
            }
        }
    };
}

CNTK_StatusCode CNTK_DefaultDevice(CNTK_DeviceDescriptor* device)
{
    if (!device)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'device' parameter is not allowed to be null");

    return ExceptionCatcher::Call([&]() {
        auto d = DeviceDescriptor::UseDefaultDevice();
        device->id = d.Id();
        device->kind = (d.Type() == DeviceKind::GPU ? CNTK_DeviceKind::CNTK_DeviceKind_GPU : CNTK_DeviceKind::CNTK_DeviceKind_CPU);
    });
}

CNTK_StatusCode CNTK_AllDevices(CNTK_DeviceDescriptor** devices, uint32_t* size)
{
    if (!devices)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'devices' parameter is not allowed to be null");

    if (!size)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'size' parameter is not allowed to be null");

    return ExceptionCatcher::Call([&]() {
        auto all = DeviceDescriptor::AllDevices();
        *devices = new CNTK_DeviceDescriptor[all.size()];
        for (size_t i = 0; i < all.size(); ++i)
        {
            (*devices)[i].id = all[i].Id();
            (*devices)[i].kind = all[i].Type() == DeviceKind::GPU ? CNTK_DeviceKind::CNTK_DeviceKind_GPU : CNTK_DeviceKind::CNTK_DeviceKind_CPU;
        }
        *size = static_cast<uint32_t>(all.size());
    });
}

CNTK_StatusCode CNTK_LoadModel(const char* modelFilePath, const CNTK_DeviceDescriptor* device, CNTK_ModelHandle* handle)
{
    if (!handle)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'handle' parameter is not allowed to be null");

    if (!modelFilePath)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'modelFilePath' parameter is not allowed to be null");

    *handle = nullptr;
    return ExceptionCatcher::Call([&]() { *handle = new CNTKEvaluatorWrapper(modelFilePath, device); });
}

CNTK_StatusCode CNTK_LoadModel_FromArray(const void* modelData, int modelDataLen,
                                         const CNTK_DeviceDescriptor* device, CNTK_ModelHandle* handle)
{
    if (!handle)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'handle' parameter is not allowed to be null");

    if (!modelData)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'modelData' parameter is not allowed to be null");

    if (modelDataLen <= 0)
        return StatusCode(CNTK_ERROR_INVALID_INPUT, "'modelDataLen' parameter must be greater than zero");

    *handle = nullptr;
    return ExceptionCatcher::Call([&]() { *handle = new CNTKEvaluatorWrapper(modelData, modelDataLen, device); });
}

CNTK_StatusCode CNTK_CloneModel(CNTK_ModelHandle model, CNTK_ParameterCloningMethod method, bool flatten, CNTK_ModelHandle* cloned)
{
    if (model == CNTK_INVALID_MODEL_HANDLE)
        return StatusCode(CNTK_INVALID_MODEL_HANDLE, "Invalid model handle");

    if (!cloned)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'handle' parameter is not allowed to be null");

    return ExceptionCatcher::Call([&]() { *cloned = ((EvaluatorWrapper*)model)->Clone(method, flatten).release(); });
}

void CNTK_ReleaseModel(CNTK_ModelHandle model)
{
    delete (EvaluatorWrapper*)model;
}

CNTK_StatusCode CNTK_GetModelArgumentsInfo(CNTK_ModelHandle model, CNTK_Variable** inputs, uint32_t* numInputs)
{
    if (model == CNTK_INVALID_MODEL_HANDLE)
        return StatusCode(CNTK_INVALID_MODEL_HANDLE, "Invalid model handle");

    if (!inputs)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'inputs' parameter is not allowed to be null");

    if(!numInputs)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'numInputs' parameter is not allowed to be null");

    return ExceptionCatcher::Call(
        [&]() { ((EvaluatorWrapper*)model)->GetModelArgumentsInfo(inputs, numInputs); });
}

CNTK_StatusCode CNTK_GetModelOutputsInfo(CNTK_ModelHandle model, CNTK_Variable** outputs, uint32_t* numOutputs)
{
    if (model == CNTK_INVALID_MODEL_HANDLE)
        return StatusCode(CNTK_INVALID_MODEL_HANDLE, "Invalid model handle");

    if (!outputs)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'outputs' parameter is not allowed to be null");

    if (!numOutputs)
        return StatusCode(CNTK_ERROR_NULL_POINTER, "'numOutputs' parameter is not allowed to be null");

    return ExceptionCatcher::Call(
        [&]() { ((EvaluatorWrapper*)model)->GetModelOutputsInfo(outputs, numOutputs); });
}

CNTK_StatusCode CNTK_EvaluateSequence(CNTK_ModelHandle model,
    const CNTK_Variable* inputs,
    const CNTK_Value* inputValues,
    const bool* inputResetFlags,
    uint32_t numInputs,
    const CNTK_Variable* outputs,
    uint32_t numOutputs,
    CNTK_Value** outputValues)
{
    if (model == CNTK_INVALID_MODEL_HANDLE)
        return StatusCode(CNTK_INVALID_MODEL_HANDLE, "Invalid model handle");

    return ExceptionCatcher::Call(
    [&]()
    {
        ((EvaluatorWrapper*)model)->EvaluateSequence(
            inputs, inputValues, inputResetFlags,
            numInputs, outputs, numOutputs, outputValues);
    });
}

void CNTK_ReleaseArray(void* array)
{
    // No destructor will be called!
    delete[] (char*)array;
}

void CNTK_CleanVariable(CNTK_Variable* variable)
{
    if (!variable)
        return;

    delete[] variable->name;
    CNTK_CleanShape(&variable->shape);
}

void CNTK_CleanValue(CNTK_Value* value)
{
    if (!value)
        return;

    delete[] value->data;
    CNTK_CleanShape(&value->shape);
}

void CNTK_CleanShape(CNTK_Shape* shape)
{
    if (!shape)
        return;

    delete[] shape->value;
    shape->size = 0;
}
