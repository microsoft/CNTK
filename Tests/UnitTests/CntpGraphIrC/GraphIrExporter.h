#pragma once
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#define _CRT_SECURE_NO_WARNINGS
#include "CNTKLibrary.h"

#include <iostream>
#include <stdio.h>
#include <string>

#pragma warning(push, 0)
#include <graphid.pb.h>
#include <google/protobuf/util/json_util.h>
#pragma warning(pop)


extern CNTK::FunctionPtr GraphIrToCntkGraph(graphIR::Graph* /*graphIrPtr*/, CNTK::FunctionPtr /*modelFuncPtr*/);
extern graphIR::Graph* CntkGraphToGraphIr(std::wstring filename, CNTK::FunctionPtr evalFunc);

extern void RetrieveInputBuffers(
    CNTK::FunctionPtr evalFunc,
    std::unordered_map<std::wstring, std::vector<float>>& inputs);

extern void ExecuteModel(
    CNTK::FunctionPtr evalFunc,
    std::unordered_map<std::wstring, std::vector<float>>& inputs,
    std::unordered_map<std::wstring, std::vector<float>>& outputs);

extern void PrintDictionaryValue(
    const std::wstring& name,
    const CNTK::DictionaryValue& value,
    int indent);

namespace GRAPHIR
{
    const google::protobuf::Message* Serialize(const CNTK::FunctionPtr& modelFuncPtr);
    const CNTK::FunctionPtr Deserialize(const CNTK::FunctionPtr& modelFuncPtr, google::protobuf::Message* message);
}
