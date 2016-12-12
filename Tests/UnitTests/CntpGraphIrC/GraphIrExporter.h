#pragma once
//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"

#include <iostream>
#include <stdio.h>
#include <string>

#pragma warning(push, 0)
#include <graphir.pb.h>
#include <google/protobuf/util/json_util.h>
#pragma warning(pop)

extern std::wstring TransformCntkToGraphIr(
    const std::string& filename,
    const CNTK::DeviceDescriptor& device);

extern std::wstring TransformGraphIrToCntk(
    const std::string& filename,
    const CNTK::DeviceDescriptor& device);

extern void ExecuteModelOnRandomData(
    std::string filename,
    std::unordered_map<std::wstring, std::vector<float>>& inputs,
    std::unordered_map<std::wstring, std::vector<float>>& outputs,
    const CNTK::DeviceDescriptor& device);

extern void ExecuteModelOnGivenData(
    CNTK::FunctionPtr evalFunc,
    std::unordered_map<std::wstring, std::vector<float>>& inputs,
    std::unordered_map<std::wstring, std::vector<float>>& outputs,
    const CNTK::DeviceDescriptor& device);


namespace GRAPHIR
{
    const graphIR::Graph* Serialize(const CNTK::FunctionPtr& modelFuncPtr);
    const CNTK::FunctionPtr Deserialize(const graphIR::Graph* message);
}
