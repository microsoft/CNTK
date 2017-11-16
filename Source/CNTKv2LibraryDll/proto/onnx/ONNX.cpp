//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "proto/onnx/core/model.h"
#include "proto/onnx/core/graph.h"
#include "Utils.h"

#include <iostream>
#include <memory>

#include "ONNXToCNTK.h"

using namespace CNTK;

namespace CNTK
{
    static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
            return;
        }

        for (auto input : function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
                "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
        }

        for (auto input : function->Inputs())
        {
            if (input.Owner() != NULL)
            {
                FunctionPtr f = input.Owner();
                PrintGraph(f, spaces + 4, useName);
            }
        }
    }
}

void ONNXFormat::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    auto model = CNTKToONNX::CreateModel(src);
#ifdef _WIN32
    ONNXIR::Model::Save(*model, filepath);
#else
    ONNXIR::Model::Save(*model, ToString(filepath));
#endif
}

FunctionPtr ONNXFormat::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice)
{
    std::shared_ptr<ONNXIR::Model> model;

#ifdef _WIN32
    Status loadStatus = ONNXIR::Model::Load(filepath, &model);
#else
    Status loadStatus = ONNXIR::Model::Load(ToString(filepath), &model);
#endif
    if (!loadStatus.Ok())
        LogicError("Failed to load the model.");

    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(model->MainGraph(), computeDevice);
    return cntkFunction;
}
