//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "proto/onnx/core/graph/model.h"
#include "proto/onnx/core/graph/graph.h"

#include "ONNX.h"
#include "CNTKToONNX.h"
#include "ONNXToCNTK.h"
#include "Utils.h"

#include <iostream>
#include <memory>

using namespace CNTK;
using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    // MaxVersion number in ONNX 1.2 is 7. Change this number (e.g. to 1 or 5) 
    // to experiment with earlier version ONNX. This is to help debugging with reshape op 
    // (and some convolution ops which only passed with newer version)
    // to do this:
    // onnx::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(LotusIR::kOnnxDomain, 1, 5);
    const int ONNX2_1MAX_VERSION = 7;

    // for debugging (and probably useful backward compatibility) propose, use this helper to tell 
    // how to implement a conversion. It is used for reshape op.
    bool IsONNX1_2Supported()
    {
        auto map = onnx::OpSchemaRegistry::DomainToVersionRange::Instance().Map();
        return map.find(onnx::ONNX_DOMAIN) != map.end() && map[onnx::ONNX_DOMAIN].second == ONNX2_1MAX_VERSION;
    }

    static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToLegacyString(ToUTF8(useName ? function->Name() : function->Uid())) + ")" + ToLegacyString(ToUTF8(function->AsString())) << std::endl;
            return;
        }

        for (auto input : function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToLegacyString(ToUTF8(useName ? function->Name() : function->Uid())) + ")" + "->" +
                        "(" + ToLegacyString(ToUTF8(useName ? input.Name() : input.Uid())) + ")" + ToLegacyString(ToUTF8(input.AsString()))
                 << std::endl;
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
    LotusIR::Model::Save(*model, filepath);
#else
    LotusIR::Model::Save(*model, ToLegacyString(ToUTF8(filepath)));
#endif
}

FunctionPtr ONNXFormat::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice)
{
    std::shared_ptr<LotusIR::Model> model;

#ifdef _WIN32
    Lotus::Common::Status loadStatus = LotusIR::Model::Load(filepath, model);
#else
    Lotus::Common::Status loadStatus = LotusIR::Model::Load(ToLegacyString(ToUTF8(filepath)), model);
#endif
    if (!loadStatus.IsOK())
        LogicError("Failed to load model: '%s'", loadStatus.ErrorMessage().c_str());

    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(model->MainGraph(), computeDevice);
    return cntkFunction;
}
