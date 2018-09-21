//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"

#include "Logger.h"
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
    std::once_flag ONNXFormat::op_schema_initializer_flag_;
    static std::string defaultLoggerId{"Default"};
    static onnxruntime::Logging::LoggingManager default_logging_manager_{ 
        std::unique_ptr<onnxruntime::Logging::ISink>{new CNTKClogSink{}},
        [](){
            onnxruntime::Logging::Severity severity;
            switch (GetTraceLevel())
            {
            case TraceLevel::Error:
                severity = onnxruntime::Logging::Severity::kERROR;
                break;
            case TraceLevel::Warning:
                severity = onnxruntime::Logging::Severity::kWARNING;
                break;
            case TraceLevel::Info:
                severity = onnxruntime::Logging::Severity::kINFO;
                break;
            default:
                severity = onnxruntime::Logging::Severity::kFATAL;
            }
            return severity;
        }(),
        false,
        onnxruntime::Logging::LoggingManager::InstanceType::Default,
        &defaultLoggerId };

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

void ONNXFormat::InitializeLotusIR()
{
    //
    // Initializing ONNX_NAMESPACE::Utils::DataTypeUtils::GetTypeStrToProtoMap()
    // 
    // This is a static unordered_map<string, TypeProto> variable that stores the mapping from type name(string) to TypeProto.
    // If used without proper initialization, we risk poluting this static map: 
    // Whenever it sees a TypeProto with an unseen type name, it tries to store that TypeProto into the map. 
    // That TypeProto object might very likely contain TensorShapeProto, which describes the shape for that particular tensor. 
    // This shape will become the default for every TypeProto object created from that type name later on. 
    // And this leads to lots of unexpected errors such as shape inference failure. 
    //
    // The solution is to initialize the map at the first run. 
    std::call_once(op_schema_initializer_flag_, [&]() {
        ONNX_NAMESPACE::OpSchema tmpSchemaForInitializingAllTensorTypes;
        tmpSchemaForInitializingAllTensorTypes.TypeConstraint("T", ONNX_NAMESPACE::OpSchema::all_tensor_types(), "");
    });
}

void ONNXFormat::Save(const FunctionPtr& src, const std::wstring& filepath)
{
    InitializeLotusIR();

    auto model = CNTKToONNX::CreateModel(src);
#ifdef _WIN32
    onnxruntime::Model::Save(*model, filepath);
#else
    onnxruntime::Model::Save(*model, ToLegacyString(ToUTF8(filepath)));
#endif
}

FunctionPtr ONNXFormat::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice)
{
    InitializeLotusIR();

    std::shared_ptr<onnxruntime::Model> model;

#ifdef _WIN32
    onnxruntime::common::Status loadStatus = onnxruntime::Model::Load(filepath, model);
#else
    onnxruntime::common::Status loadStatus = onnxruntime::Model::Load(ToLegacyString(ToUTF8(filepath)), model);
#endif
    if (!loadStatus.IsOK())
        LogicError("Failed to load model: '%s'", loadStatus.ErrorMessage().c_str());

    FunctionPtr cntkFunction = ONNXToCNTK::CreateGraph(&model->MainGraph(), computeDevice);
    return cntkFunction;
}
