#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "core/common/common.h"

namespace LotusIR {
constexpr const char* kNoOp = "NoOp";
constexpr const char* kConstant = "Constant";
constexpr const char* kFunctionOp = "_kFunctionOp";
constexpr const char* kConstantValue = "value";
constexpr const char* kOnnxDomain = "";
constexpr const char* kMLDomain = "ai.onnx.ml";
constexpr const char* kMSDomain = "com.microsoft";
constexpr const char* kCpuExecutionProvider = "CPUExecutionProvider";
constexpr const char* kCudaExecutionProvider = "CUDAExecutionProvider";
}  // namespace LotusIR
