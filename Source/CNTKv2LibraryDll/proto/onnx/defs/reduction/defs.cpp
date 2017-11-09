// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "proto/onnx/core/op.h"
#include <functional>

namespace ONNXIR {

std::function<void(OpSchema&)> ReduceDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulted
tensor has the same shape as the input if keepdims equal 1. If keepdims equal 0, then 
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Attr("axes",
                    "A list of integers, along which to reduce max.",
                    AttrType::INTS);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttrType::INT);
        schema.Input(0, "data", "An input tensor.");
        schema.Output(0, "reduced", "Reduced output tensor.");
    };
}
  
REGISTER_OPERATOR_SCHEMA(ReduceMax)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("max"));

REGISTER_OPERATOR_SCHEMA(ReduceMin)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("min"));

REGISTER_OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("sum"));

REGISTER_OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("mean"));

REGISTER_OPERATOR_SCHEMA(ReduceProd)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("product"));

REGISTER_OPERATOR_SCHEMA(ReduceLogSumExp)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ReduceDocGenerator("log sum exponent"));

std::function<void(OpSchema&)> ArgReduceDocGenerator(const char* name) {
    return [=](OpSchema& schema) {
        std::string doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the 
provided axes. The resulted tensor has the same shape as the input if keepdims equal 1. 
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.)DOC";
        ReplaceAll(doc, "{name}", name);
        schema.SetDoc(doc);
        schema.Attr("axes",
                    "A list of integers, along which to reduce max.",
                    AttrType::INTS);
        schema.Attr("keepdims",
                    "Keep the reduced dimension or not, default 1 mean keep reduced dimension.",
                    AttrType::INT);
        schema.Input(0, "data", "An input tensor.");
        schema.Output(0, "reduced", "Reduced output tensor with integer data type.");
    };
}

REGISTER_OPERATOR_SCHEMA(ArgMax)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ArgReduceDocGenerator("max"));

REGISTER_OPERATOR_SCHEMA(ArgMin)
    .NumInputs(1)
    .NumOutputs(1)
    .FillUsing(ArgReduceDocGenerator("min"));

}  // namespace ONNXIR
