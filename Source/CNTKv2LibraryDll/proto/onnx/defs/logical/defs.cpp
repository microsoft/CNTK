#include "proto/onnx/core/op.h"

namespace ONNXIR {

    #define REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(OpName)                                                  \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                            \
        .Description("Computes the elementwise comparison `"#OpName"` between "                                 \
            "`left` and `right` input tensor. The result is a tensor of type integer "                          \
            "in which `0` mean false and `1` mean true.")                                                       \
        .Input("left", "Left input tensor for the operator.", "T1")                                             \
        .Input("right", "Right input tensor for the operator.", "T1")                                           \
        .Output("output", "Result tensor of type `int`, 0 mean False and 1 mean True.", "T2")                   \
        .TypeConstraint("T1", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input to float tensors.")                   \
        .TypeConstraint("T2", { "tensor(int32)" }, "Constrain output types to int tensor.");

    //‘GREATER’, ‘LESS’, ‘EQUALS,
    REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(GT)
    REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(LE)
    REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(EQ)

    #define REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(OpName)                                                       \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                            \
        .Description("Computes the elementwise logical operation '"#OpName"' between "                          \
            "`left` and `right` input tensor. The result is a tensor of type integer "                          \
            "in which `0` mean false and `1` mean true.")                                                       \
        .Input("left", "Left input tensor for the logical operator.", "T")                                      \
        .Input("right", "Right input tensor for the logical operator.", "T")                                    \
        .Output("output", "Result tensor of type `int`, 0 mean False and 1 mean True.", "T")                    \
        .TypeConstraint("T", { "tensor(int32)" }, "Constrain input and output types to int tensor.");

    // ‘AND, ‘OR’, ‘XOR’
    REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(And)
    REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(Or)
    REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(Xor)

    REGISTER_OPERATOR_SCHEMA(Not)
        .Description("Performs element-wise negation.")
        .Input("X", "Input tensor of type bool.", "T")
        .Output("Y", "  Output tensor of type bool.", "T")
        .TypeConstraint("T", { "tensor(int32)" }, "Constrain input and output types to int tensor.");

}
