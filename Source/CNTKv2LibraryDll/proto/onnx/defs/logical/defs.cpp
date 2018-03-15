#include "proto/onnx/core/op.h"

namespace ONNXIR {

#define REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(OpName)                                                      \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                            \
        .Description("Returns the tensor resulted from performing the '"#OpName"' logical operation"            \
        "elementwise on the input tensors A and B. If broadcasting is enabled, the right-hand-side"             \
        "argument will be broadcasted to match the shape of left-hand-side argument. Refer to Add for"          \
        "a detailed description of the broadcasting rules.")                                                    \
        .Input("A", "First operand, should share the type with the second operand.", "T1")                      \
        .Input("B", "Second operand. With broadcasting can be of smaller size than A."                          \
            "If broadcasting is disabled, it should be of the same size.", "T1")                                \
        .Output("C", "Result, has same dimensions as A and type bool.", "T2")                                   \
        .TypeConstraint("T1", { "tensor(float16)", "tensor(float)", "tensor(double)" },                         \
                "Constrain input to float tensors.")                                                            \
        .TypeConstraint("T2", { "tensor(bool)" }, "Constrain output types to bool tensor.")                     \
        .Attr("axis", "If set, defines the broadcast dimensions.",                                              \
            AttrType::AttributeProto_AttributeType_INT)                                                         \
        .Attr("broadcast", "Pass 1 to enable broadcasting.",                                                    \
            AttrType::AttributeProto_AttributeType_INT);

    //‘GREATER’, ‘LESS’, ‘EQUALS,
    REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(Greater)
        REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(Less)
        REGISTER_BINARY_COMPARISON_OPERATOR_SCHEMA(Equal)

#define REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(OpName)                                                           \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                            \
        .Description("Returns the tensor resulted from performing the '"#OpName"' logical operation"            \
            "elementwise on the input tensors A and B. If broadcasting is enabled, the right-hand-side"         \
            "argument will be broadcasted to match the shape of left-hand-side argument. Refer to Add"          \
            " for a detailed description of the broadcasting rules.")                                           \
        .Input("A", "First operand.", "T")                                                                      \
        .Input("B", "Second operand. With broadcasting can be of smaller size than A. If broadcasting"          \
            "is disabled, it should be of the same size.", "T")                                                 \
        .Output("C", "Result, has same dimensions and A and type bool.", "T")                                   \
        .TypeConstraint("T", { "tensor(bool)" }, "Constrain input and output types to bool tensor.")            \
        .Attr("axis", "If set, defines the broadcast dimensions.",                                              \
            AttrType::AttributeProto_AttributeType_INT)                                                         \
        .Attr("broadcast", "Pass 1 to enable broadcasting.",                                                    \
            AttrType::AttributeProto_AttributeType_INT);


        // ‘AND, ‘OR’, ‘XOR’
        REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(And)
        REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(Or)
        REGISTER_BINARY_LOGIC_OPERATOR_SCHEMA(Xor)

        REGISTER_OPERATOR_SCHEMA(Not)
        .Description("Performs element-wise negation.")
        .Input("X", "Input tensor of type bool.", "T")
        .Output("Y", "  Output tensor of type bool.", "T")
        .TypeConstraint("T", { "tensor(bool)" }, "Constrain input and output types to bool tensor.");

}
