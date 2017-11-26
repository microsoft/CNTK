#include "proto/onnx/core/op.h"

namespace ONNXIR {

    #define REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(OpName)                                                    \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                        \
        .Description("Elementwise "#OpName" takes one or more input data (Tensor<T>) and produces one "     \
            "output data (Tensor<T>) where the declared function is applied to the input "                  \
            "tensors elementwise.")                                                                         \
        .Input("data_0", "First of the input tensors. Can be inplace.", "T")                                \
        .Output("output", "Output tensor. Same dimension as inputs.", "T")                                  \
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },                                              \
            "Constrain input and output types to float tensors.");

    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Add)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Sub)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Mul)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Div)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Max)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Min)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Sum)
    REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Mean)

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Neg)
        .Description("Neg takes one input data (Tensor<T>) and produces one output data \
            (Tensor<T>) where each element flipped sign, y = -x, is applied to \
            the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Abs)
        .Description("Absolute takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the absolute is, y = abs(x), is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
             "Constrain input and output types to float tensors.");

    // Take from ONNX
    REGISTER_OPERATOR_SCHEMA(Reciprocal)
        .Description("Reciprocal takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the reciprocal is, y = 1/x, is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Floor)
        .Description("Floor takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the floor is, y = floor(x), is applied to "
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Ceil)
        .Description("Ceil takes one input data (Tensor<T>) and produces one output data"
            "(Tensor<T>) where the ceil is, y = ceil(x), is applied to"
            "the tensor elementwise.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(Clip)
        .Description("Clip operator limits the given input within an interval. "
            "The interval is specified with arguments 'min' and 'max'. They default to "
            "numeric_limits::lowest() and numeric_limits::max() respectively. The clipping "
            "operation can be done in in-place fashion too, where the input and output blobs "
            "are the same.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("min", "Minimum value, under which element is replaced by min", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("max", "Maximum value, under which element is replaced by max", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Sqrt)
        .Description("Square root takes one input data (Tensor<T>) and produces one output "
            "data Tensor<T>) where the square root is, y = x^0.5, is applied to "
            "the tensor elementwise. If x is negative, then it will return NaN.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Exp)
        .Description("Calculates the exponential of the given input tensor, element-wise. "
            "This operation can be done in an in-place fashion too, by providing the same "
            "input and output blobs.")
        .Input("input", "input tensor", "T")
        .Output("output", "The exponential of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Log)
        .Description("Calculates the natural log of the given input tensor, element-wise. "
            "This operation can be done in an in-place fashion too, by providing the same "
            "input and output blobs.")
        .Input("input", "input tensor", "T")
        .Output("output", "The natural  log of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Pow)
        .Description("Pow takes input data (Tensor<T>) and an argument exponent, and "
            "produces one output data (Tensor<T>) where the function `f(x) = x^exponent`, "
            "is applied to the data tensor elementwise.")
        .Input("input", "input tensor", "T")
        .Output("output", "The x^exponent value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("exponent", "The exponent of the power function.", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Dot)
        .Description("Apply dot product between 2 tensors. Similar to numpy implementation: "
            "https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html")
        .Input("X", "Input tensor of any shape", "T")
        .Input("Y", "Input tensor of any shape", "T")
        .Output("output", "Output tensor the dot product between X and Y.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Gemm)
        .Description("(General Matrix multiplication: "
            "https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3 "
            "Compute Y = alpha * A * B + beta * C, where input tensor A has dimension (M X K), "
            "input tensor B has dimension (K X N), input tensor C and output tensor Y have "
            "dimension (M X N). Input tensor C can be used inplace as the output tensor Y. "
            "If attribute broadcast is non-zero, input tensor C will be broadcasted to match the "
            "dimension requirement. If A can be transposed before doing the computation if "
            "attribute transA is non-zero, same for B and transB. ")
        .Input("A", "Input tensor A", "T")
        .Input("B", "Input tensor B", "T")
        .Input("C", "Input tensor C, can be inplace.", "T")
        .Output("Y", "Output tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("transA",
              "Whether A should be transposed",
              AttrType::AttributeProto_AttributeType_INT)
        .Attr("transB",
              "Whether B should be transposed",
              AttrType::AttributeProto_AttributeType_INT)
        .Attr("broadcast",
              "Whether C should be broadcasted",
              AttrType::AttributeProto_AttributeType_INT)
        .Attr("alpha",
              "Scalar multiplier for the product of input tensors A * B",
              AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("beta",
              "Scalar multiplier for input tensor C",
              AttrType::AttributeProto_AttributeType_FLOAT);
}
