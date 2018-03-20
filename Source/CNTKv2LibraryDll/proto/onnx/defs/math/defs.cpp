#include "proto/onnx/core/op.h"

namespace ONNXIR {

#define REGISTER_ELEMENTWISE_BROADCAST_OPERATOR_SCHEMA(OpName)                                          \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                        \
        .Description(                                                                                        \
            "Performs element-wise binary "#OpName" (with limited broadcast support)."                        \
                                                                                                            \
            "If necessary, the right-hand-side argument will be broadcasted to match the shape of"            \
            "left-handside argument. When broadcasting is specified, the second tensor can either be of"    \
            "size 1 (a scalar value) or having its shape as a contiguous subset of the first tensor's"        \
            "shape. The starting of the mutually equal shape is specified by the argument \"axis\" and if"    \
            "it is not set, suffix matching is assumed. 1-dim expansion doesn't work yet. "                    \
                                                                                                            \
            "For example, the following tensor shapes are supported (with broadcast=1): "                    \
            "shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar"                                    \
            "shape(A) = (2, 3, 4, 5), shape(B) = (5,)"                                                        \
            "shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)"                                                    \
            "shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1"                                        \
            "shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0"                                            \
                                                                                                            \
            "Attribute broadcast=1 needs to be passed to enable broadcasting")                                \
        .Input("A", "First operand, should share the type with the second operand.", "T")                   \
        .Input("B", "Second operand. With broadcasting can be of smaller size than A. "                     \
            "If broadcasting is disabled it should be of the same size..", "T")                             \
        .Output("C", "Result, has same dimensions and type as A.", "T")                                     \
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },                      \
            "Constrain input and output types to float tensors.")                                           \
        .Attr("axis", "If set, defines the broadcast dimensions.",                                          \
            AttrType::AttributeProto_AttributeType_INT)                                                     \
        .Attr("broadcast", "Enable broadcasting.",                                                          \
            AttrType::AttributeProto_AttributeType_INT);

    REGISTER_ELEMENTWISE_BROADCAST_OPERATOR_SCHEMA(Add)
        REGISTER_ELEMENTWISE_BROADCAST_OPERATOR_SCHEMA(Sub)
        REGISTER_ELEMENTWISE_BROADCAST_OPERATOR_SCHEMA(Mul)
        REGISTER_ELEMENTWISE_BROADCAST_OPERATOR_SCHEMA(Div)

#define REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(OpName, output)                                            \
    REGISTER_OPERATOR_SCHEMA(OpName)                                                                        \
        .Description("Element-wise "#OpName" of each of the input tensors. The first input tensor can be "  \
            "used in-place as the output tensor, in which case the "#OpName" will be done in "              \
            "place and results will be accumulated in input0. All inputs and outputs must "                 \
            "have the same shape and data type.")                                                           \
        .Input("data_0", "First operand, should share the type with the second operand.", "T")              \
        .Output(#output, "Result, has same dimensions and type as A.", "T")                                 \
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },                      \
            "Constrain input and output types to float tensors.");

        REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Max, "max")
        REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Min, "min")
        REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Sum, "sum")
        REGISTER_ELEMENTWISE_OPERATOR_SCHEMA(Mean, "mean")

        // Taken from ONNX
        REGISTER_OPERATOR_SCHEMA(Neg)
        .Description("Neg takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where each element flipped sign, y = -x, is applied to "
            "the tensor elementwise.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Abs)
        .Description("Absolute takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the absolute is, y = abs(x), is applied to "
            "the tensor elementwise.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Take from ONNX
    REGISTER_OPERATOR_SCHEMA(Reciprocal)
        .Description("Reciprocal takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the reciprocal is, y = 1/x, is applied to "
            "the tensor elementwise.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
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
        .Attr("min", "Minimum value, under which element is replaced by min",
            AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("max", "Maximum value, under which element is replaced by max",
            AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Sqrt)
        .Description("Square root takes one input data (Tensor<T>) and produces one output "
            "data Tensor<T>) where the square root is, y = x^0.5, is applied to "
            "the tensor elementwise. If x is negative, then it will return NaN.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
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
        .Input("X", "Input tensor of any shape", "T")
        .Input("Y", "The exponent of the power function.", "T")
        .Output("Z", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(MatMul)
        .Description("Matrix product that behaves like numpy.matmul: "
            "https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html")
        .Input("A", "N-dimensional matrix A", "T")
        .Input("B", "N-dimensional matrix B", "T")
        .Output("Y", "Matrix multiply results from A * B", "T")
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
