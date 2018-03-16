#include "proto/onnx/core/op.h"

namespace ONNXIR {
    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Sigmoid)
        .Description("Sigmoid takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the "
            "tensor elementwise.")
        .Input("X", "input tensor", "T")
        .Output("Y", "The sigmoid value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Tanh)
        .Description("Calculates the hyperbolic tangent of the given input tensor element-wise. "
            "This operation can be done in an in-place fashion too, by providing the same input "
            "and output blobs.")
        .Input("input", "input tensor", "T")
        .Output("output", "The hyperbolic tangent value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Relu)
        .Description("Relu takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the rectified linear function, y = max(0, x), is "
            "applied to the tensor elementwise.")
        .Input("X", "input tensor", "T")
        .Output("Y", "The Relu value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(LeakyRelu)
        .Description("LeakyRelu takes input data (Tensor<T>) and an argument alpha, "
            "and produces one output data (Tensor<T>) where the function "
            ":`f(x) = alpha * x for x < 0`, `f(x) = x for x >= 0`, is applied to the data "
            "tensor elementwise.")
        .Input("X", "input tensor", "T")
        .Output("Y", "The LeakyRelu value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha","Coefficient of leakage", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(PRelu)
        .Description("PRelu takes input data (Tensor<T>) and slope tensor as input, "
            "and produces one output data (Tensor<T>) where the function "
            "`f(x) = slope * x for x < 0`, `f(x) = x for x >= 0`., is applied to the "
            "data tensor elementwise.")
        .Input("X", "Input tensor", "T")
        .Input("Slope", "Slope tensor. If `Slope` is of size 1, the value is shared"
            "across different channels", "T")
        .Output("Y", "The PRelu value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Elu)
        .Description("Elu takes one input data (Tensor<T>) and produces one output data"
            "(Tensor<T>) where the function `f(x) = alpha * (exp(x) - 1.) for x < 0`, "
            "`f(x) = x for x >= 0`., is applied to the tensor elementwise.")
        .Input("X", "input tensor", "T")
        .Output("Y", "The elu value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Coefficient of ELU default to 1",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0));

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Selu)
        .Description("Selu takes one input data (Tensor<T>) and produces one output data "
            "(Tensor<T>) where the scaled exponential linear unit function, "
            "`y = gamma * (alpha * e^x - alpha) for x <= 0`, `f(x) = gamma * x for x > 0`, "
            "is applied to the tensor elementwise.")
        .Input("input", "input tensor", "T")
        .Output("output", "The selu value of the input tensor computed element-wise", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Coefficient of SELU default to 1.6732.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.6732))
        .Attr("gamma", "Coefficient of SELU default to 1.0507.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0507));

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Softmax)
        .Description("The operator computes the softmax normalized values for each layer in the batch "
            "of the given input. The input is a 2-D tensor (Tensor<float>) of size "
            "(batch_size x input_feature_dimensions). The output tensor has the same shape "
            "and contains the softmax normalized values of the corresponding input. "
            "                                                                            "
            "X does not need to explicitly be a 2D vector; rather, it will be "
            "coerced into one. For an arbitrary n-dimensional tensor "
            "X in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is "
            "the axis provided, then X will be coerced into a 2-dimensional tensor with "
            "dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default "
            "case where axis=1, this means the X tensor will be coerced into a 2D tensor "
            "of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size. "
            "In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D. "
            "Each of these dimensions must be matched correctly, or else the operator "
            "will throw errors.")
        .Input("input","The input tensor that's coerced into a 2D matrix of size (NxD) "
            "as described above.", "T")
        .Output("output", "The softmax normalized output values with the same "
            "shape as input tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axis", "(int) default to 1; describes the axis of the inputs when coerced "
            "to 2D; defaults to one because the 0th axis most likely describes "
            "the batch_size", AttrType::AttributeProto_AttributeType_INT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(Affine)
        .Description("Affine takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the affine function, f(x)= alpha * x + beta is "
            "applied to the tensor elementwise.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Scalar multiplication factor", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("beta", "Scalar offset", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(HardSigmoid)
        .Description("HardSigmoid takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the hard sigmoid function, f(x) = max⁡(0,min⁡(alpha*x+beta,1)), "
            "is applied to the  tensor elementwise.")
        .Input("X", "Input tensor of any shape", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Scaling value", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("beta", "Scalar offset", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(ScaledTanh)
        .Description("ScaledTanh takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the scaled hyperbolic tangent function, "
            "f(x) = alpha*tanh⁡(beta*x), is applied to the  tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Scaling value", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("beta", "Scaling value", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(ThresholdedRelu)
        .Description("Thresholded Relu takes input data (Tensor<T>) and threshold as input, and "
            "produces one output data (Tensor<T>) where the function `f(x) = 0 for x < alpha, "
            "x for x >= alpha`, is applied to the data tensor elementwise.")
        .Input("X", "Input tensor, typically 1-D.", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Scalar threshold value", AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(LogSoftmax)
        .Description("Log Softmax takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = log(1 / sum(exp(X)) * exp(x)), is applied "
            "to the tensor elementwise.")
        .Input("input", "The input tensor that's coerced into a 2D matrix of size (NxD) as "
            "described above.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axis", "(int) default to 1; describes the axis of the inputs when coerced "
            "to 2D; defaults to one because the 0th axis most likely describes "
            "the batch_size", AttrType::AttributeProto_AttributeType_INT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(Hardmax)
        .Description("Compute the hardmax normalized values for each layer in the batch "
            "of the given input. The input is a 2-D tensor (Tensor<float>) of size "
            "(batch_size x input_feature_dimensions). The output tensor has the same shape "
            "and contains the softmax normalized values of the corresponding input. "
            "\n"
            "X does not need to explicitly be a 2D vector; rather, it will be coerced into "
            "one. For an arbitrary n-dimensional tensor X in [a_0, a_1, ..., a_{k-1}, "
            "a_k, ..., a_{n-1}] and k is the axis provided, then X will be coerced into a "
            "2-dimensional tensor with dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. "
            "For the default case where axis=1, this means the X tensor will be coerced into "
            "a 2D tensor of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the "
            "batch size.  In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D. "
            "Each of these dimensions must be matched correctly, or else the operator will "
            "throw errors.")
        .Input("input", "The input tensor that's coerced into a 2D matrix of size (NxD) as "
            "described above.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axis", "Default to 1; describes the axis of the inputs when coerced to 2D; "
            "defaults to one because the 0th axis most likely describes the batch size.",
            AttrType::AttributeProto_AttributeType_INT, int64_t(1));

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(Softsign)
        .Description("Softsign takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = x / (1 + abs(x)), is applied to the "
            "tensor elementwise.")
        .Input("input", "Input tensor, typically 1-D.", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(Softplus)
        .Description("Softplus takes one input data (Tensor<T>) and produces one output "
            "data (Tensor<T>) where the function, y = ln(1 + exp(steepness * x)), is "
            "applied to the tensor elementwise.")
        .Input("X", "Input tensor, typically 1-D.", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(ParametericSoftplus)
        .Description("Softplus takes input data (Tensor<T>) and parametric tensors, "
            "producing one output data (Tensor<T>) where the function, "
            "y = alpha * log(1 + exp(beta * x), is applied to the tensor elementwise.")
        .Input("X", "Input tensor, typically 1-D.", "T")
        .Output("Y", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("alpha", "Alpha tensor. If `alpha` is of size 1, "
            "the value is shared across different channels.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0))
        .Attr("beta", "Beta tensor. If `beta` is of size 1, "
            "the value is shared across different channels.",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0));

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(Identity)
        .Description("Identity takes one input data (Tensor<T>) and produces one "
            "output data (Tensor<T>) where the function, y = x, is applied to the "
            "tensor elementwise.")
        .Input("input", "input tensor", "T")
        .Output("output", "output tensor", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

}
