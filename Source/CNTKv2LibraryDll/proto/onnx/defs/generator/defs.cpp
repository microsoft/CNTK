#include "proto/onnx/core/op.h"

namespace ONNXIR
{
    REGISTER_OPERATOR_SCHEMA(Constant)
        .Description("A constant tensor.")
        .Attr(
            "value",
            "The value for the elements of the output tensor.",
            AttrType::AttributeProto_AttributeType_TENSOR)
        .Output("output",
                "Output tensor containing the same value of the provided tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");
}
