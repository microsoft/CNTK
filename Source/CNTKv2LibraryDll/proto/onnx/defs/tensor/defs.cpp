#include "proto/onnx/core/op.h"

namespace ONNXIR {
    // Taken fron ONNX
    REGISTER_OPERATOR_SCHEMA(Cast)
        .Description("The operator casts the elements of a given input tensor to a data type "
            "specified by the 'to' argument and returns an output tensor of the same size in "
            "the converted type. The 'to' argument must be one of the data types specified "
            "in the 'DataType' enum field in the TensorProto message. If the 'to' argument "
            "is not provided or is not one of the enumerated types in DataType, Caffe2 "
            "throws an Enforce error. "
            "NOTE: Casting to and from strings is not supported yet.")
        .Input("input", "Input tensor to be cast.", "T")
        .Output(
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast."
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttrType::AttributeProto_AttributeType_STRING);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Flatten)
        .Description("Flattens the input tensor into a 2D matrix, "
            "keeping the first dimension unchanged.")
        .Input("input", "A tensor of rank >= 2.", "T")
        .Output("output", "A tensor of rank 2 with the contents of the input tensor, "
            "with first dimension equal first dimension of input, and remaining "
            "input dimensions flatenned into the inner dimension of the output.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Reshape)
        .Description("Reshape the input tensor similar to numpy.reshape. "
            "                                                                                    "
            "It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor. "
            "                                                                             "
            "At most one dimension of the new shape can be -1. In this case, the value is "
            "inferred from the size of the tensor and the remaining dimensions. A dimensions "
            "could also be 0, in which case the actual dimension value is going to be copied "
            "from the shape argument.")
        .Input("data", "An input tensor.", "T")
        .Output("reshaped", "Reshaped data.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("shape", "Tensor of shape declarations for the output. Must be compatible with "
            "the input. At most one dimension of the new shape can be -1. In this case, the "
            "value is inferred from the size of the tensor and the remaining dimensions. A "
            "dimension could also be 0, in which case the actual dimension value is going to "
            "be copied from the input tensor.", AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Split)
        .Description("Split a tensor into a list of tensors, along the specified 'axis'. "
            "The lengths of the split can be specified using argument 'axis' or "
            "optional second input blob to the operator. Otherwise, the tensor is split "
            "to equal sized parts.")
        .Input("input", "The tensor to split", "T")
        .Input("split", "Optional list of output lengths (see also arg 'split')", "T")
        .Output("output", "A list of output tensors", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axis", "Which axis to split on", AttrType::AttributeProto_AttributeType_INT)
        .Attr("split", "Number of tensors to output.", AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Transpose)
        .Description("Transpose the input tensor similar to numpy.transpose. For example, "
            "when axes=(1, 0, 2), given an input tensor of shape (1, 2, 3), the output shape "
            "will be (2, 1, 3).")
        .Input("data", "An input tensor.", "T")
        .Output("transposed", "Transposed output.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("perm", "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given.", AttrType::AttributeProto_AttributeType_INTS);

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(Tile)
        .Description("Repeat the elements of a tensor along an axis.")
        .Input("input", "An input tensor.", "T")
        .Input("tiles", "Number of repeated copies to make of the input tensor.", "T")
        .Input("axis", "Axis along which to repeat.", "T")
        .Output("output", "Repeated output.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Concat)
        .Description("Concatenate takes as input a list of tensors, all of the same shape"
            "expect for the concatenation axis, and returns a single tensor, the concatenation"
            "of all inputs.")
        .Input("input", "A list of input tensors.", "T")
        .Output("concat_result", "Concatenated tensor", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axis", "Axis along which to concatenate", AttrType::AttributeProto_AttributeType_INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Slice)
        .Description("Produces a slice of the input tensor along multiple axes. Similar to "
            "numpy: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html "
            "                                                                              "
            "Slices are passed as two keyword argument lists with starting and end indices "
            "for each dimension of the input `data` tensor. If a negative value is passed "
            "for any of the start or end indices, it represent number of elements before "
            "the end of that dimension. "
            "                                                                            "
            "`strides` is the  step sizes when applying slicing, negative value means in "
            "reverse order.")
        .Input("input", "Tensor of data to extract slices from.", "T")
        .Output("output", "Sliced data tensor.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("starts", "List of starting indices", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("ends", "List of ending indices", AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Gather)
        .Description("Given data tensor of rank r >= 1, and indices tensor of rank q, gather "
            "entries of the outer-most dimension of data indexed by indices, and concatenate "
            "them in an output tensor of rank q + (r - 1). "
            "Example 1: data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ] "
            "           indices = [ [0, 1], [1, 2], ] "
            "           output = [ [ [1.0, 1.2], [2.3, 3.4], ], [ [2.3, 3.4], [4.5, 5.7], ], ]"
            "Example 2: data = [ [1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9], ] "
            "           indices = [0, 2], ] axis = 1, "
            "           output = [ [ [1.0, 1.9], [2.3, 3.9], [4.5, 5.9], ], ]")
        .Input("data", "Tensor of rank r >= 1.", "T")
        .Input("indices", "Tensor of int32/int64 indices, of any rank q.", "Tind")
        .Output("output", "Tensor of rank q + (r - 1).", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input types to float tensors.")
        .TypeConstraint("Tind", { "tensor(int32)", "tensor(int64)" },
            "Constrain indices types to float tensors.")
        .Attr("axis",
            "Which axis to gather on, defaults to 0. Negative value means counting dimensions "
            "from the back. Accepted range in [-r, r-1]",
            AttrType::AttributeProto_AttributeType_INT, int64_t(0));

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Squeeze)
        .Description("Remove single-dimensional entries from the shape of a tensor. "
            "Takes a  parameter `axes` with a list of axes to squeeze.")
        .Input("data", "Tensors with at least max(dims) dimensions.", "T")
        .Output("squeezed", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("axes",
            "List of positive integers, indicate the dimensions to squeeze.",
            AttrType::AttributeProto_AttributeType_INTS, int64_t(1));

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(DepthToSpace)
        .Description("DepthToSpace for 4-D tensors of type T. "
            "Rearranges (permutes) data from channel into blocks of spatial data, "
            "followed by cropping. This is the reverse transformation of "
            "SpaceToDepth. More specifically, this op outputs a copy of the input "
            "tensor where values from the channel dimension are moved in spatial "
            "blocks to the height and width dimensions, followed by cropping along "
            "the height and width dimensions.")
        .Input("input", "Input tensor of [N,C,H,W]", "T")
        .Output("output", "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, "
            "W * blocksize]", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::AttributeProto_AttributeType_INT);

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(SpaceToDepth)
        .Description("SpaceToDepth for 4-D tensors of type T. "
            "Zero-pads and then rearranges (permutes) blocks of spatial data into "
            "channel. More specifically, this op outputs a copy of the input tensor "
            "where values from the height and width dimensions are moved to the "
            "channel dimension. After the zero-padding, both height and width of the "
            "input must be divisible by the block size.")
        .Input("input", "Input tensor of [N,C,H,W]", "T")
        .Output("output", "Output tensor of [N, C * blocksize * blocksize, H/blocksize, "
            "W/blocksize]", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("blocksize", "Blocks of [blocksize,blocksize] are moved.", AttrType::AttributeProto_AttributeType_INT);

}
