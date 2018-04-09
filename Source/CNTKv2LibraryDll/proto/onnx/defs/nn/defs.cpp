#include "proto/onnx/core/op.h"
#include "proto/onnx/core/constants.h"

namespace ONNXIR {
    REGISTER_OPERATOR_SCHEMA(FC)
        .Description("Computes the result of passing an input vector X into a fully"
            "connected layer with 2D weight matrix W and 1D bias vector b.That is, "
            "the layer computes Y = X * W^T + b, where X has size(M x K), "
            "W has size(N x K), b has size(N), and Y has size(M x N), "
            "where M is often the batch size.")
        .Input("X", "input tensor that's coerced into a 2D matrix of size (MxK) ", "T")
        .Input("W", "A tensor that is coerced into a 2D blob of size (KxN) containing fully connected weight matrix", "T")
        .Input("B", "1D blob containing bias vector", "T")
        .Output("Y", "output tensor", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.")
        .Attr("axis",
            "(int32_t) default to 1; describes the axis of the inputs; "
            "defaults to one because the 0th axis most likely describes the batch_size",
            AttrType::AttributeProto_AttributeType_INT, int64_t(1))
        .Attr("axis_w",
            "(int32_t) default to 1; describes the axis of the weight matrix W; "
            "defaults to one because the 0th axis most likely describes the batch_size",
            AttrType::AttributeProto_AttributeType_INT, int64_t(1));

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Conv)
        .Description("The convolution operator consumes an input tensor and a filter, and"
            "computes the output.")
        .Input("X",
             "Input data tensor from previous layer; has size (N x C x H x W)"
             ", where N is the batch size, C is the number of channels, and"
             " H and W are the height and width. Note that this is for the 2D image."
             "Otherwise the size is (N x D1 x D2 ... x Dn)",
             "T")
        .Input("W",
             "The weight tensor that will be used in the convolutions; has size (M x C x kH x kW), "
             "where C is the number of channels, and kH and kW are the height and width of the kernel, "
             "and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be "
             "(M x C x k1 x k2 x ... x kn), where is the dimension of the kernel",
             "T")
        .Input("B",
            "Optional 1D bias to be added to the convolution, has size of M.",
            "T")
        .Output("Y",
              "Output data tensor that contains the result of the convolution. The "
              "output dimensions are functions of the kernel size, stride size, "
              "and pad lengths.",
              "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("auto_pad",
            "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER "
            "or SAME_LOWER mean pad the input so that the ouput size match the input. "
            "In case of odd number add the extra padding at the end for SAME_UPPER and "
            "at the begining for SAME_LOWER. VALID mean no padding, therefore, read the "
            "pixel values from the pads attribute.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_shape",
            "The shape of the convolution kernel.",
             AttrType::AttributeProto_AttributeType_INTS)
        .Attr("dilations",
            "dilation value along each axis of the filter.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("strides",
            "stride along each axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("pads",
            "Padding for lower and upper side along each axis, it can take any value greater "
            "than or equal to 0. The value represent the number of pixels added to the lower "
            "and upper part of the corresponding axis. So `pads` will have two values per axis, "
            "first value corresponding to the number of pixels added to the begining of the "
            "axis and the second value corresponding to the number of pixels add at the end "
            "of the axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("group",
            "number of groups input channels and output channels are divided into",
            AttrType::AttributeProto_AttributeType_INT);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(ConvTranspose)
        .Description("The convolution transpose operator consumes an input tensor and a filter,"
            "and computes the output.")
        .Input("X",
             "Input data tensor from previous layer; has size (N x C x H x W)"
             ", where N is the batch size, C is the number of channels, and"
             " H and W are the height and width. Note that this is for the 2D image."
             "Otherwise the size is (N x D1 x D2 ... x Dn)",
             "T")
        .Input("W",
             "The weight tensor that will be used in the convolutions; has size (C x M x kH x kW), "
             "where C is the number of channels, and kH and kW are the height and width of the kernel, "
             "and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be "
             "(M x C x k1 x k2 x ... x kn), where is the dimension of the kernel",
             "T")
        .Input("B",
            "Optional 1D bias to be added to the convolution, has size of C.",
            "T")
        .Output("Y",
              "Output data tensor that contains the result of the convolution. The "
              "output dimensions are functions of the kernel size, stride size, "
              "and pad lengths.",
              "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("auto_pad",
            "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER "
            "or SAME_LOWER mean pad the input so that the ouput size match the input. "
            "In case of odd number add the extra padding at the end for SAME_UPPER and "
            "at the begining for SAME_LOWER. VALID mean no padding, therefore, read the "
            "pixel values from the pads attribute.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_shape",
            "The shape of the convolution kernel.",
             AttrType::AttributeProto_AttributeType_INTS)
        .Attr("output_shape",
            "The shape of the output.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("dilations",
            "dilation value along each axis of the filter.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("strides",
            "stride along each axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("pads",
            "Padding for lower and upper side along each axis, it can take any value greater "
            "than or equal to 0. The value represent the number of pixels added to the lower "
            "and upper part of the corresponding axis. So `pads` will have two values per axis, "
            "first value corresponding to the number of pixels added to the begining of the "
            "axis and the second value corresponding to the number of pixels add at the end "
            "of the axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("group",
            "number of groups input channels and output channels are divided into",
            AttrType::AttributeProto_AttributeType_INT);

    REGISTER_OPERATOR_SCHEMA(Dropout)
        .Description("Dropout takes one input data (Tensor<float>) and produces two Tensor outputs, "
            "output (Tensor<float>) and mask (Tensor<bool>). Depending on whether it is in "
            "test mode or not, the output Y will either be a random dropout, or a simple "
            "copy of the input. Note that our implementation of Dropout does scaling in "
            "the training phase, so during testing nothing needs to be done.")
        .Input("data", "The input data as Tensor.", "T")
        .Output("output", "The output.", "T")
        .Output("mask",
            "The output mask. If is_test is nonzero, this output is not filled.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("ratio",
            "(float, default 0.5) the ratio of random dropout",
            AttrType::AttributeProto_AttributeType_FLOAT, float(0.5))
        .Attr("is_test",
            "(int, default 0) if nonzero, run dropout in test mode where "
            "the output is simply Y = X.",
            AttrType::AttributeProto_AttributeType_INT, int64_t(0));

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(AveragePool)
        .Description("AveragePool consumes an input tensor X and applies average pooling across the"
            "the tensor according to kernel sizes, stride sizes, and pad lengths."
            "Average pooling consisting of averaging all values of a subset of the"
            "input tensor according to the kernel size and downsampling the"
            "data into the output tensor Y for further processing.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from average pooling across the input tensor. "
            "Dimensions will vary based on various kernel, stride, and pad sizes.")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("auto_pad",
            "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER "
            "or SAME_LOWER mean pad the input so that the ouput size match the input. "
            "In case of odd number add the extra padding at the end for SAME_UPPER and "
            "at the begining for SAME_LOWER. VALID mean no padding, therefore, read the "
            "pixel values from the pads attribute.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_shape",
            "The size of the kernel along each axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("strides",
            "Stride along each axis.",
            AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(GlobalAveragePool)
        .Description("GlobalAveragePool consumes an input tensor X and applies average "
            "pooling across the values in the same channel. This is equivalent to "
            "AveragePool with kernel size equal to the spatial dimension of input tensor.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from pooling across the input tensor. Dimensions will "
            "be N x C x 1 x 1")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(MaxPool)
        .Description("MaxPool consumes an input tensor X and applies max pooling across the"
            "the tensor according to kernel sizes, stride sizes, and pad lengths."
            "Average pooling consisting of averaging all values of a subset of the"
            "input tensor according to the kernel size and downsampling the"
            "data into the output tensor Y for further processing.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from max pooling across the input tensor. "
            "Dimensions will vary based on various kernel, stride, and pad sizes.",
            "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("auto_pad",
            "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER "
            "or SAME_LOWER mean pad the input so that the ouput size match the input. "
            "In case of odd number add the extra padding at the end for SAME_UPPER and "
            "at the begining for SAME_LOWER. VALID mean no padding, therefore, read the "
            "pixel values from the pads attribute.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_shape",
            "The size of the kernel along each axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("strides",
            "Stride along each axis.",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("pads",
            "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("dilations",
            "Dilaton along each axis, 1 mean no dilation.",
            AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(GlobalMaxPool)
        .Description("GlobalMaxPool consumes an input tensor X and applies max pooling "
            "across the values in the same channel. This is equivalent to MaxPool "
            "with kernel size equal to the spatial dimension of input tensor.")
        .Input("X",
            "Input data tensor from the previous operator; dimensions for image case "
            "are (N x C x H x W), where N is the batch size, C is the number of channels, "
            "and H and W are the height and the width of the data. For non image case, the "
            "dimension are in the form of (N x D1 x D2 ... Dn), where N is the batch size.",
            "T")
        .Output("Y",
            "Output data tensor from pooling across the input tensor. Dimensions will "
            "be N x C x 1 x 1")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.");

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(BatchNormalization)
        .Description("Carries out batch normalization as described in the paper"
            "https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,"
            "there are multiple cases for the number of outputs, which we list below:"
            ""
            "Output case #1: Y, mean, var, saved_mean, saved_var (training mode)"
            "Output case #2: Y (test mode)")
        .Input("X",
            "The input 4-dimensional tensor of shape NCHW or NHWC depending "
            "on the order parameter.",
            "T")
        .Input("scale",
            "The scale as a 1-dimensional tensor of size C to be applied to the "
            "output.",
            "T")
        .Input("B",
            "The bias as a 1-dimensional tensor of size C to be applied to the "
            "output.",
            "T")
        .Input("mean",
            "The running mean (training) or the estimated mean (testing) "
            "as a 1-dimensional tensor of size C.",
            "T")
        .Input("var",
            "The running variance (training) or the estimated "
            "variance (testing) as a 1-dimensional tensor of size C.",
            "T")
        .Output("Y", "The output 4-dimensional tensor of the same shape as X.",
            "T")
        .Output("mean",
            "The running mean after the BatchNormalization operator. Must be in-place "
            "with the input mean. Should not be used for testing.",
            "T")
        .Output("var",
            "The running variance after the BatchNormalization operator. Must be "
            "in-place with the input var. Should not be used for testing.",
            "T")
        .Output("saved_mean",
            "Saved mean used during training to speed up gradient "
            "computation. Should not be used for testing.",
            "T")
        .Output("saved_var",
            "Saved variance used during training to speed up "
            "gradient computation. Should not be used for testing.",
            "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("is_test",
            "If set to nonzero, run spatial batch normalization in test mode.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr("momentum",
            "Factor used in computing the running mean and variance."
            "e.g., running_mean = running_mean * momentum + mean * (1 - momentum)",
            AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("spatial",
            "Compute the mean and variance across all spatial elements or per feature.",
            AttrType::AttributeProto_AttributeType_INT);

    REGISTER_OPERATOR_SCHEMA(InstanceNormalization)
        .Description("Carries out instance normalization as described in the paper "
            "https://arxiv.org/abs/1607.08022. "
            "y = scale * (x - mean) / sqrt(variance + epsilon) + B, "
            "where mean and B are computed per instance per channel.")
        .Input("input",
            "The input 4-dimensional tensor of shape NCHW.", "T")
        .Input("scale",
            "The input 1-dimensional scale tensor of size C.", "T")
        .Input("B",
            "The input 1-dimensional bias tensor of size C.", "T")
        .Output("output",
            "The output 4-dimensional tensor of the same shape as input.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("epsilon",
            "The epsilon value to use to avoid division by zero.",
            AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from Caffe2
    REGISTER_OPERATOR_SCHEMA(MaxRoiPool)
        .Description("ROI max pool consumes an input tensor X and region of interests (RoIs) to "
            "apply max pooling across each RoI, to produce output 4-D tensor of shape "
            "(num_rois, channels, pooled_shape[0], pooled_shape[1]).")
        .Input("X", "The input 4-D tensor of data. Only NCHW order is currently supported.", "T")
        .Input("rois", "RoIs (Regions of Interest) to pool over. Should be a 2-D tensor of "
            "shape (num_rois, 5) given as [[batch_id, x1, y1, x2, y2], ...].", "T")
        .Output("Y", "RoI pooled output 4-D tensor of shape "
            "(num_rois, channels, pooled_h, pooled_w).", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("pooled_shape", "ROI pool output shape (height, width).",
            AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("spatial_scale", "Multiplicative spatial scale factor to translate ROI "
            "coordinates from their input scale to the scale used when pooling (Default: 1.0).",
            AttrType::AttributeProto_AttributeType_FLOAT, float(1.0));

    REGISTER_OPERATOR_SCHEMA(LpPool)
        .Description("LpPool consumes an input blob X and applies L-p pooling across the "
            "blob according to kernel sizes, stride sizes, and pad lengths defined by the "
            "ConvPoolOpBase operator. L-p pooling consisting of taking the L-p norm of a "
            "subset of the input tensor according to the kernel size and downsampling the "
            "data into the output blob Y for further processing.")
        .Input("X", "X Input data tensor from the previous operator; dimensions depend on "
            "whether the NCHW or NHWC operators are being used. For example, in the former, "
            "the input has size (N x C x H x W), where N is the batch size, C is the number "
            "of channels, and H and W are the height and the width of the data. The "
            "corresponding permutation of dimensions is used in the latter case.", "T")
        .Output("Y", "Y Output data tensor from L-p pooling across the input tensor. "
            "Dimensions will vary based on various kernel, stride, and pad sizes.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("auto_pad",
            "auto_pad must be either SAME_UPPER, SAME_LOWER or VALID. Where SAME_UPPER "
            "or SAME_LOWER mean pad the input so that the ouput size match the input. "
            "In case of odd number add the extra padding at the end for SAME_UPPER and "
            "at the begining for SAME_LOWER. VALID mean no padding, therefore, read the "
            "pixel values from the pads attribute.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("kernel_shape", "The size of the kernel along each axis.", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("strides", "Stride along each axis.", AttrType::AttributeProto_AttributeType_INTS)
        .Attr("pads", "Padding along each axis, can take the value 0 (False) or non 0 (True)",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("p", "Value of p, default 2.", AttrType::AttributeProto_AttributeType_INT, int64_t(2));

    REGISTER_OPERATOR_SCHEMA(GlobalLpPool)
        .Description("GlobalLpPool consumes an input tensor X and applies lp-pool across the "
            "values in the same channel. This is equivalent to LpPool with kernel size equal "
            "to the spatial dimension of input tensor.")
        .Input("X", "X Input data tensor from the previous operator; dimensions depend on "
            "whether the NCHW or NHWC operators are being used. For example, in the former, "
            "the input has size (N x C x H x W), where N is the batch size, C is the number "
            "of channels, and H and W are the height and the width of the data. The "
            "corresponding permutation of dimensions is used in the latter case.", "T")
        .Output("Y", "Y Output data tensor from L-p pooling across the input tensor. Dimensions will "
            "vary based on various kernel, stride, and pad sizes.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output types to float tensors.")
        .Attr("p", "Value of p, default 2.", AttrType::AttributeProto_AttributeType_INT, int64_t(2));


    std::function<void(OperatorSchemaSetter&)> LRNDocGenerator() {
        return [=](OperatorSchemaSetter& schema) {
            schema.Description("Perform local response normalization. "
                "NOTE: Only supports Caffe across channel mode. ");
            schema.Input("X", "Input tensor of any shape", "T");
            schema.Output("Y", "Output tensor of same shape and type as input X.", "T");
            schema.TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain input and output types to float tensors.");
            schema.Attr("size", "[default 5]: the number of channels to sum over (for cross "
                  "channel LRN) or the side length of the square region to sum over (for within "
                  "channel LRN)", AttrType::AttributeProto_AttributeType_INT, int64_t(5));
            schema.Attr("alpha", "Scalar scaling factor. Default is 0.0001",
                AttrType::AttributeProto_AttributeType_FLOAT, float(0.0001));
            schema.Attr("beta", "Scalar exponent in the LRN.  Default is 0.5.",
                AttrType::AttributeProto_AttributeType_FLOAT, float(0.5));
            schema.Attr("bias", "An offset (must be positive to avoid dividing by 0). Defaults to 1.0.",
                AttrType::AttributeProto_AttributeType_FLOAT, float(1.0));
        };
    }

    REGISTER_OPERATOR_SCHEMA(LocalResponseNormalization)
        .FillUsing(LRNDocGenerator());

    // TODO: to be duplicated.
    REGISTER_OPERATOR_SCHEMA(LRN)
        .FillUsing(LRNDocGenerator());

    REGISTER_OPERATOR_SCHEMA(MeanVarianceNormalization)
        .Description("Perform mean variance normalization.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and output "
            "types to float tensors.")
        .Attr("across_channels", "If true, mean and variance are computed across channels. "
            "Default is false.", AttrType::AttributeProto_AttributeType_INT, int64_t(0))
        .Attr("normalize_variance", "If false, normalize the mean only. Default is true.",
            AttrType::AttributeProto_AttributeType_INT, int64_t(1));

    REGISTER_OPERATOR_SCHEMA(LpNormalization)
        .Description("Given a matrix, apply Lp-normalization along the provided axis. "
            "For RS4 default of p = 2 and it will perform L2 normalization. Divide each "
            "element by the square root of the sum of squares of all elements in the input tensor.")
        .Input("input", "Input tensor of any shape", "T")
        .Output("output", "Output tensor of same shape and type as input X.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(float)" }, "Constrain input and output "
            "types to float tensors.")
        .Attr("axis", "Axis along which to perform normalization.",
            AttrType::AttributeProto_AttributeType_INT)
        .Attr("p", "(int64, default 2) the order of the normalization, only 1 or 2 are supported.",
            AttrType::AttributeProto_AttributeType_INT);

    // Take from RS4
    REGISTER_OPERATOR_SCHEMA(Embedding)
        .Description("Turns positive integers (indexes) into dense vectors of fixed size. "
            "eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]] "
            "TODO: Omits use of CoreML bias parameter.")
        .Input("input", "1-D tensor of integers representing indices in the embedding "
            "dictionary with length [N] and values [0, input_dim -1]", "T1")
        .Input("W", "2-D tensor of weights [O, I]", "T2")
        .Output("output", "Output tensor of computed features [N, O].", "T2")
        .TypeConstraint("T1", { "tensor(uint64)" }, "Constrain input types to ints.")
        .TypeConstraint("T2", { "tensor(float16)", "tensor(float)", "tensor(double)" },
                "Constrain output types to float tensors.")
        .Attr("input_dim", "Size of the input vocabulary.", AttrType::AttributeProto_AttributeType_INT)
        .Attr("output_dim", "Dimension of the embedding output vectors.", AttrType::AttributeProto_AttributeType_INT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(ImageScaler)
        .Description("Alteration of image by scaling its individual values. "
            "NOTE: The current definition assumes that the bias values are stored in the "
            "same ordering as the image pixel format.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and "
            "output types to float tensors.")
        .Attr("bias", "Bias values for each channel, of shape [C]", AttrType::AttributeProto_AttributeType_FLOATS)
        .Attr("scale", "Scalar channel factor, elementwise mutliplied into every value in [C,H,W]",
            AttrType::AttributeProto_AttributeType_FLOAT);

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(Upsample)
        .Description("Scale up spatial dimensions.  Use interpolation to fill in values")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and "
            "output types to float tensors.")
        .Attr("mode", "enum {'NEAREST', 'BILINEAR' }, Nearest neighbor or bilinear upsampling.",
            AttrType::AttributeProto_AttributeType_STRING)
        .Attr("width_scale", "Scale along width dimension", AttrType::AttributeProto_AttributeType_FLOAT)
        .Attr("height_scale", "Scale along height dimension", AttrType::AttributeProto_AttributeType_FLOAT);

    REGISTER_OPERATOR_SCHEMA(Crop)
        .Description("Crop and image to the specified spatial dimensions.  If scale is given,"
            "then optionally start the crop offset by the left/top border amounts.  "
            "If scale is not provided, crop the borders as provided.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same type as X, with H and W dimensions reduced.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" }, "Constrain input and "
            "output types to float tensors.")
        .Attr("border", "A 1-D tensor of values (leftBorder, topBorder, rightBorder, bottomBorder)",
            AttrType::AttributeProto_AttributeType_INTS)
        .Attr("scale", "A 1-D tensor of values (height, width)", AttrType::AttributeProto_AttributeType_INTS);

    // Taken from ONNX
    REGISTER_OPERATOR_SCHEMA(Pad)
        .Description("Given data tensor, paddings, mode, and value. "
            "Example: Insert 0 paddings to the beginning of the second dimension. "
            "data = [ [1.0, 1.2], [2.3, 3.4], [4.5, 5.7], ] paddings = [0, 0, 2, 0] "
            "output = [ [ [0.0, 0.0, 1.0, 1.2], [0.0, 0.0, 2.3, 3.4], [0.0, 0.0, 4.5, 5.7] ] ]")
        .Input("data", "Input tensor.", "T")
        .Output("output", "Tensor after padding.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("pads",
              "List of integers indicate the padding sizes, paddings's length "
              "should be the double of input's dimension. "
              "The order should be axis_0_begin, axis_0_end, axis_1_begin, ..., "
              "axis_n_begin, axis_n_end, n is input's dimension.",
              AttrType::AttributeProto_AttributeType_INTS, int64_t(1))
        .Attr("mode",
              "Three modes: constant(default), reflect, edge",
              AttrType::AttributeProto_AttributeType_STRING, std::string("constant"))
        .Attr("value",
              "One float, indicates the value to be filled, default is 0",
              AttrType::AttributeProto_AttributeType_FLOAT, float(0));

    // Taken from RS4
    REGISTER_OPERATOR_SCHEMA(MeanSubtraction)
        .Description("Subtracts the provided mean image from the input image.")
        .Input("input", "Input tensor of shape [N,C,H,W]", "T")
        .Output("output", "Result, has same shape and type as X", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("image", "Image tensor stored as a sequence of floats [C,H,W].", AttrType::AttributeProto_AttributeType_TENSOR);

    REGISTER_OPERATOR_SCHEMA(LegacyPadding)
        .SetDomain(c_msDomain)
        .Description("his operator is designed to support CoreML's pooling operator under IncludeLastPixel padding mode.. "
            "To simulate CoreML's pooling operator, First, copy kernel shape, strides, padding "
            "amounts from the original pooling operator to this LegacyPadding operator. "
            "Second, create a pooling operator under auto_pad=VALID with the kernel and strides used in the original pooling. "
            "Third, connect the output of LegacyPadding operator with the pooling operator we just create. ")
        .Input("data", "Input tensor.", "T")
        .Output("output", "Tensor after padding.", "T")
        .TypeConstraint("T", { "tensor(float16)", "tensor(float)", "tensor(double)" },
            "Constrain input and output types to float tensors.")
        .Attr("pads",
              "Padding amounts along H- and W-axes, [pad_h, pad_w]. ",
              AttrType::AttributeProto_AttributeType_INTS, int64_t(1))
        .Attr("kernel_shape",
              "The size of the kernel along H- and W-axes, [k_h, k_w]. Notice that the kernel is a 2-D tensor. ",
              AttrType::AttributeProto_AttributeType_INTS, int64_t(1))
        .Attr("strides",
              "Stride along H- and W-axes, [stride_h, stride_w].",
              AttrType::AttributeProto_AttributeType_INTS, int64_t(1))
        .Attr("value",
              "One float, indicates the value to be filled, default is 0",
              AttrType::AttributeProto_AttributeType_FLOAT, float(0));
}

