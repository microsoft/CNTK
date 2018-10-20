//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "core/graph/graph.h"

#include "Operators.h"
#include "Utils.h"

namespace CNTK
{
namespace ONNX
{
    //
    // Support ONNX OPs from https://github.com/onnx/onnx/tree/master/onnx/defs
    //
    // The format of the below structure is simply a key which is CNTK OpName and a corresponding
    // lookup table, the  corrsponding lookup table map the OpName and all its attributes from CNTK
    // to ONNX.
    //
    // Eventually, it would be good to change CNTK OpName to match ONNX in order to avoid the need 
    // of the below table.
    //
    std::unordered_multimap<std::wstring, AttributesMapping> Operators::_cntkToONNXOpName = {
        // From nn
        { L"Pooling", { {
            { L"Pooling", "AveragePool" },
            { L"poolingWindowShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
        } } },
        { L"Pooling",  { {
            { L"Pooling",  "MaxPool" },
            { L"poolingWindowShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
        } } },
        { L"ROIPooling",{ {
            { L"ROIPooling",  "MaxRoiPool" },
            // { L"poolingType",  "" }, // always Max
            { L"roiOutputShape", "pooled_shape" },
            { L"spatialScale", "spatial_scale" },
            } } },
        { L"Convolution", { {
            { L"Convolution", "Conv" },
            { L"kernelShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
            { L"dilation", "dilations" },
            // { L"", "group" },
        } } },
        { L"Convolution", { {
            { L"ConvolutionTranspose", "ConvTranspose" },
            { L"kernelShape", "kernel_shape" },
            { L"strides", "strides" },
            { L"autoPadding", "pads" },
            { L"dilation", "dilations" },
            { L"outputShape", "output_shape" },
        } } },
        { L"GlobalMaxPooling", { {
            { L"GlobalMaxPooling", "GlobalMaxPool" },
        } } },
        { L"GlobalAveragePooling", { {
            { L"GlobalAveragePooling", "GlobalAveragePool" },
        } } },
        { L"BatchNormalization", { {
            { L"BatchNormalization", "BatchNormalization" },
            { L"spatial", "spatial" },
            // { L"", "is_test" },
            { L"epsilon", "epsilon" },
            // { L"", "momentum" },
        } } },
        // from ONNX experiment, added to test Caffe models
        // TODO: set key as BatchNormalization instead of BatchNormalizationCaffe
        { L"BatchNormalizationCaffe",{ {
            { L"BatchNormalization", "SpatialBN" },
            { L"spatial", "spatial" },
            // { L"", "is_test" },
            { L"epsilon", "epsilon" },
            // { L"", "momentum" },
        } } },
        { L"OptimizedRNNStack",{ {
            { L"OptimizedRNNStack", "OptimizedRNNStack" },
            { L"hidden_size", "hidden_size" },
            { L"num_layers", "num_layers" },
            { L"bidirectional", "bidirectional" },
            { L"recurrent_op", "recurrent_op" },
        } } },
        { L"LayerNormalization",{ {
            { L"LayerNormalization", "LayerNormalization" },
            { L"initial_scale", "initial_scale" },
            { L"initial_bias", "initial_bias" },
            { L"epsilon", "epsilon" },
        } } },
        { L"LocalResponseNormalization",{ {
            { L"LocalResponseNormalization", "LRN" },
            { L"size", "size" },
            { L"bias", "bias" },
            { L"alpha", "alpha" },
            { L"beta", "beta" },
        } } },
        { L"Dropout", { {
            { L"Dropout", "Dropout" },
            { L"dropoutRate", "ratio" },
            // { L"", "is_test" },
        } } },
        { L"Reshape",{ {
            { L"Reshape", "Reshape" },
            { L"shape", "shape" },
            } } },
        { L"Flatten",{ {
            { L"Flatten", "Flatten" },
            { L"axis", "axis" },
        } } },
        // From Generator
        { L"RandomDistribution", { {
            { L"UniformRandom", "RandomUniform" },
            // { L"", "low" },
            // { L"", "high" },
            { L"rngSeed", "seed" },
            { L"newShape", "shape" },
        } } },
        { L"RandomDistribution", { {
            { L"NormalRandom", "RandomNormal" },
            // { L"", "mean" },
            // { L"", "scale" },
            { L"rngSeed", "seed" },
            { L"newShape", "shape" },
        } } },
        { L"RandomDistribution", { {
            { L"UniformRandomLike", "RandomUniformLike" },
            // { L"", "low" },
            // { L"", "high" },
            { L"rngSeed", "seed" },
        } } },
        { L"RandomDistribution", { {
            { L"NormalRandomLike", "RandomNormalLike" },
            // { L"", "mean" },
            // { L"", "scale" },
            { L"rngSeed", "seed" },
        } } },

        // From Math 
        { L"Plus", { {
            { L"Plus", "Add" },
        } } },
        { L"Minus", { {
            { L"Minus", "Sub" },
        } } },
        { L"ElementTimes", { {
            { L"ElementTimes", "Mul" },
        } } },
        { L"ElementDivide", { {
            { L"ElementDivide", "Div" },
        } } },
        { L"And",{ {
            { L"And", "And" },
            } } },
        { L"Not",{ {
            { L"Not", "Not" },
        } } },
        { L"Or",{ {
            { L"Or", "Or" },
        } } },
        { L"Xor",{ {
            { L"Xor", "Xor" },
        } } },
        { L"Negate", { {
            { L"Negate", "Neg" },
        } } },
        { L"Abs", { {
            { L"Abs", "Abs" },
        } } },
        { L"Mean",{ {
            { L"Mean", "Mean" },
        } } },
        { L"Sum",{ {
            { L"Sum", "Sum" },
        } } },
        { L"Reciprocal", { {
            { L"Reciprocal", "Reciprocal" },
        } } },
        { L"Floor", { {
            { L"Floor", "Floor" },
        } } },
        { L"Ceil", { {
            { L"Ceil", "Ceil" },
        } } },
        { L"Clip",{ {
            { L"Clip", "Clip" },
        } } },
        { L"Sqrt", { {
            { L"Sqrt", "Sqrt" },
        } } },
        { L"ReLU", { {
            { L"ReLU", "Relu" },
        } } },
        { L"LeakyReLU", { {
            { L"LeakyReLU", "LeakyRelu" },
            { L"alpha", "alpha" },
        } } },
        { L"SELU", { {
            { L"SELU", "Selu" },
            { L"alpha", "alpha" },
            { L"gamma", "gamma" },
        } } },
        { L"ELU", { {
            { L"ELU", "Elu" },
            { L"alpha", "alpha" },
        } } },
        { L"Exp", { {
            { L"Exp", "Exp" },
        } } },
        { L"Log", { {
            { L"Log", "Log" },
        } } },
        { L"Tanh", { {
            { L"Tanh", "Tanh" },
        } } },
        { L"Pow", { {
            { L"Pow", "Pow" },
            // { L"", "exponent" },
        } } },
        { L"Times", { {
            { L"Times", "MatMul" },
        } } },
        { L"PReLU", { {
            { L"PReLU", "PRelu" },
        } } },
        { L"StableSigmoid", { {
            { L"StableSigmoid", "Sigmoid" },
        } } },
        { L"Sigmoid", { {
            { L"Sigmoid", "Sigmoid" },
        } } },
        { L"ElementMax", { {
            { L"ElementMax", "Max" },
        } } },
        { L"ElementMin", { {
            { L"ElementMin", "Min" },
        } } },
        { L"HardSigmoid",{ {
            { L"HardSigmoid", "HardSigmoid" },
            { L"alpha", "alpha" },
            { L"beta", "beta" },
        } } },
        { L"Hardmax",{ {
            { L"Hardmax", "Hardmax" },
            { L"axis", "axis" },
        } } },
        { L"Softmax", { {
            { L"Softmax", "Softmax" },
            { L"axis", "axis" },
        } } },
        { L"LogSoftmax",{ {
            { L"LogSoftmax", "LogSoftmax" },
            { L"axis", "axis" },
        } } },
        { L"Hardmax_onnx",{ {
            { L"Hardmax_onnx", "Hardmax" },
            { L"axis", "axis" },
        } } },
        { L"Softmax_onnx",{ {
            { L"Softmax_onnx", "Softmax" },
            { L"axis", "axis" },
        } } },
        { L"LogSoftmax_onnx",{ {
            { L"LogSoftmax_onnx", "LogSoftmax" },
            { L"axis", "axis" },
        } } },
        { L"Softplus",{ {
            { L"Softplus", "Softplus" },
        } } },
        { L"Softsign",{ {
            { L"Softsign", "Softsign" },
        } } },        
        { L"Equal",{ {
            { L"Equal", "Equal" },
            { L"axis ", "axis" },
            { L"broadcast", "broadcast" },
        } } },
        { L"Greater",{ {
            { L"Greater", "Greater" },
            { L"axis ", "axis" },
            { L"broadcast", "broadcast" },
            } } },
        { L"Less",{ { 
            { L"Less", "Less" }, 
            { L"axis ", "axis" }, 
            { L"broadcast", "broadcast" }, 
        } } },
        { L"Cos",{ {
            { L"Cos", "Cos" },
            } } },
        { L"Sin",{ {
            { L"Sin", "Sin" },
        } } },
        { L"Tan",{ {
            { L"Tan", "Tan" },
            } } },
        { L"Acos",{ {
            { L"Acos", "Acos" },
        } } },
        { L"Asin",{ {
            { L"Asin", "Asin" },
        } } },
        { L"Atan",{ {
            { L"Atan", "Atan" },
        } } },

        // From reduction
        { L"ReduceElements", { {
            { L"Max", "ReduceMax" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Min", "ReduceMin" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Sum", "ReduceSum" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Mean", "ReduceMean" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Prod", "ReduceProd" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"LogSum", "ReduceLogSumExp" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Argmax", "ArgMax" },
            { L"axis", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceElements", { {
            { L"Argmin", "ArgMin" },
            { L"axis", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },
        { L"ReduceL1", { {
            { L"ReduceL1", "ReduceL1" },
            { L"axis", "axes" },
            { L"keepdims", "keepdims" },
        } } },
        { L"ReduceL2",{ {
            { L"ReduceL2", "ReduceL2" },
            { L"axis", "axes" },
            { L"keepdims", "keepdims" },
        } } },
        { L"ReduceSumSquare",{ {
            { L"ReduceSumSquare", "ReduceSumSquare" },
            { L"axis", "axes" },
            { L"keepdims", "keepdims" },
        } } },
        { L"Sequence::ReduceElements",{ {
            { L"Sequence::ReduceElements", "ReduceSum" },
            { L"axisVec", "axes" },
            { L"reductionKeepDimensions", "keepdims" },
        } } },

        // From tensor
        { L"Cast", { {
            { L"Cast", "Cast" },
            { L"newDataType", "to" },
            } } },
        { L"Splice", { {
            { L"Splice", "Concat" },
            { L"axis", "axis" },
        } } },
        // { L"", "Split" },
        { L"Slice", { {
            { L"Slice", "Slice" },
            { L"axes", "axes" },
            { L"beginIndexVec", "starts" },
            { L"endIndexVec", "ends" },
        } } },
        { L"TransposeAxes", { {
            { L"TransposeAxes", "Transpose" },
            { L"axisVec", "perm" },
        } } },
        { L"Pad",{ {
            { L"Pad", "Pad" },
            { L"mode", "mode" },
            { L"pattern", "pads" },
            { L"constant_value", "value"},
            } } },
        { L"Gather", { {
            { L"Gather", "Gather" },
            { L"axis", "axis" },
        } } },
        { L"DepthToSpace",{ {
            { L"DepthToSpace", "DepthToSpace" },
        } } },
        { L"SpaceToDepth",{ {
            { L"SpaceToDepth", "SpaceToDepth" },
        } } },
        { L"Squeeze",{ {
            { L"Squeeze", "Squeeze" },
            { L"axes", "axes" },
        } } },
        { L"ImageScaler",{ {
            { L"ImageScaler", "ImageScaler" },
            } } },
        { L"MeanVarianceNormalization",{ {
            { L"MeanVarianceNormalization", "MeanVarianceNormalization" },
            { L"useStatsAcrossChannels", "across_channels" },
            { L"doVarianceScaling", "normalize_variance" },
            } } },
        { L"Embedding",{ {
            { L"Embedding", "Gather" },
            } } },
        { L"NoOp",{ {
            { L"NoOp", "Identity" },
            } } },
        { L"Alias",{ {
            { L"Alias", "Identity" },
        } } },
        { L"StopGradient",{ {
            { L"StopGradient", "Identity" },
            } } },
        { L"Gemm",{ {
            { L"Gemm", "Gemm" },
        } } },
        { L"MatMul",{ {
            { L"MatMul", "MatMul" },
        } } },
        { L"Unsqueeze",{ {
            { L"Unsqueeze", "Unsqueeze" },
        } } },
        { L"TopK",{ {
            { L"TopK", "TopK" },
            { L"axis", "axis" },
            { L"numItems", "k" },
        } } },
        { L"Sequence::Softmax",{ {
            { L"Sequence::Softmax", "Softmax" },
        } } },
        { L"StraightThrough",{ {
            { L"StraightThrough", "StraightThrough" },
        } } },
        { L"LogPlus",{ {
            { L"LogPlus", "LogPlus" },
        } } },
        { L"Crop", { {
            { L"Crop", "Crop"},
            { L"offset", "border"},
        } } },
        { L"OneHotOp", { {
            { L"OneHotOp", "OneHotEncoder"},
        } } },
    };

    // given a cntkOpName and cntk attribute OpName which is saved in CNTK::Function's attribute,
    // return a map from cntk attribute name to onnx attribute name. 
    const AttributesMapping& Operators::FindAttributeMap(const std::wstring &cntkOpName, const std::wstring& cntkAttributeOpName)
    {
        std::unordered_multimap<std::wstring, AttributesMapping>::iterator itNodeFn =
            std::find_if(_cntkToONNXOpName.begin(), _cntkToONNXOpName.end(),
                [cntkOpName, cntkAttributeOpName](std::unordered_multimap<std::wstring, AttributesMapping>::value_type nodeFn)
        {return nodeFn.first == cntkOpName && nodeFn.second.map.find(cntkAttributeOpName) != nodeFn.second.map.end(); });

        if (itNodeFn == _cntkToONNXOpName.end())
        {
            LogicError("Cannot map to ONNX op from CNTK ReduceElements operation: %s / %s",
                       Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(cntkOpName)).c_str(), Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(cntkAttributeOpName)).c_str());
        }

        return itNodeFn->second;
    }

    std::tuple<int, int> Operators::GetElementWiseInputIndices(const std::wstring& opName)
    {
        if (!SupportBroadcast(opName))
        {
            LogicError("Calling GitElementWiseInputIndices with invalid op: %s", Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(opName)).c_str());
        }

        int index0 = 0;
        while (!IsValidInputs(opName, index0))
        {
            index0++;
        }

        int index1 = index0 + 1;
        while (!IsValidInputs(opName, index1))
        {
            index1++;
        }

        return make_tuple(index0, index1);
    }
    bool Operators::SupportBroadcast(const std::wstring& cntkOpName)
    {
        return (cntkOpName == L"Plus") || (cntkOpName == L"Minus") ||
            (cntkOpName == L"ElementTimes") || (cntkOpName == L"ElementDivide") ||
            (cntkOpName == L"And") || (cntkOpName == L"Or") || (cntkOpName == L"Xor");
    }

    bool Operators::SupportBroadcastONNXOp(const std::string& onnxOpName)
    {
        return (onnxOpName == "Add") || (onnxOpName == "Sub") ||
            (onnxOpName == "Mul") || (onnxOpName == "Div") ||
            (onnxOpName == "And") || (onnxOpName == "Or") || (onnxOpName == "Xor");
    }

    bool Operators::IsLoopOp(const std::string &opName)
    {
        return opName == "PastValue" || opName == "FutureValue";
    }

    bool Operators::IsRNNOp(const std::string &opName)
    {
        return opName == "LSTM" || opName == "GRU" || opName == "RNN" || opName == "RNNStep";
    }
    
    bool Operators::IsSequenceBlockOp(const std::string &opName)
    {
        return opName == "Sequence::ReduceElements" || opName == "Sequence::BroadcastAs";
    }

    std::unordered_map<std::wstring, std::set<size_t>> Operators::_cntkBlockOPInvalidIndices = {
            { L"Clip",{ 1, 2 } },
            { L"ELU",{ 0, 1 } },
            { L"LeakyReLU",{ 0, 1 } },
            { L"SELU",{ 0, 1, 2 } },
            { L"ElementMax",{} },
            { L"ElementMin",{} },
            { L"HardSigmoid",{ 0, 1, 2, 3 } },
            { L"Mean",{ 0 } },
            { L"Softmax",{} },
            { L"LocalResponseNormalization",{ 0, 1, 2 } },
            { L"And",{ 0 } },
            { L"Or",{ 0 } },
            { L"Xor",{ 0 } },
            { L"Not",{ 0, 1 } },
            { L"Softplus",{ 0 } },
            { L"Softsign",{ 0 } },
            { L"ImageScaler",{ 0, 1, 2, 3 } },
            { L"MeanVarianceNormalization",{ 0 } },
            { L"Sequence::Slice",{ 0, 1 } },
        };

        std::unordered_map<std::wstring, std::vector<int>> Operators::_cntkToONNXInputIndices = {
            { L"Convolution",{ 1, 0 } },
            { L"ConvolutionTranspose",{ 1, 0 } },
            { L"BatchNormalization",{ 0, 1, 2, 3, 4, -1 } },
            { L"Times",{ 1, 0 } },
            { L"Gather",{ 1, 0 } },
            { L"PReLU",{ -1, 0, 1 } },
            { L"Gemm", { -1, -1, 1, 0, 2} },
        };

        //
        // CNTK Layer API needs to be treated specially.
        //
        std::set<std::wstring> Operators::_cntkLayerOPName = {
            { L"Convolution" },
            { L"ConvolutionTranspose" },
            { L"BatchNormalization" },
            { L"Dropout" },
        };

        std::set<std::wstring> Operators::_optimizedRnnStackOpNames = {
            { L"lstm" },
            { L"rnnReLU" },
            { L"rnnTanh" },
        };

        std::unordered_map<std::wstring, std::string> Operators::_optimizedRnnOpNameToOnnxOpName = {
            { L"lstm", "LSTM" },
            { L"rnnReLU", "RNN" },
            { L"rnnTanh","RNN" },
        };

        std::set<std::wstring> Operators::_cntkOpsExportedWithBatchAxis = { // This is mostly used on export side.
            { L"Convolution" },
            { L"ConvolutionTranspose" },
            { L"Pooling" },
            { L"DepthToSpace" },
            { L"SpaceToDepth" },
            { L"LocalResponseNormalization" },
            { L"MeanVarianceNormalization" },
            { L"BatchNormalization" },
            { L"ImageScaler" },
        };

        std::set<std::string> Operators::_onnxSimpleBatchAxisOps = { // List of all ONNX ops that are simple (single input, output) and have batch axis.
            { "MaxPool" },
            { "AveragePool" },
            { "GlobalAveragePool" },
            { "GlobalMaxPool" },
            { "DepthToSpace" },
            { "SpaceToDepth" },
            { "LRN" },
            { "MeanVarianceNormalization" },
            { "ImageScaler" },
            { "Crop" },
        };

    }
}
