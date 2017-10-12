//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "proto/onnx/core/model.h"
#include "proto/onnx/core/graph.h"

#include "Utils.h"
#include "Operators.h"
#include "BlockFunction.h"
#include <vector>

using namespace CNTK::ONNX;
using namespace CNTK;

//
// A helper function, to reverse any iterable container and return a copy
// of the reversed container.
//
template<typename ItrType>
ItrType reverse(ItrType v)
{
    std::reverse(std::begin(v), std::end(v));
    return v;
}

//
// Helper function to reduce the rank of a shape.
//
ONNXIR::TypeProto ReduceRank(const ONNXIR::TypeProto::TensorShapeProto* inputShape, int reductionRank, bool rightReduction)
{
    assert(inputShape != nullptr);

    int inputRank = inputShape->dim_size();
    assert(inputRank > reductionRank);

    ONNXIR::TypeProto newShape;
    int64_t reduceDim = 1;

    if (rightReduction)
    {
        for (int index = 0; index < (inputRank - reductionRank); index++)
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

        for (int index = (inputRank - reductionRank); index < inputRank; index++)
            reduceDim *= inputShape->dim(index).dim_value();

        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);
    }
    else
    {
        for (int index = 0; index < reductionRank; index++)
            reduceDim *= inputShape->dim(index).dim_value();

        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);

        for (int index = reductionRank; index < inputRank; index++)
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());
    }

    return newShape;
}

namespace CNTK
{

class CNTKToONNXHelper
{
public:
    //
    // Copy the entire CNTK graph to ONNX graph.
    //
    static void Copy(const FunctionPtr& src, ONNXIR::Graph* dst);

private:
    //
    // Recursively create ONNX nodes corresponding to each CNTK node.
    //
    static ONNXIR::Node* CreateNode(const FunctionPtr& src,
                                     ONNXIR::Graph* graph,
                                     std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
                                     std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
                                     const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    //
    // Traverse the entire graph and collect variable mapping between graph inside and outside the block.
    //
    static void TraverseGraph(const FunctionPtr& src,
                              std::set<FunctionPtr>& visited,
                              std::unordered_map<Variable, Variable>& compositeOutputsMap);

    //
    // Copy the content of NDArrayView to TensorProto, and do the needed
    // convergence.
    //
    static void CopyTensor(const NDArrayViewPtr src, ONNXIR::TensorProto& dst);

    //
    // Copy supported attributes from CNTK node to corresponding ONNX node.
    //
    static void CopyAttributes(const FunctionPtr& src, ONNXIR::Node* node);

    //
    // Convert Axis object to actual tensor index.
    //
    static int ToIndex(const Axis& axis);

    //
    // Convert NDShape and various std::vector types to TensorShape
    //
    static ONNXIR::TypeProto ToTypeProto(const NDShape& shape, bool hasBatchAxis = false);
    static ONNXIR::TypeProto ToTypeProto(const std::vector<bool>& shape);
    static ONNXIR::TypeProto ToTypeProto(const std::vector<int>& shape);
    static ONNXIR::TypeProto ToTypeProto(const std::vector<Axis>& axes);

    //
    // Convert TypeProto, NDShape and various std::vector types to std::vector
    //
    static std::vector<int64_t> ToINTS(const ONNXIR::TypeProto& shape);
    static std::vector<int64_t> ToINTS(const NDShape& shape, bool hasBatchAxis = false);
    static std::vector<int64_t> ToINTS(const std::vector<bool>& shape);
    static std::vector<int64_t> ToINTS(const std::vector<int>& shape);
    static std::vector<int64_t> ToINTS(const std::vector<Axis>& axes);

    //
    // Convert data types from CNTK to ONNX.
    //
    static void UpdateONNXType(DataType dataType, ONNXIR::TypeProto& type);

    //
    // Map CNTK OP names to ONNX OP Names.
    //
    static std::string ToOPName(const FunctionPtr& src);

    //
    // Check that the CNTK variable is compatible with ONNX.
    //
    static void ValidateVariable(const Variable& v);

    //
    // Which input to ignore during converting a CNTK block to a primitive OP in ONNX.
    //
    static bool FilterInput(const FunctionPtr& src, const CNTK::Variable& input, size_t inputIndex);

    //
    // Argument orders between CNTK and ONNX aren't always the same.
    //
    static std::vector<ONNXIR::NodeArg> MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<ONNXIR::NodeArg>& inputs);

    //
    // Add current CNTK node to ONNX graph.
    //
    static ONNXIR::Node* AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs);
};
}

std::unique_ptr<ONNXIR::Model> CNTKToONNX::CreateModel(const FunctionPtr& src)
{
    std::unique_ptr<ONNXIR::Model> model(new ONNXIR::Model("CNTKGraph", true));
    auto dstGraph = model->MainGraph();
    CNTKToONNXHelper::Copy(src, dstGraph);
    ONNXIR::Status status = dstGraph->Resolve();
    if (!status.Ok())
        LogicError("%s", status.ErrorMsg().c_str());
    return model;
}

void CNTKToONNXHelper::Copy(const FunctionPtr& src, ONNXIR::Graph* dst)
{
    std::set<FunctionPtr> visited;
    std::unordered_map<Variable, Variable> compositeOutputsMap;
    std::unordered_map<FunctionPtr, ONNXIR::Node*> functionNodes;
    std::unordered_map<Variable, ONNXIR::Node*> variableNodes;

    //
    // Traverse the graph and collect some information.
    //
    TraverseGraph(src, visited, compositeOutputsMap);

    //
    // Iterate through each node in CNTK graph and create an equivalent node
    // in ONNX graph.
    //
    CreateNode(src, dst, functionNodes, variableNodes, compositeOutputsMap);
}

void CNTKToONNXHelper::CopyTensor(const NDArrayViewPtr src, ONNXIR::TensorProto& dst)
{
    auto dataType = src->GetDataType();
    auto srcTemp = src->DeepClone();
    auto srcShape = srcTemp->Shape();
    auto totalSize = srcShape.TotalSize();

    // This is our own copy so move it to the CPU.
    srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

    switch (dataType)
    {
        case DataType::Float:
        {
            dst.set_data_type(ONNXIR::TensorProto_DataType_FLOAT);
            auto data = srcTemp->DataBuffer<float>();
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_float_data()->Add()) = data[index];

            break;
        }
        case DataType::Double:
        {
            dst.set_data_type(ONNXIR::TensorProto_DataType_DOUBLE);
            auto data = srcTemp->DataBuffer<double>();
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_double_data()->Add()) = data[index];

            break;
        }
        default:
            NOT_IMPLEMENTED;
    }

    auto dimensions = reverse(srcShape.Dimensions());
    for (auto dim : dimensions)
        *(dst.mutable_dims()->Add()) = dim;
}

int CNTKToONNXHelper::ToIndex(const Axis& axis)
{
    if ((axis == Axis::AllAxes()) || (axis == Axis::AllStaticAxes()))
        LogicError("AllAxes and AllStaticAxes are currently not supported.");

    if (axis.IsSequenceAxis())
        LogicError("Sequence axis are currently not supported.");

    if (axis.IsBatchAxis())
        return 0;
 
    return axis.StaticAxisIndex() + 1;
}

ONNXIR::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, bool hasBatchAxis)
{
    ONNXIR::TypeProto newShape;
    if (shape.HasUnboundDimension())
        LogicError("Inferred and FreeDimension aren't currently supported.");

    if (hasBatchAxis)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto dimensions = reverse(shape.Dimensions());
    for (auto dimension : dimensions)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
    }

    return newShape;
}

ONNXIR::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<bool>& shape)
{
    ONNXIR::TypeProto newShape;
    auto dimensions = reverse(shape);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension ? 1:0);

    return newShape;
}

ONNXIR::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<int>& shape)
{
    ONNXIR::TypeProto newShape;
    auto dimensions = reverse(shape);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

ONNXIR::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<Axis>& axes)
{
    std::vector<int> axesValue;
    for (auto axis : axes)
    {
        axesValue.push_back(ToIndex(axis));
    }
    std::sort(axesValue.begin(), axesValue.end());

    ONNXIR::TypeProto newShape;
    for (auto dimension : axesValue)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const ONNXIR::TypeProto& shape)
{
    std::vector<int64_t> newShape;

    for (int i = 0; i < shape.tensor_type().shape().dim_size(); i++)
        newShape.push_back((int64_t)shape.tensor_type().shape().dim(i).dim_value());

    return newShape;
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const NDShape& shape, bool hasBatchAxis)
{
    return ToINTS(ToTypeProto(shape, hasBatchAxis));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<bool>& shape)
{
    return ToINTS(ToTypeProto(shape));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<int>& shape)
{
    return ToINTS(ToTypeProto(shape));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<Axis>& axes)
{
    return ToINTS(ToTypeProto(axes));
}

void CNTKToONNXHelper::UpdateONNXType(DataType dataType, ONNXIR::TypeProto& type)
{
    switch (dataType)
    {
    case DataType::Float:
        type.mutable_tensor_type()->set_elem_type(ONNXIR::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        type.mutable_tensor_type()->set_elem_type(ONNXIR::TensorProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

std::string CNTKToONNXHelper::ToOPName(const FunctionPtr& src)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToString(src->OpName());
    if (lookup.count(src->OpName()) == 1)
    {
        auto attributesMap = lookup.find(src->OpName())->second.map;
        opName = attributesMap[src->OpName()];
    }
    else
    {
        // Some nodes map one to many.
        if (src->OpName() == L"Convolution")
        {
            auto transpose = (bool)src->Attributes()[L"transpose"].Value<bool>();
            if (transpose)
                opName = "ConvTranspose";
            else
                opName = "Conv";
        }
        else if (src->OpName() == L"Pooling")
        {
            PoolingType poolingType = (PoolingType)src->Attributes()[L"poolingType"].Value<size_t>();
            if (poolingType == PoolingType::Max)
                opName = "MaxPool";
            else
                opName = "AveragePool";
        }
    }

    return opName;
}

void CNTKToONNXHelper::ValidateVariable(const Variable& v)
{
    if ((v.HasBatchAxis() && (v.DynamicAxes().size() > 1)) ||
        (!v.HasBatchAxis() && (v.DynamicAxes().size() > 0)))
    {
        LogicError("Sequence and user defined dynamic axis are currently not supported.");
    }
}

bool CNTKToONNXHelper::FilterInput(const FunctionPtr& src, const CNTK::Variable& input, size_t inputIndex)
{
    // In CNTK block functions, they expose all constants inside the block. For block functions that
    // map directly to ONNX OP, we don't care about constanst inside the block.
    if (input.IsConstant())
        return !Operators::IsValidInputs(src->OpName(), inputIndex);
    return false;
}

//
// This is the main horsepower, it navigate CNTK graph recursivley while keep track of all visited nodes and variables, 
// and create the corresponding ONNX graph.
//
ONNXIR::Node* CNTKToONNXHelper::CreateNode(const FunctionPtr& src,
                                            ONNXIR::Graph* graph,
                                            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
                                            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes, 
                                            const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = functionNodes.find(src);
    if (iter != functionNodes.end())
        return iter->second;

    ONNXIR::Node* functionNode = nullptr;
    std::string opName = ToString(src->OpName());

    //
    // If this block node equivalent to a primitive ONNX OP, then treated as such.
    // And just maps its argument to ONNX node.
    //
    if (src->IsBlock() && 
        (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
    {
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    //
    // For compatibility of other framework that support ONNX, we will limit the list of OPs to the one
    // supported by ONNX https://github.com/onnx/onnx/tree/master/onnx/defs.
    //
    else if (Operators::IsSupportedCNTKOP(src->OpName()))
    {
        std::vector<ONNXIR::NodeArg> inputs;
        std::vector<ONNXIR::NodeArg> outputs;

        for (const auto& output : src->Outputs())
        {
            ValidateVariable(output);

            auto outputArgType = ToTypeProto(output.Shape(), output.HasBatchAxis());
            UpdateONNXType(output.GetDataType(), outputArgType);

            ONNXIR::NodeArg outputArg(ToString(output.Uid()), &outputArgType);
            outputs.push_back(outputArg);
        }

        for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
        {
            auto input = src->Inputs()[inputIndex];

            if (input.IsPlaceholder())
            {
                input = input.BlockFunctionVariableMapping();
                if (input.IsPlaceholder())
                    LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
            }
            ValidateVariable(input);

            if (src->IsBlock() && FilterInput(src, input, inputIndex))
                continue;

            //
            // Use user defined name if available otherwise use our internel unique name ID.
            //
            std::string inputName = ToString(input.Uid());
            auto inputItr = compositeOutputsMap.find(input);
            if (inputItr != compositeOutputsMap.end())
                inputName = ToString(inputItr->second.Uid());

            auto inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis());
            UpdateONNXType(input.GetDataType(), inputArgType);
            ONNXIR::NodeArg inputArg(inputName, &inputArgType);

            inputs.push_back(inputArg);

            //
            // Leaf nodes are data entry to the graph and need their own node with only output arg.
            //
            if ((input.IsParameter() || input.IsConstant()) &&
                !Operators::IgnoreConstantAndParameter(src->OpName(), inputIndex))
            {
                if (variableNodes.find(input) == variableNodes.end())
                {
                    std::vector<ONNXIR::NodeArg> varInputs;
                    std::vector<ONNXIR::NodeArg> varOutputs;

                    varOutputs.push_back({ inputArg });
                    ONNXIR::Node* variableNode = nullptr;
                    if (input.IsParameter() || input.IsConstant())
                    {
                        variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);
                        auto srcTensor = input.IsParameter() ? Parameter(input).Value() : Constant(input).Value();
                        
                        ONNXIR::TensorProto dstTensor;
                        CopyTensor(srcTensor, dstTensor);

                        variableNode->AddAttribute("value", dstTensor);
                        variableNodes.emplace(input, variableNode);
                    }
                }
            }
            //
            // If this input is output, then it is the ouput of an up stream node. Recursively add all upstream nodes.
            // Pretty much, we are doing DFS.
            //
            else if (input.IsOutput())
                CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);
        }

        //
        // Finally add a new node to ONNX graph.
        //
        functionNode = AddNode(src, graph, inputs, outputs);
    }
    else
        LogicError("Node '%S': Unsupported node.", src->AsString().c_str());

    functionNodes.emplace(src, functionNode);
    return functionNode;
}

void CNTKToONNXHelper::TraverseGraph(const FunctionPtr& src, 
                                     std::set<FunctionPtr>& visited,
                                     std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = visited.find(src);
    if (iter != visited.end())
        return;

    std::string opName = ToString(src->OpName());
    if (src->IsBlock())
    {
        if (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName()))
        {
            auto blockSrc = dynamic_cast<BlockFunction*>(src.get());
            for (auto map : blockSrc->CompositeOutputsMap())
                compositeOutputsMap.insert(map);
            TraverseGraph(src->BlockRoot(), visited, compositeOutputsMap);
        }
    }
    else
    {
        for (auto input : src->Inputs())
        {
            if (input.IsPlaceholder())
            {
                input = input.BlockFunctionVariableMapping();
                if (input.IsPlaceholder())
                    LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
            }

            if (input.IsOutput())
                TraverseGraph(input.Owner(), visited, compositeOutputsMap);
        }
    }

    visited.emplace(src);
}

void CNTKToONNXHelper::CopyAttributes(const FunctionPtr& src, ONNXIR::Node* node)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToString(src->OpName());
    if (lookup.count(src->OpName()) == 1)
    {
        auto attributesMap = lookup.find(src->OpName())->second.map;
        opName = attributesMap[src->OpName()];

        if (src->OpName() == L"BatchNormalization")
        {
            auto spatial = (int64_t)((bool)src->Attributes()[L"spatial"].Value<bool>() ? 1 : 0);
            auto normalizationTimeConstant = (float)src->Attributes()[L"normalizationTimeConstant"].Value<double>();
            // auto blendTimeConstant = (float)src->Attributes()[L"blendTimeConstant"].Value<double>();
            auto epsilon = (float)src->Attributes()[L"epsilon"].Value<double>();

            //
            // onnx: running_mean = running_mean * momentum + mean * (1 - momentum)
            // cntk: expAvgFactor * MB stats + (1-expAvgFactor) * prev running stats
            //
            auto momentum = 0.0f;
            if (!isfinite(normalizationTimeConstant))
                momentum = 1.0f;
            else if (normalizationTimeConstant > 0)
                momentum = 1.0f + expm1(-48.0f / normalizationTimeConstant);

            node->AddAttribute(attributesMap[L"spatial"], spatial);
            node->AddAttribute("is_test", (int64_t)1);
            node->AddAttribute(attributesMap[L"epsilon"], epsilon);
            node->AddAttribute("momentum", momentum);
        }
        else if (src->OpName() == L"LocalResponseNormalization")
        {
            auto depthRadius = (int64_t)src->Attributes()[L"depthRadius"].Value<size_t>();
            auto bias = (float)src->Attributes()[L"bias"].Value<double>();
            auto alpha = (float)src->Attributes()[L"alpha"].Value<double>();
            auto beta = (float)src->Attributes()[L"beta"].Value<double>();

            node->AddAttribute(attributesMap[L"depthRadius"], depthRadius);
            node->AddAttribute(attributesMap[L"bias"], bias);
            node->AddAttribute(attributesMap[L"alpha"], alpha);
            node->AddAttribute(attributesMap[L"beta"], beta);
        }
        else if ((src->OpName() == L"LeakyReLU") || (src->OpName() == L"ELU"))
        {
            auto alpha = 0.01f;
            if (src->Attributes().Contains(L"alpha"))
                alpha = (float)src->Attributes()[L"alpha"].Value<double>();
            node->AddAttribute("alpha", 0.01f);
        }
        else if (src->OpName() == L"SELU")
        {
            auto alpha = 1.6732f;
            if (src->Attributes().Contains(L"alpha"))
                alpha = (float)src->Attributes()[L"alpha"].Value<double>();

            auto gamma = 1.0507f;
            if (src->Attributes().Contains(L"gamma"))
                gamma = (float)src->Attributes()[L"gamma"].Value<double>();

            node->AddAttribute("alpha", alpha);
            node->AddAttribute("gamma", gamma);
        }
        else if (src->OpName() == L"Dropout")
        {
            auto dropoutRate = (float)src->Attributes()[L"dropoutRate"].Value<double>();
            node->AddAttribute(attributesMap[L"dropoutRate"], dropoutRate);
            node->AddAttribute("is_test", (int64_t)1);
        }
        else if ((src->OpName() == L"UniformRandom") || (src->OpName() == L"NormalRandom") ||
                 (src->OpName() == L"UniformRandomLike") || (src->OpName() == L"NormalRandomLike"))
        {
            auto randomArgs = AsVector<double>(src->Attributes()[L"randomDistributionArgs"].Value<std::vector<DictionaryValue>>());
            auto seed = (int64_t)src->Attributes()[L"rngSeed"].Value<int>();
            
            if ((src->OpName() == L"UniformRandom") || (src->OpName() == L"UniformRandomLike"))
            {
                node->AddAttribute("low", (float)randomArgs[0]);
                node->AddAttribute("high", (float)randomArgs[1]);
            }
            else
            {
                node->AddAttribute("mean", (float)randomArgs[0]);
                node->AddAttribute("scale", (float)randomArgs[1]);
            }

            node->AddAttribute(attributesMap[L"rngSeed"], seed);
            if ((src->OpName() == L"UniformRandom") || (src->OpName() == L"NormalRandom"))
            {
                auto shape = (NDShape)src->Attributes()[L"newShape"].Value<NDShape>();
                node->AddAttribute(attributesMap[L"newShape"], ToINTS(shape));
            }
        }
        else if ((src->OpName() == L"ReduceMax")  || (src->OpName() == L"ReduceMin")    ||
                 (src->OpName() == L"ReduceSum")  || (src->OpName() == L"ReduceMean")   ||
                 (src->OpName() == L"ReduceProd") || (src->OpName() == L"ReduceLogSum") ||
                 (src->OpName() == L"Argmax")     || (src->OpName() == L"Argmin"))
        {
            auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
            std::vector<Axis> reductionAxes;
            if (src->Attributes().Contains(L"axisVec"))
                reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            else if (src->Attributes().Contains(L"axis"))
                reductionAxes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));

            node->AddAttribute(attributesMap[L"reductionKeepDimensions"], keepReducedDimensions);
            node->AddAttribute("axes", ToINTS(reductionAxes));
        }
        else if (src->OpName() == L"Transpose")
        {
            std::vector<Axis> perm = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            node->AddAttribute(attributesMap[L"axisVec"], ToINTS(perm));
        }
        else if (src->OpName() == L"Reshape")
        {
            auto shape = (NDShape)src->Attributes()[L"newShape"].Value<NDShape>();
            node->AddAttribute(attributesMap[L"newShape"], ToINTS(shape, true));
        }
        else if (src->OpName() == L"Splice")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            node->AddAttribute(attributesMap[L"axis"], (int64_t)ToIndex(axis));
        }
        else if (src->OpName() == L"Slice")
        {
            std::vector<Axis> sliceAxes;
            std::vector<int> beginIndex;
            std::vector<int> endIndex;

            if (src->Attributes().Contains(L"axisVec"))
            {
                sliceAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                beginIndex = AsVector<int>(src->Attributes()[L"beginIndexVec"].Value<std::vector<DictionaryValue>>());
                endIndex = AsVector<int>(src->Attributes()[L"endIndexVec"].Value<std::vector<DictionaryValue>>());
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                sliceAxes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));
                beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
            }

            node->AddAttribute(attributesMap[L"axes"], ToINTS(sliceAxes));
            node->AddAttribute(attributesMap[L"starts"], ToINTS(beginIndex));
            node->AddAttribute(attributesMap[L"ends"], ToINTS(endIndex));
        }
        else if (src->OpName() == L"Softmax")
        {
            Axis axis = Axis(0);
            if (src->Attributes().Contains(L"axis"))
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            node->AddAttribute(attributesMap[L"axis"], (int64_t)ToIndex(axis));
        }
        else if ((src->OpName() == L"Plus") || (src->OpName() == L"Minus") ||
                 (src->OpName() == L"ElementTimes") || (src->OpName() == L"ElementDivide"))
        {
            node->AddAttribute("broadcast", (int64_t)1);
            // node->AddAttribute("axis", (int64_t)1);
        }
        else if (src->OpName() == L"Times")
        {
            size_t outputRank = src->Attributes()[L"outputRank"].Value<size_t>();
            if (outputRank > 1)
                LogicError("Output rank other than 1 is not supported.");
        }
    }
    else
    {
        // Some nodes map one to many.
        if (src->OpName() == L"Convolution")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"kernelShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());
            auto dilations = (NDShape)src->Attributes()[L"dilation"].Value<NDShape>();
            auto transpose = (bool)src->Attributes()[L"transpose"].Value<bool>();

            //
            // Remove the channel part for ONNX.
            //
            kernelShape = kernelShape.SubShape(0, kernelShape.Rank() - 1);
            strides = strides.SubShape(0, strides.Rank() - 1);
            autoPadding.pop_back();
            dilations = dilations.SubShape(0, dilations.Rank() - 1);

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            node->AddAttribute("pads", ToINTS(autoPadding));
            node->AddAttribute("dilations", ToINTS(dilations));
            node->AddAttribute("group", (int64_t)1);

            if (transpose)
            {
                auto outputShape = (NDShape)src->Attributes()[L"outputShape"].Value<NDShape>();
                node->AddAttribute("output_shape", ToINTS(outputShape, src->Inputs()[1].HasBatchAxis()));
            }
        }
        else if (src->OpName() == L"Pooling")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"poolingWindowShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            node->AddAttribute("pads", ToINTS(autoPadding));
        }
    }
}

std::vector<ONNXIR::NodeArg> CNTKToONNXHelper::MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<ONNXIR::NodeArg>& inputs)
{
    if (Operators::HasInputIndexMap(src->OpName()))
    {
        std::vector<ONNXIR::NodeArg> orderedInputs;
        std::map<int, ONNXIR::NodeArg> orderedInputsMap;
        auto map = Operators::ToONNXInputIndexMap(src->OpName());

        for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex)
        {
            if (map[inputIndex] >= 0)
                orderedInputsMap.insert(std::pair<int, ONNXIR::NodeArg>(map[inputIndex], inputs[inputIndex]));
        }

        for (const auto& item : orderedInputsMap)
            orderedInputs.push_back(item.second);

        return orderedInputs;
    }

    return inputs;
}

ONNXIR::Node* CNTKToONNXHelper::AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs)
{
    ONNXIR::Node* node = nullptr;
    auto orderedInputs = MapInputsOrderToONNX(src, inputs);
    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());

    //
    // CNTK Times OP is way more flexible for ONNX, so depend on the inputs and output shape,
    // we will need to insert some reshapes.
    //
    if (src->OpName() == L"Times")
    {
        auto input1Shape = orderedInputs[0].Shape();
        auto input2Shape = orderedInputs[1].Shape();
        auto outputShape = outputs[0].Shape();

        int input1Rank = input1Shape->dim_size();
        int input2Rank = input2Shape->dim_size();
        int outputRank = outputShape->dim_size();
        int reductionRank = (input1Rank + input2Rank - outputRank) / 2;

        if (reductionRank > 1) // We need to insert reshape.
        {
            auto input1Reshape = ReduceRank(input1Shape, reductionRank, true);
            auto input2Reshape = ReduceRank(input2Shape, reductionRank, false);

            UpdateONNXType(src->Inputs()[1].GetDataType(), input1Reshape);
            UpdateONNXType(src->Inputs()[0].GetDataType(), input2Reshape);

            ONNXIR::NodeArg inputOutput1Arg(orderedInputs[0].Name() + string("_reshape0"), &input1Reshape);
            ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape1"), &input2Reshape);

            auto reshapeNode1 = graph->AddNode(nodeName + string("_reshape0"), "Reshape", "", { orderedInputs[0] }, { inputOutput1Arg });
            auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });

            reshapeNode1->AddAttribute("shape", ToINTS(input1Reshape));
            reshapeNode2->AddAttribute("shape", ToINTS(input2Reshape));

            node = graph->AddNode(nodeName, ToOPName(src), "", { inputOutput1Arg , inputOutput2Arg }, outputs);
        }
        else
            node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
    }
    else
        node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);

    //
    // Copy and validate attributes.
    //
    CopyAttributes(src, node);

    return node;
}
