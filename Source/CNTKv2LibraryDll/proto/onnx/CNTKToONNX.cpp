//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKToONNX.h"
#include "proto/onnx/core/model.h"
#include "proto/onnx/core/graph.h"
#include "proto/onnx/core/status.h"

#include "Utils.h"
#include "Operators.h"
#include "BlockFunction.h"
#include <vector>
#include <tuple>

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
onnx::TypeProto ReduceRank(const onnx::TensorShapeProto* inputShape, int reductionRank, bool rightReduction)
{
    assert(inputShape != nullptr);

    int inputRank = inputShape->dim_size();
    assert(inputRank > reductionRank);

    onnx::TypeProto newShape;
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
        static void CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType = nullptr);

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
        static onnx::TypeProto ToTypeProto(const NDShape& shape, bool hasBatchAxis = false);
        static onnx::TypeProto ToTypeProto(const std::vector<bool>& shape);
        static onnx::TypeProto ToTypeProto(const std::vector<int>& shape, bool doReverseVec = true);
        static onnx::TypeProto ToTypeProto(const std::vector<Axis>& axes);

        //
        // Convert TypeProto, NDShape and various std::vector types to std::vector
        //
        static std::vector<int64_t> ToINTS(const onnx::TypeProto& shape);
        static std::vector<int64_t> ToINTS(const NDShape& shape, bool hasBatchAxis = false);
        static std::vector<int64_t> ToINTS(const std::vector<bool>& shape);
        static std::vector<int64_t> ToINTS(const std::vector<int>& shape, bool doReverseVec = true);
        static std::vector<int64_t> ToINTS(const std::vector<Axis>& axes);

        static std::vector<float> INTSToVecFloat(const std::vector<int64_t> &ints);
        static std::vector<int64_t> AxesToINTSArgsortIncrementBatchAxis(const std::vector<Axis> &axes);

        //
        // Convert data types from CNTK to ONNX.
        //
        static void UpdateONNXType(DataType dataType, onnx::TypeProto& type);

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
        // Given input tersor shapes of a CNTK element wise operation, figure out 
        // the shape of the second input to ONNX operation.
        // It also returns whether broadcast is required and the axis for broadcast.
        //
        static std::tuple<NDShape, bool, int> AdjustForBroadcastShape(
            const NDShape &lhs, const NDShape &rhs, bool lhsHasBatchAxis, bool rhsHasBatchAxis);
    
        //
        // Argument orders between CNTK and ONNX aren't always the same.
        //
        static std::vector<ONNXIR::NodeArg> MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<ONNXIR::NodeArg>& inputs);

        //
        // Add current CNTK node to ONNX graph.
        //
        static ONNXIR::Node* AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs);

        //
        // Get ONNX 'pads' attribute value based on CNTK node's autoPadding attribute value.
        //
        static std::pair<std::vector<int>, std::vector<int> > GetONNXPadsAttributeFromCNTKNode(const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape);

        //
        // Adds attributes 'auto_pad' or 'pads' to saved node (typically convolution or pooling).
        //
        static void PutAutopadOrPadAttrInNode(ONNXIR::Node* node, const std::vector<bool>& autoPadding, const NDShape& kernelShape);
    };
}

std::unique_ptr<ONNXIR::Model> CNTKToONNX::CreateModel(const FunctionPtr& src)
{
    std::unique_ptr<ONNXIR::Model> model(new ONNXIR::Model("CNTKGraph", true));
    auto dstGraph = model->MainGraph();
    CNTKToONNXHelper::Copy(src, dstGraph);
    ONNXIR::Common::Status status = dstGraph->Resolve();
    if (!status.Ok())
        LogicError("%s", status.ErrorMessage().c_str());
    model->SetModelversion(static_cast<ONNXIR::VERSION>(CNTK_ONNX_MODEL_VERSION)); // This is the default. Should be surfaced as graph's 'save' API input.
    model->SetProducerVersion(CNTK_ONNX_PRODUCER_VERSION);
    model->SetProducerName(CNTK_ONNX_PRODUCER_NAME);
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

void CNTKToONNXHelper::CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType /*=nullptr*/)
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
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        auto data = srcTemp->DataBuffer<float>();
        for (size_t index = 0; index < totalSize; index++)
            *(dst.mutable_float_data()->Add()) = data[index];

        break;
    }
    case DataType::Double:
    {
        dst.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        auto data = srcTemp->DataBuffer<double>();
        for (size_t index = 0; index < totalSize; index++)
            *(dst.mutable_double_data()->Add()) = data[index];

        break;
    }
    default:
        NOT_IMPLEMENTED;
    }

    // use 
    if (inputArgType != nullptr)
    {
        std::vector<int64_t> dimensions = CNTKToONNXHelper::ToINTS(*inputArgType);
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
    else
    {
        auto dimensions = reverse(srcShape.Dimensions());
        for (auto dim : dimensions)
            *(dst.mutable_dims()->Add()) = dim;
    }
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

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, bool hasBatchAxis)
{
    onnx::TypeProto newShape;
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

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<bool>& shape)
{
    onnx::TypeProto newShape;
    auto dimensions = reverse(shape);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension ? 1 : 0);

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<int>& shape,
    bool doReverseVec /* = true*/)
{
    onnx::TypeProto newShape;
    std::vector<int> dimensions(shape);
    if (doReverseVec)
        dimensions = reverse(dimensions);
    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<Axis>& axes)
{
    std::vector<int> axesValue;
    for (auto axis : axes)
    {
        axesValue.push_back(ToIndex(axis));
    }
    std::sort(axesValue.begin(), axesValue.end());

    onnx::TypeProto newShape;
    for (auto dimension : axesValue)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);

    return newShape;
}

std::vector<int64_t> CNTKToONNXHelper::AxesToINTSArgsortIncrementBatchAxis(const std::vector<Axis> &axes)
{
    std::vector<int64_t> index(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        index[i] = axes[i].StaticAxisIndex();
    }

    std::sort(index.begin(), index.end(),
        [axes](int64_t i1, int64_t i2) {return axes[i1].StaticAxisIndex() < axes[i2].StaticAxisIndex(); });

    for (int i = 0; i < axes.size(); i++)
        index[i]++;

    // add batch axis
    index.insert(index.begin(), 0);
    return index;
}

std::vector<float> CNTKToONNXHelper::INTSToVecFloat(const std::vector<int64_t> &ints)
{
    std::vector<float> vecFloat(ints.size());
    for (int i = 0; i < ints.size(); i++)
    {
        vecFloat[i] = (float)ints[i];
    }

    return vecFloat;
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const onnx::TypeProto& shape)
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

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<int>& shape,
    bool doReverseVec /* = true*/)
{
    return ToINTS(ToTypeProto(shape, doReverseVec));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<Axis>& axes)
{
    return ToINTS(ToTypeProto(axes));
}

void CNTKToONNXHelper::UpdateONNXType(DataType dataType, onnx::TypeProto& type)
{
    switch (dataType)
    {
    case DataType::Float:
        type.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        type.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_DOUBLE);
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
        else if (src->OpName() == L"ReduceElements")
        {
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();

            const AttributesMapping& attributeMap = Operators::FindAttributeMap(src->OpName(), cntkAttributeOpName);

            opName = attributeMap.map.at(cntkAttributeOpName);
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

/*
ONNX specifies braodcast for elementwise ops in following manners
shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
shape(A) = (2, 3, 4, 5), shape(B) = (5,)
shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

CNTK handles braodcast implicitely that requires rhs to match the rank of the lhs
in a way to fill dimensions with 1s if needed. For example with above example 4,
the shape of the lhs shall be:
(1, 3, 4, 1)

Note also that there is an addition batch dimension at the front of the shape in ONNX.
CNTK shapes passed in this method are only for element chapes.

This method shall be called when constructing rhs (input[0]) of an elementwise ops
and when constructing the rhs node itself.
*/
std::tuple<NDShape, bool, int> CNTKToONNXHelper::AdjustForBroadcastShape(
    const NDShape &lhs, const NDShape &rhs, bool lhsHasBatchAxis, bool rhsHasBatchAxis)
{
    // in case of batch axis, all constants need to be broadcasted.
    bool broadCast = false;
    int axis = 0;
    if (lhs.Rank() < rhs.Rank())
    {
        // this is in contradiction with ONNX. However it exists in CNTK.
        // For example, CNTK allows this:
        // C.plus([1, 2, 3], [[2,  3,  4], [ 4,  5,  6]])
        // On the otherhand, ONNX requires first input to
        // have equal or higher rank than the second input.
        return std::tuple<NDShape, bool, int>(rhs, false, 0);
    }
    else if (lhs.Rank() > rhs.Rank())
    { 
        axis = lhs.Rank() - rhs.Rank();
        if (lhsHasBatchAxis)
            axis++;
        return std::tuple<NDShape, bool, int>(rhs, true, axis);
    }

    // dimension are reversed in CNTK comparing with ONNX
    auto lhsDims = reverse(lhs.Dimensions());
    auto rhsDims = reverse(rhs.Dimensions());

    int axis_start = -1;
    int axis_stop = rhsDims.size();
    for (int i = 0; i < rhsDims.size(); i++)
    {
        if (lhsDims[i] != rhsDims[i])
        {
            broadCast = true;
            if (axis_start != -1)
            {
                axis_stop = i;
                break;
            }
        }
        else
            if (rhsDims[i] != 1 && axis_start == -1)
            {
                axis_start = i;
            }
    }

    if (!broadCast)
    {
        // now all dims matches. there is however a case
        // where lhs has batch axis but not rhs. we need to force broadcast so that
        // [#][3,4,5] with [][3,4,5] is treated as broadcast to pass caffe2.
        // in this case, start_axis is 1 (after the batch axis).
        bool forceBroadCast = lhsHasBatchAxis && !rhsHasBatchAxis;
        return std::tuple<NDShape, bool, int>(rhs, forceBroadCast, 1);
    }
    std::vector<size_t> dimensions;
    for (int i = axis_start > 0 ? axis_start : 0; i < axis_stop; i++)
    {
        dimensions.push_back(rhsDims[i]);
    }

    NDShape shape(dimensions);
    return std::tuple<NDShape, bool, int>(shape, broadCast, lhsHasBatchAxis ? (axis_start + 1) : axis_start);
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

            bool isConstant = (input.IsParameter() || input.IsConstant()) &&
                !Operators::IgnoreConstantAndParameter(src->OpName(), inputIndex);

            // element wise ops with broadcast - the second (rhs) shall be a constant to be broadcasted 
            // onto the first variable
            bool reshapeNeededForBroadcastConstant =
                inputIndex == 1 &&
                src->Inputs().size() == 2 &&
                Operators::SupportBroadcast(src->OpName());

            onnx::TypeProto inputArgType;

            if (reshapeNeededForBroadcastConstant)
            {
                NDShape broadcastShape;
                bool broadcast = false;
                int axis = 0;
                std::tie<NDShape, bool, int>(broadcastShape, broadcast, axis) =
                    AdjustForBroadcastShape(src->Inputs()[0].Shape(), src->Inputs()[1].Shape(),
                        src->Inputs()[0].HasBatchAxis(), src->Inputs()[1].HasBatchAxis());
                inputArgType = broadcast ? ToTypeProto(broadcastShape) : 
                    ToTypeProto(input.Shape(), input.HasBatchAxis());
            }
            else
            {
                inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis());
            }

            UpdateONNXType(input.GetDataType(), inputArgType);
            ONNXIR::NodeArg inputArg(inputName, &inputArgType);

            inputs.push_back(inputArg);

            //
            // Leaf nodes are data entry to the graph and need their own node with only output arg.
            //
            if (isConstant)
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

                        onnx::TensorProto dstTensor;
                        CopyTensor(srcTensor, dstTensor, &inputArgType);

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
    if (src->IsBlock() && (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
    {
        auto blockSrc = dynamic_cast<BlockFunction*>(src.get());
        for (auto map : blockSrc->CompositeOutputsMap())
            compositeOutputsMap.insert(map);
        TraverseGraph(src->BlockRoot(), visited, compositeOutputsMap);
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

        if (src->OpName() == L"Clip")
        {
            if (src->Inputs().size() != 3)
            {
                LogicError("Clip should have 3 inputs.");
            }
            float minValue = src->Inputs()[1].Value()->AsScalar<float>();
            float maxValue = src->Inputs()[2].Value()->AsScalar<float>();
            node->AddAttribute("min", minValue);
            node->AddAttribute("max", maxValue);
        }
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

            node->AddAttribute(attributesMap[L"size"], depthRadius);
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
        else if ((src->OpName() == L"RandomDistribution") ||
            (src->OpName() == L"UniformRandom") || (src->OpName() == L"NormalRandom") ||
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
        else if ((src->OpName() == L"ReduceL1") || (src->OpName() == L"ReduceL2") || (src->OpName() == L"ReduceSumSquare"))
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
        else if (src->OpName() == L"TransposeAxes")
        {
            std::vector<Axis> permutation = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            // CNTK permutation attribute is argsorted. Shall redo argsort (undo) to get the original python/ONNX perm attribute.
            std::vector<int64_t> perm = AxesToINTSArgsortIncrementBatchAxis(permutation);
            node->AddAttribute(attributesMap[L"axisVec"], perm);
        }
        else if (src->OpName() == L"Reshape")
        {
            auto shapeVec = src->Output().Shape().Dimensions();
            std::vector<int> newShapeVec;
            size_t numInferredDimensions(0);
            for (const auto& axisSize : shapeVec)
            {
                if (axisSize == NDShape::InferredDimension) 
                {
                    numInferredDimensions++;
                    if (numInferredDimensions > 1)
                        LogicError("Reshape: Multiple InferredDimension not supported by ONNX.");
                    else
                        newShapeVec.push_back(-1);
                }
                else // REVIEW SPTIWARI: Should we fill 0 for FreeDimension here?
                    newShapeVec.push_back(static_cast<int>(axisSize));
            }
            // Always add a 1 to the shape for batch axis in ONNX tensors.
            newShapeVec.push_back(1);
            node->AddAttribute(attributesMap[L"shape"], ToINTS(newShapeVec));
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
            node->AddAttribute(attributesMap[L"beginIndexVec"], ToINTS(beginIndex));
            node->AddAttribute(attributesMap[L"endIndexVec"], ToINTS(endIndex));
        }
        if (src->OpName() == L"Pad")
        {
            auto value = (float)src->Attributes()[L"paddingConstantValue"].Value<double>();
            auto mode = (size_t)src->Attributes()[L"paddingMode"].Value<size_t>();
            auto head = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingHead"].Value<std::vector<DictionaryValue>>()));
            auto foot = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingFoot"].Value<std::vector<DictionaryValue>>()));
            head.insert(head.end(), foot.begin(), foot.end());
            string modeStr;
            if (mode == 0)
                modeStr = "constant";
            else if (mode == 1)
                modeStr = "reflect";
            else if (mode == 2)
                NOT_IMPLEMENTED
            else 
                LogicError("Invalid 'mode' value encountered in CNTK Pad node.");

            node->AddAttribute("mode", modeStr);
            node->AddAttribute("pads", head);
            if (mode == 0)
                node->AddAttribute("value", value);
        }
        else if (src->OpName() == L"DepthToSpace" || src->OpName() == L"SpaceToDepth")
        {
            size_t blockSize = src->Attributes()[L"blockSize"].Value<size_t>();
            node->AddAttribute("blocksize", static_cast<int64_t>(blockSize));
        }
        else if (src->OpName() == L"Softmax" || src->OpName() == L"LogSoftmax")
        {
            Axis axis = Axis(0);
            if (src->Attributes().Contains(L"axis"))
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            node->AddAttribute(attributesMap[L"axis"], (int64_t)ToIndex(axis));
        }
        else if (Operators::SupportBroadcast(src->OpName()))
        {
            NDShape broadcastShape;
            bool broadcast = false;
            int axis = 0;
            std::tie<NDShape, bool, int>(broadcastShape, broadcast, axis) =
                AdjustForBroadcastShape(src->Inputs()[0].Shape(), src->Inputs()[1].Shape(),
                    src->Inputs()[0].HasBatchAxis(), src->Inputs()[1].HasBatchAxis());

            node->AddAttribute("broadcast", (int64_t)(broadcast ? 1 : 0));
            if (broadcast && axis >= 0)
            {
                // +1 to take into consideration the batch aies
                node->AddAttribute("axis", (int64_t)axis);
            }
        }
        else if (src->OpName() == L"Times")
        {
            size_t outputRank = src->Attributes()[L"outputRank"].Value<size_t>();
            if (outputRank > 1)
                LogicError("Output rank other than 1 is not supported.");
        }
        else if (src->OpName() == L"ROIPooling")
        {
            auto roiOutputShape = (NDShape)src->Attributes()[L"roiOutputShape"].Value<NDShape>();
            auto ints = ToINTS(roiOutputShape, false);
            std::vector<float> pooled_shape = INTSToVecFloat(ints);

            auto spatialScale = (float)src->Attributes()[L"spatialScale"].Value<double>();

            node->AddAttribute("pooled_shape", pooled_shape);
            node->AddAttribute("spatial_scale", spatialScale);
        }
        else if (src->OpName() == L"HardSigmoid")
        {
            float alpha = (float)src->Attributes()[L"alpha"].Value<float>();
            float beta = (float)src->Attributes()[L"beta"].Value<float>();
            node->AddAttribute("alpha", alpha);
            node->AddAttribute("beta", beta);
        }
        else if (src->OpName() == L"Flatten")
        {
            Axis axis(0);
            if (src->Attributes().Contains(L"axis"))
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());

            node->AddAttribute(attributesMap[L"axis"], (int64_t)ToIndex(axis));
        }
        else if (src->OpName() == L"Squeeze")
        {
            std::vector<Axis> axes;
            if (src->Attributes().Contains(L"axisVec"))
            {
                axes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                axes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));
            }
            node->AddAttribute("axes", ToINTS(axes));
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
            node->AddAttribute("dilations", ToINTS(dilations));
            node->AddAttribute("group", (int64_t)1);

            PutAutopadOrPadAttrInNode(node, autoPadding, kernelShape);

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
            if (strides.Rank() < kernelShape.Rank())
            {
                strides = strides.AppendShape(NDShape(std::vector<size_t>(kernelShape.Rank() - strides.Rank(), 1)));
            }
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            PutAutopadOrPadAttrInNode(node, autoPadding, kernelShape);
        }
        else if (src->OpName() == L"ReduceElements")
        {
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
            const AttributesMapping& attributeMap = Operators::FindAttributeMap(src->OpName(), cntkAttributeOpName);

            auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
            std::vector<Axis> reductionAxes;
            if (src->Attributes().Contains(L"axisVec"))
                reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            else if (src->Attributes().Contains(L"axis"))
                reductionAxes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));

            node->AddAttribute(attributeMap.map.at(L"reductionKeepDimensions"), keepReducedDimensions);
            node->AddAttribute("axes", ToINTS(reductionAxes));
        }
    }
}

void CNTKToONNXHelper::PutAutopadOrPadAttrInNode(ONNXIR::Node* node, const std::vector<bool>& autoPadding, const NDShape& kernelShape)
{
    // Based on the CNTK node choose to put either the auto_pad or pads attribute in the ONNX node.

    // ONNX spec says that if 'pads' attributes is specified then 'VALID'
    // for 'auto_pad' is implied, and 'auto_pad' attribute should not (must not)
    // be explicitly specified/set.
    bool isExplicitPadValueNeeded = std::find(autoPadding.begin(), autoPadding.end(), false) != autoPadding.end();
    if (isExplicitPadValueNeeded)
    {
        auto padsValueVectorsForONNX = GetONNXPadsAttributeFromCNTKNode(autoPadding, kernelShape);
        auto lowerPads = ToINTS(padsValueVectorsForONNX.first);
        auto upperPads = ToINTS(padsValueVectorsForONNX.second);
        lowerPads.insert(lowerPads.end(), upperPads.cbegin(), upperPads.cend());
        node->AddAttribute("pads", lowerPads);
    }
    else
        node->AddAttribute("auto_pad", "SAME_UPPER");
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

std::pair<std::vector<int>, std::vector<int> > CNTKToONNXHelper::GetONNXPadsAttributeFromCNTKNode(const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape)
{
    // Figure out the value for 'pads' ONNX attribute.

    // Only one of the two ONNX conv attributes, auto_pad and pads, can be specified in the saved model. 
    // It is assumed at this point that we need an explicit padding vector, pads, and not the auto_pad attribute. 
    // The 'auto_pad' atrribute is implied to be 'VALID' by ONNX specification if the 'pads' attribute is specified
    // (padsValueVector) for the dimensions for which cntkAutoPadding is true.
    assert(kernelShape.Rank() == cntkAutoPadding.size());
    std::vector<int> padsValueVectorLower(kernelShape.Rank(), 0);
    std::vector<int> padsValueVectorUpper(kernelShape.Rank(), 0);
    for (size_t i = 0; i < cntkAutoPadding.size(); ++i)
    {
        if (!cntkAutoPadding[i]) continue;
        auto q = kernelShape[i] / 2;
        padsValueVectorLower[i] = kernelShape[i] % 2 ? q : (q - 1);
        padsValueVectorUpper[i] = q;
    }
    return std::make_pair(padsValueVectorLower, padsValueVectorUpper);
}
