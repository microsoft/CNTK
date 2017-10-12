//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "ONNXToCNTK.h"
#include "proto/onnx/core/graph.h"
#include "Utils.h"
#include "Operators.h"
#include <algorithm>
#include <iostream>

using namespace ONNXIR;
using namespace CNTK;
using namespace CNTK::ONNX;

namespace CNTK
{

typedef std::unordered_map<const Node *, FunctionPtr> ONNXToCNTKMap;
class ONNXToCNTKHelper
{
public:
    //
    // Convert an ONNX graph to a CNTK graph (Function).
    // 
    static FunctionPtr FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
        const Graph* graph, const DeviceDescriptor& computeDevice);

private:

    static FunctionPtr CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
        const DeviceDescriptor& computeDevice);
    static Constant CreateConstant(const Node *node, const DeviceDescriptor& computeDevice);
    static Constant CreateConstant(const ONNXIR::TensorProto &valueProto, const std::string &nodeName,
        const DeviceDescriptor& computeDevice);
    static Variable CreateLeafVariableOrConstant(const NodeArg *nodeArg, const Graph *graph,
        const DeviceDescriptor& computeDevice);
    static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs);

    static std::vector<Axis> AttributeProtoToAxes(const AttributeProto &attributeProto);
    static ONNXIR::TypeProto FromINTS(const std::vector<int64_t> &shape);
    static NDShape FromTypeProto(const ONNXIR::TypeProto& tensorShape);
    static NDShape FromTensorShapeProto(const ONNXIR::TypeProto::TensorShapeProto& tensorShape);
    static std::vector<bool> FromTypeProtoAsBool(const ONNXIR::TypeProto& tensorShape);
    static DataType FromONNXType(ONNXIR::TypeProto type);

    static NodeAttributes::const_iterator FindAttributeIterator(const Node *node,
        const string &attributeName, bool required);
    static bool HasNamedAttribute(const Node *node, const string &attributeName);

    // Regarding following pairs of attribute getters. One of each pair takes not default value. It is
    // for attributes for which CNTK operators do not have a default value. In this case
    // we raise an error if an attribute is missing.
    // For attributes that CNTK operators do have default values, we tolerate
    // missing attributes by using CNTK default values.
    // We could require all ONNX attributes to be present and log an error if one is missing.
    // However this is the responsibility of onnx core. We want to avoid runtime error and assume
    // onnx core has already checked for this.
    static std::vector<Axis> GetNamedAttributeAsAxis(const Node *node, const string &attributeName);
    static std::vector<Axis> GetNamedAttributeAsAxis(const Node *node, const string &attributeName,
        const std::vector<Axis> &defaultAxes);

    static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName, 
        bool hasBatchAxis);
    static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName, 
        bool hasBatchAxis, const NDShape defaultShape);

    static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName);
    static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName,
        const std::vector<bool> &defaultShape);

    static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName);
    static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName, size_t defaultValue);

    static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName);
    static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName, float defaultValue);

    static std::vector<bool> GetAutoPaddingWithSymmetryConversion(const Node *node, int strideRank,
        const string &onnxAutoPaddingAttributeName, const std::vector<bool> &defaultValue);
    static void AdjustAutoPaddingAndStrideForCNTKSpecialCases(const Variable &operand, 
        std::vector<bool> &autoPadding, NDShape &strides);
    static NDShape ReverseShape(const NDShape &shape);

    static std::pair<Variable, Variable> BroadcastElementWiseInput(const Node *node, 
        const Variable &input0, const Variable &input1);
};

}

std::vector<Axis> ONNXToCNTKHelper::AttributeProtoToAxes(const AttributeProto &attributeProto)
{
    std::vector<int64_t> ints(attributeProto.ints().begin(), attributeProto.ints().end());
    std::vector<Axis> axes;
    for (std::vector<int64_t>::const_iterator it = ints.begin(); it != ints.end(); it++)
    {
        axes.push_back(Axis((int)(*it)));
    }
    return axes;
}

ONNXIR::TypeProto ONNXToCNTKHelper::FromINTS(const std::vector<int64_t> &shape)
{
    ONNXIR::TypeProto newShape;

    for (std::vector<int64_t>::const_iterator it = shape.begin(); it != shape.end(); it++)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(*it);
    }

    return newShape;
}

NDShape ONNXToCNTKHelper::FromTypeProto(const ONNXIR::TypeProto& tensorShape)
{
    return FromTensorShapeProto(tensorShape.tensor_type().shape());
}

NDShape ONNXToCNTKHelper::FromTensorShapeProto(const ONNXIR::TypeProto::TensorShapeProto& tensorShape)
{
    std::vector<size_t> dimensions;
    for (int index = 0; index < tensorShape.dim_size(); index++)
        dimensions.push_back(tensorShape.dim(index).dim_value());

    // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
    return ReverseShape(dimensions);
}

NDShape ONNXToCNTKHelper::ReverseShape(const NDShape &shape)
{
    std::vector<size_t> dimensions;
    for (int index = shape.Rank() - 1; index >= 0; index--)
    {
        dimensions.push_back(shape[index]);
    }
    return dimensions;
}

std::vector<bool> ONNXToCNTKHelper::FromTypeProtoAsBool(const ONNXIR::TypeProto& tensorShape)
{
    std::vector<bool> dimensions;
    for (int index = 0; index < tensorShape.tensor_type().shape().dim_size(); index++)
        dimensions.push_back(tensorShape.tensor_type().shape().dim(index).dim_value() == 0 ? false : true);

    // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
    std::reverse(dimensions.begin(), dimensions.end());
    return dimensions;
}

DataType ONNXToCNTKHelper::FromONNXType(ONNXIR::TypeProto type)
{
    switch (type.tensor_type().elem_type())
    {
    case ONNXIR::TensorProto_DataType_FLOAT:
        return DataType::Float;
    case ONNXIR::TensorProto_DataType_DOUBLE:
        return DataType::Double;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

// helpers copied from Lotus (Converter.cc). These functions will eventually 
// be replaced with functionalities of onnx core.
bool IsLittleEndianOrder()
{
    int n = 1;
    return (*(char *)&n == 1);
}

#pragma warning(disable : 4244) 

float UnpackFloat(const char *buf, int i)
{
    float temp = 0;
    if (IsLittleEndianOrder())
    {
        memcpy((void*)&temp, (void*)buf, sizeof(char) * 4);
    }
    else
    {
        temp = ((buf[0] << 24) |
            (buf[1] << 16) |
            (buf[2] << 8) |
            buf[3]);
    }
    return temp;
}

void RetrieveRawDataAsFloat(const ONNXIR::TensorProto &valueProto)
{
    if (!valueProto.float_data().empty())
        return;

    auto raw_data = valueProto.raw_data();
    ONNXIR::TensorProto &mutableProto = const_cast<ONNXIR::TensorProto &>(valueProto);
    ::google::protobuf::RepeatedField< float >* p_mutable_float_data = mutableProto.mutable_float_data();
    if (!raw_data.empty())
    {
        auto buff = raw_data.c_str();
        for (int i = 0; i < raw_data.size(); i += 4)
        {
            float v = UnpackFloat(buff + i, i);
            p_mutable_float_data->Add(v);
        }
    }
}

double UnpackDouble(const char *buf, int i)
{
    double temp = 0;

    if (IsLittleEndianOrder())
    {
        memcpy((void*)&temp, (void*)buf, sizeof(char) * 8);
    }
    else
    {
        // this is temperal code that will soon be replaced by onnx core.
        NOT_IMPLEMENTED;
    }
    return temp;
}

void RetrieveRawDataAsDouble(const ONNXIR::TensorProto &valueProto)
{
    if (!valueProto.double_data().empty())
        return;

    auto raw_data = valueProto.raw_data();
    ONNXIR::TensorProto &mutableProto = const_cast<ONNXIR::TensorProto &>(valueProto);
    ::google::protobuf::RepeatedField< double >* p_mutable_double_data = mutableProto.mutable_double_data();
    if (!raw_data.empty())
    {
        auto buff = raw_data.c_str();
        for (int i = 0; i < raw_data.size(); i += 4)
        {
            double v = UnpackDouble(buff + i, i);
            p_mutable_double_data->Add(v);
        }
    }
}

Constant ONNXToCNTKHelper::CreateConstant(const Node *node, const DeviceDescriptor& computeDevice)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
    const ONNXIR::TensorProto valueProto = itValue->second.t();

    return CreateConstant(valueProto, node->Name(), computeDevice);
}

Constant ONNXToCNTKHelper::CreateConstant(const ONNXIR::TensorProto &valueProto, const std::string &nodeName,
    const DeviceDescriptor& computeDevice)
{
    auto dataType = valueProto.data_type();

    NDShape shape(std::vector<size_t>(valueProto.dims().begin(), valueProto.dims().end()));

    // the following code is to revert CNTKToONNXHelper::ToTensorShape.to restore a CNTK NDArray
    NDShape reversedShape = ReverseShape(shape);

    auto totalSize = shape.TotalSize();

    switch (dataType)
    {
    case TensorProto_DataType_FLOAT:
    {
        float *data = new float[totalSize];
        if (valueProto.float_data().empty())
        {
            RetrieveRawDataAsFloat(valueProto);
        }

        if (shape.Rank() <= 2)
        {
            for (size_t index = 0; index < totalSize; index++)
            {
                data[index] = valueProto.float_data()[index];
            }
        }
        else
        {
            int outputChannels = shape[0], inputChanndels = shape[1];
            NDShape channelKernelShape(std::vector<size_t>(shape.Dimensions().begin() + 2, shape.Dimensions().end()));
            NDShape channelReversedShape = ReverseShape(channelKernelShape);
            int channelKernelSize = channelKernelShape.TotalSize();
            for (int oC = 0; oC < outputChannels; oC++)
            {
                for (int iC = 0; iC < inputChanndels; iC++)
                {
                    int channelIndex = (oC * inputChanndels + iC);
                    for (int pixel = 0; pixel < channelKernelSize; pixel++)
                    {
                        data[channelIndex * channelKernelSize + pixel] = 
                            valueProto.float_data()[channelIndex * channelKernelSize + pixel];
                    }
                }
            }
        }

        NDArrayViewPtr dstFinal(new NDArrayView(DataType::Float, reversedShape, &data[0], 
            totalSize * sizeof(float), computeDevice.CPUDevice()));
                
        if (computeDevice.Type() == DeviceKind::CPU)
        {
            Constant constantVariable(dstFinal, ToWString(nodeName));
            return constantVariable;
        }
        else
        {
            // this is the way to load values into GPU: 
            // Create a GPU NDArrayView and CopyFrom a CPU NDArrayView that holding the data.
            NDArrayViewPtr dstFinalGPU(new NDArrayView(DataType::Float, StorageFormat::Dense, reversedShape, computeDevice));
            dstFinalGPU->CopyFrom(*dstFinal);
            Constant constantVariable(dstFinalGPU, ToWString(nodeName));
            return constantVariable;
        }
    }
    break;
    case TensorProto_DataType_DOUBLE:
    {
        // TODO: refactore commom code for float and double
        double *data = new double[totalSize];
        if (valueProto.double_data().empty())
        {
            RetrieveRawDataAsDouble(valueProto);
        }

        if (shape.Rank() <= 2)
        {
            for (size_t index = 0; index < totalSize; index++)
            {
                data[index] = valueProto.double_data()[index];
            }
        }
        else
        {
            int outputChannels = shape[0], inputChanndels = shape[1];
            NDShape channelKernelShape(std::vector<size_t>(shape.Dimensions().begin() + 2, shape.Dimensions().end()));
            NDShape channelReversedShape = ReverseShape(channelKernelShape);
            int channelKernelSize = channelKernelShape.TotalSize();
            for (int oC = 0; oC < outputChannels; oC++)
            {
                for (int iC = 0; iC < inputChanndels; iC++)
                {
                    int channelIndex = (oC * inputChanndels + iC);
                    for (int pixel = 0; pixel < channelKernelSize; pixel++)
                    {
                        data[channelIndex * channelKernelSize + pixel] =
                            valueProto.double_data()[channelIndex * channelKernelSize + pixel];
                    }
                }
            }
        }

        NDArrayViewPtr dstFinal(new NDArrayView(DataType::Double, reversedShape, &data[0],
            totalSize * sizeof(double), computeDevice.CPUDevice()));

        if (computeDevice.Type() == DeviceKind::CPU)
        {
            Constant constantVariable(dstFinal, ToWString(nodeName));
            return constantVariable;
        }
        else
        {
            // this is the way to load values into GPU: 
            // Create a GPU NDArrayView and CopyFrom a CPU NDArrayView that holding the data.
            NDArrayViewPtr dstFinalGPU(new NDArrayView(DataType::Double, StorageFormat::Dense, reversedShape, computeDevice));
            dstFinalGPU->CopyFrom(*dstFinal);
            Constant constantVariable(dstFinalGPU, ToWString(nodeName));
            return constantVariable;
        }
    }
    break;
    default:
        NOT_IMPLEMENTED;
    }
}

Variable ONNXToCNTKHelper::CreateLeafVariableOrConstant(const NodeArg *nodeArg, const Graph* graph,
    const DeviceDescriptor& computeDevice)
{
    std::string nodeName = nodeArg->Name();

    ONNXIR::TensorProto valueProto;
    if (graph->GetInitialTensor(nodeName, valueProto))
    {
        return CreateConstant(valueProto, nodeName, computeDevice);
    }

    auto dataType = FromONNXType(nodeArg->ToProto().type());
    auto shapeProto = nodeArg->Shape();

    // in CNTK constants are created as Node (not a leaf) with values. 
    // in ONNX constants may also be a leaf with values saved in initializer
    // here we know it is not an ONNX constant so reshape the variable to trim off last dim;
    NDShape shape = FromTensorShapeProto(*shapeProto);
    // if (shape[shape.Rank() - 1] == NDShape::FreeDimension)
    {
        shape = shape.SubShape(0, shape.Rank() - 1);
    }

    switch (dataType)
    {
    case DataType::Float:
    {
        return InputVariable(shape, DataType::Float, ToWString(nodeArg->Name()));
    }
    case DataType::Double:
    {
        return InputVariable(shape, DataType::Double, ToWString(nodeArg->Name()));
    }
    default:
        NOT_IMPLEMENTED;
    }
}

namespace CNTK
{
void CheckForAxes(const string &nodeName, const std::vector<Axis> &axes, int requiredAxes)
{
    if (axes.size() != requiredAxes)
        LogicError("%s has %d input axis/axes. It should has %d .", nodeName.c_str(), (int)axes.size(), requiredAxes);
}
}

bool ONNXToCNTKHelper::HasNamedAttribute(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    return itValue != node->GetAttributes().end();
}

NodeAttributes::const_iterator ONNXToCNTKHelper::FindAttributeIterator(const Node *node,
    const string &attributeName, bool required)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find(attributeName);
    if (itValue == node->GetAttributes().end())
    {
        if (required)
        {
            LogicError("Node %s operator %s is missing attribute %s.", 
                node->Name().c_str(), node->OpType().c_str(), attributeName.c_str());
        }
    }
    return itValue;
}

std::vector<Axis> ONNXToCNTKHelper::GetNamedAttributeAsAxis(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    return AttributeProtoToAxes(itValue->second);
}

std::vector<Axis> ONNXToCNTKHelper::GetNamedAttributeAsAxis(const Node *node, const string &attributeName,
    const std::vector<Axis> &defaultAxes)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    if (itValue == node->GetAttributes().end())
    {
        return defaultAxes;
    }

    return AttributeProtoToAxes(itValue->second);
}

NDShape ONNXToCNTKHelper::GetNamedAttributeAsShape(const Node *node, const string &attributeName, bool hasBatchAxis)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    ::google::protobuf::RepeatedField<::google::protobuf::int64>::const_iterator itBegin =
        attributeProto.ints().begin();
    if (hasBatchAxis)
        itBegin++;
    std::vector<int64_t> shape(itBegin, attributeProto.ints().end());
    return FromTypeProto(FromINTS(shape));
}

NDShape ONNXToCNTKHelper::GetNamedAttributeAsShape(const Node *node, const string &attributeName,
    bool hasBatchAxis, const NDShape defaultShape)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultShape;
    }

    const AttributeProto &attributeProto = itValue->second;
    ::google::protobuf::RepeatedField<::google::protobuf::int64>::const_iterator itBegin =
        attributeProto.ints().begin();
    if (hasBatchAxis)
        itBegin++;
    std::vector<int64_t> shape(itBegin, attributeProto.ints().end());
    return FromTypeProto(FromINTS(shape));
}

std::vector<bool> ONNXToCNTKHelper::GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> shape(attributeProto.ints().begin(), attributeProto.ints().end());
    return FromTypeProtoAsBool(FromINTS(shape));
}

std::vector<bool> ONNXToCNTKHelper::GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName,
    const std::vector<bool> &defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> shape(attributeProto.ints().begin(), attributeProto.ints().end());
    return FromTypeProtoAsBool(FromINTS(shape));
}

size_t ONNXToCNTKHelper::GetNamedAttributeAsInt64(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    int64_t size64 = attributeProto.i();
    return size64;
}

size_t ONNXToCNTKHelper::GetNamedAttributeAsInt64(const Node *node, const string &attributeName,
    size_t defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    int64_t size64 = attributeProto.i();
    return size64;
}

float ONNXToCNTKHelper::GetNamedAttributeAsFloat(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    float floatValue = attributeProto.f();
    return floatValue;
}

float ONNXToCNTKHelper::GetNamedAttributeAsFloat(const Node *node, const string &attributeName, float defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    float floatValue = attributeProto.f();
    return floatValue;
}

namespace CNTK
{
    static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
    {
        if (function->Inputs().size() == 0)
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
            return;
        }

        for (auto input : function->Inputs())
        {
            cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
                "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString()) << std::endl;
        }

        for (auto input : function->Inputs())
        {
            if (input.Owner() != NULL)
            {
                FunctionPtr f = input.Owner();
                PrintGraph(f, spaces + 4, useName);
            }
        }
    }
}

std::pair<Variable, Variable> ONNXToCNTKHelper::BroadcastElementWiseInput(
    const Node *node, const Variable &input0, const Variable &input1)
{
    auto shape0 = input0.Shape();
    auto shape1 = input1.Shape();
    NDShape newShape;

    // set doThisAfterBoardcastInCNTKFix = true when loading FB models
    bool doThisAfterBoardcastInCNTKFix = true;
    if (doThisAfterBoardcastInCNTKFix)
    {
        if (bool broadcast = GetNamedAttributeAsInt64(node, "broadcast", 0) == 1 && HasNamedAttribute(node, "axis"))
        {
            if ((shape0.Rank() == 1) && ((shape1.Rank() > 1)))
            {
                int index = (int)GetNamedAttributeAsInt64(node, "axis") - 1 + (shape1.Rank() - 1);
                for (int i = 0; i < shape1.Rank(); i++)
                {
                    if (i != index)
                        newShape = newShape.AppendShape({ 1 });
                    else
                        newShape = newShape.AppendShape(shape0);
                }
                return{ Reshape(input0, newShape), input1 };
            }
            else if ((shape1.Rank() == 1) && ((shape0.Rank() > 1)))
            {
                int index = (int)GetNamedAttributeAsInt64(node, "axis") - 1 + (shape0.Rank() - 1);
                for (int i = 0; i < shape0.Rank(); i++)
                {
                    if (i != index)
                        newShape = newShape.AppendShape({ 1 });
                    else
                        newShape = newShape.AppendShape(shape1);
                }
                return{ input0, Reshape(input1, newShape) };
            }
        }
    }

    return { input0 , input1 };
}

std::vector<bool> ONNXToCNTKHelper::GetAutoPaddingWithSymmetryConversion(const Node *node, int strideRank,
    const string &onnxAutoPaddingAttributeName, const std::vector<bool> &defaultValue)
{
    std::vector<bool> autoPadding = GetNamedAttributeAsShapeBool(node, onnxAutoPaddingAttributeName, defaultValue);

    // This may happen if the node has asymetric padding. CNTK only support symetric padding so we pick on side
    if (autoPadding.size() == 2 * strideRank)
    {
        std::vector<bool> newPadding;
        for (size_t index = 0; index < autoPadding.size() / 2; ++index)
            newPadding.push_back(autoPadding[index]);
        autoPadding = newPadding;
    }

    return autoPadding;
}

void ONNXToCNTKHelper::AdjustAutoPaddingAndStrideForCNTKSpecialCases(const Variable &operand, std::vector<bool> &autoPadding, NDShape &strides)
{
    if ((operand.Shape().Rank() == (1 + autoPadding.size())))
    {
        autoPadding.push_back(false);
    }

    if ((operand.Shape().Rank() == (1 + strides.Rank())))
    {
        strides = strides.AppendShape({ 1 });
    }
}

FunctionPtr ONNXToCNTKHelper::CreateFunction(const Node *node, const std::vector<Variable> &inputs)
{
    string onnxOpName = node->OpType();
    
    if (onnxOpName == "FC")
    {
        // TODO: this is experimental code to load Facebook Caffe models. 
        // "FC" is not in ONNX standard. Two cases need to be handled with 
        // this type of Caffe model. 
        // 1. Make trailing dimensions of operand 1 matches the heading dimensions of operant 2.
        //  For example, with shape [1, dim0, dim1] * [dim2, dim3], we need to reshape 
        //  first operand to [1, dim0 * dim1] In this case dim0 * dim1 has to be equal to dim2.
        // 2. Broadcase bias if needed.
        Variable input0 = inputs[0], input1 = inputs[1];
        input0 = Reshape(input0, {1, NDShape::InferredDimension});

        FunctionPtr cntkFunction = Reshape(Times(input0, input1, ToWString(node->Name())), { NDShape::InferredDimension });
        cntkFunction = Plus(cntkFunction, inputs[2], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sum")
    {
        // TODO: this is experimental code to load Facebook Caffe models.
        // Need to investigate cases here operands' dimensions do not match.
        FunctionPtr cntkFunction = inputs[0];
        for (int i = 1; i < inputs.size(); i++)
        {
            cntkFunction = Plus(cntkFunction, inputs[i]);
        }
        cntkFunction->SetName(ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "LRN")
    {
        // TODO: this is experimental code to load Facebook Caffe models. 
        // Operators are added so hopefully there is not further work needed.
        size_t depthRadius = GetNamedAttributeAsInt64(node, "size");
        double bias = GetNamedAttributeAsFloat(node, "bias");
        double alpha = GetNamedAttributeAsFloat(node, "alpha");
        double beta = GetNamedAttributeAsFloat(node, "beta");
        FunctionPtr cntkFunction = LocalResponseNormalization(inputs[0],
            depthRadius, bias, alpha, beta, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "AveragePool" || onnxOpName == "MaxPool")
    {
        NDShape poolingWindowShape = GetNamedAttributeAsShape(node, "kernel_shape", false);
        NDShape strides = GetNamedAttributeAsShape(node, "strides", false);
        std::vector<bool> autoPadding = GetAutoPaddingWithSymmetryConversion(node, strides.Rank(), "pads", { false });

        AdjustAutoPaddingAndStrideForCNTKSpecialCases(inputs[0], autoPadding, strides);

        bool ceilOutDim = false;
        bool includePad = false;
        FunctionPtr cntkFunction = Pooling(inputs[0],
            onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
            poolingWindowShape, strides, autoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "GlobalAveragePool" || onnxOpName == "GlobalMaxPool")
    {
        NDShape strides = { 1 };
        std::vector<bool> autoPadding = { false };
        bool ceilOutDim = false;
        bool includePad = false;

        FunctionPtr cntkFunction = Pooling(inputs[0],
            onnxOpName == "GlobalAveragePool" ? PoolingType::Average : PoolingType::Max, 
            NDShape::Unknown(), strides, autoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Conv")
    {
        NDShape strides = GetNamedAttributeAsShape(node, "strides", false);
        NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, { 1 });
        std::vector<bool> autoPadding = GetAutoPaddingWithSymmetryConversion(node, strides.Rank(), "pads", { false });

        size_t groups = GetNamedAttributeAsInt64(node, "group", 1);

        // TODO: avoid hardcoded values
        std::vector<bool> sharing({ true });
        size_t reductionRank = 1;
        size_t maxTempMemSizeInSamples = 0;
        
        Variable convolutionMap = inputs[1];
        Variable operand = inputs[0];
        AdjustAutoPaddingAndStrideForCNTKSpecialCases(operand, autoPadding, strides);

        FunctionPtr cntkFunction = Convolution(
            convolutionMap,
            operand,
            strides,
            sharing,
            autoPadding,
            dilation,
            reductionRank,
            groups,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ConvTranspose")
    {
        NDShape strides = GetNamedAttributeAsShape(node, "strides", false);
        NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, { 1 } );
        std::vector<bool> autoPadding = GetAutoPaddingWithSymmetryConversion(node, strides.Rank(), "pads", { false });

        NDShape outputShape = GetNamedAttributeAsShape(node, "output_shape", true);

        // TODO: avoid hardcoded values
        std::vector<bool> sharing({ true });
        size_t reductionRank = 1;
        size_t maxTempMemSizeInSamples = 0;

        const Variable& convolutionMap = inputs[1];
        const Variable& operand = inputs[0];
        AdjustAutoPaddingAndStrideForCNTKSpecialCases(operand, autoPadding, strides);

        // TODO: are we sure that convolutionMap and operand are in inputs[1], inputs[0] order?
        FunctionPtr cntkFunction = ConvolutionTranspose(
            convolutionMap,
            operand,
            strides,
            sharing,
            autoPadding,
            outputShape,
            dilation,
            reductionRank,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "BatchNormalization" || onnxOpName == "SpatialBN")
    {
        // TODO: implement this right once ready.
        const Variable& operand = inputs[0];
        const Variable& scale = inputs[1];
        const Variable& bias = inputs[2];
        const Variable& runningMean = inputs[3];
        const Variable& runningInvStd = inputs[4];
        const Variable& runningCount = Constant::Scalar(0.0F);

        bool spatial = onnxOpName == "SpatialBN" || GetNamedAttributeAsInt64(node, "spatial") != 0;

        double normalizationTimeConstant = 0.0;
        if (HasNamedAttribute(node, "momentum"))
        {
            float momentum = GetNamedAttributeAsFloat(node, "momentum");
            if ((momentum > (1.0f - std::numeric_limits<float>::epsilon())) &&
                (momentum < (1.0f + std::numeric_limits<float>::epsilon())))
                normalizationTimeConstant = INFINITY;
            else if (momentum > 0.0f)
                normalizationTimeConstant = -48.0f / log1p(momentum - 1.0f);
            else
                normalizationTimeConstant = 0.0;
        }

        // TODO: avoid hardcoded values
        double blendTimeConstant = 0;
        double epsilon = 0.00001;
        bool useCuDNNEngine = true;
        FunctionPtr cntkFunction = BatchNormalization(operand,
            scale,
            bias,
            runningMean,
            runningInvStd,
            runningCount,
            spatial,
            normalizationTimeConstant,
            blendTimeConstant,
            epsilon,
            useCuDNNEngine,
            ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Dropout")
    {
        const Variable& operand = inputs[0];
        double dropoutRate = GetNamedAttributeAsFloat(node, "ratio");

        unsigned long seed = SentinelValueForAutoSelectRandomSeed;
        FunctionPtr cntkFunction = Dropout(operand, dropoutRate, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniform")
    {
        const NDShape &shape = GetNamedAttributeAsShape(node, "shape", false);

        // ONNX only has float type for random generators
        DataType dataType = DataType::Float;

        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = UniformRandom(shape, dataType, low, high, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormal")
    {
        const NDShape& shape = GetNamedAttributeAsShape(node, "shape", false);
        DataType dataType = DataType::Float;
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = NormalRandom(shape, dataType, mean, scale, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniformLike")
    {
        const Variable& operand = inputs[0];
        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = UniformRandomLike(operand, low, high, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormalLike")
    {
        const Variable& operand = inputs[0];
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = NormalRandomLike(operand, mean, scale, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Add")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = Plus(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sub")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = Minus(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mul")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementTimes(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Div")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementDivide(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Neg")
    {
        FunctionPtr cntkFunction = Negate(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Abs")
    {
        FunctionPtr cntkFunction = Abs(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reciprocal")
    {
        FunctionPtr cntkFunction = Reciprocal(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Floor")
    {
        FunctionPtr cntkFunction = Floor(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Ceil")
    {
        FunctionPtr cntkFunction = Ceil(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sqrt")
    {
        FunctionPtr cntkFunction = Sqrt(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Relu")
    {
        FunctionPtr cntkFunction = ReLU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "LeakyRelu")
    {
        FunctionPtr cntkFunction = LeakyReLU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Selu")
    {
        // TODO: get from node's attributes
        double scale = 1;
        double alpha = 0.5;
        FunctionPtr cntkFunction = SELU(inputs[0], scale, alpha, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Elu")
    {
        FunctionPtr cntkFunction = ELU(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Exp")
    {
        FunctionPtr cntkFunction = Exp(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Log")
    {
        FunctionPtr cntkFunction = Log(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Tanh")
    {
        FunctionPtr cntkFunction = Tanh(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Pow")
    {
        return nullptr;
    }
    else if (onnxOpName == "Dot")
    {
        FunctionPtr cntkFunction = Times(inputs[1], inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "PRelu")
    {
        FunctionPtr cntkFunction = PReLU(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sigmoid")
    {
        FunctionPtr cntkFunction = Sigmoid(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Max")
    {
        FunctionPtr cntkFunction = ElementMax(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Min")
    {
        FunctionPtr cntkFunction = ElementMin(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sum")
    {
        // not specified in Operators.cpp
        return nullptr;
    }
    else if (onnxOpName == "Softmax")
    {
        FunctionPtr cntkFunction = Softmax(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMax")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMax(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMin")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMin(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSum")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMean")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceMean(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceProd")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceProd(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceLogSumExp")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMax")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = Argmax(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMin")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = Argmin(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reshape")
    {
        NDShape newShape = GetNamedAttributeAsShape(node, "shape", false);
        if (inputs[0].HasBatchAxis())
            newShape = newShape.SubShape(0, newShape.Rank() - 1);

        FunctionPtr cntkFunction = Reshape(inputs[0], newShape, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Concat")
    {
        std::vector<Axis> axes;
        if (HasNamedAttribute(node, "axes"))
        {
            axes = GetNamedAttributeAsAxis(node, "axes");
        }
        else
        {
            // TODO: Make sure they all have the same rank.
            axes.push_back(Axis(inputs[0].Shape().Rank() - 1));
        }

        CheckForAxes(node->Name(), axes, 1);
        FunctionPtr cntkFunction = Splice(inputs, axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    // { L"", "Split)
    else if (onnxOpName == "Slice")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxis(node, "axes");
        std::vector<int> beginIndex;
        std::vector<int> endIndex;
        FunctionPtr cntkFunction = Slice(inputs[0], axes, beginIndex, endIndex, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Transpose")
    {
        FunctionPtr cntkFunction = Transpose(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Gather")
    {
        FunctionPtr cntkFunction = GatherOp(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else
    {
        // throw 
        return nullptr;
    }
}

FunctionPtr ONNXToCNTKHelper::FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
    const Graph* graph, const DeviceDescriptor& computeDevice)
{
    ONNXToCNTKMap::iterator itONNXToCNTKMap = constructedNodeMap.find(node);
    if (itONNXToCNTKMap != constructedNodeMap.end())
    {
        return itONNXToCNTKMap->second;
    }

    std::vector<Variable> inputs;
    const std::vector<NodeArg>& inputDefs = node->InputDefs();
    for (std::vector<NodeArg>::const_iterator it = inputDefs.begin(); it != inputDefs.end(); ++it)
    {
        NodeArg *nodeArg = const_cast<NodeArg *>(&(*it));
        const Node::EdgeEnd* inputEdgeSrcEnd = nullptr;
        Node *cNode = const_cast<Node *>(node);
        if (cNode->InputEdgeSrcEnd(nodeArg, &inputEdgeSrcEnd))
        {
            const Node* inputNode = inputEdgeSrcEnd->GetNode();
            ONNXToCNTKMap::iterator itNodeMap = constructedNodeMap.find(const_cast<Node *>(inputNode));
            if (itNodeMap != constructedNodeMap.end())
            {
                inputs.push_back(itNodeMap->second);
            }
            else
            {
                FunctionPtr input = FromONNXNode(inputNode, constructedNodeMap, graph, computeDevice);
                inputs.push_back(input);
            }
        }
        else
        {
            Variable inputVariable = CreateLeafVariableOrConstant(nodeArg, graph, computeDevice);
            inputs.push_back(inputVariable);
        }
    }

    FunctionPtr cntkFunction = CreateCNTKNode(node, inputs, computeDevice);
    constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, cntkFunction));
    return cntkFunction;
}


FunctionPtr ONNXToCNTKHelper::CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
    const DeviceDescriptor& computeDevice)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "NoOp")
    {
        // TODO: this is for sink or source - what type of variable for it?
        NDShape shape;
        Constant constantVariable(shape, 0.5F, computeDevice, ToWString(node->Name()));
        return constantVariable;
    }
    else if (onnxOpName == "Constant")
    {
        Constant constant = CreateConstant(node, computeDevice);
        return constant;
    }
    else
    {
        return CreateFunction(node, inputs);
    }
}

FunctionPtr ONNXToCNTK::CreateGraph(ONNXIR::Graph* src, const DeviceDescriptor& computeDevice)
{
    FunctionPtr cntkModel;

    // To use depth-first-traversal, keeps a collection of visited nodes.
    ONNXToCNTKMap constructedFunctions;
    for (Graph::NodeIterator it = src->Nodes_begin(); it != src->Nodes_end(); ++it)
    {
        const Node *node = *it;

        if (constructedFunctions.find(node) == constructedFunctions.end())
        {
            FunctionPtr cntkNode = ONNXToCNTKHelper::FromONNXNode(node, constructedFunctions, src, computeDevice);
        }
    }

    // ONNX puts all outputs in an graph as input to the "_Graph_Sink" node.
    ONNXToCNTKMap::iterator itNodeFn = std::find_if(constructedFunctions.begin(), constructedFunctions.end(),
        [](ONNXToCNTKMap::value_type nodeFn) {return nodeFn.first->Name() == "_Graph_Sink"; });
    if (itNodeFn == constructedFunctions.end())
    {
        return nullptr;
    }

    std::vector<FunctionPtr> functions;
    for (Node::NodeConstIterator it = itNodeFn->first->InputNodes_begin(); it != itNodeFn->first->InputNodes_end(); ++it)
    {
        functions.push_back(constructedFunctions[*it]);
    }

    if (functions.size() == 0)
    {
        return nullptr;
    }
    else if (functions.size() == 1)
    {
        return functions[0];
    }
    else
    {
        // in case multiple outputs are in a graph, combine them into one CNTK graph.
        return Combine(std::vector<Variable>(functions.begin(), functions.end()));
    }
}
