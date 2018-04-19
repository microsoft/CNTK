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
#include "RNNHelper.h"

using namespace ONNXIR;
using namespace CNTK;
using namespace CNTK::ONNX;

namespace CNTK
{

typedef std::unordered_map<const Node *, std::vector<FunctionPtr>> ONNXToCNTKMap;
typedef std::unordered_map<std::string, Variable> ONNXToCNTKVariableMap;
class ONNXToCNTKHelper
{
public:
    //
    // Convert an ONNX graph to a CNTK graph (Function).
    //
    static std::vector<FunctionPtr> FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                 ONNXToCNTKVariableMap &constructedNodeArgVariableMap,
                                                 const Graph *graph, const DeviceDescriptor &computeDevice);

private:
    static FunctionPtr CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
                                      const DeviceDescriptor &computeDevice);
    static std::vector<size_t> GetNodeDims(const Node *node);
    static Constant CreateConstant(const Node *node, const DeviceDescriptor &computeDevice);
    static Constant CreateConstant(const onnx::TensorProto &valueProto, const std::string &nodeName,
                                   const DeviceDescriptor &computeDevice);
    static Variable CreateLeafVariableOrConstant(const NodeArg *nodeArg, const Node *parentNode, const Graph *graph,
                                                 const DeviceDescriptor &computeDevice);
    static std::vector<Variable> CreateRNNLeafVariableOrConstant(const NodeArg *nodeArg,
                                                                 const Node *parentNode, const Graph *graph,
                                                                 ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const DeviceDescriptor &computeDevice);
    static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs);

    static bool IsSecondInputOfElementWiseOpsWithBroadcast(const Node *parentNode, const NodeArg *nodeArg);
    static bool FixConstantShapeForConstantVariableInputPair(const std::vector<Variable> &inputs,
                                                             std::vector<Variable> &fixedInputs);

    static const Node *GetChildNode(const Node *parentNode, const NodeArg *nodeArg);

    static std::vector<Axis> AttributeProtoToAxes(const AttributeProto &attributeProto);
    static Axis AttributeProtoToAxis(const AttributeProto &attributeProto);

    static onnx::TypeProto FromINTS(const std::vector<int64_t> &shape);
    static NDShape FromTypeProto(const onnx::TypeProto &tensorShape);
    static NDShape FromTensorShapeProto(const onnx::TensorShapeProto &tensorShape);
    static std::vector<bool> FromTypeProtoAsBool(const onnx::TypeProto &tensorShape);
    static DataType FromONNXType(onnx::TypeProto type);

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
    static std::vector<Axis> GetNamedAttributeAsAxes(const Node *node, const string &attributeName);
    static std::vector<Axis> GetNamedAttributeAsAxes(const Node *node, const string &attributeName,
                                                     const std::vector<Axis> &defaultAxes);

    static Axis GetNamedAttributeAsAxis(const Node *node, const string &attributeName);
    static Axis GetNamedAttributeAsAxis(const Node *node, const string &attributeName,
                                        const Axis &defaultAxis);

    static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName,
                                            bool hasBatchAxis);
    static NDShape GetNamedAttributeAsShape(const Node *node, const string &attributeName,
                                            bool hasBatchAxis, const NDShape defaultShape);

    static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName);
    static std::vector<bool> GetNamedAttributeAsShapeBool(const Node *node, const string &attributeName,
                                                          const std::vector<bool> &defaultShape);

    static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName);
    static size_t GetNamedAttributeAsInt64(const Node *node, const string &attributeName, size_t defaultValue);

    static std::vector<int> VecInt64ToVecInt(const std::vector<int64_t> &vecInt64);
    static std::vector<int64_t> VecIntToVecInt64(const std::vector<int> &vecInt);
    static std::vector<Axis> GetAxisVecFromIntVec(const std::vector<int> &vecInt);

    static std::vector<size_t> VecFloatToVecSize_t(const std::vector<float> &vecFloat);

    static std::vector<Axis> ConvertPermutationONNXToCNTK(const std::vector<int64_t> &permutation, bool hasBatchAxis, bool hasSequenceAxis);

    static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName);
    static float GetNamedAttributeAsFloat(const Node *node, const string &attributeName, float defaultValue);

    static string GetNamedAttributeAsString(const Node *node, const string &attributeName);
    static string GetNamedAttributeAsString(const Node *node, const string &attributeName, const string &defaultValue);

    static std::vector<std::string> GetNamedAttributeAsStringVec(const Node *node, const string &attributeName,
                                                                 const std::vector<std::string> &defaultValues);

    static std::vector<int64_t> GetNamedAttributeAsInt64Vec(const Node *node, const string &attributeName);
    static std::vector<int64_t> GetNamedAttributeAsInt64Vec(const Node *node, const string &attributeName, const std::vector<int64_t> &defaultValue);

    static std::vector<float> GetNamedAttributeAsFloatVec(const Node *node, const string &attributeName);
    static std::vector<float> GetNamedAttributeAsFloatVec(const Node *node, const string &attributeName, const std::vector<float> &defaultValue);

    static Axis ConvertONNXAxisToCNTKCppApi(int64_t axes, const Variable &input);
    static std::vector<Axis> ConvertONNXAxesToCNTKCppApi(const std::vector<int64_t> &axes, const Variable &operand);

    static void AdjustAutoPaddingAndStrideForCNTKSpecialCases(const Variable &operand,
                                                              std::vector<bool> &autoPadding, NDShape &strides);
    static std::pair<std::vector<size_t>, std::vector<size_t>> SplitAndReverseVec(std::vector<int64_t> &pads);
    static std::pair<std::vector<size_t>, std::vector<size_t>> AdjustONNXPadsVecForCNTKPadOp(const Variable &operand, std::vector<int64_t> &pads);
    static NDShape ReverseShape(const NDShape &shape);

    static std::pair<Variable, Variable> BroadcastElementWiseInput(const Node *node,
                                                                   const Variable &input0, const Variable &input1);

    static Variable GetNodeOperandWithPaddingResolved(std::vector<bool> &cntkConvAutoPadding,
                                                      NDShape &strides, const Node *node, const std::vector<Variable> &inputs, const double padValue = 0.0);

    //
    // CNTK convolution/pooling operations do not support ONNX same_low padding.
    // This method does padding accoordingly before invoking
    // Convolution/Pooling operations.
    //
    static FunctionPtr CreatePadOpForSameLowAutoPad(
        const Variable &input, NDShape kernelShape, NDShape strides, const double padValue);
    static FunctionPtr CreateCNTKConvNode(const Node *node, const std::vector<Variable> &inputs);
    static FunctionPtr CreateCNTKConvTransposeNode(const Node *node, const std::vector<Variable> &inputs);
    static FunctionPtr CreateCNTKFCNode(const std::wstring &nodeName, const std::vector<Variable> &inputs);

    //
    // Methods for creating CNTK input variables for a given node.
    //
    static std::vector<Variable> CreateCNTKInputsStartingFromIndex(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                                   ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, size_t startIndex, const DeviceDescriptor &computeDevice);
    static std::vector<Variable> CreateCNTKInputs(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                  ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, const DeviceDescriptor &computeDevice);

    //
    // Method for special checking if an ONNX node belongs to special subgraph that is created by OptimizedRNNStack export.
    //
    static std::pair<bool, std::vector<FunctionPtr>> CheckNodeBelongsToOptimizedRnnStack(const Node *node, const std::vector<Variable> &inputs,
                                                                                         ONNXToCNTKMap &constructedNodeMap, ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, const DeviceDescriptor &computeDevice);

    static ConvAutoPadType ConvertStrToConvAutoPadType(const string &str);
};

} // namespace CNTK

std::vector<Axis> ONNXToCNTKHelper::AttributeProtoToAxes(const AttributeProto &attributeProto)
{
    std::vector<Axis> axes;
    std::vector<int64_t> ints(attributeProto.ints().begin(), attributeProto.ints().end());
    // axes may get saved as collection or a single
    // int CNTKToONNXHelper::ToIndex(const Axis& axis) applies axis.StaticAxisIndex() + 1
    // to get index for ONNX. Deduct by one to get index in CNTK
    if (!ints.empty())
    {
        for (std::vector<int64_t>::const_iterator it = ints.begin(); it != ints.end(); it++)
        {
            axes.push_back(Axis((int) (*it) - 1));
        }
    }
    return axes;
}

Axis ONNXToCNTKHelper::AttributeProtoToAxis(const AttributeProto &attributeProto)
{
    Axis axis((int) (attributeProto.i()) - 1);
    return axis;
}

onnx::TypeProto ONNXToCNTKHelper::FromINTS(const std::vector<int64_t> &shape)
{
    onnx::TypeProto newShape;

    for (std::vector<int64_t>::const_iterator it = shape.begin(); it != shape.end(); it++)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(*it);
    }

    return newShape;
}

NDShape ONNXToCNTKHelper::FromTypeProto(const onnx::TypeProto &tensorShape)
{
    return FromTensorShapeProto(tensorShape.tensor_type().shape());
}

NDShape ONNXToCNTKHelper::FromTensorShapeProto(const onnx::TensorShapeProto &tensorShape)
{
    std::vector<size_t> dimensions;
    for (int i = 0; i < tensorShape.dim_size(); ++i)
    {
        auto dim_i = tensorShape.dim(i);
        if (dim_i.has_dim_value())
            dimensions.push_back(tensorShape.dim(i).dim_value());
        else if (dim_i.has_dim_param())
            dimensions.push_back(NDShape::FreeDimension);
        else
            LogicError("ONNX::TensorShapeProto_Dimension must have either dim_value or dim_param specified.");
    }

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

std::vector<bool> ONNXToCNTKHelper::FromTypeProtoAsBool(const onnx::TypeProto &tensorShape)
{
    std::vector<bool> dimensions;
    for (int index = 0; index < tensorShape.tensor_type().shape().dim_size(); index++)
        dimensions.push_back(tensorShape.tensor_type().shape().dim(index).dim_value() == 0 ? false : true);

    // CNTKToONNX ToTensorShape does reverse, need to reverse to restore CNTK shape
    std::reverse(dimensions.begin(), dimensions.end());
    return dimensions;
}

DataType ONNXToCNTKHelper::FromONNXType(onnx::TypeProto type)
{
    switch (type.tensor_type().elem_type())
    {
    case onnx::TensorProto_DataType_FLOAT:
        return DataType::Float;
    case onnx::TensorProto_DataType_DOUBLE:
        return DataType::Double;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

// helpers copied from ONNXIR (Converter.cc). These functions will eventually
// be replaced with functionalities of onnx core.
bool IsLittleEndianOrder()
{
    int n = 1;
    return (*(char *) &n == 1);
}

#pragma warning(disable : 4244)

float UnpackFloat(const char *buf, int i)
{
    float temp = 0;
    if (IsLittleEndianOrder())
    {
        memcpy((void *) &temp, (void *) buf, sizeof(char) * 4);
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

void RetrieveRawDataAsFloat(const onnx::TensorProto &valueProto)
{
    if (!valueProto.float_data().empty())
        return;

    auto raw_data = valueProto.raw_data();
    onnx::TensorProto &mutableProto = const_cast<onnx::TensorProto &>(valueProto);
    ::google::protobuf::RepeatedField<float> *p_mutable_float_data = mutableProto.mutable_float_data();
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
        memcpy((void *) &temp, (void *) buf, sizeof(char) * 8);
    }
    else
    {
        // this is temperal code that will soon be replaced by onnx core.
        NOT_IMPLEMENTED;
    }
    return temp;
}

void RetrieveRawDataAsDouble(const onnx::TensorProto &valueProto)
{
    if (!valueProto.double_data().empty())
        return;

    auto raw_data = valueProto.raw_data();
    onnx::TensorProto &mutableProto = const_cast<onnx::TensorProto &>(valueProto);
    ::google::protobuf::RepeatedField<double> *p_mutable_double_data = mutableProto.mutable_double_data();
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

std::vector<size_t> ONNXToCNTKHelper::GetNodeDims(const Node *node)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
    if (itValue != node->GetAttributes().cend())
    {
        const onnx::TensorProto valueProto = itValue->second.t();
        return std::vector<size_t>(valueProto.dims().begin(), valueProto.dims().end());
    }
    else
    {
        std::vector<size_t> ret;
        const std::vector<NodeArg> &outputArgs = node->OutputDefs();
        std::vector<NodeArg>::const_iterator it = outputArgs.begin();
        if (it != outputArgs.end())
        {
            const TensorShapeProto *shape = (*it).Shape();
            int rank = shape->dim_size();
            for (int i = 0; i < rank; i++)
                ret.push_back(shape->dim(i).dim_value());
        }
        return ret;
    }
}

Constant ONNXToCNTKHelper::CreateConstant(const Node *node, const DeviceDescriptor &computeDevice)
{
    NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
    const onnx::TensorProto valueProto = itValue->second.t();

    return CreateConstant(valueProto, node->Name(), computeDevice);
}

Constant ONNXToCNTKHelper::CreateConstant(const onnx::TensorProto &valueProto, const std::string &nodeName,
                                          const DeviceDescriptor &computeDevice)
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

const Node *ONNXToCNTKHelper::GetChildNode(const Node *parentNode, const NodeArg *nodeArg)
{
    const Node::EdgeEnd *inputEdgeSrcEnd = nullptr;
    if (parentNode->InputEdgeSrcEnd(const_cast<NodeArg *>(nodeArg), &inputEdgeSrcEnd))
    {
        return inputEdgeSrcEnd->GetNode();
    }
    return nullptr;
}

bool ONNXToCNTKHelper::IsSecondInputOfElementWiseOpsWithBroadcast(const Node *parentNode, const NodeArg *nodeArg)
{
    if (Operators::SupportBroadcastONNXOp(parentNode->OpType()))
    {
        if (HasNamedAttribute(parentNode, "broadcast") && 1 == static_cast<int>(GetNamedAttributeAsInt64(parentNode, "broadcast")))
        {
            const std::vector<NodeArg> &inputNodeArgs = parentNode->InputDefs();
            for (int index = 0; index < inputNodeArgs.size(); index++)
            {
                const NodeArg &childNodeArg = inputNodeArgs[index];
                if (childNodeArg.Name() == nodeArg->Name())
                {
                    return index == 1;
                }
            }
        }
    }
    return false;
}

// A helper for Concat case where one input has batch axis and the other is a constant without the batch axis
// in this case, we have to replace the constant input with a new constant variable of the correct shape
// (with batch axis removed)
bool ONNXToCNTKHelper::FixConstantShapeForConstantVariableInputPair(const std::vector<Variable> &inputs,
                                                                    std::vector<Variable> &fixedInputs)
{
    if (inputs.size() != 2)
        return false;

    int indexWithBatchAxis, indexConstantWithoutBatchAxis;
    if (inputs[0].HasBatchAxis() && (inputs[1].IsConstant() && !inputs[1].HasBatchAxis()))
    {
        indexWithBatchAxis = 0;
        indexConstantWithoutBatchAxis = 1;
    }
    else if (inputs[1].HasBatchAxis() && (inputs[0].IsConstant() && !inputs[0].HasBatchAxis()))
    {
        indexWithBatchAxis = 1;
        indexConstantWithoutBatchAxis = 0;
    }
    else
        return false;

    const Variable &variableInput = inputs[indexWithBatchAxis];
    Constant constantInput(inputs[indexConstantWithoutBatchAxis]);

    NDShape variableShape = variableInput.Shape();
    NDShape oldShape = constantInput.Shape();

    if (oldShape.Rank() != variableShape.Rank() + 1)
    {
        LogicError("FixConstantShapeForConstantVariableInputPair requires rank of constant input being higher then the rank of variable input by 1");
    }
    if (oldShape[oldShape.Rank() - 1] != 1)
    {
        LogicError("FixConstantShapeForConstantVariableInputPair requires size of the last dimension being 1.");
    }

    NDShape newShape = oldShape.SubShape(0, oldShape.Rank() - 1);

    fixedInputs.resize(2);
    fixedInputs[indexWithBatchAxis] = variableInput;
    fixedInputs[indexConstantWithoutBatchAxis] = Reshape(inputs[indexConstantWithoutBatchAxis], newShape);
    return true;
}

int CalculateNodeArgInputIndex(const NodeArg *nodeArg, const Node *parentNode)
{
    std::vector<NodeArg>::const_iterator it = std::find_if(parentNode->InputDefs().cbegin(),
                                                           parentNode->InputDefs().cend(), [nodeArg](const NodeArg &other) { return other.Name() == nodeArg->Name(); });
    if (it == parentNode->InputDefs().cend())
    {
        return -1;
    }
    else
        return it - parentNode->InputDefs().cbegin();
}

template <typename DType>
Constant CreateConstantWithRawData(DType *data, const NDShape &shape, const std::string &name,
                                   const DeviceDescriptor &computeDevice)
{
    DataType dataType = AsDataType<DType>();

    int totalSize = shape.TotalSize();
    NDArrayViewPtr dstFinal(new NDArrayView(dataType, shape, data,
                                            totalSize * sizeof(DType), computeDevice.CPUDevice()));

    if (computeDevice.Type() == DeviceKind::CPU)
    {
        Constant constantVariable(dstFinal, ToWString(name));
        return constantVariable;
    }
    else
    {
        // this is the way to load values into GPU:
        // Create a GPU NDArrayView and CopyFrom a CPU NDArrayView that holding the data.
        NDArrayViewPtr dstFinalGPU(new NDArrayView(dataType, StorageFormat::Dense, shape, computeDevice));
        dstFinalGPU->CopyFrom(*dstFinal);
        Constant constantVariable(dstFinalGPU, ToWString(name));
        return constantVariable;
    }
}

std::vector<Variable> CreateRNNConstant(
    const Node *parentNode, int index, const std::string &name, const onnx::TensorProto &valueProto, const DeviceDescriptor &computeDevice)
{
    std::vector<Variable> inputs;
    auto dataType = valueProto.data_type();

    switch (dataType)
    {
    case TensorProto_DataType_FLOAT:
    {
        if (valueProto.float_data().empty())
        {
            RetrieveRawDataAsFloat(valueProto);
        }
    }
    case TensorProto_DataType_DOUBLE:
    {
        if (valueProto.double_data().empty())
        {
            RetrieveRawDataAsDouble(valueProto);
        }
    }
    }

    string parentONNXOpName = parentNode->OpType();
    // index to LSTM inputs as specified in the ONNX document.
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---8
    if (parentONNXOpName == "LSTM")
    {
        switch (index)
        {
        case LSTMInputIndexX:
            // X, should not come to here
            CNTK::LogicError("input to a recurrent node shall not be a constant");
        case LSTMInputIndexW:
        case LSTMInputIndexH:
            // W, R:
            {
                // see ONNX spec for the tensor shape
                int num_directions = valueProto.dims(0);
                size_t rows = valueProto.dims(1);
                size_t cols = valueProto.dims(2);

                // CNTK cpp requires shape being (input_size, 4 * hidden_size)
                NDShape weightShape({rows, cols});

                int input_size = cols;
                int cell_size = rows / 4;

                for (int dir = 0; dir < num_directions; dir++)
                {
                    std::string nodeName = name + (index == 1 ? "_W_" : "_R_") + (char) ('0' + dir);
                    int totalSizePerDirection = rows * cols;

                    // TODO: what about double?
                    float *data = new float[totalSizePerDirection];
                    for (size_t count = 0; count < totalSizePerDirection; count++)
                    {
                        int row = count / input_size;
                        int col = count % input_size;
                        int block = row / cell_size;

                        if (block == 1)
                        {
                            // o
                            row += cell_size * 2;
                        }
                        else if (block == 3)
                        {
                            // c
                            row -= cell_size * 2;
                        }

                        int sourceIndex = dir * totalSizePerDirection + count;
                        int targetIndex = col * cell_size * 4 + row;
                        data[targetIndex] = valueProto.float_data()[sourceIndex];
                    }

                    Constant constant = CreateConstantWithRawData(&data[0], weightShape, nodeName, computeDevice);
                    inputs.push_back(constant);
                }
                return inputs;
            }
        case LSTMInputIndexB:
            // B
            {
                // see ONNX spec for the tensor shape
                int num_directions = valueProto.dims(0);
                int cell_size = valueProto.dims(1) / 8;
                // there is an ONNX spec issue with bias input. It states that
                // "This tensor has shape `[num_directions, 8*hidden_size]", which means
                // hidden and input are applied with bias separately after weight.
                // In CNTK, bias is be applied in fused form, after hidden and input
                // are element-wise added. In this case
                // the bias shape is [num_directions, 4*hidden_size]
                NDShape weightShape({(size_t)(4 * cell_size)});
                for (int dir = 0; dir < num_directions; dir++)
                {
                    std::string nodeName = name + std::string(1, (char) ('0' + dir)) + LSTMInputBiasNameHint;
                    int totalSizePerDirection = 4 * cell_size;
                    float *data = new float[totalSizePerDirection];
                    for (size_t targetIndex = 0; targetIndex < totalSizePerDirection; targetIndex++)
                    {
                        int row = targetIndex;

                        // TODO: specific to LSTM. icfo (CNTK) to iofc(ONNX)
                        int block = row / cell_size;
                        if (block == 1)
                        {
                            // c
                            row += 2 * cell_size;
                        }
                        else if (block == 3)
                        {
                            // o
                            row -= 2 * cell_size;
                        }

                        // source is column major
                        int src_index = row;
                        // "fuse"
                        data[targetIndex] =
                            valueProto.float_data()[dir * 2 * totalSizePerDirection + src_index] +
                            valueProto.float_data()[dir * 2 * totalSizePerDirection + totalSizePerDirection + src_index];
                    }

                    Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                    inputs.push_back(constant);
                }
                return inputs;
            }
        case LSTMInputIndexSequenceLens:
            // sequence length is treated as free dimension
            return inputs;
        case LSTMInputIndexinitial_h:
        case LSTMInputIndexinitial_c:
        {
            // initial_h, initial_c
            int num_directions = valueProto.dims(0);

            // TODO: batch shall be one?
            // int batchSize = valueProto.dims(1);
            int cell_size = valueProto.dims(2);
            // there is an ONNX spec issue with bias input. It states that
            // "This tensor has shape `[num_directions, 8*hidden_size]", which means
            // hidden and input are applied with bias separately after weight.
            // In CNTK, bias is be applied in fused form, after hidden and input
            // are element-wise added. In this case
            // the bias shape is [num_directions, 4*hidden_size]
            NDShape weightShape({(size_t)(cell_size)});
            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string nodeName = name + std::string(1, (char) ('0' + dir));
                if (index == 5)
                    nodeName += LSTMInputInitialHNameHint;
                else
                    nodeName += LSTMInputInitialCNameHint;

                float *data = new float[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    data[targetIndex] = valueProto.float_data()[dir * cell_size + targetIndex];
                }

                Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                inputs.push_back(constant);
            }
            return inputs;
        }
        break;
        case LSTMInputIndexP:
            // P
            {
                int num_directions = valueProto.dims(0);
                int cell_size = valueProto.dims(1) / 3;
                for (int dir = 0; dir < num_directions; dir++)
                    for (int i = 0; i < 3; i++)
                    {
                        std::string nodeName = name + ((i == 0) ? "_i" : ((i == 1) ? "_o" : "_f")) +
                                               std::string(1, (char) ('0' + dir)) + LSTMInputPeepholeNameHint;
                        float *data = new float[cell_size];
                        NDShape weightShape({(size_t)(cell_size)});
                        for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                        {
                            data[targetIndex] = valueProto.float_data()[(dir * 3 + i) * cell_size + targetIndex];
                        }

                        Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                        inputs.push_back(constant);
                    }
                return inputs;
            }
        default:
            CNTK::LogicError("CreateRNNConstant received unexpected index: %d", index);
        }
    }
    else if (parentONNXOpName == "GRU")
    {
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---6
        switch (index)
        {
        case GRUInputIndexX:
            // X, should not come to here
            CNTK::LogicError("input to a recurrent node shall not be a constant");
        case GRUInputIndexW:
        {
            // see ONNX spec for the tensor shape
            int num_directions = valueProto.dims(0);
            size_t rows = valueProto.dims(1);
            size_t cols = valueProto.dims(2);

            // CNTK cpp requires shape: (input_size, 3 * hidden_size)
            NDShape weightShape({rows, cols});

            int input_size = cols;
            int cell_size = rows / 3;

            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string nodeName = name + "_W_" + (char) ('0' + dir);
                int totalSizePerDirection = rows * cols;

                // TODO: what about double?
                float *data = new float[totalSizePerDirection];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    int targetIndex = col * cell_size * GRUWeightDimensionHiddenMultiplier + row;
                    data[targetIndex] = valueProto.float_data()[sourceIndex];
                }

                Constant constant = CreateConstantWithRawData(&data[0], weightShape, nodeName, computeDevice);
                inputs.push_back(constant);
            }
            return inputs;
        }
        case GRUInputIndexR:
        {
            // split into H and H1 for CNTK GRU implementation
            int num_directions = valueProto.dims(0);
            size_t rows = valueProto.dims(1);
            size_t cols = valueProto.dims(2);

            int input_size = cols;
            int cell_size = rows / 3;

            NDShape hShape({(size_t) cell_size * 2, (size_t) input_size});
            NDShape h1Shape({(size_t) cell_size, (size_t) input_size});

            inputs.resize(num_directions * 2);
            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string hNodeName = name + "_H_" + (char) ('0' + dir);
                std::string h1NodeName = name + "_H1_" + (char) ('0' + dir);
                int totalSizePerDirection = rows * cols;

                float *hData = new float[hShape.TotalSize()];
                float *h1Data = new float[h1Shape.TotalSize()];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int block = row / cell_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    if (block < CNTKGRUZRWeightMultiplier)
                    {
                        int targetIndex = col * cell_size * CNTKGRUZRWeightMultiplier + row;
                        hData[targetIndex] = valueProto.float_data()[sourceIndex];
                    }
                    else
                    {
                        int targetIndex = col * cell_size + row - cell_size * CNTKGRUZRWeightMultiplier;
                        h1Data[targetIndex] = valueProto.float_data()[sourceIndex];
                    }
                }

                Constant constantH = CreateConstantWithRawData(&hData[0], hShape, hNodeName, computeDevice);
                Constant constantH1 = CreateConstantWithRawData(&h1Data[0], h1Shape, h1NodeName, computeDevice);
                inputs[dir] = constantH;
                inputs[dir + num_directions] = constantH1;
            }
            return inputs;
        }
        case GRUInputIndexB:
            // B
            {
                // see ONNX spec for the tensor shape
                int num_directions = valueProto.dims(0);
                int cell_size = valueProto.dims(1) / GRUBiasDimensionHiddenMultiplier;
                // shape size is divided by 2 so that it only applies to input (CNTK)
                // TODO: this incompatibility needs further investigation.
                NDShape weightShape({(size_t)(GRUBiasDimensionHiddenMultiplier / 2 * cell_size)});
                for (int dir = 0; dir < num_directions; dir++)
                {
                    std::string nodeName = name + std::string(1, '0' + dir) + LSTMInputBiasNameHint;
                    int totalSizePerDirection = GRUBiasDimensionHiddenMultiplier / 2 * cell_size;
                    float *data = new float[totalSizePerDirection];
                    for (size_t targetIndex = 0; targetIndex < totalSizePerDirection; targetIndex++)
                    {
                        int row = targetIndex;
                        // source is column major
                        int src_index = row;
                        // "fuse"
                        data[targetIndex] =
                            valueProto.float_data()[dir * 2 * totalSizePerDirection + src_index] +
                            valueProto.float_data()[dir * 2 * totalSizePerDirection + totalSizePerDirection + src_index];
                    }

                    Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                    inputs.push_back(constant);
                }
                return inputs;
            }
        case GRUInputIndexSequenceLens:
            return inputs;
        case GRUInitialH:
        {
            // initial_h
            int num_directions = valueProto.dims(0);
            int cell_size = valueProto.dims(2);
            NDShape weightShape({(size_t)(cell_size)});
            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string nodeName = name + std::string(1, (char) ('0' + dir)) + LSTMInputInitialHNameHint;

                float *data = new float[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    data[targetIndex] = valueProto.float_data()[dir * cell_size + targetIndex];
                }

                Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                inputs.push_back(constant);
            }
            return inputs;
        }
        default:
            CNTK::LogicError("CreateRNNConstant for GRU op received unexpected index: %d", index);
        }
    }
    else if (parentONNXOpName == "RNN")
    {
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---6-1
        switch (index)
        {
        case RNNInputIndexX:
            // X, should not come to here
            CNTK::LogicError("input to a recurrent node shall not be a constant");
        case RNNInputIndexW:
        case RNNInputIndexR:
        {
            // see ONNX spec for the tensor shape
            int num_directions = valueProto.dims(0);
            size_t rows = valueProto.dims(1);
            size_t cols = valueProto.dims(2);

            // CNTK cpp requires shape: (input_size, 3 * hidden_size)
            NDShape weightShape({rows, cols});

            int input_size = cols;
            int cell_size = rows;

            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string nodeName = name + (index == RNNInputIndexW ? "_W_" : "_R_") + (char) ('0' + dir);
                int totalSizePerDirection = rows * cols;

                // TODO: what about double?
                float *data = new float[totalSizePerDirection];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    int targetIndex = col * cell_size + row;
                    data[targetIndex] = valueProto.float_data()[sourceIndex];
                }

                Constant constant = CreateConstantWithRawData(&data[0], weightShape, nodeName, computeDevice);
                inputs.push_back(constant);
            }
            return inputs;
        }
        case RNNInputIndexB:
            // B
            {
                // see ONNX spec for the tensor shape:
                // https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---6-1
                // shape of bias is [num_directions, 2*hidden_size] thus we divide dim(1) by 2
                // to get cell_size.
                int num_directions = valueProto.dims(0);
                int cell_size = valueProto.dims(1) / 2;
                NDShape weightShape({(size_t)(cell_size)});
                for (int dir = 0; dir < num_directions; dir++)
                {
                    std::string nodeName = name + std::string(1, '0' + dir) + LSTMInputBiasNameHint;
                    int totalSizePerDirection = cell_size;
                    float *data = new float[totalSizePerDirection];
                    for (size_t targetIndex = 0; targetIndex < totalSizePerDirection; targetIndex++)
                    {
                        int row = targetIndex;
                        // source is column major
                        int src_index = row;
                        // "fuse"
                        // RNN only has one bias vector. It is applied after element-wise addition
                        // of projected input and hidden states. Therefore we need to fuse two biases
                        // in ONNX into one.
                        // RNNBiasMultiplier = 2
                        data[targetIndex] =
                            valueProto.float_data()[dir * RNNBiasMultiplier * totalSizePerDirection + src_index] +
                            valueProto.float_data()[dir * RNNBiasMultiplier * totalSizePerDirection + totalSizePerDirection + src_index];
                    }

                    Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                    inputs.push_back(constant);
                }
                return inputs;
            }
        case RNNInputIndexSequenceLens:
            return inputs;
        case RNNInitialH:
        {
            // initial_h
            int num_directions = valueProto.dims(0);
            int cell_size = valueProto.dims(2);
            NDShape weightShape({(size_t)(cell_size)});
            for (int dir = 0; dir < num_directions; dir++)
            {
                std::string nodeName = name + std::string(1, (char) ('0' + dir)) + LSTMInputInitialHNameHint;

                float *data = new float[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    data[targetIndex] = valueProto.float_data()[dir * cell_size + targetIndex];
                }

                Constant constant = CreateConstantWithRawData(data, weightShape, nodeName, computeDevice);
                inputs.push_back(constant);
            }
            return inputs;
        }
        default:
            CNTK::LogicError("CreateRNNConstant for GRU op received unexpected index: %d", index);
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

std::vector<FunctionPtr> CreateRNNConstantOp(const Graph *graph, const Node *node, const Node *parentNode, int index,
                                             const DeviceDescriptor &computeDevice)
{
    const onnx::TensorProto *valueProto;
    if (!graph->GetInitialTensor(node->Name(), &valueProto))
    {
        NodeAttributes::const_iterator itValue = node->GetAttributes().find("value");
        if (itValue == node->GetAttributes().cend())
        {
            return std::vector<FunctionPtr>();
        }
        valueProto = &itValue->second.t();
    }

    std::vector<Variable> constantNodes = CreateRNNConstant(parentNode, index, node->Name(), *valueProto, computeDevice);
    std::vector<FunctionPtr> returns;
    for (auto c : constantNodes)
        returns.push_back(c);
    return returns;
}

std::vector<Variable> ONNXToCNTKHelper::CreateRNNLeafVariableOrConstant(const NodeArg *nodeArg,
                                                                        const Node *parentNode, const Graph *graph, ONNXToCNTKVariableMap &constructedNodeArgVariableMap,
                                                                        const DeviceDescriptor &computeDevice)
{
    string parentONNXOpName = parentNode->OpType();
    std::string nodeName = nodeArg->Name();
    const onnx::TensorProto *valueProto;
    if (graph->GetInitialTensor(nodeName, &valueProto))
    {
        int index = CalculateNodeArgInputIndex(nodeArg, parentNode);
        return CreateRNNConstant(parentNode, index, nodeName, *valueProto, computeDevice);
    }

    const TensorShapeProto *shapeProto = nodeArg->Shape();
    if (shapeProto == nullptr)
    {
        // dummy input,
        return std::vector<Variable>();
    }

    // std::string nodeName = nodeArg->Name();
    std::vector<Axis> dynamicAxes({Axis::OperandSequenceAxis(), Axis::DefaultBatchAxis()});

    if (parentONNXOpName == "LSTM")
    {
        // index to LSTM inputs as specified in the ONNX document.
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#inputs-3---8
        int inputIndex = CalculateNodeArgInputIndex(nodeArg, parentNode);
        switch (inputIndex)
        {
        case LSTMInputIndexX:
            // X: `[seq_length, batch_size, input_size]`.
            {
                Variable inputVariable;
                if (constructedNodeArgVariableMap.find(nodeArg->Name()) == constructedNodeArgVariableMap.end())
                {
                    DataType dataType = FromONNXType(nodeArg->ToProto().type());
                    int input_size = shapeProto->dim(2).dim_value();
                    NDShape shape({(size_t)(input_size)});
                    inputVariable = InputVariable(shape, dataType, ToWString(nodeArg->Name()), dynamicAxes);
                    constructedNodeArgVariableMap.insert(ONNXToCNTKVariableMap::value_type(nodeArg->Name(), inputVariable));
                }
                return std::vector<Variable>({constructedNodeArgVariableMap[nodeArg->Name()]});
            }
        // other inputs shall be ONNX constant node and be created as CNTK Constant in CreateRNNConstant
        case LSTMInputIndexW:            // W
        case LSTMInputIndexH:            // R
        case LSTMInputIndexB:            // B
        case LSTMInputIndexSequenceLens: // sequence_lens
        case LSTMInputIndexinitial_h:    // initial_h
        case LSTMInputIndexinitial_c:    // initial_c
        case LSTMInputIndexP:            // P
            NOT_IMPLEMENTED;
        default:
            LogicError("LSTM node has unexpected input");
        }
    }
    else if (parentONNXOpName == "GRU")
    {
        int inputIndex = CalculateNodeArgInputIndex(nodeArg, parentNode);
        switch (inputIndex)
        {
        case GRUInputIndexX:
            // X: `[seq_length, batch_size, input_size]`.
            {
                Variable inputVariable;
                if (constructedNodeArgVariableMap.find(nodeArg->Name()) == constructedNodeArgVariableMap.end())
                {
                    DataType dataType = FromONNXType(nodeArg->ToProto().type());
                    int input_size = shapeProto->dim(2).dim_value();
                    NDShape shape({(size_t)(input_size)});
                    inputVariable = InputVariable(shape, dataType, ToWString(nodeArg->Name()), dynamicAxes);
                    constructedNodeArgVariableMap.insert(ONNXToCNTKVariableMap::value_type(nodeArg->Name(), inputVariable));
                }
                return std::vector<Variable>({constructedNodeArgVariableMap[nodeArg->Name()]});
            }
        // other inputs shall be ONNX constant node and be created as CNTK Constant in CreateRNNConstant
        case GRUInputIndexW:
        case GRUInputIndexR:
        case GRUInputIndexB:
        case GRUInputIndexSequenceLens:
        case GRUInitialH:
            NOT_IMPLEMENTED;
        default:
            LogicError("GRU node has unexpected input");
        }
    }
    else if (parentONNXOpName == "RNN")
    {
        int inputIndex = CalculateNodeArgInputIndex(nodeArg, parentNode);
        switch (inputIndex)
        {
        case GRUInputIndexX:
            // X: `[seq_length, batch_size, input_size]`.
            {
                Variable inputVariable;
                if (constructedNodeArgVariableMap.find(nodeArg->Name()) == constructedNodeArgVariableMap.end())
                {
                    DataType dataType = FromONNXType(nodeArg->ToProto().type());
                    int input_size = shapeProto->dim(2).dim_value();
                    NDShape shape({(size_t)(input_size)});
                    inputVariable = InputVariable(shape, dataType, ToWString(nodeArg->Name()), dynamicAxes);
                    constructedNodeArgVariableMap.insert(ONNXToCNTKVariableMap::value_type(nodeArg->Name(), inputVariable));
                }
                return std::vector<Variable>({constructedNodeArgVariableMap[nodeArg->Name()]});
            }
        // other inputs shall be ONNX constant node and be created as CNTK Constant in CreateRNNConstant
        case GRUInputIndexW:
        case GRUInputIndexR:
        case GRUInputIndexB:
        case GRUInputIndexSequenceLens:
        case GRUInitialH:
            NOT_IMPLEMENTED;
        default:
            LogicError("RNN node has unexpected input");
        }
    }
    else
    {
        NOT_IMPLEMENTED;
    }
}

Variable ONNXToCNTKHelper::CreateLeafVariableOrConstant(const NodeArg *nodeArg,
                                                        const Node *parentNode, const Graph *graph, const DeviceDescriptor &computeDevice)
{
    string parentONNXOpName = parentNode->OpType();

    std::string nodeName = nodeArg->Name();
    const onnx::TensorProto *valueProto;
    if (graph->GetInitialTensor(nodeName, &valueProto))
    {
        return CreateConstant(*valueProto, nodeName, computeDevice);
    }

    auto shapeProto = nodeArg->Shape();

    // in CNTK constants are created as Node (not a leaf) with values.
    // in ONNX constants may also be a leaf with values saved in initializer
    // here we know it is not an ONNX constant so reshape the variable to trim off last dim;
    NDShape shape = FromTensorShapeProto(*shapeProto);
    if (!IsSecondInputOfElementWiseOpsWithBroadcast(parentNode, nodeArg))
    {
        // can only do this when broadcast is 0
        shape = shape.SubShape(0, shape.Rank() - 1);
    }

    std::vector<Axis> dynamicAxes({Axis::DefaultBatchAxis()});

    // TODO: this is not fully correct. We need to get hasSequenceAxis
    // over the traverse path. An input will have a sequence axis
    // only if it outputs to an RNN op along the path.
    // This requires support from LotusIR.
    // Now traversing starts from arbitray nodes which may miss the RNN op.
    bool hasSequenceAxis = false;
    for (Graph::NodeIterator nodeIt = (const_cast<Graph *>(graph))->Nodes_begin();
         nodeIt != (const_cast<Graph *>(graph))->Nodes_end(); ++nodeIt)
        if (Operators::IsRNNOp((*nodeIt)->OpType()))
        {
            hasSequenceAxis = true;
            break;
        }

    if (hasSequenceAxis)
    {
        shape = shape.SubShape(0, shape.Rank() - 1);
        dynamicAxes.insert(dynamicAxes.begin(), Axis::OperandSequenceAxis());
    }

    auto dataType = FromONNXType(nodeArg->ToProto().type());
    switch (dataType)
    {
    case DataType::Float:
    {
        return InputVariable(shape, DataType::Float, ToWString(nodeArg->Name()), dynamicAxes);
    }
    case DataType::Double:
    {
        return InputVariable(shape, DataType::Double, ToWString(nodeArg->Name()), dynamicAxes);
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
        LogicError("%s has %d input axis/axes. It should has %d .", nodeName.c_str(), (int) axes.size(), requiredAxes);
}

ConvAutoPadType ONNXToCNTKHelper::ConvertStrToConvAutoPadType(const string &str)
{
    if (str == "VALID" || str == "valid")
        return ConvAutoPadType::VALID;
    else if (str == "SAME_UPPER" || str == "same_upper")
        return ConvAutoPadType::SAME_UPPER;
    else if (str == "SAME_LOWER" || str == "same_lower")
        return ConvAutoPadType::SAME_LOWER;
    else
        LogicError("Unknown value for %s attribute: %s", "auto_pad", str.c_str());
}

} // namespace CNTK

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

std::vector<Axis> ONNXToCNTKHelper::GetNamedAttributeAsAxes(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    return AttributeProtoToAxes(itValue->second);
}

std::vector<Axis> ONNXToCNTKHelper::GetNamedAttributeAsAxes(const Node *node, const string &attributeName,
                                                            const std::vector<Axis> &defaultAxes)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultAxes;
    }

    return AttributeProtoToAxes(itValue->second);
}

Axis ONNXToCNTKHelper::GetNamedAttributeAsAxis(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    return AttributeProtoToAxis(itValue->second);
}

Axis ONNXToCNTKHelper::GetNamedAttributeAsAxis(const Node *node, const string &attributeName,
                                               const Axis &defaultAxis)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultAxis;
    }

    return AttributeProtoToAxis(itValue->second);
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

string ONNXToCNTKHelper::GetNamedAttributeAsString(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    return attributeProto.s();
}

string ONNXToCNTKHelper::GetNamedAttributeAsString(const Node *node, const string &attributeName, const string &defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    return attributeProto.s();
}

std::vector<std::string> ONNXToCNTKHelper::GetNamedAttributeAsStringVec(const Node *node, const string &attributeName,
                                                                        const std::vector<std::string> &defaultValues)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
        return defaultValues;

    const AttributeProto &attributeProto = itValue->second;
    return std::vector<std::string>(attributeProto.strings().begin(), attributeProto.strings().end());
}

std::vector<int64_t> ONNXToCNTKHelper::GetNamedAttributeAsInt64Vec(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> intVector(attributeProto.ints().begin(), attributeProto.ints().end());
    return intVector;
}

std::vector<int64_t> ONNXToCNTKHelper::GetNamedAttributeAsInt64Vec(const Node *node, const string &attributeName,
                                                                   const std::vector<int64_t> &defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    std::vector<int64_t> intVector(attributeProto.ints().begin(), attributeProto.ints().end());
    return intVector;
}

std::vector<float> ONNXToCNTKHelper::GetNamedAttributeAsFloatVec(const Node *node, const string &attributeName)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, true);
    const AttributeProto &attributeProto = itValue->second;
    std::vector<float> floatVector(attributeProto.floats().begin(), attributeProto.floats().end());
    return floatVector;
}

std::vector<float> ONNXToCNTKHelper::GetNamedAttributeAsFloatVec(const Node *node, const string &attributeName,
                                                                 const std::vector<float> &defaultValue)
{
    NodeAttributes::const_iterator itValue = FindAttributeIterator(node, attributeName, false);
    if (itValue == node->GetAttributes().end())
    {
        return defaultValue;
    }
    const AttributeProto &attributeProto = itValue->second;
    std::vector<float> floatVector(attributeProto.floats().begin(), attributeProto.floats().end());
    return floatVector;
}

std::vector<int> ONNXToCNTKHelper::VecInt64ToVecInt(const std::vector<int64_t> &vecInt64)
{
    std::vector<int> vecInt(vecInt64.size());
    for (int i = 0; i < vecInt64.size(); i++)
    {
        vecInt[i] = static_cast<int>(vecInt64[i]);
    }

    return vecInt;
}

std::vector<int64_t> ONNXToCNTKHelper::VecIntToVecInt64(const std::vector<int> &vecInt)
{
    std::vector<int64_t> vecInt64(vecInt.size());
    for (int i = 0; i < vecInt.size(); i++)
    {
        vecInt64[i] = vecInt[i];
    }

    return vecInt64;
}

std::vector<size_t> ONNXToCNTKHelper::VecFloatToVecSize_t(const std::vector<float> &vecFloat)
{
    std::vector<size_t> vecSize_t(vecFloat.size());
    for (int i = 0; i < vecFloat.size(); i++)
    {
        vecSize_t[i] = (size_t) vecFloat[i];
    }

    return vecSize_t;
}

// this method is to undo ConvertPermutationCNTKToONNX.
std::vector<Axis> ONNXToCNTKHelper::ConvertPermutationONNXToCNTK(const std::vector<int64_t> &permutation, bool hasBatchAxis, bool hasSequenceAxis)
{
    std::vector<int64_t> localPermutation = permutation;
    if (hasSequenceAxis)
    {
        localPermutation.erase(localPermutation.begin());
        for (int i = 0; i < localPermutation.size(); i++)
            localPermutation[i]--;
    }

    if (hasBatchAxis)
    {
        localPermutation.erase(localPermutation.begin());
        for (int i = 0; i < localPermutation.size(); i++)
            localPermutation[i]--;
    }

    std::vector<Axis> axes(localPermutation.size());
    for (int i = 0; i < localPermutation.size(); i++)
    {
        int indexToCNTKPermTable = localPermutation.size() - i - 1;
        int axisIndexInONNX = localPermutation[i];
        int axisIndexInCNTK = localPermutation.size() - axisIndexInONNX - 1;
        axes[indexToCNTKPermTable] = Axis(axisIndexInCNTK);
    }
    return axes;
}

namespace CNTK
{
static void PrintGraph(FunctionPtr function, int spaces, bool useName = false)
{
    if (function->Inputs().empty())
    {
        cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + ToString(function->AsString()) << std::endl;
        return;
    }

    for (auto input : function->Inputs())
    {
        cout << string(spaces, '.') + "(" + ToString(useName ? function->Name() : function->Uid()) + ")" + "->" +
                    "(" + ToString(useName ? input.Name() : input.Uid()) + ")" + ToString(input.AsString())
             << std::endl;
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
} // namespace CNTK

std::vector<Axis> ONNXToCNTKHelper::GetAxisVecFromIntVec(const std::vector<int> &vecInt)
{
    std::vector<Axis> vecAxis;
    for (const auto &val : vecInt)
    {
        Axis axis(val);
        vecAxis.push_back(axis);
    }
    return vecAxis;
}

std::pair<Variable, Variable> ONNXToCNTKHelper::BroadcastElementWiseInput(
    const Node *node, const Variable &input0, const Variable &input1)
{
    auto shape0 = input0.Shape();
    auto shape1 = input1.Shape();

    if (bool broadcast = GetNamedAttributeAsInt64(node, "broadcast", 0) == 1 && HasNamedAttribute(node, "axis"))
    {
        if (shape0.Rank() < shape1.Rank())
        {
            // TODO: this may happen if the ONNX op is built with input swapping (which is necessary
            // because ONNX only allows broadcast of right-hand-operand).
            // LogicError("in case of element wise binary ops, rank of lhs shall not be less than the rand of the rhs");
            return {input1, input0};
        }
        else if (shape0.Rank() > shape1.Rank() && shape1.Rank() != 0)
        {
            // axis = 3, ONNX: [2, 3, 4, 5] + [3, 4] with broadcast = 1 and axis = 1
            // coming in this call:
            // input0: 5,4,3,2
            // input1: 4,3 -> 1,4,3,1
            int axis = static_cast<int>(GetNamedAttributeAsInt64(node, "axis"));
            // axis is in ONNX. Take into consideration with input0 has batch axis.
            if (input0.HasBatchAxis())
            {
                axis--;
            }

            // fill 1 from axis 0 to axis index - 1 then append shape1
            // index is in NDShape sense (reverse of ONNX/CNTK python)
            int index = shape0.Rank() - (axis + shape1.Rank());

            NDShape newShape;
            for (int i = 0; i < index; i++)
                newShape = newShape.AppendShape({1});
            newShape = newShape.AppendShape(shape1);

            // in this case, input0 will have one less dim because it was added to match input1's batch dim.
            bool needPostRemoveInput0BatchDim = (!input0.HasBatchAxis() && input1.HasBatchAxis());

            Variable newInput1;
            if (input1.IsInput())
            {
                // as an input variable, we have to replace the original input1 keep the input argument shape
                newInput1 = InputVariable(newShape, input1.GetDataType(), input1.Name(), {Axis::DefaultBatchAxis()});
            }
            else
            {
                newInput1 = Reshape(input1, newShape);
            }

            return {needPostRemoveInput0BatchDim ? (Variable)(Reshape(input0, shape0.SubShape(0, shape0.Rank() - 1))) : input0, newInput1};
        }
        else
        {
            return {input0, input1};
        }
    }
    else
    {
        // this is to handle edge cases where one input has batch (and sequence) axis and the other does not.
        // the input with rank higher is caused by extra batch/sequence dimension which need to be squeezed away.
        // TODO: investigate if this edge case can be avoided when converting CNTK to ONNX.
        if (input0.Shape().Rank() == input1.Shape().Rank() - 1)
        {
            if (input0.HasBatchAxis() && !input1.HasBatchAxis())
                return {input0, Reshape(input1, shape1.SubShape(0, shape1.Rank() - 1))};
        }
        else if (input0.Shape().Rank() == input1.Shape().Rank() - 2 && input0.DynamicAxes().size() == 2 &&
                 input1.DynamicAxes().size() == 0)
        {
            return {input0, Reshape(input1, shape1.SubShape(0, shape1.Rank() - 2))};
        }
        else if (input0.Shape().Rank() - 1 == input1.Shape().Rank())
        {
            if (!input0.HasBatchAxis() && input1.HasBatchAxis())
                return {Reshape(input0, shape0.SubShape(0, shape0.Rank() - 1)), input1};
        }
        else if (input0.Shape().Rank() - 2 == input1.Shape().Rank() && input1.DynamicAxes().size() == 2 &&
                 input0.DynamicAxes().size() == 0)
        {
            return {Reshape(input0, shape0.SubShape(0, shape0.Rank() - 2)), input1};
        }
        return {input0, input1};
    }
}

Axis ONNXToCNTKHelper::ConvertONNXAxisToCNTKCppApi(int64_t axis, const Variable &operand)
{
    // reverse CNTKToONNXHelper::ConvertAxisToOnnx
    // note that axis is already decreased by one (assuming there is a batch axis)
    int index = axis - operand.DynamicAxes().size();
    if (index < 0)
        LogicError("ConvertAxisToCNTKCppApi cannot convert index < 0 to axis");

    return Axis(-index - 1);
}

std::vector<Axis> ONNXToCNTKHelper::ConvertONNXAxesToCNTKCppApi(const std::vector<int64_t> &axes, const Variable &operand)
{
    std::vector<Axis> cntkAxes(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        cntkAxes[i] = ConvertONNXAxisToCNTKCppApi(axes[i], operand);
    }

    return cntkAxes;
}

void ONNXToCNTKHelper::AdjustAutoPaddingAndStrideForCNTKSpecialCases(const Variable &operand, std::vector<bool> &autoPadding, NDShape &strides)
{
    if ((operand.Shape().Rank() == (1 + autoPadding.size())))
    {
        autoPadding.push_back(false);
    }

    if ((operand.Shape().Rank() == (1 + strides.Rank())))
    {
        strides = strides.AppendShape({1});
    }
}

std::pair<std::vector<size_t>, std::vector<size_t>> ONNXToCNTKHelper::SplitAndReverseVec(std::vector<int64_t> &pads)
{
    // Split this into head (lower padding) and foot (upper padding), and reverse them because
    // CNTK dimensions are in reverse order than ONNX.
    auto numOperandDims = pads.size() / 2;
    std::vector<size_t> head(pads.rbegin() + numOperandDims, pads.rend());   // The first half (lower) reversed.
    std::vector<size_t> foot(pads.rbegin(), pads.rbegin() + numOperandDims); // The second half (upper) reversed.

    return std::make_pair(head, foot);
}

std::pair<std::vector<size_t>, std::vector<size_t>> ONNXToCNTKHelper::AdjustONNXPadsVecForCNTKPadOp(const Variable &operand, std::vector<int64_t> &pads)
{
    // If there are added dimensions because of depth/channels or batch axis, then insert zeros
    // in the 'pads' vector explicitly for those dimensions to indicate that no padding is
    // needed for those dimensions.
    int nPadDims = pads.size() / 2;
    int rankDiff = operand.Shape().Rank() - nPadDims;
    pads.insert(pads.begin(), rankDiff, 0);
    pads.insert(pads.begin() + nPadDims + rankDiff, rankDiff, 0);

    return SplitAndReverseVec(pads);
}

FunctionPtr ONNXToCNTKHelper::CreateFunction(const Node *node, const std::vector<Variable> &inputs)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "Cast" && inputs[0].GetDataType() == DataType::Float && inputs[0].Owner() != nullptr)
    {
        // CNTK does not support cast op. Only float is available with ONNX support.
        // Question for having a cast op: Why not cast data as necessary internally.
        return inputs[0].Owner();
    }
    else if (onnxOpName == "LSTM")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Sigmoid", "Tanh", "Tanh"}));
        return CreateLSTM(node, inputs, direction, activations, activation_alpha, activation_beta);
    }
    else if (onnxOpName == "GRU")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Sigmoid", "Tanh"}));
        return CreateGRU(node, inputs, direction, activations, activation_alpha, activation_beta);
    }
    else if (onnxOpName == "RNN")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Tanh"}));
        return CreateRNN(node, inputs, direction, activations, activation_alpha, activation_beta);
    }
    if (onnxOpName == "FC")
    {
        return CreateCNTKFCNode(ToWString(node->Name()), inputs);
    }
    else if (onnxOpName == "Flatten")
    {
        int64_t axisIndex = (size_t) GetNamedAttributeAsInt64(node, "axis", 0);
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkFunction = Flatten(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Equal")
    {
        FunctionPtr cntkFunction = Equal(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Greater")
    {
        FunctionPtr cntkFunction = Greater(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Less")
    {
        FunctionPtr cntkFunction = Less(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mean")
    {
        FunctionPtr cntkFunction = Mean(inputs, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Clip")
    {
        double minValue = GetNamedAttributeAsFloat(node, "min");
        double maxValue = GetNamedAttributeAsFloat(node, "max");
        Constant minVariable = Constant::Scalar(DataType::Float, minValue);
        Constant maxVariable = Constant::Scalar(DataType::Float, maxValue);
        FunctionPtr cntkFunction = Clip(inputs[0], minVariable, maxVariable, ToWString(node->Name()));
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
    else if (onnxOpName == "HardSigmoid")
    {
        float alpha = GetNamedAttributeAsFloat(node, "alpha");
        float beta = GetNamedAttributeAsFloat(node, "beta");
        FunctionPtr cntkFunction = HardSigmoid(inputs[0], alpha, beta, ToWString(node->Name()));
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

        bool ceilOutDim = false;
        bool includePad = false;

        std::vector<bool> cntkPoolingAutoPadding;
        auto padValue = (onnxOpName == "AveragePool") ? 0.0 : static_cast<double>(std::numeric_limits<int>::min());
        auto poolingOperand = GetNodeOperandWithPaddingResolved(/*output arg first*/ cntkPoolingAutoPadding, strides, node, inputs, padValue);

        FunctionPtr cntkFunction = Pooling(poolingOperand,
                                           onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
                                           poolingWindowShape, strides, cntkPoolingAutoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "GlobalAveragePool" || onnxOpName == "GlobalMaxPool")
    {
        NDShape strides = {1};
        std::vector<bool> autoPadding = {false};
        bool ceilOutDim = false;
        bool includePad = false;

        FunctionPtr cntkFunction = Pooling(inputs[0],
                                           onnxOpName == "GlobalAveragePool" ? PoolingType::Average : PoolingType::Max,
                                           NDShape::Unknown(), strides, autoPadding, ceilOutDim, includePad, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "MaxRoiPool")
    {
        // ONNX spec is list of ints - however current IR spec is AttrType::AttributeProto_AttributeType_FLOATS
        std::vector<float> pooled_shape = GetNamedAttributeAsFloatVec(node, "pooled_shape");
        std::vector<size_t> dims = VecFloatToVecSize_t(pooled_shape);
        NDShape roiOutputShape(dims);

        float spatialScale = GetNamedAttributeAsFloat(node, "spatial_scale");
        FunctionPtr cntkFunction = ROIPooling(inputs[0], inputs[1],
                                              PoolingType::Max, roiOutputShape, spatialScale, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Conv")
    {
        return CreateCNTKConvNode(node, inputs);
    }
    else if (onnxOpName == "ConvTranspose")
    {
        return CreateCNTKConvTransposeNode(node, inputs);
    }
    else if (onnxOpName == "BatchNormalization" || onnxOpName == "SpatialBN")
    {
        auto is_test = GetNamedAttributeAsInt64(node, "is_test", 0);
        if (is_test == 0)
            NOT_IMPLEMENTED;
        // TODO: implement this right once ready.
        const Variable &operand = inputs[0];
        const Variable &scale = inputs[1];
        const Variable &bias = inputs[2];
        const Variable &runningMean = inputs[3];
        const Variable &runningInvStd = inputs[4];
        const Variable &runningCount = Constant::Scalar(0.0F);

        bool spatial = onnxOpName == "SpatialBN" || GetNamedAttributeAsInt64(node, "spatial", 1) != 0;

        double normalizationTimeConstant = 0.0;
        float momentum = GetNamedAttributeAsFloat(node, "momentum", 0.9f);
        if ((momentum > (1.0f - std::numeric_limits<float>::epsilon())) &&
            (momentum < (1.0f + std::numeric_limits<float>::epsilon())))
            normalizationTimeConstant = INFINITY;
        else if (momentum > 0.0f)
            normalizationTimeConstant = -48.0f / log1p(momentum - 1.0f);
        else
            normalizationTimeConstant = 0.0;

        // TODO: avoid hardcoded values
        double blendTimeConstant = 0;
        double epsilon = static_cast<double>(GetNamedAttributeAsFloat(node, "epsilon", 0.00001f));
        bool useCuDNNEngine = true;
        if ((epsilon < (0.00001f - std::numeric_limits<float>::epsilon())))
        {
            // REVIEW SPTIWARI: We are leaving some buffer in comparing with 1e-5 in the "if" condition above,
            // because 1e-5 is a common value for epsilon (ONNX default) and we do not want the model
            // to run slow for this common case because of any floating point differences. But for anything
            // clearly lower than 1e-5, we will not use cuDNN's batch normalization engine, because it floors
            // epsilon at 1e-5, and that can produce wrong numbers. For the special case when epsilon happens 
            // to be within (1e-5 , 1e-5 - std::numeric_limits<float>::epsilon()] range, cuDNN engine will be
            // used but it will print a warning that it is flooring epsilon to 1e-5.
            fprintf(stderr, "Epsilon = %0.7f, which is < 1e-5. CuDNN engine cannot be used for Batch Normalization. Could be slow.", epsilon);
            useCuDNNEngine = false;
        }
        bool disableRegularization = false;
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
                                                      disableRegularization,
                                                      ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Gemm")
    {
        float alpha = GetNamedAttributeAsFloat(node, "alpha", 1.0f);
        float beta = GetNamedAttributeAsFloat(node, "beta", 1.0f);
        float transA = GetNamedAttributeAsInt64(node, "transA", 0);
        float transB = GetNamedAttributeAsInt64(node, "transB", 0);
        float broadcast = GetNamedAttributeAsInt64(node, "broadcast", 0);
        // All the three inputs are expected to be rank=2 matrices. Only C, i.e. inputs[2]
        // can be a scalar or vector, and if the 'broadcast' attribute is non-zero then
        // we will use CNTK's automatic braodcast capability to broadcast C. But if rank < 2
        // and 'broadcast' attribute is zero, then we error out because 'broadcast' attribute
        // value is considered imperative.
        Variable input0 = inputs[0];
        Variable input1 = inputs[1];
        Variable input2 = inputs[2];
        auto cDims = input2.Shape().Dimensions();
        bool hasSingletonDim = std::any_of(cDims.begin(), cDims.end(), [](size_t i) { return i == 1; });
        if (broadcast == 0 && hasSingletonDim) // Bad ONNX node - such model/node shouldn't be serializable in ONNX at all.
            LogicError("The rank of input C in GEMM operator (A*B + C) is not 2. Either specify a value with rank=2, or set the broadcast attribute to 1.");

        FunctionPtr A = ElementTimes(input0, Constant(NDShape({1, 1}), DataType::Float, static_cast<double>(alpha)));
        FunctionPtr C = ElementTimes(input2, Constant(NDShape({1, 1}), DataType::Float, static_cast<double>(beta)));
        if (!transA && transB && broadcast) // Special case: Equivalent to FC (fully-connected) op. Takes in account broadcast of B, if needed.
        {
            return CreateCNTKFCNode(ToWString(node->Name()), {(Variable) A, input1, (Variable) C});
        }
        FunctionPtr B = (transB != 0) ? Transpose(input1) : (FunctionPtr) input1;
        FunctionPtr D = (transA != 0) ? Times(B, Transpose(A)) : Times(B, A);
        // If needed, Plus op will broadcast C automatically.
        return Plus(C, D);
    }
    else if (onnxOpName == "Dropout")
    {
        const Variable &operand = inputs[0];
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
        const NDShape &shape = GetNamedAttributeAsShape(node, "shape", false);
        DataType dataType = DataType::Float;
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = NormalRandom(shape, dataType, mean, scale, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniformLike")
    {
        const Variable &operand = inputs[0];
        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = UniformRandomLike(operand, low, high, seed, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormalLike")
    {
        const Variable &operand = inputs[0];
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
    else if (onnxOpName == "And")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementAnd(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Not")
    {
        FunctionPtr cntkFunction = ElementNot(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Or")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementOr(input0, input1, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Xor")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementXor(input0, input1, ToWString(node->Name()));
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
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 0.01F));
        FunctionPtr cntkFunction = LeakyReLU(inputs[0], alpha, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Selu")
    {
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 1.6732F));
        double gamma = static_cast<double>(GetNamedAttributeAsFloat(node, "gamma", 1.0507F));
        FunctionPtr cntkFunction = SELU(inputs[0], gamma, alpha, ToWString(node->Name()));
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
        FunctionPtr cntkFunction = Pow(inputs[0], inputs[1], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "MatMul")
    {
        FunctionPtr cntkFunction = Times(inputs[1], inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "PRelu")
    {
        FunctionPtr cntkFunction = PReLU(inputs[1], inputs[0], ToWString(node->Name()));
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
    else if (onnxOpName == "Hardmax")
    {
        if (inputs[0].IsConstant())
        {
            // onnx spec requires the first axis being batch axis for Hardmax op
            Variable input = Reshape(inputs[0], inputs[0].Shape().SubShape(0, inputs[0].Shape().Rank() - 1));
            FunctionPtr cntkFunction = Hardmax(input, ToWString(node->Name()));
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = Hardmax(inputs[0], ToWString(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "Softmax")
    {
        if (!HasNamedAttribute(node, "axis"))
        {
            FunctionPtr cntkFunction = Softmax(inputs[0], ToWString(node->Name()));
            return cntkFunction;
        }
        else
        {
            int index = static_cast<int>(GetNamedAttributeAsInt64(node, "axis", 0));
            Axis axis(index - 1);
            FunctionPtr cntkFunction = Softmax(inputs[0], axis, ToWString(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "LogSoftmax")
    {
        int index = static_cast<int>(GetNamedAttributeAsInt64(node, "axis", 0));

        Axis axis;
        if (index == 0)
        {
            axis = Axis::DefaultBatchAxis();
        }
        else
        {
            axis = Axis(index - 1);
        }

        FunctionPtr cntkFunction = LogSoftmax(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Softplus")
    {
        FunctionPtr cntkFunction = Softplus(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Softsign")
    {
        FunctionPtr cntkFunction = Softsign(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMax")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceMax(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMin")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceMin(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSum")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMean")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceMean(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceProd")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceProd(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceLogSumExp")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axes[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceL1")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        bool keepdims = GetNamedAttributeAsInt64(node, "keepdims", 1) == 1;
        FunctionPtr cntkFunction = ReduceL1(inputs[0], axes, keepdims, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceL2")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        bool keepdims = GetNamedAttributeAsInt64(node, "keepdims", 1) == 1;
        FunctionPtr cntkFunction = ReduceL2(inputs[0], axes, keepdims, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSumSquare")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        bool keepdims = GetNamedAttributeAsInt64(node, "keepdims", 1) == 1;
        FunctionPtr cntkFunction = ReduceSumSquare(inputs[0], axes, keepdims, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMax")
    {
        int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis");
        // -1 to compensate what ConvertAxisToCNTKCppApi assumes that axis is already decreased by 1
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkfunction = Argmax(inputs[0], axis, ToWString(node->Name()));
        return cntkfunction;
    }
    else if (onnxOpName == "ArgMin")
    {
        int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis");
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkFunction = Argmin(inputs[0], axis, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reshape")
    {
        // Skip reshape is it follows a LSTM.
        const Node *childNode = GetChildNode(node, &node->InputDefs()[0]);
        if (childNode != nullptr && childNode->OpType() == "LSTM")
        {
            // TODO: this is to undo reshape after LSTM in CNTKToONNX to workaround
            // output shape mismatch with input to the nexk LSTM layer.
            NDShape newShape = GetNamedAttributeAsShape(node, "shape", false);
            newShape = newShape.SubShape(0, newShape.Rank() - inputs[0].DynamicAxes().size());
            return Reshape(inputs[0], newShape, ToWString(node->Name()));
        }

        NDShape newShape = GetNamedAttributeAsShape(node, "shape", false);
        if (inputs[0].DynamicAxes().size() > 0)
        {
            if (newShape.Rank() == 1)
                LogicError("Reshape: 'shape' attribute must include element for batch axis.");
            newShape = newShape.SubShape(0, newShape.Rank() - inputs[0].DynamicAxes().size());
        }

        int inferredDim = -1;
        for (size_t i = 0; i < newShape.Dimensions().size(); ++i)
        {
            if (newShape[i] == 0)
                newShape[i] = inputs[0].Shape()[i];
            else if (newShape[i] == 0)
            {
                if (inferredDim == -1)
                    inferredDim = i;
                else
                    LogicError("Reshape: 'shape' contains more than one inferred dimension.");
            }
        }

        if (inferredDim != -1)
        {
            if (inputs[0].Shape().TotalSize() % newShape.TotalSize() != 0)
                LogicError("Reshape: 'shape' contains more than one inferred dimension.");
            int inferredDimSize = inputs[0].Shape().TotalSize() / newShape.TotalSize();
            newShape[inferredDim] = inferredDimSize;
        }

        FunctionPtr cntkFunction = Reshape(inputs[0], newShape, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Concat")
    {
        // We allow the 'axis' attribute to be optional, and not required (as
        // given in Concat's ONNX spec), to be consistent with other frameworks.
        // 'axis' can be enforced as a required attribute, if needed.
        int64_t onnxAxis = GetNamedAttributeAsInt64(node, "axis", 0);
        Axis axis = ConvertONNXAxisToCNTKCppApi(onnxAxis, inputs[0]);
        std::vector<Variable> fixedInputs;
        if (FixConstantShapeForConstantVariableInputPair(inputs, fixedInputs))
        {
            FunctionPtr cntkFunction = Splice(fixedInputs, axis, ToWString(node->Name()));
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = Splice(inputs, axis, ToWString(node->Name()));
            return cntkFunction;
        }
    }
    // { L"", "Split)
    else if (onnxOpName == "Slice")
    {
        // axes is optional so provide a default
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);

        std::vector<int64_t> starts64 = GetNamedAttributeAsInt64Vec(node, "starts");
        std::vector<int64_t> ends64 = GetNamedAttributeAsInt64Vec(node, "ends");

        if (starts64.size() != ends64.size())
        {
            LogicError("starts (of size %d) and ends (of size %d) attributes of Slice operation must be the same size.",
                       (int) starts64.size(), (int) ends64.size());
        }

        std::vector<int> starts = VecInt64ToVecInt(starts64);
        std::vector<int> ends = VecInt64ToVecInt(ends64);

        if (axes.empty())
        {
            for (int i = 0; i < starts.size(); i++)
            {
                Axis axis(i);
                axes.push_back(axis);
            }
        }

        bool workaroundONNXRT = false;
        if (workaroundONNXRT)
        {
            axes.erase(axes.begin());
            starts.erase(starts.begin());
            ends.erase(ends.begin());
        }
        FunctionPtr cntkFunction = Slice(inputs[0], axes, starts, ends, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Transpose")
    {
        std::vector<int64_t> permutation = GetNamedAttributeAsInt64Vec(node, "perm");
        std::vector<Axis> argsortedPermutation = ConvertPermutationONNXToCNTK(permutation, inputs[0].HasBatchAxis(), inputs[0].HasSequenceAxis());
        FunctionPtr cntkFunction = Transpose(inputs[0], argsortedPermutation, ToWString(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Pad")
    {
        std::vector<int64_t> pads = GetNamedAttributeAsInt64Vec(node, "pads");
        if (inputs[0].HasBatchAxis())
        {
            pads.erase(pads.begin() + pads.size() / 2);
            pads.erase(pads.begin());
        }

        if (pads.size() != 2 * inputs[0].Shape().Rank())
            LogicError("Pad: Incorrect length of 'pads' attribute in Pad op. Length of 'pads' attribute should be twice the number of dimensions in input tensor.");
        auto padsPair = SplitAndReverseVec(pads);

        CNTK::PaddingMode cntkPaddingMode;
        double cntkConstantValue = 0.0;
        auto mode = GetNamedAttributeAsString(node, "mode", "constant");
        std::transform(mode.begin(), mode.end(), mode.begin(), [](char v) { return (char) ::tolower(v); });
        if (mode == "constant")
            cntkPaddingMode = CNTK::PaddingMode::CONSTANTPAD;
        else if (mode == "reflect")
            cntkPaddingMode = CNTK::PaddingMode::REFLECTPAD;
        else if (mode == "edge")
            NOT_IMPLEMENTED
        else
            LogicError("Pad: Invalid 'mode' attribute value, %s, specified for Pad node.", mode.c_str());

        if (cntkPaddingMode == CNTK::PaddingMode::CONSTANTPAD)
            cntkConstantValue = static_cast<double>(GetNamedAttributeAsFloat(node, "value", 0.0));

        FunctionPtr cntkPadFunction = Pad(inputs[0],
                                          cntkPaddingMode,
                                          padsPair.first,
                                          padsPair.second,
                                          cntkConstantValue,
                                          ToWString(node->Name()));

        return cntkPadFunction;
    }
    else if (onnxOpName == "Gather")
    {
        if (HasNamedAttribute(node, "axis"))
        {
            int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis", 0);
            Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
            return GatherOp(inputs[1], inputs[0], axis, ToWString(node->Name()));
        }
        else
        {
            FunctionPtr cntkFunction = GatherOp(inputs[1], inputs[0], ToWString(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "DepthToSpace")
    {
        auto blockSize = GetNamedAttributeAsInt64(node, "blocksize", 1);
        return DepthToSpace(inputs[0], static_cast<size_t>(blockSize), ToWString(node->Name()));
    }
    else if (onnxOpName == "SpaceToDepth")
    {
        auto blockSize = GetNamedAttributeAsInt64(node, "blocksize", 1);
        return SpaceToDepth(inputs[0], static_cast<size_t>(blockSize), ToWString(node->Name()));
    }
    else if (onnxOpName == "Squeeze")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxes(node, "axes");
        return Squeeze(inputs[0], axes, ToWString(node->Name()));
    }
    else if (onnxOpName == "ImageScaler")
    {
        double scale = GetNamedAttributeAsFloat(node, "scale", 1);
        std::vector<float> bias = GetNamedAttributeAsFloatVec(node, "bias");
        const NDShape &inputShape = inputs[0].Shape();
        if (inputShape.Rank() == 4 && inputShape[3] == 1)
        {
            // CNTK does not have batch axis in shape dimensions.
            Variable inputReshaped = Reshape(inputs[0], inputShape.SubShape(0, 3));
            return ImageScaler(inputReshaped, scale, bias, ToWString(node->Name()));
        }
        else
        {
            return ImageScaler(inputs[0], scale, bias, ToWString(node->Name()));
        }
    }
    else if (onnxOpName == "MeanVarianceNormalization")
    {
        // REVIEW: ONNX MeanVarianceNormalization spec does not have an 'epsilon' attribute.
        // But corresponding CNTK node does. We construct the CNTK node with default value of epsilon
        // when loading the ONNX MeanVarianceNormalization node in CNTK.
        size_t acrossChannels = GetNamedAttributeAsInt64(node, "across_channels", 0);
        size_t normalizeVariance = GetNamedAttributeAsInt64(node, "normalize_variance", 1);
        return MeanVarianceNormalization(inputs[0], !!acrossChannels, !!normalizeVariance, ToWString(node->Name()));
    }
    else if (onnxOpName == "Identity")
    {
        FunctionPtr cntkFunction = Alias(inputs[0], ToWString(node->Name()));
        return cntkFunction;
    }
    else
    {
        LogicError("ONNX (%s) is not supported in CNTK", onnxOpName.c_str());
        return nullptr;
    }
}

std::pair<const Node *, int> FindParent(const Node *node)
{
    Node::NodeConstIterator it = node->OutputNodes_begin();
    if (it != node->OutputNodes_end())
    {
        const Node *parent = *it;
        int index = 0;
        for (auto nodeArg : parent->InputDefs())
        {
            // TODO: Check whether we should use node output arg name for the check below.
            if (nodeArg.Name() == node->Name())
            {
                return std::make_pair(parent, index);
            }
            index++;
        }
    }
    return std::make_pair(nullptr, -1);
}

std::pair<const Node *, int> FindParentAndChildIndex(const Node *node)
{
    Node::NodeConstIterator it = node->OutputNodes_begin();
    if (it != node->OutputNodes_end())
    {
        const Node *parent = *it;
        int index = 0;
        for (auto nodeArg : parent->InputDefs())
        {
            // TODO: Check whether we should use node output arg name for the check below.
            if (nodeArg.Name() == node->Name())
            {
                return std::make_pair(parent, index);
            }
            index++;
        }
    }
    return std::make_pair(nullptr, -1);
}

std::vector<FunctionPtr> ONNXToCNTKHelper::FromONNXNode(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                        ONNXToCNTKVariableMap &constructedNodeArgVariableMap,
                                                        const Graph *graph, const DeviceDescriptor &computeDevice)
{
    auto nodeOpStr = node->OpType();
    ONNXToCNTKMap::iterator itONNXToCNTKMap = constructedNodeMap.find(node);
    if (itONNXToCNTKMap != constructedNodeMap.end())
    {
        return std::vector<FunctionPtr>({itONNXToCNTKMap->second});
    }

    std::vector<Variable> inputs = CreateCNTKInputs(node, constructedNodeMap, constructedNodeArgVariableMap, graph, computeDevice);

    // Special check if node belongs to the subgraph created by CNTK's OptimizedRNNStack export.
    std::vector<FunctionPtr> lstmCntkFunction;
    bool isOptimizedRnnStack(false);
    std::tie<bool, std::vector<FunctionPtr>>(isOptimizedRnnStack, lstmCntkFunction) =
        CheckNodeBelongsToOptimizedRnnStack(node, inputs, constructedNodeMap, constructedNodeArgVariableMap, graph, computeDevice);
    if (isOptimizedRnnStack)
        return lstmCntkFunction;

    //
    const Node *parentNode;
    int childIndex;
    std::tie<const Node *, int>(parentNode, childIndex) = FindParentAndChildIndex(node);
    if (parentNode != nullptr && Operators::IsRNNOp(parentNode->OpType()))
    {
        std::vector<FunctionPtr> cntkFunctions = CreateRNNConstantOp(graph, node, parentNode, childIndex, computeDevice);
        if (!cntkFunctions.empty())
        {
            // TODO: make node map to vector of FunctionPtr
            constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, cntkFunctions));
        }
        return cntkFunctions;
    }
    else
    {
        FunctionPtr cntkFunction = CreateCNTKNode(node, inputs, computeDevice);
        constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, std::vector<FunctionPtr>({cntkFunction})));
        return std::vector<FunctionPtr>({cntkFunction});
    }
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs,
                                             const DeviceDescriptor &computeDevice)
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

Variable ONNXToCNTKHelper::GetNodeOperandWithPaddingResolved(std::vector<bool> &cntkConvAutoPadding,
                                                             NDShape &strides, const Node *node, const std::vector<Variable> &inputs, const double padValue)
{
    bool hasAutoPad = HasNamedAttribute(node, "auto_pad");
    bool hasPads = HasNamedAttribute(node, "pads");
    Variable operand = inputs[0];
    Variable convOperand = operand; // Important initial condition.
    if (hasAutoPad && hasPads)
    {
        LogicError("Ambiguous conv node specification. Both %s and %s attributes are specified. Only one of the two should be specified.",
                   "auto_pad", "pads");
    }
    else if (hasAutoPad)
    {
        ConvAutoPadType auto_pad = ConvertStrToConvAutoPadType(GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER"));
        switch (auto_pad)
        {
        case ConvAutoPadType::SAME_LOWER:
        {
            cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), false);
            NDShape kernelShape = GetNamedAttributeAsShape(node, "kernel_shape", false);
            convOperand = (Variable) CreatePadOpForSameLowAutoPad(inputs[0], kernelShape, strides, padValue);
        }
        break;
        case ConvAutoPadType::SAME_UPPER:
            cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), true);
            break;
        case ConvAutoPadType::VALID:
            cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), false);
            break;
        }
    }
    else if (hasPads)
    {
        // If 'pads' is specified, we pad the node and then do 'valid' convolution.
        std::vector<int64_t> pads = GetNamedAttributeAsInt64Vec(node, "pads");
        bool paddingNeeded = std::any_of(pads.begin(), pads.end(), [](int64_t i) { return i > 0; });
        if (paddingNeeded)
        {
            // Create appropriate pad node.
            auto padsPair = AdjustONNXPadsVecForCNTKPadOp(operand, pads);
            auto nodeName = node->Name().empty() ? node->Name() : node->Name() + std::string("_pad");
            FunctionPtr cntkPadFunction = Pad(operand,
                                              CNTK::PaddingMode::CONSTANTPAD,
                                              padsPair.first,
                                              padsPair.second,
                                              padValue,
                                              ToWString(nodeName));
            convOperand = (Variable) cntkPadFunction;
        }
        cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), false); // For 'VALID' convolution
    }
    else
    {
        // REVIEW SPTIWARI: Ideally this should not happen. ONNX spec says that one
        // and only one of these two attributes MUST be present. However, we are handling
        // this case leniently for now and assuming that there's no padding and behavior
        // is the same as when auto_pad == VALID.
        cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), false); // For 'VALID' convolution
    }

    AdjustAutoPaddingAndStrideForCNTKSpecialCases(operand, cntkConvAutoPadding, strides);
    return convOperand;
}

FunctionPtr ONNXToCNTKHelper::CreatePadOpForSameLowAutoPad(
    const Variable &input, NDShape kernelShape, NDShape strides, const double padValue)
{
    NDShape inputShape = input.Shape();
    std::vector<int> pads;
    for (int dim = 0; dim < kernelShape.Rank(); dim++)
    {
        // Padding could be calcualted as: p = s - (w - f) % s.
        // however, it does not ensure input size being multiplier of output size.
        // The above calculation failed with the yolo model. ONNX spec is not clear on this.
        // This following padding computation ensures: input_size = output_size * stride.
        int f = kernelShape[dim];
        int s = strides[dim];
        pads.push_back(f - s);
    }

    if (std::all_of(pads.begin(), pads.end(), [](int pad) { return (pad == 0); }))
        return input;

    if (pads.size() < inputShape.Rank())
    {
        for (int dim = pads.size(); dim < inputShape.Rank(); dim++)
            pads.push_back(0);
    }

    std::vector<size_t> begins, ends;
    for (int dim = 0; dim < pads.size(); dim++)
    {
        // SameLow: the lower (begin) side get one extra pad if total pads is odd.
        int p = pads[dim];
        int endPad = p / 2;
        ends.push_back(endPad);
        int beginPad = p - endPad;
        begins.push_back(beginPad);
    }

    CNTK::PaddingMode cntkPaddingMode = CNTK::PaddingMode::CONSTANTPAD;

    // TODO: Pad op is not intuitative or it could be a bug. One would think begins before end.
    FunctionPtr cntkPadFunction = Pad(input,
                                      cntkPaddingMode,
                                      ends,
                                      begins,
                                      padValue);

    return cntkPadFunction;
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKConvTransposeNode(const Node *node, const std::vector<Variable> &inputs)
{
    Variable inputOperand = inputs[0];
    Variable convolutionMap = inputs[1];

    NDShape strides = GetNamedAttributeAsShape(node, "strides", false);
    NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, {1});
    std::vector<bool> sharing({true});
    size_t reductionRank = 1;
    size_t maxTempMemSizeInSamples = 0;

    if (HasNamedAttribute(node, "output_shape"))
    {
        std::vector<bool> cntkConvAutoPadding;
        NDShape outputShape = GetNamedAttributeAsShape(node, "output_shape", true);
        NDShape inputShape = inputOperand.Shape();
        NDShape kernelShape = convolutionMap.Shape();

        for (int axis = 0; axis < outputShape.Rank(); axis++)
        {
            if (axis != outputShape.Rank() - 1)
            {
                int pads = (inputShape[axis] - 1) * strides[axis] + kernelShape[axis] - outputShape[axis];
                cntkConvAutoPadding.push_back(pads > 0);
            }
            else
            {
                // We assume this is the channel dimension and since ONNX does not support
                // padding (also strides, dilation) for channel dimension, we set this to
                // false when creating CNTK node.
                cntkConvAutoPadding.push_back(false);
            }
        }

        return ConvolutionTranspose(
            convolutionMap,
            inputs[0],
            strides,
            sharing,
            cntkConvAutoPadding,
            outputShape,
            dilation,
            reductionRank,
            maxTempMemSizeInSamples,
            ToWString(node->Name()));
    }
    else if (HasNamedAttribute(node, "pads"))
    {
        NOT_IMPLEMENTED;
    }
    else if (HasNamedAttribute(node, "auto_pad"))
    {
        NOT_IMPLEMENTED;
    }

    return nullptr;
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKConvNode(const Node *node, const std::vector<Variable> &inputs)
{
    NDShape strides = GetNamedAttributeAsShape(node, "strides", false);
    NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, {1});
    // TODO: avoid hardcoded values
    std::vector<bool> sharing({true});
    size_t reductionRank = 1;
    size_t maxTempMemSizeInSamples = 0;
    size_t groups = GetNamedAttributeAsInt64(node, "group", 1);

    Variable convolutionMap = inputs[1];

    std::vector<bool> cntkConvAutoPadding;
    auto convOperand = GetNodeOperandWithPaddingResolved(/*output arg first*/ cntkConvAutoPadding, strides, node, inputs);
    FunctionPtr cntkConvFunction = Convolution(
        convolutionMap,
        convOperand,
        strides,
        sharing,
        cntkConvAutoPadding,
        dilation,
        reductionRank,
        groups,
        maxTempMemSizeInSamples,
        ToWString(node->Name()));

    // TODO: support bias in CNTK op.
    if (inputs.size() == 3)
    {
        NDShape shape({1, 1, inputs[2].Shape()[0]});
        return Plus(cntkConvFunction, Reshape(inputs[2], shape));
    }
    else
        return cntkConvFunction;
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKFCNode(const std::wstring &nodeName, const std::vector<Variable> &inputs)
{
    // TODO: this is experimental code to load Facebook Caffe models.
    // "FC" is not in ONNX standard. Two cases need to be handled with
    // this type of Caffe model.
    // 1. Make trailing dimensions of operand 1 matches the heading dimensions of operant 2.
    //  For example, with shape [1, dim0, dim1] * [dim2, dim3], we need to reshape
    //  first operand to [1, dim0 * dim1] In this case dim0 * dim1 has to be equal to dim2.
    // 2. Broadcase bias if needed.
    Variable input0 = inputs[0], input1 = inputs[1];
    input0 = Reshape(input0, {1, input0.Shape().TotalSize()});
    FunctionPtr cntkFunction = Times(input0, input1, nodeName);
    cntkFunction = Reshape(cntkFunction, {cntkFunction->Output().Shape().TotalSize()});
    cntkFunction = Plus(cntkFunction, inputs[2], nodeName);
    return cntkFunction;
}

FunctionPtr ONNXToCNTK::CreateGraph(ONNXIR::Graph *src, const DeviceDescriptor &computeDevice)
{
    FunctionPtr cntkModel;

    // To use depth-first-traversal, keeps a collection of visited nodes.
    ONNXToCNTKMap constructedFunctions;
    ONNXToCNTKVariableMap constructedNodeArgVariableMap;
    for (Graph::NodeIterator it = src->Nodes_begin(); it != src->Nodes_end(); ++it)
    {
        const Node *node = *it;

        if (constructedFunctions.find(node) == constructedFunctions.end())
        {
            std::vector<FunctionPtr> cntkNode = ONNXToCNTKHelper::FromONNXNode(node,
                                                                               constructedFunctions, constructedNodeArgVariableMap, src, computeDevice);
        }
    }

    // ONNX puts all outputs in an graph as input to the "_Graph_Sink" node.
    ONNXToCNTKMap::iterator itNodeFn = std::find_if(constructedFunctions.begin(), constructedFunctions.end(),
                                                    [](ONNXToCNTKMap::value_type nodeFn) { return nodeFn.first->Name() == "_Graph_Sink"; });
    if (itNodeFn == constructedFunctions.end())
    {
        return nullptr;
    }

    std::vector<FunctionPtr> functions;
    for (Node::NodeConstIterator it = itNodeFn->first->InputNodes_begin(); it != itNodeFn->first->InputNodes_end(); ++it)
    {
        // TODO: consulting ONNXIR to see how to do this solidly.
        // https://msasg.visualstudio.com/DefaultCollection/Shared%20Data/AIToolkits-CNTK/_queries?id=1134732&_a=edit&triage=true
        std::vector<FunctionPtr> &constructedFuncts = constructedFunctions[*it];
        for (int index = 0; index < constructedFuncts.size(); index++)
        {
            FunctionPtr &constructedFunct = constructedFuncts[index];
            if (constructedFunct->RootFunction()->OpName() != L"Combine")
                functions.insert(functions.end(), constructedFunct);
        }
    }

    if (functions.empty())
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

std::vector<Variable> ONNXToCNTKHelper::CreateCNTKInputsStartingFromIndex(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                                          ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, size_t startIndex, const DeviceDescriptor &computeDevice)
{
    std::vector<Variable> inputs;
    const std::vector<NodeArg> &inputDefs = node->InputDefs();
    for (std::vector<NodeArg>::const_iterator it = inputDefs.begin() + startIndex; it != inputDefs.end(); ++it)
    {
        const NodeArg *nodeArg = &(*it);
        const Node *inputNode = GetChildNode(node, &(*it));
        if (inputNode != nullptr)
        {
            ONNXToCNTKMap::iterator itNodeMap = constructedNodeMap.find(const_cast<Node *>(inputNode));
            if (itNodeMap != constructedNodeMap.end())
            {
                inputs.insert(inputs.end(), itNodeMap->second.begin(), itNodeMap->second.end());
            }
            else
            {
                std::vector<FunctionPtr> inputVariables = FromONNXNode(inputNode, constructedNodeMap,
                                                                       constructedNodeArgVariableMap, graph, computeDevice);
                inputs.insert(inputs.end(), inputVariables.begin(), inputVariables.end());
            }
        }
        else
        {
            std::string parentONNXOpName = node->OpType();
            if (Operators::IsRNNOp(node->OpType()))
            {
                std::vector<Variable> inputVariables =
                    CreateRNNLeafVariableOrConstant(nodeArg, node, graph, constructedNodeArgVariableMap, computeDevice);
                inputs.insert(inputs.end(), inputVariables.begin(), inputVariables.end());
            }
            else
            {
                Variable inputVariable = CreateLeafVariableOrConstant(nodeArg, node, graph, computeDevice);
                inputs.push_back(inputVariable);
            }
        }
    }
    return inputs;
}

std::vector<Variable> ONNXToCNTKHelper::CreateCNTKInputs(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                         ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, const DeviceDescriptor &computeDevice)
{
    return CreateCNTKInputsStartingFromIndex(node, constructedNodeMap, constructedNodeArgVariableMap, graph, 0, computeDevice);
}

std::pair<bool, std::vector<FunctionPtr>> ONNXToCNTKHelper::CheckNodeBelongsToOptimizedRnnStack(const Node *node, const std::vector<Variable> &inputs,
                                                                                                ONNXToCNTKMap &constructedNodeMap, ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, const DeviceDescriptor &computeDevice)
{
    std::vector<FunctionPtr> lstmCntkFunction;
    bool isOptimizedRnnStack(false);
    const string shapeAdaptorFirstOpName = "Transpose"; // First op in the shape adaptor needed for ONNX LSTM.

    // The idea is that any time you see a Transpose node, there's a posibility that it was created
    // as part of export for CNTK's OptimizedRNNStack. We then check for that possibility and if it
    // is confirmed, then we do not add this Transpose node or the next Reshape node, but directly
    // create the next LSTM node (that follows the Reshape node). This special handling is needed
    // because ONNX spec, as of this comment, specifies the shape of the output of LSTM in such a
    // way that it cannnot be directly fed into the next LSTM. Therefore, during OptimizedRNNStack
    // export, when there are multiple layers, we have to insert Transpose and Reshape nodes to modify
    // the output of the previous layer to match the required shape of the input of the next layer.
    // When loading such model back in CNTK, since we really do not need these Transpose and Reshape
    // nodes (because CNTK's LSTM node's input/output shapes agree/match). So as an optimization
    // (and also because it is complicated to Transpose two axes, one of which is the batch axis,
    // in CNTK) we detect this special subgraph and if it is detected, we skip Transpose and Reshape
    // nodes.
    if (node->OpType() == shapeAdaptorFirstOpName)
    {
        std::vector<int64_t> permutation = GetNamedAttributeAsInt64Vec(node, "perm");
        const Node *firstParentNode(nullptr), *grandParentNode(nullptr);
        Node::NodeConstIterator it = node->OutputNodes_begin();
        if (it != node->OutputNodes_end())
        {
            firstParentNode = *it;
        }
        if (firstParentNode != nullptr)
        {
            it = firstParentNode->OutputNodes_begin();
            if (it != node->OutputNodes_end())
            {
                grandParentNode = *it;
            }
        }

        // This is the check that detects the special case of OptimizedRNNStac export. Criteria is as follows:
        // 1. Current node is Transpose.
        // 2. Parent node is Reshape.
        // Grandparent node is LSTM.
        // 'perm' attribute of the current Transpose node is length 4 (because that is the dimensionality of ONNX LSTM node output).
        // First input to Transpose node is rank 1 (that is the dimensionality of output of CNTK's LSTM node).
        if ((firstParentNode != nullptr) && (grandParentNode != nullptr) && firstParentNode->OpType() == "Reshape" && Operators::IsRNNOp(grandParentNode->OpType()) &&
            permutation.size() == 4 && inputs[0].Shape().Rank() == 1)
        {
            std::vector<Variable> inputsnextLSTMInputs = CreateCNTKInputsStartingFromIndex(grandParentNode, constructedNodeMap,
                                                                                           constructedNodeArgVariableMap, graph, 1, computeDevice);
            inputsnextLSTMInputs.insert(inputsnextLSTMInputs.begin(), inputs[0]);
            FunctionPtr cntkFunction = CreateCNTKNode(grandParentNode, inputsnextLSTMInputs, computeDevice);
            lstmCntkFunction.push_back(cntkFunction);
            constructedNodeMap.insert(ONNXToCNTKMap::value_type(grandParentNode, lstmCntkFunction));
            isOptimizedRnnStack = true;
        }
    }
    return std::make_pair(isOptimizedRnnStack, lstmCntkFunction);
}
