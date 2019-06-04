//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

#include "Utils.h"
#include "Operators.h"
#include <algorithm>
#include <iostream>
#include "RNNHelper.h"
#include "ONNXToCNTK.h"

using namespace onnxruntime;
using namespace CNTK;
using namespace CNTK::ONNX;
using namespace onnx;
using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
bool IsONNX1_2Supported();

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
                                                 const Graph *graph, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr,
        const DeviceDescriptor &computeDevice);

    static std::string model_location_;
private:
    static FunctionPtr CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs, const Graph *graph,
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr,
                                      const DeviceDescriptor &computeDevice);
    static std::vector<size_t> GetNodeDims(const Node *node);
    static Constant CreateConstant(const Node *node, const DeviceDescriptor &computeDevice);
    static Constant CreateConstant(const onnx::TensorProto &valueProto, const std::string &nodeName,
                                   const DeviceDescriptor &computeDevice);
    template <typename TDst, typename TSrc>
    static const CNTK::Constant CreateConstantWithTensorData(CNTK::NDShape &shape, google::protobuf::int32 tensorProtoDataType,
                                                             CNTK::DataType cntkDataType, const TSrc *srcData, CNTK::NDShape &reversedShape,
                                                             const CNTK::DeviceDescriptor &computeDevice, const std::string &nodeName);

    static Variable CreateLeafVariableOrConstant(const NodeArg *nodeArg, const Node *parentNode, const Graph *graph,
                                                 const DeviceDescriptor &computeDevice);
    static std::vector<Variable> CreateRNNLeafVariableOrConstant(const NodeArg *nodeArg,
                                                                 const Node *parentNode, const Graph *graph,
                                                                 ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const DeviceDescriptor &computeDevice);
    static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs, const Graph *graph, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr);
    static FunctionPtr CreateFunction(const Node *node, const std::vector<Variable> &inputs, const Graph *graph, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const Variable& inputPlaceholder);

    static bool IsSecondInputOfElementWiseOpsWithBroadcast(const Node *parentNode, const NodeArg *nodeArg);
    static bool FixConstantShapeForConstantVariableInputPair(const std::vector<Variable> &inputs,
                                                             std::vector<Variable> &fixedInputs);

    static const Node *GetChildNode(const Node *parentNode, const NodeArg *nodeArg, int &nodeArgIndex);

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

    static std::vector<size_t> VecInt64ToVecSize_t(const std::vector<int64_t> &vecFloat);

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

    static std::pair<std::vector<size_t>, std::vector<size_t>> SplitAndReverseVec(std::vector<int64_t> &pads);
    static std::pair<std::vector<size_t>, std::vector<size_t>> AdjustONNXPadsVecForCNTKPadOp(const Variable &operand, std::vector<int64_t> &pads);
    static NDShape ReverseShape(const NDShape &shape);

    static std::pair<std::vector<Axis>, bool> GetReduceElementsAttributes(const Node *node, const Variable &input);

    static std::pair<Variable, Variable> BroadcastElementWiseInput(const Node *node,
                                                                   const Variable &input0, const Variable &input1);

    static Variable GetNodeOperandWithPaddingResolved(std::vector<bool> &cntkConvAutoPadding,
        NDShape &strides, const Node *node, const Variable& dataOperand, const double padValue = 0.0);
    //
    // This method computes pad values for Convolution/Pooling operations under ONNX SAME_LOWER or SAME_UPPER auto_pad. 
    // Note: the shape format for inputWithBatchAxisShape is [N x C x D1 x D2 ... Dn], which includes batch axis.
    //
    static std::pair<std::vector<size_t>, std::vector<size_t>> CalcPaddingForSameLowerOrUpperAutoPad(
        const NDShape& inputWithBatchAxisShape, const NDShape& kernelShape, const NDShape& strides, bool isSameUpper);
    //
    // This method computes pad values for ConvolutionTranspose operations under ONNX SAME_LOWER or SAME_UPPER auto_pad, or output_Shape. 
    //
    static std::pair<std::vector<size_t>, std::vector<size_t>> CalcPaddingFromOutputShape(
        const NDShape& inputShape, const NDShape& kernelShape, const NDShape& strides, const NDShape& outputShape, const std::vector<int64_t>& outputPadding, bool isSameUpper);
    static std::pair<std::vector<size_t>, std::vector<size_t>> SplitONNXPads(const std::vector<size_t>& pads, bool isSameUpper);
    static std::tuple<bool, bool, bool> ConfigureConvTransposeNodePaddingOption(const Node *node);
    //
    // CNTK convolution/pooling operations do not support ONNX same_low padding.
    // This method does padding accoordingly before invoking
    // Convolution/Pooling operations.
    //
    static FunctionPtr CreatePadOpForSameLowerOrUpperAutoPad(
        const Variable &input, const NDShape& kernelShape, const NDShape& strides, const double padValue, bool isSameUpper);
    static FunctionPtr CreateCNTKConvNode(const Node *node, const std::vector<Variable> &inputs);
    static FunctionPtr CreateCNTKConvTransposeNode(const Node *node, const std::vector<Variable> &inputs);
    static FunctionPtr CreateCNTKConvTransposeNode(const Variable& inputOperand, const Variable& convolutionMap,
        const NDShape& strides, const std::vector<bool>& sharing, const std::vector<size_t>& lowerPad,
        const std::vector<size_t>&  upperPad, const NDShape& outputShape, const NDShape& dilation, size_t reductionRank, 
        size_t maxTempMemSizeInSamples, const std::string& name);
    static FunctionPtr CreateCNTKFCNode(const std::wstring &nodeName, const std::vector<Variable> &inputs);

    //
    // Methods for creating CNTK input variables for a given node.
    //
    static std::vector<Variable> CreateCNTKInputsStartingFromIndex(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                                   ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, size_t startIndex, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const DeviceDescriptor &computeDevice);
    static std::vector<Variable> CreateCNTKInputs(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                  ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const DeviceDescriptor &computeDevice);

    //
    // Method for special checking if an ONNX node belongs to special subgraph that is created by OptimizedRNNStack export.
    //
    static std::pair<bool, std::vector<FunctionPtr>> CheckNodeBelongsToOptimizedRnnStack(const Node *node, const std::vector<Variable> &inputs,
                                                                                         ONNXToCNTKMap &constructedNodeMap, 
        ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, 
        VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr,
        const DeviceDescriptor &computeDevice);

    static ConvAutoPadType ConvertStrToConvAutoPadType(const string &str);

    static std::vector<int64_t> GetShapeFromInput(const NodeArg *shapeInput, const Graph *graph);
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

CNTK::DataType ONNXToCNTKHelper::FromONNXType(onnx::TypeProto type)
{
    switch (type.tensor_type().elem_type())
    {
        // CNTK only support ONNX float and double data types.
        // For ops that take data types other than float and double,
        // CNTK will accept these data types as float.
    case onnx::TensorProto_DataType_INT64:
    case onnx::TensorProto_DataType_INT32:
    case onnx::TensorProto_DataType_BOOL:
    case onnx::TensorProto_DataType_FLOAT:
        return CNTK::DataType::Float;
    case onnx::TensorProto_DataType_FLOAT16:
        return CNTK::DataType::Float16;
    case onnx::TensorProto_DataType_DOUBLE:
        return CNTK::DataType::Double;
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

// helpers copied from onnxruntime (Converter.cc). These functions will eventually
// be replaced with functionalities of onnx core.
bool CNTKIsLittleEndianOrder()
{
    int n = 1;
    return (*(char *) &n == 1);
}

#pragma warning(disable : 4244)

float UnpackFloat(const char *buf, int i)
{
    float temp = 0;
    if (CNTKIsLittleEndianOrder())
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

    if (CNTKIsLittleEndianOrder())
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

float16 UnpackFloat16(const char *buf, int i)
{
    float16 temp = 0;

    if (CNTKIsLittleEndianOrder())
    {
        memcpy((void *) &temp, (void *) buf, sizeof(char) * 2);
    }
    else
    {
        NOT_IMPLEMENTED;
    }
    return temp;
}

void RetrieveRawDataAsFloat16(const onnx::TensorProto &valueProto)
{
    if (!valueProto.int32_data().empty())
        return;
    auto raw_data = valueProto.raw_data();
    onnx::TensorProto &mutableProto = const_cast<onnx::TensorProto &>(valueProto);
    ::google::protobuf::RepeatedField<int> *p_mutable_int32_data = mutableProto.mutable_int32_data();
    if (!raw_data.empty())
    {
        auto buff = raw_data.c_str();
        for (int i = 0; i < raw_data.size(); i += 2)
        {
            auto v = UnpackFloat16(buff + i, i);
            p_mutable_int32_data->Add(*reinterpret_cast<const uint16_t *>(&v));
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
        const ConstPointerContainer<std::vector<NodeArg *>> &outputArgs = node->OutputDefs();
        ConstPointerContainer<std::vector<NodeArg *>>::ConstIterator it = outputArgs.begin();
        if (it != outputArgs.end())
        {
            const TensorShapeProto *shape = (*it)->Shape();
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

int64_t GetTensorElementTotal(const onnx::TensorProto &tensor_proto)
{
    const ::google::protobuf::RepeatedField<::google::protobuf::int64 >& dims = tensor_proto.dims();
    int64_t count = 1;
    for (int i = 0; i < dims.size(); i++)
    {
        count *= dims[i];
    }
    return count;
}

void LoadRawDataAndUnpack(onnx::TensorProto &tensor_proto, bool doUnpack)
{
    if (tensor_proto.data_location() == TensorProto_DataLocation_EXTERNAL && tensor_proto.raw_data().empty())
    {
        ::google::protobuf::RepeatedPtrField<::onnx::StringStringEntryProto> external_data = tensor_proto.external_data();
        ::google::protobuf::RepeatedPtrField<::onnx::StringStringEntryProto>::iterator it =
            std::find_if(external_data.begin(), external_data.end(), [](const ::onnx::StringStringEntryProto& StringStringEntryProto)
        {
            return StringStringEntryProto.has_key() && StringStringEntryProto.key() == "location";
        });
        if (it != external_data.end())
        {
            std::string filename = it->value();
            std::string raw_data_from_file;
#ifdef _WIN32
            Env::Default().ReadFileAsString(ToFixedWString(ONNXToCNTKHelper::model_location_ + "/" + filename).c_str(), &raw_data_from_file);
#else
            Env::Default().ReadFileAsString((ONNXToCNTKHelper::model_location_ + "/" + filename).c_str(), &raw_data_from_file);
#endif
            const void* raw_data = raw_data_from_file.data();
            size_t raw_data_len = raw_data_from_file.size();
            tensor_proto.set_raw_data(raw_data, raw_data_len);
        }
    }

    // unpack
    if (!doUnpack)
        return;

    int64_t count = GetTensorElementTotal(tensor_proto);
    auto tensorProtoDataType = tensor_proto.data_type();
    switch (tensorProtoDataType)
    {
    case TensorProto_DataType_BOOL: 
        if (tensor_proto.int32_data().empty())
        {
            std::vector<int32_t> srcData(count);
#pragma warning (disable: 4238)
            bool* p_data = (bool*)(&srcData[0]);
            Status status = onnxruntime::utils::UnpackTensor(tensor_proto, tensor_proto.raw_data().data(), tensor_proto.raw_data().size(), p_data, count * 4);
            tensor_proto.mutable_int32_data()->Resize(count, 0);
            std::copy(srcData.begin(), srcData.end(), tensor_proto.mutable_int32_data()->begin());
        } 
    break;
    case TensorProto_DataType_INT64:
        if (tensor_proto.int64_data().empty())
        {
            std::vector<int64_t> srcData(count);
            onnxruntime::utils::UnpackTensor(tensor_proto, tensor_proto.raw_data().data(), tensor_proto.raw_data().size(), &srcData[0], count);
            tensor_proto.mutable_int64_data()->Resize(count, 0);
            std::copy(srcData.begin(), srcData.end(), tensor_proto.mutable_int64_data()->begin()); 
        } 
        break;
    case TensorProto_DataType_INT32:
        if (tensor_proto.int32_data().empty())
        {
            std::vector<int32_t> srcData(count);
            onnxruntime::utils::UnpackTensor(tensor_proto, tensor_proto.raw_data().data(), tensor_proto.raw_data().size(), &srcData[0], count);
            tensor_proto.mutable_int32_data()->Resize(count, 0);
            std::copy(srcData.begin(), srcData.end(), tensor_proto.mutable_int32_data()->begin());
        }
        break;
    case TensorProto_DataType_FLOAT16:
        RetrieveRawDataAsFloat16(tensor_proto);
        break;
    case TensorProto_DataType_FLOAT:
        RetrieveRawDataAsFloat(tensor_proto);
        break;
    case TensorProto_DataType_DOUBLE:
        RetrieveRawDataAsDouble(tensor_proto);
        break;
    default:
        break;
    }
}

Constant ONNXToCNTKHelper::CreateConstant(const onnx::TensorProto &valueProto, const std::string &nodeName,
                                          const DeviceDescriptor &computeDevice)
{
    auto tensorProtoDataType = valueProto.data_type();

    NDShape shape(std::vector<size_t>(valueProto.dims().begin(), valueProto.dims().end()));

    // the following code is to revert CNTKToONNXHelper::ToTensorShape.to restore a CNTK NDArray
    NDShape reversedShape = ReverseShape(shape);

    switch (tensorProtoDataType)
    {
    case TensorProto_DataType_BOOL:
    {
        // It does not work using vector<bool> because resulted memory layout is not what we expect.
        bool *srcData = new bool[shape.TotalSize()];
        if (valueProto.int32_data_size() == shape.TotalSize())
        {
            std::copy(valueProto.int32_data().begin(), valueProto.int32_data().end(), srcData);
        }
        else
        {
            onnxruntime::utils::UnpackTensor(valueProto, valueProto.raw_data().data(), valueProto.raw_data().size(), srcData, shape.TotalSize());
        }

        // CNTK does not support bool. We need to convert to float.
        std::vector<float> srcFloatData(shape.TotalSize());
        for (int i = 0; i < shape.TotalSize(); i++)
            srcFloatData[i] = srcData[i];
        delete[] srcData;

        return CreateConstantWithTensorData<float, float>(shape, tensorProtoDataType, CNTK::DataType::Float,
                                                          &srcFloatData[0], reversedShape, computeDevice, nodeName);
    }
    break;
    case TensorProto_DataType_INT32:
    {
        std::vector<int32_t> srcData(shape.TotalSize());
        if (valueProto.int32_data_size() == shape.TotalSize())
        {
            std::copy(valueProto.int32_data().begin(), valueProto.int32_data().end(), srcData.begin());
        }
        else
        {
            onnxruntime::utils::UnpackTensor(valueProto, valueProto.raw_data().data(), valueProto.raw_data().size(), &srcData[0], shape.TotalSize());
        }

        // CNTK does not support int. We need to convert to float.
        std::vector<float> srcFloatData(shape.TotalSize());
        for (int i = 0; i < shape.TotalSize(); i++)
            srcFloatData[i] = srcData[i];

        return CreateConstantWithTensorData<float, float>(shape, tensorProtoDataType, CNTK::DataType::Float,
                                                          &srcFloatData[0], reversedShape, computeDevice, nodeName);
    }
    break;
    case TensorProto_DataType_INT64:
    {
        std::vector<int64_t> srcData(shape.TotalSize());
        if (valueProto.int64_data_size() == shape.TotalSize())
        {
            std::copy(valueProto.int64_data().begin(), valueProto.int64_data().end(), srcData.begin());
        }
        else
        {
            onnxruntime::utils::UnpackTensor(
                valueProto, valueProto.raw_data().data(), valueProto.raw_data().size(), &srcData[0], shape.TotalSize());
        }

        // CNTK does not support int64_t. We need to convert to float.
        std::vector<float> srcFloatData(shape.TotalSize());
        for (int i = 0; i < shape.TotalSize(); i++)
            srcFloatData[i] = srcData[i];

        return CreateConstantWithTensorData<float, float>(shape, tensorProtoDataType, CNTK::DataType::Float,
                                                          &srcFloatData[0], reversedShape, computeDevice, nodeName);
    }
    break;
    case TensorProto_DataType_FLOAT:
    {
        if (valueProto.float_data().empty())
        { 
            LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(valueProto), true);
        }
        return CreateConstantWithTensorData<float, float>(shape, tensorProtoDataType, CNTK::DataType::Float,
                                                          &(valueProto.float_data()[0]), reversedShape, computeDevice, nodeName);
    }
    break;
    case TensorProto_DataType_FLOAT16:
    {
        if (valueProto.int32_data().empty())
        { 
            LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(valueProto), true);
        }

        return CreateConstantWithTensorData<uint16_t, int>(shape, tensorProtoDataType, CNTK::DataType::Float16,
                                                           &(valueProto.int32_data()[0]), reversedShape, computeDevice, nodeName);
    }
    break;
    case TensorProto_DataType_DOUBLE:
    {
        // TODO: refactore commom code for float and double
        if (valueProto.double_data().empty())
        { 
            LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(valueProto), true);
        }

        return CreateConstantWithTensorData<double, double>(shape, tensorProtoDataType, CNTK::DataType::Double,
                                                            &(valueProto.double_data()[0]), reversedShape, computeDevice, nodeName);
    }
    break;
    default:
        NOT_IMPLEMENTED;
    }
}

template <typename T>
void CopyFromProto(const onnx::TensorProto &src, T &dst, vector<int> &srcIndexRange, int dstIndex)
{
    dst[dstIndex] = 0;
    auto dtype = src.data_type();
    if (dtype == onnx::TensorProto_DataType_FLOAT16)
    {
        for (int i = 0; i < srcIndexRange.size(); i++)
        {
            dst[dstIndex] += src.int32_data()[srcIndexRange[i]];
        }
    }
    else
    {
        for (int i = 0; i < srcIndexRange.size(); i++)
        {
            dst[dstIndex] += src.float_data()[srcIndexRange[i]];
        }
    }
}

template <typename T>
void CopyFromProto(const onnx::TensorProto &src, T &dst, int srcIndex, int dstIndex)
{
    auto dtype = src.data_type();
    if (dtype == onnx::TensorProto_DataType_FLOAT16)
    {
        dst[dstIndex] = src.int32_data()[srcIndex];
    }
    else
    {
        dst[dstIndex] = src.float_data()[srcIndex];
    }
}

template <typename TDst, typename TSrc>
const CNTK::Constant CNTK::ONNXToCNTKHelper::CreateConstantWithTensorData(CNTK::NDShape &shape, google::protobuf::int32 tensorProtoDataType,
                                                                          CNTK::DataType cntkDataType, const TSrc *srcData, CNTK::NDShape &reversedShape, const CNTK::DeviceDescriptor &computeDevice, const std::string &nodeName)
{
    auto totalSize = shape.TotalSize();
    TDst *data = new TDst[totalSize];

    if (shape.Rank() <= 2)
    {
        for (size_t index = 0; index < totalSize; index++)
        {
            data[index] = srcData[index];
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
                        srcData[channelIndex * channelKernelSize + pixel];
                }
            }
        }
    }

    NDArrayViewPtr dstFinal(new NDArrayView(cntkDataType, reversedShape, &data[0],
                                            totalSize * sizeof(TDst), computeDevice.CPUDevice()));

    if (computeDevice.Type() == DeviceKind::CPU)
    {
        Constant constantVariable(dstFinal, ToFixedWStringFromMultiByte(nodeName));
        return constantVariable;
    }
    else
    {
        // this is the way to load values into GPU:
        // Create a GPU NDArrayView and CopyFrom a CPU NDArrayView that holding the data.
        NDArrayViewPtr dstFinalGPU(new NDArrayView(cntkDataType, StorageFormat::Dense, reversedShape, computeDevice));
        dstFinalGPU->CopyFrom(*dstFinal);
        Constant constantVariable(dstFinalGPU, ToFixedWStringFromMultiByte(nodeName));
        return constantVariable;
    }
}

const Node *ONNXToCNTKHelper::GetChildNode(const Node *parentNode, const NodeArg *nodeArg, int &nodeArgIndex)
{
    Node::NodeConstIterator itChildNode = parentNode->InputNodesBegin();
    for (; itChildNode != parentNode->InputNodesEnd(); ++itChildNode)
    {
        const Node *childNode = &(*itChildNode);
        const ConstPointerContainer<std::vector<NodeArg *>> &childOutputDefs = childNode->OutputDefs();
        nodeArgIndex = 0;
        for (ConstPointerContainer<std::vector<NodeArg *>>::ConstIterator itChildOutput = childOutputDefs.begin(); 
            itChildOutput != childOutputDefs.end(); ++itChildOutput, nodeArgIndex++)
        {
            const NodeArg *childOutput = *itChildOutput;
            if (childOutput == nodeArg)
                return childNode;
        }
    }
    return nullptr;
}

bool ONNXToCNTKHelper::IsSecondInputOfElementWiseOpsWithBroadcast(const Node *parentNode, const NodeArg *nodeArg)
{
    if (Operators::SupportBroadcastONNXOp(parentNode->OpType()))
    {
        if (HasNamedAttribute(parentNode, "broadcast") && 1 == static_cast<int>(GetNamedAttributeAsInt64(parentNode, "broadcast")))
        {
            const ConstPointerContainer<std::vector<NodeArg *>> &inputNodeArgs = parentNode->InputDefs();
            for (int index = 0; index < inputNodeArgs.size(); index++)
            {
                const NodeArg *childNodeArg = inputNodeArgs[index];
                if (childNodeArg->Name() == nodeArg->Name())
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
    const ConstPointerContainer<std::vector<NodeArg *>> &inputDefs = parentNode->InputDefs();
    for (int i = 0; i < inputDefs.size(); i++)
    {
        if (inputDefs[i]->Name() == nodeArg->Name())
            return i;
    }

    return -1;
}

template <typename DType>
Constant CreateConstantWithRawData(DType *data, const NDShape &shape, const std::string &name,
                                   const DeviceDescriptor &computeDevice)
{
    CNTK::DataType dataType = AsDataType<DType>();

    int totalSize = shape.TotalSize();
    NDArrayViewPtr dstFinal(new NDArrayView(dataType, shape, data,
                                            totalSize * sizeof(DType), computeDevice.CPUDevice()));

    if (computeDevice.Type() == DeviceKind::CPU)
    {
        Constant constantVariable(dstFinal, ToFixedWStringFromMultiByte(name));
        return constantVariable;
    }
    else
    {
        // this is the way to load values into GPU:
        // Create a GPU NDArrayView and CopyFrom a CPU NDArrayView that holding the data.
        NDArrayViewPtr dstFinalGPU(new NDArrayView(dataType, StorageFormat::Dense, shape, computeDevice));
        dstFinalGPU->CopyFrom(*dstFinal);
        Constant constantVariable(dstFinalGPU, ToFixedWStringFromMultiByte(name));
        return constantVariable;
    }
}

template <typename DType>
std::vector<Variable> CreateRNNConstantHelper(
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
                    DType *data = new DType[totalSizePerDirection];
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
                        CopyFromProto(valueProto, data, sourceIndex, targetIndex);
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
                    DType *data = new DType[totalSizePerDirection];
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
                        vector<int> srcIndexRange = {
                            dir * 2 * totalSizePerDirection + src_index,
                            dir * 2 * totalSizePerDirection + totalSizePerDirection + src_index};

                        CopyFromProto(valueProto, data, srcIndexRange, targetIndex);
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

                DType *data = new DType[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    CopyFromProto(valueProto, data, dir * cell_size + targetIndex, targetIndex);
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

                        DType *data = new DType[cell_size];
                        NDShape weightShape({(size_t)(cell_size)});
                        for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                        {
                            CopyFromProto(valueProto, data, (dir * 3 + i) * cell_size + targetIndex, targetIndex);
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
                DType *data = new DType[totalSizePerDirection];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    int targetIndex = col * cell_size * GRUWeightDimensionHiddenMultiplier + row;

                    CopyFromProto(valueProto, data, sourceIndex, targetIndex);
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

                DType *hData = new DType[hShape.TotalSize()];
                DType *h1Data = new DType[h1Shape.TotalSize()];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int block = row / cell_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    if (block < CNTKGRUZRWeightMultiplier)
                    {
                        int targetIndex = col * cell_size * CNTKGRUZRWeightMultiplier + row;

                        CopyFromProto(valueProto, hData, sourceIndex, targetIndex);
                    }
                    else
                    {
                        int targetIndex = col * cell_size + row - cell_size * CNTKGRUZRWeightMultiplier;

                        CopyFromProto(valueProto, h1Data, sourceIndex, targetIndex);
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
                    DType *data = new DType[totalSizePerDirection];
                    for (size_t targetIndex = 0; targetIndex < totalSizePerDirection; targetIndex++)
                    {
                        int row = targetIndex;
                        // source is column major
                        int src_index = row;
                        // "fuse"
                        vector<int> sourceIndexRange = {
                            dir * 2 * totalSizePerDirection + src_index,
                            dir * 2 * totalSizePerDirection + totalSizePerDirection + src_index};

                        CopyFromProto(valueProto, data, sourceIndexRange, targetIndex);
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

                DType *data = new DType[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    CopyFromProto(valueProto, data, dir * cell_size + targetIndex, targetIndex);
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
                DType *data = new DType[totalSizePerDirection];
                for (size_t count = 0; count < totalSizePerDirection; count++)
                {
                    int row = count / input_size;
                    int col = count % input_size;
                    int sourceIndex = dir * totalSizePerDirection + count;
                    int targetIndex = col * cell_size + row;

                    CopyFromProto(valueProto, data, sourceIndex, targetIndex);
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
                    DType *data = new DType[totalSizePerDirection];
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

                        vector<int> srcIndexRange = {
                            dir * RNNBiasMultiplier * totalSizePerDirection + src_index,
                            dir * RNNBiasMultiplier * totalSizePerDirection + totalSizePerDirection + src_index};

                        CopyFromProto(valueProto, data, srcIndexRange, targetIndex);
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

                DType *data = new DType[cell_size];
                for (size_t targetIndex = 0; targetIndex < cell_size; targetIndex++)
                {
                    CopyFromProto(valueProto, data, dir * cell_size + targetIndex, targetIndex);
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

std::vector<Variable> CreateRNNConstant(
    const Node *parentNode, int index, const std::string &name, const onnx::TensorProto &valueProto, const DeviceDescriptor &computeDevice)
{
    if (valueProto.data_type() == TensorProto_DataType_FLOAT16)
    {
        return CreateRNNConstantHelper<uint16_t>(parentNode, index, name, valueProto, computeDevice);
    }
    else
    {
        return CreateRNNConstantHelper<float>(parentNode, index, name, valueProto, computeDevice);
    }
}

std::vector<FunctionPtr> CreateRNNConstantOp(const Graph *graph, const Node *node, const Node *parentNode, int index,
                                             const DeviceDescriptor &computeDevice)
{
    const onnx::TensorProto *valueProto;
    if (!graph->GetInitializedTensor(node->Name(), valueProto))
    {
        LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(*valueProto), true);
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
    const Node *parentNode, const Graph *graph, 
    ONNXToCNTKVariableMap &constructedNodeArgVariableMap, 
    const DeviceDescriptor &computeDevice)
{
    string parentONNXOpName = parentNode->OpType();
    std::string nodeName = nodeArg->Name();
    const onnx::TensorProto *valueProto;
    if (graph->GetInitializedTensor(nodeName, valueProto))
    {
        LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(*valueProto), true);
        int index = CalculateNodeArgInputIndex(nodeArg, parentNode);
        return CreateRNNConstant(parentNode, index, nodeName, *valueProto, computeDevice);
    }

    const TensorShapeProto *shapeProto = nodeArg->Shape();
    if (shapeProto == nullptr)
    {
        // dummy input,
        return std::vector<Variable>();
    }

    std::vector<Axis> dynamicAxes({});

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
                    NDShape shape = FromTensorShapeProto(*shapeProto);
                    inputVariable = InputVariable(shape, dataType, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
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
                    NDShape shape = FromTensorShapeProto(*shapeProto);
                    inputVariable = InputVariable(shape, dataType, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
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
                    NDShape shape = FromTensorShapeProto(*shapeProto);
                    inputVariable = InputVariable(shape, dataType, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
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
    if (graph->GetInitializedTensor(nodeName, valueProto))
    {
        LoadRawDataAndUnpack(const_cast<onnx::TensorProto &>(*valueProto), true);
        return CreateConstant(*valueProto, nodeName, computeDevice); // There is no batch axis added on here.
    }

    auto shapeProto = nodeArg->Shape();

    // in CNTK constants are created as Node (not a leaf) with values.
    // in ONNX constants may also be a leaf with values saved in initializer
    // here we know it is not an ONNX constant so reshape the variable to trim off last dim;
    NDShape shape = FromTensorShapeProto(*shapeProto);
    std::vector<Axis> dynamicAxes({});

    // TODO: Do we need to take care of the sequence axis here (like before)?
    // Should it be be taken care of in RNN leaf node creation (different function)?

    auto dataType = FromONNXType(nodeArg->ToProto().type());
    switch (dataType)
    {
    case CNTK::DataType::Float:
    {
        return InputVariable(shape, CNTK::DataType::Float, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
    }
    case CNTK::DataType::Float16:
    {
        return InputVariable(shape, CNTK::DataType::Float16, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
    }
    case CNTK::DataType::Double:
    {
        return InputVariable(shape, CNTK::DataType::Double, ToFixedWStringFromMultiByte(nodeArg->Name()), dynamicAxes);
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
    else if (str == "NOTSET" || str == "notset")
        return ConvAutoPadType::NOTSET;
    else
        LogicError("Unknown value for %s attribute: %s", "auto_pad", str.c_str());
}

std::vector<int64_t> ONNXToCNTKHelper::GetShapeFromInput(const NodeArg *shapeInput, const Graph *graph)
{
    const onnx::TensorProto *valueProto;
    if (!graph->GetInitializedTensor(shapeInput->Name(), valueProto))
    {
        LogicError("Non-constant shape input for Reshape is not implemented.");
    };

    auto shapeSize = valueProto->dims(0);
    std::vector<int64_t> dimData(shapeSize);
    if (valueProto->int64_data_size() == shapeSize)
    {
        std::copy(valueProto->int64_data().begin(), valueProto->int64_data().end(), dimData.begin());
    }
    else
    {
        onnxruntime::utils::UnpackTensor(*valueProto, valueProto->raw_data().data(), valueProto->raw_data().size(), &dimData[0], shapeSize);
    }

    return dimData;
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

std::vector<size_t> ONNXToCNTKHelper::VecInt64ToVecSize_t(const std::vector<int64_t> &vecFloat)
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
        cout << string(spaces, '.') + "(" + ToLegacyString(ToUTF8(useName ? function->Name() : function->Uid())) + ")" + ToLegacyString(ToUTF8(function->AsString())) << std::endl;
        return;
    }

    for (auto input : function->Inputs())
    {
        cout << string(spaces, '.') + "(" + ToLegacyString(ToUTF8(useName ? function->Name() : function->Uid())) + ")" + "->" +
                    "(" + ToLegacyString(ToUTF8(useName ? input.Name() : input.Uid())) + ")" + ToLegacyString(ToUTF8(input.AsString()))
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
    if (input0.DynamicAxes().empty() && input1.DynamicAxes().empty())
        return std::make_pair(input0, input1);
    else
    {
        // when there is any dynamic axis, broadcast in CNTK and ONNX are not the same.
        // see GetInputAdjustmentForBroadcast for cases where extra handling is needed.
        // for example:
        // [#, 1, d] + [b, 1, d] would produce [b, 1, d] in ONNX and CNTK
        // However if input #1 in CNTK is in default variable form with dynamic axis:
        // [#] [1, d] + [b, 1, d] will produce [#] [b, d, d]
        NOT_IMPLEMENTED;
    }
}

std::pair<std::vector<Axis>, bool> ONNXToCNTKHelper::GetReduceElementsAttributes(const Node *node, const Variable &input)
{
    bool keepdims = GetNamedAttributeAsInt64(node, "keepdims", 1) == 1;
    std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes", vector<int64_t>({})), input);

    // use default of all axes according to ONNX
    if (axes.empty())
    {
        if (keepdims)
            axes = vector<Axis>({ Axis::AllAxes() });
        else
        {
            // In the case of keepdims being false, CNTK does not allow reduce on Axis::AllAxes(). 
            // We have to list out all axes instead. 
            if (input.DynamicAxes().size() != 0)
                LogicError("ReduceElements with default on all axes is not supported with input of dynamic axis.");
            axes.resize(input.Shape().Rank());
            std::generate(axes.begin(), axes.end(), [static_axis = 0]() mutable { return Axis(static_axis++); });
        }
    }
    return std::make_pair(axes, keepdims);
}

Axis ONNXToCNTKHelper::ConvertONNXAxisToCNTKCppApi(int64_t axis, const Variable &operand)
{
    if (operand.DynamicAxes().size() == 2) 
        if (axis == 0)
            return Axis::OperandSequenceAxis();
        else if (axis == 1)
            return Axis::DefaultBatchAxis();

    if (operand.DynamicAxes().size() == 1 && axis == 0) 
        return Axis::DefaultBatchAxis();

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

CNTK::DataType ConvertDataTypeTensorProtoToCNTK(TensorProto_DataType newDataType)
{
    // to TensorProto_DataType
    switch (newDataType)
    {
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
        return CNTK::DataType::Float;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
        return CNTK::DataType::Double;
    case TensorProto_DataType::TensorProto_DataType_FLOAT16:
        return CNTK::DataType::Float16;
    default:
        NOT_IMPLEMENTED;
    }
}

FunctionPtr ONNXToCNTKHelper::CreateFunction(const Node *node, const std::vector<Variable> &inputs, const Graph *graph,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr)
{
    // This method checks if the node to create is a simple vanilla batch op (such as AveragePool and MaxPool, but unlike Conv)
    // and if it is then it wraps it in pack/unpack batch ops. Otherwise it just calls CreateStandardCNTKFunction
    // to create the CNTK op(s). Some complex batch ops (e.g. convolution) are not wrapped here but the wrapping 
    // is done inside CreateStandardCNTKFunction directly.

    if (Operators::IsSimpleBatchAxisOnnxOp(node->OpType()))
    {
        // Asssumes that if CreateFunction was called with isSimpleBatchAxisOnnxOp = true
        // then the op was created with a PlaceholderVariable input.
        auto operandPlaceholder = PlaceholderVariable(inputs[0].Shape(), inputs[0].GetDataType(), L"operand", {});
        FunctionPtr operandWithBatchAxis = ToBatch(operandPlaceholder);
        auto cntkFunctionWithBatchAxis = CreateFunction(node, inputs, graph, sequenceWrapperInputToFunctionPtr, operandWithBatchAxis);
        FunctionPtr cntkFunctionWithStaticAxis = UnpackBatch(cntkFunctionWithBatchAxis, ToFixedWStringFromMultiByte(node->Name()));
        return AsBlock(std::move(cntkFunctionWithStaticAxis), { { operandPlaceholder, inputs[0] } }, 
            cntkFunctionWithBatchAxis->OpName(), ToFixedWStringFromMultiByte(node->Name()));
    }
    else
        return CreateFunction(node, inputs, graph, sequenceWrapperInputToFunctionPtr, Variable());
}

template <class T, class V>
static inline std::vector<V> CastVector(const std::vector<T>& v)
{
    std::vector<V> result;
    result.reserve(v.size());
    for (auto d : v)
        result.push_back((V)d);
    return result;
}

FunctionPtr ONNXToCNTKHelper::CreateFunction(const Node *node, const std::vector<Variable> &inputs, const Graph *graph, 
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const Variable& inputPlaceholder
    )
{
    string onnxOpName = node->OpType();
    Variable inputOperand0 = (inputPlaceholder.IsInitialized() || inputs.empty()) ? inputPlaceholder : inputs[0];

    if (onnxOpName == "LSTM")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Sigmoid", "Tanh", "Tanh"}));
        return CreateLSTM(node, inputs, direction, activations, activation_alpha, activation_beta, sequenceWrapperInputToFunctionPtr);
    }
    else if (onnxOpName == "GRU")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Sigmoid", "Tanh"}));
        return CreateGRU(node, inputs, direction, activations, activation_alpha, activation_beta, sequenceWrapperInputToFunctionPtr);
    }
    else if (onnxOpName == "RNN")
    {
        const string direction = GetNamedAttributeAsString(node, "direction");
        std::vector<float> activation_alpha = GetNamedAttributeAsFloatVec(node, "activation_alpha", std::vector<float>());
        std::vector<float> activation_beta = GetNamedAttributeAsFloatVec(node, "activation_beta", std::vector<float>());
        const std::vector<string> activations = GetNamedAttributeAsStringVec(node, "activations",
                                                                             std::vector<string>({"Tanh"}));
        return CreateRNN(node, inputs, direction, activations, activation_alpha, activation_beta, sequenceWrapperInputToFunctionPtr);
    }
    if (onnxOpName == "FC")
    {
        return CreateCNTKFCNode(ToFixedWStringFromMultiByte(node->Name()), inputs);
    }
    else if (onnxOpName == "Flatten")
    {
        int64_t axisIndex = (size_t) GetNamedAttributeAsInt64(node, "axis", 1);
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkFunction = Flatten(inputs[0], axis, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Equal")
    {
        FunctionPtr cntkFunction = Equal(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Greater")
    {
        FunctionPtr cntkFunction = Greater(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Less")
    {
        FunctionPtr cntkFunction = Less(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mean")
    {
        FunctionPtr cntkFunction = Mean(inputs, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Clip")
    {
        double minValue = GetNamedAttributeAsFloat(node, "min");
        double maxValue = GetNamedAttributeAsFloat(node, "max");
        Constant minVariable = Constant::Scalar(CNTK::DataType::Float, minValue);
        Constant maxVariable = Constant::Scalar(CNTK::DataType::Float, maxValue);
        FunctionPtr cntkFunction = Clip(inputs[0], minVariable, maxVariable, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sum")
    {
        FunctionPtr cntkFunction = Sum(inputs, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "HardSigmoid")
    {
        float alpha = GetNamedAttributeAsFloat(node, "alpha");
        float beta = GetNamedAttributeAsFloat(node, "beta");
        FunctionPtr cntkFunction = HardSigmoid(inputs[0], alpha, beta, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "LRN")
    {
        // Guard the even number size case. The size > channel case is checked at cntk side.
        size_t size = static_cast<size_t>(GetNamedAttributeAsInt64(node, "size"));
        // In ONNX the size to sum over channel axis is given by diameter, while in CNTK radius.
        // Thus we are unable to support even number diameter. 
        // Currently in Lotus we are also throwing error when diameter is even. 
        if (size % 2 != 1)
            LogicError("LRN does not support even diameter size to sum over channel axis.");
        size_t depthRadius = (size - 1)/2;
        double bias = static_cast<double>(GetNamedAttributeAsFloat(node, "bias", 1.0f));
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 1e-4f));
        double beta = static_cast<double>(GetNamedAttributeAsFloat(node, "beta", 0.75f));
        FunctionPtr cntkFunction = LocalResponseNormalization(inputOperand0, 
            depthRadius, bias, alpha, beta, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "AveragePool" || onnxOpName == "MaxPool")
    {
        NDShape poolingWindowShape = GetNamedAttributeAsShape(node, "kernel_shape", false);
        auto dim = poolingWindowShape.Rank();
        NDShape strides = GetNamedAttributeAsShape(node, "strides", false, NDShape(std::vector<size_t>(poolingWindowShape.Rank(), 1u)));
        bool includePad = GetNamedAttributeAsInt64(node, "count_include_pad", 0) != 0;
        bool hasAutoPad = HasNamedAttribute(node, "auto_pad") && GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER") != "NOTSET";
        bool hasPads = HasNamedAttribute(node, "pads");
        bool ceilOutDim = false;

        if (strides.Rank() != dim)
            LogicError("Length of attribute 'strides' should be equal to dimensionality of the kernel.");

        if (hasAutoPad && hasPads)
        {
            LogicError("Ambiguous Conv node specification. Both %s and %s attributes are specified. Only one of the two should be specified.",
                "auto_pad", "pads");
        }

        strides = strides.AppendShape({ 1 }); // Because CNTK Pooling API takes strides for channel axis also.
        std::vector<bool> cntkPoolingAutoPadding;
        std::pair<std::vector<size_t>, std::vector<size_t>> padsPair;
        FunctionPtr cntkFunction;
        if (hasAutoPad)
        {
            ConvAutoPadType auto_pad = ConvertStrToConvAutoPadType(GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER"));
            switch (auto_pad)
            {
            case ConvAutoPadType::SAME_LOWER:
            case ConvAutoPadType::SAME_UPPER:
            {
                const bool isSameUpper = auto_pad == ConvAutoPadType::SAME_UPPER;
                const NDShape& inputWithBatchAxisShape = inputs[0].Shape();
                padsPair = CalcPaddingForSameLowerOrUpperAutoPad(inputWithBatchAxisShape, poolingWindowShape, strides, /*isSameUpper=*/isSameUpper);
                cntkFunction = Pooling(inputOperand0, onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
                    poolingWindowShape, strides, padsPair.first, padsPair.second, ceilOutDim, includePad, ToFixedWStringFromMultiByte(node->Name()));
                break;
            }
            case ConvAutoPadType::VALID:
            {
                cntkPoolingAutoPadding.insert(cntkPoolingAutoPadding.begin(), dim + 1, false);
                cntkFunction = Pooling(inputOperand0, onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
                    poolingWindowShape, strides, cntkPoolingAutoPadding, ceilOutDim, includePad, ToFixedWStringFromMultiByte(node->Name()));
                break;
            }
            }
        }
        else // Either hasPads == true, i.e. pads was specified, or if pads is not specified then we use default pads value of 0.
        {
            // If 'pads' is specified, we pad the node and then do 'valid' convolution.
            std::vector<int64_t> pads = GetNamedAttributeAsInt64Vec(node, "pads", std::vector<int64_t>(2*dim, 0));
            auto padsPair = SplitAndReverseVec(pads);
            cntkFunction = Pooling(inputOperand0, onnxOpName == "AveragePool" ? PoolingType::Average : PoolingType::Max,
                poolingWindowShape, strides, padsPair.first, padsPair.second, ceilOutDim, includePad, ToFixedWStringFromMultiByte(node->Name()));
        }

        return cntkFunction;
    }
    else if (onnxOpName == "GlobalAveragePool" || onnxOpName == "GlobalMaxPool")
    {
        NDShape strides = {1};
        std::vector<bool> autoPadding = {false};
        bool ceilOutDim = false;
        bool includePad = false;

        FunctionPtr cntkFunction = Pooling(inputOperand0,
                                           onnxOpName == "GlobalAveragePool" ? PoolingType::Average : PoolingType::Max,
                                           NDShape::Unknown(), strides, autoPadding, ceilOutDim, includePad, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "MaxRoiPool")
    {
        // ONNX spec is list of ints - however current IR spec is AttrType::AttributeProto_AttributeType_FLOATS
        std::vector<int64_t> pooled_shape = GetNamedAttributeAsInt64Vec(node, "pooled_shape");
        std::vector<size_t> dims = VecInt64ToVecSize_t(pooled_shape);
        NDShape roiOutputShape(dims);

        float spatialScale = GetNamedAttributeAsFloat(node, "spatial_scale");
        FunctionPtr cntkFunction = ROIPooling(inputs[0], inputs[1],
                                              PoolingType::Max, roiOutputShape, spatialScale, ToFixedWStringFromMultiByte(node->Name()));
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
        auto operandPlaceholder = PlaceholderVariable(inputs[0].Shape(), L"operand", {});
        const Variable &operand = ToBatch(operandPlaceholder);
        const Variable &scale = PlaceholderVariable(inputs[1].Shape(), inputs[1].Name(), {});
        const Variable &bias = PlaceholderVariable(inputs[2].Shape(), inputs[2].Name(), {});
        const Variable &runningMean = PlaceholderVariable(inputs[3].Shape(), inputs[3].Name(), {});
        const Variable &runningInvStd = PlaceholderVariable(inputs[4].Shape(), inputs[4].Name(), {});
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
        FunctionPtr cntkFunctionWithBatchAxis = BatchNormalization(operand,
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
                                                      ToFixedWStringFromMultiByte(node->Name()));

        FunctionPtr cntkFunctionWithStaticAxis = UnpackBatch(cntkFunctionWithBatchAxis, ToFixedWStringFromMultiByte(node->Name()));

        vector<Variable> operands{ operandPlaceholder, scale, bias, runningMean, runningInvStd };

        vector<pair<Variable, Variable>> argsMap{ pair<Variable, Variable>{operands[0], inputs[0]} };
        for (int i = 1; i < 5; ++i)
        {
            // TODO: this does not work if mean/var inputs are not constant/parameters. 
            argsMap.push_back(pair<Variable, Variable>{ operands[i], inputs[0].GetDataType() == DataType::Float16 ? Utils::ConvertVariableType<float16, float>(inputs[i], true) : inputs[i]});
        }

        return AsBlock(std::move(cntkFunctionWithStaticAxis), argsMap,
            cntkFunctionWithBatchAxis->OpName(), ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "Gemm")
    {
        float alpha = GetNamedAttributeAsFloat(node, "alpha", 1.0f);
        float beta = GetNamedAttributeAsFloat(node, "beta", 1.0f);
        bool transA = GetNamedAttributeAsInt64(node, "transA", 0) != 0;
        bool transB = GetNamedAttributeAsInt64(node, "transB", 0) != 0;
        // we need to swap position of inputs[0] and inputs[1], since c++ has different matrix row/col major than python. 
        FunctionPtr cntkFunction = ::CNTK::Internal::Gemm(inputs[1], inputs[0], inputs[2], alpha, beta, transB, transA);
        return cntkFunction;
    }
    else if (onnxOpName == "Dropout")
    {
        const Variable &operand = inputs[0];
        double dropoutRate = GetNamedAttributeAsFloat(node, "ratio");

        unsigned long seed = SentinelValueForAutoSelectRandomSeed;
        FunctionPtr cntkFunction = Dropout(operand, dropoutRate, seed, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniform")
    {
        const NDShape &shape = GetNamedAttributeAsShape(node, "shape", false);

        TensorProto_DataType onnxDataType = static_cast<TensorProto_DataType>(GetNamedAttributeAsInt64(
            node, "dtype", TensorProto_DataType::TensorProto_DataType_FLOAT));
        CNTK::DataType dataType = ConvertDataTypeTensorProtoToCNTK(onnxDataType);

        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = UniformRandom(shape, dataType, low, high, seed, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormal")
    {
        const NDShape &shape = GetNamedAttributeAsShape(node, "shape", false);

        TensorProto_DataType onnxDataType = static_cast<TensorProto_DataType>(GetNamedAttributeAsInt64(
            node, "dtype", TensorProto_DataType::TensorProto_DataType_FLOAT));
        CNTK::DataType dataType = ConvertDataTypeTensorProtoToCNTK(onnxDataType);

        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = NormalRandom(shape, dataType, mean, scale, seed, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomUniformLike")
    {
        const Variable &operand = inputs[0];
        double low = GetNamedAttributeAsFloat(node, "low");
        double high = GetNamedAttributeAsFloat(node, "high");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = UniformRandomLike(operand, low, high, seed, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "RandomNormalLike")
    {
        const Variable &operand = inputs[0];
        double mean = GetNamedAttributeAsFloat(node, "mean");
        double scale = GetNamedAttributeAsFloat(node, "scale");
        unsigned long seed = GetNamedAttributeAsInt64(node, "seed");
        FunctionPtr cntkFunction = NormalRandomLike(operand, mean, scale, seed, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Add")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = Plus(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sub")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = Minus(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Mul")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementTimes(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Div")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementDivide(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "And")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementAnd(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Not")
    {
        FunctionPtr cntkFunction = ElementNot(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Or")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementOr(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Xor")
    {
        Variable input0, input1;
        std::tie<Variable, Variable>(input0, input1) = BroadcastElementWiseInput(node, inputs[0], inputs[1]);
        FunctionPtr cntkFunction = ElementXor(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Neg")
    {
        FunctionPtr cntkFunction = Negate(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Abs")
    {
        FunctionPtr cntkFunction = Abs(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reciprocal")
    {
        FunctionPtr cntkFunction = Reciprocal(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Floor")
    {
        FunctionPtr cntkFunction = Floor(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Ceil")
    {
        FunctionPtr cntkFunction = Ceil(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sqrt")
    {
        FunctionPtr cntkFunction = Sqrt(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Relu")
    {
        FunctionPtr cntkFunction = ReLU(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "LeakyRelu")
    {
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 0.01F));
        FunctionPtr cntkFunction = LeakyReLU(inputs[0], alpha, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Selu")
    {
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 1.6732F));
        double gamma = static_cast<double>(GetNamedAttributeAsFloat(node, "gamma", 1.0507F));
        FunctionPtr cntkFunction = SELU(inputs[0], gamma, alpha, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Elu")
    {
        double alpha = static_cast<double>(GetNamedAttributeAsFloat(node, "alpha", 1.0f));
        FunctionPtr cntkFunction = ELU(inputs[0], alpha, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Exp")
    {
        FunctionPtr cntkFunction = Exp(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Log")
    {
        FunctionPtr cntkFunction = Log(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Tanh")
    {
        FunctionPtr cntkFunction = Tanh(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Pow")
    {
        FunctionPtr cntkFunction = Pow(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "MatMul")
    {
        // in case of input with both static batch and sequence axes, need to convert them
        // to dynamic axes for MatMul to work.
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto HasBatchAndSequenceAxes = [](Variable input) {
            return input.Shape().Rank() >= 2 &&
                input.Shape()[input.Shape().Rank() - 1] == NDShape::FreeDimension &&
                input.Shape()[input.Shape().Rank() - 2] == NDShape::FreeDimension; };

        auto HasFreeDimensionAt0Axes = [](Variable input) {
            return input.Shape().Rank() >= 1 &&
                input.Shape()[input.Shape().Rank() - 1] == NDShape::FreeDimension; };

        bool input0HasBatchAndSequenceAxes = HasBatchAndSequenceAxes(inputs[0]);
        bool input1HasBatchAndSequenceAxes = HasBatchAndSequenceAxes(inputs[1]);
        bool input0HasFreeDimensionAt0Axes = HasFreeDimensionAt0Axes(inputs[0]);
        bool input1HasFreeDimensionAt0Axes = HasFreeDimensionAt0Axes(inputs[1]);
        if (input0HasBatchAndSequenceAxes || input1HasBatchAndSequenceAxes)
        {
            if (input0HasBatchAndSequenceAxes)
                input0 = ToBatchAndSequence(inputs[0], sequenceWrapperInputToFunctionPtr);
            if (input1HasBatchAndSequenceAxes)
                input1 = ToBatchAndSequence(inputs[1], sequenceWrapperInputToFunctionPtr);
            FunctionPtr cntkFunction = ::CNTK::Internal::MatMul(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
            cntkFunction = UnpackBatchAndSequence(cntkFunction);
            return cntkFunction;
        }
        else if (input0HasFreeDimensionAt0Axes || input1HasFreeDimensionAt0Axes)
        {
            if (input0HasFreeDimensionAt0Axes)
                input0 = ToBatch(inputs[0], L"");
            if (input1HasFreeDimensionAt0Axes)
                input1 = ToBatch(inputs[1], L"");
            FunctionPtr cntkFunction = ::CNTK::Internal::MatMul(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
            cntkFunction = UnpackBatch(cntkFunction, L"");
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = ::CNTK::Internal::MatMul(input0, input1, ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "PRelu")
    {
        FunctionPtr cntkFunction = PReLU(inputs[1], inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sigmoid")
    {
        FunctionPtr cntkFunction = Sigmoid(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Max")
    {
        if (inputs.size() > 1)
        {
            FunctionPtr cntkFunction = ElementMax(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
            for (int i = 2; i < inputs.size(); i++) {
                cntkFunction = ElementMax(cntkFunction, inputs[i], ToFixedWStringFromMultiByte(node->Name() + "_" + std::to_string(i)));
            }
            return cntkFunction;
        }
        else
        {
            return ElementMax(inputs[0], inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        }
    }
    else if (onnxOpName == "Min")
    {
        if (inputs.size() > 1)
        {
            FunctionPtr cntkFunction = ElementMin(inputs[0], inputs[1], ToFixedWStringFromMultiByte(node->Name()));
            for (int i = 2; i < inputs.size(); i++) {
                cntkFunction = ElementMin(cntkFunction, inputs[i], ToFixedWStringFromMultiByte(node->Name() + "_" + std::to_string(i)));
            }
            return cntkFunction;
        }
        else
        {
            return ElementMin(inputs[0], inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        }
    }
    else if (onnxOpName == "Sum")
    {
        // not specified in Operators.cpp
        return nullptr;
    }
    else if (onnxOpName == "Softmax" || onnxOpName == "LogSoftmax" || onnxOpName == "Hardmax")
    {
        int64_t onnxAxis = GetNamedAttributeAsInt64(node, "axis", 1);
        if (onnxAxis == static_cast<int>(inputs[0].Shape().Rank() + inputs[0].DynamicAxes().size() - 1))
        {
            // in case of the last axis, ONNX and CNTK are equivalent
            if (onnxOpName == "Softmax")
            {
                return Softmax(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
            }
            else if (onnxOpName == "LogSoftmax")
            {
                return LogSoftmax(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
            }
            else if (onnxOpName == "Hardmax")
            {
                return Hardmax(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
            }
        }

        auto inputOperand0Placeholder = PlaceholderVariable(inputs[0].Shape(), inputs[0].GetDataType(), L"operand", {});

        Axis axis(ConvertONNXAxisToCNTKCppApi(GetNamedAttributeAsInt64(node, "axis", 1), inputOperand0Placeholder));
        Variable input = Flatten(inputOperand0Placeholder, axis);
        FunctionPtr cntkFunction;
        if (onnxOpName == "Softmax")
        {
            cntkFunction = Softmax(input, ToFixedWStringFromMultiByte(node->Name()));
        }
        else if (onnxOpName == "LogSoftmax")
        {
            cntkFunction = LogSoftmax(input, ToFixedWStringFromMultiByte(node->Name()));
        }
        else if (onnxOpName == "Hardmax")
        {
            cntkFunction = Hardmax(input, ToFixedWStringFromMultiByte(node->Name()));
        }
        NDShape originalShape = inputOperand0Placeholder.Shape();
        assert(originalShape.Rank() > 0);
        // If original shape has free dimension(batch axis), we'll need to have reshape node infer that for us. 
        if (originalShape[originalShape.Rank() - 1] == NDShape::FreeDimension)
            originalShape[originalShape.Rank() - 1] = NDShape::InferredDimension;
        cntkFunction = Reshape(cntkFunction, originalShape);

        auto additionalProperties = Dictionary();
        additionalProperties[L"axis"] = axis;

        return AsBlock(std::move(cntkFunction), {{inputOperand0Placeholder, inputs[0]}}, std::move(additionalProperties),
            ToFixedWStringFromMultiByte(onnxOpName) + L"_onnx", ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "Softplus")
    {
        FunctionPtr cntkFunction = Softplus(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Softsign")
    {
        FunctionPtr cntkFunction = Softsign(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMax")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceMax(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMin")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceMin(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSum")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceSum(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceMean")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceMean(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceProd")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceProd(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceLogSumExp" || onnxOpName == "ReduceLogSum")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceLogSum(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceL1")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceL1(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceL2")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceL2(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ReduceSumSquare")
    {
        bool keepdims;
        std::vector<Axis> axes;

        std::tie<std::vector<Axis>, bool>(axes, keepdims) = GetReduceElementsAttributes(node, inputs[0]);

        FunctionPtr cntkFunction = ReduceSumSquare(inputs[0], axes, keepdims, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ArgMax")
    {
        int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis");
        // -1 to compensate what ConvertAxisToCNTKCppApi assumes that axis is already decreased by 1
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkfunction = Argmax(inputs[0], axis, ToFixedWStringFromMultiByte(node->Name()));
        return cntkfunction;
    }
    else if (onnxOpName == "ArgMin")
    {
        int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis");
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        FunctionPtr cntkFunction = Argmin(inputs[0], axis, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Reshape")
    {
        if (!inputs[0].DynamicAxes().empty())
            NOT_IMPLEMENTED;

        std::vector<int64_t> newShape = GetShapeFromInput(node->InputDefs()[1], graph);
        std::vector<int64_t> inputShape = CastVector<size_t, int64_t>(inputs[0].Shape().Dimensions());
        std::reverse(inputShape.begin(), inputShape.end());

        int inferredDimIndex = -1;
        int totalInputSizeExceptFreeDim = 1, totalReshapeSizeExceptFreeAndInferredDim = 1;
        // process free and inferred dimensions. ONNX dimensions are left aligned, likely starting with sequence, batch, then static axes.
        // NDShape dimension order is reversed w.r.t. ONNX.
        for (int index = 0; index < std::max(newShape.size(), inputShape.size()); index++)
        {
            if (index < inputShape.size() && newShape[index] != ReshapeKeepInputDim)
                totalInputSizeExceptFreeDim *= inputShape[index];

            if (index < newShape.size())
            {
                if (newShape[index] == ReshapeInferredDim)
                {
                    if (inferredDimIndex == -1)
                    {
                        inferredDimIndex = index;
                    }
                    else
                        LogicError("Reshape: 'shape' contains more than one inferred dimension.");
                }
                else if (newShape[index] == ReshapeKeepInputDim)
                {
                    if (index < inputShape.size())
                        newShape[index] = inputShape[index];
                    else
                        LogicError("Reshape: 'shape' has a 'keep_dimension' without matching input dimension.");
                }
                else
                {
                    totalReshapeSizeExceptFreeAndInferredDim *= newShape[index];
                }
            }
        }

        if (inferredDimIndex != -1)
        {
            if (totalInputSizeExceptFreeDim % totalReshapeSizeExceptFreeAndInferredDim != 0)
                LogicError("Reshape: inferred dimension cannot be calculated from input and new shape size.");
            newShape[inferredDimIndex] = totalInputSizeExceptFreeDim / totalReshapeSizeExceptFreeAndInferredDim;
        }

        std::reverse(newShape.begin(), newShape.end());
        NDShape newNDShape(CastVector<int64_t, size_t>(newShape));

        FunctionPtr cntkFunction = Reshape(inputs[0], newNDShape, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Unsqueeze")
    {
        std::vector<Axis> axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        FunctionPtr cntkFunction = ::CNTK::Internal::Unsqueeze(inputs[0], axes, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Concat")
    {
        // We allow the 'axis' attribute to be optional, and not required (as
        // given in Concat's ONNX spec), to be consistent with other frameworks.
        // 'axis' can be enforced as a required attribute, if needed.
        int64_t onnxAxis = GetNamedAttributeAsInt64(node, "axis", 0);
        // c.f. ConvertAxisToOnnxBroadcastOfOp where axis is computed taking into consideration 
        // dynamic axes of all inputs and possible of broadcasting.
        Axis axis = ConvertONNXAxisToCNTKCppApi(onnxAxis, inputs[0]);
        std::vector<Variable> fixedInputs;
        if (FixConstantShapeForConstantVariableInputPair(inputs, fixedInputs))
        {
            FunctionPtr cntkFunction = Splice(fixedInputs, axis, ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = Splice(inputs, axis, ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
    }
    // { L"", "Split)
    else if (onnxOpName == "Slice")
    {
        std::vector<int64_t> starts64 = GetNamedAttributeAsInt64Vec(node, "starts");
        std::vector<int64_t> ends64 = GetNamedAttributeAsInt64Vec(node, "ends");

        if (starts64.size() != ends64.size())
        {
            LogicError("starts (of size %d) and ends (of size %d) attributes of Slice operation must be the same size.",
                       (int) starts64.size(), (int) ends64.size());
        }

        std::vector<int> starts = VecInt64ToVecInt(starts64);
        std::vector<int> ends = VecInt64ToVecInt(ends64);
        for (auto &e : ends)
        {
            // CNTK treats endIndex of 0 as to (and include) the last.
            if (e == INT_MAX)
                e = 0;
        }

        std::vector<Axis> axes;
        if (HasNamedAttribute(node, "axes"))
            axes = ConvertONNXAxesToCNTKCppApi(GetNamedAttributeAsInt64Vec(node, "axes"), inputs[0]);
        // axes is optional so provide a default
        if (axes.empty())
        {
            for (int i = starts.size() - 1; i >= 0; i--)
            {
                Axis axis(i);
                axes.push_back(axis);
            }
        }

        if (axes.size() == 1 && axes[0].IsSequenceAxis())
        {
            FunctionPtr cntkFunction = Sequence::Slice(inputs[0], starts[0], ends[0], ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = Slice(inputs[0], axes, starts, ends, ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "Transpose")
    {
        std::vector<int64_t> permutation = GetNamedAttributeAsInt64Vec(node, "perm");
        Variable input = inputs[0];
        // Transpose takes permutation with static axes only. ConvertPermutationONNXToCNTK assumes batch and sequence axes, 
        // if they exist, are not involved in transpose. e.g. permutation is always in the form of [batch_perm = 0, sequence_perm = 1, perm0, perm1, ..perm1] 
        // ConvertPermutationONNXToCNTK fails if above is not true. This is the case when uppack batch/sequence is needed
        bool needToUnpack = (permutation.size() - inputs[0].DynamicAxes().size()) < 2;
        for (int i = 0; i < inputs[0].DynamicAxes().size(); i++)
        {
            if (permutation[i] != i)
            {
                needToUnpack = true;
            }
        }

        if (needToUnpack)
        {
            if (inputs[0].DynamicAxes().size() == 2)
            {
                input = Sequence::Unpack(input, 0, L"");
                input = UnpackBatch(input, L"");
            }
            else
                input = UnpackBatch(input, L"");
        }

        std::vector<Axis> argsortedPermutation = ConvertPermutationONNXToCNTK(permutation, input.HasBatchAxis(), input.HasSequenceAxis());
        FunctionPtr cntkFunction = Transpose(input, argsortedPermutation, ToFixedWStringFromMultiByte(node->Name()));
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
                                          ToFixedWStringFromMultiByte(node->Name()));

        return cntkPadFunction;
    }
    else if (onnxOpName == "Gather")
    {
        FunctionPtr indices = [&](DataType referenceDataType, DataType indicesDataType) -> FunctionPtr {
            if (referenceDataType == indicesDataType)
                return inputs[1];
            return Cast(inputs[1], referenceDataType, inputs[1].Name() + L"_cast");
        }(inputs[0].GetDataType(), inputs[1].GetDataType());

        if (HasNamedAttribute(node, "axis"))
        {
            int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis", 0);
            Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
            FunctionPtr cntkFunction = GatherOp(indices, inputs[0], axis, ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
        else
        {
            FunctionPtr cntkFunction = GatherOp(indices, inputs[0], ToFixedWStringFromMultiByte(node->Name()));
            return cntkFunction;
        }
    }
    else if (onnxOpName == "DepthToSpace")
    {
        auto blockSize = GetNamedAttributeAsInt64(node, "blocksize", 1);
        return DepthToSpace(inputOperand0, static_cast<size_t>(blockSize), ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "SpaceToDepth")
    {
        auto blockSize = GetNamedAttributeAsInt64(node, "blocksize", 1);
        return SpaceToDepth(inputOperand0, static_cast<size_t>(blockSize), ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "Squeeze")
    {
        std::vector<Axis> axes = GetNamedAttributeAsAxes(node, "axes");
        return Squeeze(inputs[0], axes, ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "ImageScaler")
    {
        float scale = GetNamedAttributeAsFloat(node, "scale", 1);
        std::vector<float> bias = GetNamedAttributeAsFloatVec(node, "bias", std::vector<float>());
        return ImageScaler(inputOperand0, scale, bias, ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "MeanVarianceNormalization")
    {
        // REVIEW: ONNX MeanVarianceNormalization spec does not have an 'epsilon' attribute.
        // But corresponding CNTK node does. We construct the CNTK node with default value of epsilon
        // when loading the ONNX MeanVarianceNormalization node in CNTK.
        std::vector<int64_t> axes = GetNamedAttributeAsInt64Vec(node, "axes");
        auto rank = inputOperand0.Shape().Rank();
        bool acrossChannels = true;
        bool supported = true;
        for (size_t i = 0; i < axes.size(); ++i)
        {
            if (i == 1 && axes[i] == 2) acrossChannels = false;
            if (static_cast<int64_t>(i) != (!acrossChannels ? axes[i] - 1 : axes[i]))
            {
                supported = false;
                break;
            }
        }
        if (!(axes.size() == rank || axes.size() == rank + 1) || !supported)
            LogicError("MeanVarianceNormalization: cntk supports only computing mean/variance over all tensor, or over channel axis. Other axes combinations are not supported");

        return MeanVarianceNormalization(inputOperand0, acrossChannels, /*normalizeVariance=*/ true, ToFixedWStringFromMultiByte(node->Name()));
    }
    else if (onnxOpName == "Identity")
    {
        FunctionPtr cntkFunction = Alias(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Sin")
    {
        FunctionPtr cntkFunction = Sin(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Asin")
    {
        FunctionPtr cntkFunction = Asin(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Cos")
    {
        FunctionPtr cntkFunction = Cos(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Acos")
    {
        FunctionPtr cntkFunction = Acos(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Cast")
    {
        TensorProto_DataType newDataType = static_cast<TensorProto_DataType>(GetNamedAttributeAsInt64(node, "to"));
        if (newDataType != TensorProto_DataType::TensorProto_DataType_FLOAT &&
            newDataType != TensorProto_DataType::TensorProto_DataType_DOUBLE &&
            newDataType != TensorProto_DataType::TensorProto_DataType_FLOAT16)
        {
            // for cast to types not supported by CNTK, we simply pass it through
            // CNTK data type is more adaptive. For example, an ONNX gather op requires
            // int64_t or int. CNTK float16, float, and double are not accepted by 
            // ONNX but it can input to an CNTK node.
            return Alias(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        }
        DataType cntkNewDataType = ConvertDataTypeTensorProtoToCNTK(newDataType);
        FunctionPtr cntkFunction = Cast(inputs[0], cntkNewDataType, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Tan")
    {
        FunctionPtr cntkFunction = Tan(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "Atan")
    {
        FunctionPtr cntkFunction = Atan(inputs[0], ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "TopK")
    {
        int64_t axisIndex = GetNamedAttributeAsInt64(node, "axis", (size_t)-1);
        Axis axis = ConvertONNXAxisToCNTKCppApi(axisIndex, inputs[0]);
        auto k = GetNamedAttributeAsInt64(node, "k", 1);
        FunctionPtr cntkFunction = TopK(inputs[0], k, axis, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "EyeLike")
    {
        // Only limited import support is provided.
        FunctionPtr cntkFunction = EyeLike(inputs[0], false, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "ConstantOfShape")
    {
        LogicError("Importing ONNX (ConstantOfShape) is not yet supported in CNTK");
        return nullptr;
    }
    else if (onnxOpName == "Crop")
    {
        // inputShape: [W, H, C] x [N]
        const NDShape& inputShape = inputOperand0.Shape();
        if (inputShape.Rank() != 3)
            RuntimeError("Crop input tensor must have shape [N,C,H,W]. ");
        std::vector<int64_t> border = GetNamedAttributeAsInt64Vec(node, "border");
        if (border.size() != 4)
            RuntimeError("Crop attribute border must be a 1-D values of (leftBorder, topBorder, rightBorder, bottomBorder).");
        const size_t leftBorder = border[0];
        const size_t topBorder = border[1];
        const size_t rightBorder = border[2];
        const size_t bottomBorder = border[3];
        NDShape targetShape = [&](){
            const size_t channelSize = inputShape[inputShape.Rank() - 1];
            if (HasNamedAttribute(node, "scale"))
            {
                // targetShape: [W, H]
                NDShape targetShape = GetNamedAttributeAsShape(node, "scale", false);
                if (targetShape.Rank() != 2)
                    RuntimeError("Crop attribute scale must be a 1-D values of (height, width).");
                // targetShape: [W, H, C]
                targetShape.AppendShape(NDShape(channelSize));
                return targetShape;
            }
            else
            {
                assert((inputShape[0] > (leftBorder + rightBorder)) && (inputShape[1] > (topBorder + bottomBorder)));
                size_t targetWidth = inputShape[0] - leftBorder - rightBorder;
                size_t targetHeight = inputShape[1] - topBorder - bottomBorder;

                return NDShape({ targetWidth, targetHeight, channelSize });
            }
        }();
        auto referent = Constant(targetShape, inputOperand0.GetDataType(), 0.0);
        FunctionPtr cntkFunction = Crop(inputOperand0, referent, leftBorder, topBorder, ToFixedWStringFromMultiByte(node->Name()));
        return cntkFunction;
    }
    else if (onnxOpName == "OneHotEncoder")
    {
        // TODO: this only works in this specific case.
        std::vector<int64_t> cats = GetNamedAttributeAsInt64Vec(node, "cats_int64s");
        int numClass = cats.size();
        Axis axis = ConvertONNXAxisToCNTKCppApi(2, inputs[0]);
        FunctionPtr cntkFunction = OneHotOp(inputs[0], numClass, false, axis);
        return cntkFunction;
    }
    else
    {
        LogicError("ONNX (%s) is not supported in CNTK", onnxOpName.c_str());
        return nullptr;
    }
}

std::pair<const Node *, int> FindParentAndChildIndex(const Node *node)
{
    Node::NodeConstIterator it = node->OutputNodesBegin();
    if (it != node->OutputNodesEnd())
    {
        const Node *parent = &(*it);
        int index = 0;
        for (auto nodeArg : parent->InputDefs())
        {
            // TODO: Check whether we should use node output arg name for the check below.
            if (nodeArg->Name() == node->Name())
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
                                                        const Graph *graph, 
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr,
    const DeviceDescriptor &computeDevice)
{
    auto nodeOpStr = node->OpType();
    ONNXToCNTKMap::iterator itONNXToCNTKMap = constructedNodeMap.find(node);
    if (itONNXToCNTKMap != constructedNodeMap.end())
    {
        return std::vector<FunctionPtr>({itONNXToCNTKMap->second});
    }

    std::vector<Variable> inputs = CreateCNTKInputs(node, constructedNodeMap, constructedNodeArgVariableMap, graph, 
        sequenceWrapperInputToFunctionPtr, computeDevice);

    // Special check if node belongs to the subgraph created by CNTK's OptimizedRNNStack export.
    std::vector<FunctionPtr> lstmCntkFunction;
    bool isOptimizedRnnStack(false);
    std::tie<bool, std::vector<FunctionPtr>>(isOptimizedRnnStack, lstmCntkFunction) =
        CheckNodeBelongsToOptimizedRnnStack(node, inputs, constructedNodeMap, constructedNodeArgVariableMap, graph, 
            sequenceWrapperInputToFunctionPtr, computeDevice);
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
        FunctionPtr cntkFunction = CreateCNTKNode(node, inputs, graph, sequenceWrapperInputToFunctionPtr, computeDevice);
        constructedNodeMap.insert(ONNXToCNTKMap::value_type(node, std::vector<FunctionPtr>({cntkFunction})));
        return std::vector<FunctionPtr>({cntkFunction});
    }
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKNode(const Node *node, const std::vector<Variable> &inputs, const Graph *graph,
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr,
                                             const DeviceDescriptor &computeDevice)
{
    string onnxOpName = node->OpType();

    if (onnxOpName == "NoOp")
    {
        // TODO: this is for sink or source - what type of variable for it?
        NDShape shape;
        Constant constantVariable(shape, 0.5F, computeDevice, ToFixedWStringFromMultiByte(node->Name()));
        return constantVariable;
    }
    else if (onnxOpName == "Constant")
    {
        Constant constant = CreateConstant(node, computeDevice);
        return constant;
    }
    else
    {
        return CreateFunction(node, inputs, graph, sequenceWrapperInputToFunctionPtr);
    }
}

Variable ONNXToCNTKHelper::GetNodeOperandWithPaddingResolved(std::vector<bool> &cntkConvAutoPadding,
                                                             NDShape &strides, const Node *node, const Variable& dataOperand, const double padValue)
{
    bool hasAutoPad = HasNamedAttribute(node, "auto_pad") && GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER") != "NOTSET";
    bool hasPads = HasNamedAttribute(node, "pads");
    Variable operand = dataOperand;
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
        case ConvAutoPadType::SAME_UPPER:
        {
            const bool isSameUpper = auto_pad == ConvAutoPadType::SAME_UPPER;
            cntkConvAutoPadding.insert(cntkConvAutoPadding.begin(), strides.Rank(), false);
            NDShape kernelShape = GetNamedAttributeAsShape(node, "kernel_shape", false);
            convOperand = (Variable) CreatePadOpForSameLowerOrUpperAutoPad(dataOperand, kernelShape, strides, padValue, /*isSameUpper=*/isSameUpper);
            break;
        }
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
                                              ToFixedWStringFromMultiByte(nodeName));
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

    return convOperand;
}

std::pair<std::vector<size_t>, std::vector<size_t>> ONNXToCNTKHelper::CalcPaddingForSameLowerOrUpperAutoPad(
    const NDShape &inputWithBatchAxisShape, const NDShape& kernelShape, const NDShape& strides, bool isSameUpper)
{
    // Quote here the ONNX definition for SAME_LOWER and SAME_UPPER, and the spec for outputShape and pads.
    //      SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
    //      pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
    if (inputWithBatchAxisShape.Rank() <= 2)
        LogicError("Convolution/Pooling, input must have shape [N x C x D1 x D2 x ... x Dn], instead of %ls", inputWithBatchAxisShape.AsString().c_str());
    std::vector<size_t> pads;
    for (int dim = 0; dim < inputWithBatchAxisShape.Rank() - 2; dim++)
    {
        // Padding could be calcualted as: pad = (outDim - 1) * stride + kernel - inDim.
        // outDim = ceil(inDim / stride).
        size_t stride = strides[dim];
        size_t kernel = kernelShape[dim];
        size_t inDim = inputWithBatchAxisShape[dim];
        size_t outDim = static_cast<size_t>(ceil(float(inDim) / stride));
        int pad = (outDim - 1) * stride + kernel - inDim;
        // Negative pad means input alone is enough and no extra pads are needed. 
        if (pad < 0)
            pad = 0;
        pads.push_back((size_t)pad);
    }

    return SplitONNXPads(pads, isSameUpper);
}

std::pair<std::vector<size_t>, std::vector<size_t>> ONNXToCNTKHelper::CalcPaddingFromOutputShape(
    const NDShape& inputShape, const NDShape& kernelShape, const NDShape& strides, const NDShape& outputShape, const std::vector<int64_t>& outputPadding, 
    bool isSameUpper)
{
    // Quote here the ONNX definition for SAME_LOWER and SAME_UPPER, and the spec for outputShape and pads.
    //      total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
    //      If(auto_pads != SAME_UPPER) : pads[start_i] = total_padding[i] / 2; pads[end_i] = total_padding[i] - (total_padding[i] / 2)
    //      Else: pads[start_i] = total_padding[i] - (total_padding[i] / 2); pads[end_i] = (total_padding[i] / 2).
    // Note the start_i and end_i is reversed here because CNTK dimensions are in reverse order than ONNX.
    std::vector<size_t> pads;
    if (inputShape.Rank() <= 2)
        LogicError("Convolution Transpose, input must have shape [N x C x D1 x D2 x ... x Dn], instead of %ls", inputShape.AsString().c_str());
    for (int dim = 0; dim < inputShape.Rank() - 2; dim++)
    {
        // Padding could be calculated as: pad = (outDim - 1) * stride + kernel - inDim.
        // outDim = ceil(inDim / stride).
        size_t stride = strides[dim];
        size_t kernel = kernelShape[dim];
        size_t inDim = inputShape[dim];
        size_t outDim = outputShape[dim];
        size_t outPad = static_cast<size_t>(outputPadding[dim]);
        int pad = stride * (inDim - 1) + outPad + kernel - outDim;
        // Negative pad means input alone is enough and no extra pads are needed. 
        if (pad < 0)
            pad = 0;
        pads.push_back((size_t)pad);
    }

    return SplitONNXPads(pads, isSameUpper);
}

std::pair<std::vector<size_t>, std::vector<size_t>> ONNXToCNTKHelper::SplitONNXPads(const std::vector<size_t>& pads, bool isSameUpper)
{
    std::vector<size_t> begins, ends;
    for (int dim = 0; dim < pads.size(); dim++)
    {
        size_t p = pads[dim];
        size_t lesserPad = p / 2;
        if (isSameUpper)
        {
            // SameUpper: the upper (end) side get one extra pad if total pads is odd.
            begins.push_back(lesserPad);
            ends.push_back(p - lesserPad);
        }
        else
        {
            // SameLower: the lower (begin) side get one extra pad if total pads is odd.
            begins.push_back(p - lesserPad);
            ends.push_back(lesserPad);
        }
    }

    return std::make_pair(begins, ends);
}

std::tuple<bool, bool, bool> ONNXToCNTKHelper::ConfigureConvTransposeNodePaddingOption(const Node *node)
{
    bool USE_OUTPUT_SHAPE = HasNamedAttribute(node, "output_shape");
    bool USE_PADS = HasNamedAttribute(node, "pads");
    bool USE_AUTO_PAD = HasNamedAttribute(node, "auto_pad") && GetNamedAttributeAsString(node, "auto_pad") != "NOTSET";
    if (USE_PADS)
    {
        auto pads = GetNamedAttributeAsInt64Vec(node, "pads");
        bool isAllZeros = std::all_of(pads.begin(), pads.end(), [](int64_t i) { return i == 0; });
        if (isAllZeros && USE_AUTO_PAD)
            USE_PADS = false;
    }
    return std::make_tuple(USE_OUTPUT_SHAPE , USE_PADS , USE_AUTO_PAD);
}

FunctionPtr ONNXToCNTKHelper::CreatePadOpForSameLowerOrUpperAutoPad(
    const Variable &input, const NDShape& kernelShape, const NDShape& strides, const double padValue, bool isSameUpper)
{
    const NDShape& inputShape = input.Shape();
    std::vector<size_t> begins;
    std::vector<size_t> ends;
    std::tie<std::vector<size_t>, std::vector<size_t>>(begins, ends) = CalcPaddingForSameLowerOrUpperAutoPad(inputShape, kernelShape, strides, isSameUpper);

    if (std::all_of(begins.begin(), begins.end(), [](size_t pad) { return (pad == 0); })
        && std::all_of(ends.begin(), ends.end(), [](size_t pad) { return (pad == 0); }))
        return input;

    while (begins.size() < inputShape.Rank())
    {
        begins.push_back(0);
    }
    while (ends.size() < inputShape.Rank())
    {
        ends.push_back(0);
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
    size_t numSpatialDim = convolutionMap.Shape().Rank() - 2; // This is conv op dimension, i.e. 2 for 2D conv, 3 for 3D conv.
    size_t groups = GetNamedAttributeAsInt64(node, "group", 1);
    if (groups > 1)
        NOT_IMPLEMENTED;

    NDShape strides = GetNamedAttributeAsShape(node, "strides", false, NDShape(std::vector<size_t>(numSpatialDim, 1u)));
    NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, NDShape(std::vector<size_t>(numSpatialDim, 1u)));

    std::vector<bool> sharing({true});
    size_t reductionRank = 1;
    size_t maxTempMemSizeInSamples = 0;
    NDShape inputShape = inputOperand.Shape();
    NDShape kernelShape = convolutionMap.Shape();
    NDShape outputShape;
    std::vector<int64_t> pads;
    std::pair<std::vector<size_t>, std::vector<size_t>> padsPair;

    bool USE_OUTPUT_SHAPE, USE_PADS, USE_AUTO_PAD;
    std::tie(USE_OUTPUT_SHAPE, USE_PADS, USE_AUTO_PAD) = ConfigureConvTransposeNodePaddingOption(node);
    pads = GetNamedAttributeAsInt64Vec(node, "pads", std::vector<int64_t>(2 * numSpatialDim, 0));
    // One of the three attributes output_shape, pads, or auto_pad should be specified.
    // If not, then we use default value (all zeros) for pads attribute below.
    if (!(USE_OUTPUT_SHAPE || USE_PADS || USE_AUTO_PAD))
    {
        fprintf(stderr, "Warning: ConvTranpose - None of the three attributes, output_shape, pads, or auto_pad are specified. Assuming the default value (all zeros) for 'pads' attribute.");
        USE_PADS = true;
        pads = std::vector<int64_t>(2 * numSpatialDim, 0);
    }

    // ONNX specified explicitly that "output_shape" has preference, quoted here.
    //      The shape of the output can be explicitly set which will cause pads values to be auto generated. 
    //      If output_shape is specified pads values are ignored. See doc for details for equations to generate pads.
    // The equations are also quoted:
    //  case 1: if output_shape is not set.
    //      output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - pads[start_i] - pads[end_i]
    //  case 2: if output_shape is set.
    //      total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + kernel_shape[i] - output_shape[i]
    //      If(auto_pads != SAME_UPPER) : pads[start_i] = total_padding[i] / 2; pads[end_i] = total_padding[i] - (total_padding[i] / 2)
    //      Else: pads[start_i] = total_padding[i] - (total_padding[i] / 2); pads[end_i] = (total_padding[i] / 2).
    if (USE_OUTPUT_SHAPE)
    {
        // Given even the same outputShape, CNTK might produce different pads values due to different equations.
        // Pads Values are explicitly generated here based on ONNX equations, so as to match up values. 
        auto outputPadding = (HasNamedAttribute(node, "output_padding")) ? GetNamedAttributeAsInt64Vec(node, "output_padding") : std::vector<int64_t>(numSpatialDim, 0);
        // ONNX spec isn't clear on the format of output_shape. But from official examples, there are at least two cases:
        //  case 1: omit batch axis and channel axis.
        //      e.g. input shape: [1, 1, 3, 3], output shape: [10, 8]
        //  case 2: full rank.
        //      e.g. input shape: [1, 1, 3, 3], output shape: [1, 2, 10, 8]
        // We only care about the spatial part of output shape, because only these axes are needed to determine pad values.
        outputShape = GetNamedAttributeAsShape(node, "output_shape", /*hasBatchAxis=*/false);
        if ((outputShape.Rank() != numSpatialDim) && (outputShape.Rank() != numSpatialDim + 2))
            LogicError("ConvTranspose node's output shape attribute is of unexpected length. It should be either equal to input shape length, or input shape length - 2");
        // NOTE: The following is for ONNX V1.3 opset8. It is subject to change in future versions.
        // For convTranspose, extra pad location is flipped compared to conv/pooling. Thus the flag 'isSameUpper' is flipped to 'notSameUpper'.
        const bool notSameUpper = ConvertStrToConvAutoPadType(GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER")) != ConvAutoPadType::SAME_UPPER;
        padsPair = CalcPaddingFromOutputShape(inputShape, kernelShape, strides, outputShape, outputPadding, notSameUpper);
    }
    else if (USE_PADS)
    {
        padsPair = SplitAndReverseVec(pads);
        auto outputPadding = (HasNamedAttribute(node, "output_padding")) ? GetNamedAttributeAsInt64Vec(node, "output_padding") : std::vector<int64_t>(numSpatialDim, 0);
        std::vector<size_t> outputShapeVect(numSpatialDim + 1, 0);
        for (int axis = 0; axis < numSpatialDim; axis++)
        {
            outputShapeVect[axis] = (inputShape[axis] - 1) * strides[axis] + kernelShape[axis] +
                static_cast<size_t>(outputPadding[axis] - padsPair.first[axis] - padsPair.second[axis]);
        }
        outputShapeVect[numSpatialDim] = kernelShape[kernelShape.Rank() - 2]; // Because kernel in C++ is in [HxWxOxI] format
        outputShape = outputShape.AppendShape(NDShape(outputShapeVect));
    }
    else if (USE_AUTO_PAD)
    {
        ConvAutoPadType auto_pad = ConvertStrToConvAutoPadType(GetNamedAttributeAsString(node, "auto_pad", "SAME_UPPER"));
        switch (auto_pad)
        {
        case ConvAutoPadType::SAME_UPPER:
        case ConvAutoPadType::SAME_LOWER:
        {
            const bool notSameUpper = auto_pad != ConvAutoPadType::SAME_UPPER;
            auto outputPadding = (HasNamedAttribute(node, "output_padding")) ? GetNamedAttributeAsInt64Vec(node, "output_padding") : std::vector<int64_t>(numSpatialDim, 0);
            // For convTranspose, extra pad location is flipped compared to conv/pooling. Thus the flag 'isSameUpper' is flipped to 'notSameUpper'.
            padsPair = CalcPaddingFromOutputShape(inputShape, kernelShape, strides, outputShape, outputPadding, notSameUpper);
            break;
        }
        case ConvAutoPadType::VALID:
            break;
        default:
            NOT_IMPLEMENTED;
        }
        outputShape = NDShape({ 0 });
    }

    // At this point length of vectors strides, dilation and padsPair must be equal to
    // number of spatial dimensions (2 for 2D conv, 3 for 3D conv). In order to match the expected input for
    // CNTK Convolution API we will append one more element in each for the "channel" axis. 
    strides = strides.AppendShape({ 1 });
    dilation = dilation.AppendShape({ 1 });

    padsPair.first.push_back(0);
    padsPair.second.push_back(0);
    if (padsPair.first.size() != padsPair.second.size())
        LogicError("ConvTranspose: producing uneven lower/upper pads rank: lower(%zu), upper(%zu). ", padsPair.first.size(), padsPair.second.size());

    // CNTK only accepts outputShape in the format of
    //  case 1: [0]. Empty shape which tells CNTK to infer output shape from other inputs. 
    //  case 2: [C x X1 x X2 ... x Xn]. Out channel axis, and all spatial axes.
    // ONNX spec isn't clear on the format of output_shape. But from official examples, there are at least two cases:
    //  case 1: omit batch axis and channel axis.
    //      e.g. input shape: [1, 1, 3, 3], output shape: [10, 8]
    //      out channel axis needs to be inserted for this case. 
    //          [10, 8] ==> [2, 10, 8]
    //  case 2: full rank.
    //      e.g. input shape: [1, 1, 3, 3], output shape: [1, 2, 10, 8]
    //      batch axis needs to be removed for this case. 
    //          [1, 2, 10, 8] ==> [2, 10, 8]
    if (outputShape.Rank() == numSpatialDim && !(outputShape.Rank() == 1 && outputShape[0] == 0))
    {
        // case 1
        outputShape = outputShape.AppendShape({kernelShape[kernelShape.Rank() - 2]}); // Because kernel in C++ is in [HxWxOxI] format
    }
    else if (outputShape.Rank() == numSpatialDim + 2)
    {
        // case 2
        outputShape = outputShape.SubShape(0, outputShape.Rank() - 1);
    }

    if (!((outputShape.Rank() == 1 && outputShape[0] == 0) || (outputShape.Rank() == numSpatialDim + 1)))
    {
        // output shape rank falls under neither CNTK case 1 nor case 2.
        LogicError("ConvTranspose: unable to produce CNTK compatible output shape from given ONNX node. ");
    }

    // cuDNN couldn't support cases of asymmetric pad values.
    // Solution: increase the outputShape with size of extra pads, and add a slice node after convTranspose to remove the padded values.
    std::vector<int> extraUpperPads(outputShape.Rank(), 0);
    if (extraUpperPads.size() > 1)
    {
        assert(padsPair.first.size() == padsPair.second.size());
        if (padsPair.first.size() != outputShape.Rank())
            LogicError("ConvTranspose: producing uneven pads rank and outputShape rank: pads(%zu), outputShape(%zu). ", padsPair.first.size(), outputShape.Rank());
        for (int idx = 0; idx < outputShape.Rank() - 1; ++idx)
        {
            if (padsPair.second[idx] != padsPair.first[idx])
            {
                extraUpperPads[idx] = padsPair.second[idx] - padsPair.first[idx];
                if (extraUpperPads[idx] > 0)
                    padsPair.second[idx] -= extraUpperPads[idx];
                else
                    padsPair.first[idx] += extraUpperPads[idx];
                outputShape[idx] += abs(extraUpperPads[idx]);
            }
        }
    }

    FunctionPtr cntkConvFunction = CreateCNTKConvTransposeNode(inputOperand, convolutionMap,
        strides, sharing, padsPair.first, padsPair.second, outputShape,
        dilation, reductionRank, maxTempMemSizeInSamples, node->Name());

    if (std::any_of(extraUpperPads.begin(), extraUpperPads.end(), [](int i) { return i != 0; }))
    {
        // Add slice node to remove output values that are considered padded.
        std::vector<Axis> axes;
        std::vector<int> beginIndices;
        std::vector<int> endIndices;
        for (int idx = 0; idx < extraUpperPads.size() - 1; ++idx)
        {
            if (extraUpperPads[idx] != 0)
            {
                int extraUpperPad = extraUpperPads[idx];
                axes.push_back(Axis(idx));
                beginIndices.push_back(extraUpperPad > 0 ? 0 : -extraUpperPad);
                endIndices.push_back(extraUpperPad > 0 ? outputShape[idx] - extraUpperPad : outputShape[idx]);
            }
        }
        cntkConvFunction = Slice(cntkConvFunction, axes, beginIndices, endIndices);
    }

    // If Bias is specified in the ONNX node.
    if (inputs.size() == 3)
    {
        NDShape shape({ 1, 1, inputs[2].Shape()[0] });
        cntkConvFunction = Plus(cntkConvFunction, Reshape(inputs[2], shape));
    }

    return cntkConvFunction;
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKConvTransposeNode(const Variable& inputOperand, const Variable& convolutionMap, const NDShape& strides,
    const std::vector<bool>& sharing, const std::vector<size_t>& lowerPad, const std::vector<size_t>&  upperPad, const NDShape& outputShape,
    const NDShape& dilation, size_t reductionRank, size_t maxTempMemSizeInSamples, const std::string& name)
{
    auto operandPlaceholder = PlaceholderVariable(inputOperand.Shape(), L"operand", {});
    auto convmapPlaceholder = PlaceholderVariable(convolutionMap.Shape(), L"convolutionMap", {});
    FunctionPtr operandWithBatchAxis = ToBatch(operandPlaceholder);
    FunctionPtr convResultWithBatchAxis = ConvolutionTranspose(
            convmapPlaceholder,
            operandWithBatchAxis,
            strides,
            sharing,
            lowerPad,
            upperPad,
            outputShape,
            dilation,
            maxTempMemSizeInSamples);
    FunctionPtr convResultWithStaticAxis = UnpackBatch(convResultWithBatchAxis, ToFixedWStringFromMultiByte(name));
    return AsBlock(std::move(convResultWithStaticAxis), { { operandPlaceholder, inputOperand },{ convmapPlaceholder, convolutionMap } },
        L"ConvolutionTranspose", ToFixedWStringFromMultiByte(name));
}

FunctionPtr ONNXToCNTKHelper::CreateCNTKConvNode(const Node *node, const std::vector<Variable> &inputs)
{
    Variable convolutionMap = inputs[1];
    size_t numSpatialDim = convolutionMap.Shape().Rank() - 2; // This is conv op dimension, i.e. 2 for 2D conv, 3 for 3D conv.

    NDShape strides = GetNamedAttributeAsShape(node, "strides", false, NDShape(std::vector<size_t>(numSpatialDim, 1u)));
    NDShape dilation = GetNamedAttributeAsShape(node, "dilations", false, NDShape(std::vector<size_t>(numSpatialDim, 1u)));
    // TODO: avoid hardcoded values
    std::vector<bool> sharing({true});
    size_t reductionRank = 1;
    size_t maxTempMemSizeInSamples = 0;
    size_t groups = GetNamedAttributeAsInt64(node, "group", 1);

    std::vector<bool> cntkConvAutoPadding;
    auto convOperand = GetNodeOperandWithPaddingResolved(/*output arg first*/ cntkConvAutoPadding, strides, node, inputs[0]);

    // At this point length of vectors strides, dilation, and cntkConvAutoPadding must be equal to
    // number of spatial dimensions (2 for 2D conv, 3 for 3D conv). 
    // In order to match the expected input for CNTK Convolution API we will append one more element
    // in each for the "channel" axis. 
    strides = strides.AppendShape({ 1 });
    dilation = dilation.AppendShape({ 1 });
    cntkConvAutoPadding.push_back(false);

    auto operandPlaceholder = PlaceholderVariable(convOperand.Shape(), L"operand", {});
    auto convmapPlaceholder = PlaceholderVariable(convolutionMap.Shape(), L"convolutionMap", {});
    FunctionPtr operandWithBatchAxis = ToBatch(operandPlaceholder);
    FunctionPtr convResultWithBatchAxis = Convolution(
        convmapPlaceholder,
        operandWithBatchAxis,
        strides,
        sharing,
        cntkConvAutoPadding,
        dilation,
        reductionRank,
        groups,
        maxTempMemSizeInSamples,
        false);
    FunctionPtr convResultWithStaticAxis = UnpackBatch(convResultWithBatchAxis, ToFixedWStringFromMultiByte(node->Name()));
    FunctionPtr cntkConvFunction = AsBlock(std::move(convResultWithStaticAxis), { { operandPlaceholder, convOperand }, { convmapPlaceholder, convolutionMap } }, L"Convolution", ToFixedWStringFromMultiByte(node->Name()));
    
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

// onnx graph library treats output NodeArgs as outputs. 
// when creating a CNTK model, we build a map from Nodes to FunctionPtrs.
// To figure out the outputs of a CNTK model, we need to filter out
// output variables of output Functions that are not in the graph outputs.
void FilterGraphOutputs(std::vector<Variable> &outputVariables)
{
    std::set<FunctionPtr> visited;
    std::vector<Variable> sinkedVariables;
    for (auto v : outputVariables)
    {
        if (v.Owner())
        {
            v.Owner()->PreorderTraverse([&visited, &sinkedVariables](const FunctionPtr& function) {
                if (visited.find(function) != visited.end())
                    return;
                visited.insert(function);
                for (auto inputVariable : function->Inputs())
                    if (std::find(sinkedVariables.begin(), sinkedVariables.end(), inputVariable) == sinkedVariables.end())
                        sinkedVariables.push_back(inputVariable);
                    
            }, false);
        }
    }

    for (std::vector<Variable>::iterator it = outputVariables.begin(); it != outputVariables.end();)
    {
        if (std::find(sinkedVariables.begin(), sinkedVariables.end(), *it) != sinkedVariables.end())
            it = outputVariables.erase(it);
        else
            ++it;
    }
}

FunctionPtr ONNXToCNTK::CreateGraph(onnxruntime::Graph *src, const DeviceDescriptor &computeDevice,
    const std::string& model_location)
{
    ONNXToCNTKHelper::model_location_ = GetRootPath(model_location);
    FunctionPtr cntkModel;

    // To use depth-first-traversal, keeps a collection of visited nodes.
    ONNXToCNTKMap constructedFunctions;
    ONNXToCNTKVariableMap constructedNodeArgVariableMap;
    VariableToFunctionPtr sequenceWrapperInputToFunctionPtr;

    const GraphNodes &nodes = src->Nodes();
    for (GraphNodes::ConstNodeIterator it = nodes.cbegin(); it != nodes.cend(); ++it)
    {
        const Node &node = *it;

        if (constructedFunctions.find(&node) == constructedFunctions.end())
        {
            std::vector<FunctionPtr> cntkNode = ONNXToCNTKHelper::FromONNXNode(&node,
                                                                               constructedFunctions, constructedNodeArgVariableMap, src, 
                sequenceWrapperInputToFunctionPtr, computeDevice);
        }
    }

    std::vector<FunctionPtr> functions;
    const std::vector<const NodeArg*>& graphOutputs = src->GetOutputs();
    // collect output Nodes based on output NodeArgs
    std::set<Node*> outputNodes;
    for (int i = 0; i < graphOutputs.size(); i++)
    {
        const NodeArg* nodeArg = graphOutputs[i];
        for (auto &node : src->Nodes())
        {
            if (std::find(outputNodes.begin(), outputNodes.end(), &node) == outputNodes.end())
            {
                for (auto nodeOutput : node.OutputDefs())
                    if (nodeOutput == nodeArg)
                    {
                        outputNodes.insert(&node);
                        break;
                    }
            }
        }
    }

    // collect output FunctionPtrs from output Nodes
    for (auto &node : outputNodes)
    {
        std::vector<FunctionPtr> &constructedFuncts = constructedFunctions[node];
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
        std::vector<Variable> outputVariables;
        for (auto f : functions)
        {
            for (auto v : f->Outputs())
            {
                outputVariables.push_back(v);
            }
        }
        if (outputVariables.size() > graphOutputs.size())
            FilterGraphOutputs(outputVariables);
        return Combine(outputVariables);
    }
}

std::vector<Variable> ONNXToCNTKHelper::CreateCNTKInputsStartingFromIndex(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                                          ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, 
    size_t startIndex, VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const DeviceDescriptor &computeDevice)
{
    std::vector<Variable> inputs;
    const ConstPointerContainer<std::vector<NodeArg *>> &inputDefs = node->InputDefs();
    for (int i = startIndex; i < inputDefs.size(); i++)
    {
        const NodeArg *nodeArg = inputDefs[i];
        // nodeArg may be one of outputDefs from another node inputNode
        // in case there are multiple outputDefs, we need to know the index of the nodeArg
        int nodeArgIndex = 0;
        const Node *inputNode = GetChildNode(node, nodeArg, nodeArgIndex);
        if (inputNode != nullptr)
        {
            ONNXToCNTKMap::iterator itNodeMap = constructedNodeMap.find(const_cast<Node *>(inputNode));
            if (itNodeMap != constructedNodeMap.end())
            {
                std::vector<FunctionPtr> inputCNTKFunctionPtrs = itNodeMap->second;
                for (auto f : inputCNTKFunctionPtrs)
                {
                    inputs.insert(inputs.end(), f->Outputs()[nodeArgIndex]);
                }
            }
            else
            {
                std::vector<FunctionPtr> inputVariables = FromONNXNode(inputNode, constructedNodeMap,
                                                                       constructedNodeArgVariableMap, graph, 
                    sequenceWrapperInputToFunctionPtr, computeDevice);
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
                if (constructedNodeArgVariableMap.find(nodeArg->Name()) == constructedNodeArgVariableMap.end())
                {
                    Variable inputVariable = CreateLeafVariableOrConstant(nodeArg, node, graph, computeDevice);
                    constructedNodeArgVariableMap.insert(ONNXToCNTKVariableMap::value_type(nodeArg->Name(), inputVariable));
                }
                inputs.push_back(constructedNodeArgVariableMap[nodeArg->Name()]);
            }
        }
    }
    return inputs;
}

std::vector<Variable> ONNXToCNTKHelper::CreateCNTKInputs(const Node *node, ONNXToCNTKMap &constructedNodeMap,
                                                         ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, 
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const DeviceDescriptor &computeDevice)
{
    return CreateCNTKInputsStartingFromIndex(node, constructedNodeMap, constructedNodeArgVariableMap, graph, 0, 
        sequenceWrapperInputToFunctionPtr, computeDevice);
}

std::pair<bool, std::vector<FunctionPtr>> ONNXToCNTKHelper::CheckNodeBelongsToOptimizedRnnStack(const Node *node, const std::vector<Variable> &inputs,
                                                                                                ONNXToCNTKMap &constructedNodeMap, ONNXToCNTKVariableMap &constructedNodeArgVariableMap, const Graph *graph, 
    VariableToFunctionPtr &sequenceWrapperInputToFunctionPtr, const DeviceDescriptor &computeDevice)
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
        Node::NodeConstIterator it = node->OutputNodesBegin();
        if (it != node->OutputNodesEnd())
        {
            firstParentNode = &(*it);
        }
        if (firstParentNode != nullptr)
        {
            it = firstParentNode->OutputNodesBegin();
            if (it != firstParentNode->OutputNodesEnd())
            {
                grandParentNode = &(*it);
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
                                                                                           constructedNodeArgVariableMap, graph, 1, 
                sequenceWrapperInputToFunctionPtr, computeDevice);
            inputsnextLSTMInputs.insert(inputsnextLSTMInputs.begin(), inputs[0]);
            FunctionPtr cntkFunction = CreateCNTKNode(grandParentNode, inputsnextLSTMInputs, graph, sequenceWrapperInputToFunctionPtr, computeDevice);
            lstmCntkFunction.push_back(cntkFunction);
            constructedNodeMap.insert(ONNXToCNTKMap::value_type(grandParentNode, lstmCntkFunction));
            isOptimizedRnnStack = true;
        }
    }
    return std::make_pair(isOptimizedRnnStack, lstmCntkFunction);
}

std::string CNTK::ONNXToCNTKHelper::model_location_;