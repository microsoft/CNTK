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
#include <numeric>
#include <iostream>
#include "RNNHelper.h"
#include "Matrix.h"

using namespace Microsoft::MSR::CNTK;
using namespace CNTK::ONNX;
using namespace CNTK;

const int FreeSequenceLen = 0;
const std::string FreeSequenceDimParam = "None";
const size_t numBiasInOnnxLstm = 2; // bias for W, and bias for R (also called H in CNTK).
// TODO: support cases where batch size is not 1.
const int FreeBatchSize = 1;

onnx::TypeProto TensorShapeProtoToTypeProto(const onnx::TensorShapeProto* inputShape)
{
    onnx::TypeProto newShape;
    int inputRank = inputShape->dim_size();
    for (int index = 0; index < inputRank; index++)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

    return newShape;
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

        static ONNXIR::Node *AddReshapeNode(const ONNXIR::NodeArg &nodeArg, const std::vector<int> &newShape, const std::string &outArgName, ONNXIR::Graph* graph);
        static ONNXIR::Node *AddMatMulNode(const ONNXIR::NodeArg &nodeArg1, const ONNXIR::NodeArg &nodeArg2, ONNXIR::Graph* graph);
        static ONNXIR::Node *AddArgMaxNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph, int axis);
        static ONNXIR::Node *AddCastNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph, const std::string &toType);

        //
        //  Insert a reshape node in front of a given node and its output node arg  
        //
        static ONNXIR::Node *InsertReshapeNodeToCNTKFunction(const FunctionPtr &src, ONNXIR::Node* node, const std::vector<int> &shape, ONNXIR::Graph* graph);

        //
        //  methods to create a RNN/LSTM/GRU node.
        //
        static ONNXIR::Node* CreateLSTMNode(const FunctionPtr& src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);
        static ONNXIR::Node *CreateGRUNode(const FunctionPtr &src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);
        static ONNXIR::Node *CreateRNNNode(const FunctionPtr &src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);

        static void PrepareRNNInput(const Variable &X, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareLSTMInitialStateNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &initialVariables, int batchSize, int cellSize, 
            const std::string &uid, std::vector<ONNXIR::NodeArg> &nodeInputs);

        static void PrepareRNNWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Ws, std::vector<ONNXIR::NodeArg> &nodeInputs,
            std::function<void(const std::vector<NDArrayViewPtr> &srcTensors,
                onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)> weightConverter);
        static void PrepareGRUZRHWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Rs, const std::vector<Variable> &Rh1s, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareGRUBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs);

        static void PrepareRNNBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs);

        static void PrepareLSTMWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Ws, double *stabilizerConstants, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareLSTMBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::vector<Variable> &Ws, std::vector<ONNXIR::NodeArg> &nodeInputs);
        static void PrepareLSTMPeepholeNode(ONNXIR::Graph* graph,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes, const std::vector<Variable> &Ps,
            const std::vector<double> &stabilizerDcCoefs, const std::vector<double> &stabilizerCCoefs,
            std::vector<ONNXIR::NodeArg> &nodeInputs);
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

        static void CopyTensorsWithMultipliers(const std::vector<NDArrayViewPtr> &srcTensors, const std::vector<double> &multipliers,
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);


        static void CopyRNNBiasTensors(const std::vector<NDArrayViewPtr> &srcTensors, 
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

        static void CopyGRUWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

        static void CopyGRUStateWeightTensors(
            const std::vector<NDArrayViewPtr> &srcZRTensors, const std::vector<NDArrayViewPtr> &srcHTensors,
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

        static void CopyRNNWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

        static void FillTensorWithScalar(const std::vector<NDArrayViewPtr>& src, onnx::TensorProto& dst, const std::vector<int> dstShape);

        //
        // Create an ONNX weight tensor for LSTM op. It handles memory mapping from CNTK to ONNX.  
        //
        static void CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(const std::vector<NDArrayViewPtr> &src, double *stabilizerConstants,
            onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

        static void CopyShapeTypeProtoToTensorProto(const onnx::TypeProto &inputArgType, onnx::TensorProto& dst);

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
        static onnx::TypeProto ToTypeProto(const NDShape& shape, int dynamicAxisCount);
        static onnx::TypeProto ToTypeProto(const NDShape& shape, bool hasBatchAxis = false, bool hasSequenceAxis = false, bool doReverseShape = true);
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
        static std::vector<int64_t> ConvertPermutationCNTKToONNX(const std::vector<Axis> &axes, bool hasBatchAxis);

        //
        // Convert data types from CNTK to ONNX.
        //
        static void UpdateONNXType(DataType dataType, onnx::TypeProto& type);

        //
        // Map CNTK OP names to ONNX OP Names.
        //
        static std::string ToOPName(const FunctionPtr& src);

        static bool OpInputsHasBatchAxis(const FunctionPtr& src);

        //
        // Which input to ignore during converting a CNTK block to a primitive OP in ONNX.
        //
        static bool FilterInput(const FunctionPtr& src, const CNTK::Variable& input, size_t inputIndex);

        //
        // Converts axis (in CNTK C++ API sense) to index in ONNX sense
        //
        static int64_t ConvertAxisToOnnx(const Axis &axis, const Variable &operand);

        //
        // Converts axes (in CNTK C++ API sense) to index in ONNX sense
        //
        static std::vector<int64_t> ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand);

        //
        // Given input tersors of a CNTK elementwise operation, figure out
        // input shapes for ONNX operation.
        // It also returns whether broadcast is required and the axis for broadcast.
        // Due to the fact that ONNX only allows braodcast of right-hand-side,
        // inputs may need to be swapped. In this case the last bool is true.
        static std::tuple<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool> AdjustForBroadcastShape(
            const Variable &input1, const Variable &input2);

        static std::tuple<std::vector<int>, bool, int, bool > CalculateBroadcastAxis(
            const std::vector<int> &dims1, const std::vector<int> &dims2);

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
        static std::pair<std::vector<int>, std::vector<int> > GetONNXPadsAttributeFromCNTKNode(
            const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, bool ceilOutDim);

        //
        // Adds attributes 'auto_pad' or 'pads' to saved node (typically convolution or pooling).
        //
        static void PutAutopadOrPadAttrInNode(ONNXIR::Node* node, const std::vector<bool>& autoPadding,
            const NDShape& kernelShape, bool ceilOutDim = false);

        //
        // Takes CNTK's OptimizedRNNStack node and converts it into a series of RNN/LSTM/GRU nodes
        // on the ONNX side.
        //
        static ONNXIR::Node* CreateONNXNodesForOptimizedRNNStack(const FunctionPtr &src,
            ONNXIR::Graph* graph,
            std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
            std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const std::unordered_map<Variable, Variable>& compositeOutputsMap);

        //
        // Takes the OptimizedRNNStack's input combined weight matrix, and splits it into individual
        // weight and bias matrices for each recurrent op layer.
        //
        static std::tuple<std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr> >
            SplitOptimzedRnnWtoIndivMats(Matrix<float>& WbigIn, size_t numLayers, size_t inputSize, size_t hiddenSize,
                bool bidirectional, wstring recurrentOp = L"lstm");

        //
        // Extracts RNN weight matrices from OptimizedRNNStack's input combined weight matrix.
        //
        static Matrix<float> GetWeightMatFromOrnnBigW(Matrix<float>& Wbig, size_t offset,
            size_t layerInputSize, size_t layerOutputSize, size_t numGates, wstring recurrentOp = L"lstm");

        //
        // Extracts RNN bias matrices from OptimizedRNNStack's input combined weight matrix.
        //
        static Matrix<float> GetBiasMatFromOrnnBigW(Matrix<float>& Wbig, size_t offset,
            size_t hiddenSize, size_t numGates, wstring recurrentOp = L"lstm");

        //
        // Takes the OptimizedRNNStack's individual weight matrix and changes the format from 
        // i,f,c,o (OptimizedRNNStack) to i,o,f,c (ONNX).
        //
        static void InplaceAdjustGateOrder(Matrix<float>& Wbig, size_t hiddenSize);

        //
        // Takes a vector of Matrix<ElemType> which are weights for each layer and each direction
        // and converts them to a vector of NDArrays, one for each layer, in ONNX LSTM format.
        //
        static std::vector<NDArrayViewPtr> ToRnnWeightPerLayerOnnxFormat(std::vector<Matrix<float> >& W, size_t numLayers,
            size_t numDirections, size_t numGates, size_t hiddenSize, size_t inputSize, bool updateInputSizeWithEachLayer);

        //
        // Takes a vector of Matrix<ElemType> which are biases for each layer and each direction
        // and converts them to a vector of NDArrays, one for each layer, in ONNX LSTM format.
        //
        static std::vector<NDArrayViewPtr> ToRnnBiasPerLayerOnnxFormat(std::vector<Matrix<float> >& W, size_t numLayers,
            size_t numDirections, size_t hiddenSize, size_t numGates);

        //
        // Create a ONNX node for input weight for a recurrence node.
        //
        static void CreateRecurrentWeightONNXNodes(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
            const Variable& Wcombined, std::vector<ONNXIR::NodeArg>& inputs, NDArrayViewPtr W, string WArgName = "");

        //
        // Method to insert reshape and transpose nodes to the output of the ONNX LSTM output
        // so that it can be fed in as input to the next ONNX LSTM node.
        //
        static ONNXIR::NodeArg LSTMOutputShapeAdapter(ONNXIR::NodeArg& inputArg, onnx::TypeProto& inputArgType, ONNXIR::Graph* graph,
            size_t numDirections, size_t hiddenSize, CNTK::DataType outputType, string adapterBasename = "");

        // A helper function, to reverse any iterable container and return a copy
        // of the reversed container.
        //
        template<typename ItrType>
        static ItrType reverse(ItrType v)
        {
            std::reverse(std::begin(v), std::end(v));
            return v;
        }

        template<class T, class V>
        static inline std::vector<V> Cast(const std::vector<T>& v)
        {
            std::vector<V> result;
            result.reserve(v.size());
            for (auto d : v)
                result.push_back((V)d);
            return result;
        }
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

void AddDataElementArrayViewToTensorProto(const NDArrayViewPtr src, int srcIndex, onnx::TensorProto& dst)
{
    DataType dataType = src->GetDataType();
    switch (dataType)
    {
    case DataType::Float:
    {
        auto data = src->DataBuffer<float>();
        *(dst.mutable_float_data()->Add()) = data[srcIndex];
    }
    break;
    case DataType::Double:
    {
        auto data = src->DataBuffer<double>();
        *(dst.mutable_double_data()->Add()) = data[srcIndex];
    }
    break;
    default:
        NOT_IMPLEMENTED;
    }
}

// LSTM gate bias order difference between CNTK (icfo) and ONNX (iofc) is 
// handled while building ONNX LSTM bias tensor.  
template<typename DType>
void AppendCNTKBiasWeightToONNXTensor(DType *data, const NDShape &shape, onnx::TensorProto& dst)
{
    auto totalSize = shape.TotalSize();
    int cell_size = shape[0] / LSTMWeightDimensionHiddenMultiplier;
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
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

        // source is collmn major
        int src_index = row;
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = (float)data[src_index];
        else if (typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = (double)data[src_index];
        else
            NOT_IMPLEMENTED;
    }

    // ONNX requires bias being 8 * cell_size with separated Wb and Rb for each gate.
    // CNTK only have bias applied to input side. put zeros for hidden side. 
    // It is numerically equivalent.
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
    {
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = 0;
        else if (typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = 0;
        else
            NOT_IMPLEMENTED;
    }
}

// CNTK data is column major. Gate weight order is icfo. 
// ONNX is row major. Gate weight order is iofc. This method does the data layout conversion.  
template<typename DType>
void AppendCNTKWeightToONNXTensor(DType *data, const NDShape &shape, onnx::TensorProto& dst, double stabilizer)
{
    if (shape.Rank() == 1)
    {
        AppendCNTKBiasWeightToONNXTensor(data, shape, dst);
        return;
    }

    auto totalSize = shape.TotalSize();
    for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
    {
        int cell_size = shape[0] / LSTMWeightDimensionHiddenMultiplier;
        int input_size = shape[1];

        bool rowMajor = true;
        int row, col;
        if (rowMajor)
        {
            // row major layout
            row = targetIndex / input_size;
            col = targetIndex % input_size;
        }
        else
        {
            row = targetIndex % (cell_size * LSTMWeightDimensionHiddenMultiplier);
            col = targetIndex / (cell_size * LSTMWeightDimensionHiddenMultiplier);
        }

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
        int src_index = LSTMWeightDimensionHiddenMultiplier * cell_size * col + row;
        if (typeid(DType) == typeid(float))
            *(dst.mutable_float_data()->Add()) = (float)(data[src_index] * stabilizer);
        else if(typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = (double)(data[src_index] * stabilizer);
        else 
            NOT_IMPLEMENTED;
    }
}

void SetTensorType(onnx::TensorProto& dst, DataType dataType)
{
    switch (dataType)
    {
    case DataType::Float:
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        break;
    case DataType::Double:
        dst.set_data_type(onnx::TensorProto_DataType_DOUBLE);
        break;
    default:
        NOT_IMPLEMENTED;
    }
}

void CNTKToONNXHelper::CopyShapeTypeProtoToTensorProto(const onnx::TypeProto &inputArgType, onnx::TensorProto& dst)
{
    std::vector<int64_t> dimensions = CNTKToONNXHelper::ToINTS(inputArgType);
    for (auto dim : dimensions)
        *(dst.mutable_dims()->Add()) = dim;
}

void CNTKToONNXHelper::CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(const std::vector<NDArrayViewPtr> &src, double *stabilizerConstants,
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    auto dataType = src[0]->GetDataType();
    SetTensorType(dst, dataType);

    for (int i = 0; i < src.size(); i++)
    {
        auto srcTemp = src[i]->DeepClone();
        auto srcShape = srcTemp->Shape();

        // This is our own copy so move it to the CPU.
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        double stabilizer = stabilizerConstants != nullptr ? stabilizerConstants[i] : 1;

        switch (dataType)
        {
        case DataType::Float:
        {
            auto data = srcTemp->DataBuffer<float>();
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        case DataType::Double:
        {
            auto data = srcTemp->DataBuffer<double>();
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyTensorsWithMultipliers(const std::vector<NDArrayViewPtr> &srcTensors, 
    const std::vector<double> &multipliers, 
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    // TODO: verify that srcTensors has consistant shapes
    if (multipliers.size() != srcTensors.size())
        LogicError("To apply multiplier when copying tensors, number of multipliers must be the same as number of tensors.");

    for (int viewIndex = 0; viewIndex < srcTensors.size(); viewIndex++)
    {
        auto view = srcTensors[viewIndex];
        double multiplier = multipliers[viewIndex];
        auto dataType = view->GetDataType();
        SetTensorType(dst, dataType);

        auto srcTemp = view->DeepClone();
        auto srcShape = srcTemp->Shape();
        auto totalSize = srcShape.TotalSize();
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
        switch (dataType)
        {
        case DataType::Float:
        {
            auto data = srcTemp->DataBuffer<float>();
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_float_data()->Add()) = (float)(data[index] * multiplier);

            break;
        }
        case DataType::Double:
        {
            auto data = srcTemp->DataBuffer<double>();
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_double_data()->Add()) = data[index] * multiplier;

            break;
        }
        default:
            NOT_IMPLEMENTED;
        }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyRNNBiasTensors(const std::vector<NDArrayViewPtr> &srcTensors,
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    if (srcTensors.empty())
        return;

    DataType dataType = srcTensors[0]->GetDataType();
    SetTensorType(dst, dataType);

    for (int i = 0; i < srcTensors.size(); i++)
    {
        auto srcTemp = srcTensors[i]->DeepClone();
        auto srcShape = srcTemp->Shape();

        // This is our own copy so move it to the CPU.
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        auto totalSize = srcShape.TotalSize();
        for (size_t index = 0; index < totalSize; index++)
        {
            AddDataElementArrayViewToTensorProto(srcTemp, index, dst);
        }

        // fill zeros for Rb[zrh] because CNTK GRU does not support Rb.
        for (size_t index = 0; index < totalSize; index++)
            switch (dataType)
            {
            case DataType::Float:
            {
                *(dst.mutable_float_data()->Add()) = 0;
            }
            break;
            case DataType::Double:
            {
                *(dst.mutable_double_data()->Add()) = 0;
            }
            break;
            }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyGRUWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    if (srcTensors.empty())
        return;

    DataType dataType = srcTensors[0]->GetDataType();
    SetTensorType(dst, dataType);

    for (int i = 0; i < srcTensors.size(); i++)
    {
        auto srcTemp = srcTensors[i]->DeepClone();
        auto srcShape = srcTemp->Shape();

        // This is our own copy so move it to the CPU.
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        auto totalSize = srcShape.TotalSize();
        for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
        {
            int cell_size = srcShape[0] / 3;
            int input_size = srcShape[1];

            // row major layout
            int row = targetIndex / input_size,
                col = targetIndex % input_size;

            // source is column major
            int srcIndex = 3 * cell_size * col + row;
            AddDataElementArrayViewToTensorProto(srcTemp, srcIndex, dst);
        }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyGRUStateWeightTensors(
    const std::vector<NDArrayViewPtr> &srcZRTensors, const std::vector<NDArrayViewPtr> &srcHTensors,
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    if (srcZRTensors.size() < 1 || srcZRTensors.size() > 2 || srcZRTensors.size() != srcHTensors.size())
        LogicError("Invalid number of GRU weight tensors");

    DataType dataType = srcZRTensors[0]->GetDataType();
    SetTensorType(dst, dataType);

    for (int i = 0; i < srcZRTensors.size(); i++)
    {
        auto srcZRTemp = srcZRTensors[i]->DeepClone();
        auto srcZRShape = srcZRTemp->Shape();

        auto srcHTemp = srcHTensors[i]->DeepClone();
        auto srcHShape = srcHTemp->Shape();

        int cell_size = srcZRShape[1];

        // This is our own copy so move it to the CPU.
        srcZRTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
        srcHTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        auto totalSize = srcZRShape.TotalSize() + srcHShape.TotalSize();
        for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
        {
            // row major layout
            int row = targetIndex / cell_size,
                col = targetIndex % cell_size;

            int src_index;
            NDArrayViewPtr srcBlockTensor;
            int block = row / cell_size;
            if (block == 0 || block == 1)
            {
                // zr blocks
                srcBlockTensor = srcZRTemp;
                src_index = 2 * cell_size * col + row;
            }
            else if (block == 2)
            {
                // h block
                srcBlockTensor = srcHTemp;
                src_index = cell_size * col + row - cell_size * 2;
            }
            else
            {
                LogicError("Invalid GRU state weight shape");
            }

            AddDataElementArrayViewToTensorProto(srcBlockTensor, src_index, dst);
        }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyRNNWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
    onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)
{
    if (srcTensors.empty())
        return;

    DataType dataType = srcTensors[0]->GetDataType();
    SetTensorType(dst, dataType);

    for (int i = 0; i < srcTensors.size(); i++)
    {
        auto srcTemp = srcTensors[i]->DeepClone();
        auto srcShape = srcTemp->Shape();

        int cell_size = srcShape[0];
        int input_size = srcShape[1];

        // This is our own copy so move it to the CPU.
        srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

        auto totalSize = srcShape.TotalSize();
        for (size_t targetIndex = 0; targetIndex < totalSize; targetIndex++)
        {
            // row major layout
            int row = targetIndex / input_size,
                col = targetIndex % input_size;

            // source is column major
            int srcIndex = cell_size * col + row;
            AddDataElementArrayViewToTensorProto(srcTemp, srcIndex, dst);
        }
    }

    CopyShapeTypeProtoToTensorProto(inputArgType, dst);
}

void CNTKToONNXHelper::CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType /*=nullptr*/)
{
    auto dataType = src->GetDataType();
    auto srcTemp = src->DeepClone();
    auto srcShape = srcTemp->Shape();
    auto totalSize = srcShape.TotalSize();

    // This is our own copy so move it to the CPU.
    srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());

    SetTensorType(dst, dataType);

    switch (dataType)
    {
    case DataType::Float:
    {
        auto data = srcTemp->DataBuffer<float>();
        for (size_t index = 0; index < totalSize; index++) 
            *(dst.mutable_float_data()->Add()) = data[index];

        break;
    }
    case DataType::Double:
    {
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
        CopyShapeTypeProtoToTensorProto(*inputArgType, dst);
    }
    else
    {
        auto dimensions = CNTKToONNXHelper::reverse(srcShape.Dimensions());
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

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, int dynamicAxisCount)
{
    onnx::TypeProto newShape;
    if (shape.HasInferredDimension())
    {
        LogicError("This model has tensor dimensions marked as InferredDimension. Please evaluate"
            "the model with test data at least once and try saving it again.");
    }

    for (int i = 0; i < dynamicAxisCount; i++)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto dimensions = reverse(shape.Dimensions());
    for (auto dimension : dimensions)
    {
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
    }

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const NDShape& shape, bool hasBatchAxis, bool hasSequenceAxis, bool doReverseShape)
{
    onnx::TypeProto newShape;
    if (shape.HasInferredDimension())
    {
        LogicError("This model has tensor dimensions marked as InferredDimension. Please evaluate"
            "the model with test data at least once and try saving it again.");
    }

    // Sequence dimension should be before batch axis after we reverse the shape (reversal happens below).
    if (hasSequenceAxis)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(FreeSequenceDimParam);

    if (hasBatchAxis)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    auto dimensions = shape.Dimensions();
    if (doReverseShape)
        dimensions = reverse(dimensions);
    for (auto dimension : dimensions)
    {
        if (dimension == NDShape::FreeDimension)
        {
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(FreeSequenceDimParam);
        }
        else
        {
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
        }
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

// this method is to undo an idempotent convertion in sanitize_permutation:
// Find the permutation such that when it is applied to the reverse
// of an input gives the reverse of perm applied to the input
// Example:
// input is[a, b, c, d], perm is[3, 0, 2, 1], perm of input is[d, a, c, b]
// we are looking for[2, 1, 3, 0] because when we apply it to[d, c, b, a]
// the result is[b, c, a, d] which is the revese of[d, a, c, b]
std::vector<int64_t> CNTKToONNXHelper::ConvertPermutationCNTKToONNX(const std::vector<Axis> &axes, bool hasBatchAxis)
{
    std::vector<int64_t> permutation(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        int indexToONNXPermTable = axes.size() - i - 1;
        int axisIndexInCNTK = axes[i].StaticAxisIndex();
        int axisIndexInONNX = axes.size() - axisIndexInCNTK - 1;
        permutation[indexToONNXPermTable] = axisIndexInONNX;
    }
    if (hasBatchAxis)
    {
        for (int i = 0; i < permutation.size(); i++)
            permutation[i]++;
        permutation.insert(permutation.begin(), 0);
    }
    return permutation;
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

// whether this op has any input with batch axis
bool CNTKToONNXHelper::OpInputsHasBatchAxis(const FunctionPtr& src)
{
    std::vector<Variable> inputs = src->Inputs();
    for (std::vector<Variable>::const_iterator it = inputs.cbegin(); it != inputs.cend(); it++)
    {
        if ((*it).HasBatchAxis())
            return true;
    }
    return false;
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
CNTK python static axis is zero based. Free/Inferred axis is not static.
ONNX batch axis, if exists, is 0. in this case static axes start from 1.
CNTK cpp get static axis in a dis-normalized form (e.g. -axis - 1)
In general CNTK node attribute contains axis in this dis-normalized form.
This function converts dis-normalized form to ONNX form.
*/
int64_t CNTKToONNXHelper::ConvertAxisToOnnx(const Axis &axis, const Variable &operand)
{
    if (axis.IsBatchAxis())
    {
        if (operand.DynamicAxes().size() == 1)
            return 0;
        else if (operand.DynamicAxes().size() == 2)
            return 1;
        else
            LogicError("Inconsitant Axis in ConvertAxisToOnnx");
    }
    else if (axis.IsSequenceAxis())
    {
        return 0;
    }

    NDShape inputShape = operand.Shape();
    Axis normalizedAxis = NormalizeStaticAxis(const_cast<Axis &>(axis), inputShape.Rank());
    int64_t ax = inputShape.Rank() - normalizedAxis.StaticAxisIndex() - 1;
    ax += operand.DynamicAxes().size();
    return ax;
}

std::vector<int64_t> CNTKToONNXHelper::ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand)
{
    if (std::any_of(axes.cbegin(), axes.cend(), [](const Axis &axis) {return axis== Axis::AllStaticAxes(); }))
    {
        std::vector<int64_t> onnxAxes;
        for (int i = 0; i < operand.Shape().Rank(); i++)
        {
            onnxAxes.push_back(i + operand.DynamicAxes().size());
        }
        return onnxAxes;
    }

    std::vector<int64_t> onnxAxes(axes.size());
    for (int i = 0; i < axes.size(); i++)
    {
        onnxAxes[i] = ConvertAxisToOnnx(axes[i], operand);
    }
    return onnxAxes;
}

/*
ONNX specifies braodcast for elementwise ops in following manners
shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
shape(A) = (2, 3, 4, 5), shape(B) = (5,)
shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

CNTK handles braodcast implicitely as numpy does. For example with above example 4,
the shape of the shall be:
(1, 3, 4, 1) or (3, 4, 1)

more general cases:
same rank:
case1: [a, b, c] + [1, b, 1] - broadcast
case1: [a, b, c] + [a, b, 1] - broadcast
case1: [a, b, c] + [1, b, c] - broadcast
case2: [1, b, 1] + [a, b, c] - swap to become case1 then broadcast
case2: [a, b, 1] + [a, b, c] - swap to become case1 then broadcast
case2: [1, b, c] + [a, b, c] - swap to become case1 then broadcast
case3: [a2, b, c2] + [a, b, c]: cannot broadcast

different ranks:
[a, b, c] + [b, 1]: reshape input[1] to [1, b, 1] to become case 1
[a, b, c] + [b, c]: reshape input[1] to [1, b, c] to become case 1

[b, 1] + [a, b, c]: reshape input[0] to [1, b, 1] to become case 2
[b, c] + [a, b, c]: reshape input[0] to [1, b, c] to become case 2

[a2, b, c2] + [b, c]: reshape input[1] to [1, b, c] to become case 3 (cannot broadcast)
[b, c2] + [a, b, c]: reshape input[0] to [1, b, c2] to become case 3 (cannot broadcast)

Note that there is an addition batch dimension at the front of the shape in ONNX.

*/
std::tuple<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool> CNTKToONNXHelper::AdjustForBroadcastShape(
    const Variable &input1, const Variable &input2)
{
    bool broadcast;
    int axis = 0;
    NDShape shape1 = input1.Shape(), shape2 = input2.Shape();
    bool swapInput = false;

    bool hasAnyBatchAxis = input1.HasBatchAxis() || input2.HasBatchAxis();
    bool hasAnySequenceAxis = input1.HasSequenceAxis() || input2.HasSequenceAxis();

    // CNTK and ONNX dimensions are reversed.
    // Reverse the dimension so that broadcast and axis calculation is in ONNX sense.
    std::vector<int> dims1(reverse(Cast<size_t, int>(shape1.Dimensions())));
    std::vector<int> dims2(reverse(Cast<size_t, int>(shape2.Dimensions())));

    if ((shape1.TotalSize() > 1 && shape2.TotalSize() == 1) || (shape1.TotalSize() == 1 && shape2.TotalSize() > 1))
    {
        broadcast = true;
        swapInput = (shape1.TotalSize() == 1 && shape2.TotalSize() > 1);

        if (swapInput)
            std::swap(dims1, dims2);
        if (hasAnySequenceAxis)
            dims1.insert(dims1.begin(), 1);
        if (hasAnyBatchAxis)
            dims1.insert(dims1.begin(), 1);

        return make_tuple(std::pair<std::vector<int>, std::vector<int>>(dims1, dims2), broadcast, axis, swapInput);
    }

    if (shape1.Rank() < shape2.Rank())
    {
        // This is a case of [b, c] + [a, b, c].
        // Need to swap the inputs to fit into ONNX spec - only right-hand-side argument will be broadcasted.
        std::swap(dims1, dims2);
        swapInput = true;
    }

    if (dims1.size() > dims2.size())
    {
        // This is a case like [a, b, c] + [b, 1]. Make it [a, b, c] + [1, b, 1].
        dims2.insert(dims2.begin(), dims1.size() - dims2.size(), 1);
    }

    // Append batch dimension if needed.
    if (hasAnySequenceAxis)
    {
        dims1.insert(dims1.begin(), 1);
        dims2.insert(dims2.begin(), 1);
    }
    if (hasAnyBatchAxis)
    {
        dims1.insert(dims1.begin(), 1);
        dims2.insert(dims2.begin(), 1);
    }

    bool swapInputDueToDims;
    std::tie<std::vector<int>, bool, int>(dims2, broadcast, axis, swapInputDueToDims) = CalculateBroadcastAxis(dims1, dims2);

    if (broadcast && swapInput && swapInputDueToDims)
    {
        LogicError("Shapes of elementwise binary operation are not compatible.");
    }

    return make_tuple(std::pair<std::vector<int>, std::vector<int>>(dims1, dims2), broadcast, axis, swapInput || swapInputDueToDims);
}

/*
For example with:
case1: [a, b, c] + [ b, 1] - broadcast
broadcast shape = [b], broadcast = true, axis = 1
*/
std::tuple<std::vector<int>, bool, int, bool> CNTKToONNXHelper::CalculateBroadcastAxis(
    const std::vector<int> &dims1, const std::vector<int> &dims2)
{
    bool swapInput = false;
    // this method assumes dims1.size() == dims2.size(), which is granted by caller AdjustForBroadcastShape.
    bool broadCast = false;
    int axis_start = -1;
    int axis_stop = dims2.size();
    for (int i = 0; i < dims2.size(); i++)
    {
        if (dims1[i] != dims2[i])
        {
            if (dims1[i] == 1)
                swapInput = true;

            broadCast = true;
            if (axis_start != -1)
            {
                axis_stop = i;
                break;
            }
        }
        else
            if (dims2[i] != 1 && axis_start == -1)
            {
                axis_start = i;
            }
    }

    if (!broadCast)
    {
        return make_tuple(dims2, broadCast, axis_start, swapInput);
    }

    axis_start = std::max(0, axis_start);

    const std::vector<int> broadcaseInputDims = swapInput ? dims1 : dims2;
    // sanity check;
    for (int i = 0; i < broadcaseInputDims.size(); i++)
    {
        if ((i < axis_start || i >= axis_stop) && broadcaseInputDims[i] != 1)
        {
            LogicError("dimension %d cannot be broadcasted", i);
        }
        else if (i >= axis_start && i < axis_stop && dims1[i] != dims2[i])
        {
            LogicError("dimension %d cannot be broadcasted", i);
        }
    }
    std::vector<int> dimensions;
    for (int i = axis_start; i < axis_stop; i++)
    {
        dimensions.push_back(broadcaseInputDims[i]);
    }

    return make_tuple(dimensions, broadCast, axis_start, swapInput);
}

// prepare an input node arg with correct name and meta data so that LotusIR can make the connection. 
void CNTKToONNXHelper::PrepareRNNInput(const Variable &input, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    std::string inputName = ToString(input.Uid());
    onnx::TypeProto inputArgType = ToTypeProto(input.Shape(), (int)(input.DynamicAxes().size()));
    
    if (input.IsInput() && input.HasSequenceAxis())
        (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_param(FreeSequenceDimParam);

    UpdateONNXType(input.GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(inputName, &inputArgType);
    nodeInputs.push_back(inputArg);
}

void CNTKToONNXHelper::PrepareLSTMInitialStateNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &initialVariables, int batchSize, int cellSize, 
    const std::string &uid, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    std::vector<int> shape({ (int)initialVariables.size(), batchSize , cellSize });
    bool doReverseVec = false;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(initialVariables[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(uid, &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < initialVariables.size(); i++)
    {
        const Variable &variable = initialVariables[i];
        auto srcTensor = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value();
        if (srcTensor->Shape().Rank() == 0 || srcTensor->Shape().TotalSize() == 1)
        {
            srcTensors.push_back(srcTensor);
        }
        else
        {
            // TODO:
            NOT_IMPLEMENTED;
        }
    }

    onnx::TensorProto dstTensor;
    FillTensorWithScalar(srcTensors, dstTensor, shape);

    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(initialVariables[0], variableNode);
}

void CNTKToONNXHelper::PrepareLSTMPeepholeNode(ONNXIR::Graph* graph,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes, const std::vector<Variable> &Ps,
    const std::vector<double> &stabilizerDcCoefs, const std::vector<double> &stabilizerCCoefs,
    std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // this method is called when all Ps are valid parameter/constant variable.
    int hidden_size = Ps[0].Shape()[0];
    int directions = Ps.size() / 3;
    bool doReverseVec = false;
    std::vector<int> shape({ directions, 3 * hidden_size });
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ps[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Ps[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", std::vector<ONNXIR::NodeArg>(), varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    std::vector<double> multipliers;
    for (int i = 0; i < Ps.size(); i++)
    {
        // Because ONNX does not support stabilizer internal to LSTM, we have to fuse 
        // stabilizer operation with peephole weight. Notice that element wise times is
        // applied to stabilizer and peephole weight, it is safe to adjust peephole
        // weight with stabilizer coefficient.
        // Ps is in iof order, 
        // apply dc stabilizer coefficient to i and f
        // apply c stabilizer coefficient to o
        int dir = i / 3;
        switch (i % 3)
        {
        case 0:
        case 2:
            multipliers.push_back(stabilizerDcCoefs[dir]);
            break;
        case 1:
            multipliers.push_back(stabilizerCCoefs[dir]);
            break;
        }
        const Variable &variable = Ps[i];
        
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    CopyTensorsWithMultipliers(srcTensors, multipliers, dstTensor, inputArgType);

    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Ps[0], variableNode);
}

void CNTKToONNXHelper::PrepareLSTMBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true 
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int> shape = Cast<size_t, int>((NDShape({ Bs.size() }).AppendShape(Bs[0].Shape())).Dimensions());

    // ONNX LSTM spec has 2 bias, for forward and backward.
    shape[1] *= 2;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Bs[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, nullptr, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Bs[0], variableNode);
}

void CNTKToONNXHelper::PrepareLSTMWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Ws, double *stabilizerConstants, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true 
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int> shape = Cast<size_t, int>((NDShape({ Ws.size() }).AppendShape(Ws[0].Shape())).Dimensions());
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ws[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Ws[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name(); 
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Ws.size(); i++)
    {
        const Variable &variable = Ws[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    
    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, stabilizerConstants, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Ws[0], variableNode);
}

std::string DeriveDirectionString(const std::vector<FunctionPtr> lstms,
    std::map<RNNDirection, int> directionCount)
{
    return lstms.size() == 2 ? RNNDirectionBidirection :
        (directionCount[RNNDirection::Backward] == 1 ? RNNDirectionReverse : RNNDirectionForward);
}

void AddEmptyInput(std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    ONNXIR::NodeArg inputArg("", nullptr);
    nodeInputs.emplace_back(inputArg);
}

void SanityCheckForConstantOrParameters(const std::vector<Variable> &variables)
{
    for (auto variable : variables)
    {
        if (variable.IsInitialized() && !variable.IsConstant() && !variable.IsParameter())
            CNTK::LogicError("Input to RNN op is not a constant or parameter: Variable Name: %S, Variable Uid: %S",
                variable.Name().c_str(), 
                variable.Uid().c_str());
    }
}

ONNXIR::Node* CNTKToONNXHelper::CreateLSTMNode(const FunctionPtr &src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<FunctionPtr> lstms = GetRNNBlocksFromSingleOrBidirectionalRNN(src, "LSTM");
    
    // order forward, backward
    std::map<RNNDirection, int> directionCount({ { RNNDirection::Forward, 0 } ,{ RNNDirection::Backward, 0 } });

    // The following construct refers to ONNX spec:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm
    // specifically, for attrubute and variable dimension. 
    // We use the term from the spec as possible as we can to maintain a close correlation
    // to the ONNX specification. 
    
    int num_directions = lstms.size();
    // A list of 3 (or 6 if bidirectional) activation functions for input, output, forget, cell, and hidden.
    std::vector<std::string> activations(num_directions * 3);

    // TODO:
    // In principle all these variables shall be treated as either constant or op output. 
    // In reality except X, all other inputs to LSTM can be treated as constant.
    // We will investigate when there is any model that failed above assumption.
    std::vector<Variable> Xs(num_directions), Ws(num_directions), Rs(num_directions), Bs(num_directions),
        initialHs(num_directions), initialCs(num_directions), Ps(num_directions * 3);

    std::vector<Variable> Yhs(lstms.size()), Ycs(lstms.size());

    // 
    std::vector<double> stabilizerDhCoefs(lstms.size()), stabilizerDcCoefs(lstms.size()), stabilizerCCoefs(lstms.size());

    for (std::vector<FunctionPtr>::const_iterator itLSTMBlock = lstms.cbegin(); itLSTMBlock != lstms.cend(); itLSTMBlock++)
    {
        // src has to be an LSTM node. 
        const FunctionPtr& lstm = *itLSTMBlock;
        string f_activation, g_activation, h_activation;
        RNNDirection direction;
        Variable initStateH, initStateC;
        Variable peepholeCi, peepholeCo, peepholeCf;
        double stabilizer_dh = 1, stabilizer_dc = 1, stabilizer_c = 1;
        TraceLSTMPathes(lstm, f_activation, g_activation, h_activation, direction,
            initStateH, initStateC,
            peepholeCi, peepholeCo, peepholeCf, stabilizer_dh, stabilizer_dc, stabilizer_c);

        directionCount[direction]++;

        int directionIndex = lstms.size() == 1 ? 0 : (direction ? 1 : 0);

        initialHs[directionIndex] = initStateH;
        initialCs[directionIndex] = initStateC;

        Ps[LSTMPeepholeCount * directionIndex + LSTMPeepholeCountCiIndex] = peepholeCi;
        Ps[LSTMPeepholeCount * directionIndex + LSTMPeepholeCountCoIndex] = peepholeCo;
        Ps[LSTMPeepholeCount * directionIndex + LSTMPeepholeCountCfIndex] = peepholeCf;

        activations[directionIndex * LSTMActivationCount + LSTMActivationFIndex] = f_activation;
        activations[directionIndex * LSTMActivationCount + LSTMActivationGIndex] = g_activation;
        activations[directionIndex * LSTMActivationCount + LSTMActivationHIndex] = h_activation;
    
        std::vector<Variable> inputs = lstm->Inputs();

        stabilizerDhCoefs[directionIndex] = stabilizer_dh;
        stabilizerDcCoefs[directionIndex] = stabilizer_dc;
        stabilizerCCoefs[directionIndex] = stabilizer_c;

        // input (always the last one), weight, hidden weight, and bias have fixed indices. 
        // Thus we do not bother obtain them through traversing.
        int inputIndex = inputs.size() - 1;
        Xs[directionIndex] = inputs[inputIndex];

        Ws[directionIndex] = inputs[CNTKLSTMWeightIndex];
        Rs[directionIndex] = inputs[CNTKLSTMHiddenWeightIndex];
        Bs[directionIndex] = inputs[CNTKLSTMBiasIndex];

        std::vector<Variable> outputs = lstm->Outputs();

        Yhs[directionIndex] = outputs[CNTKLSTMOutputYhIndex];
        Ycs[directionIndex] = outputs[CNTKLSTMOutputChIndex];
    }

    SanityCheckForConstantOrParameters(initialHs);
    SanityCheckForConstantOrParameters(initialCs);
    SanityCheckForConstantOrParameters(Ps);

    // ensure that if there is one direction, it is not backward.
    // if there two directions, they are forward and backward, and
    // that the inputs (Xs) are the same.
    if (std::any_of(directionCount.begin(), directionCount.end(), [](std::map<RNNDirection, int>::value_type &v) {return v.second > 1; }))
    {
        LogicError("LSTM node is invalid because there should be no more than one path in each direction.");
    }
    if (lstms.size() == 2 && Xs[0] != Xs[1])
    {
        LogicError("Bi-directional LSTM node is invalid because the two LSTM nodes do not share one same input.");
    }

    string direction = DeriveDirectionString(lstms, directionCount);

    // TODO: following commented out attributes are not supported. Use default.
    // float clip; // no clip yet
    // std::vector<float> activation_alpha;    // no supported activation need alpha.
    // std::vector<float> activation_beta;    // no supported activation need beta.
    int hidden_size = lstms[0]->Outputs()[0].Shape()[0];
    int output_sequence = RNNOutputSequence;        // LSTM in CNTK always output full sequence of output

    // TODO: implement peephole
    // Variable P;

    // inputs
    std::vector<ONNXIR::NodeArg> nodeInputs;
    PrepareRNNInput(Xs[0], nodeInputs);
    PrepareLSTMWeightNode(graph, variableNodes, Ws, nullptr, nodeInputs);
    PrepareLSTMWeightNode(graph, variableNodes, Rs, &stabilizerDhCoefs[0], nodeInputs);
    
    {
        bool hasBias = std::all_of(Bs.begin(), Bs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (hasBias)
        {
            PrepareLSTMBiasNode(graph, variableNodes, Bs, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }

        // TODO: enable sequence_lens. It requires additional model input of batched sequence data layout. 
        // Need to investigate how this is done with CNTK python API.
        bool has_sequence_lens = false;
        std::string sequence_lens_inputName = "sequence_lens___";
        if (has_sequence_lens)
        {
            onnx::TypeProto inputArgType = ToTypeProto(std::vector<int>({ 1 }), false);
            inputArgType.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
            ONNXIR::NodeArg inputArg(sequence_lens_inputName, &inputArgType);
            nodeInputs.push_back(inputArg);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); }); 
        if (has_initial_h)
        {
            std::string hiddenUid = ToString(Yhs[0].Uid()) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }

        bool has_initial_c = std::all_of(initialCs.begin(), initialCs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_c)
        {
            std::string cellUid = ToString(Ycs[0].Uid()) + "_initial_c";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialCs, FreeBatchSize, hidden_size, cellUid, nodeInputs);
        } else
        {
            AddEmptyInput(nodeInputs);
        }

        // peephole
        bool hasPeephole = std::all_of(Ps.begin(), Ps.end(), [](Variable &v) {return v.IsInitialized(); });
        if (hasPeephole)
        {
            PrepareLSTMPeepholeNode(graph, variableNodes, Ps, stabilizerDcCoefs, stabilizerCCoefs, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }
    }

    std::vector<ONNXIR::NodeArg> nodeOutputs;
    {
        if (output_sequence == 1)
        {
            std::string nodeName;
            if (lstms.size() == 1)
                nodeName = ToString(Yhs[0].Uid());
            else
                nodeName = ToString(src->Output().Uid());

            auto outputArgType = ToTypeProto(std::vector<int>({ FreeSequenceLen, (int)Yhs.size(), FreeBatchSize, (int)Yhs[0].Shape()[0] }), false);
            UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        else
        {
            ONNXIR::NodeArg outputArg("", nullptr);
            nodeOutputs.push_back(outputArg);
        }

        {
            Variable Yh = Yhs[0];
            std::string nodeName = ToString(Yh.Uid()) + "_h";
            // TODO: batchSize is fixed to one. Needs to find out how to handle bacth axis as a free dimension.
            const int batchSize = 1;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Yhs.size(), batchSize, (int)Yh.Shape()[0]}), false);
            UpdateONNXType(Yh.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        {
            Variable Yc = Ycs[0];
            std::string nodeName = ToString(Yc.Uid()) + "_c";
            int batchSize = 1;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Ycs.size(), batchSize, (int)Yc.Shape()[0] }), false);
            UpdateONNXType(Yc.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
    }

    // TODO: Except X, all other inputs to LSTM are treated as constant.
    // It is highly unlikely that any other input is an output of an op. 
    // We will investigate once it is real.
    if (Xs[0].Owner().get() != nullptr)
        CreateNode(Xs[0].Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());
    ONNXIR::Node *lstmNode = graph->AddNode(nodeName, "LSTM", "", nodeInputs, nodeOutputs);

    lstmNode->AddAttribute("activations", activations);
    lstmNode->AddAttribute("direction", direction);
    lstmNode->AddAttribute("hidden_size", (int64_t)hidden_size);
    lstmNode->AddAttribute("output_sequence", (int64_t)output_sequence);
    
    // TODO: make bidirectional LSTM work by figuring out output data 
    // layout transpose in InsertReshapeNodeToCNTKFunction. 
    if (lstms.size() == 2)
        NOT_IMPLEMENTED;

    // squeeze direction axis out. This is safe because it is not bi-directional node.

    std::vector<int> shape({ FreeSequenceLen, 1, hidden_size });

    ONNXIR::Node *squeezedLSTMNode = InsertReshapeNodeToCNTKFunction(src, lstmNode, shape, graph);

    functionNodes.emplace(src, squeezedLSTMNode);
    return squeezedLSTMNode;
}

void CNTKToONNXHelper::PrepareGRUBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;
    int numDirections = Bs.size();
    int hiddenSize = Bs[0].Shape()[0] / GRUWeightDimensionHiddenMultiplier;

    std::vector<int> shape({ numDirections, GRUBiasDimensionHiddenMultiplier * hiddenSize });

    // ONNX GRU spec has 2 bias, for forward and backward.
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Bs[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    CopyRNNBiasTensors(srcTensors, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Bs[0], variableNode);
}

void CNTKToONNXHelper::PrepareGRUZRHWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes, 
    const std::vector<Variable> &Rzrs, const std::vector<Variable> &Rhs, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    int numDirections = Rzrs.size();
    int hiddenSize = Rzrs[0].Shape().Dimensions()[1];
    std::vector<int> shape({ numDirections, GRUWeightDimensionHiddenMultiplier * hiddenSize, hiddenSize });
    onnx::TypeProto inputArgType = ToTypeProto(shape, false);
    UpdateONNXType(Rzrs[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Rzrs[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcZRTensors, srcHTensors;
    for (int i = 0; i < Rzrs.size(); i++)
    {
        const Variable &variable = Rzrs[i];
        srcZRTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());

        const Variable &variableH1 = Rhs[i];
        srcHTensors.push_back(variableH1.IsParameter() ? Parameter(variableH1).Value() : Constant(variableH1).Value());
    }

    onnx::TensorProto dstTensor;

    CopyGRUStateWeightTensors(srcZRTensors, srcHTensors, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Rzrs[0], variableNode);
}

void CNTKToONNXHelper::PrepareRNNWeightNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Ws, std::vector<ONNXIR::NodeArg> &nodeInputs, 
    std::function<void(const std::vector<NDArrayViewPtr> &srcTensors,
        onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)> weightConverter)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;

    std::vector<int> shape = Cast<size_t, int>((NDShape({ Ws.size() }).AppendShape(Ws[0].Shape())).Dimensions());
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ws[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Ws[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Ws.size(); i++)
    {
        const Variable &variable = Ws[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    weightConverter(srcTensors, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Ws[0], variableNode);
}

ONNXIR::Node *CNTKToONNXHelper::CreateGRUNode(const FunctionPtr &src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<FunctionPtr> grus = GetRNNBlocksFromSingleOrBidirectionalRNN(src, "GRU");

    // order forward, backward
    std::map<RNNDirection, int> directionCount({ { RNNDirection::Forward, 0 } ,{ RNNDirection::Backward, 0 } });

    // The following construct refers to ONNX spec:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm
    // specifically, for attrubute and variable dimension. 
    // We use the term from the spec as possible as we can to maintain a close correlation
    // to the ONNX specification. 

    int num_directions = grus.size();
    // A list of 3 (or 6 if bidirectional) activation functions for input, output, forget, cell, and hidden.
    std::vector<std::string> activations(num_directions * GRUActivationCount);

    // TODO:
    // In principle all these variables shall be treated as either constant or op output. 
    // In reality except X, all other inputs to LSTM can be treated as constant.
    std::vector<Variable> Xs(num_directions), Ws(num_directions), Rzrs(num_directions), 
        Rhs(num_directions), Bs(num_directions),
        initialHs(num_directions);
    
    std::vector<Variable> Yhs(grus.size());

    for (std::vector<FunctionPtr>::const_iterator itGRUBlock = grus.cbegin(); itGRUBlock != grus.cend(); itGRUBlock++)
    {
        // src has to be an GRU node. 
        const FunctionPtr& gru = *itGRUBlock;
        std::vector<Variable> inputs = gru->Inputs();
        if (inputs.size() != CNTKGRUInputCount)
            LogicError("Unkown GRU configuration. The GRU node might be created with self stabilization. Such GRU ops cannot be converted to ONNX.");

        string f_activation, g_activation;
        RNNDirection direction;
        Variable initStateH;
        TraceGRUPathes(gru, f_activation, g_activation, direction, initStateH);

        directionCount[direction]++;

        int directionIndex = grus.size() == 1 ? 0 : (direction ? 1 : 0);

        initialHs[directionIndex] = initStateH;

        activations[directionIndex * GRUActivationCount + GRUActivationFIndex] = f_activation;
        activations[directionIndex * GRUActivationCount + GRUActivationGIndex] = g_activation;

        // input (always the last one), weight, hidden weight, and bias have fixed indices. 
        // Thus we do not bother obtain them through traversing.
        int inputIndex = inputs.size() - 1;
        Xs[directionIndex] = inputs[inputIndex];

        Ws[directionIndex] = inputs[CNTKGRUWeightIndex];
        SanityCheckForConstantOrParameters(Ws);

        Rzrs[directionIndex] = inputs[CNTKGRUHiddenWeightZRIndex];
        SanityCheckForConstantOrParameters(Rzrs);

        Rhs[directionIndex] = inputs[CNTKGRUHiddenWeightHIndex];
        SanityCheckForConstantOrParameters(Rhs);

        Bs[directionIndex] = inputs[CNTKGRUBiasIndex];
        SanityCheckForConstantOrParameters(Bs);

        std::vector<Variable> outputs = gru->Outputs();

        Yhs[directionIndex] = outputs[CNTKLSTMOutputYhIndex];
    }

    // ensure that if there is one direction, it is not backward.
    // if there two directions, they are forward and backward, and
    // that the inputs (Xs) are the same.
    if (std::any_of(directionCount.begin(), directionCount.end(), [](std::map<RNNDirection, int>::value_type &v) {return v.second > 1; }))
    {
        LogicError("GRU node is invalid because there should be no more than one path in each direction.");
    }
    if (grus.size() == 2 && Xs[0] != Xs[1])
    {
        LogicError("Bi-directional GRU node is invalid because the two LSTM nodes do not share one same input.");
    }

    string direction = DeriveDirectionString(grus, directionCount);

    // an RNN output size is the hidden size
    int hidden_size = grus[0]->Outputs()[0].Shape()[0];

    // inputs
    std::vector<ONNXIR::NodeArg> nodeInputs;
    PrepareRNNInput(Xs[0], nodeInputs);
    PrepareRNNWeightNode(graph, variableNodes, Ws, nodeInputs, CopyGRUWeightTensors);
    PrepareGRUZRHWeightNode(graph, variableNodes, Rzrs, Rhs, nodeInputs);

    {
        bool hasBias = std::all_of(Bs.begin(), Bs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (hasBias)
        {
            PrepareGRUBiasNode(graph, variableNodes, Bs, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }

        {
            // sequence_lens is not supported
            AddEmptyInput(nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_h)
        {
            std::string hiddenUid = ToString(Yhs[0].Uid()) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }
    }

    const int output_sequence = RNNOutputSequence;       // GRU in CNTK always output full sequence of output
    std::vector<ONNXIR::NodeArg> nodeOutputs;
    {
        if (output_sequence == 1)
        {
            std::string nodeName;
            if (grus.size() == 1)
                nodeName = ToString(Yhs[0].Uid());
            else
                nodeName = ToString(src->Output().Uid());

            auto outputArgType = ToTypeProto(std::vector<int>({ FreeSequenceLen, (int)Yhs.size(), FreeBatchSize, (int)Yhs[0].Shape()[0] }), false);
            UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        else
        {
            ONNXIR::NodeArg outputArg("", nullptr);
            nodeOutputs.push_back(outputArg);
        }

        {
            Variable Yh = Yhs[0];
            std::string nodeName = ToString(Yh.Uid()) + "_h";
            // TODO: batchSize is fixed to one. Needs to find out how to handle bacth axis as a free dimension.
            const int batchSize = 1;
            const bool doReverseVec = false;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Yhs.size(), batchSize, (int)Yh.Shape()[0] }), doReverseVec);
            UpdateONNXType(Yh.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
    }

    // TODO: Except X, all other inputs to GRU are treated as constant.
    // It is highly unlikely that any other input is an output of an op. 
    // We will investigate once it is real.
    if (Xs[0].Owner().get() != nullptr)
        CreateNode(Xs[0].Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());
    ONNXIR::Node *gruNode = graph->AddNode(nodeName, "GRU", "", nodeInputs, nodeOutputs);

    gruNode->AddAttribute("activations", activations);
    gruNode->AddAttribute("direction", direction);
    gruNode->AddAttribute("hidden_size", (int64_t)hidden_size);
    gruNode->AddAttribute("output_sequence", (int64_t)output_sequence);

    // TODO: make bidirectional GRU work by figuring out output data 
    // layout transpose in InsertReshapeNodeToCNTKFunction. 
    if (grus.size() == 2)
        NOT_IMPLEMENTED;

    // TODO: uncomment this code once LotusRT output shape matches ONNX
    // squeeze direction axis out. This is safe because it is not bi-directional node.
    std::vector<int> shape({ FreeSequenceLen, 1, hidden_size });
    ONNXIR::Node *squeezedLSTMNode = InsertReshapeNodeToCNTKFunction(src, gruNode, shape, graph);
    functionNodes.emplace(src, squeezedLSTMNode);
    return squeezedLSTMNode;
}

void CNTKToONNXHelper::PrepareRNNBiasNode(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::vector<Variable> &Bs, std::vector<ONNXIR::NodeArg> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;
    int numDirections = Bs.size();
    int hiddenSize = Bs[0].Shape()[0];

    std::vector<int> shape({ numDirections, 2 * hiddenSize });

    // ONNX GRU spec has 2 bias, for forward and backward.
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    ONNXIR::NodeArg inputArg(ToString(Bs[0].Uid()), &inputArgType);
    std::vector<ONNXIR::NodeArg> varOutputs({ inputArg });
    std::vector<ONNXIR::NodeArg> varInputs;
    std::string inputName = inputArg.Name();
    ONNXIR::Node* variableNode = graph->AddNode(inputName, "Constant", "", varInputs, varOutputs);

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;

    CopyRNNBiasTensors(srcTensors, dstTensor, inputArgType);
    variableNode->AddAttribute("value", dstTensor);
    nodeInputs.push_back(inputArg);

    variableNodes.emplace(Bs[0], variableNode);
}


ONNXIR::Node *CNTKToONNXHelper::CreateRNNNode(const FunctionPtr &src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<FunctionPtr> rnns = GetRNNBlocksFromSingleOrBidirectionalRNN(src, "RNNStep");

    // order forward, backward
    std::map<RNNDirection, int> directionCount({ { RNNDirection::Forward, 0 } ,{ RNNDirection::Backward, 0 } });

    // The following construct refers to ONNX spec:
    // https://github.com/onnx/onnx/blob/master/docs/Operators.md#lstm
    // specifically, for attrubute and variable dimension. 
    // We use the term from the spec as possible as we can to maintain a close correlation
    // to the ONNX specification. 

    int num_directions = rnns.size();
    // A list of 3 (or 6 if bidirectional) activation functions for input, output, forget, cell, and hidden.
    std::vector<std::string> activations(num_directions);

    // TODO:
    // In principle all these variables shall be treated as either constant or op output. 
    // In reality except X, all other inputs to LSTM can be treated as constant.
    std::vector<Variable> Xs(num_directions), Ws(num_directions), Rs(num_directions),
        Bs(num_directions), initialHs(num_directions);

    std::vector<Variable> Yhs(rnns.size());

    for (std::vector<FunctionPtr>::const_iterator itRNNBlock = rnns.cbegin(); itRNNBlock != rnns.cend(); itRNNBlock++)
    {
        // src has to be an RNN node. 
        const FunctionPtr& rnn = *itRNNBlock;
        std::vector<Variable> inputs = rnn->Inputs();
        if (inputs.size() != CNTKRNNInputCount)
            LogicError("A RNN block does not have expected input count (%d). Actual input count is %d", (int)CNTKRNNInputCount, (int)inputs.size());

        string activation;
        RNNDirection direction;
        Variable initStateH;
        TraceRNNPathes(rnn, activation, direction, initStateH);

        directionCount[direction]++;

        int directionIndex = rnns.size() == 1 ? 0 : (direction ? 1 : 0);

        initialHs[directionIndex] = initStateH;

        activations[directionIndex] = activation;

        Xs[directionIndex] = inputs[CNTKRNNInputIndex];

        Ws[directionIndex] = inputs[CNTKRNNWeightIndex];

        Rs[directionIndex] = inputs[CNTKRNNHweightIndex];

        Bs[directionIndex] = inputs[CNTKRNNBiasIndex];

        std::vector<Variable> outputs = rnn->Outputs();

        Yhs[directionIndex] = outputs[CNTKRNNOutputYhIndex];
    }

    SanityCheckForConstantOrParameters(Ws);
    SanityCheckForConstantOrParameters(Rs);
    SanityCheckForConstantOrParameters(Bs);

    // ensure that if there is one direction, it is not backward.
    // if there two directions, they are forward and backward, and
    // that the inputs (Xs) are the same.
    if (std::any_of(directionCount.begin(), directionCount.end(), [](std::map<RNNDirection, int>::value_type &v) {return v.second > 1; }))
    {
        LogicError("RNN node is invalid because there should be no more than one path in each direction.");
    }
    if (rnns.size() == 2 && Xs[0] != Xs[1])
    {
        LogicError("Bi-directional RNN node is invalid because the two RNN nodes do not share one same input.");
    }

    string direction = DeriveDirectionString(rnns, directionCount);

    // an RNN output size is the hidden size
    int hidden_size = rnns[0]->Outputs()[0].Shape()[0];

    // inputs
    std::vector<ONNXIR::NodeArg> nodeInputs;
    PrepareRNNInput(Xs[0], nodeInputs);
    PrepareRNNWeightNode(graph, variableNodes, Ws, nodeInputs, CopyRNNWeightTensors);
    PrepareRNNWeightNode(graph, variableNodes, Rs, nodeInputs, CopyRNNWeightTensors);

    {
        bool hasBias = std::all_of(Bs.begin(), Bs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (hasBias)
        {
            PrepareRNNBiasNode(graph, variableNodes, Bs, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }

        {
            // sequence_lens is not supported
            AddEmptyInput(nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_h)
        {
            std::string hiddenUid = ToString(Yhs[0].Uid()) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(nodeInputs);
        }
    }

    const int output_sequence = RNNOutputSequence;       // RNN in CNTK always output full sequence of output
    std::vector<ONNXIR::NodeArg> nodeOutputs;
    {
        if (output_sequence == 1)
        {
            std::string nodeName;
            if (rnns.size() == 1)
                nodeName = ToString(Yhs[0].Uid());
            else
                nodeName = ToString(src->Output().Uid());

            auto outputArgType = ToTypeProto(std::vector<int>({ FreeSequenceLen, (int)Yhs.size(), FreeBatchSize, (int)Yhs[0].Shape()[0] }), false);
            UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
        else
        {
            ONNXIR::NodeArg outputArg("", nullptr);
            nodeOutputs.push_back(outputArg);
        }

        {
            Variable Yh = Yhs[0];
            std::string nodeName = ToString(Yh.Uid()) + "_h";

            const int batchSize = 1;
            const bool doReverseVec = false;
            auto outputArgType = ToTypeProto(std::vector<int>({ (int)Yhs.size(), batchSize, (int)Yh.Shape()[0] }), doReverseVec);
            UpdateONNXType(Yh.GetDataType(), outputArgType);
            ONNXIR::NodeArg outputArg(nodeName, &outputArgType);
            nodeOutputs.push_back(outputArg);
        }
    }

    if (Xs[0].Owner().get() != nullptr)
        CreateNode(Xs[0].Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());
    ONNXIR::Node *rnnNode = graph->AddNode(nodeName, "RNN", "", nodeInputs, nodeOutputs);

    rnnNode->AddAttribute("activations", activations);
    rnnNode->AddAttribute("direction", direction);
    rnnNode->AddAttribute("hidden_size", (int64_t)hidden_size);
    rnnNode->AddAttribute("output_sequence", (int64_t)output_sequence);

    //// TODO: make bidirectional RNN work by figuring out output data 
    //// layout transpose in InsertReshapeNodeToCNTKFunction. 
    if (rnns.size() == 2)
        NOT_IMPLEMENTED;

    //// TODO: uncomment this code once LotusRT output shape matches ONNX
    //// squeeze direction axis out. This is safe because it is not bi-directional node.
    std::vector<int> shape({ FreeSequenceLen, 1, hidden_size });
    ONNXIR::Node *squeezedRNNNode = InsertReshapeNodeToCNTKFunction(src, rnnNode, shape, graph);
    functionNodes.emplace(src, squeezedRNNNode);
    return squeezedRNNNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddReshapeNode(const ONNXIR::NodeArg &nodeArg, const std::vector<int> &newShape, const std::string &outArgName, ONNXIR::Graph* graph)
{
    ONNXIR::NodeArg outputArg(outArgName, nullptr);
    ONNXIR::Node* reshapeNode = graph->AddNode(nodeArg.Name() + string("_reshape"), "Reshape", "", { nodeArg }, { outputArg });
    reshapeNode->AddAttribute("shape", ToINTS(newShape, false));
    return reshapeNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddMatMulNode(const ONNXIR::NodeArg &nodeArg1, const ONNXIR::NodeArg &nodeArg2, ONNXIR::Graph* graph)
{
    ONNXIR::NodeArg outputArg(nodeArg1.Name() + "matmul_out", nullptr);
    ONNXIR::Node* argMatMulNode = graph->AddNode(
        nodeArg1.Name() + string("_matmul"), "MatMul", "", { nodeArg1, nodeArg2 }, { outputArg });
    return argMatMulNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddArgMaxNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph, int axis)
{
    // ONNXIR::NodeArg inputArg(nodeArg.Name(), nullptr);
    ONNXIR::NodeArg outputArg(nodeArg.Name() + "argmax_out", nullptr);
    ONNXIR::Node* argMaxNode = graph->AddNode(nodeArg.Name() + string("_argmax"), "ArgMax", "", { nodeArg }, { outputArg });
    argMaxNode->AddAttribute("axis", (int64_t)axis);
    return argMaxNode;
}

ONNXIR::Node *CNTKToONNXHelper::AddCastNode(const ONNXIR::NodeArg &nodeArg, ONNXIR::Graph* graph, const std::string &toType)
{
    // ONNXIR::NodeArg inputArg(nodeArg.Name(), nullptr);
    ONNXIR::NodeArg outputArg(nodeArg.Name() + "cast_out", nullptr);
    ONNXIR::Node* castNode = graph->AddNode(nodeArg.Name() + string("_cast"), "Cast", "", { nodeArg }, { outputArg });
    castNode->AddAttribute("to", toType);
    return castNode;
}

// This method is to workaround the fact that ONNX LSTM spec does not allow easy layer stacking.
// Mapping memory layout from a bidirectional LSTM may need some work.
// For now we simply treat a bidirectional LSTM as two separate LSTMs. We use this method to reshape 
// LSTM output to squeeze away the direction dimension.
// TODO: extend this method to handle bidirection LSTMs.
ONNXIR::Node *CNTKToONNXHelper::InsertReshapeNodeToCNTKFunction(const FunctionPtr &src, ONNXIR::Node* node, const std::vector<int> &shape, ONNXIR::Graph* graph)
{
    FunctionPtr blockRoot = src->BlockRoot();
    Variable output;
    if (Operators::IsRNNOp(ToString(src->OpName())))
        output = src->Outputs()[0];
    else
        // a bidirection LSTM case
        NOT_IMPLEMENTED
    
    std::string nodeName = ToString(blockRoot->Uid());
    
    // NodeArg name of the output of the reshaped node
    std::string outputNodeArgName = ToString(output.Uid());

    // We need to name reshape node's output arg with LSTM output name.
    // Thus we need to give LSTM node output a different name.   
    std::vector<ONNXIR::NodeArg>& outputArgs = node->Mutable_OutputDefs();
    TensorShapeProto inputShape(*outputArgs[0].Shape());

    // replace LSTM output arg with one of different name and same shape. 
    std::string lstmToReshapeNodeArgName = outputNodeArgName + "_tmp";
    outputArgs[0] = ONNXIR::NodeArg(lstmToReshapeNodeArgName, nullptr);
    outputArgs[0].SetShape(inputShape);

    // 
    ONNXIR::NodeArg inputArg = ONNXIR::NodeArg(lstmToReshapeNodeArgName, nullptr);
    inputArg.SetShape(inputShape);

    // this is the output NodeArg of the reshaped node. It has to be named 
    // with the original node's output NodeArg so that ONNXIR can make a the connection. 
    onnx::TypeProto typeProto = ToTypeProto(shape, false);
    ONNXIR::NodeArg outputArg(outputNodeArgName, &typeProto);

    ONNXIR::Node* reshapeNode = graph->AddNode(nodeName + string("_reshape"), "Reshape", "", { inputArg }, { outputArg });
    reshapeNode->AddAttribute("shape", ToINTS(shape, false));
    return reshapeNode;
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

    // TODO: uncomment this code once bidirectional LSTM is supprted.
    //if (opName == "Splice")
    //{ 
    //    std::vector<Variable> inputs = src->Inputs();
    //    bool bidiectionalLSTM = inputs.size() == 2 &&
    //        std::all_of(inputs.begin(), inputs.end(), [](Variable &input) {return input.Owner() != nullptr && input.Owner()->OpName() == L"LSTM"; });
    //    if (bidiectionalLSTM)
    //        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    //}
    //else 
    if (opName == "RNNStep")
    { 
        return CreateRNNNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (opName == "GRU")
    {
        return CreateGRUNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (opName == "LSTM")
    {
        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (opName == "Combine")
    {
        for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
        {
            auto input = src->Inputs()[inputIndex];
            CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);
        }

        // not a single node, 
        return nullptr;
    }
    else if (opName == "OptimizedRNNStack")
        return CreateONNXNodesForOptimizedRNNStack(src, graph, functionNodes, variableNodes, compositeOutputsMap);
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
            auto outputArgType = ToTypeProto(output.Shape(), output.HasBatchAxis(), output.HasSequenceAxis());
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

            // Special case handling of LayerNormalization layer because it changes
            // ops dynamically based on value of inputs. If more such cases ops are seen, 
            // this should be abstracted out from here. 
            if (ToString(src->OpName()) == "LayerNormalization")
            {
                // If non-zero epsilon was specified, a fourth input is included 
                // which must be ignored because we cannot export epsilon to ONNX.
                // See LayerNormalization branch in AddNode() below.
                if (src->Inputs().size() == 4 && inputIndex == 0 && input.IsConstant())
                    continue;
            }

            if (FilterInput(src, input, inputIndex))
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

            onnx::TypeProto inputArgType;

            bool broadcastSwapped = false;
            if (Operators::SupportBroadcast(src->OpName()))
            {
                std::pair<std::vector<int>, std::vector<int>> adjustedDims;
                bool broadcast = false;
                int axis = 0;
                int index0, index1;
                std::tie<int, int>(index0, index1) = Operators::GetElementWiseInputIndices(src->OpName());

                if (index0 != inputIndex && index1 != inputIndex)
                    continue;

                std::tie<std::pair<std::vector<int>, std::vector<int>>, bool, int, bool>(adjustedDims, broadcast, axis, broadcastSwapped) =
                    AdjustForBroadcastShape(src->Inputs()[index0], src->Inputs()[index1]);
                if (inputIndex == index0)
                    inputArgType = ToTypeProto(adjustedDims.first, false);
                else if (inputIndex == index1)
                    inputArgType = ToTypeProto(adjustedDims.second, false);
            }
            else if (opName == "Splice")
            {
                // for ops like Concat, batch axis may exist in one of the operand
                // CNTK allows the other operand(s) not having batch axis. But ONNX 
                // requires operands to have the same rank
                inputArgType = ToTypeProto(input.Shape(), OpInputsHasBatchAxis(src));
            }
            else if (opName == "Hardmax" || opName == "ImageScaler")
            {
                // ONNX specifies that hardmax, ImageScaler always need a batch axis
                inputArgType = ToTypeProto(input.Shape(), true);
            }
            else
            {
                if (isConstant && opName == "BatchNormalization" && (inputIndex > 0 && inputIndex <= 4)
                    && input.Shape().Rank() == 2)
                    // this is a workaround for brainscript models that have rank = 2 for BN inputs.
                    inputArgType = ToTypeProto(input.Shape().SubShape(0, input.Shape().Rank() - 1));
                else
                    inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis(), input.HasSequenceAxis());
                if (input.IsInput() && input.HasSequenceAxis())
                    (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_param(FreeSequenceDimParam);
            }

            UpdateONNXType(input.GetDataType(), inputArgType);
            ONNXIR::NodeArg inputArg(inputName, &inputArgType);

            inputs.push_back(inputArg);

            if (broadcastSwapped && inputs.size() == 2)
                swap(inputs[0], inputs[1]);
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
    if (Operators::IsLoopOp(opName))
    {
        // avoid infinite loop 
        return;
    }

    if (!Operators::IsRNNOp(opName) &&
        src->IsBlock() && (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
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
                if (!Operators::IsRNNOp(opName) && input.IsPlaceholder())
                    LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
            }

            if (input.IsInitialized() && input.IsOutput())
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
            node->AddAttribute("alpha", alpha);
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

            std::vector<int64_t> axes = ConvertAxesToOnnx(reductionAxes, src->Inputs()[0]);
            node->AddAttribute("axes", axes);
        }
        else if (src->OpName() == L"TransposeAxes")
        {
            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> permutation = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                // CNTK permutation attribute is argsorted. Shall redo argsort (undo) to get the original python/ONNX perm attribute.
                std::vector<int64_t> perm = ConvertPermutationCNTKToONNX(permutation, src->Inputs()[0].HasBatchAxis());
                node->AddAttribute(attributesMap[L"axisVec"], perm);
            }
            else if (src->Attributes().Contains(L"axis1") && src->Attributes().Contains(L"axis2"))
            {
                // swapaxis: permutation is between two axes
                int rank = src->Output().Shape().Rank();
                std::vector<int64_t> perm;
                bool hasBatchAxis = src->Inputs()[0].HasBatchAxis();
                for (int index = 0; index < (hasBatchAxis ? (rank + 1) : rank); index++)
                {
                    perm.push_back(index);
                }

                Axis axis1 = (Axis)(src->Attributes()[L"axis1"].Value<Axis>()).StaticAxisIndex();
                Axis axis2 = (Axis)(src->Attributes()[L"axis2"].Value<Axis>()).StaticAxisIndex();
                int64_t axisIndex1 = ConvertAxisToOnnx(axis1, src->Inputs()[0]);
                int64_t axisIndex2 = ConvertAxisToOnnx(axis2, src->Inputs()[0]);
                std::swap(perm[axisIndex1], perm[axisIndex2]);
                node->AddAttribute(attributesMap[L"axisVec"], perm);
            }
        }
        else if (src->OpName() == L"Reshape")
        {
            // TODO: handle CNTK reshape with begin and end axes.
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
            if ((src->Inputs().size() > 0) && (src->Inputs()[0].HasBatchAxis()))
                newShapeVec.push_back(1);
            node->AddAttribute(attributesMap[L"shape"], ToINTS(newShapeVec));
        }
        else if (src->OpName() == L"Splice")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], axisIndex);
        }
        else if (src->OpName() == L"Slice")
        {
            std::vector<int> beginIndex;
            std::vector<int> endIndex;

            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> sliceAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                node->AddAttribute(attributesMap[L"axes"], ToINTS(sliceAxes));

                beginIndex = AsVector<int>(src->Attributes()[L"beginIndexVec"].Value<std::vector<DictionaryValue>>());
                endIndex = AsVector<int>(src->Attributes()[L"endIndexVec"].Value<std::vector<DictionaryValue>>());
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
                bool workaroundONNXRT = false;
                // this code is to workaround a ONNXRT bug that fails
                // to take axes attribute into consideration.
                // we need to convert op attribute to a default ONNX case
                // where axes is not set (or set to ordered indices).
                if (workaroundONNXRT)
                {
                    bool hasBatchAxis = src->Inputs()[0].HasBatchAxis();
                    NDShape inputShape = src->Inputs()[0].Shape();
                    std::vector<int64_t> sliceAxes;
                    int numDims = hasBatchAxis ? (inputShape.Rank() + 1) : inputShape.Rank();
                    for (int onnxAxis = 0; onnxAxis < numDims; onnxAxis++)
                    {
                        sliceAxes.push_back(onnxAxis);
                        if (onnxAxis == 0 && hasBatchAxis)
                        {
                            // batch axis
                            beginIndex.push_back(0);
                            endIndex.push_back(1);
                        }
                        else
                        {
                            if (axisIndex == onnxAxis)
                            {
                                beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                                endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
                            }
                            else
                            {
                                int cntkAxisIndex = numDims - onnxAxis - 1;
                                beginIndex.push_back(0);
                                endIndex.push_back(inputShape[cntkAxisIndex]);
                            }
                        }
                    }
                    node->AddAttribute(attributesMap[L"axes"], sliceAxes);
                }
                else
                {
                    std::vector<int64_t> sliceAxes;
                    sliceAxes.push_back(axisIndex);
                    node->AddAttribute(attributesMap[L"axes"], sliceAxes);

                    beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                    endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
                }
            }

            std::vector<int64_t> beginIndex64 = Cast<int, int64_t>(beginIndex);
            std::vector<int64_t> endIndex64 = Cast<int, int64_t>(endIndex);

            node->AddAttribute(attributesMap[L"beginIndexVec"], beginIndex64);
            node->AddAttribute(attributesMap[L"endIndexVec"], endIndex64);
        }
        if (src->OpName() == L"Pad")
        {
            auto value = (float)src->Attributes()[L"paddingConstantValue"].Value<double>();
            auto mode = (size_t)src->Attributes()[L"paddingMode"].Value<size_t>();
            auto head = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingHead"].Value<std::vector<DictionaryValue>>()));
            auto foot = ToINTS(AsVector<size_t>(src->Attributes()[L"paddingFoot"].Value<std::vector<DictionaryValue>>()));
            if (OpInputsHasBatchAxis(src))
            {
                head.insert(head.begin(), 0);
                foot.insert(foot.begin(), 0);
            }

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
            std::pair<std::vector<int>, std::vector<int>> adjustedDims;
            bool broadcast = false, swapInput = false;
            int axis = 0;
            int index0, index1;
            std::tie<int, int>(index0, index1) = Operators::GetElementWiseInputIndices(src->OpName());
            std::tie<std::pair<std::vector<int>, std::vector<int>>, bool, int>(adjustedDims, broadcast, axis, swapInput) =
                AdjustForBroadcastShape(src->Inputs()[index0], src->Inputs()[index1]);


            if (src->Inputs()[1].IsConstant() && src->Inputs()[1].Shape().Rank() == 0 &&
                src->Inputs()[0].DynamicAxes().size() != 0)
            {
                // TODO: move into AdjustForBroadcastShape
                // a scalar with dynamic access elementwise a constant scalar.   
                broadcast = true;
            }

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
            {
                axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            }
            int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], ax);
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
        else if (src->OpName() == L"Gather")
        {
            if (src->Attributes().Contains(L"axis"))
            {
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);
                node->AddAttribute(attributesMap[L"axis"], ax);
            }
        }
        else if (src->OpName() == L"ImageScaler")
        {
            float scale = (float)(src->Attributes()[L"Scaler"].Value<float>());
            std::vector<float> biases = AsVector<float>(src->Attributes()[L"Biases"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("scale", scale);
            node->AddAttribute("bias", biases);
        }
        else if (src->OpName() == L"MeanVarianceNormalization")
        {
            auto useStatsAcrossChannels = (int64_t)(src->Attributes()[L"useStatsAcrossChannels"].Value<bool>());
            auto doVarianceScaling = (int64_t)(src->Attributes()[L"doVarianceScaling"].Value<bool>());
            // REVIEW: MeanVarianceNormalization attribute 'epsilon' is not exported to ONNX because
            // ONNX MeanVarianceNormalization does not have a corresponding attribute. This should be
            // added if and when the attribute is added to MeanVarianceNormalization node's ONNX spec.
            node->AddAttribute(attributesMap[L"useStatsAcrossChannels"], useStatsAcrossChannels);
            node->AddAttribute(attributesMap[L"doVarianceScaling"], doVarianceScaling);
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
            auto groups = (size_t)src->Attributes()[L"groups"].Value<size_t>();

            //
            // Remove the channel part for ONNX. This is because ONNX, unlike CNTK, does
            // not support padding (pads), dilation, or strides for channel dimension.
            kernelShape = kernelShape.SubShape(0, kernelShape.Rank() - 1);
            strides = strides.SubShape(0, strides.Rank() - 1);
            autoPadding.pop_back();
            dilations = dilations.SubShape(0, dilations.Rank() - 1);

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            node->AddAttribute("dilations", ToINTS(dilations));
            node->AddAttribute("group", (int64_t)groups);

            if (transpose)
            {
                auto outputShape = (NDShape)src->Attributes()[L"outputShape"].Value<NDShape>();
                node->AddAttribute("output_shape", ToINTS(outputShape, src->Inputs()[1].HasBatchAxis()));
            }
            PutAutopadOrPadAttrInNode(node, autoPadding, kernelShape);
        }
        else if (src->OpName() == L"Pooling")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"poolingWindowShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            bool ceilOutDim = (bool)src->Attributes()[L"ceilOutDim"].Value<bool>();
            if (strides.Rank() < kernelShape.Rank())
            {
                // TODO: Try removing this branch. May not be needed after batch dimension fix.
                strides = strides.AppendShape(NDShape(std::vector<size_t>(kernelShape.Rank() - strides.Rank(), 1)));
            }
            if ((strides.Rank() - kernelShape.Rank()) == 1)
            {
                // This can happen, for example, because a CNTK node includes strides for the channel axis as well. 
                strides = strides.SubShape(0, strides.Rank() - 1);
            }
            else if ((strides.Rank() - kernelShape.Rank()) > 1)
            {
                // This means that the length of kernel shape and strides is off by two or more which should not happen.
                LogicError("Node '%S': kernel shape and strides dimensionality does not match.", src->AsString().c_str());
            }
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));
            PutAutopadOrPadAttrInNode(node, autoPadding, kernelShape, ceilOutDim);
        }
        else if (src->OpName() == L"ReduceElements")
        {
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
            const AttributesMapping& attributeMap = Operators::FindAttributeMap(src->OpName(), cntkAttributeOpName);

            auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
            node->AddAttribute(attributeMap.map.at(L"reductionKeepDimensions"), keepReducedDimensions);

            if (src->Attributes().Contains(L"axisVec"))
            {
                std::vector<Axis> reductionAxes;
                reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
                std::vector<int64_t> axes = ConvertAxesToOnnx(reductionAxes, src->Inputs()[0]);
                node->AddAttribute("axes", axes);
            }
            else if (src->Attributes().Contains(L"axis"))
            {
                // py axis -> cpp (-axis -1) -> normalize (rank + axis)
                Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
                int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);

                node->AddAttribute("axis", ax);
            }
        }
    }
}

void CNTKToONNXHelper::PutAutopadOrPadAttrInNode(ONNXIR::Node* node,
    const std::vector<bool>& autoPadding, const NDShape& kernelShape, bool ceilOutDim)
{
    // Based on the CNTK node choose to put either the auto_pad or pads attribute in the ONNX node.

    // ONNX spec says that if 'pads' attributes is specified then 'VALID'
    // for 'auto_pad' is implied, and 'auto_pad' attribute should not (must not)
    // be explicitly specified/set.
    bool isExplicitPadValueNeeded = std::find(autoPadding.begin(), autoPadding.end(), false) != autoPadding.end();
    if (isExplicitPadValueNeeded && !ceilOutDim)
    {
        auto padsValueVectorsForONNX = GetONNXPadsAttributeFromCNTKNode(autoPadding, kernelShape, ceilOutDim);
        auto lowerPads = ToINTS(padsValueVectorsForONNX.first);
        auto upperPads = ToINTS(padsValueVectorsForONNX.second);
        lowerPads.insert(lowerPads.end(), upperPads.cbegin(), upperPads.cend());
        node->AddAttribute("pads", lowerPads);
    }
    else if (ceilOutDim)
        node->AddAttribute("auto_pad", "SAME_LOWER");
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

ONNXIR::Node* FindByName(ONNXIR::Graph* graph, const std::string &name)
{
    for (ONNXIR::Graph::NodeIterator it = graph->Nodes_begin(); it != graph->Nodes_end(); ++it)
    {
        ONNXIR::Node *node = *it;

        const std::vector<ONNXIR::NodeArg>& outputNodeArgs = node->OutputDefs();
        for (int i = 0; i < outputNodeArgs.size(); i++)
        {
            if (outputNodeArgs[i].Name() == name)
            {
                return node;
            }
        }
    }
    return nullptr;
}

ONNXIR::Node* CNTKToONNXHelper::AddNode(const FunctionPtr& src, ONNXIR::Graph* graph, const std::vector<ONNXIR::NodeArg>& inputs, const std::vector<ONNXIR::NodeArg>& outputs)
{
    ONNXIR::Node* node = nullptr;
    std::vector<ONNXIR::NodeArg> orderedInputs = MapInputsOrderToONNX(src, inputs);
    auto nodeName = src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name());

    if (L"Embedding" == src->OpName())
    {
        // WinML does not allow Cast between float and int. To workaround it, set workaroundWinMLNotSupportCastOfInt = true.
        // otherwise use argmax, cast, gather which make more sense for an embedding operation.
        bool workaroundWinMLNotSupportCastOfInt = true;
        if (workaroundWinMLNotSupportCastOfInt)
        {
            ONNXIR::Node* argMatMul = AddMatMulNode(orderedInputs[1], orderedInputs[0], graph);
            int input_size = src->Output().Shape()[0];
            std::vector<int> newShape({ FreeSequenceLen, 1, input_size });
            AddReshapeNode(argMatMul->OutputDefs()[0], newShape, outputs[0].Name(), graph);
        }
        else
        {
            int inputDataAxis = src->Inputs()[1].DynamicAxes().size();
            ONNXIR::Node* argMax = AddArgMaxNode(orderedInputs[1], graph, inputDataAxis);
            ONNXIR::Node* int32Cast = AddCastNode(argMax->OutputDefs()[0], graph, "INT32");

            bool reshapeGather = true;
            if (reshapeGather)
            {
                ONNXIR::NodeArg gatherIndexInputNodeArg(int32Cast->OutputDefs()[0].Name(), nullptr);
                ONNXIR::NodeArg gatherSourceInputNodeArg(orderedInputs[0].Name(), nullptr);
                ONNXIR::NodeArg gatherOutputArg(nodeName + "_gather_tmp", nullptr);
                ONNXIR::Node* gatherNode = graph->AddNode(nodeName + "_tmp", "Gather", "", { gatherSourceInputNodeArg , gatherIndexInputNodeArg }, { gatherOutputArg });

                ONNXIR::NodeArg reshapeInputNodeArg(gatherNode->OutputDefs()[0].Name(), nullptr);
                ONNXIR::Node* reshapedGather = graph->AddNode(nodeName, "Reshape", "", { reshapeInputNodeArg }, outputs);
                int input_size = src->Output().Shape()[0];
                // std::vector<int> newShape({ SequenceLen, 1, input_size });
                std::vector<int> newShape({ FreeSequenceLen, 1, input_size });
                reshapedGather->AddAttribute("shape", ToINTS(newShape, false));
                return reshapedGather;
            }
            else
            {
                ONNXIR::NodeArg gatherIndexInputNodeArg(int32Cast->OutputDefs()[0].Name(), nullptr);
                graph->AddNode(nodeName, "Gather", "", { orderedInputs[0] , gatherIndexInputNodeArg }, outputs);
            }
        }
    }
    else if (Operators::SupportBroadcast(src->OpName()))
    {
        // when converting CNTK to ONNX with broadcasting, the boardcasting input at right-hand-side
        // needs to be reshaped. Reshape is not needed if the broadcasting input is a constant. In such case
        // CreateNode already created a constant with the needed shape. 
        // If the broadcasting input is not a constant, a reshape operation needs to be inserted. 
        // The following code does this reshape insertion.
        const TensorShapeProto* input1Shape = orderedInputs[0].Shape();
        const TensorShapeProto* input2Shape = orderedInputs[1].Shape();
        int input1Rank = input1Shape->dim_size();
        int input2Rank = input2Shape->dim_size();
        ONNXIR::Node* inputNode2 = FindByName(graph, orderedInputs[1].Name());
        if (input2Rank < input1Rank && inputNode2 != nullptr && inputNode2->OpType() != "Constant" && input2Rank != 0)
        {
            // The conditions for inserting a reshape op (the if statement logic above) are:
            // 1. input2Rank < input1Rank : Broadcast is needed. 
            // 2. inputNode2->OpType() != "Constant" : Because if it is Constant we create a 
            //    node for it explicitly in CreateNode() method above.
            // 3. input2Rank != 0 : That is, the second input is not a scalar. If it is then
            //    Reshape is not needed.
            ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape1"), nullptr);
            inputOutput2Arg.SetShape(*input2Shape);

            auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });

            onnx::TypeProto reshapeTypeProto2 = TensorShapeProtoToTypeProto(input2Shape);

            reshapeNode2->AddAttribute("shape", ToINTS(reshapeTypeProto2));

            node = graph->AddNode(nodeName, ToOPName(src), "", { orderedInputs[0] , inputOutput2Arg }, outputs);
        }
        else
        {
            if (src->Inputs()[0].DynamicAxes().size() == 2 && src->Inputs()[1].DynamicAxes().size() == 0 &&
                input1Shape->dim().size() > 2 && input1Shape->dim().size() == input2Shape->dim().size())
            {
                // TODO: apply workaround to MatMul by wrapping it with reshape ops. 
                // This shall be done after code refactoring.
                // in one of this cases (Dense), "Plus" comes after matmul which collaped the first 2 axis (sequence and batch)
                // into one. need to recover it assuming batch size = 1.
                std::vector<int64_t> shape1 = ToINTS(TensorShapeProtoToTypeProto(input1Shape));
                std::vector<int64_t> shape2 = ToINTS(TensorShapeProtoToTypeProto(input2Shape));

                ONNXIR::NodeArg inputOutput2Arg(orderedInputs[1].Name() + string("_reshape2"), nullptr);
                {
                    auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape2"), "Reshape", "", { orderedInputs[1] }, { inputOutput2Arg });
                    // remove batch and sequence dimensions
                    shape2.erase(shape2.begin());
                    shape2.erase(shape2.begin());
                    reshapeNode2->AddAttribute("shape", shape2);
                }

                ONNXIR::NodeArg inputOutput1Arg(orderedInputs[0].Name() + string("_reshape1"), nullptr);
                {
                    auto reshapeNode1 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", { orderedInputs[0] }, { inputOutput1Arg });
                    // (const_cast<TensorShapeProto*>(input1Shape))->mutable_dim(0)->set_dim_value(SequenceLen);
                    (const_cast<TensorShapeProto*>(input1Shape))->mutable_dim(0)->set_dim_value(FreeSequenceLen);

                    onnx::TypeProto reshapeTypeProto1 = TensorShapeProtoToTypeProto(input1Shape);
                    reshapeNode1->AddAttribute("shape", ToINTS(reshapeTypeProto1));
                }

                node = graph->AddNode(nodeName, ToOPName(src), "", { inputOutput1Arg, inputOutput2Arg }, outputs);
                node->AddAttribute("broadcast", (int64_t)1);
            }
            else
                node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
        }
    }
    else
    {
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
        else if (src->OpName() == L"LayerNormalization")
        {
            // Special handling of LayerNormalization to use MeanVarianceNormalization (and not reduce_mean op).
            auto numInputs = src->Inputs().size();
            if (numInputs != 3 && numInputs != 4)
                LogicError("Number of inputs to LayerNormalization is must be either 3 or 4.");

            const size_t operandIndexInCntkInputs = (numInputs == 3) ? 2 : 3; // This changes depending on whether non-zero epsilon was specified.
            const size_t operandIndexInOnnxInputs = 2; // ONNX input indices don't change because we have already filtered epsilon input from ONNX inputs in CreateNode() above.
            const size_t scaleIndexInOnnxInputs = 0;
            const size_t biasIndexInOnnxInputs = 1;

            auto input0 = inputs[operandIndexInOnnxInputs];
            onnx::TypeProto input0ArgType = ToTypeProto(src->Inputs()[operandIndexInCntkInputs].Shape(), src->Inputs()[operandIndexInCntkInputs].HasBatchAxis());
            UpdateONNXType(src->Inputs()[operandIndexInCntkInputs].GetDataType(), input0ArgType);
            ONNXIR::NodeArg mvnTensorOutputArg(nodeName + string("_mvn_output0"), &input0ArgType);
            ONNXIR::Node* mvnNode = graph->AddNode(nodeName + string("_MVN"), "MeanVarianceNormalization",
                "", { input0 }, { mvnTensorOutputArg });
            mvnNode->AddAttribute("across_channels", static_cast<int64_t>(1));
            mvnNode->AddAttribute("normalize_variance", static_cast<int64_t>(1));

            auto input1 = inputs[scaleIndexInOnnxInputs];
            ONNXIR::NodeArg mulTensorOutputArg(nodeName + string("_mul_output0"), &input0ArgType);
            ONNXIR::Node* mulNode = graph->AddNode(nodeName + string("_mul"), "Mul",
                "", { mvnTensorOutputArg, input1 }, { mulTensorOutputArg });
            mulNode->AddAttribute("broadcast", static_cast<int64_t>(1));

            auto input2 = inputs[biasIndexInOnnxInputs];
            ONNXIR::NodeArg addTensorOutputArg(nodeName + string("_Output_0"), &input0ArgType);
            node = graph->AddNode(nodeName + string("_add"), "Add",
                "", { mulTensorOutputArg, input2 }, { addTensorOutputArg });
            node->AddAttribute("broadcast", static_cast<int64_t>(1));
        }
        else
            node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
    }

    //
    // Copy and validate attributes.
    //
    CopyAttributes(src, node);

    return node;
}

std::pair<std::vector<int>, std::vector<int> > CNTKToONNXHelper::GetONNXPadsAttributeFromCNTKNode(
    const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, bool ceilOutDim)
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

void CNTKToONNXHelper::FillTensorWithScalar(const std::vector<NDArrayViewPtr> &srcs,
    onnx::TensorProto& dst, const std::vector<int> dstShape)
{
    auto dataType = srcs[0]->GetDataType();
    SetTensorType(dst, dataType);
    // the first dimension is for srcs count
    int eachSrcSize = std::accumulate(dstShape.begin() + 1, dstShape.end(), 1, std::multiplies<int>());
    switch (dataType)
    {
    case DataType::Float:
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (size_t index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_float_data()->Add()) = scalar;
            }
        }

        break;
    }
    case DataType::Double:
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (size_t index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_double_data()->Add()) = scalar;
            }
        }

        break;
    }
    default:
        NOT_IMPLEMENTED;
    }

    for (auto dim : dstShape)
        *(dst.mutable_dims()->Add()) = dim;
}

ONNXIR::Node* CNTKToONNXHelper::CreateONNXNodesForOptimizedRNNStack(const FunctionPtr &src,
    ONNXIR::Graph* graph,
    std::unordered_map<FunctionPtr, ONNXIR::Node*>& functionNodes,
    std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto numLayers = (size_t)src->Attributes()[L"numLayers"].Value<size_t>();
    auto hiddenSize = (size_t)src->Attributes()[L"hiddenSize"].Value<size_t>();
    auto bidirectional = (bool)(src->Attributes()[L"bidirectional"].Value<bool>());
    auto recurrentOp = (wstring)src->Attributes()[L"recurrentOp"].Value<wstring>();

    if (!Operators::IsOptimizedRnnStackOp(recurrentOp))
        InvalidArgument("Recurrent op used for OptimizedRNNStack is not supported for ONNX export.");

    size_t numDirections = bidirectional ? 2 : 1;
    size_t inputSize = src->Inputs()[0].Shape()[0];
    auto Wcombined = src->Inputs()[1];
    auto WcombinedShape = Wcombined.Shape();

    // Step 1: Read out the OptimzedRNNStack input weight matrix (the big one that combines all weights and biases).
    NDArrayViewPtr srcTensor = Wcombined.IsParameter() ? Parameter(Wcombined).Value() : Constant(Wcombined).Value();
    NDArrayViewPtr srcTemp = srcTensor->DeepClone();
    // Ensure our copy is on the CPU.
    srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
    float* Wdata = srcTemp->WritableDataBuffer<float>();
    Matrix<float> Wm(WcombinedShape[0], WcombinedShape[1], Wdata, CPUDEVICE, MatrixType::DENSE, MatrixFormat::matrixFormatDense);
    
    // Step 2: Extract individual weight and bias matrices for each layer from the big weight matrix.
    std::vector<NDArrayViewPtr> W, R, B;
    std::tie<std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr> >
        (W, R, B) = SplitOptimzedRnnWtoIndivMats(Wm, numLayers, inputSize, hiddenSize, bidirectional, recurrentOp);

    // Step 3: Create ONNX nodes mirroring the implementation of OptimizedRNNStack.
    ONNXIR::Node* functionNode = nullptr;
    bool inputNeedsShapeAdapter(false);
    auto ornnInput = src->Inputs()[0]; // CNTK OptimizedRNNStack node's input operand.

    if (ornnInput.Owner().get() != nullptr)
        CreateNode(ornnInput.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto ornnInputArgType = ToTypeProto(ornnInput.Shape(), ornnInput.HasBatchAxis(), ornnInput.HasSequenceAxis());
    UpdateONNXType(ornnInput.GetDataType(), ornnInputArgType);
    auto ornnOutput = src->Outputs()[0];
    auto outArgType1 = ToTypeProto({ ornnOutput.Shape()[0] / numDirections, numDirections }, ornnOutput.HasBatchAxis(), ornnOutput.HasSequenceAxis());
    TensorShapeProto outArgShape = outArgType1.mutable_tensor_type()->shape();
    int opRank = outArgShape.dim_size();
    std::vector<int> x(opRank);
    std::iota(x.begin(), x.end(), 0);
    std::swap(x[opRank - 2], x[opRank - 3]); // swap (last but one) annd (last but two)
    onnx::TypeProto ornnOutputArgType;
    for (int index = 0; index < opRank; index++)
    {
        if (outArgShape.dim(x[index]).has_dim_param()) // For sequence axis, which is a dynamic axis.
            ornnOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(outArgShape.dim(x[index]).dim_param());
        else
            ornnOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(outArgShape.dim(x[index]).dim_value());
    }
    UpdateONNXType(ornnOutput.GetDataType(), ornnOutputArgType);

    // Note: Keep the ONNX node input name same as the CNTK node as done below.
    std::string ornnInputName = ToString(ornnInput.Uid());
    auto inputItr = compositeOutputsMap.find(ornnInput);
    if (inputItr != compositeOutputsMap.end())
        ornnInputName = ToString(inputItr->second.Uid());

    // Create ONNX LSTM layers
    ONNXIR::NodeArg layerInputOperandArg(ornnInputName, &ornnInputArgType);
    for (size_t i = 0; i < numLayers; ++i)
    {
        std::vector<ONNXIR::NodeArg> inputs;
        std::vector<ONNXIR::NodeArg> outputs;

        // ==== Step 4. Create input nodes =====
        // Input operand X
        if (inputNeedsShapeAdapter)
        {
            std::string adapterBasename = (src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name())) + "_Adapter_" + std::to_string(i);
            ONNXIR::NodeArg shapeAdaptedInputOperandArg = LSTMOutputShapeAdapter(layerInputOperandArg, ornnOutputArgType, graph,
                numDirections, hiddenSize, ornnOutput.GetDataType(), adapterBasename);
            inputs.push_back(shapeAdaptedInputOperandArg);
        }
        else
            inputs.push_back(layerInputOperandArg);

        // Create node for input weight tensor W
        auto WArgName = ToString(Wcombined.Uid()) + "_W_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, W[i], WArgName);
        // Create node for input weight tensor R (equivalent to CNTK's H)
        auto RArgName = ToString(Wcombined.Uid()) + "_R_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, R[i], RArgName);
        // Create node for input bias tensor B
        auto BArgName = ToString(Wcombined.Uid()) + "_B_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, B[i], BArgName);

        // Create other optional input args
        // sequence_lens (optional)
        auto seqLenArgName = ToString(Wcombined.Uid()) + "_seq_len_" + std::to_string(i);
        ONNXIR::NodeArg inputArg_sequence_lens(seqLenArgName, nullptr);
        inputs.push_back(inputArg_sequence_lens);
        // initial_h (optional)
        auto initalHArgName = ToString(Wcombined.Uid()) + "_initial_h_" + std::to_string(i);
        ONNXIR::NodeArg inputArg_initial_h(initalHArgName, nullptr);
        inputs.push_back(inputArg_initial_h);
        // initial_c (optional)
        auto initalCArgName = ToString(Wcombined.Uid()) + "_initial_c_" + std::to_string(i);
        ONNXIR::NodeArg inputArg_initial_c(initalCArgName, nullptr);
        inputs.push_back(inputArg_initial_c);
        // P (peepholes) (optional)
        auto initalPArgName = ToString(Wcombined.Uid()) + "_P_" + std::to_string(i);
        ONNXIR::NodeArg inputArg_P(initalPArgName, nullptr);
        inputs.push_back(inputArg_P);

        // ==== Step 5. Create output nodes =====
        // For now, we always output Y. So this attribute value is 1.
        int64_t outputSequence = 1; 
        //Note: Important to keep the output arg name the same.
        auto outArgName = (i == numLayers-1) ? ToString(ornnOutput.Uid()) : ToString(ornnOutput.Uid()) + "_" + std::to_string(i);
        ONNXIR::NodeArg outputArg_Y(outArgName, &ornnOutputArgType);
        outputs.push_back(outputArg_Y);

        // Dummy output arg Y_h
        auto outputYhArgName = ToString(ornnOutput.Uid()) + "_Y_h_" + std::to_string(i);
        auto outputYhArgType = ToTypeProto(std::vector<int>({ 1, 1, static_cast<int>(hiddenSize) }), false);
        UpdateONNXType(ornnOutput.GetDataType(), outputYhArgType);
        ONNXIR::NodeArg outputArg_Yh(outputYhArgName, &outputYhArgType);
        outputs.push_back(outputArg_Yh);

        // Dummy output arg Y_c
        auto outputYcArgName = ToString(ornnOutput.Uid()) + "_Y_c_" + std::to_string(i);
        auto outputYcArgType = ToTypeProto(std::vector<int>({ 1, 1, static_cast<int>(hiddenSize) }), false);
        UpdateONNXType(ornnOutput.GetDataType(), outputYcArgType);
        ONNXIR::NodeArg outputArg_Yc(outputYcArgName, &outputYcArgType);
        outputs.push_back(outputArg_Yc);

        // ==== Step 6. Add ONNX LSTM node ====
        auto rnnOpNameLookup = Operators::OptimizedRnnToOnnxOpLookup();
        auto rnnNodeName = (src->Name().empty() ? ToString(src->Uid()) : ToString(src->Name())) + std::to_string(i);
        functionNode = graph->AddNode(rnnNodeName, rnnOpNameLookup[recurrentOp], "", inputs, outputs);

        std::vector<std::string> singleDirectionActivation;
        if (recurrentOp == L"lstm")
            singleDirectionActivation = { "Sigmoid", "Tanh", "Tanh" };
        else if (recurrentOp == L"rnnReLU")
            singleDirectionActivation = { "Relu" };
        else if (recurrentOp == L"rnnTanh")
            singleDirectionActivation = { "Tanh" };
        std::vector<std::string> activations;
        activations.insert(activations.end(), singleDirectionActivation.begin(), singleDirectionActivation.end());
        if (bidirectional)
            activations.insert(activations.end(), singleDirectionActivation.begin(), singleDirectionActivation.end());
        functionNode->AddAttribute("activations", activations);
        functionNode->AddAttribute("direction", bidirectional ? "bidirectional" : "forward");
        functionNode->AddAttribute("hidden_size", (int64_t)hiddenSize);
        functionNode->AddAttribute("output_sequence", outputSequence);

        layerInputOperandArg = outputArg_Y; // Output of this layer is the input to the next layer in the loop.
        inputNeedsShapeAdapter = true; // To enable shape adapter to allow stacking for next layer. 
    }

    functionNodes.emplace(src, functionNode);
    return functionNode;
}

std::tuple<std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr> >
CNTKToONNXHelper::SplitOptimzedRnnWtoIndivMats(Matrix<float>& WbigIn,
    size_t numLayers, size_t inputSize, size_t hiddenSize, bool bidirectional, wstring recurrentOp)
{
    size_t numDirections = bidirectional ? 2 : 1;
    size_t numGates;
    if (recurrentOp == L"lstm")
        numGates = 4;
    else if (recurrentOp == L"rnnReLU" || recurrentOp == L"rnnTanh")
        numGates = 1;
    else
        InvalidArgument("Unsupported recurrent op value.");

    std::vector<Matrix<float> >  W;
    std::vector<Matrix<float> >  R;
    std::vector<Matrix<float> >  B;

    // The next two operations will make a deep copy and flatten the matrix
    // in the same order as the Python matrix Wbig (row-major).
    Matrix<float> Wbig = WbigIn.Transpose(); // Deep copy.
    Wbig.Reshape(1, WbigIn.GetNumElements()); 

    // Step 2: Extracting the weights W and R from big weight matrix (including backward ones in case of bidirectional op).
    size_t offset(0);
    size_t layerInputSize(inputSize);
    for (size_t i = 0; i < numLayers; ++i)
    {
        Matrix<float> fW = GetWeightMatFromOrnnBigW(Wbig, offset, layerInputSize, hiddenSize, numGates, recurrentOp);
        offset += layerInputSize * hiddenSize * numGates;
        W.push_back(Matrix<float>(fW, CPUDEVICE));
        Matrix<float> fR = GetWeightMatFromOrnnBigW(Wbig, offset, hiddenSize, hiddenSize, numGates, recurrentOp);
        offset += hiddenSize * hiddenSize * numGates;
        R.push_back(Matrix<float>(fR, CPUDEVICE));

        if (bidirectional)
        {
            Matrix<float> bW = GetWeightMatFromOrnnBigW(Wbig, offset, layerInputSize, hiddenSize, numGates, recurrentOp);
            offset += layerInputSize * hiddenSize * numGates;
            W.push_back(Matrix<float>(bW, CPUDEVICE));
            Matrix<float> bR = GetWeightMatFromOrnnBigW(Wbig, offset, hiddenSize, hiddenSize, numGates, recurrentOp);
            offset += hiddenSize * hiddenSize * numGates;
            R.push_back(Matrix<float>(bR, CPUDEVICE));
        }

        layerInputSize = hiddenSize * numDirections;
    }

    // Step 3: Extracting the biases B from big weight matrix (including backward ones in case of bidirectional op).
    // NOTE: that 'offset' should be set correctly based on the extraction of weight matrices W and R as in Step 2.
    // In Step 3 we cannot start with offset = 0 for biases, since they start from somewhere in the middle of the
    // big weight matrix.
    for (size_t i = 0; i < numLayers; ++i)
    {
        Matrix<float> fB = GetBiasMatFromOrnnBigW(Wbig, offset, hiddenSize, numGates, recurrentOp);
        offset += numBiasInOnnxLstm * hiddenSize * numGates;
        B.push_back(Matrix<float>(fB, CPUDEVICE));
        if (bidirectional)
        {
            Matrix<float> bB = GetBiasMatFromOrnnBigW(Wbig, offset, hiddenSize, numGates, recurrentOp);
            offset += numBiasInOnnxLstm * hiddenSize * numGates;
            B.push_back(Matrix<float>(bB, CPUDEVICE));
        }
    }

    // Step 4: Convert weight matrices into NDArrayView;
    std::vector<NDArrayViewPtr> Wonnx = ToRnnWeightPerLayerOnnxFormat(W, numLayers, numDirections, numGates, hiddenSize, inputSize, true);
    std::vector<NDArrayViewPtr> Ronnx = ToRnnWeightPerLayerOnnxFormat(R, numLayers, numDirections, numGates, hiddenSize, hiddenSize, false);

    // Step 5: Convert bias matrices into NDArrayView;
    std::vector<NDArrayViewPtr> Bonnx = ToRnnBiasPerLayerOnnxFormat(B, numLayers, numDirections, hiddenSize, numGates);

    return std::make_tuple(std::move(Wonnx), std::move(Ronnx), std::move(Bonnx));
}

Matrix<float> CNTKToONNXHelper::GetWeightMatFromOrnnBigW(Matrix<float>& Wbig, size_t offset,
    size_t layerInputSize, size_t layerOutputSize, size_t numGates, wstring recurrentOp)
{
    Matrix<float> W0(CPUDEVICE);
    W0.SetValue(Wbig.ColumnSlice(offset, layerInputSize*layerOutputSize*numGates));
    W0.Reshape(layerInputSize, layerOutputSize*numGates);
    if (recurrentOp == L"lstm") // rnnReLU and rnnTanh have one gate so reordering is moot.
        InplaceAdjustGateOrder(W0, layerOutputSize);
    return W0;
}

Matrix<float> CNTKToONNXHelper::GetBiasMatFromOrnnBigW(Matrix<float>& Wbig, size_t offset,
    size_t hiddenSize, size_t numGates, wstring recurrentOp)
{
    Matrix<float> b(1, numBiasInOnnxLstm * hiddenSize*numGates, CPUDEVICE);

    Matrix<float> b1(CPUDEVICE), b2(CPUDEVICE);
    b1.SetValue(Wbig.ColumnSlice(offset, hiddenSize*numGates));
    auto nextoffset = offset + hiddenSize * numGates; // Note that 'offset' still must be updated outside.
    b2.SetValue(Wbig.ColumnSlice(nextoffset, hiddenSize*numGates));
    // Creating bias vector b as [W_b, R_b]. Creating these values as done in
    // optimized_rnnstack_converter.py. W_bias is b1 + b2. R_bias is just zeros.
    b1.AssignSumOf(b1, b2);
    if (recurrentOp == L"lstm") // rnnReLU and rnnTanh have only one gates so reordering is moot.
        InplaceAdjustGateOrder(b1, hiddenSize);
    b.SetColumnSlice(b1, 0, hiddenSize * numGates);
    b.SetColumnSlice(Matrix<float>::Zeros(1, hiddenSize * numGates, CPUDEVICE), hiddenSize * numGates, hiddenSize * numGates);
    return b;
}

void CNTKToONNXHelper::InplaceAdjustGateOrder(Matrix<float>& W, size_t hiddenSize)
{
    // REVIEW sptiwari: Written just for LSTM. Assumes numGates = 4. GRU to be included later.

    size_t offset(0);

    Matrix<float> Wi(CPUDEVICE), Wf(CPUDEVICE), Wc(CPUDEVICE), Wo(CPUDEVICE);
    Wi.SetValue(W.ColumnSlice(offset, hiddenSize));
    offset += hiddenSize;
    Wf.SetValue(W.ColumnSlice(offset, hiddenSize));
    offset += hiddenSize;
    Wc.SetValue(W.ColumnSlice(offset, hiddenSize));
    offset += hiddenSize;
    Wo.SetValue(W.ColumnSlice(offset, hiddenSize));

    offset = 0;
    W.SetColumnSlice(Wi, offset, hiddenSize);
    offset += hiddenSize;
    W.SetColumnSlice(Wo, offset, hiddenSize);
    offset += hiddenSize;
    W.SetColumnSlice(Wf, offset, hiddenSize);
    offset += hiddenSize;
    W.SetColumnSlice(Wc, offset, hiddenSize);
}

std::vector<NDArrayViewPtr> CNTKToONNXHelper::ToRnnWeightPerLayerOnnxFormat(std::vector<Matrix<float> >& W, size_t numLayers,
    size_t numDirections, size_t numGates, size_t hiddenSize, size_t inputSize, bool updateInputSizeWithEachLayer)
{
    std::vector<NDArrayViewPtr> Wonnx;
    // First layer input size is inputSize. Other layers' input size is numDirections*hiddenSize.
    size_t layerInputSize(inputSize);
    for (size_t i = 0; i < numLayers; ++i)
    {
        // Here we create a currLayerWeightMatrix which has 3D tensor in a 2D matrix data buffer.
        // The format is [Plane 1 Plane2]. This is the buffer format needed to pass in to NDArrayView
        // for creating a 3D tensor as needed by ONNX LSTM.
        Matrix<float> currLayerWeightMatrix(hiddenSize * numGates, layerInputSize * numDirections, CPUDEVICE);
        size_t offset = 0;
        for (size_t j = 0; j < numDirections; ++j)
        {
            Matrix<float> temp(W[i*numDirections + j].GetNumCols(), W[i*numDirections + j].GetNumRows(), W[i*numDirections + j].Data(), CPUDEVICE); 
            currLayerWeightMatrix.SetColumnSlice(temp, offset, layerInputSize);
            offset += layerInputSize;
        }
        NDArrayView currLayerWeightNDArray(::CNTK::DataType::Float, NDShape({ numDirections, hiddenSize*numGates, layerInputSize }),
            (void*)currLayerWeightMatrix.Data(), currLayerWeightMatrix.BufferSize(), DeviceDescriptor::CPUDevice());
        Wonnx.push_back(currLayerWeightNDArray.DeepClone(DeviceDescriptor::CPUDevice()));

        if (updateInputSizeWithEachLayer)
        {
            // Except for first layer (so starting from second layer), the layer input sizes are 
            // hiddenSize for one directional, and 2*hiddenSize for bidirectional. This is needed
            // for W matrix which is based on inputSize, but not the H (sometimes called R) matrix, 
            // which is just based on hiddenSize which does not change.
            layerInputSize = hiddenSize * numDirections;
        }
    }
    return Wonnx;
}

std::vector<NDArrayViewPtr> CNTKToONNXHelper::ToRnnBiasPerLayerOnnxFormat(std::vector<Matrix<float> >& B, size_t numLayers,
    size_t numDirections, size_t hiddenSize, size_t numGates)
{
    std::vector<NDArrayViewPtr> Bonnx;
    for (size_t i = 0; i < numLayers; ++i)
    {
        // Here we create a currLayerWeightMatrix which has 3D tensor in a 2D matrix data buffer.
        // The format is [Plane 1 Plane2]. This is the buffer format needed to pass in to NDArrayView
        // for creating a 3D tensor as needed by ONNX LSTM.
        Matrix<float> currLayerBiasMatrix(2 * hiddenSize * numGates, numDirections, CPUDEVICE);
        size_t offset = 0;
        for (size_t j = 0; j < numDirections; ++j)
        {
            Matrix<float> temp(B[i*numDirections + j].GetNumCols(), B[i*numDirections + j].GetNumRows(), B[i*numDirections + j].Data(), CPUDEVICE);
            currLayerBiasMatrix.SetColumnSlice(temp, offset, 1);
            ++offset;
        }
        NDArrayView currLayerBiasNDArray(::CNTK::DataType::Float, NDShape({ numDirections, 2 * hiddenSize*numGates }),
            (void*)currLayerBiasMatrix.Data(), currLayerBiasMatrix.BufferSize(), DeviceDescriptor::CPUDevice());
        Bonnx.push_back(currLayerBiasNDArray.DeepClone(DeviceDescriptor::CPUDevice()));
    }
    return Bonnx;
}

void CNTKToONNXHelper::CreateRecurrentWeightONNXNodes(ONNXIR::Graph* graph, std::unordered_map<Variable, ONNXIR::Node*>& variableNodes,
    const Variable& Wcombined, std::vector<ONNXIR::NodeArg>& inputs, NDArrayViewPtr W, string WArgName)
{
    auto WArgType = ToTypeProto(W->Shape(), false, false, false); // Last arg is false because we don't want shape reversal here.
    UpdateONNXType(Wcombined.GetDataType(), WArgType);
    ONNXIR::NodeArg WArg(WArgName, &WArgType);
    inputs.push_back(WArg);

    std::vector<ONNXIR::NodeArg> varInputs;
    std::vector<ONNXIR::NodeArg> varOutputs;

    varOutputs.push_back({ WArg });
    ONNXIR::Node* variableNode = graph->AddNode(WArgName, "Constant", "", varInputs, varOutputs);
    onnx::TensorProto dstTensor;
    CopyTensor(W, dstTensor, &WArgType);
    variableNode->AddAttribute("value", dstTensor);
    variableNodes.emplace(Wcombined, variableNode);
}

ONNXIR::NodeArg CNTKToONNXHelper::LSTMOutputShapeAdapter(ONNXIR::NodeArg& inputArg, onnx::TypeProto& inputArgType, ONNXIR::Graph* graph,
    size_t numDirections, size_t hiddenSize, CNTK::DataType outputType, string adapterBasename)
{
    // This adapter changes input format (this is output of previous layer) from
    // [S, numDirections, B, hiddenSize] --> Output format (input to new LSTM layer) [S, B, numDirections*hiddenSize]

    // Transpose 2nd and 3rd axes, i.e. [S, numDirections, B, hiddenSize] --> [S, B, numDirections, hiddenSize]
    TensorShapeProto inputShape = inputArgType.mutable_tensor_type()->shape();
    onnx::TypeProto transposeOutputArgType;
    int inputRank = inputShape.dim_size();
    std::vector<int64_t> x(inputRank);
    std::iota(x.begin(), x.end(), 0);
    std::swap(x[inputRank - 2], x[inputRank - 3]); // swap (last but one) and (last but two)
    for (int index = 0; index < inputRank; index++)
    {
        if (inputShape.dim(static_cast<int>(x[index])).has_dim_param()) // For sequence axis, which is a dynamic axis.
            transposeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(inputShape.dim(static_cast<int>(x[index])).dim_param());
        else
            transposeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape.dim(static_cast<int>(x[index])).dim_value());
    }
    UpdateONNXType(outputType, transposeOutputArgType);
    ONNXIR::NodeArg transposeOutputArg(adapterBasename + "_Transpose_Output", &transposeOutputArgType);
    auto transposeNode = graph->AddNode(adapterBasename + "_Transpose", "Transpose", "", { inputArg }, { transposeOutputArg });
    transposeNode->AddAttribute("perm", x);

    // Reshape to combine last two axes, i.e. [S, B, numDirections, hiddenSize] --> [S, B, numDirections*hiddenSize]
    TensorShapeProto lastShape = transposeOutputArgType.mutable_tensor_type()->shape();
    int lastShapeRank = lastShape.dim_size();
    if (lastShapeRank != 4)
        LogicError("Rank of the LSTM output from previous layer must be 4.");
    if (lastShape.dim(2).has_dim_param() || lastShape.dim(3).has_dim_param())
        LogicError("Sequence axis cannot be amongst the last two axis. It must be the first one.");
    onnx::TypeProto reshapeOutputArgType;
    if (lastShape.dim(0).has_dim_param())
        reshapeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(lastShape.dim(0).dim_param());
    else
        reshapeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(lastShape.dim(0).dim_value());
    if (lastShape.dim(1).has_dim_param())
        reshapeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(lastShape.dim(1).dim_param());
    else
        reshapeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(lastShape.dim(1).dim_value());
    reshapeOutputArgType.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(lastShape.dim(2).dim_value()*lastShape.dim(3).dim_value());
    UpdateONNXType(outputType, reshapeOutputArgType);
    ONNXIR::NodeArg reshapeOutputArg(adapterBasename + "_Reshape_Output", &reshapeOutputArgType);
    auto reshapeNode = graph->AddNode(adapterBasename + "_Reshape", "Reshape", "", { transposeOutputArg }, { reshapeOutputArg });
    std::vector<int64_t> shape({ 0, 0, -1 });
    reshapeNode->AddAttribute("shape", shape);

    return reshapeOutputArg;
}
