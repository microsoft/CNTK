//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "core/graph/model.h"
#include "core/graph/graph.h"
#include "core/common/status.h"
#include "core/graph/schema_registry.h"

#include "CNTKToONNX.h"
#include "Utils.h"
#include "Operators.h"
#include "PrimitiveFunctionAttribute.h"
#include "BlockFunction.h"
#include <vector>
#include <tuple>
#include <numeric>
#include <iostream>
#include "RNNHelper.h"
#include "Matrix.h"
#include "ConvolveGeometry.h"

using namespace Microsoft::MSR::CNTK;
using namespace CNTK::ONNX;
using namespace CNTK;
using namespace onnxruntime;
using namespace onnx;

namespace CNTK
{
class CNTKToONNXHelper
{
public:
    //
    // Copy the entire CNTK graph to ONNX graph.
    //
    static void Copy(const FunctionPtr& src, onnxruntime::Graph* dst);

private:
    //
    // CNTK uses Combine op to aggregate multiple outputs of a model.
    // onnxruntime creates outputs by collecting dangling output NodeArgs in a graph (cf. BuildConnections). 
    // Often an aggregated CNTK output node is also an input to another node in the graph.
    // This type of nodes are not dangling and thus are not treated as ONNX outputs.
    // To solve this issue, we search for all such nodes and append to it a NoOp
    // to make it dangling at the end.
    //
    static void HandleRootCombineOp(const FunctionPtr& src, onnxruntime::Graph* dst);

    //
    // Recursively create ONNX nodes corresponding to each CNTK node.
    //
    static onnxruntime::Node* CreateNode(const FunctionPtr& src,
        onnxruntime::Graph* graph,
                                    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                    const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    
    static bool CheckCorrectTransposeAxisToSkipForSequenceAxisOpWrapper(FunctionPtr currentOp);
    static bool MatchOpSequence(const FunctionPtr src, std::vector<wstring> opSequence, FunctionPtr &op);
    static bool MatchInputSequence(const Variable &input, std::vector<wstring> opSequence, Variable &inputSkipTo);
    static Variable SkipBatchAndSequenceAxisInput(const Variable input);
    static FunctionPtr SkipBatchAndSequenceAxisOp(const FunctionPtr src);

    static onnxruntime::Node* CreateSoftmaxLikeNode(const FunctionPtr& src,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);
    static onnxruntime::Node* CreateSequenceSliceNode(const FunctionPtr& src,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    // Create an ONNX NodeArg of desired shape with constant 0s as initial values. 
    // The NodeArg is used to expand inputs of a CNTK splice op to a desired shape via broadcast.
    static onnxruntime::NodeArg &AddZerosConstantNodeArg(Graph *graph, const string &nodeArgName,
        const std::vector<int64_t> &shape, CNTK::DataType dataType);

    static onnxruntime::Node *AddReshapeNodeImpl(Graph *graph, const string &nodeName, NodeArg *input, NodeArg *output, const std::vector<int64_t>& newShape);

    static NodeArg* GetInputAdjustmentForBroadcast(onnxruntime::Graph* graph, const FunctionPtr src, const Variable &input, int inputIndex, onnx::TypeProto &inputArgType);

    // Processes inputs of a src CNTK op, creating ONNX nodes needed for the inputs.
    static void ProcessInputs(const FunctionPtr& src,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap,
        std::vector<onnxruntime::NodeArg *>& inputs);
    static void ProcessInputsForBatchAxisOp(const FunctionPtr& rootNode,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap,
        std::vector<onnxruntime::NodeArg *>& inputs);

    // Processes outputs of a src CNTK op.
    static void ProcessOutputs(const FunctionPtr& src,
        std::vector<onnxruntime::NodeArg *>& outputs, Graph *graph);
    static void ProcessOutputsForBatchAxisOp(const FunctionPtr& rootNode,
        std::vector<onnxruntime::NodeArg *>& outputs, Graph *graph);

    static onnxruntime::Node *AddReshapeNode(onnxruntime::NodeArg &nodeArg, const std::vector<int64_t> &newShape, const std::string &outArgName,
        onnxruntime::Graph* graph);
    static onnxruntime::Node *AddMatMulNode(onnxruntime::NodeArg &nodeArg1, onnxruntime::NodeArg &nodeArg2, onnxruntime::Graph* graph,
        const std::string &out_arg_name);
    static onnxruntime::Node *AddAddNode(onnxruntime::NodeArg &nodeArg1, onnxruntime::NodeArg &nodeArg2, onnxruntime::Graph* graph,
        const std::string &out_arg_name);
    static onnxruntime::Node *AddIdentityOp(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, const std::string &out_arg_name);
    static onnxruntime::Node *AddArgMaxNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, int axis);
    static onnxruntime::Node *AddCastNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, onnx::TensorProto_DataType toType);
    static onnxruntime::Node *AddTransposeNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, const std::vector<int64_t> &perm,
        onnx::TypeProto& transposeOutputArgType, const std::string &outputNodeArgName);

    static std::vector<int64_t> BroadcastInputs(std::vector<onnxruntime::NodeArg *> &orderedInputs, const std::set<int64_t>& ignoreAxes,
        const FunctionPtr& src, onnxruntime::Graph* graph);

    //
    //  Insert a reshape node in front of a given node and its output node arg
    //
    static onnxruntime::Node *InsertReshapeNodeToCNTKFunction(const FunctionPtr &src, onnxruntime::Node* node, const std::vector<int64_t> &shape, onnxruntime::Graph* graph,
        const std::string &nodeOutputName);

    //
    //  methods to create a RNN/LSTM/GRU node.
    //
    static onnxruntime::Node* CreateLSTMNode(const FunctionPtr& src,
        onnxruntime::Graph* graph,
                                        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                        const std::unordered_map<Variable, Variable>& compositeOutputsMap);
    static onnxruntime::Node *CreateGRUNode(const FunctionPtr &src,
        onnxruntime::Graph* graph,
                                       std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                       std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                       const std::unordered_map<Variable, Variable>& compositeOutputsMap);
    static onnxruntime::Node *CreateRNNNode(const FunctionPtr &src,
        onnxruntime::Graph* graph,
                                       std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                       std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                       const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    static void PrepareRNNInput(const Variable &X, Graph *graph, std::vector<onnxruntime::NodeArg*> &nodeInputs);
    static void PrepareLSTMInitialStateNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                            const std::vector<Variable> &initialVariables, int batchSize, int cellSize,
                                            const std::string &uid, std::vector<onnxruntime::NodeArg *> &nodeInputs);

    static void PrepareRNNWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                     const std::vector<Variable> &Ws, std::vector<onnxruntime::NodeArg *> &nodeInputs,
                                     std::function<void(const std::vector<NDArrayViewPtr> &srcTensors,
                                                        onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)>
                                         weightConverter);
    static void PrepareGRUZRHWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node *>& variableNodes,
                                        const std::vector<Variable> &Rs, const std::vector<Variable> &Rh1s, std::vector<onnxruntime::NodeArg *> &nodeInputs);
    static void PrepareGRUBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                   const std::vector<Variable> &Bs, std::vector<onnxruntime::NodeArg *> &nodeInputs);

    static void PrepareRNNBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                   const std::vector<Variable> &Bs, std::vector<onnxruntime::NodeArg*> &nodeInputs);

    static void PrepareLSTMWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                      const std::vector<Variable> &Ws, double *stabilizerConstants, std::vector<onnxruntime::NodeArg *> &nodeInputs);
    static void PrepareLSTMBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                    const std::vector<Variable> &Ws, std::vector<onnxruntime::NodeArg *> &nodeInputs);
    static void PrepareLSTMPeepholeNode(onnxruntime::Graph* graph,
                                        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes, const std::vector<Variable> &Ps,
                                        const std::vector<double> &stabilizerDcCoefs, const std::vector<double> &stabilizerCCoefs,
                                        std::vector<onnxruntime::NodeArg *> &nodeInputs);
    //
    // Traverse the entire graph and collect variable mapping between graph inside and outside the block.
    //
    static void TraverseGraph(const FunctionPtr& src,
                              std::set<FunctionPtr>& visited,
                              std::unordered_map<Variable, Variable>& compositeOutputsMap);

    static void SetTensorType(onnx::TensorProto& dst, CNTK::DataType dataType);

    //
    // Copy the content of NDArrayView to TensorProto, and do the needed
    // convergence.
    //
    static void CopyTensor(const NDArrayViewPtr src, onnx::TensorProto& dst, onnx::TypeProto *inputArgType = nullptr);

    static void CopyTensorsWithMultipliers(const std::vector<NDArrayViewPtr> &srcTensors, const std::vector<double> &multipliers,
                                           onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

    static void CopyRNNBiasTensors(const std::vector<NDArrayViewPtr>& srcTensors,
                                   onnx::TensorProto& dst, const onnx::TypeProto& inputArgType);

    static void CopyGRUWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
                                     onnx::TensorProto &dst, const onnx::TypeProto &inputArgType);

    static void CopyGRUStateWeightTensors(
        const std::vector<NDArrayViewPtr> &srcZRTensors, const std::vector<NDArrayViewPtr> &srcHTensors,
        onnx::TensorProto& dst, const onnx::TypeProto &inputArgType);

    static void CopyRNNWeightTensors(const std::vector<NDArrayViewPtr> &srcTensors,
                                     onnx::TensorProto &dst, const onnx::TypeProto &inputArgType);

    static void FillTensorWithScalar(const std::vector<NDArrayViewPtr> &src, onnx::TensorProto &dst, const std::vector<int64_t> dstShape);

    static onnxruntime::NodeArg& CreateScalarNode(Graph *graph, const string &nodeArgName, CNTK::DataType dataType, double value);

    //
    // Create an ONNX weight tensor for LSTM op. It handles memory mapping from CNTK to ONNX.
    //
    static void CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(const std::vector<NDArrayViewPtr> &src, double* stabilizerConstants,
                                                                    onnx::TensorProto &dst, const onnx::TypeProto &inputArgType);

    static void CopyShapeTypeProtoToTensorProto(const onnx::TypeProto &inputArgType, onnx::TensorProto &dst);

    //
    // Copy supported attributes from CNTK node to corresponding ONNX node.
    //
    static void CopyAttributes(const FunctionPtr &src, onnxruntime::Node* node);

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
    static onnx::TypeProto ToTypeProto(const std::vector<int64_t>& shape, bool doReverseVec = true);
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
    // Converts axis (in CNTK C++ API sense) to index in ONNX sense assuming op may do broadcast
    // across multiple inputs. In such case, it shall take the highest axis.  
    //
    static int64_t ConvertAxisToOnnxBroadcastOfOp(const Axis &axis, const FunctionPtr &src);
    
    //
    // Converts axis (in CNTK C++ API sense) to index in ONNX sense
    //
    static int64_t ConvertAxisToOnnx(const Axis &axis, const Variable &operand);

    //
    // Converts axes (in CNTK C++ API sense) to index in ONNX sense
    //
    static std::vector<int64_t> ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand);

    //
    // Argument orders between CNTK and ONNX aren't always the same.
    //
    static std::vector<onnxruntime::NodeArg* > MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<onnxruntime::NodeArg* >& inputs);

    //
    // Add current CNTK node to ONNX graph.
    //
    static onnxruntime::Node* AddNode(const FunctionPtr& src, onnxruntime::Graph* graph, const std::vector<onnxruntime::NodeArg*>& inputs, const std::vector<onnxruntime::NodeArg* >& outputs);

    //
    // set node attribute for ReduceElements ops
    // 
    static void SetReduceElementsAttributes(const FunctionPtr src, Node *node);

    //
    // Check if CNTK node's pad attribute value is correct with regard to ceilOutDim.
    //
    static void ValidatePadValueForCeilOutDim(const std::vector<int64_t> lowerPad, const std::vector<int64_t> upperPad, const std::vector<bool>& autoPadding,
        const NDShape& kernelShape, const NDShape& inputShape, const NDShape& strides, const NDShape& dilation, bool transpose = false);
    //
    // Get ONNX 'pads' attribute value based on CNTK node's autoPadding attribute value.
    //
    static std::pair<std::vector<int>, std::vector<int> > GetONNXPadsAttributeFromCNTKNode(
        const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, const NDShape& inputShape, 
        const NDShape& strides, const NDShape& dilation, bool ceilOutDim, bool transpose);

    //
    // Adds attributes 'pads' to saved node (typically convolution or pooling).
    //
    static void PutPadAttrInNode(onnxruntime::Node* node, const std::vector<bool>& autoPadding,
        const NDShape& kernelShape, const NDShape& inputShape, const NDShape& strides, const NDShape& dilation, bool ceilOutDim = false, bool transpose = false);

    //
    // Takes CNTK's StraightThrough estimator node and converts it into a series of Greater+Cast+Mul+Sub
    // nodes on the ONNX side.
    //
    static onnxruntime::Node* CreateONNXNodesForStraightThrough(const FunctionPtr &src,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);
    //
    // Takes CNTK's OptimizedRNNStack node and converts it into a series of RNN/LSTM/GRU nodes
    // on the ONNX side.
    //
    static onnxruntime::Node* CreateONNXNodesForOptimizedRNNStack(const FunctionPtr &src,
        onnxruntime::Graph* graph,
                                                             std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                                             std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
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
    static std::vector<NDArrayViewPtr> ToRnnWeightPerLayerOnnxFormat(std::vector<Matrix<float>>& W, size_t numLayers,
                                                                     size_t numDirections, size_t numGates, size_t hiddenSize, size_t inputSize, bool updateInputSizeWithEachLayer);

    //
    // Takes a vector of Matrix<ElemType> which are biases for each layer and each direction
    // and converts them to a vector of NDArrays, one for each layer, in ONNX LSTM format.
    //
    static std::vector<NDArrayViewPtr> ToRnnBiasPerLayerOnnxFormat(std::vector<Matrix<float>>& W, size_t numLayers,
                                                                   size_t numDirections, size_t hiddenSize, size_t numGates);

    //
    // Create a ONNX node for input weight for a recurrence node.
    //
    static void CreateRecurrentWeightONNXNodes(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                               const Variable& Wcombined, std::vector<onnxruntime::NodeArg*>& inputs, NDArrayViewPtr W, string WArgName = "");

    //
    // Method to insert reshape and transpose nodes to the output of the ONNX LSTM output
    // so that it can be fed in as input to the next ONNX LSTM node.
    //
    static onnxruntime::NodeArg* LSTMOutputShapeAdapter(onnxruntime::NodeArg& inputArg, onnx::TypeProto& inputArgType, onnxruntime::Graph* graph,
                                                   size_t numDirections, size_t hiddenSize, CNTK::DataType outputType, string adapterBasename = "");

    // Takes CNTK's Select node and converts it into a series of ONNX nodes.
    static onnxruntime::Node * CreateONNXNodesForSelect(const FunctionPtr & src,
        onnxruntime::Graph * graph, std::unordered_map<FunctionPtr,
        onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    // Takes CNTK's TimesTranspose node and converts it into a series of ONNX nodes.
    static onnxruntime::Node * CreateONNXNodesForTimesTranspose(const FunctionPtr & src,
        onnxruntime::Graph * graph, std::unordered_map<FunctionPtr,
        onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    //
    // Method to create ONNX nodes that have an explicit batch axis from their CNTK
    // counterparts.
    //
    static onnxruntime::Node* CreateNodeForBatchAxisOp(const FunctionPtr &src,
        onnxruntime::Graph* graph,
        std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
        std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
        const std::unordered_map<Variable, Variable>& compositeOutputsMap);

    // A helper function, to reverse any iterable container and return a copy
    // of the reversed container.
    //
    template<typename ItrType>
    static ItrType reverse(ItrType v)
    {
        std::reverse(std::begin(v), std::end(v));
        return v;
    }

    template <class T, class V>
    static inline std::vector<V> Cast(const std::vector<T>& v)
    {
        std::vector<V> result;
        result.reserve(v.size());
        for (auto d : v)
            result.push_back((V) d);
        return result;
    }

    static onnx::TypeProto MakeTypeProtoWithShape()
    {
        onnx::TypeProto typeProtoWithShape;
        // this is to ensure a scalar has a tensor shape of zero dimenstion.
        typeProtoWithShape.mutable_tensor_type()->mutable_shape();
        return typeProtoWithShape;
    }

    static onnx::TypeProto TensorShapeProtoToTypeProto(const onnx::TensorShapeProto* inputShape)
    {
        onnx::TypeProto newShape = MakeTypeProtoWithShape();

        int inputRank = inputShape->dim_size();
        for (int index = 0; index < inputRank; index++)
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

        return newShape;
    }

    //
    // Helper function to reduce the rank of a shape.
    //
    static onnx::TypeProto ReduceRank(const onnx::TensorShapeProto* inputShape, int reductionRank, bool rightReduction, bool hasBatchAxis)
    {
        assert(inputShape != nullptr);

        int inputRank = inputShape->dim_size();
        assert(inputRank > reductionRank);

        onnx::TypeProto newShape = MakeTypeProtoWithShape();

        int64_t reduceDim = 1;

        if (rightReduction)
        {
            // This code detects and skips batch axis automatically. 
            // If we support outputRank > 1 in the future, we can deduct if we have emulated batch axis based on reductionRank and outputRank.
            for (int index = 0; index < (inputRank - reductionRank); index++)
                newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());

            for (int index = (inputRank - reductionRank); index < inputRank; index++)
                reduceDim *= inputShape->dim(index).dim_value();

            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);
        }
        else
        {
            // Update: if there is batch axis, we skip the first axis, since axis (1, ) is emulated batch axis.
            // There is no way to deduct if there is batch axis from here.
            // a valid shape could be [1, reductionRank, postfix] where postfix can have arbitrary rank. 
            // Currently however we are assuming postfix as 1. This part should also be updated when we support outputRank > 1.
            size_t startIdx = hasBatchAxis ? 1 : 0;
            if (hasBatchAxis)
                newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(0).dim_value());

            for (int index = startIdx; index < reductionRank + startIdx; index++)
                reduceDim *= inputShape->dim(index).dim_value();

            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(reduceDim);

            for (int index = reductionRank + startIdx; index < inputRank; index++)
                newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(inputShape->dim(index).dim_value());
        }

        return newShape;
    }
};
} // namespace CNTK

std::unique_ptr<onnxruntime::Model> CNTKToONNX::CreateModel(const FunctionPtr& src)
{
    std::unique_ptr<onnxruntime::Model> model(new onnxruntime::Model("CNTKGraph", true));
    auto &dstGraph = model->MainGraph();
    CNTKToONNXHelper::Copy(src, &dstGraph);
    onnxruntime::common::Status status = dstGraph.Resolve();
    if (!status.IsOK())
        LogicError("%s", status.ErrorMessage().c_str());

    model->SetModelversion(static_cast<onnxruntime::Version>(CNTK_ONNX_MODEL_VERSION)); // REVIEW sptiwari: This is the default. This and doc_string should be surfaced as graph's 'save' API input.
    model->SetDomain(CNTK_ONNX_MODEL_DOMAIN);
    model->SetProducerVersion(CNTK_ONNX_PRODUCER_VERSION);
    model->SetProducerName(CNTK_ONNX_PRODUCER_NAME);
    return model;
}

void CNTKToONNXHelper::Copy(const FunctionPtr& src, onnxruntime::Graph* dst)
{
    std::set<FunctionPtr> visited;
    std::unordered_map<Variable, Variable> compositeOutputsMap;
    std::unordered_map<FunctionPtr, onnxruntime::Node*> functionNodes;
    std::unordered_map<Variable, onnxruntime::Node*> variableNodes;

    //
    // Traverse the graph and collect some information.
    //
    TraverseGraph(src, visited, compositeOutputsMap);

    // this is in case the last node is wrapped with batch and sequence pack/unpack plus transpose axis ops (via importing).
    // in this case, the (un)packing op sequence will not get skipped in ProcessInputs.
    // we need to handle this specific case at begining.
    FunctionPtr srcSkipped = SkipBatchAndSequenceAxisOp(src);
    //
    // Iterate through each node in CNTK graph and create an equivalent node
    // in ONNX graph.
    //
    CreateNode(srcSkipped, dst, functionNodes, variableNodes, compositeOutputsMap);


    if (srcSkipped->OpName() == L"Combine")
    {
        HandleRootCombineOp(srcSkipped, dst);
    }
}

void CNTKToONNXHelper::HandleRootCombineOp(const FunctionPtr& src, onnxruntime::Graph* dst)
{
    // Graph::BuildConnections connects ending nodes to the sink node
    // so that the ending nodes become output nodes. 
    // When input to the last "Combine" node is also input to another node, it
    // is not an ending node thus cannot become an output node. As for this, they cannot be used for inferencing.
    // We fix this problem by connecting the node with a NoOp node to make it an ending node.
    if (src->OpName() != L"Combine")
        return;

    const GraphNodes &nodes = dst->Nodes();
    for (auto input : src->Inputs())
    {
        std::string nodeArgName = ToLegacyString(ToUTF8(input.Uid()));
        const NodeArg* nodeArg = dst->FindNodeArg(nodeArgName);
        if (!nodeArg)
            continue;

        bool foundOutputToInputNodeArg = false;
        for (GraphNodes::ConstNodeIterator it = nodes.cbegin(); it != nodes.cend() && !foundOutputToInputNodeArg; ++it)
        {
            const Node &node = *it;
            auto inputNodeArgs = node.InputDefs();
            for (int i = 0; i < inputNodeArgs.size(); i++)
            {
                if (inputNodeArgs[i]->Name() == nodeArgName)
                {
                    foundOutputToInputNodeArg = true;
                    break;
                }
            }
        }

        if (foundOutputToInputNodeArg)
        {
            // This nodeArg is not dangling. Append to it a NoOp so that it can be treated as output by onnxruntime.
            std::string out_arg_name = nodeArg->Name() + "_attach_noop_";
            AddIdentityOp(const_cast<NodeArg &>(*nodeArg), dst, out_arg_name);
        }
    }
}

void AddDataElementArrayViewToTensorProto(const NDArrayViewPtr src, int srcIndex, onnx::TensorProto& dst)
{
    CNTK::DataType dataType = src->GetDataType();
    switch (dataType)
    {
    case CNTK::DataType::Float:
    {
        auto data = src->DataBuffer<float>();
        *(dst.mutable_float_data()->Add()) = data[srcIndex];
    }
    break;
    case CNTK::DataType::Float16:
    {
        auto data = reinterpret_cast<const uint16_t*>(src->DataBuffer<float16>());
        *(dst.mutable_int32_data()->Add()) = data[srcIndex];
    }
    break;
    case CNTK::DataType::Double:
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
template <typename DType>
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
        else if (typeid(DType) == typeid(uint16_t))
            *(dst.mutable_int32_data()->Add()) = (uint16_t)data[src_index];
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
        else if (typeid(DType) == typeid(uint16_t))
            *(dst.mutable_int32_data()->Add()) = 0;
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
        else if (typeid(DType) == typeid(uint16_t))
            *(dst.mutable_int32_data()->Add()) = (uint16_t)(data[src_index] * stabilizer);
        else if(typeid(DType) == typeid(double))
            *(dst.mutable_double_data()->Add()) = (double)(data[src_index] * stabilizer);
        else
            NOT_IMPLEMENTED;
    }
}

void CNTKToONNXHelper::SetTensorType(onnx::TensorProto& dst, CNTK::DataType dataType)
{
    switch (dataType)
    {
    case CNTK::DataType::Float:
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        break;
    case CNTK::DataType::Float16:
        dst.set_data_type(onnx::TensorProto_DataType_FLOAT16);
        break;
    case CNTK::DataType::Double:
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
        case CNTK::DataType::Float:
        {
            auto data = srcTemp->DataBuffer<float>();
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        case CNTK::DataType::Float16:
        {
            auto data = reinterpret_cast<const uint16_t*>(srcTemp->DataBuffer<float16>());
            AppendCNTKWeightToONNXTensor(data, srcShape, dst, stabilizer);
            break;
        }
        case CNTK::DataType::Double:
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
        case CNTK::DataType::Float:
        {
            auto data = srcTemp->DataBuffer<float>();
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_float_data()->Add()) = (float)(data[index] * multiplier);

            break;
        }
        case CNTK::DataType::Float16:
        {
            auto data = reinterpret_cast<const uint16_t*>(srcTemp->DataBuffer<float16>());
            for (size_t index = 0; index < totalSize; index++)
                *(dst.mutable_int32_data()->Add()) = (uint16_t) (data[index] * multiplier);
            break;
        }
        case CNTK::DataType::Double:
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
            case CNTK::DataType::Float:
            {
                *(dst.mutable_float_data()->Add()) = 0;
            }
            break;
            case CNTK::DataType::Float16:
            {
                *(dst.mutable_int32_data()->Add()) = 0;
            }
            break;
            case CNTK::DataType::Double:
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

    switch (dataType)
    {
        // To handle ONNX data types other than float and double,
        // to always convert CNTK type to onnx TensorProto 
        // of the right ONNX type. 
    case CNTK::DataType::Float:
    {
        if (!inputArgType->has_tensor_type())
            dst.set_data_type(onnx::TensorProto_DataType_FLOAT);
        else
            dst.set_data_type(inputArgType->tensor_type().elem_type());
        auto data = srcTemp->DataBuffer<float>();
        for (size_t index = 0; index < totalSize; index++)
            switch (inputArgType->tensor_type().elem_type())
            {
            case onnx::TensorProto_DataType_FLOAT:
            case onnx::TensorProto_DataType_UNDEFINED:
                *(dst.mutable_float_data()->Add()) = data[index];
                break;
            case onnx::TensorProto_DataType_BOOL:
                *(dst.mutable_int32_data()->Add()) = (int) data[index];
                break;
            case onnx::TensorProto_DataType_INT32:
                *(dst.mutable_int32_data()->Add()) = (int) data[index];
                break;
            }
        break;
    }
    case CNTK::DataType::Float16:
    {
        if (!inputArgType->has_tensor_type())
            dst.set_data_type(onnx::TensorProto_DataType_FLOAT16);
        else
            dst.set_data_type(inputArgType->tensor_type().elem_type());

        auto data = reinterpret_cast<const uint16_t*>(srcTemp->DataBuffer<float16>());
        for (size_t index = 0; index < totalSize; index++)
            *(dst.mutable_int32_data()->Add()) = data[index];
        break;
    }
    case CNTK::DataType::Double:
    {
        // TODO: ONNX data types other than float and double are
        // not supported if the original CNTK data type is double.
        if (inputArgType->has_tensor_type() &&
            inputArgType->tensor_type().elem_type() != onnx::TensorProto_DataType_DOUBLE)
        {
            NOT_IMPLEMENTED;
        }

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
    onnx::TypeProto newShape = MakeTypeProtoWithShape();

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
    onnx::TypeProto newShape = MakeTypeProtoWithShape();

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
    onnx::TypeProto newShape = MakeTypeProtoWithShape();

    auto dimensions = reverse(shape);

    for (auto dimension : dimensions)
        newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension ? 1 : 0);

    return newShape;
}

onnx::TypeProto CNTKToONNXHelper::ToTypeProto(const std::vector<int64_t>& shape,
                                              bool doReverseVec /* = true*/)
{
    onnx::TypeProto newShape = MakeTypeProtoWithShape();

    std::vector<int64_t> dimensions(shape);
    if (doReverseVec)
        dimensions = reverse(dimensions);
    

    for (auto dimension : dimensions)
        if (dimension == NDShape::FreeDimension)
        {
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param(FreeSequenceDimParam);
        }
        else
        {
            newShape.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dimension);
        }

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

    onnx::TypeProto newShape = MakeTypeProtoWithShape();

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
    return ToINTS(ToTypeProto(Cast<int, int64_t>(shape), doReverseVec));
}

std::vector<int64_t> CNTKToONNXHelper::ToINTS(const std::vector<Axis>& axes)
{
    return ToINTS(ToTypeProto(axes));
}

bool IsUnSupportedLayerNormalization(const FunctionPtr src)
{
    std::string cntkOpName = ToLegacyString(ToUTF8(src->OpName()));
    return cntkOpName == "LayerNormalization" && src->Output().HasSequenceAxis();
}

bool CNTKToONNXHelper::CheckCorrectTransposeAxisToSkipForSequenceAxisOpWrapper(FunctionPtr currentOp)
{
    if (currentOp->OpName() != L"TransposeAxes")
        return true;

    if (currentOp->Attributes().Contains(L"axis1") && currentOp->Attributes().Contains(L"axis2") &&
        !currentOp->Inputs()[0].HasBatchAxis() && !currentOp->Inputs()[0].HasSequenceAxis())
    {
        Axis axis1 = (Axis)(currentOp->Attributes()[L"axis1"].Value<Axis>()).StaticAxisIndex();
        Axis axis2 = (Axis)(currentOp->Attributes()[L"axis2"].Value<Axis>()).StaticAxisIndex();
        int64_t axisIndex1 = ConvertAxisToOnnx(axis1, currentOp->Inputs()[0]);
        int64_t axisIndex2 = ConvertAxisToOnnx(axis2, currentOp->Inputs()[0]);
        // transpose between first and second of unpacked batch and sequence axes.
        return axisIndex1 + axisIndex2 == 1;
    }
    return false;
}

bool CNTKToONNXHelper::MatchOpSequence(const FunctionPtr src, std::vector<wstring> opSequence, FunctionPtr &op)
{
    FunctionPtr currentOp = src;
    for (auto opName : opSequence)
    {
        if (currentOp == nullptr || currentOp->OpName() != opName || !CheckCorrectTransposeAxisToSkipForSequenceAxisOpWrapper(currentOp))
        {
            return false;
        }
        currentOp = currentOp->Inputs().size() == 1 ? currentOp->Inputs()[0].Owner() : nullptr;
    }
    op = currentOp;
    return true;
}

bool CNTKToONNXHelper::MatchInputSequence(const Variable &input, std::vector<wstring> opSequence, Variable &inputSkipTo)
{
    FunctionPtr currentOp = input.Owner();
    for (int i = 0; i < opSequence.size(); i++)
    {
        auto opName = opSequence[i];
        if (currentOp == nullptr || currentOp->OpName() != opName || !CheckCorrectTransposeAxisToSkipForSequenceAxisOpWrapper(currentOp))
        {
            return false;
        }

        if (i < opSequence.size() - 1)
        {
            currentOp = currentOp->Inputs().size() > 0 ? currentOp->Inputs()[0].Owner() : nullptr;
        }
    }
    inputSkipTo = currentOp->Inputs()[0];
    return true;
}

// Here (in SkipBatchAndSequenceAxisInput and SkipBatchAndSequenceAxisOp) we heuristically skip wrapping op sequences 
// that have been added when importing a model with ops that require sequence axis.
// Skipping sequence of wrapping ops is a good to have to make the re-exported model clean and efficient.
// Even we do not or have missed skipping of any wrapping sequence, 
// the exported model shall still be valid and produce matching numbers.
Variable CNTKToONNXHelper::SkipBatchAndSequenceAxisInput(const Variable input)
{
    std::vector<wstring> toSequenceBatchOps({ L"ToSequenceOp", L"ToBatchAxis", L"TransposeAxes" });
    std::vector<wstring> unpackSequenceBatchOps({ L"TransposeAxes", L"Reshape", L"UnpackBatchAxis", L"UnpackSequenceOp" });

    Variable nextInput = input;
    bool skipped = false;
    while (MatchInputSequence(nextInput, toSequenceBatchOps, nextInput) ||
        MatchInputSequence(nextInput, unpackSequenceBatchOps, nextInput))
    {
        skipped = true;
    }

    if (!skipped)
        return input;

    if (nextInput.Owner() && nextInput.Owner()->OpName() == L"NoOp")
        nextInput = nextInput.Owner()->Inputs()[0];

    return nextInput;
}

FunctionPtr CNTKToONNXHelper::SkipBatchAndSequenceAxisOp(const FunctionPtr src)
{
    std::vector<wstring> toSequenceBatchOps({ L"ToSequenceOp", L"ToBatchAxis", L"TransposeAxes" });
    std::vector<wstring> unpackSequenceBatchOps({ L"TransposeAxes", L"UnpackBatchAxis", L"UnpackSequenceOp" });

    FunctionPtr op = src;
    while (MatchOpSequence(op, toSequenceBatchOps, op) ||
        MatchOpSequence(op, unpackSequenceBatchOps, op))
        ;
    return op;
}

bool IsBatchAxisOp(const FunctionPtr src)
{
    // This method checks for the following pattern to determine whether
    // we have a batch axis op:
    // "UnpackBatchAxis" --> Supported batch op --> "ToBatchAxis"
    bool isBatchAxisOp = false;
    if (ToLegacyString(ToUTF8(src->OpName())) == "UnpackBatchAxis")
    {
        auto parentNode = src->Inputs()[0].Owner();
        if (Operators::IsOpExportedWithBatchAxis(parentNode->OpName()))
        {
            for (size_t inputIndex = 0; inputIndex < parentNode->Inputs().size(); ++inputIndex)
            {
                auto input = parentNode->Inputs()[inputIndex];
                if (input.IsOutput())
                {
                    return ToLegacyString(ToUTF8(input.Owner()->OpName())) == "ToBatchAxis";
                }
            }
        }
    }
    return isBatchAxisOp;
}

TensorProto_DataType ConvertDataTypeCNTKToTensorProto(
    CNTK::DataType newDataType)
{
    // to TensorProto_DataType
    switch (newDataType)
    {
    case CNTK::DataType::Float:
        return TensorProto_DataType::TensorProto_DataType_FLOAT;
    case CNTK::DataType::Double:
        return TensorProto_DataType::TensorProto_DataType_DOUBLE;
    case CNTK::DataType::Float16:
        return TensorProto_DataType::TensorProto_DataType_FLOAT16;
    default:
        NOT_IMPLEMENTED;
    }
}

bool OpNeedONNXTypeMap(const std::string &cntkType)
{
    const vector<string> ops({"And", "Equal", "Greater", "Less", "Not", "Or", "Xor", "Gather", "ArgMax", "ArgMin", "TopK" });
    for (auto o : ops)
    {
        if (cntkType == o)
            return true;
    }
    return false;
}

// Generate ONNX nodes with correct tensor types.
// We call this function to work around the type compatiblity issue between CNTK and ONNX.
TensorProto_DataType MapAndUpdateONNXType(const std::string &op, bool inputArg, int argOrder, CNTK::DataType dataType,
                          onnx::TypeProto *type)
{
    if (op == "And" || op == "Not" || op == "Or" || op == "Xor")
    {
        if (type)
            type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_BOOL);
        return onnx::TensorProto_DataType_BOOL;
    }
    else if (!inputArg && (op == "ArgMax" || op == "ArgMin"))
    {
        if (type)
            type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);
        return onnx::TensorProto_DataType_INT64;
    }
    else if (op == "Equal")
    {
        if (inputArg)
        {
            if (type)
                type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
            return onnx::TensorProto_DataType_INT32;
        }
        else
        {
            if (type)
                type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_BOOL);
            return onnx::TensorProto_DataType_BOOL;
        }
    }
    else if (op == "Gather" && inputArg && argOrder == 0)
    {
        // Gather input order are switched so as a quick workaround I simply assume the swap
        if (type)
            type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
        return onnx::TensorProto_DataType_INT32;
    }
    else if ((op == "Greater" || op == "Less") && !inputArg)
    {
        if (type)
            type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_BOOL);
        return onnx::TensorProto_DataType_BOOL;
    }
    else if (op == "TopK" && !inputArg && argOrder == 1)
    {
        // the second output of TopK is index tensor of int64
        if (type)
            type->mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);
        return onnx::TensorProto_DataType_INT64;
    }
    else
    {
        onnx::TensorProto_DataType tensorProto_DataType = ConvertDataTypeCNTKToTensorProto(dataType);
        if (type)
            type->mutable_tensor_type()->set_elem_type(tensorProto_DataType);
        return tensorProto_DataType;
    }
}

void CNTKToONNXHelper::UpdateONNXType(CNTK::DataType dataType, onnx::TypeProto &type)
{
    TensorProto_DataType tensorProtoDataType = ConvertDataTypeCNTKToTensorProto(dataType);
    type.mutable_tensor_type()->set_elem_type(tensorProtoDataType);
}

std::string CNTKToONNXHelper::ToOPName(const FunctionPtr& src)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToLegacyString(ToUTF8(src->OpName()));
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
            bool hasAttr = src->Attributes().Contains(L"transpose");
            if (hasAttr &&
                (bool) src->Attributes()[L"transpose"].Value<bool>())
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
            wstring cntkAttributeOpName = (wstring)src->Attributes()[PrimitiveFunctionAttribute::AttributeNameReductionOpName].Value<wstring>();

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

int64_t CNTKToONNXHelper::ConvertAxisToOnnxBroadcastOfOp(const Axis &axis, const FunctionPtr &src)
{
    // TODO: what if:
    // [#][1] + [1, 2], axis = 1 => [#, 1, 2]  onnx_axis = 
    // onnx_axis = 2 with input 1  
    // onnx_axis = 1 with input 1  
    // final onnx_axis = 2
    int64_t onnx_axis = 0;
    for (int i = 0; i < src->Inputs().size(); i++)
    {
        onnx_axis = std::max(onnx_axis, ConvertAxisToOnnx(axis, src->Inputs()[i]));
    }
    return onnx_axis;
}

/*
CNTK python static axis is zero based. Batch and Sequence axis is not static axis.
CNTK cpp get static axis in a sanitized form (e.g. -axis - 1 by sanitize_axis)
In general CNTK node attribute contains axis
in a dis-normalized form (e.g. index from the last dimension).
This function converts axis to ONNX form
(e.g. index from the first dimension of the shape including both static and dynamic axes).
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
            LogicError("Inconsistent Axis in ConvertAxisToOnnx");
    }
    else if (axis.IsSequenceAxis())
    {
        return 0;
    }

    NDShape inputShape = operand.Shape();
    Axis normalizedAxis = NormalizeStaticAxis(const_cast<Axis &>(axis), inputShape.Rank());
    int64_t ax = inputShape.Rank() - normalizedAxis.StaticAxisIndex() - 1;
    ax += operand.DynamicAxes().size();
    
    // this is a special case for Sequence::ReduceElement op. axis is on sequence axis which is 1 in CNTK. it is 0 in ONNX. 
    if (ax == 1 && operand.DynamicAxes().size() == 1 && operand.Shape().Rank() > 0 && operand.Shape()[operand.Shape().Rank() - 1] == NDShape::FreeDimension)
        ax = 0;

    return ax;
}

std::vector<int64_t> CNTKToONNXHelper::ConvertAxesToOnnx(const std::vector<Axis> &axes, const Variable &operand)
{
    if (std::any_of(axes.cbegin(), axes.cend(), [](const Axis &axis) {return axis == Axis::AllStaticAxes() || axis == Axis::AllAxes(); }))
    {
        std::vector<int64_t> onnxAxes;
        if (std::any_of(axes.cbegin(), axes.cend(), [](const Axis &axis) {return axis == Axis::AllAxes(); }))
        {
            for (int i = 0; i<operand.DynamicAxes().size(); i++)
            {
                onnxAxes.push_back(i);
            }
        }

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

// prepare an input node arg with correct name and meta data so that onnxruntime can make the connection.
void CNTKToONNXHelper::PrepareRNNInput(const Variable &X, Graph *graph, std::vector<onnxruntime::NodeArg*> &nodeInputs)
{
    Variable input;
    wstring opName = X.Owner() ? X.Owner()->OpName() : L"";
    
    if (X.BlockFunctionVariableMapping().IsInitialized() && !Operators::IsRNNOp(ToLegacyString(ToUTF8(opName))) && opName != L"Embedding")
    {
        // Embedding block output name is the block name already so we shall not mape ro the root function argument.
        input = X.BlockFunctionVariableMapping();
    }
    else
    {
        input = X;
    }

    std::string inputName = ToLegacyString(ToUTF8(input.Uid()));
    onnx::TypeProto inputArgType = ToTypeProto(input.Shape(), (int)(input.DynamicAxes().size()));

    if (input.IsInput() && input.HasSequenceAxis())
        (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_param(FreeSequenceDimParam);

    UpdateONNXType(input.GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(inputName, &inputArgType);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareLSTMInitialStateNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                                   const std::vector<Variable> &initialVariables, int batchSize, int cellSize,
                                                   const std::string &uid, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    std::vector<int64_t> shape({ (int64_t)initialVariables.size(), batchSize , cellSize });
    bool doReverseVec = false;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(initialVariables[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg& inputArg = graph->GetOrCreateNodeArg(uid, &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < initialVariables.size(); i++)
    {
        const Variable &variable = initialVariables[i];
        auto srcTensor = variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value(); 
        srcTensors.push_back(srcTensor);
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    FillTensorWithScalar(srcTensors, dstTensor, shape);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareLSTMPeepholeNode(onnxruntime::Graph* graph,
                                               std::unordered_map<Variable, onnxruntime::Node*>& variableNodes, const std::vector<Variable> &Ps,
                                               const std::vector<double> &stabilizerDcCoefs, const std::vector<double> &stabilizerCCoefs,
                                               std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    // this method is called when all Ps are valid parameter/constant variable.
    int hidden_size = Ps[0].Shape()[0];
    int directions = Ps.size() / 3;
    bool doReverseVec = false;
    std::vector<int64_t> shape({ directions, 3 * hidden_size });
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ps[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg& inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Ps[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

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
    dstTensor.set_name(inputName);
    CopyTensorsWithMultipliers(srcTensors, multipliers, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareLSTMBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                           const std::vector<Variable> &Bs, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int64_t> shape = Cast<size_t, int64_t>((NDShape({ Bs.size() }).AppendShape(Bs[0].Shape())).Dimensions());

    // ONNX LSTM spec has 2 bias, for forward and backward.
    shape[1] *= 2;
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Bs[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg*> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg*> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, nullptr, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareLSTMWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                             const std::vector<Variable> &Ws, double *stabilizerConstants, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    // NDShape is in reversed order relative CNTK python so doReverseVec need to be true
    // when converting to ONNX tensor.
    // However with LSTM, CNTK python weight tensor shape is already reversed relative to ONNX.
    // We do not want to reverse again.
    bool doReverseVec = false;

    std::vector<int64_t> shape = Cast<size_t, int64_t>((NDShape({ Ws.size() }).AppendShape(Ws[0].Shape())).Dimensions());
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ws[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Ws[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({&inputArg});
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Ws.size(); i++)
    {
        const Variable &variable = Ws[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    CopyTensorsWithCNTKToONNXLSTMWeightLayoutConversion(srcTensors, stabilizerConstants, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

std::string DeriveDirectionString(const std::vector<FunctionPtr> lstms,
                                  std::map<RNNDirection, int> directionCount)
{
    return lstms.size() == 2 ? RNNDirectionBidirection :(directionCount[RNNDirection::Backward] == 1 ? RNNDirectionReverse : RNNDirectionForward);
}

void AddEmptyInput(Graph *graph, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg("", nullptr);
    nodeInputs.emplace_back(&inputArg);
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

std::pair<string, string> MakeRNNAndPostReshapeOutputNames(const std::vector<FunctionPtr> &lstms,
    const std::vector<Variable> &Yhs, const FunctionPtr &src)
{
    std::string nodeOutputName;
    if (lstms.size() == 1)
        nodeOutputName = ToLegacyString(ToUTF8(Yhs[0].Uid()));
    else
        nodeOutputName = ToLegacyString(ToUTF8(src->Output().Uid()));
    std::string nodeOutputNameBeforeReshape = nodeOutputName + "_before_reshape";
    return std::make_pair(nodeOutputName, nodeOutputNameBeforeReshape);
}

Variable FindInputToRNN(int startIndex, std::vector<Variable> &inputs)
{
    // input is the one other than bias, weights (ordered before startIndex), 
    // and past/future ops. 
    int inputIndex = inputs.size() - 1;
    for (; inputIndex >= startIndex; inputIndex--)
    {
        if (inputs[inputIndex].Owner() == nullptr ||
            (inputs[inputIndex].Owner()->OpName() != L"PastValue" && inputs[inputIndex].Owner()->OpName() != L"FutureValue"))
            break;
    }
    return inputs[inputIndex];
}

onnxruntime::Node* CNTKToONNXHelper::CreateLSTMNode(const FunctionPtr &src,
                                               onnxruntime::Graph* graph,
                                               std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                               std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
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

        Xs[directionIndex] = FindInputToRNN(CNTKLSTMHiddenWeightIndex + 1, inputs);

        // weight, hidden weight, and bias have fixed indices.
        // Thus we do not bother obtain them through traversing.
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

    // TODO: implement peephole
    // Variable P;

    // inputs
    std::vector<onnxruntime::NodeArg *> nodeInputs;
    Variable input = SkipBatchAndSequenceAxisInput(Xs[0]);
    PrepareRNNInput(input, graph, nodeInputs);
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
            AddEmptyInput(graph, nodeInputs);
        }

        // TODO: enable sequence_lens. It requires additional model input of batched sequence data layout.
        // Need to investigate how this is done with CNTK python API.
        bool has_sequence_lens = false;
        std::string sequence_lens_inputName = "sequence_lens___";
        if (has_sequence_lens)
        {
            onnx::TypeProto inputArgType = ToTypeProto(std::vector<int64_t>({ 1 }), false);
            inputArgType.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT32);
            onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(sequence_lens_inputName, &inputArgType);
            nodeInputs.push_back(&inputArg);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_h)
        {
            std::string hiddenUid = ToLegacyString(ToUTF8(Yhs[0].Uid())) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }

        bool has_initial_c = std::all_of(initialCs.begin(), initialCs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_c)
        {
            std::string cellUid = ToLegacyString(ToUTF8(Ycs[0].Uid())) + "_initial_c";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialCs, FreeBatchSize, hidden_size, cellUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }

        // peephole
        bool hasPeephole = std::all_of(Ps.begin(), Ps.end(), [](Variable &v) {return v.IsInitialized(); });
        if (hasPeephole)
        {
            PrepareLSTMPeepholeNode(graph, variableNodes, Ps, stabilizerDcCoefs, stabilizerCCoefs, nodeInputs);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }
    }

    std::vector<onnxruntime::NodeArg *> nodeOutputs;
    std::string nodeOutputName, 
        nodeOutputNameBeforeReshape;
    std::tie<std::string, std::string>(nodeOutputName, nodeOutputNameBeforeReshape) = MakeRNNAndPostReshapeOutputNames(lstms, Yhs, src);

    {
        auto outputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)NDShape::FreeDimension, 
            (int64_t)Yhs.size(), FreeBatchSize, (int64_t)Yhs[0].Shape()[0]}), false);
        UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
        onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeOutputNameBeforeReshape, &outputArgType);
        nodeOutputs.push_back(&outputArg);
    }

    // TODO: Except X, all other inputs to LSTM are treated as constant.
    // It is highly unlikely that any other input is an output of an op.
    // We will investigate once it is real.
    if (input.Owner().get() != nullptr)
        CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = ToLegacyString(ToUTF8(src->Uid()));
    onnxruntime::Node *lstmNode = graph->AddNode(nodeName, "LSTM", "", nodeInputs, nodeOutputs);

    lstmNode->AddAttribute("activations", activations);
    lstmNode->AddAttribute("direction", direction);
    lstmNode->AddAttribute("hidden_size", (int64_t)hidden_size);

    // TODO: make bidirectional LSTM work by figuring out output data
    // layout transpose in InsertReshapeNodeToCNTKFunction.
    if (lstms.size() == 2)
        NOT_IMPLEMENTED;

    // squeeze direction axis out. This is safe because it is not bi-directional node.

    std::vector<int64_t> shape({ (int64_t)NDShape::FreeDimension, 1, hidden_size });

    onnxruntime::Node *squeezedLSTMNode = InsertReshapeNodeToCNTKFunction(src, lstmNode, shape, graph, nodeOutputName);

    functionNodes.emplace(src, squeezedLSTMNode);
    return squeezedLSTMNode;
}

void CNTKToONNXHelper::PrepareGRUBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                          const std::vector<Variable> &Bs, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;
    int numDirections = Bs.size();
    int hiddenSize = Bs[0].Shape()[0] / GRUWeightDimensionHiddenMultiplier;

    std::vector<int64_t> shape({ numDirections, GRUBiasDimensionHiddenMultiplier * hiddenSize });

    // ONNX GRU spec has 2 bias, for forward and backward.
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Bs[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    CopyRNNBiasTensors(srcTensors, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareGRUZRHWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                               const std::vector<Variable> &Rzrs, const std::vector<Variable> &Rhs, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    int numDirections = Rzrs.size();
    int hiddenSize = Rzrs[0].Shape().Dimensions()[1];
    std::vector<int64_t> shape({ numDirections, GRUWeightDimensionHiddenMultiplier * hiddenSize, hiddenSize });
    onnx::TypeProto inputArgType = ToTypeProto(shape, false);
    UpdateONNXType(Rzrs[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Rzrs[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcZRTensors, srcHTensors;
    for (int i = 0; i < Rzrs.size(); i++)
    {
        const Variable &variable = Rzrs[i];
        srcZRTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());

        const Variable &variableH1 = Rhs[i];
        srcHTensors.push_back(variableH1.IsParameter() ? Parameter(variableH1).Value() : Constant(variableH1).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    CopyGRUStateWeightTensors(srcZRTensors, srcHTensors, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

void CNTKToONNXHelper::PrepareRNNWeightNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                            const std::vector<Variable> &Ws, std::vector<onnxruntime::NodeArg *> &nodeInputs,
                                            std::function<void(const std::vector<NDArrayViewPtr> &srcTensors,
                                                               onnx::TensorProto& dst, const onnx::TypeProto &inputArgType)> weightConverter)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;

    std::vector<int64_t> shape = Cast<size_t, int64_t>((NDShape({Ws.size()}).AppendShape(Ws[0].Shape())).Dimensions());
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Ws[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Ws[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({&inputArg});
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Ws.size(); i++)
    {
        const Variable &variable = Ws[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    weightConverter(srcTensors, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

onnxruntime::Node *CNTKToONNXHelper::CreateGRUNode(const FunctionPtr &src,
                                              onnxruntime::Graph* graph,
                                              std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                              std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
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

        Xs[directionIndex] = FindInputToRNN(CNTKGRUHiddenWeightHIndex + 1, inputs);

        // Weight, hidden weight, and bias have fixed indices.
        // Thus we do not bother obtain them through traversing.
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
    std::vector<onnxruntime::NodeArg *> nodeInputs;
    Variable input = SkipBatchAndSequenceAxisInput(Xs[0]);
    PrepareRNNInput(input, graph, nodeInputs);
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
            AddEmptyInput(graph, nodeInputs);
        }

        {
            // sequence_lens is not supported
            AddEmptyInput(graph, nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_h)
        {
            std::string hiddenUid = ToLegacyString(ToUTF8(Yhs[0].Uid())) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }
    }

    std::string nodeOutputName, nodeOutputNameBeforeReshape;
    std::tie<std::string, std::string>(nodeOutputName, nodeOutputNameBeforeReshape) = MakeRNNAndPostReshapeOutputNames(grus, Yhs, src);

    std::vector<onnxruntime::NodeArg *> nodeOutputs;
    {
        auto outputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)NDShape::FreeDimension, 
            (int64_t)Yhs.size(), FreeBatchSize, (int64_t)Yhs[0].Shape()[0] }), false);
        UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
        onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeOutputNameBeforeReshape, &outputArgType);
        nodeOutputs.push_back(&outputArg);

        {
            Variable Yh = Yhs[0];
            std::string nodeName = ToLegacyString(ToUTF8(Yh.Uid())) + "_h";
            // TODO: batchSize is fixed to one. Needs to find out how to handle bacth axis as a free dimension.
            const int batchSize = 1;
            const bool doReverseVec = false;
            auto outputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)Yhs.size(), batchSize, (int)Yh.Shape()[0] }), doReverseVec);
            UpdateONNXType(Yh.GetDataType(), outputArgType);
            onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeName, &outputArgType);
            nodeOutputs.push_back(&outputArg);
        }
    }

    // TODO: Except X, all other inputs to GRU are treated as constant.
    // It is highly unlikely that any other input is an output of an op.
    // We will investigate once it is real.
    if (input.Owner().get() != nullptr)
        CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = ToLegacyString(ToUTF8(src->Uid()));
    onnxruntime::Node *gruNode = graph->AddNode(nodeName, "GRU", "", nodeInputs, nodeOutputs);

    gruNode->AddAttribute("activations", activations);
    gruNode->AddAttribute("direction", direction);
    gruNode->AddAttribute("hidden_size", (int64_t)hidden_size);

    // TODO: make bidirectional GRU work by figuring out output data
    // layout transpose in InsertReshapeNodeToCNTKFunction.
    if (grus.size() == 2)
        NOT_IMPLEMENTED;

    // TODO: uncomment this code once LotusRT output shape matches ONNX
    // squeeze direction axis out. This is safe because it is not bi-directional node.
    std::vector<int64_t> shape({ (int64_t)NDShape::FreeDimension, 1, hidden_size });
    onnxruntime::Node *squeezedLSTMNode = InsertReshapeNodeToCNTKFunction(src, gruNode, shape, graph, nodeOutputName);
    functionNodes.emplace(src, squeezedLSTMNode);
    return squeezedLSTMNode;
}

void CNTKToONNXHelper::PrepareRNNBiasNode(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                          const std::vector<Variable> &Bs, std::vector<onnxruntime::NodeArg *> &nodeInputs)
{
    // TODO: sanity check for all variables to have the same shape and data types.
    bool doReverseVec = false;
    int numDirections = Bs.size();
    int hiddenSize = Bs[0].Shape()[0];

    std::vector<int64_t> shape({ numDirections, 2 * hiddenSize });

    // ONNX GRU spec has 2 bias, for forward and backward.
    onnx::TypeProto inputArgType = ToTypeProto(shape, doReverseVec);
    UpdateONNXType(Bs[0].GetDataType(), inputArgType);
    onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(Bs[0].Uid())), &inputArgType);
    std::vector<onnxruntime::NodeArg *> varOutputs({ &inputArg });
    std::vector<onnxruntime::NodeArg *> varInputs;
    std::string inputName = inputArg.Name();

    std::vector<NDArrayViewPtr> srcTensors;
    for (int i = 0; i < Bs.size(); i++)
    {
        const Variable &variable = Bs[i];
        srcTensors.push_back(variable.IsParameter() ? Parameter(variable).Value() : Constant(variable).Value());
    }

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputName);
    CopyRNNBiasTensors(srcTensors, dstTensor, inputArgType);

    graph->AddInitializedTensor(dstTensor);
    nodeInputs.push_back(&inputArg);
}

onnxruntime::Node *CNTKToONNXHelper::CreateRNNNode(const FunctionPtr &src,
                                              onnxruntime::Graph* graph,
                                              std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                              std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
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

        Xs[directionIndex] = FindInputToRNN(CNTKRNNBiasIndex + 1, inputs);

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
    std::vector<onnxruntime::NodeArg *> nodeInputs;
    Variable input = SkipBatchAndSequenceAxisInput(Xs[0]);
    PrepareRNNInput(input, graph, nodeInputs);
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
            AddEmptyInput(graph, nodeInputs);
        }

        {
            // sequence_lens is not supported
            AddEmptyInput(graph, nodeInputs);
        }

        bool has_initial_h = std::all_of(initialHs.begin(), initialHs.end(), [](Variable &v) {return v.IsInitialized(); });
        if (has_initial_h)
        {
            std::string hiddenUid = ToLegacyString(ToUTF8(Yhs[0].Uid())) + "_initial_h";
            PrepareLSTMInitialStateNode(graph, variableNodes, initialHs, FreeBatchSize, hidden_size, hiddenUid, nodeInputs);
        }
        else
        {
            AddEmptyInput(graph, nodeInputs);
        }
    }

    std::string nodeOutputName, nodeOutputNameBeforeReshape;
    std::tie<std::string, std::string>(nodeOutputName, nodeOutputNameBeforeReshape) = MakeRNNAndPostReshapeOutputNames(rnns, Yhs, src);

    std::vector<onnxruntime::NodeArg *> nodeOutputs;
    {
        auto outputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)NDShape::FreeDimension, 
            (int64_t)Yhs.size(), FreeBatchSize, (int64_t)Yhs[0].Shape()[0] }), false);
        UpdateONNXType(Yhs[0].GetDataType(), outputArgType);
        onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeOutputNameBeforeReshape, &outputArgType);
        nodeOutputs.push_back(&outputArg);
    }

    if (input.Owner().get() != nullptr)
        CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto nodeName = ToLegacyString(ToUTF8(src->Uid()));
    onnxruntime::Node *rnnNode = graph->AddNode(nodeName, "RNN", "", nodeInputs, nodeOutputs);

    rnnNode->AddAttribute("activations", activations);
    rnnNode->AddAttribute("direction", direction);
    rnnNode->AddAttribute("hidden_size", (int64_t)hidden_size);

    //// TODO: make bidirectional RNN work by figuring out output data
    //// layout transpose in InsertReshapeNodeToCNTKFunction.
    if (rnns.size() == 2)
        NOT_IMPLEMENTED;

    //// TODO: uncomment this code once LotusRT output shape matches ONNX
    //// squeeze direction axis out. This is safe because it is not bi-directional node.
    std::vector<int64_t> shape({ (int64_t)NDShape::FreeDimension, 1, hidden_size });
    onnxruntime::Node *squeezedRNNNode = InsertReshapeNodeToCNTKFunction(src, rnnNode, shape, graph, nodeOutputName);
    functionNodes.emplace(src, squeezedRNNNode);
    return squeezedRNNNode;
}

// Create an ONNX NodeArg of desired shape with constant 0s as initial values. 
onnxruntime::NodeArg &CNTKToONNXHelper::AddZerosConstantNodeArg(Graph *graph, const string &nodeArgName,
    const std::vector<int64_t> &shape, CNTK::DataType dataType)
{
    onnx::TypeProto shapeInputArgType = ToTypeProto(shape, false);
    shapeInputArgType.mutable_tensor_type()->set_elem_type(ConvertDataTypeCNTKToTensorProto(dataType));
    onnxruntime::NodeArg &shapeInputArg = graph->GetOrCreateNodeArg(nodeArgName, &shapeInputArgType);

    onnx::TensorProto dstTensor;
    dstTensor.set_name(shapeInputArg.Name());
    dstTensor.set_data_type(ConvertDataTypeCNTKToTensorProto(dataType));

    if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) {return dim <= 0; }))
        LogicError("Invalid splice inputs shape");

    int64_t totalSize = std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());
    switch (dataType)
    { 
    case CNTK::DataType::Float16:
        dstTensor.mutable_int32_data()->Resize((int)totalSize, 0);
        break;
    case CNTK::DataType::Float:
        dstTensor.mutable_float_data()->Resize((int)totalSize, (float)0);
        break;
    case CNTK::DataType::Double:
        dstTensor.mutable_double_data()->Resize((int)totalSize, 0);
        break;
    default:
        NOT_IMPLEMENTED;
    }

    for (int index = 0; index < shape.size(); index++)
        *(dstTensor.mutable_dims()->Add()) = shape[index];

    graph->AddInitializedTensor(dstTensor);
    return shapeInputArg;
}

onnxruntime::Node *CNTKToONNXHelper::AddReshapeNodeImpl(Graph *graph, const string &nodeName, NodeArg *input, NodeArg *output, const std::vector<int64_t> &newShape)
{
    onnx::TypeProto shapeInputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)newShape.size() }));
    shapeInputArgType.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);

    onnxruntime::NodeArg &shapeInputArg = graph->GetOrCreateNodeArg(output->Name() + "_shape", &shapeInputArgType);

    onnx::TensorProto dstTensor;
    dstTensor.set_name(shapeInputArg.Name());
    dstTensor.set_data_type(onnx::TensorProto_DataType_INT64);
    for (size_t index = 0; index < newShape.size(); index++)
        if (newShape[index] == NDShape::FreeDimension)
        {
                
            *(dstTensor.mutable_int64_data()->Add()) = ReshapeKeepInputDim;
        }
        else if (newShape[index] == NDShape::InferredDimension)
        {
            // TODO: add a test case for this code path.
            *(dstTensor.mutable_int64_data()->Add()) = ReshapeInferredDim;
        }
        else
        {
            *(dstTensor.mutable_int64_data()->Add()) = newShape[index];
        }
    *(dstTensor.mutable_dims()->Add()) = newShape.size();
    graph->AddInitializedTensor(dstTensor);

    auto reshapeNode1 = graph->AddNode(nodeName, "Reshape", "", { input, &shapeInputArg }, { output });
    return reshapeNode1;
}


onnxruntime::Node *CNTKToONNXHelper::AddReshapeNode(onnxruntime::NodeArg &nodeArg, const std::vector<int64_t> &newShape, const std::string &outArgName, 
    onnxruntime::Graph *graph)
{
    onnx::TypeProto typeProto = ToTypeProto(newShape, false);
    onnx::TensorProto_DataType elemType = nodeArg.TypeAsProto()->tensor_type().elem_type();
    typeProto.mutable_tensor_type()->set_elem_type(elemType);

    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(outArgName, &typeProto);
    auto reshapeNode = AddReshapeNodeImpl(graph, nodeArg.Name() + string("_reshape"), 
        const_cast<onnxruntime::NodeArg *>(&nodeArg), &outputArg, newShape);
    return reshapeNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddMatMulNode(onnxruntime::NodeArg &nodeArg1, onnxruntime::NodeArg &nodeArg2, onnxruntime::Graph* graph, 
    const std::string &out_arg_name)
{
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(out_arg_name, nullptr);
    onnxruntime::Node* argMatMulNode = graph->AddNode(
        nodeArg1.Name() + string("_matmul"), "MatMul", "", { &nodeArg1, &nodeArg2 }, { &outputArg });
    return argMatMulNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddAddNode(onnxruntime::NodeArg &nodeArg1, onnxruntime::NodeArg &nodeArg2, onnxruntime::Graph* graph,
    const std::string &out_arg_name)
{
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(out_arg_name, nullptr);
    onnxruntime::Node* argMatMulNode = graph->AddNode(
        nodeArg1.Name() + string("_add"), "Add", "", { &nodeArg1, &nodeArg2 }, { &outputArg });
    return argMatMulNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddIdentityOp(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, const std::string &out_arg_name)
{
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(out_arg_name, nullptr);
    onnxruntime::Node* identityNode = graph->AddNode(
        nodeArg.Name() + string("_identity"), "Identity", "", { &nodeArg}, { &outputArg });
    return identityNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddArgMaxNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, int axis)
{
    // onnxruntime::NodeArg inputArg(nodeArg.Name(), nullptr);
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeArg.Name() + "argmax_out", nullptr);
    onnxruntime::Node* argMaxNode = graph->AddNode(nodeArg.Name() + string("_argmax"), "ArgMax", "", { &nodeArg }, { &outputArg });
    argMaxNode->AddAttribute("axis", (int64_t)axis);
    return argMaxNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddCastNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph, onnx::TensorProto_DataType toType)
{
    // onnxruntime::NodeArg inputArg(nodeArg.Name(), nullptr);
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeArg.Name() + "cast_out", nullptr);
    onnxruntime::Node* castNode = graph->AddNode(nodeArg.Name() + string("_cast"), "Cast", "", { &nodeArg }, { &outputArg });
    castNode->AddAttribute("to", (int64_t)toType);
    return castNode;
}

onnxruntime::Node *CNTKToONNXHelper::AddTransposeNode(onnxruntime::NodeArg &nodeArg, onnxruntime::Graph* graph,
    const std::vector<int64_t> &perm, onnx::TypeProto& transposeOutputArgType, const std::string &outputNodeArgName)
{
    onnxruntime::NodeArg &outputArg = graph->GetOrCreateNodeArg(nodeArg.Name() + "transpose_out", &transposeOutputArgType);
    onnx::TensorProto_DataType elementType = nodeArg.TypeAsProto()->tensor_type().elem_type();
    const_cast<TypeProto*>(outputArg.TypeAsProto())->mutable_tensor_type()->set_elem_type(elementType);
    onnxruntime::Node* transposeNode = graph->AddNode(nodeArg.Name() + string("_transpose"), "Transpose", "", { &nodeArg }, { &outputArg });
    transposeNode->AddAttribute("perm", perm);
    return transposeNode;
}

// This method is to workaround the fact that ONNX LSTM spec does not allow easy layer stacking.
// Mapping memory layout from a bidirectional LSTM may need some work.
// For now we simply treat a bidirectional LSTM as two separate LSTMs. We use this method to reshape
// LSTM output to squeeze away the direction dimension.
// TODO: extend this method to handle bidirection LSTMs.
onnxruntime::Node *CNTKToONNXHelper::InsertReshapeNodeToCNTKFunction(const FunctionPtr &src, onnxruntime::Node* node, const std::vector<int64_t> &shape, onnxruntime::Graph* graph,
    const std::string &nodeOutputName)
{
    FunctionPtr blockRoot = src->BlockRoot();
    Variable output;
    if (Operators::IsRNNOp(ToLegacyString(ToUTF8(src->OpName()))))
        output = src->Outputs()[0];
    else
        // a bidirection LSTM case
        NOT_IMPLEMENTED

        std::string nodeName = ToLegacyString(ToUTF8(blockRoot->Uid()));

    // We need to name reshape node's output arg with LSTM output name.
    // Thus we need to give LSTM node output a different name.
    auto outputArgs = node->OutputDefs();

    std::string lstmToReshapeNodeArgName = nodeOutputName;
    onnx::TypeProto typeProto = ToTypeProto(shape, false);
    UpdateONNXType(src->Outputs()[0].GetDataType(), typeProto);
    onnxruntime::NodeArg *outputArg = &graph->GetOrCreateNodeArg(lstmToReshapeNodeArgName, &typeProto);

    auto reshapeNode = AddReshapeNodeImpl(graph, nodeName + string("_reshape"),
        const_cast<NodeArg *>(outputArgs.at(0)), outputArg, shape);

    return reshapeNode;
}

// to handle discrepancy between CNTK and ONNX for softmax ops.
onnxruntime::Node* CNTKToONNXHelper::CreateSoftmaxLikeNode(const FunctionPtr& src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::string cntkOpName = ToLegacyString(ToUTF8(src->OpName()));
    std::string nodeName = ToLegacyString(ToUTF8(src->Uid()));
    std::string onnxOpName = cntkOpName;

    if (cntkOpName != "Softmax" && cntkOpName != "LogSoftmax" && cntkOpName != "Sequence::Softmax")
    {
        LogicError("CreateSoftmaxLikeNode is called with incorrect CNTK function (%s)", cntkOpName.c_str());
    }

    int onnxRank = src->Inputs()[0].DynamicAxes().size() + src->Inputs()[0].Shape().Rank();
    int64_t axisIndex;
    if (src->Attributes().Contains(L"axis"))
    {
        Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
        axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
    }
    else
    {
        if (cntkOpName == "Sequence::Softmax")
        {
            // sequence axis index is 0 in ONNX
            axisIndex = 0;
            onnxOpName = "Softmax";
        }
        else 
            // cntk default to the last dim 
            axisIndex = onnxRank - 1;
    }

    bool needTranspose = (axisIndex != onnxRank - 1);

    std::vector<onnxruntime::NodeArg *> inputs;
    ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);
    std::vector<onnxruntime::NodeArg *> outputs;
    ProcessOutputs(src, outputs, graph);

    // set up NodeArg names so graph is properly connected.
    std::string outterInputName = inputs[0]->Name();
    std::string outterOutputName = outputs[0]->Name();

    std::string preTransposeInputNodeArgName = outterInputName;
    std::string preTransposeNodeName = nodeName + "_preTranspose";
    std::string preTransposeOutputNodeArgName = preTransposeNodeName + "_output_0";

    std::string postTransposeNodeName = nodeName + "_postTranspose";
    std::string innerSoftmaxOutputNodeArgName = needTranspose ? (postTransposeNodeName + "_input") : outterOutputName;
    std::string postTransposeInputNodeArgName = innerSoftmaxOutputNodeArgName;
    std::string postTransposeOutputNodeArgName = outterOutputName;

    // transpose shape and permutation
    std::vector<int64_t>  perm(onnxRank);
    std::generate(perm.begin(), perm.end(), [axis = 0]() mutable { return axis++; });
    std::swap(perm[axisIndex], perm[onnxRank - 1]);

    // k := axis
    // [d_0, ..., d_k, ..., d_n]
    std::vector<int64_t> transposeOutputShape = ToINTS(*inputs[0]->TypeAsProto());

    onnxruntime::NodeArg *inputToInnerSoftmaxArgNode = inputs[0];
    if (needTranspose)
    {
        // [d_0, ..., d_k, ..., d_n] ->[d_0, ..., ..., d_n, d_k]
        std::swap(transposeOutputShape[axisIndex], transposeOutputShape[onnxRank - 1]);
        onnx::TypeProto transposeOutputArgType = ToTypeProto(transposeOutputShape, false);
        UpdateONNXType(src->Output().GetDataType(), transposeOutputArgType);

        auto functionNodeTransposed = AddTransposeNode(const_cast<NodeArg &>(*inputToInnerSoftmaxArgNode), graph, perm, 
            transposeOutputArgType, preTransposeOutputNodeArgName);

        inputToInnerSoftmaxArgNode = const_cast<NodeArg *>(functionNodeTransposed->OutputDefs()[0]);
    }

    onnx::TypeProto softmaxLikeOutputArgType = ToTypeProto(transposeOutputShape, false);
    UpdateONNXType(src->Output().GetDataType(), softmaxLikeOutputArgType);

    onnxruntime::NodeArg &innerSoftmaxLikeOutputArg = graph->GetOrCreateNodeArg(innerSoftmaxOutputNodeArgName, &softmaxLikeOutputArgType);
    onnxruntime::Node* softmaxLikeNode = graph->AddNode(nodeName, onnxOpName, "", { inputToInnerSoftmaxArgNode }, { &innerSoftmaxLikeOutputArg });

    // always softmax on the last axes
    softmaxLikeNode->AddAttribute("axis", (int64_t)onnxRank - 1);

    onnxruntime::NodeArg *outputFromInnerSoftmaxArgNode = const_cast<onnxruntime::NodeArg *>(softmaxLikeNode->OutputDefs()[0]);
    if (needTranspose)
    {
        // [d_0, ..., ..., d_n, d_k] -> [d_0, ..., d_k, ..., d_n]
        std::swap(transposeOutputShape[axisIndex], transposeOutputShape[onnxRank - 1]);
        onnx::TypeProto transposeOutputArgType = ToTypeProto(transposeOutputShape, false);
        UpdateONNXType(src->Output().GetDataType(), transposeOutputArgType);

        softmaxLikeNode = AddTransposeNode(*outputFromInnerSoftmaxArgNode, graph, perm,
            transposeOutputArgType, postTransposeOutputNodeArgName);
    }

    functionNodes.emplace(src, softmaxLikeNode);
    return softmaxLikeNode;
}

// To parse Sequence.Slice node graph to collect axis/begin index/end index
// and to build an ONNX Slice op.
// IMPORTANT NOTE:
// This function convert a CNTK Sequence::Slice op to ONNX Slice op. 
// CNTK Sequence::Slice has ability to handle input of zigged arrays (i.e. sequences of various lengths).
// ONNX Slice does not support zigged arrays data format. 
// Therefore in case of batch size larger than 1 and input data are a zigged arrays, 
// we do not expect model evaludation to generate marching numbers between CNTK and ONNX.
// with this following CNTK example:
// model = C.sequence.slice(C.sequence.input_variable((1)), -2, -1)
// model.eval([[0, 1, 2], [0, 1, 2, 3, 4]])
// CNTK output is:
// array([[1.], [3.]], dtype = float32)
// output from exported ONNX model will be:
// array([[padding_value], [3.]], dtype = float32)
onnxruntime::Node* CNTKToONNXHelper::CreateSequenceSliceNode(const FunctionPtr& src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto f = src->BlockRoot();
    int64_t beginIndex = 0, endIndex = 0;

    auto packedIndex = f->Inputs()[1].Owner();
    auto whereFunc = packedIndex->Inputs()[1].Owner();
    auto inputToWhere = whereFunc->Inputs()[0].Owner();
    // input to Where node can be:
    // ElementTimes - both indices are non-zero, beginIndex/endIndex are from First/Second inputs 
    // FutureValue - beginIndex is negative, endIndex is zero
    // PastValue - endIndex is positive, beginIndex is zero
    // 1 Minus FutureValue - endIndex is negative, beginIndex is zero
    // 1 Minus PastValue - beginIndex is positive, endIndex is zero
    auto reportLogicError = [&src]()
    {
        LogicError("Failed to parse Sequence.Slice node %s(%s).", ToLegacyString(ToUTF8(src->Name())).c_str(), ToLegacyString(ToUTF8(src->Uid())).c_str());
    };
    if (inputToWhere->OpName() == L"ElementTimes")
    {
        {
            auto beginToWhere = inputToWhere->Inputs()[0].Owner();
            if (beginToWhere->OpName() == L"Minus")
            {
                auto beginToMinusMustBeAPastValueOp = beginToWhere->Inputs()[1].Owner();
                if (beginToMinusMustBeAPastValueOp->OpName() == L"PastValue")
                    beginIndex = static_cast<int64_t>(beginToMinusMustBeAPastValueOp->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
                else
                    reportLogicError();
            }
            else if (beginToWhere->OpName() == L"FutureValue")
            {
                beginIndex = -static_cast<int64_t>(beginToWhere->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
            }
            else
                reportLogicError();
        }
        {
            auto endToWhere = inputToWhere->Inputs()[1].Owner();
            if (endToWhere->OpName() == L"Minus")
            {
                auto endToMinusMustBeAFutureValueOp = endToWhere->Inputs()[1].Owner();
                if (endToMinusMustBeAFutureValueOp->OpName() == L"FutureValue")
                    endIndex = -static_cast<int64_t>(endToMinusMustBeAFutureValueOp->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
                else
                    reportLogicError();
            }
            else if (endToWhere->OpName() == L"PastValue")
            {
                endIndex = static_cast<int64_t>(endToWhere->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
            }
            else
                reportLogicError();
        }
    }
    else if (inputToWhere->OpName() == L"FutureValue")
    {
        beginIndex = -static_cast<int64_t>(inputToWhere->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
    }
    else if (inputToWhere->OpName() == L"PastValue")
    {
        endIndex = static_cast<int64_t>(inputToWhere->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
    }
    else if (inputToWhere->OpName() == L"Minus")
    {
        auto inputToMinus = inputToWhere->Inputs()[1].Owner();
        if (inputToMinus->OpName() == L"FutureValue")
        {
            endIndex = -static_cast<int64_t>(inputToMinus->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
        }
        else if (inputToMinus->OpName() == L"PastValue")
        {
            beginIndex = static_cast<int64_t>(inputToMinus->Attributes()[PrimitiveFunctionAttribute::AttributeNameOffset].Value<size_t>());
        }
    }

    if (endIndex == 0)
        // this is where CNTK and numpy disagree. numpy will output an empty matrix
        // where CNTK outputs from beginIndex to (and include) the last.
        endIndex = INT_MAX;

    std::vector<onnxruntime::NodeArg *> inputs;
    ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);

    //std::vector<onnxruntime::NodeArg *> outputs;
    //ProcessOutputs(src, outputs, graph);
    auto outputArgType = ToTypeProto(src->Output().Shape(), src->Output().HasBatchAxis(), src->Output().HasSequenceAxis());
    UpdateONNXType(src->Output().GetDataType(), outputArgType);

    std::string outputName = ToLegacyString(ToUTF8(src->BlockRoot()->Output().Uid()));
    std::string sliceOutputName = outputName;
    bool seq_dim_is_1 = endIndex - beginIndex == 1 || (endIndex == INT_MAX && beginIndex == -1);
    if (seq_dim_is_1)
    {
        // it appears that sequence.slice squeezes sequence axis out if slice length is 1
        sliceOutputName += "_PreReshape";
    }

    onnxruntime::NodeArg &outputNodeArg = graph->GetOrCreateNodeArg(sliceOutputName, &outputArgType);

    const std::string & nodeName = ToLegacyString(ToUTF8(src->Name()));
    onnxruntime::Node *sequenceSliceNode = graph->AddNode(nodeName, "Slice", "", { inputs[inputs.size() - 1] }, { &outputNodeArg });
    sequenceSliceNode->AddAttribute("axes", std::vector<int64_t>({ int64_t(0) }));
    sequenceSliceNode->AddAttribute("ends", std::vector<int64_t>({ endIndex }));
    sequenceSliceNode->AddAttribute("starts", std::vector<int64_t>({ beginIndex }));
    if (seq_dim_is_1)
    {
        // CNTK Sequence.Slice op squeezes the sequence axis if it is of dimension 1.
        // insert reshape to remove sequence axis
        std::vector<int64_t> newShape(reverse(Cast<size_t, int64_t>(src->Output().Shape().Dimensions())));
        // add batch size at end
        newShape.insert(newShape.begin(), 1);
        const std::string outArgName = sliceOutputName;
        return AddReshapeNode(outputNodeArg, newShape, outputName, graph);
    }
    else
        return sequenceSliceNode;
}

//
// This is the main horsepower, it navigate CNTK graph recursivley while keep track of all visited nodes and variables,
// and create the corresponding ONNX graph.
//
onnxruntime::Node* CNTKToONNXHelper::CreateNode(const FunctionPtr& src,
                                           onnxruntime::Graph* graph,
                                           std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                           std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                           const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = functionNodes.find(src);
    if (iter != functionNodes.end())
        return iter->second;
    
    onnxruntime::Node* functionNode = nullptr;
    std::string cntkOpName = ToLegacyString(ToUTF8(src->OpName()));
    std::string onnxOpName = ToOPName(src);

    // TODO: uncomment this code once bidirectional LSTM is supprted.
    //if (cntkOpName == "Splice")
    //{
    //    std::vector<Variable> inputs = src->Inputs();
    //    bool bidiectionalLSTM = inputs.size() == 2 &&
    //        std::all_of(inputs.begin(), inputs.end(), [](Variable &input) {return input.Owner() != nullptr && input.Owner()->OpName() == L"LSTM"; });
    //    if (bidiectionalLSTM)
    //        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    //}
    //else
    if (cntkOpName == "Sequence::Slice")
    {
        return CreateSequenceSliceNode(src,
            graph,
            functionNodes,
            variableNodes,
            compositeOutputsMap);
    }
    else if (cntkOpName == "Softmax" || cntkOpName == "LogSoftmax" || cntkOpName == "Sequence::Softmax")
    {
        return CreateSoftmaxLikeNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "RNNStep")
    {
        return CreateRNNNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "GRU")
    {
        return CreateGRUNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "LSTM")
    {
        return CreateLSTMNode(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "Combine")
    {
        for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
        {
            auto input = src->Inputs()[inputIndex];
            CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);
        }

        // not a single node,
        return nullptr;
    }
    else if (cntkOpName == "OptimizedRNNStack")
    {
        return CreateONNXNodesForOptimizedRNNStack(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "StraightThrough")
    {
        return CreateONNXNodesForStraightThrough(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "Select")
    {
        return CreateONNXNodesForSelect(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "TransposeTimes")
    {
        return CreateONNXNodesForTimesTranspose(src, graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (cntkOpName == "UnpackBatchAxis")
    {
        if (IsBatchAxisOp(src))
            return CreateNodeForBatchAxisOp(src, graph, functionNodes, variableNodes, compositeOutputsMap);
        else
        {
            // this is a normal use of UnpackBatchAxis. ONNX does not treat batch axis specially so 
            // we shall skip the op.
            auto blockMapping = src->Inputs()[0].BlockFunctionVariableMapping();
            if (blockMapping.IsInitialized())
                return CreateNode(blockMapping.Owner(),
                    graph,
                    functionNodes,
                    variableNodes,
                    compositeOutputsMap);
            else if (src->Inputs()[0].Owner())
                return CreateNode(src->Inputs()[0].Owner(),
                    graph,
                    functionNodes,
                    variableNodes,
                    compositeOutputsMap);
        }
    }

    //
    // If this block node equivalent to a primitive ONNX OP, then treated as such.
    // And just maps its argument to ONNX node.
    //
    if (src->IsBlock() &&
        (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())))
    {
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    else if (IsUnSupportedLayerNormalization(src))
    {
        // LayerNormalization is build with a MeanVarianceNormalization op which requires
        // input to be of shape NCHW. For other cases such as language models with
        // features in 1-D, we have to fallback to unblocking the op into its subgraph. 
        // TODO: make ONNX MeanVarianceNormalization and CNTK test work with sequential models.
        functionNode = CreateNode(src->BlockRoot(), graph, functionNodes, variableNodes, compositeOutputsMap);
    }
    //
    // For compatibility of other framework that support ONNX, we will limit the list of OPs to the one
    // supported by ONNX https://github.com/onnx/onnx/tree/master/onnx/defs.
    //
    else if (Operators::IsSupportedCNTKOP(src->OpName()))
    {
        std::vector<onnxruntime::NodeArg *> inputs;
        ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);

        std::vector<onnxruntime::NodeArg *> outputs;
        ProcessOutputs(src, outputs, graph);

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

Variable SkipDynamicAxisPackUnpack(Variable input, bool &dynamicAxisPackUnpackSkipped)
{
    dynamicAxisPackUnpackSkipped = false;
    std::set<std::wstring> ops({ L"UnpackBatchAxis" , L"ToBatchAxis" , L"UnpackSequenceOp" , L"UnpackBatchAxis" });
    while (input.Owner() && ops.find(input.Owner()->OpName()) != ops.end())
    {
        input = input.Owner()->Inputs()[0];
        dynamicAxisPackUnpackSkipped = true;
    } 
    
    return input;
}

bool TryMatchNodeArgType(onnx::TypeProto &argType, onnxruntime::Graph* graph, const std::string &nodeArgName)
{
    const NodeArg* inputNodeArg = graph->FindNodeArg(nodeArgName);
    if (inputNodeArg)
    {
        onnx::TensorProto_DataType inputType = inputNodeArg->TypeAsProto()->tensor_type().elem_type();
        argType.mutable_tensor_type()->set_elem_type(inputType);
        return true;

    }
    return false;
}

// cases where adjustment may be needed for CNTK broadcast to match ONNX
// it shows (hopefully complete) cases where adjustment is needed. They fell under a few conditions.
// inputs need to be adjusted (reshaped) are listed after each case (e.g. "None", "1", "1, 2") 
// cases where inputs do not have the same dynamic axes:
// [ ][   2] + [#   ][3, 2] -> [#, 3, 2]       : None
// [ ][3, 2] + [#   ][   2] -> [#, 3, 2]       : 2          Cond. A: has dynamic axis but lower shape rank
// [ ][   2] + [#, *][3, 2] -> [#, *, 3, 2]    : None
// [ ][3, 2] + [#, *][   2] -> [#, *, 3, 2]    : 2          Cond. A
// [#][   2] + [#, *][3, 2] -> [#, *, 3, 2]    : 1          Cond. A
// [#][3, 2] + [#, *][   2] -> [#, *, 3, 2]    : 1, 2       Cond. B: has fewer dynamic axis; Cond A
// [#][   2] + [#, *][   2] -> [#, *, 2]       : 1          Cond. B
//
// cases where inputs have same dynamic axis:
// [#   ][   2] + [#   ][3, 2] -> [#, 3, 2]    : 1          Cond. A
// [#, *][   2] + [#, *][3, 2] -> [#, *, 3, 2] : 1          Cond. A
// 
// two sample models involving above broadcast cases:
// 1. With ReduceMean
//      shape_x=(2,)
//      z = C.reduce_mean(C.sequence.input_variable(shape_x) - C.reduce_mean(C.sequence.input_variable(shape_x), 0, False), 0, False)
// 2. input of different dynamic axes (this will get ReconcileDynamicAxis op which is not supported in ONNX)
//      z = C.sequence.input_variable((2,)) + C.input_variable((3,2))
//
// input is not necessarily an input to src. It may be obtained via skipping of batch/sequence pack/unpack wrappers. 
NodeArg* CNTKToONNXHelper::GetInputAdjustmentForBroadcast(onnxruntime::Graph* graph, const FunctionPtr src, 
    const Variable &input, int inputIndex, onnx::TypeProto &inputArgType)
{
    // TODO: do we need to get blockroot if it is a block function?
    if (!Operators::SupportBroadcast(src->OpName()))
        return nullptr;
    else
    {
        if (input.DynamicAxes().size() == 0)
            // ONNX and CNTK broadcasts match.
            return nullptr;

        int dynamicAxesTotal = 0;
        int rankMax = 0;
        for (int n = 0; n < src->Inputs().size(); n++)
        {
            Variable i = n == inputIndex ? input : src->Inputs()[n];
            dynamicAxesTotal = dynamicAxesTotal > i.DynamicAxes().size() ? dynamicAxesTotal : i.DynamicAxes().size();
            rankMax = rankMax > i.Shape().Rank() ? rankMax : i.Shape().Rank();
        }

        if (input.Shape().Rank() == rankMax && input.DynamicAxes().size() == dynamicAxesTotal)
            return nullptr;

        std::vector<int64_t> newShape;
        if (dynamicAxesTotal == 1)
        {
            // batch axis
            newShape.push_back(FreeBatchSize);
        }
        else
        {
            // batch and sequence axis
            newShape.push_back(NDShape::FreeDimension);
            newShape.push_back(FreeBatchSize);
        }

        for (int staticIndex = 0; staticIndex < rankMax; staticIndex++)
        {
            if (staticIndex < rankMax - input.Shape().Rank())
                newShape.push_back(1);
            else
            {
                int indexToInputShape = staticIndex - (rankMax - input.Shape().Rank()); 
                newShape.push_back(input.Shape()[indexToInputShape]);
            }
        }
        
        //auto inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis(), input.HasSequenceAxis());
        //inputArgType.mutable_tensor_type()->set_elem_type(inputArgType.tensor_type().elem_type());
        //UpdateONNXType(input.GetDataType(), inputArgType);
        std::string inputNodeArgName = ToLegacyString(ToUTF8(input.Uid()));
        std::string outputArgName = inputNodeArgName + "_reshaped_for_broadcast";
        onnxruntime::NodeArg &nodeArg = graph->GetOrCreateNodeArg(inputNodeArgName, &inputArgType);
        Node *reshapeNode = AddReshapeNode(nodeArg, newShape, outputArgName, graph);
        return const_cast<NodeArg *>(reshapeNode->OutputDefs()[0]);
    }
}

void CNTKToONNXHelper::ProcessInputs(const FunctionPtr& src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap,
    std::vector<onnxruntime::NodeArg *>& inputs)
{
    std::string cntkOpName = ToLegacyString(ToUTF8(src->OpName()));
    std::string onnxOpName = ToOPName(src);

    for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
    {
        auto input = src->Inputs()[inputIndex];

        if (input.IsPlaceholder())
        {
            input = input.BlockFunctionVariableMapping();
            if (input.IsPlaceholder())
                LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
        }

        // first try to skip complex patterns otherwise SkipDynamicAxisPackUnpack may skip an element 
        // of the pattern so that complex patterns become not skipped as a whole.
        // retry SkipBatchAndSequenceAxisInput shall be good enough.
        input = SkipBatchAndSequenceAxisInput(input);
        // UnpackBatchAxis and ToBatchAxis is a noop in ONNX 
        bool dynamicAxisPackUnpackSkipped = false;
        input = SkipDynamicAxisPackUnpack(input, dynamicAxisPackUnpackSkipped);
        if (dynamicAxisPackUnpackSkipped)
            input = SkipBatchAndSequenceAxisInput(input);

        // Special case handling of LayerNormalization layer because it changes
        // ops dynamically based on value of inputs. If more such cases ops are seen,
        // this should be abstracted out from here.
        if (ToLegacyString(ToUTF8(src->OpName())) == "LayerNormalization")
        {
            // If non-zero epsilon was specified, a fourth input is included
            // which must be ignored because we cannot export epsilon to ONNX.
            // See LayerNormalization branch in AddNode() below.
            if (src->Inputs().size() == 4 && inputIndex == 0 && input.IsConstant())
                continue;
        }
        else if (ToLegacyString(ToUTF8(src->OpName())) == "Crop")
        {
            // Export only the first input. In ONNX Crop accepts only 1 input, and there is no notion of referent input.
            if (inputIndex > 0)
                continue;
        }

        if (FilterInput(src, input, inputIndex))
            continue;

        //
        // Use user-defined name if available, otherwise use our internal unique name ID.
        //
        std::string inputName = ToLegacyString(ToUTF8(input.Uid()));
        auto inputItr = compositeOutputsMap.find(input);
        if (inputItr != compositeOutputsMap.end())
            inputName = ToLegacyString(ToUTF8(inputItr->second.Uid()));

        bool isConstant = (input.IsParameter() || input.IsConstant()) &&
            !Operators::IgnoreConstantAndParameter(src->OpName(), inputIndex);

        //
        // If this input is output, then it is the ouput of an up stream node. Recursively add all upstream nodes.
        // Pretty much, we are doing DFS.
        //
        if (input.IsOutput())
            CreateNode(input.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

        onnx::TypeProto inputArgType;

        if (cntkOpName == "Splice")
        {
            // for ops like Concat, batch axis may exist in one of the operand
            // CNTK allows the other operand(s) not having batch axis. But ONNX
            // requires operands to have the same rank
            inputArgType = ToTypeProto(input.Shape(), OpInputsHasBatchAxis(src), input.HasSequenceAxis());
        }
        else if (cntkOpName == "ImageScaler")
        {
            // TODO: verify - ONNX specifies that ImageScaler always need a batch axis
            inputArgType = ToTypeProto(input.Shape(), true);
        }
        else if (cntkOpName == "Convolution")
        {
            const size_t ConvWeightIndex = 0u;
            const size_t ConvOperandIndex = 1u;
            NDShape inputShape = input.Shape();
            if (inputIndex == ConvWeightIndex)
            {
                // CNTK kernel shape can omit the out channel axis if its value equals to 1.
                // On the other hand, ONNX spec requires out channel axis to be explicitly set. 
                // w: [O x C x W x H], operand: [N] x [C x W x H].
                // Thus insert the emulated out channel axis if needed. 
                const NDShape& operandShape = src->Inputs()[ConvOperandIndex].Shape();
                if (operandShape.Rank() >= inputShape.Rank())
                    inputShape = inputShape.AppendShape({1});
                assert(inputShape.Rank() == (operandShape.Rank() + 1));
            }
            inputArgType = ToTypeProto(inputShape, input.HasBatchAxis(), input.HasSequenceAxis());
        }
        else
        {
            inputArgType = ToTypeProto(input.Shape(), input.HasBatchAxis(), input.HasSequenceAxis());

            if (isConstant && cntkOpName == "BatchNormalization" && (inputIndex > 0 && inputIndex <= 4))
            {
                // In case of BatchNormalization, if data (input[0]) is of type FP16, then all BN stats(inputs[1:4])
                // need to be converted from FP32 to FP16 prior to getting exported to ONNX
                if (src->Inputs()[0].GetDataType() == DataType::Float16)
                    input = Utils::ConvertVariableType<float, float16>(input, true);

                // This is a workaround allowing CNTK V1 pretrained models to continue running after removal of sequence axis from input
                if (input.Shape().Rank() > 1)
                    inputArgType = ToTypeProto(input.Shape().SubShape(0, 1), input.HasBatchAxis(), input.HasSequenceAxis());
            }
        }

        // TODO: if it is an identity op, we shall peek its input node to find the correct tensor element type.

        if (onnxOpName == "Identity")
        {
            // shall match the type of the same name NodeArg from upstream. 
            string inputNodeArgName = ToLegacyString(ToUTF8(input.Uid()));
            if (!TryMatchNodeArgType(inputArgType, graph, inputNodeArgName))
                UpdateONNXType(src->Inputs()[0].GetDataType(), inputArgType);
        }
        else if (OpNeedONNXTypeMap(cntkOpName))
        {
            if (!input.IsOutput())
            {
                MapAndUpdateONNXType(onnxOpName, true, inputIndex, input.GetDataType(), &inputArgType);
            }
            else
            {
                // input NodeArg has already been created as an output NodeArg of the previous function node. 
                // a Cast op needs to be inserted to get the desired type in ONNX.
                TensorProto_DataType onnx_type = MapAndUpdateONNXType(onnxOpName, true, inputIndex, input.GetDataType(), nullptr);
                if (ConvertDataTypeCNTKToTensorProto(input.GetDataType()) != onnx_type)
                {
                    UpdateONNXType(input.GetDataType(), inputArgType);
                    onnxruntime::NodeArg &castInputArg = graph->GetOrCreateNodeArg(inputName, &inputArgType);
                    onnxruntime::Node* castNode = AddCastNode(castInputArg, graph, onnx_type);
                    inputs.push_back(const_cast<NodeArg *>(castNode->OutputDefs()[0]));

                    // we already completed preparation of this input and can proceed to the next input.
                    continue;
                }
            }
        }
        else
        {
            UpdateONNXType(input.GetDataType(), inputArgType);
        }

        //
        // Leaf nodes are data entry to the graph and need their own node with only output arg.
        //
        if (isConstant)
        {
            if (variableNodes.find(input) == variableNodes.end())
            {
                if (input.IsParameter() || input.IsConstant())
                {
                    auto srcTensor = input.IsParameter() ? Parameter(input).Value() : Constant(input).Value();

                    onnx::TensorProto dstTensor;
                    dstTensor.set_name(inputName);
                    CopyTensor(srcTensor, dstTensor, &inputArgType);

                    graph->AddInitializedTensor(dstTensor);
                }
            }
        }

        onnxruntime::NodeArg *adjusted = GetInputAdjustmentForBroadcast(graph, src, input, inputIndex, inputArgType);

        onnxruntime::NodeArg &inputArg = adjusted == nullptr ? graph->GetOrCreateNodeArg(inputName, &inputArgType) : *adjusted;

        inputs.push_back(&inputArg);

        if (cntkOpName == "Reshape")
        {
            // ONNX1.2 reshape node take shape as input instead of attribute. 

            // We can construct the shape input for onnx by two ways: 1. cntk node output shape, or 2. cntk node attribute "newShape".
            // If there attribute "newShape" is missing, or attributes "beginAxis" and "endAxis" exists, we use cntk node output shape.
            // such that we don't need to duplicate the shape inference logic here. 
            // Otherwise we use the cntk node attribute "newShape". 
            bool useOutputShape = [&]() {
                if (!src->Attributes().Contains(L"newShape") || ((NDShape)src->Attributes()[L"newShape"].Value<NDShape>()).Rank() == 0)
                    return true;
                if (src->Attributes().Contains(L"beginAxis") && ((Axis)src->Attributes()[L"beginAxis"].Value<Axis>()).StaticAxisIndex() != 0)
                    return true;
                if (src->Attributes().Contains(L"endAxis") && ((Axis)src->Attributes()[L"endAxis"].Value<Axis>()).StaticAxisIndex() != src->Inputs()[0].Shape().Rank())
                    return true;
                return false;
            }();
            const NDShape shape = useOutputShape ? src->Output().Shape() : (NDShape)src->Attributes()[L"newShape"].Value<NDShape>();

            std::vector<int> newShapeVec;
            size_t numInferredDimensions(0);
            for (const auto& axisSize : shape.Dimensions())
            {
                if (axisSize == NDShape::InferredDimension)
                {
                    numInferredDimensions++;
                    if (numInferredDimensions > 1)
                        LogicError("Reshape: Multiple InferredDimension not supported by ONNX.");
                    else
                        newShapeVec.push_back(ReshapeInferredDim);
                }
                else // REVIEW SPTIWARI: Should we fill 0 for FreeDimension here?
                    newShapeVec.push_back(static_cast<int>(axisSize));
            }

            // If output has batch axis, then create an output shape (which goes in as input to the
            // ONNX node) with an additional axis for batch axis (1). 
            if (src->Output().HasBatchAxis())
                newShapeVec.push_back(1u);
            std::reverse(newShapeVec.begin(), newShapeVec.end());
            onnx::TypeProto shapeInputArgType = ToTypeProto(std::vector<int64_t>({ (int64_t)newShapeVec.size() }));
            shapeInputArgType.mutable_tensor_type()->set_elem_type(onnx::TensorProto_DataType_INT64);

            onnxruntime::NodeArg &shapeInputArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(src->Output().Uid())) + "_shape", &shapeInputArgType);

            inputs.push_back(&shapeInputArg);

            onnx::TensorProto dstTensor;
            dstTensor.set_name(shapeInputArg.Name());
            dstTensor.set_data_type(onnx::TensorProto_DataType_INT64);
            for (size_t index = 0; index < newShapeVec.size(); index++)
                *(dstTensor.mutable_int64_data()->Add()) = (int)newShapeVec[index];
            *(dstTensor.mutable_dims()->Add()) = newShapeVec.size();
            graph->AddInitializedTensor(dstTensor);
        }
    }
}

void CNTKToONNXHelper::ProcessOutputs(const FunctionPtr& src,
    std::vector<onnxruntime::NodeArg *>& outputs, Graph *graph)
{
    std::string onnxOpName = ToOPName(src);
    int outputIndex = 0;
    for (const auto& output : src->Outputs())
    {
        auto outputArgType = ToTypeProto(output.Shape(), output.HasBatchAxis(), output.HasSequenceAxis());
        if (onnxOpName == "Identity")
        {
            // shall match the type of this Identity node's input NodeArg.
            string inputNodeArgName = ToLegacyString(ToUTF8(src->Inputs()[0].Uid()));
            if (!TryMatchNodeArgType(outputArgType, graph, inputNodeArgName))
                UpdateONNXType(src->Inputs()[0].GetDataType(), outputArgType);
        }
        else if (OpNeedONNXTypeMap(onnxOpName))
        {
            MapAndUpdateONNXType(onnxOpName, false, outputIndex, output.GetDataType(), &outputArgType);
        }
        else
        {
            UpdateONNXType(output.GetDataType(), outputArgType);
        }
        onnxruntime::NodeArg &outputNodeArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(output.Uid())), &outputArgType);
        outputs.emplace_back(&outputNodeArg);
        outputIndex++;
    }
}

void CNTKToONNXHelper::TraverseGraph(const FunctionPtr& src,
                                     std::set<FunctionPtr>& visited,
                                     std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    auto iter = visited.find(src);
    if (iter != visited.end()) 
        return;

    std::string opName = ToLegacyString(ToUTF8(src->OpName()));
    if (Operators::IsLoopOp(opName))
    {
        // avoid infinite loop
        return;
    }

    if (!Operators::IsRNNOp(opName) &&
        src->IsBlock() && 
        (!Operators::IsSupportedCNTKOP(src->OpName()) || Operators::IsLayerCNTKOP(src->OpName())) ||
        IsUnSupportedLayerNormalization(src))
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

void CNTKToONNXHelper::CopyAttributes(const FunctionPtr& src, onnxruntime::Node* node)
{
    auto lookup = Operators::CntkToONNXLookup();
    assert(lookup.count(src->OpName()) != 0);

    std::string opName = ToLegacyString(ToUTF8(src->OpName()));
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
        else if (src->OpName() == L"Cast")
        {
            DataType newDataType = static_cast<DataType>(src->Attributes()[L"newDataType"].Value<int>());
            int64_t to = static_cast<int64_t>(ConvertDataTypeCNTKToTensorProto(newDataType));
            node->AddAttribute(attributesMap[L"newDataType"], to);
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
            node->AddAttribute(attributesMap[L"epsilon"], epsilon);
            node->AddAttribute("momentum", momentum);
        }
        else if (src->OpName() == L"LocalResponseNormalization")
        {
            auto depthRadius = (int64_t)src->Attributes()[L"depthRadius"].Value<size_t>();
            auto bias = (float)src->Attributes()[L"bias"].Value<double>();
            auto alpha = (float)src->Attributes()[L"alpha"].Value<double>();
            auto beta = (float)src->Attributes()[L"beta"].Value<double>();

            node->AddAttribute(attributesMap[L"size"], 2*depthRadius + 1);
            node->AddAttribute(attributesMap[L"bias"], bias);
            node->AddAttribute(attributesMap[L"alpha"], alpha);
            node->AddAttribute(attributesMap[L"beta"], beta);
        }
        else if (src->OpName() == L"ELU")
        {
            float alpha = 1.0f;
            if (src->Attributes().Contains(L"alpha"))
                alpha = (float)src->Attributes()[L"alpha"].Value<double>();
            node->AddAttribute("alpha", alpha);
        }
        else if (src->OpName() == L"LeakyReLU")
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
        if (src->OpName() == L"ReduceL1" || src->OpName() == L"ReduceL2" || src->OpName() == L"ReduceSumSquare")
        {
            SetReduceElementsAttributes(src, node);
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
                // It is safe here to assume that the axis is a static axis.
                int64_t axisIndex1 = ConvertAxisToOnnx(axis1, src->Inputs()[0]);
                int64_t axisIndex2 = ConvertAxisToOnnx(axis2, src->Inputs()[0]);
                std::swap(perm[axisIndex1], perm[axisIndex2]);
                node->AddAttribute(attributesMap[L"axisVec"], perm);
            }
        }
        else if (src->OpName() == L"Reshape")
        {
        }
        else if (src->OpName() == L"Splice")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            int64_t axisIndex = ConvertAxisToOnnxBroadcastOfOp(axis, src);
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
                // CNTK slice only support single axis slice. 
                // It is safe to assume that the axis is a static axis.
                int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
                std::vector<int64_t> sliceAxes;
                sliceAxes.push_back(axisIndex);
                node->AddAttribute(attributesMap[L"axes"], sliceAxes);

                beginIndex.push_back((int)(src->Attributes()[L"beginIndex"].Value<int>()));
                endIndex.push_back((int)(src->Attributes()[L"endIndex"].Value<int>()));
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
        else if (src->OpName() == L"Hardmax")
        {
            int numDims = src->Inputs()[0].Shape().Rank();
            if (numDims == 0)
            {
                LogicError("Zero-rank input is not supported for ONNX export.");
            }
            int64_t axisIndex = numDims - 1 + src->Inputs()[0].DynamicAxes().size();
            node->AddAttribute(attributesMap[L"axis"], axisIndex);
        }
        else if (src->OpName() == L"Softmax_onnx" || src->OpName() == L"LogSoftmax_onnx" || src->OpName() == L"Hardmax_onnx")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            int64_t axisIndex = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], axisIndex);
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
            auto pooled_shape = ToINTS(roiOutputShape, false);

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
            // Flatten op takes single axis. It is safe here to assume that the axis is a static axis.
            int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]) + 1 /* TODO: Figure out how to remove this hardcoded 1 */;
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
                // Gather op takes single axis. It is safe here to assume that the axis is a static axis.
                // axis is used to apply to reference input - the second input. 
                int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[1]);
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
        else if (src->OpName() == L"Gemm")
        {
            float alpha = static_cast<float>(src->Attributes()[L"alpha"].Value<float>());
            float beta = static_cast<float>(src->Attributes()[L"beta"].Value<float>());
            int64_t transA = static_cast<int64_t>(src->Attributes()[L"transA"].Value<bool>());
            int64_t transB = static_cast<int64_t>(src->Attributes()[L"transB"].Value<bool>());

            node->AddAttribute("alpha", alpha);
            node->AddAttribute("beta", beta);
            // Swap transpose attribute to match the swapped inputs in ONNX order. 
            node->AddAttribute("transA", transB);
            node->AddAttribute("transB", transA);
        }
        else if (src->OpName() == L"Unsqueeze")
        {
            std::vector<Axis> axes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
            // Pass in output operand, such that Unsqueeze axes can be converted based on output rank. 
            std::vector<int64_t> ax = ConvertAxesToOnnx(axes, src->Outputs()[0]);

            node->AddAttribute("axes", ax);
        }
        else if (src->OpName() == L"TopK")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            // TopK op takes single axis. It is safe here to assume that the axis is a static axis.
            int64_t ax = ConvertAxisToOnnx(axis, src->Inputs()[0]);
            node->AddAttribute(attributesMap[L"axis"], ax);

            size_t k = src->Attributes()[L"numItems"].Value<size_t>();
            node->AddAttribute(attributesMap[L"numItems"], static_cast<int64_t>(k));
        }
        else if (src->OpName() == L"Crop")
        {
            const NDShape& inputShape = src->Inputs()[0].Shape();
            const NDShape& targetShape = src->Inputs()[1].Shape();

            // ONNX Crop supports only input tensor of shape [N,C,H,W]. The spatial rank for both input and referent should equal to 3.
            if (inputShape.Rank() != 3 || targetShape.Rank() != 3)
                RuntimeError("ONNX Crop supports only input tensor of shape [N,C,H,W]. Including batch axis, input has rank %zu, referent has rank %zu. ",
                    inputShape.Rank()+1, targetShape.Rank()+1);

            size_t xOffset = inputShape[0] - targetShape[0];
            size_t yOffset = inputShape[1] - targetShape[1];

            if (src->Attributes().Contains(L"offset"))
            {
                // crop_manual
                std::vector<size_t> offsets = AsVector<size_t>(src->Attributes()[L"offset"].Value<std::vector<DictionaryValue>>());
                offsets.push_back(xOffset - offsets[0]);
                offsets.push_back(yOffset - offsets[1]);
                std::reverse(offsets.begin(), offsets.end());
                node->AddAttribute(attributesMap[L"offset"], ToINTS(offsets));
            }
            else
            {
                // TODO : crop_automatic
                RuntimeError("Exporting crop_automatic to ONNX is not supported yet.");
            }
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
            size_t groups = (src->Attributes().Contains(L"groups")) ? (size_t)src->Attributes()[L"groups"].Value<size_t>() : 1u;
            bool ceilOutDim = (src->Attributes().Contains(L"ceilOutDim")) ? (bool)src->Attributes()[L"ceilOutDim"].Value<bool>() : false;

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
                if(outputShape != NDShape({ 0 }))
                    node->AddAttribute("output_shape", ToINTS(outputShape, src->Inputs()[1].HasBatchAxis()));
                // Notes on outputShape vs pads for convolution and convTranspose.
                // In onnx spec for convTranspose, quoted here "If output_shape is specified pads values are ignored". 
                // If outputShape is exported for convTranspose, autopad/pads can be skipped. 
            }
            const NDShape& inputShape = src->Inputs()[1].Shape();
            PutPadAttrInNode(node, autoPadding, kernelShape, inputShape, strides, dilations, ceilOutDim, transpose);
        }
        else if (src->OpName() == L"Pooling")
        {
            auto kernelShape = (NDShape)src->Attributes()[L"poolingWindowShape"].Value<NDShape>();
            auto strides = (NDShape)src->Attributes()[L"strides"].Value<NDShape>();
            bool ceilOutDim = src->Attributes().Contains(L"ceilOutDim") ? (bool)src->Attributes()[L"ceilOutDim"].Value<bool>() : false;
            auto autoPadding = AsVector<bool>(src->Attributes()[L"autoPadding"].Value<std::vector<DictionaryValue>>());
            const NDShape& inputShape = src->Inputs()[0].Shape();

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
            
            // This is a workaround allowing CNTK V1 pretrained models to continue running after removal of sequence axis from inuput
            if (src->Inputs()[0].Shape().Rank() - 1 != kernelShape.Rank() && kernelShape.Dimensions()[kernelShape.Rank() - 1] == 1)
                kernelShape = kernelShape.SubShape(0, kernelShape.Rank() - 1);

            if (src->Inputs()[0].Shape().Rank() - 1 != strides.Rank() && strides.Dimensions()[strides.Rank() - 1] == 1)
                strides = strides.SubShape(0, strides.Rank() - 1);

            node->AddAttribute("kernel_shape", ToINTS(kernelShape));
            node->AddAttribute("strides", ToINTS(strides));

            auto lowerPad = ToINTS(src->Attributes()[L"lowerPad"].Value<NDShape>());
            auto upperPad = ToINTS(src->Attributes()[L"upperPad"].Value<NDShape>());
            // lowerPad/upperPad is set to NDShape({0}) by default. 
            // If this node has explicitly set the lowerPad and upperPad values(i.e. nodes that are constructed with lowerPad/upperPad values and autoPadding=False),
            // export these values directly. Otherwise, check autoPadding and export accordingly. 
            bool isAllPadsZero = std::all_of(lowerPad.begin(), lowerPad.end(), [](int64_t i) { return i == 0; });
            isAllPadsZero = isAllPadsZero & std::all_of(upperPad.begin(), upperPad.end(), [](int64_t i) { return i == 0; });
            bool isAnyAutoPadTrue = std::any_of(autoPadding.begin(), autoPadding.end(), [](bool i) { return i; });
            if (lowerPad.size() > 0 && upperPad.size() > 0 
                && !(lowerPad.size() == 1 && upperPad.size() == 1 && lowerPad[0] == 0 && upperPad[0] == 0)
                && !(isAllPadsZero && ceilOutDim)
                && !isAnyAutoPadTrue)
            {
                if (ceilOutDim)
                    ValidatePadValueForCeilOutDim(lowerPad, upperPad, autoPadding, kernelShape, inputShape, strides,
                        /*dilation=*/std::vector<size_t>(kernelShape.Rank(), 1), /*transpose=*/false);
                lowerPad.insert(lowerPad.end(), upperPad.cbegin(), upperPad.cend());
                node->AddAttribute("pads", lowerPad);
            }
            else
            {
                PutPadAttrInNode(node, autoPadding, kernelShape, inputShape, strides, /*dilation=*/std::vector<size_t>(kernelShape.Rank(), 1), ceilOutDim, /*transpose=*/false);
            }
        }
        else if (src->OpName() == L"ReduceElements")
        {
            SetReduceElementsAttributes(src, node);
        }
    }
}

void CNTKToONNXHelper::SetReduceElementsAttributes(const FunctionPtr src, Node *node)
{
    std::wstring reductionOpName = src->OpName();
    if (reductionOpName == L"ReduceElements")
    {
        reductionOpName = src->Attributes()[L"reductionOpName"].Value<wstring>();
    }

    auto keepReducedDimensions = (int64_t)((bool)src->Attributes()[L"reductionKeepDimensions"].Value<bool>() ? 1 : 0);
    bool forceKeepReducedDimensions = false;

    std::vector<Axis> reductionAxes;
    if (src->Attributes().Contains(L"axisVec"))
        reductionAxes = AsVector<Axis>(src->Attributes()[L"axisVec"].Value<std::vector<DictionaryValue>>());
    else if (src->Attributes().Contains(L"axis"))
        reductionAxes.push_back((Axis)(src->Attributes()[L"axis"].Value<Axis>()));

    // Reduction on batch axis in CNTK removes the batch axis, even if keepdims is true. 
    // For ONNX export we need to make sure we export keepdims as 0 (false). 
    // The same applies for AllStaticAxes. 
    if (!forceKeepReducedDimensions &&
        (reductionAxes.size() == 1
            && (reductionAxes[0] == Axis::DefaultBatchAxis()
                || reductionAxes[0] == Axis::AllStaticAxes()
                || reductionAxes[0] == Axis::AllAxes())))
        keepReducedDimensions = 0;
    std::vector<int64_t> axes = ConvertAxesToOnnx(reductionAxes, src->Inputs()[0]);

    if (reductionOpName == L"Argmax" || reductionOpName == L"Argmin")
        node->AddAttribute("axis", axes[0]);
    else
        if (reductionAxes[0] != Axis::AllAxes())
            node->AddAttribute("axes", axes); 

    node->AddAttribute("keepdims", keepReducedDimensions);
}

void CNTKToONNXHelper::ValidatePadValueForCeilOutDim(const std::vector<int64_t> lowerPad, const std::vector<int64_t> upperPad, const std::vector<bool>& autoPadding,
    const NDShape& kernelShape, const NDShape& inputShape, const NDShape& strides, const NDShape& dilations, bool transpose)
{
    auto padsValueVectorsForONNX = GetONNXPadsAttributeFromCNTKNode(autoPadding, kernelShape, inputShape, strides, dilations, /*ceilOutDim=*/true, transpose);
    auto onnxLowerPads = ToINTS(padsValueVectorsForONNX.first);
    auto onnxUpperPads = ToINTS(padsValueVectorsForONNX.second);

    for (int i=0; i < onnxLowerPads.size(); ++i)
    {
        if (i >= lowerPad.size() || i >= upperPad.size() || (lowerPad[i] + upperPad[i] < onnxLowerPads[i] + onnxUpperPads[i]))
            LogicError("Convolution/Convolution Transpose/Pooling. Pads value provided is not enough when ceilOutDim is true. ");
    }
}


void CNTKToONNXHelper::PutPadAttrInNode(onnxruntime::Node* node,
    const std::vector<bool>& autoPadding, const NDShape& kernelShape, const NDShape& inputShape, const NDShape& strides,
    const NDShape& dilations, bool ceilOutDim, bool transpose)
{
    // To fully support CNTK exporting of convolution & pooling ops for all input settings,
    // The padding attributes must be exported in 'pads' instead of 'autoPad'.
    // When padding is required, ONNX spec for autoPad has two choices of either "SAME_UPPER" or "SAME_LOWER".
    // However in CNTK it is possible that one op has dimensions which exploits both options. 
    // E.g.
    //   operand shape: [7, 8], kernel shape: [2, 3], strides: [2, 2].
    // The pad values CNTK generates is [0, 1, 1, 0]. This cannot be expressed in one single "SAME_UPPER" nor "SAME_LOWER". 
    auto padsValueVectorsForONNX = GetONNXPadsAttributeFromCNTKNode(autoPadding, kernelShape, inputShape, strides, dilations, ceilOutDim, transpose);
    auto lowerPads = ToINTS(padsValueVectorsForONNX.first);
    auto upperPads = ToINTS(padsValueVectorsForONNX.second);
    lowerPads.insert(lowerPads.end(), upperPads.cbegin(), upperPads.cend());
    node->AddAttribute("pads", lowerPads);
}

std::vector<onnxruntime::NodeArg *> CNTKToONNXHelper::MapInputsOrderToONNX(const FunctionPtr& src, const std::vector<onnxruntime::NodeArg *>& inputs)
{
    if (Operators::HasInputIndexMap(src->OpName()))
    {
        std::vector<onnxruntime::NodeArg *> orderedInputs;
        std::map<int, onnxruntime::NodeArg *> orderedInputsMap;
        auto map = Operators::ToONNXInputIndexMap(src->OpName());

        for (size_t inputIndex = 0; inputIndex < inputs.size(); ++inputIndex)
        {
            if (map[inputIndex] >= 0)
                orderedInputsMap.insert(std::pair<int, onnxruntime::NodeArg *>(map[inputIndex], inputs[inputIndex]));
        }

        for (const auto &item : orderedInputsMap)
            orderedInputs.push_back(item.second);

        return orderedInputs;
    }

    return inputs;
}

onnxruntime::Node* FindByName(onnxruntime::Graph* graph, const std::string &name)
{
    GraphNodes &nodes = graph->Nodes();

    for (onnxruntime::GraphNodes::MutableNodeIterator it = nodes.begin(); it != nodes.begin(); ++it)
    {
        onnxruntime::Node &node = *it;

        auto outputNodeArgs = node.OutputDefs();
        for (int i = 0; i < outputNodeArgs.size(); i++)
        {
            if (outputNodeArgs[i]->Name() == name)
            {
                return &node;
            }
        }
    }
    return nullptr;
}

std::vector<int64_t> GetShapeFromNodeArg(onnxruntime::NodeArg *nodeArg)
{
    std::vector<int64_t> shape;
    const TypeProto *typeProto = nodeArg->TypeAsProto();
    for (int dim = 0; dim < typeProto->tensor_type().shape().dim_size(); dim++)
    {
        shape.push_back(typeProto->tensor_type().shape().dim()[dim].dim_value());
    }
    return shape;
}

// CNTK splice allows broadcast of inputs before applying concatenation.
// ONNX Concat is limited to matching input shape cases
// i.e. inputs' dimensions shall be the equal except for the concatenation axis.
// for an example, see test_Concat_With_Broadcast in onnx_op_test.py.
// This function broadcasts the inputs for axes excluding ignoreAxes.
// Returns the broadcasted shape.
std::vector<int64_t> CNTKToONNXHelper::BroadcastInputs(std::vector<onnxruntime::NodeArg *> &orderedInputs, const std::set<int64_t> &ignoreAxes,
    const FunctionPtr& src, onnxruntime::Graph* graph)
{
    std::vector<std::vector<int64_t>> shapes;
    int max_rank = 0;
    for (auto nodeArg : orderedInputs)
    {
        shapes.push_back(GetShapeFromNodeArg(nodeArg));
        max_rank = std::max(max_rank, shapes.rbegin()->size());
    }

    std::vector<int64_t> broadcast_shape(max_rank, 1);
    for (int i = 0; i < shapes.size(); i++)
    {
        std::vector<int64_t> &shape_i = shapes[i];
        for (int index_to_shape_i = 0; index_to_shape_i < shape_i.size(); index_to_shape_i++)
        {
            int onnx_axis = index_to_shape_i + (max_rank - shape_i.size());
            if (ignoreAxes.find(onnx_axis) != ignoreAxes.end())
                // only check and update non ignoreAxes dimensions
                continue;
            else if (broadcast_shape[onnx_axis] == 1)
                broadcast_shape[onnx_axis] = shape_i[index_to_shape_i];
            else if (broadcast_shape[onnx_axis] != shape_i[index_to_shape_i] && shape_i[index_to_shape_i] != 1)
                LogicError("Invalid broadcast inputs shape");
        }
    }

    // TODO: use ONNX Expand once ONNX version 7 is supported
    // Without Expand op, we create a zeros constant of expected shape and apply broadcast add
    // to get input to the right shape for concatination.    
    for (int i = 0; i < orderedInputs.size(); i++)
    {
        std::vector<int64_t> &shape_i = shapes[i];
        bool need_broadcast = shape_i.size() < max_rank;
        while (shape_i.size() < max_rank)
            shape_i.insert(shape_i.begin(), 1);

        for (int onnx_axis = 0; onnx_axis < shape_i.size(); onnx_axis++)
        {
            if (ignoreAxes.find(onnx_axis) == ignoreAxes.end() && shape_i[onnx_axis] != broadcast_shape[onnx_axis])
            {
                shape_i[onnx_axis] = broadcast_shape[onnx_axis];
                need_broadcast = true;
            }
        }

        if (!need_broadcast)
            continue;

        onnxruntime::NodeArg *nodeArg = orderedInputs[i];

        // We insert an "Add" with broadcast to get desired shape that can be accepted by ONNX Concat. 
        onnxruntime::NodeArg &nodeArg2 = AddZerosConstantNodeArg(graph, nodeArg->Name() + "_braodcast_for_desired_shape",
            shape_i, src->Inputs()[i].GetDataType());
        const std::string out_arg_name = nodeArg->Name() + "_post_braodcasted_with_desired_shape";
        onnxruntime::Node *node = AddAddNode(*nodeArg, nodeArg2, graph, out_arg_name);
        orderedInputs[i] = const_cast<NodeArg*>(node->OutputDefs()[0]);
    }

    return broadcast_shape;
}

onnxruntime::Node* CNTKToONNXHelper::AddNode(const FunctionPtr& src, onnxruntime::Graph* graph, const std::vector<onnxruntime::NodeArg *>& inputs, const std::vector<onnxruntime::NodeArg *>& outputs)
{
    onnxruntime::Node* node = nullptr;
    std::vector<onnxruntime::NodeArg *> orderedInputs = MapInputsOrderToONNX(src, inputs);
    auto nodeName = ToLegacyString(ToUTF8(src->Uid()));

    if (L"Embedding" == src->OpName())
    {
        onnxruntime::Node* argMatMul = AddMatMulNode(*orderedInputs[1], *orderedInputs[0], graph, outputs[0]->Name());
    }
    else
    {
        //
        // CNTK Times OP is way more flexible for ONNX, so depend on the inputs and output shape,
        // we will need to insert some reshapes.
        //
        if (src->OpName() == L"Times")
        {
            auto input1Shape = orderedInputs[0]->Shape();
            auto input2Shape = orderedInputs[1]->Shape();
            auto outputShape = outputs[0]->Shape();

            int input1Rank = input1Shape->dim_size();
            int input2Rank = input2Shape->dim_size();
            int outputRank = outputShape->dim_size();
            int reductionRank = (input1Rank + input2Rank - outputRank) / 2;
            // Currently we don't support outputRank > 1. Thus input1 shape has format [(1), a, outputRank],
            // where (1) is the optional batch axis and a the axis correspond to outputRank = 1.
            // When we support outputRank, the flag will be defined as (input1Rank - reductionRank - outputRank) == 1.
            bool input1HasBatchAxis = (input1Rank - reductionRank) == 2;
            bool input2HasBatchAxis = (input2Rank - reductionRank) == 2;

            if (reductionRank > 1) // We need to insert reshape.
            {
                onnx::TypeProto input1Reshape = ReduceRank(input1Shape, reductionRank, true, input1HasBatchAxis);
                onnx::TypeProto input2Reshape = ReduceRank(input2Shape, reductionRank, false, input2HasBatchAxis);

                UpdateONNXType(src->Inputs()[1].GetDataType(), input1Reshape);
                UpdateONNXType(src->Inputs()[0].GetDataType(), input2Reshape);

                onnxruntime::NodeArg &inputOutput1Arg = graph->GetOrCreateNodeArg(orderedInputs[0]->Name() + string("_reshape0"), &input1Reshape);
                onnxruntime::NodeArg &inputOutput2Arg = graph->GetOrCreateNodeArg(orderedInputs[1]->Name() + string("_reshape1"), &input2Reshape);

                //auto reshapeNode1 = graph->AddNode(nodeName + string("_reshape0"), "Reshape", "", {orderedInputs[0]}, {&inputOutput1Arg});
                //auto reshapeNode2 = graph->AddNode(nodeName + string("_reshape1"), "Reshape", "", {orderedInputs[1]}, {&inputOutput2Arg});

                //reshapeNode1->AddAttribute("shape", ToINTS(input1Reshape));
                //reshapeNode2->AddAttribute("shape", ToINTS(input2Reshape));

                AddReshapeNodeImpl(graph, nodeName + "_reshape0", orderedInputs[0], &inputOutput1Arg, ToINTS(input1Reshape));
                AddReshapeNodeImpl(graph, nodeName + "_reshape1", orderedInputs[1], &inputOutput2Arg, ToINTS(input2Reshape));

                node = graph->AddNode(nodeName, ToOPName(src), "", {&inputOutput1Arg, &inputOutput2Arg}, outputs);
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
            const size_t operandIndexInOnnxInputs = 2;                        // ONNX input indices don't change because we have already filtered epsilon input from ONNX inputs in CreateNode() above.
            const size_t scaleIndexInOnnxInputs = 0;
            const size_t biasIndexInOnnxInputs = 1;

            auto input0 = inputs[operandIndexInOnnxInputs];
            onnx::TypeProto input0ArgType = ToTypeProto(src->Inputs()[operandIndexInCntkInputs].Shape(), src->Inputs()[operandIndexInCntkInputs].HasBatchAxis());
            UpdateONNXType(src->Inputs()[operandIndexInCntkInputs].GetDataType(), input0ArgType);
            onnxruntime::NodeArg &mvnTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_mvn_output0"), &input0ArgType);
            onnxruntime::Node* mvnNode = graph->AddNode(nodeName + string("_MVN"), "MeanVarianceNormalization",
                                                   "", { input0 }, { &mvnTensorOutputArg });
            mvnNode->AddAttribute("across_channels", static_cast<int64_t>(1));
            mvnNode->AddAttribute("normalize_variance", static_cast<int64_t>(1));

            auto input1 = inputs[scaleIndexInOnnxInputs];
            onnxruntime::NodeArg &mulTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_mul_output0"), &input0ArgType);
            onnxruntime::Node* mulNode = graph->AddNode(nodeName + string("_mul"), "Mul",
                                                   "", { &mvnTensorOutputArg, input1 }, { &mulTensorOutputArg });

            auto input2 = inputs[biasIndexInOnnxInputs];
            onnxruntime::NodeArg &addTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_Output_0"), &input0ArgType);
            node = graph->AddNode(nodeName + string("_add"), "Add",
                                  "", { &mulTensorOutputArg, input2 }, { &addTensorOutputArg });
        }
        else if (src->OpName() == L"Splice")
        {
            Axis axis = (Axis)(src->Attributes()[L"axis"].Value<Axis>());
            BroadcastInputs(orderedInputs, { ConvertAxisToOnnxBroadcastOfOp(axis, src) }, src, graph);
            node = graph->AddNode(nodeName, ToOPName(src), "", orderedInputs, outputs);
        }
        else if (src->OpName() == L"LogPlus")
        {
            // CNTK LogPlus is the equivalent to numpy.logaddexp
            // ONNX has a different but similar op: ReduceLogSumExp
            onnx::TensorProto_DataType tensorType = orderedInputs[0]->TypeAsProto()->tensor_type().elem_type();
            std::vector<int64_t> broadcastShape = BroadcastInputs(orderedInputs, /*ignoreAxes=*/{}, src, graph);
            // Now both inputs should have the same shape.
            // Add another axis in front. This will be the axis to be reduced over later.
            std::vector<int64_t> unsqueezeOutputShape = broadcastShape;
            unsqueezeOutputShape.insert(unsqueezeOutputShape.begin(), 1);
            std::vector<int64_t> concatOutputShape = broadcastShape;
            concatOutputShape.insert(concatOutputShape.begin(), 2);

            auto unsqueezeInputFunc = [&](int inputIndex) -> onnxruntime::NodeArg& {
                onnx::TypeProto outputArgType = ToTypeProto(unsqueezeOutputShape);
                outputArgType.mutable_tensor_type()->set_elem_type(tensorType);
                onnxruntime::NodeArg &unsqueezeTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_unsqueeze" + std::to_string(inputIndex) + "_output0"), &outputArgType);
                onnxruntime::Node* unsqueezeNode = graph->AddNode(nodeName + string("_Unsqueeze") + std::to_string(inputIndex), "Unsqueeze", "", { orderedInputs[inputIndex] }, { &unsqueezeTensorOutputArg });
                unsqueezeNode->AddAttribute("axes", std::vector<int64_t>(1, 0));
                return unsqueezeTensorOutputArg;
            };

            onnxruntime::NodeArg &unsqueezeTensorOutputArg0 = unsqueezeInputFunc(0);
            onnxruntime::NodeArg &unsqueezeTensorOutputArg1 = unsqueezeInputFunc(1);

            onnx::TypeProto concatOutputArgType = ToTypeProto(concatOutputShape);
            concatOutputArgType.mutable_tensor_type()->set_elem_type(tensorType);
            onnxruntime::NodeArg &concatTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_concat_output0"), &concatOutputArgType);
            onnxruntime::Node* concatNode = graph->AddNode(nodeName + string("_Concat"), "Concat", "", { &unsqueezeTensorOutputArg0, &unsqueezeTensorOutputArg1 },
                { &concatTensorOutputArg });
            concatNode->AddAttribute("axis", static_cast<int64_t>(0));

            onnx::TypeProto outputArgType = ToTypeProto(broadcastShape);
            outputArgType.mutable_tensor_type()->set_elem_type(tensorType);
            onnxruntime::NodeArg &reduceLogSumExpTensorOutputArg = graph->GetOrCreateNodeArg(nodeName + string("_Output_0"), &outputArgType);
            node = graph->AddNode(nodeName + string("_reduce_log_sum_exp"), "ReduceLogSumExp", "", { &concatTensorOutputArg }, { &reduceLogSumExpTensorOutputArg });
            // reduce over the first axis.
            node->AddAttribute("axes", std::vector<int64_t>(1, 0));
            node->AddAttribute("keepdims", static_cast<int64_t>(0));
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

std::pair<std::vector<int>, std::vector<int>> CNTKToONNXHelper::GetONNXPadsAttributeFromCNTKNode(
    const std::vector<bool>& cntkAutoPadding, const NDShape& kernelShape, const NDShape& inputShape, const NDShape& strides, 
    const NDShape& dilations, bool ceilOutDim, bool transpose)
{
    // Reuse ConvolveGeometry to compute outputShape and pad values. 
    // The difference here is that ConvolveGeometry expects parameters to be non-spatial shapes, which includes the channel info that 
    // we have just excluded. Thus emulated channel axis is inserted.
    assert(inputShape.Rank() > 0);
    const size_t channelSize = inputShape[inputShape.Rank() - 1];
    const NDShape& kernelShapeWithChannel = kernelShape.AppendShape({ channelSize });
    const NDShape& stridesWithChannel = strides.Rank() == 1 ? strides : strides.AppendShape({ channelSize });
    const NDShape& dilationsWithChannel = dilations.Rank() == 1 ? dilations : dilations.AppendShape({ 1 });
    std::vector<bool> cntkAutoPaddingWithChannel = cntkAutoPadding.size() > 0 ? cntkAutoPadding : std::vector<bool>({ false });
    if (cntkAutoPaddingWithChannel.size() > 1 && cntkAutoPaddingWithChannel.size() < inputShape.Rank())
    {
        while (cntkAutoPaddingWithChannel.size() < inputShape.Rank() - 1)
            cntkAutoPaddingWithChannel.push_back(cntkAutoPaddingWithChannel[0]);
        cntkAutoPaddingWithChannel.push_back(false);
    }

    NDShape convOperandShape = inputShape;
    if (transpose)
    {
        // For convolution transpose. Reuse logic in ConvolveGeometry to compute the actual pads value. 
        // First, get CNTK convTranspose outputShape by ConvolveGeometry::ComputeInputShape.
        // Next, treat this as normal convolution, and use the achieved outputShape as inputShape, to compute the pads values. 
        convOperandShape = AsNDShape(ConvolveGeometry::ComputeInputShape(AsTensorShape(inputShape), AsTensorShape(kernelShapeWithChannel),
            /*mapCount=*/AsTensorShape({ 1 }), AsTensorShape(stridesWithChannel), /*sharing=*/std::vector<bool>({ true }), cntkAutoPaddingWithChannel, 
            /*lowerPad=*/AsTensorShape({ 0 }), /*upperPad=*/AsTensorShape({ 0 }), AsTensorShape(dilationsWithChannel), /*groups=*/1, 
            ceilOutDim, /*(UNUSED)needsDynamicValidation=*/false, /*(UNUSED)isFinalValidationPass=*/false));
    }

    auto geometry = std::make_shared<ConvolveGeometry>(AsTensorShape(convOperandShape), AsTensorShape(kernelShapeWithChannel),
        /*mapCount=*/AsTensorShape({1}), AsTensorShape(stridesWithChannel), /*sharing=*/std::vector<bool>({true}), cntkAutoPaddingWithChannel, /*lowerPad=*/AsTensorShape({0}),
        /*upperPad=*/AsTensorShape({0}), AsTensorShape(dilationsWithChannel), ceilOutDim, /*groups=*/1);

    // Figure out the value for 'pads' ONNX attribute.
    std::vector<int> padsValueVectorLower(kernelShape.Rank(), 0);
    std::vector<int> padsValueVectorUpper(kernelShape.Rank(), 0);
    for (size_t i = 0; i < kernelShape.Rank(); ++i)
    {
        int upperPad = 0;
        int lowerPad = 0;
        if ((i >= cntkAutoPadding.size() && !cntkAutoPadding[cntkAutoPadding.size() - 1]) || (i < cntkAutoPadding.size() && !cntkAutoPadding[i]))
        {
            // When autoPadding is False
            if (ceilOutDim)
            {
                // This is a special case handling for Pooling, where ceilOutDim = True and cntkAutoPadding = False.
                // In CNTK, no paddings should be generated since autoPadding is False. 
                // Yet due to ceilOutDim = True, the outputShape might end up 1 size larger, requiring
                // an input of dimension that actually exceeds the current input. 
                // E.g.
                //      input dim: 112, kernel size: 3, stride: 2
                // The output dim will end up 56. 
                // This will require an input dim of 113. 
                // In CNTK, this issue is covered in ConvolveGeometry, where the mapping from output cell to input index,
                // namely MpRowCol/MpRowIndices is computed. 
                // In short, zeros are appended at upper at compute time. 
                padsValueVectorUpper[i] = geometry->GetExtraCellsCount(i);
            }
            else
            {
                continue;
            }
        }
        else
        {
            // When autoPadding is True
            padsValueVectorLower[i] = geometry->GetLowerPad(i);
            padsValueVectorUpper[i] = geometry->GetUpperPad(i);
            if (padsValueVectorLower[i] < 0)
                padsValueVectorLower[i] = 0;
            if (padsValueVectorUpper[i] < 0)
                padsValueVectorUpper[i] = 0;
        }
    }
    return std::make_pair(padsValueVectorLower, padsValueVectorUpper);
}

void CNTKToONNXHelper::FillTensorWithScalar(const std::vector<NDArrayViewPtr> &srcs,
                                            onnx::TensorProto& dst, const std::vector<int64_t> dstShape)
{
    auto dataType = srcs[0]->GetDataType();
    SetTensorType(dst, dataType);
    // the first dimension is for srcs count
    int64_t eachSrcSize = std::accumulate(dstShape.begin() + 1, dstShape.end(), (int64_t)1, std::multiplies<int64_t>());
    switch (dataType)
    {
    case CNTK::DataType::Float:
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (int index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_float_data()->Add()) = scalar;
            }
        }

        break;
    }
    case CNTK::DataType::Float16:
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            auto scalar = reinterpret_cast<const uint16_t*>(srcTemp->DataBuffer<float16>());

            for (int index = 0; index < eachSrcSize; index++)
            {
                *(dst.mutable_int32_data()->Add()) = *scalar;
            }
        }

         break;
    }
    case CNTK::DataType::Double:
    {
        for (int i = 0; i < srcs.size(); i++)
        {
            auto srcTemp = srcs[i]->DeepClone();
            srcTemp->ChangeDevice(DeviceDescriptor::CPUDevice());
            float scalar = *srcTemp->DataBuffer<float>();

            for (int index = 0; index < eachSrcSize; index++)
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

onnxruntime::NodeArg& CNTKToONNXHelper::CreateScalarNode(Graph *graph, const string &nodeArgName,
                                                         CNTK::DataType dataType, double value)
{
    // Notes:
    // 1. Although we intend to create a scalar node, which should ideally be zero-dimensional,
    //    we are creating a one-dimensional tensor with dim size 1. This is because that it is
    //    possible that some machinery (such as broadcast for binary ops) may not handle empty 
    //    tensor shape (zero-dimensional tensor) well.
    // 2. We take the value of the scalar as a double type, but then typecast it to the suitable
    //    type based on the CNTK::DataType provided as input.
    onnx::TypeProto argTypeProto = ToTypeProto(std::vector<int64_t>({ 1 }), false);
    argTypeProto.mutable_tensor_type()->set_elem_type(ConvertDataTypeCNTKToTensorProto(dataType));
    onnxruntime::NodeArg &inputNodeArg = graph->GetOrCreateNodeArg(nodeArgName, &argTypeProto);

    onnx::TensorProto dstTensor;
    dstTensor.set_name(inputNodeArg.Name());
    dstTensor.set_data_type(ConvertDataTypeCNTKToTensorProto(dataType));

    switch (dataType)
    {
    case CNTK::DataType::Float16:
        dstTensor.mutable_int32_data()->Add(static_cast<int>(value));
        break;
    case CNTK::DataType::Float:
        dstTensor.mutable_float_data()->Add(static_cast<float>(value));
        break;
    case CNTK::DataType::Double:
        dstTensor.mutable_double_data()->Add(value);
        break;
    default:
        NOT_IMPLEMENTED;
    }

    *(dstTensor.mutable_dims()->Add()) = static_cast<int64_t>(1);
    graph->AddInitializedTensor(dstTensor);
    return inputNodeArg;
}

onnxruntime::Node* CNTKToONNXHelper::CreateONNXNodesForStraightThrough(const FunctionPtr &src,
                                                                    onnxruntime::Graph* graph,
                                                                    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                                                    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                                                    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    // This method exports CNTK's StraighThrough estimator op through an ONNX sub-graph. 
    // ONNX subgraph consists of Greater + Cast + Mul + Sub ops. 
    std::vector<onnxruntime::NodeArg*> inputs;
    ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);

    std::vector<onnxruntime::NodeArg*> outputs;
    ProcessOutputs(src, outputs, graph);

    const std::string& outputName = outputs[0]->Name();

    onnxruntime::NodeArg& scalarZeroOutputArg = CreateScalarNode(graph, outputName + "_zero_out", src->Inputs()[0].GetDataType(), 0.0);
    onnxruntime::NodeArg& greaterOutputArg = graph->GetOrCreateNodeArg(outputName + "_greater_out", nullptr);
    onnxruntime::Node* greaterNode = graph->AddNode(outputName + "_greater", "Greater", "", { inputs[0], &scalarZeroOutputArg }, { &greaterOutputArg });

    onnxruntime::NodeArg& castOutputArg = graph->GetOrCreateNodeArg(outputName + "_cast_out", nullptr);
    onnxruntime::Node* castNode = graph->AddNode(outputName + "_cat", "Cast", "", { &greaterOutputArg }, { &castOutputArg });
    castNode->AddAttribute("to", static_cast<int64_t>(ConvertDataTypeCNTKToTensorProto(src->Inputs()[0].GetDataType())));

    onnxruntime::NodeArg& scalarTwoOutputArg = CreateScalarNode(graph, outputName + "_two_out", src->Inputs()[0].GetDataType(), 2.0);

    onnxruntime::NodeArg& mulOutputArg = graph->GetOrCreateNodeArg(outputName + "_mul_out", nullptr);
    onnxruntime::Node* mulNode = graph->AddNode(outputName + "_mul", "Mul", "", { &castOutputArg, &scalarTwoOutputArg }, { &mulOutputArg });

    onnxruntime::NodeArg& scalarOneOutputArg = CreateScalarNode(graph, outputName + "_one_out", src->Inputs()[0].GetDataType(), 1.0);
    onnxruntime::NodeArg& subOutputArg = graph->GetOrCreateNodeArg(outputName + "_sub_out", nullptr);
    onnxruntime::Node* subNode = graph->AddNode(outputName + "_sub", "Sub", "", { &mulOutputArg, &scalarOneOutputArg }, { outputs[0] });

    functionNodes.emplace(src, subNode);
    return subNode;
}

onnxruntime::Node* CNTKToONNXHelper::CreateONNXNodesForOptimizedRNNStack(const FunctionPtr &src,
                                                                    onnxruntime::Graph* graph,
                                                                    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
                                                                    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
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
    float *Wdata = srcTemp->WritableDataBuffer<float>();
    Matrix<float> Wm(WcombinedShape[0], WcombinedShape[1], Wdata, CPUDEVICE, MatrixType::DENSE, MatrixFormat::matrixFormatDense);

    // Step 2: Extract individual weight and bias matrices for each layer from the big weight matrix.
    std::vector<NDArrayViewPtr> W, R, B;
    std::tie<std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>>(W, R, B) = SplitOptimzedRnnWtoIndivMats(Wm, numLayers, inputSize, hiddenSize, bidirectional, recurrentOp);

    // Step 3: Create ONNX nodes mirroring the implementation of OptimizedRNNStack.
    onnxruntime::Node *functionNode = nullptr;
    bool inputNeedsShapeAdapter(false);
    auto ornnInput = src->Inputs()[0]; // CNTK OptimizedRNNStack node's input operand.

    if (ornnInput.Owner().get() != nullptr)
        CreateNode(ornnInput.Owner(), graph, functionNodes, variableNodes, compositeOutputsMap);

    auto ornnInputArgType = ToTypeProto(ornnInput.Shape(), ornnInput.HasBatchAxis(), ornnInput.HasSequenceAxis());
    UpdateONNXType(ornnInput.GetDataType(), ornnInputArgType);
    auto ornnOutput = src->Outputs()[0];
    auto outArgType1 = ToTypeProto({ornnOutput.Shape()[0] / numDirections, numDirections}, ornnOutput.HasBatchAxis(), ornnOutput.HasSequenceAxis());
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
    std::string ornnInputName = ToLegacyString(ToUTF8(ornnInput.Uid()));
    auto inputItr = compositeOutputsMap.find(ornnInput);
    if (inputItr != compositeOutputsMap.end())
        ornnInputName = ToLegacyString(ToUTF8(inputItr->second.Uid()));

    // Create ONNX LSTM layers
    onnxruntime::NodeArg *layerInputOperandArg = &graph->GetOrCreateNodeArg(ornnInputName, &ornnInputArgType);
    for (size_t i = 0; i < numLayers; ++i)
    {
        std::vector<onnxruntime::NodeArg *> inputs;
        std::vector<onnxruntime::NodeArg *> outputs;

        // ==== Step 4. Create input nodes =====
        // Input operand X
        if (inputNeedsShapeAdapter)
        {
            std::string adapterBasename = ToLegacyString(ToUTF8(src->Uid())) + "_Adapter_" + std::to_string(i);
            onnxruntime::NodeArg* shapeAdaptedInputOperandArg = LSTMOutputShapeAdapter(*layerInputOperandArg, ornnOutputArgType, graph,
                                                                                  numDirections, hiddenSize, ornnOutput.GetDataType(), adapterBasename);
            inputs.push_back(shapeAdaptedInputOperandArg);
        }
        else
            inputs.push_back(layerInputOperandArg);

        // Create node for input weight tensor W
        auto WArgName = ToLegacyString(ToUTF8(Wcombined.Uid())) + "_W_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, W[i], WArgName);
        // Create node for input weight tensor R (equivalent to CNTK's H)
        auto RArgName = ToLegacyString(ToUTF8(Wcombined.Uid())) + "_R_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, R[i], RArgName);
        // Create node for input bias tensor B
        auto BArgName = ToLegacyString(ToUTF8(Wcombined.Uid())) + "_B_" + std::to_string(i);
        CreateRecurrentWeightONNXNodes(graph, variableNodes, Wcombined, inputs, B[i], BArgName);

        // ==== Step 5. Create output nodes =====
        // For now, we always output Y. So this attribute value is 1.
        int64_t outputSequence = 1;
        //Note: Important to keep the output arg name the same.
        auto outArgName = (i == numLayers - 1) ? ToLegacyString(ToUTF8(ornnOutput.Uid())) : ToLegacyString(ToUTF8(ornnOutput.Uid())) + "_" + std::to_string(i);
        onnxruntime::NodeArg &outputArg_Y = graph->GetOrCreateNodeArg(outArgName, &ornnOutputArgType);
        outputs.push_back(&outputArg_Y);

        // ==== Step 6. Add ONNX LSTM node ====
        auto rnnOpNameLookup = Operators::OptimizedRnnToOnnxOpLookup();
        auto rnnNodeName = ToLegacyString(ToUTF8(src->Uid())) + std::to_string(i);
        functionNode = graph->AddNode(rnnNodeName, rnnOpNameLookup[recurrentOp], "", inputs, outputs);

        std::vector<std::string> singleDirectionActivation;
        if (recurrentOp == L"lstm")
            singleDirectionActivation = {"Sigmoid", "Tanh", "Tanh"};
        else if (recurrentOp == L"rnnReLU")
            singleDirectionActivation = {"Relu"};
        else if (recurrentOp == L"rnnTanh")
            singleDirectionActivation = {"Tanh"};
        std::vector<std::string> activations;
        activations.insert(activations.end(), singleDirectionActivation.begin(), singleDirectionActivation.end());
        if (bidirectional)
            activations.insert(activations.end(), singleDirectionActivation.begin(), singleDirectionActivation.end());
        functionNode->AddAttribute("activations", activations);
        functionNode->AddAttribute("direction", bidirectional ? "bidirectional" : "forward");
        functionNode->AddAttribute("hidden_size", (int64_t) hiddenSize);

        layerInputOperandArg = &outputArg_Y; // Output of this layer is the input to the next layer in the loop.
        inputNeedsShapeAdapter = true;       // To enable shape adapter to allow stacking for next layer.
    }

    // this following code maps ONNX output tensor to CNTK output. 
    // [*, dir, #, H] -> [#, *][dir * H] 
    // NDShape::FreeDimension, numDirections, FreeBatchSize, hidden -> FreeBatchSize, NDShape::FreeDimension, numDirections * hidden
    int hidden = src->Output().Shape().Dimensions()[0] / numDirections;
    std::vector<int64_t> perm({ 2, 0, 1, 3 });
    auto inputArgs = functionNode->OutputDefs();
    std::vector<int64_t> transposeOutputShape({ FreeBatchSize, (int64_t)NDShape::FreeDimension, (int64_t)numDirections * hidden });

    onnx::TypeProto transposeOutputArgType = ToTypeProto(transposeOutputShape, false);
    UpdateONNXType(src->Output().GetDataType(), transposeOutputArgType);

    auto functionNodeTransposed = AddTransposeNode(const_cast<NodeArg &>(*inputArgs.at(0)), graph, perm, 
        transposeOutputArgType, inputArgs[0]->Name() + "transpose_out");
    auto transposedOutputArgs = functionNodeTransposed->OutputDefs();

    std::vector<int64_t> newShape = Cast<size_t, int64_t>(src->Output().Shape().Dimensions());
    newShape.insert(newShape.begin(), NDShape::FreeDimension);
    newShape.insert(newShape.begin(), FreeBatchSize);
    const std::string reshapedOutArgName = inputArgs.at(0)->Name() + "transposed_and_reshaped";
    auto functionNodeReshaped = AddReshapeNode(const_cast<NodeArg &>(*transposedOutputArgs.at(0)), newShape, reshapedOutArgName, graph);

    functionNodes.emplace(src, functionNodeReshaped);
    return functionNodeReshaped;
}

std::tuple<std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>, std::vector<NDArrayViewPtr>>
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

Matrix<float> CNTKToONNXHelper::GetBiasMatFromOrnnBigW(Matrix<float>&Wbig, size_t offset,
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

void CNTKToONNXHelper::CreateRecurrentWeightONNXNodes(onnxruntime::Graph* graph, std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
                                                      const Variable& Wcombined, std::vector<onnxruntime::NodeArg *>& inputs, NDArrayViewPtr W, string WArgName)
{
    auto WArgType = ToTypeProto(W->Shape(), false, false, false); // Last arg is false because we don't want shape reversal here.
    UpdateONNXType(Wcombined.GetDataType(), WArgType);
    onnxruntime::NodeArg &WArg = graph->GetOrCreateNodeArg(WArgName, &WArgType);
    inputs.push_back(&WArg);

    std::vector<onnxruntime::NodeArg *> varInputs;
    std::vector<onnxruntime::NodeArg *> varOutputs;

    varOutputs.push_back({&WArg});

    onnx::TensorProto dstTensor;
    dstTensor.set_name(WArgName);
    CopyTensor(W, dstTensor, &WArgType);

    graph->AddInitializedTensor(dstTensor);
}

onnxruntime::NodeArg* CNTKToONNXHelper::LSTMOutputShapeAdapter(onnxruntime::NodeArg& inputArg, onnx::TypeProto& inputArgType, onnxruntime::Graph* graph,
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
    onnxruntime::NodeArg &transposeOutputArg = graph->GetOrCreateNodeArg(adapterBasename + "_Transpose_Output", &transposeOutputArgType);
    auto transposeNode = graph->AddNode(adapterBasename + "_Transpose", "Transpose", "", { &inputArg }, { &transposeOutputArg });
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
    onnxruntime::NodeArg &reshapeOutputArg = graph->GetOrCreateNodeArg(adapterBasename + "_Reshape_Output", &reshapeOutputArgType);
    std::vector<int64_t> shape({ (int64_t)NDShape::FreeDimension, FreeBatchSize, (int64_t)(numDirections * hiddenSize) });
    AddReshapeNodeImpl(graph, adapterBasename + "_Reshape", &transposeOutputArg, &reshapeOutputArg, shape);
    return &reshapeOutputArg;
}

onnxruntime::Node* CNTKToONNXHelper::CreateONNXNodesForTimesTranspose(const FunctionPtr &src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<onnxruntime::NodeArg *> inputs; 
    ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);

    std::vector<onnxruntime::NodeArg *> outputs;
    ProcessOutputs(src, outputs, graph);

    const std::string & outputName = outputs[0]->Name();

    int rightInputRank = inputs[0]->Shape()->dim_size() - 1;

    onnxruntime::NodeArg &transposeOutputArg = graph->GetOrCreateNodeArg(outputName + "_transpose_out", nullptr);
    onnxruntime::Node* transposeNode = graph->AddNode(outputName + "_transpose", "Transpose", "", { inputs[0] }, { &transposeOutputArg });
    transposeNode->AddAttribute("perm", ToINTS(rightInputRank == 2 ? vector<int>({ 1, 2, 0 }) : vector<int>({ 0, 1 })));

    onnxruntime::NodeArg &matmulOutputArg = graph->GetOrCreateNodeArg(outputName + "_matmul_out", nullptr);
    onnxruntime::Node* matmulNode = graph->AddNode(outputName + "_matmul", "MatMul", "", { inputs[1], &transposeOutputArg }, { &matmulOutputArg });
    
    functionNodes.emplace(src, matmulNode);
    return matmulNode;
}

onnxruntime::Node* CNTKToONNXHelper::CreateONNXNodesForSelect(const FunctionPtr &src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<onnxruntime::NodeArg *> inputs;
    ProcessInputs(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);
    assert(inputs.size() == 3);

    std::vector<onnxruntime::NodeArg *> outputs;
    ProcessOutputs(src, outputs, graph);
    assert(outputs.size() == 1);

    // CNTK's select(flag, value_if_true, value_if_false) can be represented with ONNX ops as
    // flag01 * value_if_true + (1 - flag01) * value_if_false,
    // where flag01 = ceil(min(abs(flag), 1)).

    const std::string & outputName = outputs[0]->Name();

    onnxruntime::NodeArg &absOutputArg = graph->GetOrCreateNodeArg(outputName + "_abs_out", nullptr);
    graph->AddNode(outputName + "_abs", "Abs", "", { inputs[0] }, { &absOutputArg });

    // Add a Clip node equivalent to min(abs(flag), 1).
    onnxruntime::NodeArg &clipOutputArg = graph->GetOrCreateNodeArg(outputName + "_clip_out", nullptr);
    onnxruntime::Node* clipNode = graph->AddNode(outputName + "_clip", "Clip", "", { &absOutputArg }, { &clipOutputArg });
    clipNode->AddAttribute("min", 0.0f); // Should be unnecesary for ONNX, but currently required by CNTK.
    clipNode->AddAttribute("max", 1.0f);

    onnxruntime::NodeArg &ceilOutputArg = graph->GetOrCreateNodeArg(outputName + "_ceil_out", nullptr);
    graph->AddNode(outputName + "_ceil", "Ceil", "", { &clipOutputArg }, { &ceilOutputArg });

    onnxruntime::NodeArg &mulTrueOutputArg = graph->GetOrCreateNodeArg(outputName + "_mul_true_out", nullptr);
    graph->AddNode(outputName + "_mul_true", "Mul", "", { &ceilOutputArg, inputs[1] }, { &mulTrueOutputArg });

    onnxruntime::NodeArg &oneOutputArg = graph->GetOrCreateNodeArg(outputName + "_one_out", nullptr);
    onnxruntime::Node* oneNode = graph->AddNode(outputName + "_one", "Constant", "", {}, { &oneOutputArg });
    onnx::TensorProto oneTensor;
    oneTensor.set_data_type(onnx::TensorProto::FLOAT);
    oneTensor.add_float_data(1.0f);
    oneNode->AddAttribute("value", oneTensor);

    onnxruntime::NodeArg &oneSubOutputArg = graph->GetOrCreateNodeArg(outputName + "_one_sub_out", nullptr);
    graph->AddNode(outputName + "_sub_one", "Sub", "", { &oneOutputArg, &ceilOutputArg }, { &oneSubOutputArg });

    onnxruntime::NodeArg &mulFalseOutputArg = graph->GetOrCreateNodeArg(outputName + "_mul_false_out", nullptr);
    graph->AddNode(outputName + "_mul_false", "Mul", "", { &oneSubOutputArg, inputs[2] }, { &mulFalseOutputArg });

    onnxruntime::Node* sumNode = graph->AddNode(outputName + "_sum", "Sum", "", { &mulTrueOutputArg, &mulFalseOutputArg }, { outputs[0] });

    functionNodes.emplace(src, sumNode);
    return sumNode;
}

onnxruntime::Node* CNTKToONNXHelper::CreateNodeForBatchAxisOp(const FunctionPtr &src,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap)
{
    std::vector<onnxruntime::NodeArg *> inputs;
    ProcessInputsForBatchAxisOp(src, graph, functionNodes, variableNodes, compositeOutputsMap, inputs);

    std::vector<onnxruntime::NodeArg *> outputs;
    ProcessOutputsForBatchAxisOp(src, outputs, graph);

    auto batchOp = src->Inputs()[0].Owner();
    // Add a new node to ONNX graph.
    auto functionNode = AddNode(batchOp, graph, inputs, outputs);

    functionNodes.emplace(batchOp, functionNode);
    return functionNode;
}

void CNTKToONNXHelper::ProcessInputsForBatchAxisOp(const FunctionPtr& rootNode,
    onnxruntime::Graph* graph,
    std::unordered_map<FunctionPtr, onnxruntime::Node*>& functionNodes,
    std::unordered_map<Variable, onnxruntime::Node*>& variableNodes,
    const std::unordered_map<Variable, Variable>& compositeOutputsMap,
    std::vector<onnxruntime::NodeArg *>& inputs)
{
    // This method assumes that it has been validated already that 
    // rootNode is a batch axis op with the following pattern: 
    // "UnpackBatchAxis" (rootNode) --> Supported batch op (src) --> "ToBatchAxis" (topNode)
    FunctionPtr topNode = nullptr;
    Variable inputTopNode;
    size_t inputIdxToReplace(0u);
    auto src = rootNode->Inputs()[0].Owner();
    for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
    {
        auto input = src->Inputs()[inputIndex];
        if (input.IsOutput())
        {
            inputIdxToReplace = inputIndex;
            topNode = input.Owner();
        }
    }
    if (topNode != nullptr)
        inputTopNode = topNode->Inputs()[0];
    else
        LogicError("Invalid top node encountered when exporting CNTK op as ONNX batch axis op.");

    std::string cntkOpName = ToLegacyString(ToUTF8(src->OpName()));
    std::string onnxOpName = ToOPName(src);

    for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
    {
        Variable input = (inputIndex == inputIdxToReplace) ? inputTopNode : src->Inputs()[inputIndex];
        if (input.IsPlaceholder())
        {
            input = input.BlockFunctionVariableMapping();
            if (input.IsPlaceholder())
                LogicError("Node '%S': Placeholder isn't supported currently.", src->AsString().c_str());
        }

        if (FilterInput(src, input, inputIndex))
            continue;

        // Use user-defined name if available, otherwise use our internal unique name ID.
        std::string inputName = ToLegacyString(ToUTF8(input.Uid()));
        auto inputItr = compositeOutputsMap.find(input);
        if (inputItr != compositeOutputsMap.end())
            inputName = ToLegacyString(ToUTF8(inputItr->second.Uid()));

        bool isConstant = (input.IsParameter() || input.IsConstant()) &&
            !Operators::IgnoreConstantAndParameter(src->OpName(), inputIndex);

        onnx::TypeProto inputArgType;
        inputArgType = ToTypeProto(input.Shape(), false, false, true); // Explicitly turning off batch and sequence axis.
        if (input.IsInput() && input.HasSequenceAxis())
            (*inputArgType.mutable_tensor_type()->mutable_shape()->mutable_dim())[0].set_dim_param(FreeSequenceDimParam);

        // In case of BatchNormalization, if data (input[0]) is of type FP16, then all BN stats(inputs[1:4])
        // need to be converted from FP32 to FP16 prior to getting exported to ONNX.
        if (isConstant && cntkOpName == "BatchNormalization" && (inputIndex > 0 && inputIndex <= 4) && src->Inputs()[0].GetDataType() == DataType::Float16)
            input = Utils::ConvertVariableType<float, float16>(input, true);

        if (OpNeedONNXTypeMap(cntkOpName))
        {
            MapAndUpdateONNXType(onnxOpName, true, inputIndex, input.GetDataType(), &inputArgType); // TODO: Is this needed? Probably not.
        }
        else
        {
            UpdateONNXType(input.GetDataType(), inputArgType);
        }

        onnxruntime::NodeArg &inputArg = graph->GetOrCreateNodeArg(inputName, &inputArgType);
        inputs.push_back(&inputArg);

        //
        // Leaf nodes are data entry to the graph and need their own node with only output arg.
        //
        if (isConstant)
        {
            if (variableNodes.find(input) == variableNodes.end())
            {
                if (input.IsParameter() || input.IsConstant())
                {
                    auto srcTensor = input.IsParameter() ? Parameter(input).Value() : Constant(input).Value();

                    onnx::TensorProto dstTensor;
                    dstTensor.set_name(inputName);
                    CopyTensor(srcTensor, dstTensor, &inputArgType);

                    graph->AddInitializedTensor(dstTensor);
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
}

void CNTKToONNXHelper::ProcessOutputsForBatchAxisOp(const FunctionPtr& rootNode,
    std::vector<onnxruntime::NodeArg *>& outputs, Graph *graph)
{
    FunctionPtr topNode = nullptr;
    NDShape topNodeInputShape;
    auto src = rootNode->Inputs()[0].Owner();
    for (size_t inputIndex = 0; inputIndex < src->Inputs().size(); ++inputIndex)
    {
        auto input = src->Inputs()[inputIndex];
        if (input.IsOutput())
        {
            topNode = input.Owner();
        }
    }
    if (topNode != nullptr)
        topNodeInputShape = topNode->Inputs()[0].Shape();
    else
        LogicError("Invalid top node encountered when exporting CNTK op as ONNX batch axis op.");

    std::string onnxOpName = ToOPName(src);
    // TODO: Below line assumes that the first output of the op (e.g. conv)
    // is the one which UnpackBatch gets applied on. If some other output is
    // the one that is unpacked (i.e. gets passed on to UnpackBatchAxis) then 
    // this needs to be updated and we need to change this dynamically for 
    // different ops. 
    size_t outputIdxToReplace(0u); 
    for (size_t outputIndex = 0; outputIndex < src->Outputs().size(); ++outputIndex)
    {
        Variable output;
        NDShape outputShape;
        if (outputIndex == outputIdxToReplace)
        {   // This is the batch axis op's output that needs to be replaced with UnpackBatchAxis output.
            output = rootNode->Outputs()[0];
            outputShape = output.Shape().SubShape(0, output.Shape().Rank() - 1);
            outputShape = outputShape.AppendShape(topNodeInputShape.SubShape(topNodeInputShape.Rank() - 1, topNodeInputShape.Rank()));
        }
        else
        {
            output = src->Outputs()[outputIndex];
            outputShape = output.Shape();
        }

        auto outputArgType = ToTypeProto(outputShape, false, output.HasSequenceAxis(), true);
        if (OpNeedONNXTypeMap(onnxOpName))
        {
            MapAndUpdateONNXType(onnxOpName, false, 0, output.GetDataType(), &outputArgType); // TODO: Is this needed? Probably not.
        }
        else
        {
            UpdateONNXType(output.GetDataType(), outputArgType);
        }
        onnxruntime::NodeArg &outputNodeArg = graph->GetOrCreateNodeArg(ToLegacyString(ToUTF8(output.Uid())), &outputArgType);
        outputs.emplace_back(&outputNodeArg);
    }
}
