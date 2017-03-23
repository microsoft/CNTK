//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "BackCompat.h"
#include "PrimitiveFunction.h"
#include "CompositeFunction.h"
#include "ComputationNetworkBuilder.h"
#include "Utils.h"
#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "NonlinearityNodes.h"
#include "LinearAlgebraNodes.h"
#include "RecurrentNodes.h"
#include "EvaluationNodes.h"
#include "TrainingNodes.h"
#include "ReshapingNodes.h"
#include "DeprecatedNodes.h"
#include "RNNNodes.h"
#include "PreComputeNodes.h"
#include "DeprecatedNodes.h"
#include "SpecialPurposeNodes.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    namespace Internal
    {
        // Helper class to resolve variables in the model.
        class VariableResolver final
        {
            std::unordered_map<Variable, Variable> m_placeholderReplacements;
            std::unordered_map<ComputationNodeBasePtr, Variable> m_nodeToVariableMap;
            std::unordered_set<FunctionPtr> m_allPrimitiveFunctions;

        public:
            const std::unordered_map<Variable, Variable>& GetPlaceHolders() const
            {
                return m_placeholderReplacements;
            }

            template<class ElementType>
            Variable GetVariable(const ComputationNodeBasePtr& node)
            {
                auto iter = m_nodeToVariableMap.find(node);
                if (iter != m_nodeToVariableMap.end())
                    return iter->second;

                Variable var;
                if (node->IsLeaf())
                    var = ResolveLeaf<ElementType>(node);
                else
                {
                    // This is a non-leaf node and maps to a primitive Function
                    NDShape varShape = AsNDShape(node->GetSampleLayout());
                    auto placeholderVar = PlaceholderVariable(varShape);
                    m_nodeToVariableMap[node] = placeholderVar;

                    std::vector<Variable> inputVars(node->GetNumInputs());
                    for (size_t i = 0; i < inputVars.size(); ++i)
                    {
                        inputVars[i] = GetVariable<ElementType>(node->Input(i));
                        if (inputVars[i].IsPlaceholder())
                            m_placeholderReplacements[inputVars[i]] = Variable();
                    }

                    var = ResolveFunction<ElementType>(node, inputVars);

                    if (m_placeholderReplacements.find(placeholderVar) != m_placeholderReplacements.end())
                        m_placeholderReplacements[placeholderVar] = var;
                }

                m_nodeToVariableMap[node] = var;
                return var;
            }

        private:

            template<class ElementType>
            Variable CreateParameterOrConstantFromNodeValue(const ComputationNodeBasePtr& node, bool isConstant)
            {
                auto& matrix = node->As<ComputationNode<ElementType>>()->Value();
                auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorViewShape(node->GetSampleLayout()));
                NDArrayViewPtr value = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), AsNDShape(node->GetSampleLayout()), false, tensorView);

                auto kind = isConstant ? VariableKind::Constant : VariableKind::Parameter;

                std::wstring varUid, varName;
                std::tie(varUid, varName) = UidAndNameFromCNTKInternalNodeName(node->NodeName(), kind);

                return isConstant ? (Variable)Constant(value, varName, varUid) : Parameter(value, varName, varUid);
            }

            template<class ElementType>
            Variable ResolveLeaf(const ComputationNodeBasePtr& node)
            {
                NDShape variableShape = AsNDShape(node->GetSampleLayout());
                std::wstring varUid, varName;
                if (node->Is<InputValueBase<ElementType>>())
                {
                    std::tie(varUid, varName) = UidAndNameFromCNTKInternalNodeName(node->NodeName(), VariableKind::Input);

                    bool isSparse = node->Is<SparseInputValue<ElementType>>();
                    if (node->HasMBLayout())
                    {
                        // TODO: Currently only default dynamic axis is supported
                        auto inputNodeInternalDynamicAxisName = node->As<InputValueBase<ElementType>>()->GetRequestedDynamicAxis();
                        std::vector<Axis> inputVarDynamicAxes = DynamicAxesFromInternalDynamicAxisName(inputNodeInternalDynamicAxisName);

                        return Variable(variableShape, isSparse, AsDataType<ElementType>(), node->GetLearningRateMultiplier() != 0, varName, inputVarDynamicAxes, varUid);
                    }

                    // TODO: Allow creating inputs without a dynamic axis
                    LogicError("LoadLegacyModel: Found InputNode '%S' with no dynamic axes which is currently unsupported.", node->NodeName().c_str());
                }

                if (node->Is<LearnableParameter<ElementType>>())
                {
                    bool isConstant = (node->GetLearningRateMultiplier() == 0);
                    return CreateParameterOrConstantFromNodeValue<ElementType>(node, isConstant);
                }

                LogicError("LoadLegacyModel: Unsupported legacy CNTK node named '%S'.", node->NodeName().c_str());
                return Variable();// make compiler happy.
            }

            template<class ElementType>
            Variable ResolveFunction(const ComputationNodeBasePtr& node, std::vector<Variable>& inputVars)
            {
                PrimitiveOpType opType;
                Dictionary primitiveFunctionConfigParameters;
                if (node->OperationName() == OperationNameOf(NegateNode))
                    opType = PrimitiveOpType::Negate;
                else if (node->OperationName() == OperationNameOf(SigmoidNode))
                    opType = PrimitiveOpType::Sigmoid;
                else if (node->OperationName() == OperationNameOf(TanhNode))
                    opType = PrimitiveOpType::Tanh;
                else if (node->OperationName() == OperationNameOf(CosineNode))
                    opType = PrimitiveOpType::Cos;
                else if (node->OperationName() == OperationNameOf(SinNode))
                    opType = PrimitiveOpType::Sin;
                else if (node->OperationName() == OperationNameOf(PassNode))
                    opType = PrimitiveOpType::Pass;
                else if (node->OperationName() == OperationNameOf(LabelsToGraphNode))
                    opType = PrimitiveOpType::LabelsToGraph;
                else if (node->OperationName() == OperationNameOf(RectifiedLinearNode))
                    opType = PrimitiveOpType::ReLU;
                else if (node->OperationName() == OperationNameOf(ExpNode))
                    opType = PrimitiveOpType::Exp;
                else if (node->OperationName() == OperationNameOf(LogNode))
                    opType = PrimitiveOpType::Log;
                else if (node->OperationName() == OperationNameOf(SqrtNode))
                    opType = PrimitiveOpType::Sqrt;
                else if (node->OperationName() == OperationNameOf(FloorNode))
                    opType = PrimitiveOpType::Floor;
                else if (node->OperationName() == OperationNameOf(AbsNode))
                    opType = PrimitiveOpType::Abs;
                else if (node->OperationName() == OperationNameOf(ReciprocalNode))
                    opType = PrimitiveOpType::Reciprocal;
                else if (node->OperationName() == OperationNameOf(SoftmaxNode))
                    opType = PrimitiveOpType::Softmax;
                else if (node->OperationName() == OperationNameOf(HardmaxNode))
                    opType = PrimitiveOpType::Hardmax;
                else if (node->OperationName() == OperationNameOf(TransposeDimensionsNode))
                {
                    auto transposeDimensionsNode = node->As<TransposeDimensionsNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis1] = AsAxis(transposeDimensionsNode->Axis1());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis2] = AsAxis(transposeDimensionsNode->Axis2());

                    opType = PrimitiveOpType::TransposeAxes;
                }
                else if (node->OperationName() == OperationNameOf(WhereNode))
                {
                    auto internalDynamicAxisName = node->As<WhereNode<ElementType>>()->DynamicAxisName();
                    std::vector<Axis> dynamicAxes = DynamicAxesFromInternalDynamicAxisName(internalDynamicAxisName);
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNewDynamicAxes] = AsDictionaryValueVector(dynamicAxes);

                    opType = PrimitiveOpType::Where;
                }
                else if (node->OperationName() == OperationNameOf(SliceNode))
                {
                    auto sliceNode = node->As<SliceNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = AsDictionaryValueVector(AsAxis(sliceNode->Axis()));
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameBeginIndex] = AsDictionaryValueVector(sliceNode->BeginIndex());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameEndIndex] = AsDictionaryValueVector(sliceNode->EndIndex());

                    opType = PrimitiveOpType::Slice;
                }
                else if (node->OperationName() == OperationNameOf(RandomSampleNode))
                {
                    auto randomSampleNode = node->As<RandomSampleNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAllowDuplicates] = randomSampleNode->GetAllowDuplicates();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNumSamples] = randomSampleNode->GetNumSamples();

                    opType = PrimitiveOpType::RandomSample;
                }
                else if (node->OperationName() == OperationNameOf(RandomSampleInclusionFrequencyNode))
                {
                    auto randomSampleInclusionFrequencyNode = node->As<RandomSampleInclusionFrequencyNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAllowDuplicates] = randomSampleInclusionFrequencyNode->GetAllowDuplicates();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNumSamples] = randomSampleInclusionFrequencyNode->GetNumSamples();

                    opType = PrimitiveOpType::RandomSampleInclusionFrequency;
                }
                else if (node->OperationName() == OperationNameOf(DropoutNode))
                {
                    auto dropoutNode = node->As<DropoutNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameDropoutRate] = dropoutNode->GetDropoutRate();

                    opType = PrimitiveOpType::Dropout;
                }
                else if (node->OperationName() == OperationNameOf(ReshapeNode))
                {
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNewShape] = AsNDShape(node->GetSampleLayout());

                    opType = PrimitiveOpType::Reshape;
                }
                else if (node->OperationName() == OperationNameOf(SumElementsNode))
                    opType = PrimitiveOpType::SumAll;
                else if (node->OperationName() == OperationNameOf(PlusNode))
                    opType = PrimitiveOpType::Plus;
                else if (node->OperationName() == OperationNameOf(LogPlusNode))
                    opType = PrimitiveOpType::LogPlus;
                else if (node->OperationName() == OperationNameOf(MinusNode))
                    opType = PrimitiveOpType::Minus;
                else if (node->OperationName() == OperationNameOf(ElementTimesNode))
                    opType = PrimitiveOpType::ElementTimes;
                else if (node->OperationName() == OperationNameOf(EqualNode))
                    opType = PrimitiveOpType::Equal;
                else if (node->OperationName() == OperationNameOf(NotEqualNode))
                    opType = PrimitiveOpType::NotEqual;
                else if (node->OperationName() == OperationNameOf(LessNode))
                    opType = PrimitiveOpType::Less;
                else if (node->OperationName() == OperationNameOf(LessEqualNode))
                    opType = PrimitiveOpType::LessEqual;
                else if (node->OperationName() == OperationNameOf(GreaterNode))
                    opType = PrimitiveOpType::Greater;
                else if (node->OperationName() == OperationNameOf(GreaterEqualNode))
                    opType = PrimitiveOpType::GreaterEqual;
                else if (node->OperationName() == OperationNameOf(PackedIndexNode))
                    opType = PrimitiveOpType::PackedIndex;
                else if (node->OperationName() == OperationNameOf(GatherPackedNode))
                    opType = PrimitiveOpType::GatherPacked;
                else if (node->OperationName() == OperationNameOf(ScatterPackedNode))
                    opType = PrimitiveOpType::ScatterPacked;
                else if (node->OperationName() == OperationNameOf(TimesNode))
                {
                    // Deal with abuse of * in legacy configs/models
                    if (inputVars[0].Shape().Rank() == 0 || inputVars[1].Shape().Rank() == 0)
                        opType = PrimitiveOpType::ElementTimes;
                    else
                    {
                        auto timesNode = node->As<TimesNode<ElementType>>();
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOutputRank] = timesNode->OutputRank();
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameInferInputRankToMap] = timesNode->InferInputRankToMap();
                        opType = PrimitiveOpType::Times;
                    }
                }
                else if (node->OperationName() == OperationNameOf(TransposeTimesNode))
                {
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOutputRank] = node->As<TransposeTimesNode<ElementType>>()->OutputRank();
                    opType = PrimitiveOpType::TransposeTimes;
                }
                else if (node->OperationName() == OperationNameOf(PastValueNode))
                {
                    if (inputVars.size() == 1)
                    {
                        auto initialStateVar = Constant::Scalar(node->As<PastValueNode<ElementType>>()->InitialActivationValue(), AsDeviceDescriptor(node->GetDeviceId()));
                        inputVars.push_back(initialStateVar);
                    }

                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOffset] = (size_t)node->As<PastValueNode<ElementType>>()->TimeStep();
                    opType = PrimitiveOpType::PastValue;
                }
                else if (node->OperationName() == OperationNameOf(FutureValueNode))
                {
                    if (inputVars.size() == 1)
                    {
                        auto initialStateVar = Constant::Scalar(node->As<FutureValueNode<ElementType>>()->InitialActivationValue(), AsDeviceDescriptor(node->GetDeviceId()));
                        inputVars.push_back(initialStateVar);
                    }

                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOffset] = (size_t)node->As<FutureValueNode<ElementType>>()->TimeStep();
                    opType = PrimitiveOpType::FutureValue;
                }
                else if (node->OperationName() == OperationNameOf(CosDistanceNode))
                    opType = PrimitiveOpType::CosDistance;
                else if (node->OperationName() == OperationNameOf(LogisticNode))
                    opType = PrimitiveOpType::Logistic;
                else if (node->OperationName() == OperationNameOf(SquareErrorNode))
                    opType = PrimitiveOpType::SquaredError;
                else if (node->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode))
                    opType = PrimitiveOpType::CrossEntropyWithSoftmax;
                else if (node->OperationName() == OperationNameOf(ClassificationErrorNode))
                    opType = PrimitiveOpType::ClassificationError;
                else if (node->OperationName() == OperationNameOf(LambdaRankNode))
                    opType = PrimitiveOpType::LambdaRank;
                else if (node->OperationName() == OperationNameOf(NDCG1EvalNode))
                    opType = PrimitiveOpType::NDCG;
                else if (node->OperationName() == OperationNameOf(ReduceElementsNode))
                {
                    auto reduceElementsNode = node->As<ReduceElementsNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = AsAxis(reduceElementsNode->ReductionAxis());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameReductionOpName] = reduceElementsNode->ReductionOpName();

                    opType = PrimitiveOpType::ReduceElements;
                }
                else if (node->OperationName() == OperationNameOf(SumColumnElementsNode))
                {
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = Axis(0);
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameReductionOpName] = PrimitiveFunction::InternalSumReductionOpName;

                    opType = PrimitiveOpType::ReduceElements;
                }
                else if (node->OperationName() == OperationNameOf(ConvolutionNode))
                {
                    auto convolutionNode = node->As<ConvolutionNode<ElementType>>();

                    // Some legacy CNTK v1 models store the convolution filter parameters in 2D form with the trailing
                    // tensor dimensions flattended into the column dimension of the 2D paramater matrix
                    // We need to recover the actual tensor shape of the parameter in this case
                    auto& convolutionMapVar = inputVars[0];
                    if (convolutionNode->IsConvolution2D() || (convolutionMapVar.Shape().Rank() == 2))
                    {
                        assert(convolutionMapVar.Shape().Rank() == 2);
                        assert(convolutionMapVar.IsConstant() || convolutionMapVar.IsParameter());
                        auto kernelShape = AsNDShape(convolutionNode->KernelShape());
                        NDShape actualConvolutionMapShape = kernelShape.AppendShape({ convolutionMapVar.Shape()[0] });

                        if (actualConvolutionMapShape.TotalSize() != convolutionMapVar.Shape().TotalSize())
                            LogicError("The convolution map tensor's shape '%S' size does not match the size (%d) of the legacy 2D convolution map shape '%S'.",
                                        actualConvolutionMapShape.AsString().c_str(), (int)convolutionMapVar.Shape().TotalSize(), convolutionMapVar.Shape().AsString().c_str());

                        auto oldConvolutionMapValue = convolutionMapVar.IsConstant() ? Constant(convolutionMapVar).Value() : Parameter(convolutionMapVar).Value();
                        auto oldConvolutionMapMatrix = oldConvolutionMapValue->GetMatrix<ElementType>();

                        auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(oldConvolutionMapMatrix->AsReference()), AsTensorViewShape(actualConvolutionMapShape));
                        auto newConvolutionMapValue = MakeSharedObject<NDArrayView>(oldConvolutionMapValue->GetDataType(), oldConvolutionMapValue->Device(), oldConvolutionMapValue->GetStorageFormat(), actualConvolutionMapShape, oldConvolutionMapValue->IsReadOnly(), tensorView);

                        // Lets replace the convolutionMapVar with a new properly reshaped Parameter/Constant
                        convolutionMapVar = convolutionMapVar.IsConstant() ? Variable(Constant(newConvolutionMapValue, convolutionMapVar.Name(), convolutionMapVar.Uid())) : Variable(Parameter(newConvolutionMapValue, convolutionMapVar.Name(), convolutionMapVar.Uid()));
                    }

                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameStrides] = AsNDShape(convolutionNode->Strides());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(convolutionNode->Sharing());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(convolutionNode->AutoPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameLowerPad] = AsNDShape(convolutionNode->LowerPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUpperPad] = AsNDShape(convolutionNode->UpperPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameTranspose] = convolutionNode->Transpose();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOutputShape] = AsNDShape(convolutionNode->OutputShape());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples] = convolutionNode->MaxTempMemSizeInSamples();

                    opType = PrimitiveOpType::Convolution;
                }
                else if (node->OperationName() == OperationNameOf(ROIPoolingNode))
                {
                    auto roiPoolingNode = node->As<ROIPoolingNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameROIOutputShape] = AsNDShape(roiPoolingNode->ROIOutputShape());

                    opType = PrimitiveOpType::ROIPooling;
                }
                else if (node->OperationName() == OperationNameOf(MaxUnpoolingNode))
                {
                    auto unpoolingNode = node->As<MaxUnpoolingNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNamePoolingType] = (size_t)PoolingType::Max;
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUnpoolingWindowShape] = AsNDShape(unpoolingNode->KernelShape());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameStrides] = AsNDShape(unpoolingNode->Strides());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(unpoolingNode->AutoPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameLowerPad] = AsNDShape(unpoolingNode->LowerPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUpperPad] = AsNDShape(unpoolingNode->UpperPad());

                    opType = PrimitiveOpType::Unpooling;
                }
                else if (node->OperationName() == OperationNameOf(PoolingNode))
                {
                    auto poolingNode = node->As<PoolingNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNamePoolingType] = (size_t)(AsPoolingType(poolingNode->PoolingKind()));
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNamePoolingWindowShape] = AsNDShape(poolingNode->KernelShape());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameStrides] = AsNDShape(poolingNode->Strides());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(poolingNode->AutoPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameLowerPad] = AsNDShape(poolingNode->LowerPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUpperPad] = AsNDShape(poolingNode->UpperPad());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameCeilOutDim] = poolingNode->CeilOutDim();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameIncludePad] = poolingNode->PoolPadMode();

                    opType = PrimitiveOpType::Pooling;
                }
                // Legacy pooling node.
                else if ((node->OperationName() == OperationNameOf(MaxPoolingNode)) ||
                         (node->OperationName() == OperationNameOf(AveragePoolingNode)))
                {
                    auto poolingNode = node->As<PoolingNodeBase<ElementType>>();
                    if (poolingNode->IsImageLayoutCHW())
                    {
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNamePoolingType] = (size_t)(AsPoolingType(poolingNode->PoolingKind()));
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNamePoolingWindowShape] = AsNDShape(poolingNode->KernelShape());
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameStrides] = AsNDShape(poolingNode->Strides());
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(poolingNode->AutoPad());
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameLowerPad] = AsNDShape(poolingNode->LowerPad());
                        primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUpperPad] = AsNDShape(poolingNode->UpperPad());

                        opType = PrimitiveOpType::Pooling;
                    }
                    else
                        LogicError("Unsupported data layout for ComputationNode with OperationName='%S' found when loading legacy CNTK model", node->OperationName().c_str());
                }
                else if (node->OperationName() == OperationNameOf(BatchNormalizationNode))
                {
                    auto batchNormalizationNode = node->As<BatchNormalizationNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameSpatial] = batchNormalizationNode->Spatial();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNormalizationTimeConstant] = batchNormalizationNode->NormalizationTimeConstant();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameBlendTimeConstant] = batchNormalizationNode->BlendTimeConstant();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameEpsilon] = batchNormalizationNode->Epsilon();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUseCuDNNEngine] = !batchNormalizationNode->UseCNTKEngine();

                    opType = PrimitiveOpType::BatchNormalization;
                }
                else if (node->OperationName() == OperationNameOf(ClipNode))
                    opType = PrimitiveOpType::Clip;
                else if (node->OperationName() == OperationNameOf(IfNode))
                    opType = PrimitiveOpType::Select;
                else if (node->OperationName() == OperationNameOf(RowStackNode))
                {
                    // Internal CNTK SliceNode uses 1 based axis indices instead of 0 based
                    auto rowStackNode = node->As<RowStackNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = AsAxis(rowStackNode->GetSpliceDim());

                    opType = PrimitiveOpType::Splice;
                }
                else if (node->OperationName() == OperationNameOf(OptimizedRNNStackNode))
                {
                    auto optimizedRNNStackNode = node->As<OptimizedRNNStackNode<ElementType>>();
                    auto attributes = optimizedRNNStackNode->Attributes();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameBidirectional] = attributes.m_bidirectional;
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameHiddenSize] = attributes.m_hiddenSize;
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameNumLayers] = attributes.m_numLayers;
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameRecurrentOp] = attributes.m_recurrentOp;

                    opType = PrimitiveOpType::OptimizedRNNStack;
                }
                else if (node->OperationName() == OperationNameOf(ReconcileDynamicAxisNode))
                {
                    opType = PrimitiveOpType::ReconcileDynamicAxis;
                }
                else if (node->OperationName() == OperationNameOf(LogSoftmaxNode))
                {
                    opType = PrimitiveOpType::LogSoftmax;
                }
                else if (node->OperationName() == OperationNameOf(EditDistanceErrorNode)) 
                {
                    auto edNode = node->As<EditDistanceErrorNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameInsertionPenalty] = edNode->InsertionPenalty();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameDeletionPenalty] = edNode->DeletionPenalty();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameSubstitutionPenalty] = edNode->SubstitutionPenalty();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameSquashInputs] = edNode->SquashInputs();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameTokensToIgnore] = AsDictionaryValueVector(edNode->TokensToIgnore());

                    opType = PrimitiveOpType::EditDistanceError;
                }
                else if (node->OperationName() == OperationNameOf(ForwardBackwardNode))
                {
                    auto edNode = node->As<ForwardBackwardNode<ElementType>>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameDelayConstraint] = edNode->DelayConstraint();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameBlankTokenId] = edNode->BlankTokenId();

                    opType = PrimitiveOpType::ForwardBackward;
                }
                else if (node->OperationName() == OperationNameOf(CosDistanceWithNegativeSamplesNode))
                {
                    opType = PrimitiveOpType::CosDistanceWithNegativeSamples;
                }
                else if ((node->OperationName() == OperationNameOf(MeanNode)) || (node->OperationName() == OperationNameOf(InvStdDevNode)))
                {
                    auto precomputeNode = node->As<MeanInvStdDevNodeBase<ElementType>>();
                    if (!precomputeNode->HasComputed())
                        InvalidArgument("Cannot load a CNTK legacy V1 model containing a Mean/InvStdDev node '%S' which is not precomputed.", node->NodeName().c_str());

                    return CreateParameterOrConstantFromNodeValue<ElementType>(node, /* isConstant =*/ true);
                }
                else if (node->OperationName() == OperationNameOf(PerDimMeanVarNormalizationNode))
                {
                    auto meanValue = Constant(inputVars[1]).Value();
                    auto invStdDevValue = Constant(inputVars[2]).Value();

                    std::wstring uid, name;
                    std::tie(uid, name) = UidAndNameFromCNTKInternalNodeName(node->NodeName());

                    return PerDimMeanVarianceNormalize(inputVars[0], meanValue, invStdDevValue, name);
                }
                else
                    InvalidArgument("Unsupported ComputationNode with OperationName='%S' found when loading legacy CNTK model.\n"
                                    "This is likely a deprecated operation; loading Brainscript/NDL models that contain deprecated operations, is not supported in Python/C++ API.\n"
                                    "Please refer to CNTK documentation and edit/modify your Brainscript model/script to replace the deprecated operation with a supported operation.\n" , node->OperationName().c_str());

                if (node->Is<RngUser>())
                {
                    auto rngUserNode = node->As<RngUser>();
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameRngSeed] = static_cast<size_t>(rngUserNode->GetRngSeed());
                    primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameRngOffset] = static_cast<size_t>(rngUserNode->GetRngOffset());
                }

                // Let's reorder inputVars properly since the ordering of inputs of CNTK internal ComputationNode may be different from the PrimitiveFunction inputs ordering
                ReorderAsPrimitiveFunctionInputs(opType, inputVars);

                std::wstring functionUid, functionName;
                std::tie(functionUid, functionName) = UidAndNameFromCNTKInternalNodeName(node->NodeName(), opType);

                FunctionPtr primitiveFunction = MakeSharedObject<PrimitiveFunction>(opType, inputVars, std::move(primitiveFunctionConfigParameters), functionName, functionUid);
                m_allPrimitiveFunctions.insert(primitiveFunction);
                return primitiveFunction->Output();
            }
        };

        FunctionPtr LoadLegacyModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
        {
            ComputationNetworkPtr net = make_shared<ComputationNetwork>(AsCNTKImplDeviceId(computeDevice));
            net->SetTraceLevel(Internal::GetComputationNetworkTraceLevel());
            net->SetTrackGapNans(Internal::GetComputationNetworkTrackGapNans());

            auto dataType = DetectLegacyModelDataType(modelFile);
            switch (dataType)
            {
            case LegacyModelDataType::Auto:
                net->Load<float>(modelFile); // the actual template type will be ignored.
                break;
            case LegacyModelDataType::Float:
                net->Load<float>(modelFile);
                break;
            case LegacyModelDataType::Double:
                net->Load<double>(modelFile);
                break;
            default:
                NOT_IMPLEMENTED;
            }

            return ConvertFromLegacyModel(net);
        }

        FunctionPtr ConvertFromLegacyModel(const ComputationNetworkPtr& net)
        {
            // Traverse the model and construct the Function graph
            std::unordered_map<ComputationNodeBasePtr, Variable> nodeToVariableMap;
            std::unordered_map<Variable, Variable> placeholderReplacements;
            std::vector<Variable> rootVariables;
            VariableResolver resolver;
            auto& networkRoots = net->RootNodes();
            for (auto& rootNode : networkRoots)
            {
                if (rootNode->IsLeaf())
                    continue;

                if (ComputationNetwork::IsNodePtr<ComputationNode<float>>(rootNode))
                {
                    auto var = resolver.GetVariable<float>(rootNode);
                    rootVariables.push_back(var.IsOutput() ? (Variable)var.Owner() : var);
                }
                else if (ComputationNetwork::IsNodePtr<ComputationNode<double>>(rootNode))
                {
                    auto var = resolver.GetVariable<double>(rootNode);
                    rootVariables.push_back(var.IsOutput() ? (Variable)var.Owner() : var);
                }
                else
                    LogicError("ConvertFromLegacyModel(): computation node '%S' has invalid element type.", rootNode->NodeName().c_str());
            }

            auto rootComposite = Combine(rootVariables);
            rootComposite->ReplacePlaceholders(resolver.GetPlaceHolders());
            return rootComposite;
        }

        void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile)
        {
            CompositeFunction* compositeFunction = dynamic_cast<CompositeFunction*>(rootFunction.get());
            if (compositeFunction == nullptr)
                InvalidArgument("Primitive (i.e. non-composite) Function '%S' instance cannot be saved.", rootFunction->AsString().c_str());

            ComputationNetworkPtr computationNetwork;
            DataType dataType = rootFunction->Outputs()[0].GetDataType();
            DeviceDescriptor device = DeviceDescriptor::CPUDevice();
            if (compositeFunction->m_computationNetwork == nullptr)
            {
                auto parameters = compositeFunction->Parameters();
                if (!parameters.empty())
                    device = parameters.front().Value()->Device();
            }
            else
                device = AsDeviceDescriptor(compositeFunction->m_computationNetwork->GetDeviceId());

            switch (dataType)
            {
            case DataType::Float:
                computationNetwork = compositeFunction->GetComputationNetwork<float>(device, {}, {}, {}, false);
                break;
            case DataType::Double:
                computationNetwork = compositeFunction->GetComputationNetwork<double>(device, {}, {}, {}, false);
                break;
            default:
                LogicError("SaveAsLegacyModel: Function '%S' has unknown DataType %s.", rootFunction->AsString().c_str(), DataTypeName(dataType));
            }

            computationNetwork->Save(modelFile);
        }

        LegacyModelDataType DetectLegacyModelDataType(const std::wstring& modelFile)
        {
            File fstream(modelFile, FileOptions::fileOptionsBinary | FileOptions::fileOptionsRead);
            fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BCN");

            // model version
            size_t modelVersion = CNTK_MODEL_VERSION_1; // if version info is not there it is version 1
            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BVersion"))
            {
                fstream >> modelVersion;
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EVersion");
            }

            if (modelVersion > CNTK_MODEL_VERSION_7)
            {
                return LegacyModelDataType::Auto;
            }

            char b = 0x42;
            std::wstring bmat = L"BMAT";
            for (;;)
            {
                fstream.SkipToDelimiter(b); // skip to the next 'B' character.
                ungetc(b, fstream); // but the character back into the stream.
                if (fstream.TryGetMarker(fileMarkerBeginSection, bmat))
                {
                    size_t elementSize;
                    fstream >> elementSize;
                    if (elementSize == sizeof(float))
                        return LegacyModelDataType::Float;
                    else if (elementSize == sizeof(double))
                        return LegacyModelDataType::Double;
                    else
                        RuntimeError("DetectLegacyModelDataType(): invalid element size %zu.", elementSize);
                    }
                fgetc(fstream); // consume 'B' character to avoid an infinite cycle.
            }
        }
    }
}
