//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Function.h"
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

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    template <typename ElementType>
    Variable GetVariable(const ComputationNodeBasePtr& node,
                         std::unordered_map<ComputationNodeBasePtr, Variable>& nodeToVariableMap,
                         std::unordered_map<Variable, Variable>& placeholderReplacements,
                         std::unordered_set<FunctionPtr>& allPrimitiveFunctions)
    {
        auto iter = nodeToVariableMap.find(node);
        if (iter != nodeToVariableMap.end())
            return iter->second;

        Variable var;
        NDShape varShape = AsNDShape(node->GetSampleLayout());

        if (node->IsLeaf())
        {
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

                    var = Variable(varShape, isSparse, AsDataType<ElementType>(), node->GetLearningRateMultiplier() != 0, varName, inputVarDynamicAxes, varUid);
                }
                else
                {
                    // TODO: Allow creating inputs without a dynamic axis
                    LogicError("Found InputNode with no dynamic axes which is currently unsupported");
                }
            }
            else if (node->Is<LearnableParameter<ElementType>>())
            {
                bool isConstant = (node->GetLearningRateMultiplier() == 0);
                auto& matrix = node->As<ComputationNode<ElementType>>()->Value();
                auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorViewShape(node->GetSampleLayout()));
                NDArrayViewPtr value = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), varShape, false, tensorView);
                if (isConstant)
                {
                    std::tie(varUid, varName) = UidAndNameFromCNTKInternalNodeName(node->NodeName(), VariableKind::Constant);
                    var = Constant(value, varName, varUid);
                }
                else
                {
                    std::tie(varUid, varName) = UidAndNameFromCNTKInternalNodeName(node->NodeName(), VariableKind::Parameter);
                    var = Parameter(value, varName, varUid);
                }
            }
            else
                LogicError("CNTK::LoadLegacyModel: Unsupported legacy CNTK node named '%S'", node->NodeName().c_str());
        }
        else
        {
            // This is a non-leaf node and maps to a primitive Function
            auto placeholderVar = PlaceholderVariable(varShape);
            nodeToVariableMap[node] = placeholderVar;

            std::vector<Variable> inputVars(node->GetNumInputs());
            for (size_t i = 0; i < inputVars.size(); ++i)
            {
                inputVars[i] = GetVariable<ElementType>(node->Input(i), nodeToVariableMap, placeholderReplacements, allPrimitiveFunctions);
                if (inputVars[i].IsPlaceholder())
                    placeholderReplacements[inputVars[i]] = Variable();
            }

            PrimitiveOpType opType;
            Dictionary primitiveFunctionConfigParameters;
            if (node->OperationName() == OperationNameOf(NegateNode))
                opType = PrimitiveOpType::Negate;
            else if (node->OperationName() == OperationNameOf(SigmoidNode))
                opType = PrimitiveOpType::Sigmoid;
            else if (node->OperationName() == OperationNameOf(TanhNode))
                opType = PrimitiveOpType::Tanh;
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
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = AsAxis(sliceNode->Axis());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameBeginIndex] = sliceNode->BeginIndex();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameEndIndex] = sliceNode->EndIndex();

                opType = PrimitiveOpType::Slice;
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
                auto timesNode = node->As<TimesNode<ElementType>>();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameOutputRank] = timesNode->OutputRank();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameInferInputRankToMap] = (size_t)timesNode->InferInputRankToMap();
                opType = PrimitiveOpType::Times;
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
            else if (node->OperationName() == OperationNameOf(SquareErrorNode))
                opType = PrimitiveOpType::SquaredError;
            else if (node->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode))
                opType = PrimitiveOpType::CrossEntropyWithSoftmax;
            else if (node->OperationName() == OperationNameOf(ClassificationErrorNode))
                opType = PrimitiveOpType::ClassificationError;
            else if (node->OperationName() == OperationNameOf(ReduceElementsNode))
            {
                auto reduceElementsNode = node->As<ReduceElementsNode<ElementType>>();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAxis] = AsAxis(reduceElementsNode->ReductionAxis());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameReductionOpName] = reduceElementsNode->ReductionOpName();

                opType = PrimitiveOpType::ReduceElements;
            }
            else if (node->OperationName() == OperationNameOf(ConvolutionNode))
            {
                auto convolutionNode = node->As<ConvolutionNode<ElementType>>();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameStrides] = AsNDShape(convolutionNode->Strides());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(convolutionNode->Sharing());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(convolutionNode->AutoPad());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameLowerPad] = AsNDShape(convolutionNode->LowerPad());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameUpperPad] = AsNDShape(convolutionNode->UpperPad());
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameTranspose] = convolutionNode->Transpose();
                primitiveFunctionConfigParameters[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples] = convolutionNode->MaxTempMemSizeInSamples();

                opType = PrimitiveOpType::Convolution;
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

                opType = PrimitiveOpType::Pooling;
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
            else
                LogicError("Unsupported ComputationNode with OperationName='%S' found when loading legacy CNTK model", node->OperationName().c_str());

            // Let's reorder inputVars properly since the ordering of inputs of CNTK internal ComputationNode may be different from the PrimitiveFunction inputs ordering
            ReorderAsPrimitiveFunctionInputs(opType, inputVars);

            FunctionPtr primitiveFunction = MakeSharedObject<PrimitiveFunction>(opType, inputVars, std::move(primitiveFunctionConfigParameters), node->NodeName());
            allPrimitiveFunctions.insert(primitiveFunction);
            var = primitiveFunction->Output();
            if (placeholderReplacements.find(placeholderVar) != placeholderReplacements.end())
                placeholderReplacements[placeholderVar] = var;
        }

        nodeToVariableMap[node] = var;
        return var;
    }

    template <typename ElementType>
    FunctionPtr LoadLegacyModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        ComputationNetworkPtr net = make_shared<ComputationNetwork>(AsCNTKImplDeviceId(computeDevice));
        net->Load<ElementType>(modelFile);

        // Now traverse the model and construct the Function graph
        std::unordered_map<ComputationNodeBasePtr, Variable> nodeToVariableMap;
        std::unordered_map<Variable, Variable> placeholderReplacements;
        std::unordered_set<FunctionPtr> allPrimitiveFunctions;
        std::vector<Variable> rootVariables;
        auto& networkRoots = net->RootNodes();
        for (auto& rootNode : networkRoots)
        {
            if (rootNode->IsLeaf())
                continue;

            rootVariables.push_back(GetVariable<ElementType>(rootNode, nodeToVariableMap, placeholderReplacements, allPrimitiveFunctions).Owner());
        }

        auto rootComposite = Combine(rootVariables);
        rootComposite->ReplacePlaceholders(placeholderReplacements);

        return rootComposite;
    }

    FunctionPtr LoadLegacyModel(DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        switch (dataType)
        {
        case DataType::Float:
            return LoadLegacyModel<float>(modelFile, computeDevice);
        case DataType::Double:
            return LoadLegacyModel<double>(modelFile, computeDevice);
        default:
            LogicError("Unknown DataType %s", DataTypeName(dataType));
        }
    }

    void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile)
    {
        CompositeFunction* compositeFunction = dynamic_cast<CompositeFunction*>(rootFunction.get());
        if (compositeFunction == nullptr)
            InvalidArgument("Primitive (aka non-composite) Function instances cannot be saved");

        ComputationNetworkPtr computationNetwork;
        DataType dataType = rootFunction->Outputs()[0].GetDataType();
        auto device = (compositeFunction->m_computationNetwork == nullptr) ? DeviceDescriptor::CPUDevice() : AsDeviceDescriptor(compositeFunction->m_computationNetwork->GetDeviceId());
        switch (dataType)
        {
        case DataType::Float:
            computationNetwork = compositeFunction->GetComputationNetwork<float>(device, {}, false);
            break;
        case DataType::Double:
            computationNetwork = compositeFunction->GetComputationNetwork<double>(device, {}, false);
            break;
        default:
            LogicError("Unknown DataType %s", DataTypeName(dataType));
        }

        computationNetwork->Save(modelFile);
    }
}
