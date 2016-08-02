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

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    template <typename ElementType>
    Variable GetVariable(const ComputationNodeBasePtr& node,
                         std::unordered_map<ComputationNodeBasePtr, Variable>& nodeToVariableMap,
                         std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                         std::unordered_set<FunctionPtr>& allPrimitiveFunctions)
    {
        auto iter = nodeToVariableMap.find(node);
        if (iter != nodeToVariableMap.end())
            return iter->second;

        Variable var;
        NDShape varShape = AsNDShape(node->GetSampleLayout());
        // The CNTK sample layouts may have trailing axes with dimension size of 1 which are automatically
        // added when converting from NDShape to CNTK internal TensorShapes and are not present in the original
        // shapes specified by the user. These should be truncated.
        if (varShape.NumAxes() <= 2)
        {
            size_t numTrailingDimsToRemove = 0;
            for (int i = varShape.NumAxes() - 1; i >= 0; --i)
            {
                if (varShape[i] == 1)
                    numTrailingDimsToRemove++;
                else
                    break;
            }
            varShape = varShape.SubShape(0, varShape.NumAxes() - numTrailingDimsToRemove);
        }

        if (node->IsLeaf())
        {
            if (node->Is<InputValueBase<ElementType>>())
            {
                auto inputNode = node->As<InputValueBase<ElementType>>();
                bool isSparse = node->Is<SparseInputValue<ElementType>>();
                if (node->HasMBLayout())
                {
                    // TODO: Currently only default dynamic axis is supported
                    const std::wstring defaultCNTKDynamicAxisName = L"";
                    if (inputNode->GetRequestedDynamicAxis() != defaultCNTKDynamicAxisName)
                        LogicError("Found dynamic axis named '%S' while currently only default dynamic axis named '%S' is supported!", node->GetMBLayout()->GetAxisName(), defaultCNTKDynamicAxisName.c_str());

                    var = Variable(varShape, isSparse, AsDataType<ElementType>(), node->GetLearningRateMultiplier() != 0, node->GetName());
                }
                else
                {
                    // TODO: Allow creating inputs without a dynamic axis
                    LogicError("Found InputNode with no dynamic axis which is currently unsupported");
                }
            }
            else if (node->Is<LearnableParameter<ElementType>>())
            {
                auto& matrix = node->As<ComputationNode<ElementType>>()->Value();
                auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), node->GetSampleLayout());
                NDArrayViewPtr parameterValue = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), varShape, false, tensorView);
                var = Parameter(parameterValue, node->GetName());
            }
            else
                LogicError("CNTK::LoadLegacyModel: Unsupported legacy CNTK node named '%S'", node->NodeName().c_str());
        }
        else
        {
            // This is a non-leaf node and maps to a primitive Function
            auto placeholderVar = Placeholder(varShape);
            nodeToVariableMap[node] = placeholderVar;

            std::vector<Variable> inputVars(node->GetNumInputs());
            for (size_t i = 0; i < inputVars.size(); ++i)
            {
                inputVars[i] = GetVariable<ElementType>(node->Input(i), nodeToVariableMap, placeholderReplacements, allPrimitiveFunctions);
                if (inputVars[i].IsPlaceholder())
                    placeholderReplacements[Placeholder(inputVars[i])] = Variable();
            }

            PrimitiveOpType opType;
            Dictionary primitiveFunctionConfigParameters;
            if (node->OperationName() == OperationNameOf(TanhNode))
                opType = PrimitiveOpType::Tanh;
            else if (node->OperationName() == OperationNameOf(SigmoidNode))
                opType = PrimitiveOpType::Sigmoid;
            else if (node->OperationName() == OperationNameOf(ExpNode))
                opType = PrimitiveOpType::Exp;
            else if (node->OperationName() == OperationNameOf(TimesNode))
                opType = PrimitiveOpType::Times;
            else if (node->OperationName() == OperationNameOf(PlusNode))
                opType = PrimitiveOpType::Plus;
            else if (node->OperationName() == OperationNameOf(PastValueNode))
            {
                if (inputVars.size() == 1)
                {
                    auto initialStateVar = Constant({}, node->As<PastValueNode<ElementType>>()->InitialActivationValue(), AsDeviceDescriptor(node->GetDeviceId()));
                    inputVars.insert(inputVars.begin(), initialStateVar);
                }
                primitiveFunctionConfigParameters[L"stepSize"] = DictionaryValue((size_t)node->As<PastValueNode<ElementType>>()->TimeStep());
                opType = PrimitiveOpType::PastValue;
            }
            else if (node->OperationName() == OperationNameOf(FutureValueNode))
            {
                if (inputVars.size() == 1)
                {
                    auto initialStateVar = Constant({}, node->As<FutureValueNode<ElementType>>()->InitialActivationValue(), AsDeviceDescriptor(node->GetDeviceId()));
                    inputVars.insert(inputVars.begin(), initialStateVar);
                }
                primitiveFunctionConfigParameters[L"stepSize"] = DictionaryValue((size_t)node->As<FutureValueNode<ElementType>>()->TimeStep());
                opType = PrimitiveOpType::FutureValue;
            }
            else if (node->OperationName() == OperationNameOf(CrossEntropyWithSoftmaxNode))
            {
                std::swap(inputVars[0], inputVars[1]);
                opType = PrimitiveOpType::CrossEntropyWithSoftmax;
            }
            else if (node->OperationName() == OperationNameOf(ErrorPredictionNode))
            {
                std::swap(inputVars[0], inputVars[1]);
                opType = PrimitiveOpType::ClassificationError;
            }
            else if (node->OperationName() == OperationNameOf(ElementTimesNode))
                opType = PrimitiveOpType::ElementTimes;
            else if (node->OperationName() == OperationNameOf(SumElementsNode))
                opType = PrimitiveOpType::ReduceSum;
            else
                LogicError("Unsupported ComputationNode with OperationName='%S' found when loading legacy CNTK model", node->OperationName().c_str());

            FunctionPtr primitiveFunction = MakeSharedObject<PrimitiveFunction>(opType, inputVars, std::move(primitiveFunctionConfigParameters), node->GetName());
            allPrimitiveFunctions.insert(primitiveFunction);
            var = primitiveFunction->Output();
            if (placeholderReplacements.find(placeholderVar) != placeholderReplacements.end())
                placeholderReplacements[placeholderVar] = var;
        }

        nodeToVariableMap[node] = var;
        return var;
    }

    template <typename ElementType>
    FunctionPtr LoadLegacyModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::DefaultDevice()*/)
    {
        ComputationNetworkPtr net = make_shared<ComputationNetwork>(AsCNTKImplDeviceId(computeDevice));
        net->Load<ElementType>(modelFile);

        // Now traverse the model and construct the Function graph
        std::unordered_map<ComputationNodeBasePtr, Variable> nodeToVariableMap;
        std::unordered_map<Placeholder, Variable> placeholderReplacements;
        std::unordered_set<FunctionPtr> allPrimitiveFunctions;
        std::vector<FunctionPtr> rootFunctions;
        auto& networkRoots = net->RootNodes();
        for (auto& rootNode : networkRoots)
        {
            if (rootNode->IsLeaf())
                continue;

            rootFunctions.push_back(GetVariable<ElementType>(rootNode, nodeToVariableMap, placeholderReplacements, allPrimitiveFunctions).Owner());
        }

        auto rootComposite = Combine(rootFunctions);
        rootComposite->ReplacePlaceholders(placeholderReplacements);

        return rootComposite;
    }

    template <typename ElementType>
    void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile)
    {
        CompositeFunction* compositeFunction = dynamic_cast<CompositeFunction*>(rootFunction.get());
        if (compositeFunction == nullptr)
            InvalidArgument("Primitive (aka non-composite) Function instances cannot be saved");

        auto computationNetwork = compositeFunction->GetComputationNetwork<ElementType>(DeviceDescriptor::CPUDevice(), {});
        computationNetwork->Save(modelFile);
    }

    // Template instantiations
    template CNTK_API FunctionPtr LoadLegacyModel<float>(const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::DefaultDevice()*/);
    template CNTK_API FunctionPtr LoadLegacyModel<double>(const std::wstring& modelFile, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::DefaultDevice()*/);

    template CNTK_API void SaveAsLegacyModel<float>(const FunctionPtr& rootFunction, const std::wstring& modelFile);
    template CNTK_API void SaveAsLegacyModel<double>(const FunctionPtr& rootFunction, const std::wstring& modelFile);
}
