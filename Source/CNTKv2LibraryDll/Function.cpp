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

using namespace Microsoft::MSR::CNTK;

bool g_shareNodeValueMatrices = true;

namespace CNTK
{
    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetNode(const Variable& variable, ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        if (variableToNodeMap.find(variable) != variableToNodeMap.end())
            return variableToNodeMap[variable];

        std::shared_ptr<ComputationNode<ElementType>> computationNodePtr;
        if (variable.IsParameter() || variable.IsConstant())
        {
            computationNodePtr = builder.CreateLearnableParameter(variable.Name(), AsTensorShape(variable.Shape()));
            if (!variable.NeedsGradient())
                computationNodePtr->SetLearningRateMultiplier(0.0);

            NDArrayViewPtr value = variable.IsConstant() ? Constant(variable).Value() : Parameter(variable).Value();
            auto matrix = variable.IsConstant() ? value->GetMatrix<ElementType>()->AsReference() : value->GetWritableMatrix<ElementType>()->AsReference();
            computationNodePtr->Value() = std::move(matrix);
        }
        else if (variable.Kind() == VariableKind::Input)
        {
            // TODO: Specify dynamic axis
            computationNodePtr = builder.CreateInputNode(variable.Name(), AsTensorShape(variable.Shape()));
            if (variable.NeedsGradient())
            {
                // Set a dummy learning rate multiplier to force gradient computation for the input computation node since by default
                // gradients are not computed for Input nodes
                computationNodePtr->SetLearningRateMultiplier(0.00001f);
            }
        }
        else
        {
            assert(variable.Kind() == VariableKind::Output);
            computationNodePtr = GetOutputVariableNode(variable, builder, variableToNodeMap, isVariableRootMap)->As<ComputationNode<ElementType>>()->shared_from_this();
        }

        variableToNodeMap[variable] = computationNodePtr;
        isVariableRootMap[variable] = (variable.Kind() == VariableKind::Output);
        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetOutputVariableNode(const Variable& variable, ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        assert(variable.Kind() == VariableKind::Output);
        Function* function = variable.Owner();
        ComputationNodeBasePtr computationNodePtr;
        if (dynamic_cast<PrimitiveFunction*>(function) != nullptr)
        {
            PrimitiveFunction* primitiveFunction = dynamic_cast<PrimitiveFunction*>(function);

            // Create the nodes corresponding to the inputs
            auto functionInputs = primitiveFunction->Inputs();
            std::shared_ptr<ComputationNode<ElementType>> input0Node = GetNode(functionInputs[0], builder, variableToNodeMap, isVariableRootMap)->As<ComputationNode<ElementType>>()->shared_from_this();

            std::shared_ptr<ComputationNode<ElementType>> input1Node;
            if (functionInputs.size() > 1)
                input1Node = GetNode(functionInputs[1], builder, variableToNodeMap, isVariableRootMap)->As<ComputationNode<ElementType>>()->shared_from_this();

            PrimitiveOpType op = primitiveFunction->OpType();
            switch (op)
            {
            case PrimitiveOpType::Plus:
                computationNodePtr = builder.Plus(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Times:
                // TODO: The output rank of the times operation is currently hardcoded to 1
                computationNodePtr = builder.Times(input0Node, input1Node, 1, function->Name());
                break;
            case PrimitiveOpType::Sigmoid:
                computationNodePtr = builder.Sigmoid(input0Node, function->Name());
                break;
            case PrimitiveOpType::CrossEntropyWithSoftmax:
                computationNodePtr = builder.CrossEntropyWithSoftmax(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::PredictionError:
                computationNodePtr = builder.ErrorPrediction(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::Combine:
                for (size_t i = 0; i < functionInputs.size(); ++i)
                    GetNode(functionInputs[i], builder, variableToNodeMap, isVariableRootMap);

                computationNodePtr = variableToNodeMap[variable];

                break;
            default:
                LogicError("Specified op %s not yet supported", PrimitiveOpTypeName(op));
                break;
            }

            if (op != PrimitiveOpType::Combine)
            {
                for (size_t i = 0; i < functionInputs.size(); ++i)
                    isVariableRootMap[functionInputs[i]] = false;
            }
        }
        else
        {
            LogicError("User defined Functions are currently unsupported!");
        }

        return computationNodePtr;
    }

    template <typename ElementType>
    ComputationNetworkPtr CompositeFunction::GetComputationNetwork(const DeviceDescriptor& device, const _Internal::_SimpleSet<Variable>& backpropRoots)
    {
        if (m_computationNetwork != nullptr)
        {
            // TODO: We should either invalidate and readapt the network if he backpropRoots change compared to what was specified when the network
            // was last constructed, to just recreate a new network.
            // For now just disallow changing the backpropRoots after the network is created
            if (m_currentBackpropRoots != *backpropRoots.m_set)
                LogicError("Changing backprop roots across different Forward calls on a CNTK composite Function is currently unsupported");

            // TODO: Support changing the device across different invocations of the forward method on a Function instance
            if (AsDeviceDescriptor(m_computationNetwork->GetDeviceId()) != device)
                LogicError("Changing device across different Forward calls on a CNTK composite Function is currently unsupported");
        }

        if (m_computationNetwork == nullptr)
        {
            m_computationNetwork = std::make_shared<ComputationNetwork>(AsCNTKImplDeviceId(device));

            ComputationNetworkBuilder<ElementType> builder(*m_computationNetwork);

            // TODO: We current only support one backprop root
            if (backpropRoots.Size() > 1)
                LogicError("More than one backprop roots is currently unsupported");

            ComputationNodeBasePtr backpropRootNode;

            // Now recursively create the network in a top-down fashion
            auto rootFunction = RootFunction();
            auto rootFunctionOutputs = rootFunction->Outputs();
            std::vector<ComputationNodeBasePtr> forwardRootNodes;
            for (size_t i = 0; i < rootFunctionOutputs.size(); ++i)
            {
                auto currentRootNode = GetNode(rootFunctionOutputs[i], builder, m_variableToNodeMap, m_isVariableRootMap);
                forwardRootNodes.push_back(currentRootNode);

                if (backpropRoots.Contains(rootFunctionOutputs[i]))
                    backpropRootNode = m_variableToNodeMap[rootFunctionOutputs[i]];
            }

            // If any of the function outputs is not a root node, we need to explicitly add it to the 'output' group of the ComputationNetwork
            for (size_t i = 0; i < rootFunctionOutputs.size(); ++i)
            {
                if (!m_isVariableRootMap[rootFunctionOutputs[i]])
                    m_computationNetwork->AddToNodeGroup(L"output", m_variableToNodeMap[rootFunctionOutputs[i]]);
            }

            m_currentBackpropRoots = backpropRoots;

            m_computationNetwork->CompileNetwork();

            // Verify that the shapes of the output Variables that we computed match the corresponding nodes in the ComputationNetwork
            for (auto iter = m_variableToNodeMap.begin(); iter != m_variableToNodeMap.end(); ++iter)
            {
                if (iter->first.Kind() == VariableKind::Output)
                {
                    auto outputVar = iter->first;
                    auto computationNodePtr = m_variableToNodeMap[outputVar];
                    auto outputShape = outputVar.Shape();
                    auto computationNodeSampleLayout = computationNodePtr->GetSampleLayout();
                    if (((outputShape.NumAxes() == 0) && (computationNodeSampleLayout[0] != 1)) ||
                        ((outputShape.NumAxes() != 0) && (computationNodeSampleLayout != AsTensorShape(outputShape))))
                    {
                        LogicError("The output Variable shape %s does not match the SampleLayout shape %s of the corresponding ComputationNode in the network", AsString(outputShape).c_str(), ((std::string)computationNodeSampleLayout).c_str());
                    }
                }
            }

            m_computationNetwork->AllocateAllMatrices(forwardRootNodes, {}, backpropRootNode);
        }

        return m_computationNetwork;
    }

    /*static*/ void CompositeFunction::CopyNDArrayViewToComputationNodeValue(const NDArrayViewPtr& arrayView, ComputationNodeBasePtr node)
    {
        switch (arrayView->DataType())
        {
        case DataType::Float:
        {
            auto& nodeData = node->As<ComputationNode<float>>()->Value();
            nodeData.AssignValuesOf(*(arrayView->GetMatrix<float>()));
            break;
        }
        case DataType::Double:
        {
            auto& nodeData = node->As<ComputationNode<double>>()->Value();
            nodeData.AssignValuesOf(*(arrayView->GetMatrix<double>()));
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(arrayView->DataType()));
            break;
        }
    }

    /*static*/ void CompositeFunction::CopyNDArrayViewToComputationNodeGradient(const NDArrayViewPtr& arrayView, ComputationNodeBasePtr node)
    {
        switch (arrayView->DataType())
        {
        case DataType::Float:
            node->As<ComputationNode<float>>()->ResetGradient(*(arrayView->GetMatrix<float>()));
            break;
        case DataType::Double:
            node->As<ComputationNode<double>>()->ResetGradient(*(arrayView->GetMatrix<double>()));
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(arrayView->DataType()));
            break;
        }
    }

    void CompositeFunction::PopulateNetworkInputs(const _Internal::_SimpleMap<Variable, const ValuePtr>& arguments)
    {
        auto functionArguments = this->Arguments();
        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto iter = functionArguments.begin(); iter != functionArguments.end(); ++iter)
        {
            // Ensure we have values for all arguments of the function
            if (!arguments.Contains(*iter))
                InvalidArgument("Value not specified for required Function Argument");

            auto argumentComputationNode = m_variableToNodeMap[*iter];
            inputNodes.push_back(argumentComputationNode);

            ValuePtr argumentValue = arguments[*iter];
            CopyNDArrayViewToComputationNodeValue(argumentValue->Data(), argumentComputationNode);

            // TODO: No sequence support for now
            // The number of axes for argument Value can be at most 1 larger than the number of axes of the variable's shape
            if (argumentValue->Data()->Shape().NumAxes() != (iter->Shape().NumAxes() + 1))
                InvalidArgument("Argument value's number of axes should be 1 larger than the argument variable's number of axes");

            size_t numSamples = argumentValue->Data()->Shape()[iter->Shape().NumAxes()];

            argumentComputationNode->GetMBLayout()->InitAsFrameMode(numSamples);
        }
        m_computationNetwork->BumpEvalTimeStamp(inputNodes);
    }

    void CompositeFunction::PopulateNetworkGradients(const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients)
    {
        auto functionOutputs = this->Outputs();
        std::unordered_map<Variable, const ValuePtr>& gradientsValueMap = *gradients.m_map;
        for (auto iter = gradientsValueMap.begin(); iter != gradientsValueMap.end(); ++iter)
        {
            // Only gradients for roots of the function can be specified
            if (std::find(functionOutputs.begin(), functionOutputs.end(), iter->first) == functionOutputs.end())
                InvalidArgument("Gradients cannot be specified for a Variable that is not an Output of the Function");

            auto outputComputationNode = m_variableToNodeMap[iter->first];

            ValuePtr gradientValue = iter->second;
            CopyNDArrayViewToComputationNodeGradient(gradientValue->Data(), outputComputationNode);
        }
    }

    /*static*/ void CompositeFunction::CopyComputationNodeDataToNDArrayView(const Microsoft::MSR::CNTK::ComputationNodeBasePtr& node, NDArrayViewPtr arrayView, bool copyGradient)
    {
        switch (arrayView->DataType())
        {
        case DataType::Float:
        {
            auto& outputMatrix = copyGradient ? node->As<ComputationNode<float>>()->Gradient() : node->As<ComputationNode<float>>()->Value();
            auto arrayViewMatrix = arrayView->GetWritableMatrix<float>();
            arrayViewMatrix->AssignValuesOf(outputMatrix);
            break;
        }
        case DataType::Double:
        {
            auto& outputMatrix = copyGradient ? node->As<ComputationNode<double>>()->Gradient() : node->As<ComputationNode<double>>()->Value();
            auto arrayViewMatrix = arrayView->GetWritableMatrix<double>();
            arrayViewMatrix->AssignValuesOf(outputMatrix);
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(arrayView->DataType()));
            break;
        }
    }

    static NDShape GetValueShape(const Variable& var, const ComputationNodeBasePtr& computationNodePtr)
    {
        size_t outputValueNumAxes = var.Shape().NumAxes();
        if (computationNodePtr->GetMBLayout() != nullptr)
            outputValueNumAxes++;

        std::vector<size_t> outputShapeDims(outputValueNumAxes);
        for (size_t i = 0; i < var.Shape().NumAxes(); ++i)
            outputShapeDims[i] = computationNodePtr->GetSampleLayout().GetDim(i);

        if (computationNodePtr->GetMBLayout() != nullptr)
            outputShapeDims[var.Shape().NumAxes()] = computationNodePtr->GetMBLayout()->GetNumParallelSequences();

        return NDShape(outputShapeDims);
    }

    void CompositeFunction::GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs)
    {
        // Now copy the Forward values of output nodes from the network to outputs' Value objects
        for (auto iter = outputs.begin(); iter != outputs.end(); ++iter)
        {
            auto computationNodePtr = m_variableToNodeMap[iter->first];
            auto outputValuePtr = iter->second;

            auto outputShape = GetValueShape(iter->first, computationNodePtr);
            if (outputValuePtr != nullptr)
            {
                // TODO: The shape of the specified output Value object must match the actual output shape
                if (outputValuePtr->Data()->Shape() != outputShape)
                    InvalidArgument("The shape %s of the specified Value object for output does not match the actual output shape %s", AsString(outputValuePtr->Data()->Shape()).c_str(), AsString(outputShape).c_str());
            }
            else
            {
                outputValuePtr = new Value(new NDArrayView(iter->first.DataType(), outputShape, nullptr, 0, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())));
            }

            CopyComputationNodeDataToNDArrayView(computationNodePtr, outputValuePtr->Data(), false);
            outputs[iter->first] = outputValuePtr;
        }
    }

    void CompositeFunction::GetNetworkGradients(std::unordered_map<Variable, ValuePtr>& gradients)
    {
        auto networkInputs = this->Inputs();
        // Now copy the gradient values of input nodes of the network to gradients' Value objects
        for (auto iter = gradients.begin(); iter != gradients.end(); ++iter)
        {
            // Only gradients corresponding to inputs of the network can be obtained
            if (std::find(networkInputs.begin(), networkInputs.end(), iter->first) == networkInputs.end())
                InvalidArgument("Backpropagated gradient values can only be obtained for inputs of a Function");

            // Gradients can only be obtained for parameter variables or input variables that NeedsGradient
            if (!iter->first.NeedsGradient())
                InvalidArgument("Gradient value incorrectly requested for an Output or Constant Variable, or an Input Variable with NeedsGradient setting of false");

            auto computationNodePtr = m_variableToNodeMap[iter->first];
            auto gradientValuePtr = iter->second;

            auto gradientShape = GetValueShape(iter->first, computationNodePtr);
            if (gradientValuePtr != nullptr)
            {
                // TODO: The shape of the specified output Value object must match the actual output shape
                if (gradientValuePtr->Data()->Shape() != gradientShape)
                    InvalidArgument("The shape %s of the specified Value object for gradient does not match the actual gradient shape %s", AsString(gradientValuePtr->Data()->Shape()).c_str(), AsString(gradientShape).c_str());
            }
            else
            {
                gradientValuePtr = new Value(new NDArrayView(iter->first.DataType(), gradientShape, nullptr, 0, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())));
            }

            if (!computationNodePtr->NeedsGradient())
                LogicError("Backpropagated gradient value cannot be read from a ComputationNode that has NeedsGradient set to false");

            CopyComputationNodeDataToNDArrayView(computationNodePtr, gradientValuePtr->Data(), true);
            gradients[iter->first] = gradientValuePtr;
        }
    }

    /*virtual*/ BackPropStatePtr CompositeFunction::Forward(const _Internal::_SimpleMap<Variable, const ValuePtr>& arguments,
                                                            _Internal::_SimpleMap<Variable, ValuePtr>& outputs,
                                                            const _Internal::_SimpleSet<Variable>& outputsToRetainBackwardStateFor,
                                                            const DeviceDescriptor& computeDevice)
    {
        // TODO: How about zero argument functions?
        // TODO: We need a better way to determine the ElementType for the network
        auto dataType = arguments.m_map->begin()->second->Data()->DataType();
        if (dataType == DataType::Float)
            GetComputationNetwork<float>(computeDevice, outputsToRetainBackwardStateFor);
        else
            GetComputationNetwork<double>(computeDevice, outputsToRetainBackwardStateFor);

        // TODO: Avoid copying the data when possible

        // Feed data into the arguments of the network
        PopulateNetworkInputs(arguments);

        std::unordered_set<Variable> functionOutputs = _Internal::_SimpleVector<Variable>::CreateSimpleVector(this->Outputs()).GetAsUnorderedSet();
        std::vector<ComputationNodeBasePtr> outputsToEvaluate;

        for (auto iter = outputs.m_map->begin(); iter != outputs.m_map->end(); ++iter)
        {
            // Ensure that only a subset of this function's outputs are being asked to be evaluated
            if (functionOutputs.find(iter->first) == functionOutputs.end())
                InvalidArgument("Requested output is not an Ouptut of the Function");

            auto outputComputationNode = m_variableToNodeMap[iter->first];
            outputsToEvaluate.push_back(outputComputationNode);
        }

        // The 'outputsToRetainBackwardStateFor' nodes also need to be evaluated if not already specified in 'outputs'
        for (auto iter = outputsToRetainBackwardStateFor.m_set->begin(); iter != outputsToRetainBackwardStateFor.m_set->end(); ++iter)
        {
            if (outputs.m_map->find(*iter) == outputs.m_map->end())
                outputsToEvaluate.push_back(m_variableToNodeMap[*iter]);
        }

        m_computationNetwork->ForwardProp(outputsToEvaluate);

        GetNetworkOutputs(*(outputs.m_map));

        // TODO: How to deal with the specified 'computeDevice'

        return (outputsToRetainBackwardStateFor.Size() > 0) ? new CNTKBackPropState(this, m_variableToNodeMap[arguments.m_map->begin()->first]->GetEvalTimeStamp()) : nullptr;
    }

    /*virtual*/ void CompositeFunction::Backward(const BackPropStatePtr& state,
                                                 const _Internal::_SimpleMap<Variable, const ValuePtr>& rootGradientValues,
                                                 _Internal::_SimpleMap<Variable, ValuePtr>& backPropagatedGradientValuesForInputs)
    {
        if ((state == nullptr) || (dynamic_cast<const CNTKBackPropState*>(state.GetPtr()) == nullptr))
            InvalidArgument("Invalid backprop state specified");

        // TODO: Support multiple concurrent backprop states
        auto backpropState = dynamic_cast<const CNTKBackPropState*>(state.GetPtr());
        if (backpropState->EvalTimeStamp() != m_variableToNodeMap[*(this->Arguments().begin())]->GetEvalTimeStamp())
            LogicError("The specified backprop state specified cannot be used for backpropagation as the Function's internal state was modified by subsequent Forward calls to the function."
                       "This is not a user error but a shortcoming of the current implementation where multiple independent backprop states are not simultaneously supported");

        // TODO: Avoid copying the data when possible

        // Feed data into the arguments of the network
        PopulateNetworkGradients(rootGradientValues);

        // Zero all gradients of nodes below the root nodes
        for (auto iter = rootGradientValues.m_map->begin(); iter != rootGradientValues.m_map->end(); ++iter)
            m_computationNetwork->ZeroInputGradients(m_variableToNodeMap[iter->first]);

        if (rootGradientValues.Size() > 1)
            LogicError("Currently gradient backprop from only one of the Function Outputs is supported");

        // Backpropagate through the network
        auto rootComputationNodePtr = m_variableToNodeMap[rootGradientValues.m_map->begin()->first];
        m_computationNetwork->GetNestedNetwork(rootComputationNodePtr)->Backprop(FrameRange(nullptr), true, true);

        GetNetworkGradients(*(backPropagatedGradientValuesForInputs.m_map));

        // TODO: How to deal with the specified 'computeDevice'
    }

    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Times, { leftOperand, rightOperand }, Dictionary(), name), name);
    }

    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Plus, { leftOperand, rightOperand }, Dictionary(), name), name);
    }

    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Sigmoid, { operand }, Dictionary(), name), name);
    }

    FunctionPtr _Combine(const _Internal::_SimpleVector<FunctionPtr>& operands, const std::wstring& name/* = L""*/)
    {
        _Internal::_SimpleSet<FunctionPtr> uniqueOperands;
        std::vector<Variable> inputs;
        for (size_t i = 0; i < operands.Size(); ++i)
        {
            if (uniqueOperands.Contains(operands[i]))
                LogicError("All function operands specified to Combine must be unique");

            uniqueOperands.Insert(operands[i]);
            auto currentFunctionOutputs = operands[i]->Outputs();
            std::copy(currentFunctionOutputs.begin(), currentFunctionOutputs.end(), std::back_inserter(inputs));
        }

        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Combine, inputs, Dictionary(), name), name);

    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& output, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::CrossEntropyWithSoftmax, { output, labels }, Dictionary(), name), name);
    }

    FunctionPtr PredictionError(const Variable& prediction, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::PredictionError, { prediction, labels }, Dictionary(), name), name);
    }
}
