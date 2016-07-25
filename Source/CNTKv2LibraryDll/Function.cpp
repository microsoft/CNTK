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
#include "ReshapingNodes.h"

using namespace Microsoft::MSR::CNTK;

bool g_shareNodeValueMatrices = true;

namespace CNTK
{
    std::shared_ptr<std::vector<Variable>> Function::InputsImpl() const
    {
        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        std::vector<Variable> inputs;
        if (compositeFunction == nullptr)
            inputs = m_inputs;
        else
            inputs = compositeFunction->DetermineInputs();

        return std::shared_ptr<std::vector<Variable>>(new std::vector<Variable>(std::move(inputs)), [](std::vector<Variable>* ptr) { delete ptr; });
    }

    FunctionPtr Function::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements)
    {
        // Cannot be called on primitive functions
        if (RootFunction() == nullptr)
            InvalidArgument("ReplacePlaceholders should never be called on primitive functions");

        std::unordered_set<const Function*> visitedFunctions;
        std::unordered_set<Placeholder> replacedPlaceholders;
        ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        for (auto replacementPair : placeholderReplacements)
        {
            if (replacedPlaceholders.find(replacementPair.first) == replacedPlaceholders.end())
                InvalidArgument("At least one of the placeholders specified for replacement was not found in the function");
        }

        return this->shared_from_this();
    }

    // Placeholders can be replaced incrementally - i.e. not all placeholders need to replaced in one go.
    // The only requirement is that they must all be replaced before making any 'Forward' calls on the Function instance.
    /*virtual*/ void Function::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                                   std::unordered_set<const Function*>& visitedFunctions,
                                                   std::unordered_set<Placeholder>& replacedPlaceholders)
    {
        visitedFunctions.insert(this);

        for (auto& inputVar : m_inputs)
        {
            if (inputVar.IsPlaceholder())
            {
                Placeholder placeholder(inputVar);
                if (placeholderReplacements.find(placeholder) != placeholderReplacements.end())
                {
                    inputVar = placeholderReplacements.at(placeholder);
                    replacedPlaceholders.insert(placeholder);
                }
            }
            else if (inputVar.IsOutput() && (visitedFunctions.find(inputVar.Owner().get()) == visitedFunctions.end()))
                inputVar.Owner()->ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);
        }
    }

    // Replace any PlaceHolder Variables in the graph of Functions underlying 'this' CompositeFunction. All PlaceHolder variables
    // should have been replaced before performing any Forward compute of 'this' Function.
    /*virtual*/ void CompositeFunction::ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                                            std::unordered_set<const Function*>& visitedFunctions,
                                                            std::unordered_set<Placeholder>& replacedPlaceholders)
    {
        RootFunction()->ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        // If any of the placeholders were replaced with Output variables, let's add the graph of function underneath each of those to 'm_allPrimitiveFunctions' set
        for (auto replacedPlaceholder : replacedPlaceholders)
        {
            auto replacingVariable = placeholderReplacements.at(replacedPlaceholder);
            if (replacingVariable.IsOutput())
            {
                auto ownerFunc = replacingVariable.Owner();
                std::unordered_set<FunctionPtr> visitedFunctions;
                DetermineInputs(ownerFunc, visitedFunctions);

                // Add the newly visited functions to 'm_allPrimitiveFunctions' set
                m_allPrimitiveFunctions.insert(visitedFunctions.begin(), visitedFunctions.end());
            }
        }
    }

    // Recursively create a sub-network of ComputationNode instances corresponding to the graph of Functions 
    // underlying the specified 'variable' and return the ComputationNode instance that corresponds to the 
    // top level 'variable'
    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetNode(const Variable& variable,
                                                                 Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                 ComputationNetworkBuilder<ElementType>& builder,
                                                                 std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap,
                                                                 std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        auto iter = variableToNodeMap.find(variable);
        if (iter != variableToNodeMap.end())
            return iter->second;

        // Lets add a null entry in the map for this variable, to break infinite recursion when processing recurrent graphs
        variableToNodeMap[variable] = nullptr;

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
        else if (variable.IsInput())
        {
            // TODO: Support inputs with > 1 dynamic axes
            if (variable.DynamicAxes().size() != 1)
                LogicError("Currently only Input variables with one dynamic axis are supported");

            auto dynamicAxis = variable.DynamicAxes()[0];
            if (dynamicAxis != Axis::DefaultDynamicAxis())
                LogicError("Currently only Input variables with DefaultDynamicAxis are supported");
            if (IsSparseInput(variable))
                computationNodePtr = builder.CreateSparseInputNode(variable.Name(), AsTensorShape(variable.Shape()));
            else
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
            assert(variable.IsOutput());
            computationNodePtr = GetOutputVariableNode(variable, network, builder, variableToNodeMap, isVariableRootMap)->template As<ComputationNode<ElementType>>()->shared_from_this();
        }

        variableToNodeMap[variable] = computationNodePtr;
        isVariableRootMap[variable] = variable.IsOutput();
        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetOutputVariableNode(const Variable& variable,
                                                                               Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                               ComputationNetworkBuilder<ElementType>& builder,
                                                                               std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap,
                                                                               std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        assert(variable.IsOutput());

        Function* function = variable.Owner().get();
        ComputationNodeBasePtr computationNodePtr;
        if (dynamic_cast<PrimitiveFunction*>(function))
        {
            PrimitiveFunction* primitiveFunction = dynamic_cast<PrimitiveFunction*>(function);

            // Create the nodes corresponding to the inputs
            auto functionInputs = primitiveFunction->Inputs();
            auto input0BaseNodePtr = GetNode(functionInputs[0], network, builder, variableToNodeMap, isVariableRootMap);
            std::shared_ptr<ComputationNode<ElementType>> input0Node = (input0BaseNodePtr != nullptr) ? input0BaseNodePtr->template As<ComputationNode<ElementType>>()->shared_from_this() : nullptr;

            std::shared_ptr<ComputationNode<ElementType>> input1Node;
            if (functionInputs.size() > 1)
            {
                auto input1BaseNodePtr = GetNode(functionInputs[1], network, builder, variableToNodeMap, isVariableRootMap);
                input1Node = (input1BaseNodePtr != nullptr) ? input1BaseNodePtr->template As<ComputationNode<ElementType>>()->shared_from_this() : nullptr;
            }

            PrimitiveOpType op = primitiveFunction->OpType();
            switch (op)
            {
            case PrimitiveOpType::Negate:
                computationNodePtr = builder.Negate(input0Node, function->Name());
                break;
            case PrimitiveOpType::Sigmoid:
                computationNodePtr = builder.Sigmoid(input0Node, function->Name());
                break;
            case PrimitiveOpType::Tanh:
                computationNodePtr = builder.Tanh(input0Node, function->Name());
                break;
            case PrimitiveOpType::ReLU:
                computationNodePtr = builder.RectifiedLinear(input0Node, function->Name());
                break;
            case PrimitiveOpType::Exp:
                computationNodePtr = builder.Exp(input0Node, function->Name());
                break;
            case PrimitiveOpType::Log:
                computationNodePtr = builder.Log(input0Node, function->Name());
                break;
            case PrimitiveOpType::Sqrt:
                computationNodePtr = builder.Sqrt(input0Node, function->Name());
                break;
            case PrimitiveOpType::Floor:
                computationNodePtr = builder.Floor(input0Node, function->Name());
                break;
            case PrimitiveOpType::Abs:
                computationNodePtr = builder.Abs(input0Node, function->Name());
                break;
            case PrimitiveOpType::Reciprocal:
                computationNodePtr = builder.Reciprocal(input0Node, function->Name());
                break;
            case PrimitiveOpType::Softmax:
                if (functionInputs[0].Shape().NumAxes() > 1)
                    InvalidArgument("Softmax operation can only be applied to a 1D input");

                computationNodePtr = builder.Softmax(input0Node, function->Name());
                break;
            case PrimitiveOpType::Plus:
                computationNodePtr = builder.Plus(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Minus:
                computationNodePtr = builder.Minus(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::ElementTimes:
                computationNodePtr = builder.ElementTimes(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Equal:
                computationNodePtr = builder.Equal(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::NotEqual:
                computationNodePtr = builder.NotEqual(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Less:
                computationNodePtr = builder.Less(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::LessEqual:
                computationNodePtr = builder.LessEqual(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Greater:
                computationNodePtr = builder.Greater(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::GreaterEqual:
                computationNodePtr = builder.GreaterEqual(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::Times:
                // TODO: The output rank of the times operation is currently hardcoded to 1
                computationNodePtr = builder.Times(input0Node, input1Node, 1, function->Name());
                break;
            case PrimitiveOpType::SquaredError:
                computationNodePtr = builder.SquareError(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::CrossEntropyWithSoftmax:
                computationNodePtr = builder.CrossEntropyWithSoftmax(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::ClassificationError:
                computationNodePtr = builder.ErrorPrediction(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::PastValue:
            case PrimitiveOpType::FutureValue:
            {
                Variable initialStateVar = functionInputs[0];
                Variable inputOperandVar = functionInputs[1];
                // TODO: Current we only support a scalar initial state
                if (!initialStateVar.IsConstant() || (initialStateVar.Shape().NumAxes() > 0))
                    LogicError("Currently PastValue/FutureValue Function only supports scalar initial state");

                // TODO: We currently only support input operand with 1 static axis for PastValue/FutureValue
                if (inputOperandVar.Shape().NumAxes() != 1)
                    LogicError("Currently PastValue/FutureValue Function only supports input operand with 1 static axis");

                // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
                if (inputOperandVar.DynamicAxes().size() != 1)
                    LogicError("Currently PastValue/FutureValue Function only supports input operand with 1 dynamic axis");

                // Get the intial state of the PastValue/FutureValue operation
                ElementType initStateValue;
                NDArrayView tempView({}, &initStateValue, 1, DeviceDescriptor::CPUDevice());
                tempView.CopyFrom(*Constant(initialStateVar).Value());

                if (op == PrimitiveOpType::PastValue)
                    computationNodePtr = builder.PastValue(input1Node, (float)initStateValue, inputOperandVar.Shape()[0], primitiveFunction->FunctionConfig()[L"stepSize"].GetValue<size_t>(), function->Name());
                else
                    computationNodePtr = builder.FutureValue(input1Node, (float)initStateValue, inputOperandVar.Shape()[0], primitiveFunction->FunctionConfig()[L"stepSize"].GetValue<size_t>(), function->Name());

                break;
            }
            case PrimitiveOpType::ReduceSum:
            {
                // TODO: Use the new ReduceElements node instead of the legacy SumElements node for reduction. Currently ReduceElements has incorrect MBLayout inference.
                //computationNodePtr = network->AddNodeToNetAndAttachInputs(New<ReduceElementsNode<ElementType>>(network->GetDeviceId(), function->Name(), L"Sum", 0), { input0Node });
                computationNodePtr = builder.Sum(input0Node, function->Name());
                break;
            }
            case PrimitiveOpType::Combine:
                // This operation is just a no-op and is a means to combine multiple functions to create a single Function
                // whose outputs are a union of tyhe outputs of the Functions being combined.
                for (auto inputVar : functionInputs)
                    GetNode(inputVar, network, builder, variableToNodeMap, isVariableRootMap);

                computationNodePtr = variableToNodeMap[variable];

                break;
            default:
                LogicError("Specified op %s not yet supported", PrimitiveOpTypeName(op));
                break;
            }

            if (op != PrimitiveOpType::Combine)
            {
                for (auto inputVar : functionInputs)
                    isVariableRootMap[inputVar] = false;
            }
        }
        else
        {
            LogicError("User defined Functions are currently unsupported!");
        }

        return computationNodePtr;
    }

    template <typename ElementType>
    ComputationNetworkPtr CompositeFunction::GetComputationNetwork(const DeviceDescriptor& device, const std::unordered_set<Variable>& backpropRoots)
    {
        if (m_computationNetwork != nullptr)
        {
            // TODO: We should either invalidate and readapt the network if he backpropRoots change compared to what was specified when the network
            // was last constructed, to just recreate a new network.
            // For now just disallow changing the backpropRoots after the network is created
            if (m_currentBackpropRoots != backpropRoots)
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
            if (backpropRoots.size() > 1)
                LogicError("More than one backprop roots is currently unsupported");

            ComputationNodeBasePtr backpropRootNode;

            // Now recursively create the network in a top-down fashion
            auto rootFunction = RootFunction();
            auto rootFunctionOutputs = rootFunction->Outputs();
            std::vector<ComputationNodeBasePtr> forwardRootNodes;
            for (auto rootOutput : rootFunctionOutputs)
            {
                auto currentRootNode = GetNode(rootOutput, m_computationNetwork, builder, m_variableToNodeMap, m_isVariableRootMap);
                forwardRootNodes.push_back(currentRootNode);

                if (backpropRoots.find(rootOutput) != backpropRoots.end())
                    backpropRootNode = m_variableToNodeMap[rootOutput];
            }

            // If any of the function outputs is not a root node, we need to explicitly add it to the 'output' group of the ComputationNetwork
            for (auto rootOutput : rootFunctionOutputs)
            {
                if (!m_isVariableRootMap[rootOutput])
                    m_computationNetwork->AddToNodeGroup(L"output", m_variableToNodeMap[rootOutput]);
            }

            m_currentBackpropRoots = backpropRoots;

            // In case of recurrence, the inputs of some of the ComputationNodes are not attached due to cycles.
            // Now attach those after we have created all ComputationNodes in the network
            for (auto varNodePair : m_variableToNodeMap)
            {
                auto currentComputationNodeInputs = varNodePair.second->GetInputs();

                // TODO: Can any node other than a non PastValue/FutureValue Function have a null input attached after the first pass is finished?
                if (std::find(currentComputationNodeInputs.begin(), currentComputationNodeInputs.end(), nullptr) != currentComputationNodeInputs.end())
                {
                    // We found a null input; this variable must correspond to a PastValue or FutureValue function
                    const PrimitiveFunction* primitiveFunc = dynamic_cast<const PrimitiveFunction*>(varNodePair.first.Owner().get());
                    if ((primitiveFunc == nullptr) || ((primitiveFunc->OpType() != PrimitiveOpType::PastValue) && (primitiveFunc->OpType() != PrimitiveOpType::FutureValue)))
                        InvalidArgument("Invalid Function graph detected; recurrence found at a Function that is not a PastValue/FutureValue function");

                    // The 2nd input of the PastValue/FutureValue function denotes the recurrent input
                    auto actualInput = m_variableToNodeMap[primitiveFunc->Inputs()[1]];
                    varNodePair.second->AttachInputs({ actualInput });
                }
            }

            m_computationNetwork->CompileNetwork();

            // Verify that the shapes of the output Variables that we computed match the corresponding nodes in the ComputationNetwork
            for (auto varNodePair : m_variableToNodeMap)
            {
                if (varNodePair.first.IsOutput())
                {
                    auto outputVar = varNodePair.first;
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

    template <typename ElementType>
    /*static*/ std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CompositeFunction::GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value)
    {
        if (var.GetDataType() != value->Data()->GetDataType())
            LogicError("The Variable's DataType %s does not match the corresponding Value's DataType %s", DataTypeName(var.GetDataType()), DataTypeName(value->Data()->GetDataType()));

        if (AsDataType<ElementType>() != value->Data()->GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(value->Data()->GetDataType()));

        // TODO: Is supplying dense data for an Input variable tagged as sparse, a fatal error?
        if (IsSparseInput(var) && !value->Data()->IsSparse())
            InvalidArgument("Dense input data supplied for a sparse input Variable");

        if (IsSparseInput(var) && (value->Data()->GetStorageFormat() != StorageFormat::SparseCSC))
            InvalidArgument("Sparse Input data must be in SparseCSC format");

        if (value->Data()->Shape().NumAxes() == var.Shape().NumAxes())
            return{ value->Data()->GetMatrix<ElementType>(), nullptr };

        if (value->Data()->Shape().NumAxes() != (var.Shape().NumAxes() + var.DynamicAxes().size() + 1))
            InvalidArgument("Value's number of axes should be larger than the Variable's number of axes by 1 + number of dynamic axes");

        if (var.DynamicAxes().size() > 1)
            LogicError("More than one dynamic axis for a variable is currently unsupported");

        size_t maxNumTimeSteps = value->Data()->Shape()[var.Shape().NumAxes()];
        size_t numSequences = value->Data()->Shape()[var.Shape().NumAxes() + 1];

        auto mask = value->Mask();
        if ((mask != nullptr) && ((var.Shape().NumAxes() + mask->Shape().NumAxes()) != value->Data()->Shape().NumAxes()))
            InvalidArgument("Invalid Value object; the sum of the #axes of the mask and data does not equal the Variable's number of axes by 1 + number of dynamic axes");

        if ((numSequences == 1) || (maxNumTimeSteps == 1))
        {
            // The data need not be shuffled
            std::shared_ptr<const Matrix<ElementType>> matrixData = value->Data()->GetMatrix<ElementType>(var.Shape().NumAxes());
            auto layout = std::make_shared<MBLayout>();
            if (maxNumTimeSteps == 1)
                layout->InitAsFrameMode(numSequences);
            else
            {
                layout->Init(1, maxNumTimeSteps);
                layout->AddSequence(0, 0, 0, maxNumTimeSteps);
            }

            return{ matrixData , layout};
        }
        else
        {
            std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
            if (mask != nullptr)
            {
                // Determine the sequence lengths from the mask
                std::unique_ptr<char[]> maskData(mask->GetMatrix()->CopyToArray());
                for (size_t i = 0; i < numSequences; ++i)
                {
                    size_t currentSequenceLength = 0;
                    bool currentSequenceEndAlreadyFound = false;
                    for (size_t j = 0; j < maxNumTimeSteps; ++j)
                    {
                        if (maskData[(i * maxNumTimeSteps) + j] == 1)
                        {
                            if (currentSequenceEndAlreadyFound)
                                InvalidArgument("Invalid Value object; only trailing steps of a sequence can be masked");

                            currentSequenceLength++;
                        }
                        else
                        {
                            currentSequenceEndAlreadyFound = true;
                        }
                    }

                    sequenceLengths[i] = currentSequenceLength;
                }
            }

            // The data needs to be rearranged since CNTK requires sequences to be interleaved across timesteps
            std::vector<MBLayout::SequenceInfo> sequences;
            for (size_t i = 0; i < numSequences; ++i)
                sequences.push_back({ i, SIZE_MAX, 0, sequenceLengths[i]});

            auto layout = std::make_shared<MBLayout>();
            std::vector<std::pair<size_t, size_t>> placement;
            std::vector<size_t> rowAllocations;
            layout->InitAsPackedSequences(sequences, placement, rowAllocations);
            if (maxNumTimeSteps != layout->GetNumTimeSteps())
                LogicError("The number of time steps in the packed MBLayout does not match the longest sequence's length in the Value object");

            if (numSequences != layout->GetNumSequences())
                LogicError("The number of sequences in the packed MBLayout does not match the sequence count in the Value object");

            // Now generate the gather indices
            auto matrixData = std::make_shared<Matrix<ElementType>>(var.Shape().TotalSize(),
                                                                    layout->GetNumCols(),
                                                                    AsCNTKImplDeviceId(value->Data()->Device()),
                                                                    value->Data()->IsSparse() ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                    AsCNTKImplMatrixFormat(value->Data()->GetStorageFormat()));

            std::vector<size_t> sequencesShorterThanLongestSequence;
            for (size_t i = 0; i < numSequences; ++i)
                if (sequenceLengths[i] != maxNumTimeSteps)
                    sequencesShorterThanLongestSequence.push_back(i);

            // Set the source location for all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
            size_t sourceColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
            std::vector<ElementType> gatherIndicesVector(layout->GetNumCols(), (ElementType)sourceColIdxForInvalidColumns);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t targetParallelStreamIdx = placement[i].first;
                size_t targetStartIdxInParallelStream = placement[i].second;
                for (size_t j = 0; j < sequenceLengths[i]; ++j)
                    gatherIndicesVector[((targetStartIdxInParallelStream + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = (ElementType)((i * maxNumTimeSteps) + j);
            }

            auto gatherIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), gatherIndicesVector.data(), AsCNTKImplDeviceId(value->Data()->Device()));
            matrixData->DoGatherColumnsOf(0, *gatherIdxMatrix, *(value->Data()->GetMatrix<ElementType>(var.Shape().NumAxes())), 1);
            return{ matrixData, layout };
        }
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        NDShape valueDataShape = sampleShape;
        if (layout != nullptr)
            valueDataShape = valueDataShape.AppendShape({ layout->GetNumTimeSteps(), layout->GetNumSequences() });

        // No data shuffling needed if no layout or the layout has just one time-step or just one sequence
        if ((layout == nullptr) || (layout->GetNumTimeSteps() == 1) || (layout->GetNumSequences() == 1))
        {
            // Just create a view over the existing matrix itself
            auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorShape(valueDataShape));
            auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), valueDataShape, readOnly, tensorView);
            return MakeSharedObject<Value>(data);
        }

        if (layout->GetNumCols() != matrix.GetNumCols())
            LogicError("Bad MBLayout: The number of columns in the MBLayout does not match the number of columns in the data matrix!");

        size_t maxNumTimeSteps = layout->GetNumTimeSteps();
        size_t numSequences = layout->GetNumSequences();

        std::vector<size_t> sequenceLengths;
        auto& layoutSequences = layout->GetAllSequences();
        for (auto sequenceInfo : layoutSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
                sequenceLengths.push_back(sequenceInfo.GetNumTimeSteps());
        }

        // Reshuffle to data to unpack and uninterleave the CNTK form data
        // Now generate the gather indices
        auto shuffledMatrixData = std::make_shared<Matrix<ElementType>>(matrix.GetNumRows(), maxNumTimeSteps * numSequences, matrix.GetDeviceId());

        std::vector<size_t> sequencesShorterThanLongestSequence;
        for (size_t i = 0; i < numSequences; ++i)
            if (sequenceLengths[i] != maxNumTimeSteps)
                sequencesShorterThanLongestSequence.push_back(i);

        // Set the target location of all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
        size_t targetColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
        std::vector<ElementType> scatterIndicesVector(layout->GetNumCols(), (ElementType)targetColIdxForInvalidColumns);
        size_t i = 0;
        for (auto sequenceInfo : layoutSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
            {
                size_t targetParallelStreamIdx = sequenceInfo.s;
                size_t targetStartIdxInParallelStream = sequenceInfo.tBegin;
                for (size_t j = 0; j < sequenceInfo.GetNumTimeSteps(); ++j)
                    scatterIndicesVector[((targetStartIdxInParallelStream + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = (ElementType)((i * maxNumTimeSteps) + j);

                i++;
            }
        }

        auto scatterIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), scatterIndicesVector.data(), matrix.GetDeviceId());
        shuffledMatrixData->DoScatterColumnsOf(0, *scatterIdxMatrix, matrix, 1);

        // Create the mask if needed
        NDMaskPtr mask;
        if (!sequencesShorterThanLongestSequence.empty())
        {
            mask = MakeSharedObject<NDMask>(NDShape({ maxNumTimeSteps, numSequences }), AsDeviceDescriptor(matrix.GetDeviceId()));
            for (auto shortSequenceIdx : sequencesShorterThanLongestSequence)
            {
                mask->MaskSection({ sequenceLengths[shortSequenceIdx], shortSequenceIdx }, { NDShape::InferredDimension, 1 });
            }
        }

        auto tensorView = new TensorView<ElementType>(shuffledMatrixData, AsTensorShape(valueDataShape));
        auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), StorageFormat::Dense, valueDataShape, readOnly, tensorView);
        return MakeSharedObject<Value>(data, mask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        if (var.DynamicAxes().size() > 1)
            LogicError("More than one dynamic axis for a variable is currently unsupported");

        if (AsDataType<ElementType>() != var.GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(var.GetDataType()));

        if ((layout != nullptr) && (matrix.GetNumRows() != var.Shape().TotalSize()))
            LogicError("Unexpected matrix layout: The number of rows in the matrix does not match the sample size of the Variable");

        return GetValueObjectFromCNTKImplMatrixAndMBLayout(var.Shape(), matrix, layout, readOnly);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeValue(const std::pair<Variable, ValuePtr>& variableValue, ComputationNodeBasePtr& computationNode)
    {
        auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableValue.first, variableValue.second);
        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;

        auto& nodeData = computationNode->As<ComputationNode<ElementType>>()->Value();

        // Switch the node matrix to the right matrix type
        nodeData.SwitchToMatrixType(CNTKMatrixAndMBLayout.first->GetMatrixType(), CNTKMatrixAndMBLayout.first->GetFormat(), false);
        nodeData.AssignValuesOf(*CNTKMatrixAndMBLayout.first);
        computationNode->GetMBLayout()->CopyFrom(layout);
    }

    void CompositeFunction::PopulateNetworkInputs(const std::unordered_map<Variable, ValuePtr>& arguments)
    {
        auto functionArguments = this->Arguments();
        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto argument : functionArguments)
        {
            // Ensure we have values for all arguments of the function
            if (arguments.find(argument) == arguments.end())
                InvalidArgument("Value not specified for required Function Argument");

            auto argumentComputationNode = m_variableToNodeMap[argument];
            inputNodes.push_back(argumentComputationNode);

            ValuePtr argumentValue = arguments.at(argument);

            MBLayoutPtr layout;
            switch (argumentValue->Data()->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeValue<float>({ argument, argumentValue }, argumentComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeValue<double>({ argument, argumentValue }, argumentComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(argumentValue->Data()->GetDataType()));
                break;
            }
        }

        m_computationNetwork->BumpEvalTimeStamp(inputNodes);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeGradient(const std::pair<Variable, ValuePtr>& variableGradient, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode)
    {
        auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableGradient.first, variableGradient.second);
        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;
        auto nodeLayout = computationNode->GetMBLayout();
        if (((layout == nullptr) != (nodeLayout == nullptr)) || ((layout != nullptr) && (*layout != *nodeLayout)))
            InvalidArgument("The layout of the specified gradient Value in incompatible with the layout of the corresponding Variable computed during Forward call");
        computationNode->As<ComputationNode<ElementType>>()->AssignGradient(*CNTKMatrixAndMBLayout.first);
    }

    // Assign the supplied gradients corresponding to the root(s) of the network to be backpropagated through the graph
    void CompositeFunction::PopulateNetworkGradients(const std::unordered_map<Variable, ValuePtr>& gradients)
    {
        auto functionOutputs = this->Outputs();
        for (auto gradientVarValuePair : gradients)
        {
            // Only gradients for roots of the function can be specified
            if (std::find(functionOutputs.begin(), functionOutputs.end(), gradientVarValuePair.first) == functionOutputs.end())
                InvalidArgument("Gradients cannot be specified for a Variable that is not an Output of the Function");

            auto outputComputationNode = m_variableToNodeMap[gradientVarValuePair.first];
            ValuePtr gradientValue = gradientVarValuePair.second;

            switch (gradientValue->Data()->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeGradient<float>(gradientVarValuePair, outputComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeGradient<double>(gradientVarValuePair, outputComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(gradientValue->Data()->GetDataType()));
                break;
            }
        }
    }

    static NDShape GetValueShape(const Variable& var, const ComputationNodeBasePtr& computationNodePtr)
    {
        size_t outputValueNumAxes = var.Shape().NumAxes();

        // Add the batch and dynamic axes if needed
        if (computationNodePtr->GetMBLayout() != nullptr)
            outputValueNumAxes += 2;

        std::vector<size_t> outputShapeDims(outputValueNumAxes);
        for (size_t i = 0; i < var.Shape().NumAxes(); ++i)
            outputShapeDims[i] = computationNodePtr->GetSampleLayout().GetDim(i);

        if (computationNodePtr->GetMBLayout() != nullptr)
        {
            outputShapeDims[var.Shape().NumAxes()] = computationNodePtr->GetMBLayout()->GetNumTimeSteps();
            outputShapeDims[var.Shape().NumAxes() + 1] = computationNodePtr->GetMBLayout()->GetNumSequences();
        }

        return NDShape(outputShapeDims);
    }

    void CompositeFunction::GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs)
    {
        // Now copy the Forward values of output nodes from the network to outputs' Value objects
        for (auto outputVarValuePair : outputs)
        {
            auto computationNodePtr = m_variableToNodeMap[outputVarValuePair.first];
            auto outputValuePtr = outputVarValuePair.second;

            auto outputShape = GetValueShape(outputVarValuePair.first, computationNodePtr);
            if (outputValuePtr != nullptr)
            {
                // TODO: The shape of the specified output Value object must match the actual output shape
                if (outputValuePtr->Data()->Shape() != outputShape)
                    InvalidArgument("The shape %s of the specified Value object for output does not match the actual output shape %s", AsString(outputValuePtr->Data()->Shape()).c_str(), AsString(outputShape).c_str());
            }

            ValuePtr nodeValue;
            switch (outputVarValuePair.first.GetDataType())
            {
            case DataType::Float:
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(outputVarValuePair.first, computationNodePtr->As<ComputationNode<float>>()->Value(), computationNodePtr->GetMBLayout());
                break;
            case DataType::Double:
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(outputVarValuePair.first, computationNodePtr->As<ComputationNode<double>>()->Value(), computationNodePtr->GetMBLayout());
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(outputVarValuePair.first.GetDataType()));
                break;
            }

            if (outputValuePtr == nullptr)
            {
                auto data = MakeSharedObject<NDArrayView>(outputVarValuePair.first.GetDataType(), outputShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId()));
                auto mask = (nodeValue->Mask() != nullptr) ? MakeSharedObject<NDMask>(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()) : nullptr;
                outputValuePtr = MakeSharedObject<Value>(data, mask);
            }
            outputValuePtr->CopyFrom(*nodeValue);
            outputs[outputVarValuePair.first] = outputValuePtr;
        }
    }

    void CompositeFunction::GetNetworkGradients(std::unordered_map<Variable, ValuePtr>& gradients)
    {
        auto networkInputs = this->Inputs();
        // Now copy the gradient values of input nodes of the network to gradients' Value objects
        for (auto gradientVarValuePair : gradients)
        {
            // Only gradients corresponding to inputs of the network can be obtained
            if (std::find(networkInputs.begin(), networkInputs.end(), gradientVarValuePair.first) == networkInputs.end())
                InvalidArgument("Backpropagated gradient values can only be obtained for inputs of a Function");

            // Gradients can only be obtained for parameter variables or input variables that NeedsGradient
            if (!gradientVarValuePair.first.NeedsGradient())
                InvalidArgument("Gradient value incorrectly requested for an Output or Constant Variable, or an Input Variable with NeedsGradient setting of false");

            auto computationNodePtr = m_variableToNodeMap[gradientVarValuePair.first];
            auto gradientValuePtr = gradientVarValuePair.second;

            auto gradientShape = GetValueShape(gradientVarValuePair.first, computationNodePtr);
            if (gradientValuePtr != nullptr)
            {
                // TODO: The shape of the specified output Value object must match the actual output shape
                if (gradientValuePtr->Data()->Shape() != gradientShape)
                    InvalidArgument("The shape %s of the specified Value object for gradient does not match the actual gradient shape %s", AsString(gradientValuePtr->Data()->Shape()).c_str(), AsString(gradientShape).c_str());
            }

            if (!computationNodePtr->NeedsGradient())
                LogicError("Backpropagated gradient value cannot be read from a ComputationNode that has NeedsGradient set to false");

            ValuePtr nodeValue;
            switch (gradientVarValuePair.first.GetDataType())
            {
            case DataType::Float:
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(gradientVarValuePair.first, computationNodePtr->As<ComputationNode<float>>()->Gradient(), computationNodePtr->GetMBLayout());
                break;
            case DataType::Double:
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(gradientVarValuePair.first, computationNodePtr->As<ComputationNode<double>>()->Gradient(), computationNodePtr->GetMBLayout());
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(gradientVarValuePair.first.GetDataType()));
                break;
            }

            if (gradientValuePtr == nullptr)
            {
                auto data = MakeSharedObject<NDArrayView>(gradientVarValuePair.first.GetDataType(), gradientShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId()));
                auto mask = (nodeValue->Mask() != nullptr) ? MakeSharedObject<NDMask>(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()) : nullptr;
                gradientValuePtr = MakeSharedObject<Value>(data, mask);
            }

            gradientValuePtr->CopyFrom(*nodeValue);
            gradients[gradientVarValuePair.first] = gradientValuePtr;
        }
    }

    /*virtual*/ BackPropStatePtr CompositeFunction::Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                                            std::unordered_map<Variable, ValuePtr>& outputs,
                                                            const DeviceDescriptor& computeDevice,
                                                            const std::unordered_set<Variable>& outputsToRetainBackwardStateFor)
    {
        // TODO: How about zero argument functions?
        // TODO: We need a better way to determine the ElementType for the network
        auto dataType = arguments.begin()->second->Data()->GetDataType();
        if (dataType == DataType::Float)
            GetComputationNetwork<float>(computeDevice, outputsToRetainBackwardStateFor);
        else
            GetComputationNetwork<double>(computeDevice, outputsToRetainBackwardStateFor);

        // TODO: Avoid copying the data when possible

        // Feed data into the arguments of the network
        PopulateNetworkInputs(arguments);

        std::unordered_set<Variable> functionOutputs(this->Outputs().begin(), this->Outputs().end());
        std::vector<ComputationNodeBasePtr> outputsToEvaluate;

        for (auto outputVarValuePair : outputs)
        {
            // Ensure that only a subset of this function's outputs are being asked to be evaluated
            if (functionOutputs.find(outputVarValuePair.first) == functionOutputs.end())
                InvalidArgument("Requested output is not an Ouptut of the Function");

            auto outputComputationNode = m_variableToNodeMap[outputVarValuePair.first];
            outputsToEvaluate.push_back(outputComputationNode);
        }

        // The 'outputsToRetainBackwardStateFor' nodes also need to be evaluated if not already specified in 'outputs'
        for (auto rootVarForBackprop : outputsToRetainBackwardStateFor)
        {
            if (outputs.find(rootVarForBackprop) == outputs.end())
                outputsToEvaluate.push_back(m_variableToNodeMap[rootVarForBackprop]);
        }

        m_computationNetwork->ForwardProp(outputsToEvaluate);

        GetNetworkOutputs(outputs);

        // TODO: How to deal with the specified 'computeDevice'

        return (outputsToRetainBackwardStateFor.size() > 0) ? MakeSharedObject<CNTKBackPropState>(this->shared_from_this(), std::make_pair(arguments.begin()->first, m_variableToNodeMap[arguments.begin()->first]->GetEvalTimeStamp())) : nullptr;
    }

    /*virtual*/ void CompositeFunction::Backward(const BackPropStatePtr& state,
                                                 const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                                                 std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs)
    {
        auto backpropState = dynamic_cast<const CNTKBackPropState*>(state.get());
        if (backpropState == nullptr)
            InvalidArgument("Invalid backprop state specified");

        // TODO: Support multiple concurrent backprop states
        if (backpropState->EvalTimeStamp().second != m_variableToNodeMap[backpropState->EvalTimeStamp().first]->GetEvalTimeStamp())
            LogicError("The specified backprop state specified cannot be used for backpropagation as the Function's internal state was modified by subsequent Forward calls to the function."
                       "This is not a user error but a shortcoming of the current implementation where multiple independent backprop states are not simultaneously supported");

        if (rootGradientValues.size() > 1)
            LogicError("Currently gradient backprop from only one of the Function Outputs is supported");

        // TODO: Avoid copying the data when possible

        // Zero all gradients of nodes below the root nodes
        for (auto rootGradientVarValuePair : rootGradientValues)
            m_computationNetwork->ZeroInputGradients(m_variableToNodeMap[rootGradientVarValuePair.first]);

        // Feed data into the arguments of the network
        PopulateNetworkGradients(rootGradientValues);

        // Backpropagate through the network
        auto rootComputationNodePtr = m_variableToNodeMap[rootGradientValues.begin()->first];
        m_computationNetwork->GetNestedNetwork(rootComputationNodePtr)->Backprop(FrameRange(nullptr), true, true);

        GetNetworkGradients(backPropagatedGradientValuesForInputs);

        // TODO: How to deal with the specified 'computeDevice'
    }

    FunctionPtr UnaryOp(PrimitiveOpType op, const Variable& operand, Dictionary&& opConfig, const std::wstring& name)
    {
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, std::vector<Variable>({ operand }), std::move(opConfig), name), name);
    }

    FunctionPtr Negate(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Negate, operand, Dictionary(), name);
    }

    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Sigmoid, operand, Dictionary(), name);
    }

    FunctionPtr Tanh(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Tanh, operand, Dictionary(), name);
    }

    FunctionPtr ReLU(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::ReLU, operand, Dictionary(), name);
    }

    FunctionPtr Exp(const Variable& operand, const std::wstring& name/* = L""*/)
        {
        return UnaryOp(PrimitiveOpType::Exp, operand, Dictionary(), name);
    }

    FunctionPtr Log(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Log, operand, Dictionary(), name);
        }

    FunctionPtr Square(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return ElementTimes(operand, operand, name);
    }

    FunctionPtr Sqrt(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Sqrt, operand, Dictionary(), name);
    }

    FunctionPtr Round(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return Floor(Plus(operand, Constant(NDShape({}), 0.5f)), name);
    }

    FunctionPtr Floor(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Floor, operand, Dictionary(), name);
    }

    FunctionPtr Ceil(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return Negate(Floor(Negate(operand)), name);
    }

    FunctionPtr Abs(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Abs, operand, Dictionary(), name);
    }

    FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Reciprocal, operand, Dictionary(), name);
    }

    FunctionPtr Softmax(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::Softmax, operand, Dictionary(), name);
    }

    FunctionPtr BinaryOp(PrimitiveOpType op, const Variable& leftOperand, const Variable& rightOperand, Dictionary&& opConfig, const std::wstring& name)
    {
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, std::vector<Variable>({ leftOperand, rightOperand }), std::move(opConfig), name), name);
    }

    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Plus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Minus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::ElementTimes, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return ElementTimes(leftOperand, Reciprocal(rightOperand), name);
    }

    FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Equal, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::NotEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Less, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::LessEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Greater, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::GreaterEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::Times, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::SquaredError, prediction, targets, Dictionary(), name);
    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::CrossEntropyWithSoftmax, prediction, labels, Dictionary(), name);
    }

    FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name/* = L""*/)
    {
        return BinaryOp(PrimitiveOpType::ClassificationError, prediction, labels, Dictionary(), name);
    }

    FunctionPtr PastValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 1)
            InvalidArgument("PastValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic axis");

        auto additionalProperties = Dictionary();
        additionalProperties[L"stepSize"] = DictionaryValue(stepSize);
        return BinaryOp(PrimitiveOpType::PastValue, initialState, operand, std::move(additionalProperties), name);
    }

    FunctionPtr FutureValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 1)
            InvalidArgument("FutureValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic axis");

        auto additionalProperties = Dictionary();
        additionalProperties[L"stepSize"] = DictionaryValue(stepSize);
        return BinaryOp(PrimitiveOpType::FutureValue, initialState, operand, std::move(additionalProperties), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return UnaryOp(PrimitiveOpType::ReduceSum, operand, Dictionary(), name);
    }

    FunctionPtr Combine(const std::vector<FunctionPtr>& operands, const std::wstring& name/* = L""*/)
    {
        std::unordered_set<FunctionPtr> uniqueOperands;
        std::vector<Variable> inputs;
        for (auto operand : operands)
        {
            if (uniqueOperands.find(operand) != uniqueOperands.end())
                LogicError("All function operands specified to Combine must be unique");

            uniqueOperands.insert(operand);
            auto currentFunctionOutputs = operand->Outputs();
            std::copy(currentFunctionOutputs.begin(), currentFunctionOutputs.end(), std::back_inserter(inputs));
        }

        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Combine, inputs, Dictionary(), name), name);
    }
}
