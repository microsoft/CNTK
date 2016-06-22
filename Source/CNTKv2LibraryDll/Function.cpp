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
    _Internal::_SimpleVector<Variable> Function::_Inputs() const
    {
        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        if (compositeFunction == nullptr)
            return m_inputs;
        else
            return _Internal::_SimpleVector<Variable>::CreateSimpleVector(compositeFunction->DetermineInputs());
    }

    /*virtual*/ void Function::_ReplacePlaceholders(const _Internal::_SimpleMap<Placeholder, Variable>& placeholderReplacements, _Internal::_SimpleSet<const Function*>& visitedFunctions, _Internal::_SimpleSet<Placeholder>& replacedPlaceholders)
    {
        visitedFunctions.Insert(this);

        for (auto iter = m_inputs.m_vector->begin(); iter != m_inputs.m_vector->end(); ++iter)
        {
            if (iter->IsPlaceholder())
            {
                Placeholder placeholder(*iter);
                if (placeholderReplacements.Contains(placeholder))
                {
                    *iter = placeholderReplacements[placeholder];
                    replacedPlaceholders.Insert(placeholder);
                }
            }
            else if ((iter->Kind() == VariableKind::Output) && !visitedFunctions.Contains(iter->Owner()))
                iter->Owner()->_ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);
        }
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        if (variableToNodeMap.find(variable) != variableToNodeMap.end())
            return variableToNodeMap[variable];

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
        else if (variable.Kind() == VariableKind::Input)
        {
            // TODO: Specify dynamic axis
            if (variable.IsSparseInput())
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
            assert(variable.Kind() == VariableKind::Output);
            computationNodePtr = GetOutputVariableNode(variable, network, builder, variableToNodeMap, isVariableRootMap)->template As<ComputationNode<ElementType>>()->shared_from_this();
        }

        variableToNodeMap[variable] = computationNodePtr;
        isVariableRootMap[variable] = (variable.Kind() == VariableKind::Output);
        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::GetOutputVariableNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap)
    {
        assert(variable.Kind() == VariableKind::Output);

        Function* function = variable.Owner();
        ComputationNodeBasePtr computationNodePtr;
        if (dynamic_cast<PrimitiveFunction*>(function) != nullptr)
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
            case PrimitiveOpType::Tanh:
                computationNodePtr = builder.Tanh(input0Node, function->Name());
                break;
            case PrimitiveOpType::CrossEntropyWithSoftmax:
                computationNodePtr = builder.CrossEntropyWithSoftmax(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::PredictionError:
                computationNodePtr = builder.ErrorPrediction(input1Node, input0Node, function->Name());
                break;
            case PrimitiveOpType::Exp:
                computationNodePtr = builder.Exp(input0Node, function->Name());
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
            case PrimitiveOpType::ElementTimes:
                computationNodePtr = builder.ElementTimes(input0Node, input1Node, function->Name());
                break;
            case PrimitiveOpType::ReduceSum:
            {
                // TODO: Use the new ReduceElements node instead of the legacy SumElements node for reduction. Currently ReduceElements has incorrect MBLayout inference.
                //computationNodePtr = network->AddNodeToNetAndAttachInputs(New<ReduceElementsNode<ElementType>>(network->GetDeviceId(), function->Name(), L"Sum", 0), { input0Node });
                computationNodePtr = builder.Sum(input0Node, function->Name());
                break;
            }
            case PrimitiveOpType::Combine:
                for (size_t i = 0; i < functionInputs.size(); ++i)
                    GetNode(functionInputs[i], network, builder, variableToNodeMap, isVariableRootMap);

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
                auto currentRootNode = GetNode(rootFunctionOutputs[i], m_computationNetwork, builder, m_variableToNodeMap, m_isVariableRootMap);
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

            // In case of recurrence, the inputs of some of the ComputationNodes are not attached due to cycles.
            // Now attach those after we have created all ComputationNodes in the network
            for (auto iter = m_variableToNodeMap.begin(); iter != m_variableToNodeMap.end(); ++iter)
            {
                auto currentComputationNodeInputs = iter->second->GetInputs();

                // TODO: Can any node other than a non PastValue/FutureValue Function have a null input attached after the first pass is finished?
                if (std::find(currentComputationNodeInputs.begin(), currentComputationNodeInputs.end(), nullptr) != currentComputationNodeInputs.end())
                {
                    // We found a null input; this variable must correspond to a PastValue or FutureValue function
                    const PrimitiveFunction* primitiveFunc = dynamic_cast<const PrimitiveFunction*>(iter->first.Owner().GetPtr());
                    if ((primitiveFunc == nullptr) || ((primitiveFunc->OpType() != PrimitiveOpType::PastValue) && (primitiveFunc->OpType() != PrimitiveOpType::FutureValue)))
                        InvalidArgument("Invalid Function graph detected; recurrence found at a Function that is not a PastValue/FutureValue function");

                    // The 2nd input of the PastValue/FutureValue function denotes the recurrent input
                    auto actualInput = m_variableToNodeMap[primitiveFunc->Inputs()[1]];
                    iter->second->AttachInputs({ actualInput });
                }
            }

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

    template <typename ElementType>
    /*static*/ std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CompositeFunction::GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value)
    {
        if (var.GetDataType() != value->Data()->GetDataType())
            LogicError("The Variable's DataType %s does not match the corresponding Value's DataType %s", DataTypeName(var.GetDataType()), DataTypeName(value->Data()->GetDataType()));

        if (AsDataType<ElementType>() != value->Data()->GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(value->Data()->GetDataType()));

        // TODO: Is supplying dense data for an Input variable tagged as sparse, a fatal error?
        if (var.IsSparseInput() && !value->Data()->IsSparse())
            InvalidArgument("Dense input data supplied for a sparse input Variable");

        if (var.IsSparseInput() && (value->Data()->GetStorageFormat() != StorageFormat::SparseCSC))
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
                                                                    AsCNTKMatrixFormat(value->Data()->GetStorageFormat()));

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
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout)
    {
        if (var.DynamicAxes().size() > 1)
            LogicError("More than one dynamic axis for a variable is currently unsupported");

        if (AsDataType<ElementType>() != var.GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(var.GetDataType()));

        if ((layout != nullptr) && (matrix.GetNumRows() != var.Shape().TotalSize()))
            LogicError("Unexpected matrix layout: The number of rows in the matrix does not match the sample size of the Variable");

        NDShape valueDataShape = var.Shape();
        if (layout != nullptr)
            valueDataShape = valueDataShape.AppendShape({ layout->GetNumTimeSteps(), layout->GetNumSequences() });

        // No data shuffling needed if no layout or the layout has just one time-step or just one sequence
        if ((layout == nullptr) || (layout->GetNumTimeSteps() == 1) || (layout->GetNumSequences() == 1))
        {
            // Just create a view over the existing matrix itself
            auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorShape(valueDataShape));
            auto data = NDArrayViewPtr(new NDArrayView(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), valueDataShape, true, tensorView), [](_ReferenceCounter* ptr) { delete ptr; });
            return ValuePtr(new Value(data), [](_ReferenceCounter* ptr) { delete ptr; });
        }

        if (layout->GetNumCols() != matrix.GetNumCols())
            LogicError("Bad MBLayout: The number of columns in the MBLayout does not match the number of columns in the data matrix!");

        size_t maxNumTimeSteps = layout->GetNumTimeSteps();
        size_t numSequences = layout->GetNumSequences();

        std::vector<size_t> sequenceLengths;
        auto& layoutSequences = layout->GetAllSequences();
        for (auto iter = layoutSequences.begin(); iter != layoutSequences.end(); ++iter)
        {
            if (iter->seqId != GAP_SEQUENCE_ID)
                sequenceLengths.push_back(iter->GetNumTimeSteps());
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
        for (auto iter = layoutSequences.begin(); iter != layoutSequences.end(); ++iter)
        {
            if (iter->seqId != GAP_SEQUENCE_ID)
            {
                size_t targetParallelStreamIdx = iter->s;
                size_t targetStartIdxInParallelStream = iter->tBegin;
                for (size_t j = 0; j < iter->GetNumTimeSteps(); ++j)
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
            mask = NDMaskPtr(new NDMask({ maxNumTimeSteps, numSequences }, AsDeviceDescriptor(matrix.GetDeviceId())), [](_ReferenceCounter* ptr) { delete ptr; });
            for (size_t i = 0; i < sequencesShorterThanLongestSequence.size(); ++i)
            {
                size_t shorterSequenceIdx = sequencesShorterThanLongestSequence[i];
                mask->MaskSection({ sequenceLengths[shorterSequenceIdx], shorterSequenceIdx }, { NDShape::InferredDimension, 1 });
            }
        }

        auto tensorView = new TensorView<ElementType>(shuffledMatrixData, AsTensorShape(valueDataShape));
        auto data = NDArrayViewPtr(new NDArrayView(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), StorageFormat::Dense, valueDataShape, true, tensorView), [](_ReferenceCounter* ptr) { delete ptr; });
        return ValuePtr(new Value(data, mask), [](_ReferenceCounter* ptr) { delete ptr; });
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

            MBLayoutPtr layout;
            switch (argumentValue->Data()->GetDataType())
            {
            case DataType::Float:
            {
                auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<float>(*iter, argumentValue);
                layout = CNTKMatrixAndMBLayout.second;

                auto& nodeData = argumentComputationNode->As<ComputationNode<float>>()->Value();
                // Switch the node matrix to the right matrix type
                nodeData.SwitchToMatrixType(CNTKMatrixAndMBLayout.first->GetMatrixType(), CNTKMatrixAndMBLayout.first->GetFormat(), false);
                nodeData.AssignValuesOf(*CNTKMatrixAndMBLayout.first);
                break;
            }
            case DataType::Double:
            {
                auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<double>(*iter, argumentValue);
                layout = CNTKMatrixAndMBLayout.second;

                auto& nodeData = argumentComputationNode->As<ComputationNode<double>>()->Value();
                // Switch the node matrix to the right matrix type
                nodeData.SwitchToMatrixType(CNTKMatrixAndMBLayout.first->GetMatrixType(), CNTKMatrixAndMBLayout.first->GetFormat(), false);
                nodeData.AssignValuesOf(*CNTKMatrixAndMBLayout.first);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(argumentValue->Data()->GetDataType()));
                break;
            }

            argumentComputationNode->GetMBLayout()->CopyFrom(layout);
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
            auto nodeLayout = outputComputationNode->GetMBLayout();

            ValuePtr gradientValue = iter->second;

            MBLayoutPtr layout;
            switch (gradientValue->Data()->GetDataType())
            {
            case DataType::Float:
            {
                auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<float>(iter->first, gradientValue);
                layout = CNTKMatrixAndMBLayout.second;
                if (((layout == nullptr) != (nodeLayout == nullptr)) || ((layout != nullptr) && (*layout != *nodeLayout)))
                    InvalidArgument("The layout of the specified gradient Value in incompatible with the layout of the corresponding Variable computed during Forward call");
                outputComputationNode->As<ComputationNode<float>>()->ResetGradient(*CNTKMatrixAndMBLayout.first);
                break;
            }
            case DataType::Double:
            {
                auto CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<double>(iter->first, gradientValue);
                layout = CNTKMatrixAndMBLayout.second;
                if (((layout == nullptr) != (nodeLayout == nullptr)) || ((layout != nullptr) && (*layout != *nodeLayout)))
                    InvalidArgument("The layout of the specified gradient Value in incompatible with the layout of the corresponding Variable computed during Forward call");
                outputComputationNode->As<ComputationNode<double>>()->ResetGradient(*CNTKMatrixAndMBLayout.first);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(gradientValue->Data()->GetDataType()));
                break;
            }
        }
    }

    static NDShape GetValueShape(const Variable& var, const ComputationNodeBasePtr& computationNodePtr)
    {
        size_t outputValueNumAxes = var.Shape().NumAxes();
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

            switch (iter->first.GetDataType())
            {
            case DataType::Float:
            {
                auto nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(iter->first, computationNodePtr->As<ComputationNode<float>>()->Value(), computationNodePtr->GetMBLayout());
                if (outputValuePtr == nullptr)
                {
                    auto data = NDArrayViewPtr(new NDArrayView(iter->first.GetDataType(), outputShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())), [](_ReferenceCounter* ptr) { delete ptr; });
                    auto mask = (nodeValue->Mask() != nullptr) ? NDMaskPtr(new NDMask(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()), [](_ReferenceCounter* ptr) { delete ptr; }) : nullptr;
                    outputValuePtr = ValuePtr(new Value(data, mask), [](_ReferenceCounter* ptr) { delete ptr; });
                }
                outputValuePtr->CopyFrom(*nodeValue);
                break;
            }
            case DataType::Double:
            {
                auto nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(iter->first, computationNodePtr->As<ComputationNode<double>>()->Value(), computationNodePtr->GetMBLayout());
                if (outputValuePtr == nullptr)
                {
                    auto data = NDArrayViewPtr(new NDArrayView(iter->first.GetDataType(), outputShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())), [](_ReferenceCounter* ptr) { delete ptr; });
                    auto mask = (nodeValue->Mask() != nullptr) ? NDMaskPtr(new NDMask(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()), [](_ReferenceCounter* ptr) { delete ptr; }) : nullptr;
                    outputValuePtr = ValuePtr(new Value(data, mask), [](_ReferenceCounter* ptr) { delete ptr; });
                }
                outputValuePtr->CopyFrom(*nodeValue);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(iter->first.GetDataType()));
                break;
            }

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

            if (!computationNodePtr->NeedsGradient())
                LogicError("Backpropagated gradient value cannot be read from a ComputationNode that has NeedsGradient set to false");

            switch (iter->first.GetDataType())
            {
            case DataType::Float:
            {
                auto nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(iter->first, computationNodePtr->As<ComputationNode<float>>()->Gradient(), computationNodePtr->GetMBLayout());
                if (gradientValuePtr == nullptr)
                {
                    auto data = NDArrayViewPtr(new NDArrayView(iter->first.GetDataType(), gradientShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())), [](_ReferenceCounter* ptr) { delete ptr; });
                    auto mask = NDMaskPtr((nodeValue->Mask() != nullptr) ? new NDMask(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()) : nullptr, [](_ReferenceCounter* ptr) { delete ptr; });
                    gradientValuePtr = ValuePtr(new Value(data, mask), [](_ReferenceCounter* ptr) { delete ptr; });
                }
                gradientValuePtr->CopyFrom(*nodeValue);
                break;
            }
            case DataType::Double:
            {
                auto nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(iter->first, computationNodePtr->As<ComputationNode<double>>()->Gradient(), computationNodePtr->GetMBLayout());
                if (gradientValuePtr == nullptr)
                {
                    auto data = NDArrayViewPtr(new NDArrayView(iter->first.GetDataType(), gradientShape, AsDeviceDescriptor(computationNodePtr->ValuePtr()->GetDeviceId())), [](_ReferenceCounter* ptr) { delete ptr; });
                    auto mask = NDMaskPtr((nodeValue->Mask() != nullptr) ? new NDMask(nodeValue->Mask()->Shape(), nodeValue->Mask()->Device()) : nullptr, [](_ReferenceCounter* ptr) { delete ptr; });
                    gradientValuePtr = ValuePtr(new Value(data, mask), [](_ReferenceCounter* ptr) { delete ptr; });

                }
                gradientValuePtr->CopyFrom(*nodeValue);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(iter->first.GetDataType()));
                break;
            }

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
        auto dataType = arguments.m_map->begin()->second->Data()->GetDataType();
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

        return (outputsToRetainBackwardStateFor.Size() > 0) ? BackPropStatePtr(new CNTKBackPropState(this, { arguments.m_map->begin()->first, m_variableToNodeMap[arguments.m_map->begin()->first]->GetEvalTimeStamp() }), [](_ReferenceCounter* ptr) { delete ptr; }) : nullptr;
    }

    /*virtual*/ void CompositeFunction::Backward(const BackPropStatePtr& state,
                                                 const _Internal::_SimpleMap<Variable, const ValuePtr>& rootGradientValues,
                                                 _Internal::_SimpleMap<Variable, ValuePtr>& backPropagatedGradientValuesForInputs)
    {
        if ((state == nullptr) || (dynamic_cast<const CNTKBackPropState*>(state.GetPtr()) == nullptr))
            InvalidArgument("Invalid backprop state specified");

        // TODO: Support multiple concurrent backprop states
        auto backpropState = dynamic_cast<const CNTKBackPropState*>(state.GetPtr());
        if (backpropState->EvalTimeStamp().second != m_variableToNodeMap[backpropState->EvalTimeStamp().first]->GetEvalTimeStamp())
            LogicError("The specified backprop state specified cannot be used for backpropagation as the Function's internal state was modified by subsequent Forward calls to the function."
                       "This is not a user error but a shortcoming of the current implementation where multiple independent backprop states are not simultaneously supported");

        if (rootGradientValues.Size() > 1)
            LogicError("Currently gradient backprop from only one of the Function Outputs is supported");

        // TODO: Avoid copying the data when possible

        // Zero all gradients of nodes below the root nodes
        for (auto iter = rootGradientValues.m_map->begin(); iter != rootGradientValues.m_map->end(); ++iter)
            m_computationNetwork->ZeroInputGradients(m_variableToNodeMap[iter->first]);

        // Feed data into the arguments of the network
        PopulateNetworkGradients(rootGradientValues);

        // Backpropagate through the network
        auto rootComputationNodePtr = m_variableToNodeMap[rootGradientValues.m_map->begin()->first];
        m_computationNetwork->GetNestedNetwork(rootComputationNodePtr)->Backprop(FrameRange(nullptr), true, true);

        GetNetworkGradients(*(backPropagatedGradientValuesForInputs.m_map));

        // TODO: How to deal with the specified 'computeDevice'
    }

    /*virtual*/ void CompositeFunction::_ReplacePlaceholders(const _Internal::_SimpleMap<Placeholder, Variable>& placeholderReplacements, _Internal::_SimpleSet<const Function*>& visitedFunctions, _Internal::_SimpleSet<Placeholder>& replacedPlaceholders)
    {
        RootFunction()->_ReplacePlaceholders(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        // If any of the placeholders were replaced with Output variables, let's add the graph of function underneath each of those to 'm_allPrimitiveFunctions' set
        for (auto iter = replacedPlaceholders.m_set->begin(); iter != replacedPlaceholders.m_set->end(); ++iter)
        {
            auto replacingVariable = placeholderReplacements[*iter];
            if (replacingVariable.Kind() == VariableKind::Output)
            {
                auto ownerFunc = replacingVariable.Owner();
                _Internal::_SimpleSet<FunctionPtr> visitedFunctions;
                _DetermineInputs(ownerFunc, visitedFunctions);

                // Add the newly visited functions to 'm_allPrimitiveFunctions' set
                m_allPrimitiveFunctions.m_set->insert(visitedFunctions.m_set->begin(), visitedFunctions.m_set->end());
            }
        }
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

    FunctionPtr Tanh(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Tanh, { operand }, Dictionary(), name), name);
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

    FunctionPtr Exp(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::Exp, { operand }, Dictionary(), name), name);
    }

    FunctionPtr PastValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 1)
            InvalidArgument("PastValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic axis");

        auto additionalProperties = Dictionary();
        additionalProperties[L"stepSize"] = DictionaryValue(stepSize);
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::PastValue, { initialState, operand }, std::move(additionalProperties), name), name);
    }

    FunctionPtr FutureValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name/* = L""*/)
    {
        if (operand.DynamicAxes().size() != 1)
            InvalidArgument("FutureValue overload that does not explicitly specify a dynamic axis can only be used for operands with exactly one dynamic axis");

        auto additionalProperties = Dictionary();
        additionalProperties[L"stepSize"] = DictionaryValue(stepSize);
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::FutureValue, { initialState, operand }, std::move(additionalProperties), name), name);
    }

    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::ElementTimes, { leftOperand, rightOperand }, Dictionary(), name), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name/* = L""*/)
    {
        return CompositeFunction::Create(new PrimitiveFunction(PrimitiveOpType::ReduceSum, { operand }, Dictionary(), name), name);
    }
}
