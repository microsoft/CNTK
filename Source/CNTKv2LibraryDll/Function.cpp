//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveFunction.h"
#include "CompositeFunction.h"

using namespace Microsoft::MSR::CNTK;

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

    Function::Function(const std::vector<Variable>& inputs, const std::vector<Variable>& outputs, Dictionary&& functionConfig, const FunctionPtr& rootFunction, const std::wstring& name, const std::wstring& uid)
        : m_rootFunction(rootFunction), m_name(name != L"" ? name : uid), m_uid(uid), m_attributes(std::move(functionConfig))
    {
        for (auto inputVar : inputs)
        {
            m_inputs.push_back(inputVar);

            if (!inputVar.IsInput() &&
                !inputVar.IsOutput() &&
                !inputVar.IsParameter() &&
                !inputVar.IsConstant() &&
                !inputVar.IsPlaceholder())
            {
                InvalidArgument("Function input has invalid VariableKind!");
            }
        }

        std::unordered_set<Variable> uniqueOutputs;
        for (auto outputVar : outputs)
        {
            if (uniqueOutputs.find(outputVar) != uniqueOutputs.end())
                RuntimeError("Same variable appears multiple times in the outputs vector passed to Function constructor");

            if (m_rootFunction == nullptr && outputVar.IsOutput() && outputVar.m_dataFields->m_ownerFunction == this)
            {
                // in case of a primitive function, set uid of output vars to owner function uid + "_Output_" + output index.
                outputVar.m_dataFields->m_uid = m_uid + L"_" + VariableKindName(outputVar.Kind()) + L"_" + std::to_wstring(m_outputs.size());
            }

            m_outputs.push_back(outputVar);
            uniqueOutputs.insert(outputVar);
        }
    }

    /*virtual*/ Function::~Function() {}

    // Placeholders can be replaced incrementally - i.e. not all placeholders need to replaced in one go.
    // The only requirement is that they must all be replaced before making any 'Forward' calls on the Function instance.
    /*virtual*/ void Function::ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                          std::unordered_set<const Function*>& visitedFunctions,
                                                          std::unordered_set<Variable>& replacedPlaceholders)
    {
        visitedFunctions.insert(this);

        auto replacePlaceholder = [&placeholderReplacements, &replacedPlaceholders](Variable& var) {
            if (var.IsPlaceholder())
            {
                auto placeholder = var;
                if (placeholderReplacements.find(placeholder) != placeholderReplacements.end())
                {
                    var = placeholderReplacements.at(placeholder);
                    replacedPlaceholders.insert(placeholder);

                    // If shape or dynamic axes of the placeholder are known but unknown in the replacement, we update the replacement's shape/dynamic axes
                    if (var.Shape().IsUnknown() && !placeholder.Shape().IsUnknown())
                        var.m_dataFields->m_shape = placeholder.Shape();

                    if ((var.DynamicAxes() == Axis::UnknownDynamicAxes()) && (placeholder.DynamicAxes() != Axis::UnknownDynamicAxes()))
                        var.m_dataFields->m_dynamicAxes = placeholder.DynamicAxes();
                }
            }
        };

        for (auto& inputVar : m_inputs)
        {
            replacePlaceholder(inputVar);

            if (inputVar.IsOutput() && (visitedFunctions.find(inputVar.Owner().get()) == visitedFunctions.end()))
            {
                inputVar.Owner()->ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);
            }
        }

        auto primitiveFunction = dynamic_cast<PrimitiveFunction*>(this);
        if (primitiveFunction != nullptr && primitiveFunction->OpType() == PrimitiveOpType::Combine)
        {
            // Again, combine needs a special treatment, since m_inputs and m_outputs are two 
            // different vectors, and m_outputs is created by copying m_inputs content, there
            // still can be placeholders cached in m_outputs, need to replace those as well.
            for (auto& outputVar : m_outputs)
            {
                replacePlaceholder(outputVar);
            }
        }
    }

    void Function::ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions, bool& recurrentNodeOutputModified)
    {
        auto primitiveFunction = dynamic_cast<PrimitiveFunction*>(this);
        if (primitiveFunction == nullptr)
            LogicError("ValidateOrUpdateOutputs currently only supported for PrimitiveFunction instances");

        assert(visitedFunctions.find(this) == visitedFunctions.end());
        visitedFunctions[this] = 1;

        // Validate each of the inputs first
        for (auto input : m_inputs)
        {
            if (input.IsOutput())
            {
                auto owner = input.Owner().get();
                if (visitedFunctions.find(owner) == visitedFunctions.end())
                    owner->ValidateOrUpdateOutputs(visitedFunctions, recurrentNodeOutputModified);
                else
                    visitedFunctions[owner]++;
            }
        }

        auto outputsUsingNewInputs = PrimitiveFunction::GetOutputVariables(primitiveFunction->OpType(), m_inputs, this, primitiveFunction->m_attributes, true, primitiveFunction->Name());
        auto currentOutputs = Outputs();
        for (size_t i = 0; i < currentOutputs.size(); ++i)
        {
            auto newOutputVar = outputsUsingNewInputs[i];
            auto currentOutputVar = currentOutputs[i];

            if (visitedFunctions[this] > 1)
            {
                if (!newOutputVar.Shape().IsUnknown() && (currentOutputVar.Shape() != newOutputVar.Shape()))
                {
                    recurrentNodeOutputModified = true;
                    currentOutputVar.m_dataFields->m_shape = newOutputVar.Shape();
                }

                if ((currentOutputVar.GetDataType() == DataType::Unknown) && (currentOutputVar.GetDataType() != newOutputVar.GetDataType()))
                {
                    recurrentNodeOutputModified = true;
                    currentOutputVar.m_dataFields->m_dataType = newOutputVar.GetDataType();
                }

                if ((currentOutputVar.DynamicAxes() == Axis::UnknownDynamicAxes()) && (currentOutputVar.DynamicAxes() != newOutputVar.DynamicAxes()))
                {
                    recurrentNodeOutputModified = true;
                    currentOutputVar.m_dataFields->m_dynamicAxes = newOutputVar.DynamicAxes();
                }

                if ((!newOutputVar.Shape().IsUnknown() && (currentOutputVar.Shape() != newOutputVar.Shape())) ||
                    ((newOutputVar.GetDataType() != DataType::Unknown) && (currentOutputVar.GetDataType() != newOutputVar.GetDataType())) ||
                    ((newOutputVar.DynamicAxes() != Axis::UnknownDynamicAxes()) && (currentOutputVar.DynamicAxes() != newOutputVar.DynamicAxes())))
                {
                    InvalidArgument("Inconsistency in output variable shape, DataType or Dynamic axes computed after replaced placeholders vs. existing output properties, for the Recurrent Function");
                }
            }
            else
            {
                currentOutputVar.m_dataFields->m_shape = newOutputVar.Shape();
                currentOutputVar.m_dataFields->m_dataType = newOutputVar.GetDataType();
                currentOutputVar.m_dataFields->m_dynamicAxes = newOutputVar.DynamicAxes();
            }
        }
    }

    void Function::SaveModel(const std::wstring& modelFilePath, bool usinglegacyModelFormat)
    {
        if (usinglegacyModelFormat)
        {
            Internal::SaveAsLegacyModel(shared_from_this(), modelFilePath);
        }
        else
        {
            Dictionary model = Serialize();
            auto stream = GetFstream(modelFilePath, false);
            *stream << model;
            stream->flush();
        }
    }

    /*static*/ FunctionPtr Function::LoadModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice)
    {
        auto stream = GetFstream(modelFile, true);
        if (!Internal::IsLegacyModel(*stream))
        {
            Dictionary model;
            *stream >> model;
            return Function::Deserialize(model, computeDevice);
        }
        else
        {
            return Internal::LoadLegacyModel(modelFile, computeDevice);
        }
    }

    void Function::RestoreModel(const std::wstring& modelFilePath)
    {
        auto stream = GetFstream(modelFilePath, true);
        if (!Internal::IsLegacyModel(*stream))
        {
            Dictionary model;
            *stream >> model;
            RestoreFromCheckpoint(model);
            return;
        }

        auto loadedModelFunction = Internal::LoadLegacyModel(modelFilePath, DeviceDescriptor::CPUDevice());

        // TODO: Make sure that the loaded model is the same as the trainer's model through UID matching in the V2 format
        // TODO: For V1 format models make sure that the loaded model is isomorphic to the trainer's model
        auto loadedModelLeafVariables = loadedModelFunction->Inputs();
        auto trainerModelLeafVariables = Inputs();
        if (trainerModelLeafVariables.size() != loadedModelLeafVariables.size())
            InvalidArgument("The loaded model's leaf variables do not match the trainer model's leaf variables");

        std::map<std::wstring, Variable> loadedModelLeafVariablesMap;
        for (auto leafVar : loadedModelLeafVariables)
            loadedModelLeafVariablesMap[leafVar.Uid()] = leafVar;

        std::map<std::wstring, Variable> trainerModelLeafVariablesMap;
        for (auto leafVar : trainerModelLeafVariables)
            trainerModelLeafVariablesMap[leafVar.Uid()] = leafVar;

        // Remove the initial state inputs of PastValue and FutureValue functions from the maps if they are a scalar constant
        // since these are not part of the internal CNTK serialized computation graph
        auto removePastAndFutureValueInitialStateScalarConstants = [](const std::unordered_set<FunctionPtr>& allPrimitiveFunctions, std::map<std::wstring, Variable>& modelLeafVariableMap) {
            for (auto funcPtr : allPrimitiveFunctions)
            {
                auto primitiveFunction = dynamic_cast<const PrimitiveFunction*>(funcPtr.get());
                if ((primitiveFunction->OpType() == PrimitiveOpType::PastValue) || (primitiveFunction->OpType() == PrimitiveOpType::FutureValue))
                {
                    auto initialStateInput = primitiveFunction->Inputs()[1];
                    if (initialStateInput.IsConstant() && (initialStateInput.Shape().TotalSize() == 1))
                        modelLeafVariableMap.erase(initialStateInput.Uid());
                }
            }
        };

        auto loadedModelCompositeFunction = dynamic_cast<const CompositeFunction*>(loadedModelFunction.get());
        removePastAndFutureValueInitialStateScalarConstants(loadedModelCompositeFunction->m_allPrimitiveFunctions, loadedModelLeafVariablesMap);

        auto trainerModelCompositeFunction = dynamic_cast<CompositeFunction*>(this);
        removePastAndFutureValueInitialStateScalarConstants(trainerModelCompositeFunction->m_allPrimitiveFunctions, trainerModelLeafVariablesMap);

        // Now update the trainer's model parameters and constants with those from the loaded model
        for (auto nameVarPair : trainerModelLeafVariablesMap)
        {
            auto trainerModelLeafVar = nameVarPair.second;

            auto areVariablesEquivalent = [](const Variable& left, const Variable& right) {
                return Internal::AreEquivalent(left, right) && (left.Uid() == right.Uid());
            };

            auto correspondingLoadedModelVar = loadedModelLeafVariablesMap.at(trainerModelLeafVar.Uid());

            if (!areVariablesEquivalent(correspondingLoadedModelVar, trainerModelLeafVar))
                InvalidArgument("The loaded model's leaf variables do not match the trainer model's leaf variables");

            if ((trainerModelLeafVar.IsConstant() && !Constant(trainerModelLeafVar).Value()->IsReadOnly()) || trainerModelLeafVar.IsParameter())
            {
                auto trainerModelVarValue = trainerModelLeafVar.IsConstant() ? Constant(trainerModelLeafVar).Value() : Parameter(trainerModelLeafVar).Value();
                auto loadedModelVarValue = correspondingLoadedModelVar.IsConstant() ? Constant(correspondingLoadedModelVar).Value() : Parameter(correspondingLoadedModelVar).Value();
                trainerModelVarValue->CopyFrom(*loadedModelVarValue);
            }
        }

        trainerModelCompositeFunction->CopyState(*loadedModelCompositeFunction);
    }

    static Variable GetCorrespondingOutputVariableFromClone(const Variable& cloneeOutput, const FunctionPtr& cloneeFunction, const FunctionPtr& clonedFunction)
    {
        size_t outputVarIndex = 0;
        for (auto output : cloneeFunction->Outputs())
        {
            if (output == cloneeOutput)
                break;

            outputVarIndex++;
        }

        return clonedFunction->Outputs()[outputVarIndex];
    }

    FunctionPtr Function::ReplacePlaceholder(const Variable& placeholderReplacement)
    {
        auto placeholders = Placeholders();
        if (placeholders.size() != 1)
            InvalidArgument("Function::ReplacePlaceholders called with a single replacement variable but this Function has %d placeholders", (int)placeholders.size());

        return ReplacePlaceholders({ { *(placeholders.begin()), placeholderReplacement } });
    }

    FunctionPtr Function::ReplacePlaceholders(const std::unordered_map<Variable, Variable>& placeholderReplacements)
    {
        std::unordered_set<const Function*> visitedFunctions;
        std::unordered_set<Variable> replacedPlaceholders;
        ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        for (auto replacementPair : placeholderReplacements)
        {
            if (replacedPlaceholders.find(replacementPair.first) == replacedPlaceholders.end())
                InvalidArgument("At least one of the placeholders specified for replacement was not found in the function");
        }

        return this->shared_from_this();
    }

    FunctionPtr Function::Clone(const FunctionPtr& clonee,
                                ParameterCloningMethod parameterCloneMethod,
                                const std::unordered_map<Variable, Variable>& replacements,
                                std::unordered_map<const Function*, FunctionPtr>& cloneMap,
                                std::unordered_map<Variable, Variable>& leafVariablesCloneMap,
                                std::unordered_map<Variable, Variable>& placeholderReplacements)
    {
        const PrimitiveFunction* primitiveFunction = dynamic_cast<const PrimitiveFunction*>(clonee.get());
        if (primitiveFunction == nullptr)
            LogicError("Currently cloning of user defined Functions is unsupported");

        if (cloneMap.find(clonee.get()) != cloneMap.end())
            LogicError("Cloning an already visited Function");

        cloneMap[clonee.get()] = nullptr;

        std::vector<Variable> inputs;
        auto cloneeInputs = clonee->Inputs();
        for (auto cloneeInput : cloneeInputs)
        {
            Variable clonedInput;
            if (replacements.find(cloneeInput) != replacements.end())
            {
                clonedInput = PlaceholderVariable(cloneeInput.Shape(), cloneeInput.DynamicAxes());
                placeholderReplacements[clonedInput] = replacements.at(cloneeInput);
            }
            else
                {
                // This is not a replacement. Lets create a fresh clone

                // Is it a leaf
                if (cloneeInput.IsInput() || cloneeInput.IsConstant() || cloneeInput.IsPlaceholder() || cloneeInput.IsParameter())
                {
                    if (leafVariablesCloneMap.find(cloneeInput) != leafVariablesCloneMap.end())
                        clonedInput = leafVariablesCloneMap.at(cloneeInput);
                    else
                    {
                        if (cloneeInput.IsParameter() || cloneeInput.IsConstant())
                        {
                            switch (parameterCloneMethod)
                            {
                            case ParameterCloningMethod::Clone:
                                clonedInput = cloneeInput.Clone();
                                leafVariablesCloneMap[cloneeInput] = clonedInput;
                                break;
                            case ParameterCloningMethod::Share:
                                clonedInput = cloneeInput;
                                break;
                            case ParameterCloningMethod::Freeze:
                                if (cloneeInput.IsParameter())
                                    clonedInput = Constant(Parameter(cloneeInput).Value(), cloneeInput.Name());
                                else
                                    clonedInput = Constant(Constant(cloneeInput).Value(), cloneeInput.Name());

                                leafVariablesCloneMap[cloneeInput] = clonedInput;
                                break;
                            default:
                                LogicError("Unknown ParameterCloningMethod");
                            }
                        }
                        else
                        {
                            clonedInput = cloneeInput.Clone();
                            leafVariablesCloneMap[cloneeInput] = clonedInput;
                        }
                    }
                }
                else
                {
                    assert(cloneeInput.IsOutput());

                    // If this is an already visited Function's output then we use a placeholder
                    if (cloneMap.find(cloneeInput.Owner().get()) != cloneMap.end())
                    {
                        if (cloneMap.at(cloneeInput.Owner().get()) == nullptr)
                        {
                            // See if we already created a placeholder for this already visited Function's output
                            auto existingPlaceholderReplacement = std::find_if(placeholderReplacements.begin(), placeholderReplacements.end(), [cloneeInput](const std::pair<Variable, Variable>& placeholderReplacement) {
                                return (placeholderReplacement.second == cloneeInput);
                            });

                            if (existingPlaceholderReplacement == placeholderReplacements.end())
                            {
                                clonedInput = PlaceholderVariable(cloneeInput.Shape(), cloneeInput.DynamicAxes());
                                placeholderReplacements[clonedInput] = cloneeInput;
                            }
                            else
                                clonedInput = existingPlaceholderReplacement->first;
                        }
                        else
                            clonedInput = GetCorrespondingOutputVariableFromClone(cloneeInput, cloneeInput.Owner(), cloneMap.at(cloneeInput.Owner().get()));
                    }
                    else
                    {
                        auto clonedFunction = Clone(cloneeInput.Owner(), parameterCloneMethod, replacements, cloneMap, leafVariablesCloneMap, placeholderReplacements);
                        clonedInput = GetCorrespondingOutputVariableFromClone(cloneeInput, cloneeInput.Owner(), clonedFunction);
                    }
                }
            }

            inputs.push_back(clonedInput);
        }

        Dictionary attributesCopy(primitiveFunction->Attributes());
        auto clonedFunction = MakeSharedObject<PrimitiveFunction>(primitiveFunction->OpType(), inputs, std::move(attributesCopy), primitiveFunction->Name());
        cloneMap[primitiveFunction] = clonedFunction;

        return clonedFunction;
    }

    FunctionPtr Function::Clone(ParameterCloningMethod parameterCloneMethod, const std::unordered_map<Variable, Variable>& replacements) const
    {
        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        if (compositeFunction == nullptr)
            LogicError("Currently only cloning of composite functions is supported");

        std::unordered_map<const Function*, FunctionPtr> cloneMap;
        std::unordered_map<Variable, Variable> leafVariablesCloneMap;
        std::unordered_map<Variable, Variable> placeholderReplacements;
        auto clonedRootFunction = Function::Clone(compositeFunction->RootFunction(), parameterCloneMethod, replacements, cloneMap, leafVariablesCloneMap, placeholderReplacements);

        // Patch the values in the placeholderReplacements map with newly cloned Variables where applicable
        std::unordered_set<FunctionPtr> replacementClones;
        for (auto& varPair : placeholderReplacements)
        {
            if (varPair.second.IsOutput())
            {
                if (cloneMap.find(varPair.second.Owner().get()) != cloneMap.end())
                    placeholderReplacements[varPair.first] = GetCorrespondingOutputVariableFromClone(varPair.second, varPair.second.Owner(), cloneMap.at(varPair.second.Owner().get()));
                else
                {
                    // Check if any of the inputs or intermediate functions in the graph underneath the replacement
                    // are one of the cloned Functions or inputs; if so then we should clone the graph underneath
                    // this replacement too
                    std::unordered_set<FunctionPtr> visitedFunctions;
                    auto visitedLeaves = CompositeFunction::DetermineInputs(varPair.second.Owner(), visitedFunctions);
                    std::unordered_map<Variable, Variable> cloningReplacementsForPlaceholderReplacement;
                    for (auto visitedLeaf : visitedLeaves)
                    {
                        if (leafVariablesCloneMap.find(visitedLeaf) != leafVariablesCloneMap.end())
                            cloningReplacementsForPlaceholderReplacement[visitedLeaf] = leafVariablesCloneMap[visitedLeaf];
                    }

                    for (auto visitedFunction : visitedFunctions)
                    {
                        if (cloneMap.find(visitedFunction.get()) != cloneMap.end())
                        {
                            auto visitedFunctionOutputs = visitedFunction->Outputs();
                            for (auto visitedFunctionOutput : visitedFunctionOutputs)
                                cloningReplacementsForPlaceholderReplacement[visitedFunctionOutput] = GetCorrespondingOutputVariableFromClone(visitedFunctionOutput, visitedFunction, cloneMap.at(visitedFunction.get()));
                        }
                    }

                    if (!cloningReplacementsForPlaceholderReplacement.empty())
                    {
                        auto replacementToClone = CompositeFunction::Create(varPair.second.Owner());
                        auto replacementClone = replacementToClone->Clone(parameterCloneMethod, cloningReplacementsForPlaceholderReplacement);
                        replacementClones.insert(replacementClone);
                        placeholderReplacements[varPair.first] = GetCorrespondingOutputVariableFromClone(varPair.second, varPair.second.Owner(), replacementClone->RootFunction());
                    }
                }
            }
            else
            {
                if (leafVariablesCloneMap.find(varPair.second) != leafVariablesCloneMap.end())
                    placeholderReplacements[varPair.first] = leafVariablesCloneMap.at(varPair.second);
            }
        }

        auto clonedComposite = CompositeFunction::Create(clonedRootFunction, compositeFunction->Name());
        clonedComposite->ReplacePlaceholders(placeholderReplacements);
        return clonedComposite;
    }


    /*virtual*/ void Function::RestoreFromCheckpoint(const Dictionary& modelDictionary)
    {
        CompositeFunction* compositeFunction = dynamic_cast<CompositeFunction*>(this);
        if (compositeFunction == nullptr)
            InvalidArgument("Primitive (aka non-composite) Function instances cannot be restored from a checkpoint.");

        auto restoredFunction = Function::Deserialize(modelDictionary, DeviceDescriptor::CPUDevice());

        if (!Internal::AreEquivalent(shared_from_this(), restoredFunction))
            InvalidArgument("'This' function is not equivalent (isomorphic) to the function restored from a checkpoint.");

        auto parameters = Parameters();
        auto restoredParameters = restoredFunction->Parameters();

        assert(parameters.size() == restoredParameters.size());

        for (int i = 0; i < parameters.size(); i++)
        {
            assert(Internal::AreEquivalent(parameters[i], restoredParameters[i]));

            // Additionally, to be super-safe, compare parameter UIDs.
            if (parameters[i].Uid() != restoredParameters[i].Uid())
            {
                 InvalidArgument("'This' function parameters and restored function parameters do not have identical UIDs.");
            }

            parameters[i].Value()->CopyFrom(*(restoredParameters[i].Value().get()));
        }

        auto restoredCompositeFunction = dynamic_cast<const CompositeFunction*>(restoredFunction.get());
        compositeFunction->CopyState(*restoredCompositeFunction);
    }

    /*static*/ FunctionPtr Function::Deserialize(const Dictionary& modelDictionary, const CNTK::DeviceDescriptor& device)
    {
        return CompositeFunction::Deserialize(modelDictionary, device);
    }

    void Function::PrintGraph() const
    {
        CompositeFunction::Traverse(RootFunction(), [](const FunctionPtr& function) {
        });
    }

    FunctionPtr UnaryOp(PrimitiveOpType op, const Variable& operand, Dictionary&& opConfig, const std::wstring& name)
    {
        std::vector<Variable> operands = { operand };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, operands, std::move(opConfig), name), name);
    }

    FunctionPtr Negate(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Negate, operand, Dictionary(), name);
    }

    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Sigmoid, operand, Dictionary(), name);
    }

    FunctionPtr Tanh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Tanh, operand, Dictionary(), name);
    }

    FunctionPtr Sin(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Sin, operand, Dictionary(), name);
    }

    FunctionPtr Cos(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Cos, operand, Dictionary(), name);
    }

    FunctionPtr ReLU(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::ReLU, operand, Dictionary(), name);
    }

    FunctionPtr Exp(const Variable& operand, const std::wstring& name)
        {
        return UnaryOp(PrimitiveOpType::Exp, operand, Dictionary(), name);
    }

    FunctionPtr Log(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Log, operand, Dictionary(), name);
        }

    FunctionPtr Square(const Variable& operand, const std::wstring& name)
    {
        return ElementTimes(operand, operand, name);
    }

    FunctionPtr Sqrt(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Sqrt, operand, Dictionary(), name);
    }

    FunctionPtr Round(const Variable& operand, const std::wstring& name)
    {
        return Floor(Plus(operand, Constant::Scalar(0.5f)), name);
    }

    FunctionPtr Floor(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Floor, operand, Dictionary(), name);
    }

    FunctionPtr Ceil(const Variable& operand, const std::wstring& name)
    {
        return Negate(Floor(Negate(operand)), name);
    }

    FunctionPtr Abs(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Abs, operand, Dictionary(), name);
    }

    FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Reciprocal, operand, Dictionary(), name);
    }

    FunctionPtr Softmax(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Softmax, operand, Dictionary(), name);
    }

    FunctionPtr Hardmax(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Hardmax, operand, Dictionary(), name);
    }

    FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis1] = axis1;
        additionalProperties[PrimitiveFunction::AttributeNameAxis2] = axis2;
        return UnaryOp(PrimitiveOpType::TransposeAxes, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Transpose(const Variable& operand, const std::wstring& name)
    {
        if (operand.Shape().Rank() <= 2)
            InvalidArgument("Transpose can already be called for 1D or 2D operands");

        return TransposeAxes(operand, Axis(0), Axis(1), name);
    }

    FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name)
    {
        if (axis.IsStaticAxis())
        {
            if ((endIndex - beginIndex) <= 0)
                InvalidArgument("CNTK::Slice: endIndex (%d) - beginIndex (%d) must be a positive number", endIndex, beginIndex);

            return Internal::Slice(operand, axis, beginIndex, endIndex, name);
        }

        if (axis == Axis::DefaultBatchAxis())
            LogicError("Slice is currently unsupported along the batch axis");

        LogicError("CNTK::Slice: Invalid axis argument provided. To slice a sequence along its ordered dynamic axis use Sequence::Slice.");
    }

    FunctionPtr RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNumSamples] = numSamples;
        additionalProperties[PrimitiveFunction::AttributeNameAllowDuplicates] = allowDuplicates;

        return UnaryOp(PrimitiveOpType::RandomSample, operand, std::move(additionalProperties), name);
    }

    FunctionPtr RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNumSamples] = numSamples;
        additionalProperties[PrimitiveFunction::AttributeNameAllowDuplicates] = allowDuplicates;

        return UnaryOp(PrimitiveOpType::RandomSampleInclusionFrequency, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Dropout(const Variable& operand, double dropoutRate, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameDropoutRate] = dropoutRate;

        return UnaryOp(PrimitiveOpType::Dropout, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNewShape] = newShape;

        return UnaryOp(PrimitiveOpType::Reshape, operand, std::move(additionalProperties), name);
    }

    FunctionPtr BinaryOp(PrimitiveOpType op, const Variable& leftOperand, const Variable& rightOperand, Dictionary&& opConfig, const std::wstring& name)
    {
        std::vector<Variable> operands = { leftOperand, rightOperand };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(op, operands, std::move(opConfig), name), name);
    }

    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Plus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::LogPlus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Minus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::ElementTimes, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return ElementTimes(leftOperand, Reciprocal(rightOperand), name);
    }

    FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Equal, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::NotEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Less, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::LessEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Greater, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::GreaterEqual, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, int inferInputRankToMap, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOutputRank] = outputRank;
        additionalProperties[PrimitiveFunction::AttributeNameInferInputRankToMap] = inferInputRankToMap;
        return BinaryOp(PrimitiveOpType::Times, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank /*= 1*/, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOutputRank] = outputRank;
        return BinaryOp(PrimitiveOpType::TransposeTimes, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr CosineDistance(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::CosDistance, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name)
    {
        std::vector<Variable> operands = { prediction, targets };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Logistic, operands, Dictionary(), name), name);
    }

    FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name)
    {
        std::vector<Variable> operands = { prediction, targets, weights };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Logistic, operands, Dictionary(), name), name);
    }

    FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name)
    {
        auto difference = Minus(prediction, targets);
        auto squaredDifference = ElementTimes(difference, difference);
        return Internal::ReduceElements(squaredDifference, PrimitiveFunction::InternalSumReductionOpName, Axis::AllStaticAxes(), name);
    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name)
    {
        if (axis == Axis(0))
            return Minus(ReduceLogSum(prediction, axis), TransposeTimes(labels, prediction), name);
        else
            return Minus(ReduceLogSum(prediction, axis), ReduceSum(ElementTimes(labels, prediction), axis), name);
    }

    FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name)
    {
        if (topN == 0)
            InvalidArgument("ClassificationError: The topN argument must be > 0!");

        if (topN == 1)
        {
            if (axis == Axis(0))
                return Minus(Constant::Scalar(1.0f), TransposeTimes(labels, Hardmax(prediction)), name);
            else
            {
                auto axMax = ReduceMax(prediction, axis);
                auto pred = Equal(prediction, axMax);
                auto wrongPred = NotEqual(labels, pred);
                auto axErr = ReduceSum(wrongPred, axis);
                auto capErr = GreaterEqual(axErr, Constant::Scalar(1.0f));
                return ReduceMean(capErr, Axis::AllStaticAxes(), name);
            }
        }
        else
        {
            if (axis != Axis(0))
                LogicError("ClassificationError along a specific axis does not support topN!");

            std::vector<Variable> operands = { prediction, labels, Constant::Scalar((float)topN) };
            return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::ClassificationError, operands, Dictionary(), name), name);
        }
    }

    FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::PastValue, operand, initialState, std::move(additionalProperties), name);
    }

    FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::FutureValue, operand, initialState, std::move(additionalProperties), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::SumAll, operand, Dictionary(), name);
    }

    FunctionPtr ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalLogSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMeanReductionOpName, axis, name);
    }

    FunctionPtr ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMaxReductionOpName, axis, name);
    }

    FunctionPtr ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMinReductionOpName, axis, name);
    }
    FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name)
    {
        Constant meanVar(mean);
        Constant invStdDevVar(invStdDev);

        return ElementTimes(Minus(operand, meanVar), invStdDevVar, name);
    }

    FunctionPtr Convolution(const Variable& convolutionMap,
                            const Variable& operand,
                            const NDShape& strides,
                            const std::vector<bool>& sharing,
                            const std::vector<bool>& autoPadding,
                            const NDShape& lowerPad,
                            const NDShape& upperPad,
                            bool transpose,
                            size_t maxTempMemSizeInSamples,
                            const std::wstring& name)
    {
        // Currently we require that the Convolution function's operand have a dynamic axis since otherwise
        // the internal implementation incorrectly infers the batch axis dimension by picking up the first axis as 
        // the sample shape and considering the rest to be part of the batch axis
        if (operand.DynamicAxes().empty())
            LogicError("Convolution currently requires the main operand to have dynamic axes");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = lowerPad;
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = upperPad;
        additionalProperties[PrimitiveFunction::AttributeNameTranspose] = transpose;
        additionalProperties[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples] = maxTempMemSizeInSamples;

        return BinaryOp(PrimitiveOpType::Convolution, convolutionMap, operand, std::move(additionalProperties), name);
    }

    FunctionPtr ROIPooling(const Variable& convolutionMap, const Variable& rois, const NDShape& roiOutputShape, const std::wstring& name/* = L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameROIOutputShape] = roiOutputShape;
        return BinaryOp(PrimitiveOpType::ROIPooling, convolutionMap, rois, std::move(additionalProperties), name);
    }

    FunctionPtr Pooling(const Variable& operand,
                        PoolingType poolingType,
                        const NDShape& poolingWindowShape,
                        const NDShape& strides,
                        const std::vector<bool>& autoPadding,
                        const NDShape& lowerPad,
                        const NDShape& upperPad,
                        const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePoolingType] = (size_t)poolingType;
        additionalProperties[PrimitiveFunction::AttributeNamePoolingWindowShape] = poolingWindowShape;
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = lowerPad;
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = upperPad;

        return UnaryOp(PrimitiveOpType::Pooling, operand, std::move(additionalProperties), name);
    }

    FunctionPtr BatchNormalization(const Variable& operand,
                                   const Variable& scale,
                                   const Variable& bias,
                                   const Variable& runningMean,
                                   const Variable& runningInvStd,
                                   bool spatial,
                                   double normalizationTimeConstant,
                                   double blendTimeConstant,
                                   double epsilon,
                                   bool useCuDNNEngine,
                                   const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameSpatial] = spatial;
        additionalProperties[PrimitiveFunction::AttributeNameNormalizationTimeConstant] = normalizationTimeConstant;
        additionalProperties[PrimitiveFunction::AttributeNameBlendTimeConstant] = blendTimeConstant;
        additionalProperties[PrimitiveFunction::AttributeNameEpsilon] = epsilon;
        additionalProperties[PrimitiveFunction::AttributeNameUseCuDNNEngine] = useCuDNNEngine;

        std::vector<Variable> operands = { operand, scale, bias, runningMean, runningInvStd };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::BatchNormalization,
                                                                             operands,
                                                                             std::move(additionalProperties),
                                                                             name),
                                         name);
    }

    FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name)
    {
        std::vector<Variable> operands = { operand, min, max };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Clip, operands, Dictionary(), name), name);
    }

    FunctionPtr ElementSelect(const Variable& condition, const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        // TODO: If the condition is a scalar constant, we can just pass-through the appropriate operand
        std::vector<Variable> operands = { condition, leftOperand, rightOperand };
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Select, operands, Dictionary(), name), name);
    }

    FunctionPtr Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;

        std::vector<Variable> operandsCopy = operands;
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Splice, operandsCopy, std::move(additionalProperties), name), name);
    }

    FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name)
    {
        std::unordered_set<Variable> uniqueOperands;
        for (auto operand : operands)
        {
            if (uniqueOperands.find(operand) != uniqueOperands.end())
                LogicError("All operands specified to Combine must be unique");

            uniqueOperands.insert(operand);
        }

        std::vector<Variable> operandsCopy = operands;
        return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Combine, operandsCopy, Dictionary(), name), name);
    }

    FunctionPtr Alias(const Variable& operand, const std::wstring& name)
    {
        // TODO: This is a temporary and expensive hack until we have a real alias implementation
        // that does not waste memory and compute cycles
        return Plus(operand, Constant::Scalar(0.0f), name);
    }

    FunctionPtr OptimizedRNNStack(const Variable& operand, const Variable& weights, size_t hiddenSize, size_t numLayers, bool bidirectional, const std::wstring& recurrentOp, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameHiddenSize] = hiddenSize;
        additionalProperties[PrimitiveFunction::AttributeNameNumLayers] = numLayers;
        additionalProperties[PrimitiveFunction::AttributeNameBidirectional] = bidirectional;
        additionalProperties[PrimitiveFunction::AttributeNameRecurrentOp] = recurrentOp;

        return BinaryOp(PrimitiveOpType::OptimizedRNNStack, operand, weights, std::move(additionalProperties), name);
    }

    namespace Sequence
    {
        void VerifyIsSequence(const Variable& operand)
        {
            // The operand must have at least one dynamic axis
            if (operand.DynamicAxes().empty())
                InvalidArgument("A sequence function can only be applied on operands with at least one dynamic axis and whose first dynamic axis is ordered");
        }

        FunctionPtr IsFirst(const Variable& operand, const std::wstring& name)
        {
            return Internal::IsWithin(operand, 1, name);
        }

        FunctionPtr IsLast(const Variable& operand, const std::wstring& name)
        {
            return Internal::IsWithin(operand, -1, name);
        }

        FunctionPtr Slice(const Variable& operand, int beginIndex, int endIndex, const std::wstring& name)
        {
            VerifyIsSequence(operand);

            if ((beginIndex == 0) && (endIndex == 0))
                return operand;

            auto beginFlagsLambda = [beginIndex, operand]() {
                return (beginIndex > 0) ? Minus(Constant::Scalar(1.0f), Internal::IsWithin(operand, beginIndex)) : Internal::IsWithin(operand, beginIndex);
            };

            auto endFlagsLambda = [endIndex, operand]() {
                return (endIndex > 0) ? Internal::IsWithin(operand, endIndex) : Minus(Constant::Scalar(1.0f), Internal::IsWithin(operand, endIndex));
            };

            FunctionPtr flags;
            if (beginIndex == 0)
                flags = endFlagsLambda();
            else if (endIndex == 0)
                flags = beginFlagsLambda();
            else
                flags = ElementTimes(beginFlagsLambda(), endFlagsLambda());

            int sliceLength = (endIndex - beginIndex);
            size_t multiplicativeFactor = (sliceLength > 0) ? 0 : 1;

            return Internal::Gather(operand, flags, { multiplicativeFactor, sliceLength }, name);
        }

        FunctionPtr First(const Variable& operand, const std::wstring& name)
        {
            return Sequence::Slice(operand, 0, 1, name);
        }

        FunctionPtr Last(const Variable& operand, const std::wstring& name)
        {
            return Sequence::Slice(operand, -1, 0, name);
        }

        FunctionPtr Where(const Variable& condition, const std::wstring& name)
        {
            return UnaryOp(PrimitiveOpType::Where, condition, Dictionary(), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::Gather(operand, condition, name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::Scatter(operand, condition, name);
        }

        FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name)
        {
            auto dataPadded = Internal::Scatter(operand, Sequence::IsFirst(broadcastAs), std::make_pair<size_t, int>(0, 1));
            auto placeHolderOutput = PlaceholderVariable(operand.Shape(), broadcastAs.DynamicAxes());
            auto output = ElementSelect(Sequence::IsFirst(broadcastAs), dataPadded, PastValue(placeHolderOutput), name);
            return output->ReplacePlaceholders({ { placeHolderOutput, output } });
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::wstring& name)
        {
            using namespace std::placeholders;

            std::function<FunctionPtr(const Variable& leftOperand, const Variable& rightOperand)> reductionFunctor;
            if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                reductionFunctor = std::bind(Plus, _1, _2, L"");
            else
                LogicError("%S reduction along dynamic axis is currently unsupported", reductionOpName.c_str());

            // We are reducing over a dynamic axis which is currently implemented using recurrence
            auto cumulativeSumFunctionPlaceholder = PlaceholderVariable(operand.Shape());
            auto prevAccumulatedValuesFunction = PastValue(cumulativeSumFunctionPlaceholder);
            auto cumulativeSumFunction = reductionFunctor(prevAccumulatedValuesFunction, operand);
            cumulativeSumFunction->ReplacePlaceholders({ { cumulativeSumFunctionPlaceholder, cumulativeSumFunction } });

            return Sequence::Slice(cumulativeSumFunction, -1, 0, name);
        }

        FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name)
        {
            return ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, name);
        }
    }

    namespace Internal
    {
        FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name)
        {
            Sequence::VerifyIsSequence(operand);

            if (offset == 0)
                InvalidArgument("CNTK::Sequence::IsWithin: The offset must be positive");

            if (offset > 0)
                return PastValue(Internal::ZeroesWithDynamicAxesLike(operand), Constant::Scalar(1.0f), offset, name);
            else
                return FutureValue(Internal::ZeroesWithDynamicAxesLike(operand), Constant::Scalar(1.0f), -offset, name);
        }

        FunctionPtr PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name)
        {
            return BinaryOp(PrimitiveOpType::PackedIndex, operand, index, Dictionary(), name);
        }

        FunctionPtr GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name)
        {
            return BinaryOp(PrimitiveOpType::GatherPacked, operand, packedIndex, Dictionary(), name);
        }

        FunctionPtr ScatterPacked(const Variable& operand, const Variable& packedIndex, const Variable& condition, const std::wstring& name)
        {
            std::vector<Variable> operands = { operand, packedIndex, condition };
            return CompositeFunction::Create(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::ScatterPacked, operands, Dictionary(), name), name);
        }

        FunctionPtr ZeroesWithDynamicAxesLike(const Variable& operand)
        {
            if (operand.IsSparse())
            {
                if (operand.Shape().Rank() > 1)
                    LogicError("Internal::ZeroesWithDynamicAxesLike: Currently only 1D sparse inputs are supported!");

                // TODO: A matrix multiplication is too expensive for something like this
                // Replace this with a cheaper operation.
                return Times(Constant({ 1, operand.Shape()[0] }, operand.GetDataType(), 0.0), operand);
            }
            else
            {
                auto reduceAllStaticAxesFunc = Internal::ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, Axis::AllStaticAxes());
                return Minus(reduceAllStaticAxesFunc, reduceAllStaticAxesFunc);
            }
        }

        FunctionPtr Where(const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name)
        {
            auto additionalProperties = Dictionary();
            additionalProperties[PrimitiveFunction::AttributeNameNewSequenceAxisLengthScalingFactor] = newDerivedSequenceAxisScalingAndAdditiveFactor.first;
            additionalProperties[PrimitiveFunction::AttributeNameNewSequenceAxisLengthAdditiveFactor] = newDerivedSequenceAxisScalingAndAdditiveFactor.second;
            return UnaryOp(PrimitiveOpType::Where, condition, std::move(additionalProperties), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::GatherPacked(operand, Internal::PackedIndex(/*layout of*/ operand, Sequence::Where(condition)), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name)
        {
            return Internal::GatherPacked(operand, Internal::PackedIndex(/*layout of*/ operand, Where(condition, newDerivedSequenceAxisScalingAndAdditiveFactor)), name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::ScatterPacked(operand, Internal::PackedIndex(/*layout of*/ condition, Sequence::Where(condition)), /*layout of*/ condition, name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name)
        {
            return Internal::ScatterPacked(operand, Internal::PackedIndex(/*layout of*/ condition, Where(condition, newDerivedSequenceAxisScalingAndAdditiveFactor)), /*layout of*/ condition, name);
        }

        FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name)
        {
            auto additionalProperties = Dictionary();
            additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
            additionalProperties[PrimitiveFunction::AttributeNameBeginIndex] = beginIndex;
            additionalProperties[PrimitiveFunction::AttributeNameEndIndex] = endIndex;

            return UnaryOp(PrimitiveOpType::Slice, operand, std::move(additionalProperties), name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name)
        {
            if (axis.IsStaticAxis() || (axis == Axis::AllStaticAxes()))
            {
                auto additionalProperties = Dictionary();
                additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
                additionalProperties[PrimitiveFunction::AttributeNameReductionOpName] = reductionOpName;
                return UnaryOp(PrimitiveOpType::ReduceElements, operand, std::move(additionalProperties), name);
            }

            if (axis == Axis::DefaultBatchAxis())
                LogicError("Reduction is currently unsupported along the batch axis");

            LogicError("CNTK::ReduceElements: Invalid axis argument provided. To reduce a sequence along its ordered dynamic axis use Sequence::ReduceElements.");
        }
   }
}
