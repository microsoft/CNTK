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
#include "EvaluationNodes.h"
#include "TrainingNodes.h"
#include "LinearAlgebraNodes.h"
#include "InputAndParamNodes.h"
#include "NonlinearityNodes.h"
#include "RecurrentNodes.h"
#include "Value.h"

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

    // Placeholders can be replaced incrementally - i.e. not all placeholders need to replaced in one go.
    // The only requirement is that they must all be replaced before making any 'Forward' calls on the Function instance.
    /*virtual*/ void Function::ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                          std::unordered_set<const Function*>& visitedFunctions,
                                                          std::unordered_set<Variable>& replacedPlaceholders)
    {
        visitedFunctions.insert(this);

        for (auto& inputVar : m_inputs)
        {
            if (inputVar.IsPlaceholder())
            {
                auto placeholder = inputVar;
                if (placeholderReplacements.find(placeholder) != placeholderReplacements.end())
                {
                    inputVar = placeholderReplacements.at(placeholder);
                    replacedPlaceholders.insert(placeholder);
                }
            }
            else if (inputVar.IsOutput() && (visitedFunctions.find(inputVar.Owner().get()) == visitedFunctions.end()))
                inputVar.Owner()->ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);
        }
    }

    void Function::ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions)
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
                    owner->ValidateOrUpdateOutputs(visitedFunctions);
                else
                    visitedFunctions[owner]++;
            }
        }

        bool inferDimensions = (visitedFunctions[this] == 1);
        auto outputsUsingNewInputs = PrimitiveFunction::GetOutputVariables(primitiveFunction->OpType(), m_inputs, this, primitiveFunction->m_attributes, inferDimensions, primitiveFunction->Name());
        auto currentOutputs = Outputs();
        for (size_t i = 0; i < currentOutputs.size(); ++i)
        {
            auto newOutputVar = outputsUsingNewInputs[i];
            auto currentOutputVar = currentOutputs[i];
            if (!inferDimensions)
            {
                if (currentOutputVar.Shape().IsUnknown())
                    currentOutputVar.m_dataFields->m_shape = newOutputVar.Shape();

                if (currentOutputVar.GetDataType() == DataType::Unknown)
                    currentOutputVar.m_dataFields->m_dataType = newOutputVar.GetDataType();

                if (currentOutputVar.DynamicAxes() == Axis::UnknownDynamicAxes)
                    currentOutputVar.m_dataFields->m_dynamicAxes = newOutputVar.DynamicAxes();

                if ((currentOutputVar.Shape() != newOutputVar.Shape()) ||
                    (currentOutputVar.GetDataType() != newOutputVar.GetDataType()) ||
                    (currentOutputVar.DynamicAxes() != newOutputVar.DynamicAxes()))
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

    void Function::RestoreFromLegacyModel(const std::wstring& modelFilePath)
    {
        auto loadedModelFunction = LoadLegacyModel(Outputs()[0].GetDataType(), modelFilePath, DeviceDescriptor::CPUDevice());

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

        auto trainerModelCompositeFunction = dynamic_cast<const CompositeFunction*>(this);
        removePastAndFutureValueInitialStateScalarConstants(trainerModelCompositeFunction->m_allPrimitiveFunctions, trainerModelLeafVariablesMap);

        // Now update the trainer's model parameters and constants with those from the loaded model
        for (auto nameVarPair : trainerModelLeafVariablesMap)
        {
            auto trainerModelLeafVar = nameVarPair.second;

            auto areVariablesEquivalent = [](const Variable& left, const Variable& right) {
                bool areDynamicAxesCompatible = (left.DynamicAxes().size() == right.DynamicAxes().size());
                auto numAxes = left.DynamicAxes().size();
                for (size_t i = 0; areDynamicAxesCompatible && (i < numAxes); ++i)
                    areDynamicAxesCompatible = (left.DynamicAxes()[i].IsOrdered() == right.DynamicAxes()[i].IsOrdered());

                return ((left.Kind() == right.Kind()) &&
                    ((left.Shape() == right.Shape()) || (AsTensorShape(left.Shape()) == AsTensorShape(right.Shape()))) &&
                    (left.GetDataType() == right.GetDataType()) &&
                    areDynamicAxesCompatible &&
                    (left.NeedsGradient() == right.NeedsGradient()) &&
                    (left.Uid() == right.Uid()) &&
                    (left.IsSparse() == right.IsSparse()));
            };

            auto correspondingLoadedModelVar = loadedModelLeafVariablesMap.at(trainerModelLeafVar.Uid());

            if (!areVariablesEquivalent(correspondingLoadedModelVar, trainerModelLeafVar))
                InvalidArgument("The loaded model's leaf variables do not match the trainer model's leaf variables");

            if (trainerModelLeafVar.IsConstant() || trainerModelLeafVar.IsParameter())
            {
                auto trainerModelVarValue = trainerModelLeafVar.IsConstant() ? Constant(trainerModelLeafVar).Value() : Parameter(trainerModelLeafVar).Value();
                auto loadedModelVarValue = correspondingLoadedModelVar.IsConstant() ? Constant(correspondingLoadedModelVar).Value() : Parameter(correspondingLoadedModelVar).Value();
                trainerModelVarValue->CopyFrom(*loadedModelVarValue);
            }
        }
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

    // Names for the reduction operations as used by the CNTK ReduceElementsNode
    /*static*/ const std::wstring PrimitiveFunction::InternalSumReductionOpName = L"Sum";
    /*static*/ const std::wstring PrimitiveFunction::InternalLogSumReductionOpName = L"LogSum";
    /*static*/ const std::wstring PrimitiveFunction::InternalMeanReductionOpName = L"Mean";
    /*static*/ const std::wstring PrimitiveFunction::InternalMaxReductionOpName = L"Max";
    /*static*/ const std::wstring PrimitiveFunction::InternalMinReductionOpName = L"Min";
    /*static*/ const std::wstring PrimitiveFunction::InternalAllReductionOpName = L"All";
    /*static*/ const std::wstring PrimitiveFunction::InternalAnyReductionOpName = L"Any";

    // Names of the various attributes of CNTK primitive Functions
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis = L"axis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis1 = L"axis1";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis2 = L"axis2";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDropoutRate = L"dropoutRate";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewShape = L"newShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOutputRank = L"outputRank";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameInferInputRankToMap = L"inferInputRankToMap";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOffset = L"offset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameStrides = L"strides";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSharing = L"sharing";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAutoPadding = L"autoPadding";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameLowerPad = L"lowerPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUpperPad = L"upperPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTranspose = L"transpose";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples = L"maxTempMemSizeInSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingType = L"poolingType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingWindowShape = L"poolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSpatial = L"spatial";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNormalizationTimeConstant = L"normalizationTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlendTimeConstant = L"blendTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEpsilon = L"epsilon";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUseCuDNNEngine = L"useCuDNNEngine";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewDynamicAxes = L"newDynamicAxes";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginIndex = L"beginIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndIndex = L"endIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameReductionOpName = L"reductionOpName";

    /*static*/ std::vector<Variable> PrimitiveFunction::GetOutputVariables(PrimitiveOpType op,
                                                                           std::vector<Variable>& inputs,
                                                                           Function* owner,
                                                                           Dictionary& functionConfig,
                                                                           bool inferDimensions,
                                                                           const std::wstring& functionName)
    {
        if (op == PrimitiveOpType::Combine)
            return inputs;

        // We use the first non-constant input operand's DataType as the output DataType
        // In case there are no non-constant known DataTypes, we just pick the first known operand DataType
        // Also, all the known DataTypes of operands should match except for constants where coercion is allowed
        DataType firstKnownInputDataType = DataType::Unknown;
        DataType outputDataType = DataType::Unknown;
        size_t i = 0;
        while (i < inputs.size())
        {
            auto input = inputs[i++];
            auto inputDataType = input.GetDataType();
            if (inputDataType != DataType::Unknown)
            {
                if (firstKnownInputDataType == DataType::Unknown)
                    firstKnownInputDataType = inputDataType;

                if (outputDataType == DataType::Unknown)
                {
                    if (!input.IsConstant())
                        outputDataType = inputDataType;
                }
                else
                {
                    // The DataType of all operands should match except for Constants where we allow coercion
                    if ((inputDataType != DataType::Unknown) && (inputDataType != outputDataType) && !input.IsConstant())
                        InvalidArgument("Primitive function with op type %S has operands with different DataTypes %s and %s", PrimitiveOpTypeName(op).c_str(), DataTypeName(outputDataType), DataTypeName(inputDataType));
                }
            }
        }

        if (outputDataType == DataType::Unknown)
            outputDataType = firstKnownInputDataType;

        // We currently require that the inputs' dynamic axes if any match
        std::vector<Axis> outputDynamicAxes;
        if ((op == PrimitiveOpType::SumAll) || (op == PrimitiveOpType::SquaredError) || (op == PrimitiveOpType::CrossEntropyWithSoftmax) || (op == PrimitiveOpType::ClassificationError))
            outputDynamicAxes = std::vector<Axis>({});
        else if (op == PrimitiveOpType::Where)
            outputDynamicAxes = AsVector<Axis>(functionConfig[PrimitiveFunction::AttributeNameNewDynamicAxes].Value<std::vector<DictionaryValue>>());
        else if (op == PrimitiveOpType::ScatterPacked)
            outputDynamicAxes = inputs[2].DynamicAxes();
        else if ((op == PrimitiveOpType::PackedIndex) || (op == PrimitiveOpType::GatherPacked))
            outputDynamicAxes = inputs[1].DynamicAxes();
        else
        {
            auto allInputDynamicAxesEmpty = std::find_if(inputs.begin(), inputs.end(), [](const Variable& input) { return !input.DynamicAxes().empty(); }) == inputs.end();
            if (!allInputDynamicAxesEmpty)
            {
                outputDynamicAxes = Axis::UnknownDynamicAxes;
                for (auto inputVar : inputs)
                {
                    auto currentInputDynamicAxes = inputVar.DynamicAxes();
                    if (!currentInputDynamicAxes.empty() && (currentInputDynamicAxes != Axis::UnknownDynamicAxes))
                    {
                        if (outputDynamicAxes == Axis::UnknownDynamicAxes)
                            outputDynamicAxes = currentInputDynamicAxes;
                        else
                        {
                            if (currentInputDynamicAxes != outputDynamicAxes)
                                LogicError("Currently if an operand of a elementwise operation has any dynamic axes, those must match the dynamic axes of the other operands");
                        }
                    }
                }
            }
        }

        NDShape outputShape;
        bool areAnyInputShapesUnknown = (std::find_if(inputs.begin(), inputs.end(), [](const Variable& input) { return input.Shape().IsUnknown(); }) != inputs.end());
        if (areAnyInputShapesUnknown)
            outputShape = NDShape::Unknown;
        else
        {
            switch (op)
            {
            case PrimitiveOpType::Negate:
            case PrimitiveOpType::Sigmoid:
            case PrimitiveOpType::Tanh:
            case PrimitiveOpType::ReLU:
            case PrimitiveOpType::Exp:
            case PrimitiveOpType::Log:
            case PrimitiveOpType::Sqrt:
            case PrimitiveOpType::Floor:
            case PrimitiveOpType::Abs:
            case PrimitiveOpType::Reciprocal:
            case PrimitiveOpType::Softmax:
            case PrimitiveOpType::Hardmax:
            case PrimitiveOpType::Dropout:
            case PrimitiveOpType::Where:
            {
                assert(inputs.size() == 1);
                outputShape = UnaryElementwiseOpOutputShape(inputs[0].Shape());
                break;
            }
            case PrimitiveOpType::TransposeAxes:
            {
                assert(inputs.size() == 1);

                auto axis1 = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis1].Value<Axis>(), inputs[0].Shape());
                auto axis2 = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis2].Value<Axis>(), inputs[0].Shape());

                if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                    LogicError("TransposeAxes operation currently does not support transposing dynamic axes");

                VerifyStaticAxis(axis1, inputs[0].Shape());
                VerifyStaticAxis(axis2, inputs[0].Shape());

                outputShape = inputs[0].Shape();
                std::swap(outputShape[axis1.StaticAxisIndex()], outputShape[axis2.StaticAxisIndex()]);
                break;
            }
            case PrimitiveOpType::Slice:
            {
                assert(inputs.size() == 1);
                auto axis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), inputs[0].Shape());

                auto beginIndex = functionConfig[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                auto endIndex = functionConfig[PrimitiveFunction::AttributeNameEndIndex].Value<int>();
                if (!axis.IsStaticAxis())
                    LogicError("Built-in Slice operation currently does not support slicing along dynamic axis");

                VerifyStaticAxis(axis, inputs[0].Shape());

                size_t sliceAxisDim = inputs[0].Shape()[axis.StaticAxisIndex()];
                int realBeginIndex = (beginIndex >= 0) ? beginIndex : beginIndex + sliceAxisDim;
                int realEndIndex = (endIndex > 0) ? endIndex : endIndex + sliceAxisDim;
                if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                    RuntimeError("Slice operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input's shape ([%S]).",
                    beginIndex,
                    endIndex,
                    realBeginIndex,
                    realEndIndex,
                    AsStringForErrorReporting(inputs[0].Shape()).c_str());

                auto outputTensorShape = AsTensorShape(inputs[0].Shape());

                // propagate as much as we can
                if ((axis.StaticAxisIndex() < (int)outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim))
                    outputTensorShape.NarrowTo(axis.StaticAxisIndex(), realBeginIndex, realEndIndex);

                outputShape = AsNDShape(outputTensorShape, /*allowNonFlattenableTensorShapes = */ true);
                break;
            }
            case PrimitiveOpType::Reshape:
            {
                auto newShape = functionConfig[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();
                outputShape = ReshapeOutputShape(inputs[0].Shape(), newShape);
                break;
            }
            case PrimitiveOpType::Pooling:
            {
                assert(inputs.size() == 1);
                auto poolingWindowsShape = functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
                auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                NDShape outputMapCount = { 1 };
                std::vector<bool> sharing = { true };
                outputShape = ConvolutionOpOutputShape(op, inputs[0].Shape(), poolingWindowsShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, inferDimensions);
                break;
            }
            case PrimitiveOpType::SumAll:
                assert(inputs.size() == 1);
                outputShape = {};
                break;
            case PrimitiveOpType::Plus:
            case PrimitiveOpType::Minus:
            case PrimitiveOpType::ElementTimes:
            case PrimitiveOpType::Equal:
            case PrimitiveOpType::NotEqual:
            case PrimitiveOpType::Less:
            case PrimitiveOpType::LessEqual:
            case PrimitiveOpType::Greater:
            case PrimitiveOpType::GreaterEqual:
                assert(inputs.size() == 2);
                outputShape = BinaryElementwiseOpOutputShape(op, inputs[0], inputs[1], true, inferDimensions);
                break;
            case PrimitiveOpType::Times:
            {
                assert(inputs.size() == 2);
                auto outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                auto inferInputRankToMap = (int)functionConfig[PrimitiveFunction::AttributeNameInferInputRankToMap].Value<size_t>();
                outputShape = TimesOpOutputShape(inputs[0], inputs[1], outputRank, inferInputRankToMap, inferDimensions);
                break;
            }
            case PrimitiveOpType::TransposeTimes:
            {
                assert(inputs.size() == 2);

                auto transposeShapeFunc = [](const NDShape& shape) {
                    NDShape transposedShape(std::max<size_t>(2, shape.Rank()), 1);
                    for (size_t i = 0; i < shape.Rank(); ++i)
                        transposedShape[transposedShape.Rank() - i - 1] = shape[i];

                    return transposedShape;
                };

                if (inputs[0].Shape().Rank() > 2)
                    LogicError("TransposeTimes operation currently only supports %s operands of rank 1 or 2", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left");

                NDShape transposedLeftOperandShape = transposeShapeFunc(inputs[0].Shape());
                Variable dummyLeftOperand = PlaceholderVariable(transposedLeftOperandShape);
                size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                outputShape = TimesOpOutputShape(dummyLeftOperand, inputs[1], outputRank, -1, inferDimensions);
                if (dummyLeftOperand.Shape() != transposedLeftOperandShape)
                    inputs[0].m_dataFields->m_shape = transposeShapeFunc(dummyLeftOperand.Shape());

                break;
            }
            case PrimitiveOpType::Convolution:
            {
                assert(inputs.size() == 2);
                auto& strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                auto& lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                auto& upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                auto sharing = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
                auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                bool transpose = functionConfig[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
                if (inputs[0].Shape().Rank() < inputs[1].Shape().Rank())
                    InvalidArgument("The convolution map should have at least as many axes as the shape of the input it operates on!");

                NDShape outputMapCount, kernelShape;
                std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(inputs[0].Shape(), inputs[1].Shape());
                auto originalKernelShape = kernelShape;
                outputShape = ConvolutionOpOutputShape(op, inputs[1].Shape(), kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose, inferDimensions);
                if (originalKernelShape != kernelShape)
                {
                    for (size_t i = 0; i < kernelShape.Rank(); ++i)
                        inputs[0].m_dataFields->m_shape[i] = kernelShape[i];
                }

                functionConfig[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
                functionConfig[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
                break;
            }
            case PrimitiveOpType::SquaredError:
            case PrimitiveOpType::CrossEntropyWithSoftmax:
            case PrimitiveOpType::ClassificationError:
            {
                if (op == PrimitiveOpType::ClassificationError)
                    assert(inputs.size() >= 2);
                else
                    assert(inputs.size() == 2);

                if ((inputs[0].Shape().Rank() > 2) || ((inputs[0].Shape().Rank() > 1) && (inputs[0].Shape()[1] != 1)))
                    InvalidArgument("The shape of input operands for the %S operation should have at most one axis", PrimitiveOpTypeName(op).c_str());

                auto predictionShape = inputs[0].Shape();
                auto labelsShape = inputs[1].Shape();
                if (predictionShape != labelsShape)
                    RuntimeError("Prediction output operand's shape %S is incompatible with label operand's shape %S for the %S operation", AsStringForErrorReporting(predictionShape).c_str(), AsStringForErrorReporting(labelsShape).c_str(), PrimitiveOpTypeName(op).c_str());

                std::vector<int> reductionAxes;
                for (int i = 0; i < (int)inputs[0].Shape().Rank(); ++i)
                    reductionAxes.push_back(i);

                outputShape = ReductionOpOutputShape(op, predictionShape, reductionAxes, /*preserveReductionAxes =*/ false);
                break;
            }
            case PrimitiveOpType::PastValue:
            case PrimitiveOpType::FutureValue:
            {
                assert(inputs.size() == 2);
                Variable inputOperandVar = inputs[0];
                Variable initialStateVar = inputs[1];

                // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
                if ((inputOperandVar.DynamicAxes() != Axis::UnknownDynamicAxes) && (inputOperandVar.DynamicAxes().size() != 2))
                    LogicError("Currently PastValue/FutureValue Function only supports input operand with 2 dynamic axis (1 sequence-axis and 1 batch-axis)");

                if (!initialStateVar.DynamicAxes().empty())
                    LogicError("Currently PastValue/FutureValue Function does not support initial state operand with dynamic axes!");

                outputShape = BinaryElementwiseOpOutputShape(op, inputs[0], inputs[1], true, inferDimensions);
                break;
            }
            case PrimitiveOpType::ReduceElements:
            {
                assert(inputs.size() == 1);
                auto reductionAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), inputs[0].Shape());
                if (reductionAxis == Axis::AllStaticAxes())
                    outputShape = {};
                else
                {
                    std::vector<int> reductionAxes = { reductionAxis.StaticAxisIndex() };
                    outputShape = ReductionOpOutputShape(op, inputs[0].Shape(), reductionAxes, /*preserveReductionAxes =*/ true);
                }
                break;
            }
            case PrimitiveOpType::BatchNormalization:
            {
                assert(inputs.size() == 5);
                auto spatial = functionConfig[PrimitiveFunction::AttributeNameSpatial].Value<bool>();
                outputShape = BatchNormalizationOutputShape(inputs, spatial, inferDimensions);
                break;
            }
            case PrimitiveOpType::PackedIndex:
                outputShape = UnaryElementwiseOpOutputShape(inputs[1].Shape());
                break;
            case PrimitiveOpType::GatherPacked:
            {
                bool sourceHasDynamicAxis = !inputs[0].DynamicAxes().empty();

                // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
                if (sourceHasDynamicAxis)
                    outputShape = inputs[0].Shape();
                else
                {
                    if (inputs[0].Shape().Rank() > 1)
                        outputShape = outputShape.SubShape(0, outputShape.Rank() - 1);
                    else
                        outputShape = {};
                }

                break;
            }
            case PrimitiveOpType::ScatterPacked:
            {
                if (inputs[0].DynamicAxes().empty() || inputs[1].DynamicAxes().empty() || inputs[2].DynamicAxes().empty())
                    InvalidArgument("ScatterPacked requires all its operands to have dynamic axes");

                outputShape = inputs[0].Shape();
                break;
            }
            case PrimitiveOpType::Clip:
                assert(inputs.size() == 3);
                outputShape = UnaryElementwiseOpOutputShape(inputs[0].Shape());
                break;
            case PrimitiveOpType::Select:
                assert(inputs.size() == 3);
                outputShape = NaryElementwiseOpOutputShape(op, inputs, true);
                break;
            case PrimitiveOpType::Splice:
            {
                assert(inputs.size() >= 2);
                auto maxInputRank = MaxInputRank(inputs);
                auto spliceAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), NDShape(maxInputRank));

                if (!spliceAxis.IsStaticAxis())
                    LogicError("Splice operation currently does not support splicing along dynamic axis");

                if (spliceAxis.StaticAxisIndex() < 0)
                    InvalidArgument("Splice: The axis argument's static axis index must be >= 0!");

                outputShape = SpliceOutputShape(inputs, spliceAxis.StaticAxisIndex());
                break;
            }
            default:
                LogicError("Specified op %S not yet supported", PrimitiveOpTypeName(op).c_str());
                break;
            }
        }

        return { OutputVariable(outputShape, outputDataType, owner, outputDynamicAxes, functionName.empty() ? L"" : functionName + L"_output") };
    }

    /*static*/ const std::wstring CompositeFunction::CompositeFunctionOpName = L"CompositeFunctionOpName";
    /*static*/ std::atomic<unsigned int> CompositeFunction::s_nextAutoGeneratedDynamicAxis(0);

    // Names of the dynamic axes in the CNTK engine for some special sets of dynamic axes values
    // Note: The no sequence axis corresponds to a special case where there is no sequence axis (i.e. has been reduced over)
    // and the special name is used to identify this when loading back a model saved in CNTK v1 format. This will not really be needed
    // when the new CNTK v2 model serialization format is ready.
    /*static*/ const std::wstring CompositeFunction::InternalDefaultDynamicAxisName = L"*";
    /*static*/ const std::wstring CompositeFunction::InternalNoSequenceAxisName = L"__noSequenceAxis";

    // Replace any PlaceHolder Variables in the graph of Functions underlying 'this' CompositeFunction. All PlaceHolder variables
    // should have been replaced before performing any Forward compute of 'this' Function.
    /*virtual*/ void CompositeFunction::ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                                   std::unordered_set<const Function*>& visitedFunctions,
                                                                   std::unordered_set<Variable>& replacedPlaceholders)
    {
        RootFunction()->ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);

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
        std::unordered_map<const Function*, size_t> functionVisitCounts;
        RootFunction()->ValidateOrUpdateOutputs(functionVisitCounts);
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
        {
            isVariableRootMap[variable] = false;
            return iter->second;
        }

        // The DataType, Shape and DynamicAxes of the variable must be known by now
        if (variable.GetDataType() == DataType::Unknown)
            InvalidArgument("Variable%S with unknown DataType detected when compiling the Function graph!", ParanthesizedName(variable.Name()).c_str());

        if (variable.Shape().IsUnknown())
            InvalidArgument("Variable%S with unknown shape detected when compiling the Function graph!", ParanthesizedName(variable.Name()).c_str());

        if (variable.Shape().HasInferredDimension())
            InvalidArgument("Variable%S with InferredDimension for at least one axis in it's shape, detected when compiling the Function graph!", ParanthesizedName(variable.Name()).c_str());

        if (variable.DynamicAxes() == Axis::UnknownDynamicAxes)
            InvalidArgument("Variable%S with unknown dynamic axes detected when compiling the Function graph!", ParanthesizedName(variable.Name()).c_str());

        // Lets add a null entry in the map for this variable, to break infinite recursion when processing recurrent graphs
        variableToNodeMap[variable] = nullptr;

        std::shared_ptr<ComputationNode<ElementType>> computationNodePtr;
        auto internalNodeName = CNTKInternalNodeNameFromUidAndName(variable.Uid(), variable.Name());
        if (variable.IsParameter() || variable.IsConstant())
        {
            computationNodePtr = builder.CreateLearnableParameter(internalNodeName, AsTensorShape(variable.Shape()));
            network->InitLearnableParameters(computationNodePtr, L"fixedValue", 0); // must call this to follow protocol; can overwrite later
            if (!variable.NeedsGradient())
                computationNodePtr->SetLearningRateMultiplier(0.0);

            NDArrayViewPtr value = variable.IsConstant() ? Constant(variable).Value() : Parameter(variable).Value();
            std::shared_ptr<const Matrix<ElementType>> valueMatrix = variable.IsConstant() ? value->GetMatrix<ElementType>() : value->GetWritableMatrix<ElementType>();
            if (variable.IsParameter() || (valueMatrix->GetDeviceId() == network->GetDeviceId()))
                computationNodePtr->Value() = valueMatrix->AsReference();
            else
            {
                Matrix<ElementType> clonedMatrix(valueMatrix->GetNumRows(), valueMatrix->GetNumCols(), network->GetDeviceId(), valueMatrix->GetMatrixType(), valueMatrix->GetFormat());
                clonedMatrix.AssignValuesOf(*valueMatrix);
                computationNodePtr->Value() = std::move(clonedMatrix);
            }
        }
        else if (variable.IsInput())
        {
            // TODO: Input variables currently are required to have the default batch axis
            auto dynamicAxes = variable.DynamicAxes();
            auto foundDefaultBatchAxis = std::find(dynamicAxes.begin(), dynamicAxes.end(), Axis::DefaultBatchAxis());
            if (foundDefaultBatchAxis == dynamicAxes.end())
                LogicError("Currently Input Variables are required to have the DefaultBatchAxis as one of their dynamic axes");

            if (dynamicAxes.back() != Axis::DefaultBatchAxis())
                LogicError("Currently Input Variables are required to have the DefaultBatchAxis as their last dynamic axes");

            // TODO: Support inputs with > 1 dynamic axes
            if ((dynamicAxes.size() < 1) || (dynamicAxes.size() > 2))
                LogicError("Currently only Input variables with 1 or 2 dynamic axis are supported");

            // Construct the dynamic axis name to be used internally for the CNTK InputNodes
            std::wstring internalDynamicAxisName = InternalDynamicAxisNameFromDynamicAxes(dynamicAxes);

            if (!internalDynamicAxisName.empty() && !network->NodeNameExists(internalDynamicAxisName))
                network->AddNodeToNetAndAttachInputs(New<DynamicAxisNode<ElementType>>(network->GetDeviceId(), internalDynamicAxisName), {});

            if (IsSparseInput(variable))
                computationNodePtr = builder.CreateSparseInputNode(internalNodeName, AsTensorShape(variable.Shape()), internalDynamicAxisName);
            else
                computationNodePtr = builder.CreateInputNode(internalNodeName, AsTensorShape(variable.Shape()), internalDynamicAxisName);

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
        if (isVariableRootMap.find(variable) == isVariableRootMap.end())
            isVariableRootMap[variable] = variable.IsOutput();
        return computationNodePtr;
    }

    template <typename ElementType>
    /*static*/ ComputationNodeBasePtr CompositeFunction::CreateComputationNode(const Variable& variable,
                                                                               PrimitiveFunction* primitiveFunction,
                                                                               const std::vector<std::shared_ptr<ComputationNode<ElementType>>>& inputNodes,
                                                                               Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                               std::unordered_map<Variable, ComputationNodeBasePtr>& variableToNodeMap)
    {
        ComputationNodeBasePtr computationNodePtr;

        auto functionName = primitiveFunction->Name();
        auto& functionConfig = primitiveFunction->Attributes();
        auto functionInputs = primitiveFunction->Inputs();
        PrimitiveOpType op = primitiveFunction->OpType();

        switch (op)
        {
        case PrimitiveOpType::Negate:
            computationNodePtr = New<NegateNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Sigmoid:
            computationNodePtr = New<SigmoidNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Tanh:
            computationNodePtr = New<TanhNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::ReLU:
            computationNodePtr = New<RectifiedLinearNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Exp:
            computationNodePtr = New<ExpNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Log:
            computationNodePtr = New<LogNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Sqrt:
            computationNodePtr = New<SqrtNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Floor:
            computationNodePtr = New<FloorNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Abs:
            computationNodePtr = New<AbsNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Reciprocal:
            computationNodePtr = New<ReciprocalNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Softmax:
            computationNodePtr = New<SoftmaxNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Hardmax:
            computationNodePtr = New<HardmaxNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::TransposeAxes:
        {
            auto axis1 = functionConfig[PrimitiveFunction::AttributeNameAxis1].Value<Axis>();
            auto axis2 = functionConfig[PrimitiveFunction::AttributeNameAxis2].Value<Axis>();

            // The axis ids passed to the internal CNTK TransposeDimensionsNode are 1 based instead of 0 based
            computationNodePtr = New<TransposeDimensionsNode<ElementType>>(network->GetDeviceId(), functionName, AsCNTKInternalAxisIdx(axis1), AsCNTKInternalAxisIdx(axis2));
            break;
        }
        case PrimitiveOpType::Where:
        {
            auto dynamicAxes = variable.DynamicAxes();
            auto internalCNTKWhereNodeDynamicAxisName = InternalDynamicAxisNameFromDynamicAxes(dynamicAxes);
            computationNodePtr = New<WhereNode<ElementType>>(network->GetDeviceId(), functionName, internalCNTKWhereNodeDynamicAxisName);
            break;
        }
        case PrimitiveOpType::Slice:
        {
            auto axis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            auto beginIndex = functionConfig[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
            auto endIndex = functionConfig[PrimitiveFunction::AttributeNameEndIndex].Value<int>();

            // Internal CNTK SliceNode takes 1 based axis indices instead of 0 based
            computationNodePtr = New<SliceNode<ElementType>>(network->GetDeviceId(), functionName, beginIndex, endIndex, AsCNTKInternalAxisIdx(axis));
            break;
        }
        case PrimitiveOpType::Dropout:
        {
            auto dropoutRate = functionConfig[PrimitiveFunction::AttributeNameDropoutRate].Value<double>();
            computationNodePtr = New<DropoutNode<ElementType>>(network->GetDeviceId(), functionName);
            computationNodePtr->As<DropoutNode<ElementType>>()->SetDropoutRate(dropoutRate);
            break;
        }
        case PrimitiveOpType::Reshape:
        {
            auto newShape = functionConfig[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();
            computationNodePtr = New<ReshapeNode<ElementType>>(network->GetDeviceId(), functionName, AsTensorShape(newShape));
            break;
        }
        case PrimitiveOpType::Pooling:
        {
            PoolingType poolingType = (PoolingType)(functionConfig[PrimitiveFunction::AttributeNamePoolingType].Value<size_t>());
            auto poolingWindowsShape = functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            computationNodePtr = New<PoolingNode<ElementType>>(network->GetDeviceId(), functionName, AsCNTKPoolKind(poolingType), AsTensorShape(poolingWindowsShape), AsTensorShape(strides), autoPadding, AsTensorShape(lowerPad), AsTensorShape(upperPad), ImageLayoutKind::CHW);
            break;
        }
        case PrimitiveOpType::SumAll:
            computationNodePtr = New<SumElementsNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Plus:
            computationNodePtr = New<PlusNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Minus:
            computationNodePtr = New<MinusNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::ElementTimes:
            computationNodePtr = New<ElementTimesNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Equal:
            computationNodePtr = New<EqualNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::NotEqual:
            computationNodePtr = New<NotEqualNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Less:
            computationNodePtr = New<LessNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::LessEqual:
            computationNodePtr = New<LessEqualNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Greater:
            computationNodePtr = New<GreaterNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::GreaterEqual:
            computationNodePtr = New<GreaterEqualNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Times:
        {
            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            auto inferInputRankToMap = (int)functionConfig[PrimitiveFunction::AttributeNameInferInputRankToMap].Value<size_t>();
            computationNodePtr = New<TimesNode<ElementType>>(network->GetDeviceId(), functionName, outputRank, inferInputRankToMap);
            break;
        }
        case PrimitiveOpType::TransposeTimes:
        {
            size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
            computationNodePtr = New<TransposeTimesNode<ElementType>>(network->GetDeviceId(), functionName, outputRank);
            break;
        }
        case PrimitiveOpType::Convolution:
        {
            NDShape outputMapCount, kernelShape;
            std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(functionInputs[0].Shape(), functionInputs[1].Shape());
            auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
            auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
            auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
            auto sharing = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
            auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
            auto transpose = functionConfig[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
            auto maxTempMemSizeInSamples = functionConfig[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples].Value<size_t>();
            computationNodePtr = New<ConvolutionNode<ElementType>>(network->GetDeviceId(), functionName, AsTensorShape(kernelShape), AsTensorShape(outputMapCount), AsTensorShape(strides), sharing, autoPadding, AsTensorShape(lowerPad), AsTensorShape(upperPad), transpose, ImageLayoutKind::CHW, maxTempMemSizeInSamples);
            break;
        }
        case PrimitiveOpType::SquaredError:
            computationNodePtr = New<SquareErrorNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::CrossEntropyWithSoftmax:
            computationNodePtr = New<CrossEntropyWithSoftmaxNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::ClassificationError:
            computationNodePtr = New<ClassificationErrorNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::PastValue:
        case PrimitiveOpType::FutureValue:
        {
            Variable inputOperandVar = functionInputs[0];
            Variable initialStateVar = functionInputs[1];

            size_t offset = primitiveFunction->Attributes()[PrimitiveFunction::AttributeNameOffset].Value<size_t>();
            if (op == PrimitiveOpType::PastValue)
                computationNodePtr = New<PastValueNode<ElementType>>(network->GetDeviceId(), functionName, AsTensorShape(inputOperandVar.Shape()), offset);
            else
                computationNodePtr = New<FutureValueNode<ElementType>>(network->GetDeviceId(), functionName, AsTensorShape(inputOperandVar.Shape()), offset);

            break;
        }
        case PrimitiveOpType::ReduceElements:
        {
            auto reductionAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            auto reductionOpName = functionConfig[PrimitiveFunction::AttributeNameReductionOpName].Value<std::wstring>();
            computationNodePtr = New<ReduceElementsNode<ElementType>>(network->GetDeviceId(), functionName, reductionOpName, AsCNTKInternalAxisIdx(reductionAxis));
            break;
        }
        case PrimitiveOpType::BatchNormalization:
        {
            auto spatial = functionConfig[PrimitiveFunction::AttributeNameSpatial].Value<bool>();
            auto normalizationTimeConstant = functionConfig[PrimitiveFunction::AttributeNameNormalizationTimeConstant].Value<double>();
            auto blendTimeConstant = functionConfig[PrimitiveFunction::AttributeNameBlendTimeConstant].Value<double>();
            auto epsilon = functionConfig[PrimitiveFunction::AttributeNameEpsilon].Value<double>();
            auto useCuDNNEngine = functionConfig[PrimitiveFunction::AttributeNameUseCuDNNEngine].Value<bool>();
            computationNodePtr = New<BatchNormalizationNode<ElementType>>(network->GetDeviceId(), functionName, spatial, normalizationTimeConstant, blendTimeConstant, epsilon, !useCuDNNEngine, ImageLayoutKind::CHW);
            break;
        }
        case PrimitiveOpType::Combine:
            // This operation is just a no-op and is a means to combine multiple functions to create a single Function
            // whose outputs are a union of the outputs of the Functions being combined.
            computationNodePtr = variableToNodeMap[variable];
            break;
        case PrimitiveOpType::PackedIndex:
            computationNodePtr = New<PackedIndexNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::GatherPacked:
            computationNodePtr = New<GatherPackedNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::ScatterPacked:
            computationNodePtr = New<ScatterPackedNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Clip:
            computationNodePtr = New<ClipNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Select:
            computationNodePtr = New<IfNode<ElementType>>(network->GetDeviceId(), functionName);
            break;
        case PrimitiveOpType::Splice:
        {
            Axis spliceAxis = functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
            computationNodePtr = New<RowStackNode<ElementType>>(network->GetDeviceId(), functionName, AsCNTKInternalAxisIdx(spliceAxis));
            break;
        }
        default:
            LogicError("Specified op %S not yet supported", PrimitiveOpTypeName(op).c_str());
            break;
        }

        std::vector<ComputationNodeBasePtr> inputNodesBasePtrs;
        for (auto inputNode : inputNodes)
            inputNodesBasePtrs.push_back(inputNode);

        // Let's reorder inputNodesBasePtrs properly since the ordering of inputs of CNTK internal ComputationNode may be different from the PrimitiveFunction inputs ordering
        ReorderAsCNTKComputationNodeInputs(op, inputNodesBasePtrs);
        if (computationNodePtr->Is<INumInputs>())
        {
            auto computationNodeExpectedInputCount = computationNodePtr->As<INumInputs>()->GetExpectedNumInputs();
            if (computationNodeExpectedInputCount != inputNodesBasePtrs.size())
                LogicError("Input count mismatch: The Primitive function for op %S has %d inputs while the corresponding ComputationNode has %d inputs",
                           PrimitiveOpTypeName(op).c_str(),
                           (int)inputNodesBasePtrs.size(),
                           (int)computationNodeExpectedInputCount);
        }

        network->AddNodeToNetAndAttachInputs(computationNodePtr, inputNodesBasePtrs);

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
            PrimitiveOpType op = primitiveFunction->OpType();
            auto& functionInputs = primitiveFunction->m_inputs;

            DataType nonConstInputDataType = DataType::Unknown;
            for (auto& inputVar : functionInputs)
            {
                if (!inputVar.IsConstant() && (inputVar.GetDataType() != DataType::Unknown))
                {
                    nonConstInputDataType = inputVar.GetDataType();
                    break;
                }
            }
            
            // Create the nodes corresponding to the inputs
            std::vector<std::shared_ptr<ComputationNode<ElementType>>> inputNodes;
            for (auto& inputVar : functionInputs)
            {
                // If the inputVar is a constant and not the right DataType let's coerce it to the right type
                if (inputVar.IsConstant() && (nonConstInputDataType != DataType::Unknown) && (inputVar.GetDataType() != nonConstInputDataType))
                {
                    auto constantValueCPU = Constant(inputVar).Value()->DeepClone(DeviceDescriptor::CPUDevice(), true);
                    NDArrayViewPtr newConstantValue = CloneAsDataType(constantValueCPU, nonConstInputDataType, true);
                    inputVar = Constant(newConstantValue);
                }

                auto baseNodePtr = GetNode(inputVar, network, builder, variableToNodeMap, isVariableRootMap);
                inputNodes.push_back((baseNodePtr != nullptr) ? baseNodePtr->template As<ComputationNode<ElementType>>()->shared_from_this() : nullptr);
            }

            computationNodePtr = CreateComputationNode(variable, primitiveFunction, inputNodes, network, variableToNodeMap);
            if (op != PrimitiveOpType::Combine)
            {
                for (auto inputVar : functionInputs)
                    isVariableRootMap[inputVar] = false;
            }
        }
        else
            LogicError("User defined Functions are currently unsupported!");

        return computationNodePtr;
    }

    template <typename ElementType>
    ComputationNetworkPtr CompositeFunction::GetComputationNetwork(const DeviceDescriptor& device, const std::unordered_set<Variable>& backpropRoots, bool allocateNetworkMatrices)
    {
        if (m_computationNetwork != nullptr)
        {
            // TODO: We should either invalidate and readapt the network if he backpropRoots change compared to what was specified when the network
            // was last constructed, to just recreate a new network.
            // For now just disallow changing the backpropRoots after the network is created
            if (!backpropRoots.empty() && (m_currentBackpropRoots != backpropRoots))
                LogicError("Changing backprop roots across different Forward calls on a CNTK composite Function is currently unsupported");

            // TODO: Support changing the device across different invocations of the forward method on a Function instance
            if (AsDeviceDescriptor(m_computationNetwork->GetDeviceId()) != device)
                LogicError("Changing device across different Forward calls on a CNTK composite Function is currently unsupported");

        }
        else
        {
            m_computationNetwork = std::make_shared<ComputationNetwork>(AsCNTKImplDeviceId(device));

            ComputationNetworkBuilder<ElementType> builder(*m_computationNetwork);

            // TODO: We current only support one backprop root
            if (backpropRoots.size() > 1)
                LogicError("More than one backprop roots is currently unsupported");

            // Now recursively create the network in a top-down fashion
            auto rootFunction = RootFunction();
            auto rootFunctionOutputs = rootFunction->Outputs();
            for (auto rootOutput : rootFunctionOutputs)
                GetNode(rootOutput, m_computationNetwork, builder, m_variableToNodeMap, m_isVariableRootMap);

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
                auto& currentComputationNode = varNodePair.second;
                auto& currentComputationNodeInputs = currentComputationNode->GetInputs();
                auto& currentVar = varNodePair.first;

                if (std::find(currentComputationNodeInputs.begin(), currentComputationNodeInputs.end(), nullptr) != currentComputationNodeInputs.end())
                {
                    // This ComputationNode has at least one null input which now needs to be properly attached

                    const PrimitiveFunction* primitiveFunc = dynamic_cast<const PrimitiveFunction*>(currentVar.Owner().get());

                    // Let's reorder properly since the ordering of inputs of CNTK internal ComputationNode may be different from the PrimitiveFunction inputs ordering
                    auto inputVars = primitiveFunc->Inputs();
                    ReorderAsCNTKComputationNodeInputs(primitiveFunc->OpType(), inputVars);
                    inputVars.resize(currentComputationNode->GetNumInputs());

                    std::vector<ComputationNodeBasePtr> inputNodesBasePtrs;
                    for (auto inputVar : inputVars)
                        inputNodesBasePtrs.push_back(m_variableToNodeMap[inputVar]);

                    currentComputationNode->AttachInputs(inputNodesBasePtrs);
                }
            }

            m_computationNetwork->SetTraceLevel(Internal::GetComputationNetworkTraceLevel());
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
                    if (((outputShape.Rank() == 0) && (computationNodeSampleLayout[0] != 1)) ||
                        ((outputShape.Rank() != 0) && (computationNodeSampleLayout != AsTensorViewShape(outputShape)) && (computationNodeSampleLayout != AsTensorShape(outputShape))))
                    {
                        LogicError("The output Variable shape %S does not match the SampleLayout shape %s of the corresponding ComputationNode in the network", outputShape.AsString().c_str(), ((std::string)computationNodeSampleLayout).c_str());
                    }
                }
            }
        }


        if (!m_networkMatricesAllocated && allocateNetworkMatrices)
        {
            ComputationNodeBasePtr backpropRootNode;

            // Now recursively create the network in a top-down fashion
            auto rootFunction = RootFunction();
            auto rootFunctionOutputs = rootFunction->Outputs();
            std::vector<ComputationNodeBasePtr> forwardRootNodes;
            for (auto rootOutput : rootFunctionOutputs)
            {
                auto currentRootNode = m_variableToNodeMap[rootOutput];
                forwardRootNodes.push_back(currentRootNode);

                if (m_currentBackpropRoots.find(rootOutput) != m_currentBackpropRoots.end())
                    backpropRootNode = currentRootNode;
            }

            m_computationNetwork->AllocateAllMatrices(forwardRootNodes, {}, backpropRootNode);
            m_networkMatricesAllocated = allocateNetworkMatrices;
        }

        return m_computationNetwork;
    }

    template <typename ElementType>
    /*static*/ std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CompositeFunction::GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value)
    {
        if (var.GetDataType() != value->GetDataType())
            LogicError("The Variable's DataType %s does not match the corresponding Value's DataType %s", DataTypeName(var.GetDataType()), DataTypeName(value->GetDataType()));

        if (AsDataType<ElementType>() != value->GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(value->GetDataType()));

        // TODO: Is supplying dense data for an Input variable tagged as sparse, a fatal error?
        if (IsSparseInput(var) && !value->IsSparse())
            InvalidArgument("Dense input data supplied for a sparse input Variable");

        if (IsSparseInput(var) && (value->GetStorageFormat() != StorageFormat::SparseCSC))
            InvalidArgument("Sparse Input data must be in SparseCSC format");

        auto varShape = var.Shape();
        auto valueShape = value->Shape();
        if (valueShape.Rank() < varShape.Rank())
            InvalidArgument("Value's rank should be >= the Variable's rank");

        size_t maxAddionalValueAxes = std::max<size_t>(2, var.DynamicAxes().size());
        if (valueShape.Rank() > (varShape.Rank() + maxAddionalValueAxes))
            InvalidArgument("Value rank should be larger than the Variable%S rank at most by number of dynamic axes", ParanthesizedName(var.Name()).c_str());

        if (valueShape.SubShape(0, varShape.Rank()) != varShape)
        {
            InvalidArgument("The %s dimensions of the Value shape %S do not match the shape of the variable %S that it corresponds to!",
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading",
                            AsStringForErrorReporting(valueShape).c_str(),
                            AsStringForErrorReporting(varShape).c_str());
        }

        if (var.DynamicAxes().empty())
            return{ value->Data()->GetMatrix<ElementType>(), nullptr };

        if (var.DynamicAxes().size() > 2)
            LogicError("More than 2 dynamic axis for a variable is currently unsupported");

        auto mask = value->Mask();
        if ((mask != nullptr) && ((varShape.Rank() + mask->Shape().Rank()) != valueShape.Rank()))
            InvalidArgument("Invalid Value object; the sum of the rank of the mask and data does not equal the Variable's rank + number of dynamic axes");

        auto getNumTimeStepsAndSequencesFunc = [](const NDShape& maskShape) {
            size_t maxNumTimeSteps = 1;
            size_t numSequences = 1;
            if (maskShape.Rank() > 0)
                maxNumTimeSteps = maskShape[0];

            if (maskShape.Rank() > 1)
                numSequences = maskShape[1];

            return std::pair<size_t, size_t>(maxNumTimeSteps, numSequences);
        };

        size_t maxNumTimeSteps, numSequences;
        std::tie(maxNumTimeSteps, numSequences) = getNumTimeStepsAndSequencesFunc(valueShape.SubShape(varShape.Rank()));

        auto getSequenceStartsAndLengthsFunc = [&getNumTimeStepsAndSequencesFunc](const NDMaskPtr& mask, std::vector<ptrdiff_t>& sequenceBeginIndices, std::vector<size_t>& sequenceLengths) {
            auto cpuMask = mask;
            if (mask->Device() != DeviceDescriptor::CPUDevice())
                cpuMask = mask->DeepClone(DeviceDescriptor::CPUDevice());

            const MaskKind* maskBuffer = cpuMask->DataBuffer();
            size_t maxNumTimeSteps, numSequences;
            std::tie(maxNumTimeSteps, numSequences) = getNumTimeStepsAndSequencesFunc(mask->Shape());

            for (size_t i = 0; i < numSequences; ++i)
            {
                MaskKind firstMaskEntry = maskBuffer[i * maxNumTimeSteps];
                if (firstMaskEntry == MaskKind::SequenceBegin)
                    sequenceBeginIndices[i] = 0;
                else if (firstMaskEntry == MaskKind::Valid)
                    sequenceBeginIndices[i] = Microsoft::MSR::CNTK::SentinelValueIndicatingUnspecifedSequenceBeginIdx;
                else
                    LogicError("The first entry of a mask should be Valid or SequenceBegin");

                size_t currentSequenceLength = 1;
                bool currentSequenceEndAlreadyFound = false;
                for (size_t j = 1; j < maxNumTimeSteps; ++j)
                {
                    if (maskBuffer[(i * maxNumTimeSteps) + j] == MaskKind::Invalid)
                        currentSequenceEndAlreadyFound = true;
                    else
                    {
                        if (currentSequenceEndAlreadyFound)
                            InvalidArgument("Invalid Value object; only trailing steps of a sequence can be masked");

                        currentSequenceLength++;
                    }
                }

                sequenceLengths[i] = currentSequenceLength;
            }
        };

        if ((numSequences == 1) || (maxNumTimeSteps == 1))
        {
            // The data need not be shuffled
            std::shared_ptr<const Matrix<ElementType>> matrixData = value->Data()->GetMatrix<ElementType>(varShape.Rank());
            auto layout = std::make_shared<MBLayout>();
            if (!mask)
            {
                if (maxNumTimeSteps == 1)
                    layout->InitAsFrameMode(numSequences);
                else
                {
                    layout->Init(numSequences, maxNumTimeSteps);
                    layout->AddSequence(0, 0, 0, maxNumTimeSteps);
                }
            }
            else
            {
                layout->Init(numSequences, maxNumTimeSteps);

                std::vector<ptrdiff_t> sequenceBeginIndices(numSequences, 0);
                std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
                getSequenceStartsAndLengthsFunc(mask, sequenceBeginIndices, sequenceLengths);

                for (size_t i = 0; i < numSequences; ++i)
                    layout->AddSequence(i, i, sequenceBeginIndices[i], sequenceLengths[i]);
            }

            return{ matrixData , layout};
        }
        else
        {
            std::vector<ptrdiff_t> sequenceBeginIndices(numSequences, 0);
            std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
            if (mask != nullptr)
                getSequenceStartsAndLengthsFunc(mask, sequenceBeginIndices, sequenceLengths);

            bool hasTruncatedSequences = std::find_if(sequenceBeginIndices.begin(), sequenceBeginIndices.end(), [](const int& val) { return (val < 0); }) != sequenceBeginIndices.end();

            auto layout = std::make_shared<MBLayout>();
            std::vector<std::pair<size_t, size_t>> placement;
            if (!hasTruncatedSequences)
            {
                std::vector<MBLayout::SequenceInfo> sequences;
                for (size_t i = 0; i < numSequences; ++i)
                    sequences.push_back({ i, SIZE_MAX, sequenceBeginIndices[i], sequenceLengths[i] });

                std::vector<size_t> rowAllocations;
                layout->InitAsPackedSequences(sequences, placement, rowAllocations);
            }
            else
            {
                layout->Init(numSequences, maxNumTimeSteps);

                // We cannot pack as some of the sequences are truncated and thus all sequences have to be
                // kept in their original parallel streams
                placement.resize(numSequences);
                for (size_t i = 0; i < numSequences; ++i)
                {
                    layout->AddSequence(i, i, sequenceBeginIndices[i], sequenceLengths[i]);

                    // Add the gap if there is one
                    if (sequenceLengths[i] < maxNumTimeSteps)
                        layout->AddSequence(GAP_SEQUENCE_ID, i, sequenceLengths[i], maxNumTimeSteps);

                    placement[i] = std::make_pair(i, 0);
                }
            }

            if (maxNumTimeSteps != layout->GetNumTimeSteps())
                LogicError("The number of time steps in the packed MBLayout does not match the longest sequence's length in the Value object");

            if (numSequences != layout->GetNumSequences())
                LogicError("The number of sequences in the packed MBLayout does not match the sequence count in the Value object");

            // The data needs to be rearranged since CNTK requires sequences to be interleaved across timesteps
            // Now generate the gather indices
            auto matrixData = std::make_shared<Matrix<ElementType>>(varShape.TotalSize(),
                                                                    layout->GetNumCols(),
                                                                    AsCNTKImplDeviceId(value->Device()),
                                                                    value->IsSparse() ? MatrixType::SPARSE : MatrixType::DENSE,
                                                                    AsCNTKImplMatrixFormat(value->GetStorageFormat()));

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

            auto gatherIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), gatherIndicesVector.data(), AsCNTKImplDeviceId(value->Device()));
            matrixData->DoGatherColumnsOf(0, *gatherIdxMatrix, *(value->Data()->GetMatrix<ElementType>(varShape.Rank())), 1);
            return{ matrixData, layout };
        }
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        NDShape valueDataShape = sampleShape;

        size_t maxNumTimeSteps = 1;
        size_t numSequences = 1;
        if (layout != nullptr)
        {
            maxNumTimeSteps = layout->GetNumTimeSteps();
            numSequences = layout->GetNumSequences();
            valueDataShape = valueDataShape.AppendShape({ maxNumTimeSteps, numSequences });
        }

        auto createMaskFunc = [](const MBLayoutPtr& layout, const DeviceDescriptor& device, std::vector<size_t>& sequencesShorterThanLongestSequence) {
            std::vector<bool> sequenceBeginFlags;
            std::vector<size_t> sequenceLengths;
            sequencesShorterThanLongestSequence.clear();

            size_t maxNumTimeSteps = layout->GetNumTimeSteps();
            size_t numSequences = layout->GetNumSequences();
            auto& layoutSequences = layout->GetAllSequences();

            size_t sequenceIdx = 0;
            bool allSequencesStartInThisMB = true;
            bool allSequencesSameLength = true;
            for (auto sequenceInfo : layoutSequences)
            {
                if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
                {
                    auto currentSequenceBeginIdx = std::max<ptrdiff_t>(0, sequenceInfo.tBegin);
                    auto currentSequenceEndIdx = std::min(maxNumTimeSteps, sequenceInfo.tEnd);
                    auto currentSequenceLength = (currentSequenceEndIdx - currentSequenceBeginIdx);
                    auto isCurrentSequenceBeginningInsideThisMB = sequenceInfo.tBegin >= 0;

                    allSequencesStartInThisMB = allSequencesStartInThisMB && isCurrentSequenceBeginningInsideThisMB;
                    allSequencesSameLength = allSequencesSameLength && (currentSequenceLength == maxNumTimeSteps);

                    sequenceBeginFlags.push_back(isCurrentSequenceBeginningInsideThisMB);
                    sequenceLengths.push_back(currentSequenceLength);

                    if (currentSequenceLength != maxNumTimeSteps)
                        sequencesShorterThanLongestSequence.push_back(sequenceIdx);

                    sequenceIdx++;
                }
            }

            if (!allSequencesStartInThisMB && (numSequences != layout->GetNumParallelSequences()))
                LogicError("Cannot create an unpacked Value object from packed data where one or more sequences are truncated");

            bool maskNeeded = !allSequencesSameLength || !allSequencesStartInThisMB;

            NDMaskPtr mask;
            if (maskNeeded)
            {
                mask = MakeSharedObject<NDMask>(NDShape({ maxNumTimeSteps, numSequences }), device);
                for (size_t i = 0; i < numSequences; ++i)
                    if (sequenceBeginFlags[i])
                        mask->MarkSequenceBegin({0, i});

                for (auto shortSequenceIdx : sequencesShorterThanLongestSequence)
                    mask->InvalidateSection({ sequenceLengths[shortSequenceIdx], shortSequenceIdx }, { NDShape::InferredDimension, 1 });
            }

            return mask;
        };

        // No data shuffling needed if no layout or the layout has just one time-step or just one sequence
        std::vector<size_t> sequencesShorterThanLongestSequence;
        if ((maxNumTimeSteps == 1) || (numSequences == 1))
        {
            // Just create a view over the existing matrix itself
            auto tensorView = new TensorView<ElementType>(std::make_shared<Matrix<ElementType>>(matrix.AsReference()), AsTensorViewShape(valueDataShape));
            auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), valueDataShape, readOnly, tensorView);
            if (layout == nullptr)
                return MakeSharedObject<Value>(data);
            else
            {
                auto mask = createMaskFunc(layout, AsDeviceDescriptor(matrix.GetDeviceId()), sequencesShorterThanLongestSequence);
                return MakeSharedObject<Value>(data, mask);
            }
        }

        if (layout->GetNumCols() != matrix.GetNumCols())
            LogicError("Bad MBLayout: The number of columns in the MBLayout does not match the number of columns in the data matrix!");

        // Reshuffle to data to unpack and uninterleave the CNTK form packed data
        // Now generate the scatter indices
        auto shuffledMatrixData = std::make_shared<Matrix<ElementType>>(matrix.GetNumRows(), maxNumTimeSteps * numSequences, matrix.GetDeviceId(), matrix.GetMatrixType(), matrix.GetFormat());
        auto mask = createMaskFunc(layout, AsDeviceDescriptor(matrix.GetDeviceId()), sequencesShorterThanLongestSequence);

        // Set the target location of all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
        size_t targetColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
        std::vector<ElementType> scatterIndicesVector(layout->GetNumCols(), (ElementType)targetColIdxForInvalidColumns);

        size_t i = 0;
        auto& layoutSequences = layout->GetAllSequences();
        for (auto sequenceInfo : layoutSequences)
        {
            if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
            {
                size_t targetParallelStreamIdx = sequenceInfo.s;
                auto currentSequenceBeginIdx = std::max<ptrdiff_t>(0, sequenceInfo.tBegin);
                auto currentSequenceEndIdx = std::min(maxNumTimeSteps, sequenceInfo.tEnd);
                size_t currentSequenceLength = (currentSequenceEndIdx - currentSequenceBeginIdx);

                for (size_t j = 0; j < currentSequenceLength; ++j)
                    scatterIndicesVector[((currentSequenceBeginIdx + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx] = (ElementType)((i * maxNumTimeSteps) + j);

                i++;
            }
        }

        auto scatterIdxMatrix = std::make_shared<Matrix<ElementType>>(1, layout->GetNumCols(), scatterIndicesVector.data(), matrix.GetDeviceId());
        shuffledMatrixData->DoScatterColumnsOf(0, *scatterIdxMatrix, matrix, 1);

        auto tensorView = new TensorView<ElementType>(shuffledMatrixData, AsTensorViewShape(valueDataShape));
        auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(shuffledMatrixData->GetFormat()), valueDataShape, readOnly, tensorView);
        return MakeSharedObject<Value>(data, mask);
    }

    template <typename ElementType>
    /*static*/ ValuePtr CompositeFunction::GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        if (var.DynamicAxes().size() > 2)
            LogicError("More than 2 dynamic axis for a variable is currently unsupported");

        if (AsDataType<ElementType>() != var.GetDataType())
            LogicError("The specified ElementType %s does not match the DataType %s", typeid(ElementType).name(), DataTypeName(var.GetDataType()));

        if ((layout != nullptr) && (matrix.GetNumRows() != var.Shape().TotalSize()))
            LogicError("Unexpected matrix layout: The number of rows in the matrix does not match the sample size of the Variable");

        return GetValueObjectFromCNTKImplMatrixAndMBLayout(var.Shape(), matrix, layout, readOnly);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeValue(const std::pair<Variable, ValuePtr>& variableValue, ComputationNodeBasePtr& computationNode)
    {
        std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CNTKMatrixAndMBLayout;
        auto packedValue = dynamic_cast<PackedValue*>(variableValue.second.get());
        if (packedValue)
            CNTKMatrixAndMBLayout = packedValue->PackedData<ElementType>();
        else
            CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableValue.first, variableValue.second);

        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;

        auto& nodeData = computationNode->As<ComputationNode<ElementType>>()->Value();

        // Switch the node matrix to the right matrix type
        nodeData.AssignValuesOf(*CNTKMatrixAndMBLayout.first);
        computationNode->GetMBLayout()->CopyFrom(layout);
    }

    void CompositeFunction::PopulateNetworkInputs(const std::unordered_map<Variable, ValuePtr>& arguments)
    {
        std::vector<ComputationNodeBasePtr> inputNodes;
        for (auto argumentValuePair : arguments)
        {
            auto argument = argumentValuePair.first;
            auto argumentComputationNode = m_variableToNodeMap[argument];
            assert(argumentComputationNode);
            inputNodes.push_back(argumentComputationNode);

            ValuePtr argumentValue = arguments.at(argument);

            MBLayoutPtr layout;
            switch (argumentValue->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeValue<float>({ argument, argumentValue }, argumentComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeValue<double>({ argument, argumentValue }, argumentComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(argumentValue->GetDataType()));
                break;
            }
        }

        m_computationNetwork->BumpEvalTimeStamp(inputNodes);
    }

    template <typename ElementType>
    /*static*/ void CompositeFunction::PopulateComputationNodeGradient(const std::pair<Variable, ValuePtr>& variableGradient, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode)
    {
        std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> CNTKMatrixAndMBLayout;
        auto packedValue = dynamic_cast<PackedValue*>(variableGradient.second.get());
        if (packedValue)
            CNTKMatrixAndMBLayout = packedValue->PackedData<ElementType>();
        else
            CNTKMatrixAndMBLayout = GetCNTKImplMatrixAndMBLayoutFromValueObject<ElementType>(variableGradient.first, variableGradient.second);

        MBLayoutPtr layout = CNTKMatrixAndMBLayout.second;
        auto nodeLayout = computationNode->GetMBLayout();
        if (((layout == nullptr) != (nodeLayout == nullptr)) || ((layout != nullptr) && (*layout != *nodeLayout)))
            InvalidArgument("The layout of the specified gradient Value is incompatible with the layout of the corresponding Variable computed during Forward call");
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

            switch (gradientValue->GetDataType())
            {
            case DataType::Float:
                PopulateComputationNodeGradient<float>(gradientVarValuePair, outputComputationNode);
                break;
            case DataType::Double:
                PopulateComputationNodeGradient<double>(gradientVarValuePair, outputComputationNode);
                break;
            default:
                LogicError("Unsupported DataType %s", DataTypeName(gradientValue->GetDataType()));
                break;
            }
        }
    }

    static NDShape GetValueShape(const Variable& var, const ComputationNodeBasePtr& computationNodePtr)
    {
        size_t outputValueNumAxes = var.Shape().Rank();

        // Add the batch and dynamic axes if needed
        if (computationNodePtr->GetMBLayout() != nullptr)
            outputValueNumAxes += 2;

        std::vector<size_t> outputShapeDims(outputValueNumAxes);
        for (size_t i = 0; i < var.Shape().Rank(); ++i)
            outputShapeDims[i] = computationNodePtr->GetSampleLayout().GetDim(i);

        if (computationNodePtr->GetMBLayout() != nullptr)
        {
            outputShapeDims[var.Shape().Rank()] = computationNodePtr->GetMBLayout()->GetNumTimeSteps();
            outputShapeDims[var.Shape().Rank() + 1] = computationNodePtr->GetMBLayout()->GetNumSequences();
        }

        return NDShape(outputShapeDims);
    }

    /*static*/ void CompositeFunction::GetNodeOutputOrGradient(Variable var, ValuePtr& varValue, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode, bool getGradient)
    {
        auto valueShape = GetValueShape(var, computationNode);
        if (varValue != nullptr)
        {
            // TODO: The shape of the specified output Value object must match the actual output shape
            if (varValue->Shape() != valueShape)
                InvalidArgument("The shape %S of the specified Value object for %s does not match the actual shape %S", AsStringForErrorReporting(varValue->Shape()).c_str(), getGradient ? "gradient" : "output", AsStringForErrorReporting(valueShape).c_str());
        }

        ValuePtr nodeValue;
        auto layout = computationNode->GetMBLayout();
        switch (var.GetDataType())
        {
        case DataType::Float:
        {
            auto& matrix = getGradient ? computationNode->As<ComputationNode<float>>()->Gradient() : computationNode->As<ComputationNode<float>>()->Value();
            if (varValue == nullptr)
                nodeValue = MakeSharedObject<PackedValue>(var.Shape(), std::make_shared<Matrix<float>>(matrix.AsReference()), layout, /*readOnly =*/ false);
            else
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(var, matrix, layout);
            break;
        }
        case DataType::Double:
        {
            auto& matrix = getGradient ? computationNode->As<ComputationNode<double>>()->Gradient() : computationNode->As<ComputationNode<double>>()->Value();
            if (varValue == nullptr)
                nodeValue = MakeSharedObject<PackedValue>(var.Shape(), std::make_shared<Matrix<double>>(matrix.AsReference()), layout, /*readOnly =*/ false);
            else
                nodeValue = GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(var, matrix, layout);
            break;
        }
        default:
            LogicError("Unsupported DataType %s", DataTypeName(var.GetDataType()));
            break;
        }

        if (varValue == nullptr)
            varValue = nodeValue->DeepClone();
        else
            varValue->CopyFrom(*nodeValue);
    }

    void CompositeFunction::GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs)
    {
        // Now copy the Forward values of output nodes from the network to outputs' Value objects
        for (auto outputVarValuePair : outputs)
            GetNodeOutputOrGradient(outputVarValuePair.first, outputs[outputVarValuePair.first], m_variableToNodeMap[outputVarValuePair.first], false /*getGradient*/);
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

            if (!computationNodePtr->NeedsGradient())
                LogicError("Backpropagated gradient value cannot be read from a ComputationNode that has NeedsGradient set to false");

            GetNodeOutputOrGradient(gradientVarValuePair.first, gradients[gradientVarValuePair.first], computationNodePtr, true /*getGradient*/);
        }
    }

    const std::vector<Variable>& CompositeFunction::GetArgumentDependencies(const Variable& output)
    {
        assert(output.IsOutput());

        auto iter = m_perOutputVarArgumentDependencies.find(output);
        if (iter != m_perOutputVarArgumentDependencies.end())
            return iter->second;

        auto wrappedComposite = CompositeFunction::Create(output.Owner());
        m_perOutputVarArgumentDependencies[output] = wrappedComposite->Arguments();

        return m_perOutputVarArgumentDependencies[output];
    }

    /*virtual*/ BackPropStatePtr CompositeFunction::Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                                            std::unordered_map<Variable, ValuePtr>& outputs,
                                                            const DeviceDescriptor& computeDevice,
                                                            const std::unordered_set<Variable>& outputsToRetainBackwardStateFor)
    {
        // Validate arguments and outputs
        if (outputs.empty())
            InvalidArgument("CompositeFunction::Forward: At least one output has to be specified!");

        // Make sure that the DataType of the variables and corresponding values match
        // TODO: We need a better way to determine the ElementType for the network
        auto dataType = DataType::Unknown;
        for (auto variableValuePair : arguments)
        {
            if (dataType == DataType::Unknown)
                dataType = variableValuePair.first.GetDataType();
            else if (dataType != variableValuePair.first.GetDataType())
                LogicError("CompositeFunction::Forward: The DataType of all arguments of the Function must be same");
        }

        if (dataType == DataType::Unknown)
        {
            for (auto variableValuePair : outputs)
            {
                if (dataType == DataType::Unknown)
                    dataType = variableValuePair.first.GetDataType();
            }
        }

        if (dataType == DataType::Float)
            GetComputationNetwork<float>(computeDevice, outputsToRetainBackwardStateFor, true);
        else if (dataType == DataType::Double)
            GetComputationNetwork<double>(computeDevice, outputsToRetainBackwardStateFor, true);
        else
            InvalidArgument("Unsupported DataType %s", DataTypeName(dataType));

        std::unordered_set<Variable> functionOutputs(this->Outputs().begin(), this->Outputs().end());
        std::vector<ComputationNodeBasePtr> outputsToEvaluate;
        std::unordered_set<Variable> requiredArguments;
        for (auto outputVarValuePair : outputs)
        {
            // Ensure that only a subset of this function's outputs are being asked to be evaluated
            if (functionOutputs.find(outputVarValuePair.first) == functionOutputs.end())
                InvalidArgument("Requested output is not an Ouptut of the Function");

            auto& requiredArgumentsForCurrentOutput = GetArgumentDependencies(outputVarValuePair.first);
            requiredArguments.insert(requiredArgumentsForCurrentOutput.begin(), requiredArgumentsForCurrentOutput.end());

            auto outputComputationNode = m_variableToNodeMap[outputVarValuePair.first];
            outputsToEvaluate.push_back(outputComputationNode);
        }

        // TODO: Avoid copying the data when possible

        // We should have argument values supplied for all required argument dependencies for the requested outputs
        for (auto requiredArgument : requiredArguments)
        {
            if (arguments.find(requiredArgument) == arguments.end())
                InvalidArgument("Function::Forward: Required argument's (%S) value that the requested output(s) depend on has not been provided", requiredArgument.Name().c_str());
        }

        // Feed data into the arguments of the network
        PopulateNetworkInputs(arguments);

        // Dropout nodes have an implicit input in the form of the random mask that is applied to its explicit input
        // This mask is regerated every minibatch and hence dropout nodes with a non-zero dropout rate must me marked outdated
        // w.r.t. inputs to force evaluation in each minibatch
        list<ComputationNodeBasePtr> dropoutNodes = m_computationNetwork->GetNodesWithType(OperationNameOf(DropoutNode));
        for (auto& nodeIter : dropoutNodes)
            nodeIter->SetEvalTimeStampOutdatedWrtAll();

        // The 'outputsToRetainBackwardStateFor' nodes also need to be evaluated if not already specified in 'outputs'
        for (auto rootVarForBackprop : outputsToRetainBackwardStateFor)
        {
            if (functionOutputs.find(rootVarForBackprop) == functionOutputs.end())
                InvalidArgument("Requested outputs to retain backward state for is not an Ouptut of the Function");

            if (outputs.find(rootVarForBackprop) == outputs.end())
                outputsToEvaluate.push_back(m_variableToNodeMap[rootVarForBackprop]);
        }

        // TODO: Verify that values were supplied for all inputs that requested outputs depend on

        ScopedNetworkOperationMode modeGuard(m_computationNetwork, outputsToRetainBackwardStateFor.empty() ? NetworkOperationMode::inferring : NetworkOperationMode::training);

        m_computationNetwork->ForwardProp(outputsToEvaluate);

        GetNetworkOutputs(outputs);

        // TODO: How to deal with the specified 'computeDevice'
        Variable evalTimeStampVariable;
        if (arguments.empty())
            evalTimeStampVariable = Inputs()[0];
        else
            evalTimeStampVariable = arguments.begin()->first;

        return (outputsToRetainBackwardStateFor.size() > 0) ? MakeSharedObject<CNTKBackPropState>(this->shared_from_this(), std::make_pair(evalTimeStampVariable, m_variableToNodeMap[evalTimeStampVariable]->GetEvalTimeStamp())) : nullptr;
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
        ScopedNetworkOperationMode modeGuard(m_computationNetwork, NetworkOperationMode::training);

        auto rootComputationNodePtr = m_variableToNodeMap[rootGradientValues.begin()->first];
        m_computationNetwork->GetNestedNetwork(rootComputationNodePtr)->Backprop(FrameRange(nullptr), true, true);

        GetNetworkGradients(backPropagatedGradientValuesForInputs);

        // TODO: How to deal with the specified 'computeDevice'
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
        return Floor(Plus(operand, Constant::Scalar(operand.GetDataType(), 0.5)), name);
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
        if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
            LogicError("TransposeAxes currently does not support transposing dynamic axes");

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
        if (axis == Axis::DefaultBatchAxis())
            LogicError("Slice is currently unsupported along the batch axis");

        if (axis.IsStaticAxis())
        {
            if ((endIndex - beginIndex) <= 0)
                InvalidArgument("CNTK::Slice: endIndex (%d) - beginIndex (%d) must be a positive number", endIndex, beginIndex);

            return Internal::Slice(operand, axis, beginIndex, endIndex, name);
        }

        if ((beginIndex == 0) && (endIndex == 0))
            return operand;

        auto operandAxes = operand.DynamicAxes();
        auto findAxis = std::find(operandAxes.begin(), operandAxes.end(), axis);
        if (findAxis == operandAxes.end())
            InvalidArgument("The specified dynamic axis named %S does not match any of the dynamic axes of the operand", axis.Name().c_str());

        auto beginFlagsLambda = [beginIndex, operand]() {
            return (beginIndex > 0) ? Minus(Constant::Scalar(operand.GetDataType(), 1.0), Internal::IsWithin(operand, beginIndex)) : Internal::IsWithin(operand, beginIndex);
        };

        auto endFlagsLambda = [endIndex, operand]() {
            return (endIndex > 0) ? Internal::IsWithin(operand, endIndex) : Minus(Constant::Scalar(operand.GetDataType(), 1.0), Internal::IsWithin(operand, endIndex));
        };

        FunctionPtr flags;
        if (beginIndex == 0)
            flags = endFlagsLambda();
        else if (endIndex == 0)
            flags = beginFlagsLambda();
        else
            flags = ElementTimes(beginFlagsLambda(), endFlagsLambda());

        // Since we are slicing along a dynamic axis, the output variable's dynamic axes will be different than the operand
        std::vector<Axis> newDynamicAxes;
        for (auto operandAxis : operandAxes)
        {
            if (operandAxis == axis)
            {
                int sliceLength = (endIndex - beginIndex);
                size_t multiplicativeFactor = (sliceLength > 0) ? 0 : 1;
                auto derivedDynamicAxes = GetDerivedDynamicAxes(operandAxis, multiplicativeFactor, sliceLength);
                std::copy(derivedDynamicAxes.begin(), derivedDynamicAxes.end(), std::back_inserter(newDynamicAxes));
            }
            else
                newDynamicAxes.push_back(operandAxis);
        }

        return Internal::Gather(operand, flags, newDynamicAxes, name);
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
        additionalProperties[PrimitiveFunction::AttributeNameInferInputRankToMap] = (size_t)inferInputRankToMap;
        return BinaryOp(PrimitiveOpType::Times, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank /*= 1*/, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOutputRank] = outputRank;
        return BinaryOp(PrimitiveOpType::TransposeTimes, leftOperand, rightOperand, std::move(additionalProperties), name);
    }

    FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name)
    {
        auto difference = Minus(prediction, targets);
        auto squaredDifference = ElementTimes(difference, difference);
        return Internal::ReduceElements(squaredDifference, PrimitiveFunction::InternalSumReductionOpName, Axis::AllStaticAxes(), name);
    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name)
    {
        return Minus(ReduceLogSum(prediction, Axis(0)), TransposeTimes(labels, prediction), name);
    }

    FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const std::wstring& name)
    {
        if (topN == 0)
            InvalidArgument("ClassificationError: The topN argument must be > 0!");

        if (topN == 1)
            return Minus(Constant::Scalar(prediction.GetDataType(), 1.0), TransposeTimes(labels, Hardmax(prediction)), name);
        else
        {
            std::vector<Variable> operands = { prediction, labels, Constant::Scalar(prediction.GetDataType(), (double)topN) };
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
        additionalProperties[L"spatial"] = spatial;
        additionalProperties[L"normalizationTimeConstant"] = normalizationTimeConstant;
        additionalProperties[L"blendTimeConstant"] = blendTimeConstant;
        additionalProperties[L"epsilon"] = epsilon;
        additionalProperties[L"useCuDNNEngine"] = useCuDNNEngine;

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
        return Plus(operand, Constant::Scalar(operand.GetDataType(), 0), name);
    }

    namespace Sequence
    {
        void VerifyIsSequence(const Variable& operand)
        {
            // The operand must have at least one dynamic axis and it's first dynamic axis must be ordered
            if (operand.DynamicAxes().empty() || !operand.DynamicAxes()[0].IsOrdered())
                InvalidArgument("A sequence function can only be applied on operands with at least one dynamic axis and whose first dynamic axis is ordered");
        }

        FunctionPtr IsFirst(const Variable& operand, const std::wstring& name)
        {
            VerifyIsSequence(operand);
            return Internal::IsWithin(operand, 1, name);
        }

        FunctionPtr IsLast(const Variable& operand, const std::wstring& name)
        {
            VerifyIsSequence(operand);
            return Internal::IsWithin(operand, -1, name);
        }

        FunctionPtr First(const Variable& operand, const std::wstring& name)
        {
            VerifyIsSequence(operand);
            return Slice(operand, operand.DynamicAxes()[0], 0, 1, name);
        }

        FunctionPtr Last(const Variable& operand, const std::wstring& name)
        {
            VerifyIsSequence(operand);
            return Slice(operand, operand.DynamicAxes()[0], -1, 0, name);
        }

        std::vector<Axis> WhereOpDynamicAxes(const Variable& operand)
        {
            VerifyIsSequence(operand);

            std::vector<Axis> newDynamicAxes = { Axis::NewUniqueDynamicAxis(L"whereNodeDynamicAxis") };
            for (size_t i = 1; i < operand.DynamicAxes().size(); ++i)
                newDynamicAxes.push_back(operand.DynamicAxes()[i]);

            return newDynamicAxes;
        }

        FunctionPtr Where(const Variable& condition, const std::wstring& name)
        {
            return Internal::Where(condition, WhereOpDynamicAxes(condition), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::Gather(operand, condition, WhereOpDynamicAxes(condition), name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            return Internal::Scatter(operand, condition, WhereOpDynamicAxes(condition), name);
        }

        FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name)
        {
            auto dataPadded = Internal::Scatter(operand, Sequence::IsFirst(broadcastAs), broadcastAs.DynamicAxes());
            auto placeHolderOutput = PlaceholderVariable(operand.Shape(), broadcastAs.DynamicAxes());
            auto output = ElementSelect(Sequence::IsFirst(broadcastAs), dataPadded, PastValue(placeHolderOutput), name);
            return output->ReplacePlaceholders({ { placeHolderOutput, output } });
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
                return PastValue(Internal::ZeroesWithDynamicAxesLike(operand), Constant::Scalar(operand.GetDataType(), 1.0), offset, name);
            else
                return FutureValue(Internal::ZeroesWithDynamicAxesLike(operand), Constant::Scalar(operand.GetDataType(), 1.0), -offset, name);
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

        FunctionPtr Where(const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name)
        {
            auto additionalProperties = Dictionary();
            additionalProperties[PrimitiveFunction::AttributeNameNewDynamicAxes] = AsDictionaryValueVector(newDynamicAxes);
            return UnaryOp(PrimitiveOpType::Where, condition, std::move(additionalProperties), name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name)
        {
            return Internal::GatherPacked(operand, Internal::PackedIndex(/*layout of*/ operand, Where(condition, newDynamicAxes)), name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::vector<Axis>& newDynamicAxes, const std::wstring& name)
        {
            return Internal::ScatterPacked(operand, Internal::PackedIndex(/*layout of*/ condition, Where(condition, newDynamicAxes)), /*layout of*/ condition, name);
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
            using namespace std::placeholders;

            if (axis.IsStaticAxis() || (axis == Axis::AllStaticAxes()))
            {
                auto additionalProperties = Dictionary();
                additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
                additionalProperties[PrimitiveFunction::AttributeNameReductionOpName] = reductionOpName;
                return UnaryOp(PrimitiveOpType::ReduceElements, operand, std::move(additionalProperties), name);
            }

            if (axis == Axis::DefaultBatchAxis())
                LogicError("Reduction is currently unsupported along the batch axis");

            if (reductionOpName != PrimitiveFunction::InternalSumReductionOpName)
                LogicError("%S reduction along dynamic axis is currently unsupported", reductionOpName.c_str());

            std::function<FunctionPtr(const Variable& leftOperand, const Variable& rightOperand)> reductionFunctor;
            if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                reductionFunctor = std::bind(Plus, _1, _2, L"");

            // We are reducing over a dynamic axis which is currently implemented using recurrence
            auto cumulativeSumFunctionPlaceholder = PlaceholderVariable(operand.Shape());
            auto prevAccumulatedValuesFunction = PastValue(cumulativeSumFunctionPlaceholder);
            auto cumulativeSumFunction = reductionFunctor(prevAccumulatedValuesFunction, operand);
            cumulativeSumFunction->ReplacePlaceholders({ { cumulativeSumFunctionPlaceholder, cumulativeSumFunction } });

            return CNTK::Slice(cumulativeSumFunction, axis, -1, 0, name);
        }
   }
}
