//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveFunction.h"
#include "CompositeFunction.h"
#include "BlockFunction.h"
#include "Utils.h"
#include "UserFunctionFactory.h"
#include "TrainingNodes.h"
#include "proto/onnx/ONNX.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    /*static*/ UserFunctionFactoryPtr Function::s_userFunctionFactory = std::make_shared<UserFunctionFactory>();

    /*static*/ void Function::RegisterNativeUserFunction(const std::wstring& uniqueOpName, const std::wstring& moduleName, const std::wstring& factoryMethodName)
    {
        s_userFunctionFactory->Register(uniqueOpName, moduleName, factoryMethodName);
    }

    /*static*/ FunctionPtr Function::NativeUserFunction(const std::wstring& opId, const std::vector<Variable>& operands, const Dictionary& functionConfig, const std::wstring& userFunctionInstanceName)
    {
        return AsComposite(s_userFunctionFactory->CreateInstance(opId, operands, functionConfig, userFunctionInstanceName), userFunctionInstanceName);
    }

    bool Internal::IsNativeUserFunctionRegistered(const std::wstring& uniqueOpName)
    {
        return Function::s_userFunctionFactory->IsRegistered(uniqueOpName);
    }

    static std::unordered_map<std::wstring, UDFDeserializeCallbackPtr> udfCallbackMap;
    static std::mutex udfCallbackMapMutex;

    /*static*/ void Function::RegisterUDFDeserializeCallback(const std::wstring& uniqueOpName, const UDFDeserializeCallback& deserializer)
    {
        std::unique_lock<std::mutex> lock(udfCallbackMapMutex);
        auto result = udfCallbackMap.insert({ uniqueOpName, make_shared<UDFDeserializeCallback>(deserializer) });
        if (!result.second)
            InvalidArgument("A callback for the UserFunction with op name '%S' has already been registered.", uniqueOpName.c_str());
    }

    /*static*/ UDFDeserializeCallbackPtr Function::GetUDFDeserializeCallback(const std::wstring& uniqueOpName)
    {
        std::unique_lock<std::mutex> lock(udfCallbackMapMutex);
        if (udfCallbackMap.find(uniqueOpName) == udfCallbackMap.end())
            return nullptr;
        return udfCallbackMap.at(uniqueOpName);
    }

    std::vector<Variable>& Function::InitOutputs()
    {
        if (std::this_thread::get_id() == m_outputInitializingByThreadId)
        {
            // std::call_once may deadlock when re-entering from the same thread that's running the lambda, early exit
            RuntimeError("Re-enter Function::InitOutputs() from Function::InitOutputs(), outputs are not initialized yet");
        }
        std::call_once(m_outputsInitFlag, [this]() {
            m_outputInitializingByThreadId = std::this_thread::get_id();
            std::vector<Variable> outputs;
            outputs.reserve(Function::MaxNumOutputs);
            InferOutputs(outputs);
            for (auto outputVar : outputs)
            {
                if (outputVar.IsOutput() && !outputVar.Owner())
                    outputVar.SetOwner(shared_from_this());

                if (m_rootFunction == nullptr && outputVar.IsOutput() && outputVar.Owner().get() == this)
                {
                    // in case of a primitive function, set uid of output vars to owner function uid + "_Output_" + output index.
                    outputVar.m_dataFields->m_uid = m_uid + L"_" + VariableKindName(outputVar.Kind()) + L"_" + std::to_wstring(m_outputs.size());
                }

                m_outputs.push_back(outputVar);
                if (m_outputs.back().m_outputComposite != nullptr)
                {
                    // Nuke the composite ptr to allow release of cyclic graphs.
                    m_outputs.back().m_outputComposite = nullptr;
                }
            }
            m_outputInitializingByThreadId = std::thread::id();
        });

        return m_outputs;
    }

    std::vector<Variable>& Function::RawOutputs() const
    {
        return const_cast<Function*>(this)->InitOutputs();
    }

    std::shared_ptr<std::vector<Variable>> Function::InputsImpl(bool pythonOperandOrder) const
    {
        std::vector<Variable> inputs;

        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        if (compositeFunction == nullptr)
        {
            // For the Times and TransposeTimes primitive functions, if we want the python operand order
            // then we need to reorder the operands as stored in m_inputs
            const PrimitiveFunction* primitiveFunction = dynamic_cast<const PrimitiveFunction*>(this);
            if (pythonOperandOrder && primitiveFunction && ((primitiveFunction->OpType() == PrimitiveOpType::Times) || (primitiveFunction->OpType() == PrimitiveOpType::TransposeTimes)))
            {
                assert(m_inputs.size() == 2);
                inputs = { m_inputs[1], m_inputs[0] };
            }
            else
                inputs = m_inputs;
        }
        else
            inputs = compositeFunction->DetermineInputs(pythonOperandOrder);

        return std::shared_ptr<std::vector<Variable>>(new std::vector<Variable>(std::move(inputs)), [](std::vector<Variable>* ptr) { delete ptr; });
    }

    std::shared_ptr<std::vector<Variable>> Function::OutputsImpl() const
    {
        std::vector<Variable> outputs;
        std::shared_ptr<const Function> composite = IsComposite() ? this->shared_from_this() : AsComposite(const_cast<Function*>(this)->shared_from_this());
        for (auto& v : RawOutputs())
            outputs.push_back(v.CompositePreservingCopy(composite));

        return std::shared_ptr<std::vector<Variable>>(new std::vector<Variable>(std::move(outputs)), [](std::vector<Variable>* ptr) { delete ptr; });
    }

    Function::Function(const std::vector<Variable>& inputs, const Dictionary& functionConfig, const FunctionPtr& rootFunction, const std::wstring& name, const std::wstring& uid)
        : m_rootFunction(rootFunction), m_name(name), m_uid(uid), m_attributes(std::move(functionConfig))
    {
        for (auto inputVar : inputs)
        {
            m_inputs.push_back(inputVar);
            if (m_inputs.back().m_outputComposite != nullptr)
            {
                // Nuke the composite ptr to allow release of cyclic graphs.
                m_inputs.back().m_outputComposite = nullptr;
            }

            if (!inputVar.IsInput() &&
                !inputVar.IsOutput() &&
                !inputVar.IsParameter() &&
                !inputVar.IsConstant() &&
                !inputVar.IsPlaceholder())
            {
                InvalidArgument("Function '%S' input has invalid VariableKind.", inputVar.AsString().c_str());
            }
        }
    }

    /*virtual*/ Function::~Function() {}

    /*virtual*/ const std::wstring& Function::OpName() const
    {
        static const std::wstring defaultUserFunctionOpName = L"UserFunction";
        return defaultUserFunctionOpName;
    }

    BackPropStatePtr Function::Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
        std::unordered_map<Variable, ValuePtr>& outputs,
        const DeviceDescriptor& computeDevice,
        const std::unordered_set<Variable>& outputsToRetainBackwardStateFor,
        const std::unordered_set<Variable>& inputsToExcludeGradientsFor)
    {
        auto compositeFunction = dynamic_cast<CompositeFunction*>(this);
        if (compositeFunction)
            return compositeFunction->Forward(arguments, outputs, computeDevice, outputsToRetainBackwardStateFor, inputsToExcludeGradientsFor);

        std::vector<ValuePtr> inputValues;
        auto functionInputs = Inputs();
        for (const auto& input : functionInputs)
        {
            ValuePtr inputValue;
            if (input.IsConstant())
                inputValue = MakeSharedObject<Value>(Constant(input).Value());
            else if (input.IsParameter())
                inputValue = MakeSharedObject<Value>(Parameter(input).Value());
            else
            {
                if (arguments.find(input) == arguments.end())
                    InvalidArgument("No value specified for input Variable '%S' of Function '%S'.", input.AsString().c_str(), this->AsString().c_str());

                inputValue = arguments.at(input);
            }

            inputValues.push_back(inputValue);
        }

        return Forward(inputValues, outputs, computeDevice, outputsToRetainBackwardStateFor);
    }

    /*virtual*/ void Function::Backward(const BackPropStatePtr& /*state*/,
        const std::unordered_map<Variable, ValuePtr>& /*rootGradientValues*/,
        std::unordered_map<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/)
    {
        NOT_IMPLEMENTED;
    }

    void Function::Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
        std::unordered_map<Variable, ValuePtr>& gradients,
        std::unordered_map<Variable, ValuePtr>& outputsToEvaluate,
        const DeviceDescriptor& computeDevice)
    {
        auto gradientRoot = Output();
        Gradients(arguments, gradientRoot, gradients, outputsToEvaluate, computeDevice);
    }

    void Function::Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
        Variable& gradientRoot,
        std::unordered_map<Variable, ValuePtr>& gradients,
        std::unordered_map<Variable, ValuePtr>& outputsToEvaluate,
        const DeviceDescriptor& computeDevice)
    {
        if (!this->IsComposite())
            LogicError("Function '%S': Currently 'Gradients' method is only supported for composite Functions.", AsString().c_str());

        auto outputs = outputsToEvaluate;
        if (outputsToEvaluate.find(gradientRoot) == outputsToEvaluate.end())
            outputs.insert({gradientRoot , nullptr});

        // TODO: Exclude inputs not belonging to 'gradients' from the gradient computation
        auto backPropState = this->Forward(arguments, outputs, computeDevice, {gradientRoot});

        for (auto outputVarValuePair : outputsToEvaluate)
            outputsToEvaluate[outputVarValuePair.first] = outputs[outputVarValuePair.first];

        auto gradientRootOutputValue = outputs[gradientRoot];
        auto rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(gradientRoot.GetDataType(), gradientRootOutputValue->Shape(), computeDevice), gradientRootOutputValue->Mask());
        rootGradientValue->Data()->SetValue(1.0f);

        this->Backward(backPropState, {{gradientRoot, rootGradientValue}}, gradients);
    }

    void Function::SetName(const std::wstring& name)
    {
        if (!Name().empty() && !Internal::IsRenamingFunctionsAllowed())
            InvalidArgument("Illegal to set name of a Function '%S' with an existing name '%S'", this->AsString().c_str(), Name().c_str());

        m_name = name;
    }

    bool Function::IsBlock() const
    {
        auto blockFunction = dynamic_cast<const BlockFunction*>(this);
        return (blockFunction != nullptr);
    }

    FunctionPtr Function::BlockRoot() const
    {
        if (!IsBlock())
            InvalidArgument("Function::BlockRoot() called for a Function '%S' which is not a block.", this->AsString().c_str());

        auto blockFunction = dynamic_cast<const BlockFunction*>(this);
        return blockFunction->Composite()->RootFunction();
    }

    std::shared_ptr<std::vector<std::pair<Variable, Variable>>> Function::BlockArgumentsMappingImpl() const
    {
        if (!IsBlock())
            InvalidArgument("Function::BlockArgumentsMapping() called for a Function '%S' which is not a block.", this->AsString().c_str());

        auto blockFunction = dynamic_cast<const BlockFunction*>(this);
        return std::shared_ptr<std::vector<std::pair<Variable, Variable>>>(new std::vector<std::pair<Variable, Variable>>(std::move(blockFunction->CompositeArgumentsMap())), [](std::vector<std::pair<Variable, Variable>>* ptr) { delete ptr; });
    }

    /*static*/ void Function::ReplacePlaceholderInPlace(Variable& var,
        const std::unordered_map<Variable, Variable>& placeholderReplacements,
        std::unordered_set<Variable>& replacedPlaceholders)
    {
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
    }

    // Placeholders can be replaced incrementally - i.e. not all placeholders need to replaced in one go.
    // The only requirement is that they must all be replaced before making any 'Forward' calls on the Function instance.
    /*virtual*/ void Function::ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
        std::unordered_set<const Function*>& visitedFunctions,
        std::unordered_set<Variable>& replacedPlaceholders)
    {
        FunctionPtr primitiveRootFunction = shared_from_this();
        if (IsComposite())
        {
            primitiveRootFunction = RootFunction();
            primitiveRootFunction->ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);
        }
        else
        {
            visitedFunctions.insert(this);

            for (auto& inputVar : m_inputs)
            {
                ReplacePlaceholderInPlace(inputVar, placeholderReplacements, replacedPlaceholders);
                if (inputVar.m_outputComposite != nullptr)
                {
                    // Nuke the composite ptr to allow release of cyclic graphs.
                    inputVar.m_outputComposite = nullptr;
                }

                if (inputVar.IsOutput() && (visitedFunctions.find(inputVar.Owner().get()) == visitedFunctions.end()))
                    inputVar.Owner()->ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);
            }
        }

        auto primitiveRootFunctionPtr = dynamic_cast<const PrimitiveFunction*>(primitiveRootFunction.get());
        if (primitiveRootFunctionPtr && (primitiveRootFunctionPtr->OpType() == PrimitiveOpType::Combine))
        {
            // Combine's outputs are just a copy of its inputs and any replacements need to be properly
            // reflected in the outputs as well
            for (auto& outputVar : InitOutputs())
                ReplacePlaceholderInPlace(outputVar, placeholderReplacements, replacedPlaceholders);
        }

        OnPlaceholdersReplaced(placeholderReplacements, replacedPlaceholders);
    }

    /*virtual*/ void Function::OnPlaceholdersReplaced(const std::unordered_map<Variable, Variable>& /*placeholderReplacements*/,
        std::unordered_set<Variable>& /*replacedPlaceholders*/)
    {}

    /*static*/ bool Function::ValidateOrUpdateOutput(const Variable& currentOutputVar, const Variable& newOutputVar, bool alwaysUpdate)
    {
        bool updated = false;
        if (!alwaysUpdate)
        {
            if (!newOutputVar.Shape().IsUnknown() && (currentOutputVar.Shape() != newOutputVar.Shape()))
            {
                updated = true;
                currentOutputVar.m_dataFields->m_shape = newOutputVar.Shape();
            }

            if (!newOutputVar.Shape().IsUnknown() && (currentOutputVar.NeedsGradient() != newOutputVar.NeedsGradient()))
            {
                updated = true;
                currentOutputVar.m_dataFields->m_needsGradient = newOutputVar.NeedsGradient();
            }

            if ((currentOutputVar.GetDataType() == DataType::Unknown) && (currentOutputVar.GetDataType() != newOutputVar.GetDataType()))
            {
                updated = true;
                currentOutputVar.m_dataFields->m_dataType = newOutputVar.GetDataType();
            }

            if ((currentOutputVar.DynamicAxes() == Axis::UnknownDynamicAxes()) && (currentOutputVar.DynamicAxes() != newOutputVar.DynamicAxes()))
            {
                updated = true;
                currentOutputVar.m_dataFields->m_dynamicAxes = newOutputVar.DynamicAxes();
            }

            if ((!newOutputVar.Shape().IsUnknown() && (currentOutputVar.Shape() != newOutputVar.Shape())) ||
                ((newOutputVar.GetDataType() != DataType::Unknown) && (currentOutputVar.GetDataType() != newOutputVar.GetDataType())) ||
                ((newOutputVar.DynamicAxes() != Axis::UnknownDynamicAxes()) && (currentOutputVar.DynamicAxes() != newOutputVar.DynamicAxes())) ||
                (!newOutputVar.Shape().IsUnknown() && (currentOutputVar.NeedsGradient() != newOutputVar.NeedsGradient())))
            {
                InvalidArgument("New output Variable Shape, DataType, NeedsGradient, Dynamic axes after replaced placeholders does not match previous output Variable, for the Recurrent Function.\n"
                    "New = %S\n"
                    "Previous = %S\n",
                    newOutputVar.AsString().c_str(), currentOutputVar.AsString().c_str());
            }
        }
        else
        {
            currentOutputVar.m_dataFields->m_shape = newOutputVar.Shape();
            currentOutputVar.m_dataFields->m_dataType = newOutputVar.GetDataType();
            currentOutputVar.m_dataFields->m_needsGradient = newOutputVar.NeedsGradient();
            currentOutputVar.m_dataFields->m_dynamicAxes = newOutputVar.DynamicAxes();
            updated = true;
        }

        if (currentOutputVar.Owner()->IsBlock())
            currentOutputVar.m_dataFields->m_blockFunctionVariableMapping = newOutputVar.BlockFunctionVariableMapping();

        return updated;
    }

    void Function::ValidateOrUpdateOutputs()
    {
        // Validate/update the output shapes, data types etc. to reflect any changes
        // in inputs due to placeholder replacements
        std::unordered_map<const Function*, size_t> functionVisitCounts;

        // An arbitrary cap on changing output shape of recurrent nodes, to detect infinite inference loops
        const size_t maxNumValidationPassesAllowed = 128;
        bool recurrentNodeOutputModified = false;
        size_t numValidationPasses = 0;
        std::vector<Variable> outputVarBuffer;
        outputVarBuffer.reserve(Function::MaxNumOutputs);
        do
        {
            recurrentNodeOutputModified = false;
            functionVisitCounts.clear();
            RootFunction()->ValidateOrUpdateOutputs(functionVisitCounts, recurrentNodeOutputModified, outputVarBuffer);
            numValidationPasses++;
        } while (recurrentNodeOutputModified && (numValidationPasses < maxNumValidationPassesAllowed));

        if (numValidationPasses >= maxNumValidationPassesAllowed)
            LogicError("Function '%S' ReplacePlaceholders: A recurrent node output shape change happened in max allowed (%d) successive validation passes "
                "indicating a potential infinite inference loop.", AsString().c_str(), (int)numValidationPasses);
    }

    void Function::ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions, bool& recurrentNodeOutputModified, std::vector<Variable>& outputsUsingNewInputs)
    {
        assert(visitedFunctions.find(this) == visitedFunctions.end());
        visitedFunctions[this] = 1;

        // Validate each of the inputs first
        for (auto input : m_inputs)
        {
            if (input.IsOutput())
            {
                auto owner = input.Owner().get();
                if (visitedFunctions.find(owner) == visitedFunctions.end())
                {
                    outputsUsingNewInputs.clear();
                    owner->ValidateOrUpdateOutputs(visitedFunctions, recurrentNodeOutputModified, outputsUsingNewInputs);
                }
                else
                    visitedFunctions[owner]++;
            }
        }

        outputsUsingNewInputs.clear();
        this->InferOutputs(outputsUsingNewInputs);
        auto currentOutputs = RawOutputs();
        for (size_t i = 0; i < currentOutputs.size(); ++i)
        {
            auto newOutputVar = outputsUsingNewInputs[i];
            auto currentOutputVar = currentOutputs[i];

            bool isRecurrent = (visitedFunctions[this] > 1);
            bool outputUpdated = ValidateOrUpdateOutput(currentOutputVar, newOutputVar, !isRecurrent);
            recurrentNodeOutputModified = recurrentNodeOutputModified || (isRecurrent && outputUpdated);
        }
    }

    void Function::Evaluate(const std::unordered_map<Variable, ValuePtr>& arguments,
        std::unordered_map<Variable, ValuePtr>& outputs,
        const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        Forward(arguments, outputs, computeDevice, {});
    }

    void Function::Save(std::vector<unsigned char> &vectorBuf)
    {
        Dictionary model = Serialize();
        std::ostringstream stream;
        stream << model;
        stream.flush();

        std::string const& s = stream.str();
        vectorBuf.reserve(s.size());
        vectorBuf.assign(s.begin(), s.end());
    }

    void Function::Save(const std::wstring& filepath, ModelFormat format)
    {
        switch (format)
        {
        case ModelFormat::CNTKv2:
        {
            Dictionary model = Serialize();
            auto stream = GetFstream(filepath, false);
            *stream << model;
            stream->flush();
            break;
        }

        case ModelFormat::ONNX:
        {
            ONNXFormat::Save(RootFunction(), filepath);
            break;
        }
        }
    }

    /*static*/ FunctionPtr Function::Load(const std::wstring& filepath, const DeviceDescriptor& computeDevice, ModelFormat format)
    {
        switch (format)
        {
        case ModelFormat::CNTKv2:
        {
            auto stream = GetFstream(filepath, true);
            if (!Internal::IsLegacyModel(*stream))
            {
                Dictionary model;
                *stream >> model;
                return Function::Deserialize(model, computeDevice);
            }
            else
            {
                return Internal::LoadLegacyModel(filepath, computeDevice); // throw an exception if deserializer != nullptr?
            }
            break;
        }

        case ModelFormat::ONNX:
            return ONNXFormat::Load(filepath, computeDevice);
            break;
        }

        return nullptr;
    }

    /*static*/ FunctionPtr Function::Load(const char *buffer, size_t length, const DeviceDescriptor& computeDevice)
    {
        if ((buffer == nullptr) || (length <= 0))
            InvalidArgument("The model buffer should not be null and its length should be greater than 0");

        struct modelStreamBuffer : std::streambuf
        {
            modelStreamBuffer(const char* start, size_t size) {
                // std::streambuf::setg() requires char *
                char* first = const_cast<char*>(start);
                this->setg(first, first, first + size);
            }
        };

        if (Internal::IsLegacyModel(buffer, length))
            InvalidArgument("Loading a legacy model from byte array is not supported.");
        else
        {
            modelStreamBuffer buf(buffer, length);
            std::istream modelStream(&buf);

            return Load(modelStream, computeDevice);
        }
    }

    /*static*/ FunctionPtr Function::Load(std::istream& inputStream, const DeviceDescriptor& computeDevice)
    {
        Dictionary model;
        inputStream >> model;
        return Function::Deserialize(model, computeDevice);
    }

    void Function::Restore(const std::wstring& filepath)
    {
        auto stream = GetFstream(filepath, true);
        if (!Internal::IsLegacyModel(*stream))
        {
            Dictionary model;
            *stream >> model;
            RestoreFromCheckpoint(model);
            return;
        }

        auto loadedModelFunction = Internal::LoadLegacyModel(filepath, DeviceDescriptor::CPUDevice());
        // TODO: Make sure that the loaded model is the same as the trainer's model through UID matching in the V2 format
        // TODO: For V1 format models make sure that the loaded model is isomorphic to the trainer's model
        auto loadedModelLeafVariables = loadedModelFunction->Inputs();
        auto trainerModelLeafVariables = Inputs();
        if (trainerModelLeafVariables.size() != loadedModelLeafVariables.size())
            InvalidArgument("The loaded Function '%S' leaf variables do not match those of the Function '%S' being restored.",
                loadedModelFunction->AsString().c_str(), this->AsString().c_str());

        std::map<std::wstring, Variable> loadedModelLeafVariablesMap;
        for (auto leafVar : loadedModelLeafVariables)
            loadedModelLeafVariablesMap[leafVar.Uid()] = leafVar;

        std::map<std::wstring, Variable> trainerModelLeafVariablesMap;
        for (auto leafVar : trainerModelLeafVariables)
            trainerModelLeafVariablesMap[leafVar.Uid()] = leafVar;

        // Remove the initial state inputs of PastValue and FutureValue functions from the maps if they are a scalar constant
        // since these are not part of the internal CNTK serialized computation graph
        std::function<void(const std::unordered_set<FunctionPtr>&, std::map<std::wstring, Variable>&)> RemovePastAndFutureValueInitialStateScalarConstants;
        RemovePastAndFutureValueInitialStateScalarConstants = [&RemovePastAndFutureValueInitialStateScalarConstants](const std::unordered_set<FunctionPtr>& allPrimitiveFunctions, std::map<std::wstring, Variable>& modelLeafVariableMap) {
            for (auto funcPtr : allPrimitiveFunctions)
            {
                auto primitiveFunction = dynamic_cast<const PrimitiveFunction*>(funcPtr.get());
                if ((primitiveFunction->OpType() == PrimitiveOpType::PastValue) || (primitiveFunction->OpType() == PrimitiveOpType::FutureValue))
                {
                    auto initialStateInput = primitiveFunction->Inputs()[1];
                    if (initialStateInput.IsConstant() && (initialStateInput.Shape().TotalSize() == 1))
                        modelLeafVariableMap.erase(initialStateInput.Uid());
                }
                else if (primitiveFunction->OpType() == PrimitiveOpType::Block)
                {
                    auto blockFunction = dynamic_cast<const BlockFunction*>(primitiveFunction);
                    auto blockComposite = dynamic_cast<const CompositeFunction*>(blockFunction->Composite().get());
                    RemovePastAndFutureValueInitialStateScalarConstants(blockComposite->m_allPrimitiveFunctions, modelLeafVariableMap);
                }
            }
        };

        auto loadedModelCompositeFunction = dynamic_cast<const CompositeFunction*>(loadedModelFunction.get());
        RemovePastAndFutureValueInitialStateScalarConstants(loadedModelCompositeFunction->m_allPrimitiveFunctions, loadedModelLeafVariablesMap);

        auto trainerModelCompositeFunction = dynamic_cast<CompositeFunction*>(this);
        RemovePastAndFutureValueInitialStateScalarConstants(trainerModelCompositeFunction->m_allPrimitiveFunctions, trainerModelLeafVariablesMap);

        // Now update the trainer's model parameters and constants with those from the loaded model
        for (auto nameVarPair : trainerModelLeafVariablesMap)
        {
            auto trainerModelLeafVar = nameVarPair.second;

            auto areVariablesEquivalent = [](const Variable& left, const Variable& right) {
                return Internal::AreEquivalent(left, right) && (left.Uid() == right.Uid());
            };

            auto correspondingLoadedModelVar = loadedModelLeafVariablesMap.at(trainerModelLeafVar.Uid());

            if (!areVariablesEquivalent(correspondingLoadedModelVar, trainerModelLeafVar))
                InvalidArgument("The loaded Function '%S' leaf variables do not match those of the Function '%S' being restored.",
                    loadedModelFunction->AsString().c_str(), this->AsString().c_str());

            if ((trainerModelLeafVar.IsConstant() && !Constant(trainerModelLeafVar).Value()->IsReadOnly()) || trainerModelLeafVar.IsParameter())
            {
                auto trainerModelVarValue = trainerModelLeafVar.IsConstant() ? Constant(trainerModelLeafVar).Value() : Parameter(trainerModelLeafVar).Value();
                auto loadedModelVarValue = correspondingLoadedModelVar.IsConstant() ? Constant(correspondingLoadedModelVar).Value() : Parameter(correspondingLoadedModelVar).Value();
                trainerModelVarValue->CopyFrom(*loadedModelVarValue);
            }
        }

        trainerModelCompositeFunction->CopyState(*loadedModelCompositeFunction);
    }

    Variable GetCorrespondingOutputVariableFromClone(const Variable& cloneeOutput, const FunctionPtr& cloneeFunction, const FunctionPtr& clonedFunction)
    {
        size_t outputVarIndex = 0;
        for (auto output : cloneeFunction->RawOutputs())
        {
            if (output == cloneeOutput)
                break;

            outputVarIndex++;
        }

        return clonedFunction->RawOutputs()[outputVarIndex];
    }

    FunctionPtr Function::ReplacePlaceholder(const Variable& placeholderReplacement)
    {
        auto placeholders = Placeholders();
        if (placeholders.size() != 1)
            InvalidArgument("ReplacePlaceholder called with a single replacement Variable '%S' but this Function '%S' has %d placeholders '%S'",
                placeholderReplacement.AsString().c_str(),
                this->AsString().c_str(),
                (int)placeholders.size(),
                NamedListString(placeholders).c_str());

        return ReplacePlaceholders({ { *(placeholders.begin()), placeholderReplacement } });
    }

    FunctionPtr Function::ReplacePlaceholders(const std::unordered_map<Variable, Variable>& placeholderReplacements)
    {
        std::unordered_set<const Function*> visitedFunctions;
        std::unordered_set<Variable> replacedPlaceholders;
        ReplacePlaceholdersInPlace(placeholderReplacements, visitedFunctions, replacedPlaceholders);

        // Validate/update the output shapes, data types etc. to reflect any changes in inputs due to placeholder replacements
        RootFunction()->ValidateOrUpdateOutputs();

        for (auto replacementPair : placeholderReplacements)
        {
            if (replacedPlaceholders.find(replacementPair.first) == replacedPlaceholders.end())
                InvalidArgument("Placeholder '%S' specified for replacement not found in the Function '%S'.",
                    replacementPair.first.AsString().c_str(), this->AsString().c_str());
        }

        return this->shared_from_this();
    }

    FunctionPtr Function::FlattenFunction(const FunctionPtr& clonee, const std::vector<Variable>& clonedInputs)
    {
        FunctionPtr clonedFunction;
        const BlockFunction* blockFunction = dynamic_cast<const BlockFunction*>(clonee.get());
        if (!blockFunction)
            return clonee->Clone(clonedInputs);

        std::unordered_map<Variable, Variable> cloneeToClonedInputMap;
        auto cloneeInputs = clonee->Inputs();
        std::transform(cloneeInputs.begin(), cloneeInputs.end(), clonedInputs.begin(),
            std::inserter(cloneeToClonedInputMap, cloneeToClonedInputMap.end()),
            std::make_pair<const Variable&, const Variable&>);

        auto cloneeComposite = blockFunction->Composite();
        auto cloneeCompositeInputs = cloneeComposite->Inputs();
        std::unordered_map<Variable, Variable> cloneeCompositeReplacements;

        // Make sure we that during cloning we substitue all block arguments with the corresponding
        // cloned inputs.
        for (auto cloneeArgumentMapping : blockFunction->BlockArgumentsMapping())
            cloneeCompositeReplacements.insert({ cloneeArgumentMapping.first, cloneeToClonedInputMap.at(cloneeArgumentMapping.second) });

        for (size_t i = 0; i < clonedInputs.size(); ++i)
        {
            const auto& cloneeInput = cloneeInputs[i];
            const auto& clonedInput = clonedInputs[i];
            if ((cloneeInput != clonedInput) && (cloneeInput.IsParameter() || cloneeInput.IsConstant()))
            {
                auto iter = std::find(cloneeCompositeInputs.begin(), cloneeCompositeInputs.end(), cloneeInput);
                if (iter != cloneeCompositeInputs.end())
                {
                    auto cloneeCompositeInput = *iter;
                    cloneeCompositeReplacements.insert({ cloneeCompositeInput, clonedInput });
                }
            }
        }
        return cloneeComposite->CloneImpl(ParameterCloningMethod::Share, cloneeCompositeReplacements, FlattenFunction);
    }

    FunctionPtr Function::Clone(const FunctionPtr& clonee,
        ParameterCloningMethod parameterCloneMethod,
        const std::unordered_map<Variable, Variable>& replacements,
        std::unordered_map<const Function*, FunctionPtr>& cloneMap,
        std::unordered_map<Variable, Variable>& leafVariablesCloneMap,
        std::unordered_map<Variable, Variable>& placeholderReplacements,
        std::function<FunctionPtr(const FunctionPtr&, const std::vector<Variable>&)> clone)
    {
        if (cloneMap.find(clonee.get()) != cloneMap.end())
            LogicError("Function::Clone: Cloning an already visited Function '%S'.", clonee->AsString().c_str());

        cloneMap[clonee.get()] = nullptr;

        std::unordered_map<Variable, Variable> cloneeToClonedInputMap;
        std::vector<Variable> inputs;
        auto cloneeInputs = clonee->Inputs();
        for (auto cloneeInput : cloneeInputs)
        {
            Variable clonedInput;
            if (replacements.find(cloneeInput) != replacements.end())
            {
                auto replacement = replacements.at(cloneeInput);
                clonedInput = PlaceholderLike(replacement);
                placeholderReplacements[clonedInput] = replacement;
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
                                {
                                    //parameter values can be updated so we need our own copy
                                    const auto& ndav = Parameter(cloneeInput).Value();
                                    clonedInput = Constant(ndav->DeepClone(ndav->Device(), ndav->IsReadOnly()), cloneeInput.Name());
                                }
                                else
                                {
                                    //constants can also be updated via non-sgd means
                                    const auto& ndav = Constant(cloneeInput).Value();
                                    clonedInput = Constant(ndav->DeepClone(ndav->Device(), ndav->IsReadOnly()), cloneeInput.Name());
                                }
                                leafVariablesCloneMap[cloneeInput] = clonedInput;
                                break;
                            default:
                                LogicError("Function::Clone: Unknown ParameterCloningMethod.");
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
                                clonedInput = PlaceholderVariable();
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
                        auto clonedFunction = Clone(cloneeInput.Owner(), parameterCloneMethod, replacements, cloneMap, leafVariablesCloneMap, placeholderReplacements, clone);
                        clonedInput = GetCorrespondingOutputVariableFromClone(cloneeInput, cloneeInput.Owner(), clonedFunction);
                    }
                }
            }

            inputs.push_back(clonedInput);
            cloneeToClonedInputMap.insert({cloneeInput, clonedInput});
        }

        FunctionPtr clonedFunction = clone(clonee, inputs);
        cloneMap[clonee.get()] = clonedFunction;
        return clonedFunction;
    }

    FunctionPtr Function::CloneFunction(const FunctionPtr& clonee, const std::vector<Variable>& clonedInputs)
    {
        auto inputs = clonedInputs;
        auto cloneeInputs = clonee->Inputs();

        FunctionPtr clonedFunction;
        const BlockFunction* blockFunction = dynamic_cast<const BlockFunction*>(clonee.get());
        if (!blockFunction)
            return clonee->Clone(inputs);

        std::unordered_map<Variable, Variable> cloneeToClonedInputMap;
        std::transform(cloneeInputs.begin(), cloneeInputs.end(), clonedInputs.begin(),
            std::inserter(cloneeToClonedInputMap, cloneeToClonedInputMap.end()),
            std::make_pair<const Variable&, const Variable&>);

        {
            auto cloneeComposite = blockFunction->Composite();
            auto cloneeCompositeInputs = cloneeComposite->Inputs();
            std::unordered_map<Variable, Variable> cloneeCompositeReplacements;
            std::vector<std::pair<Variable, Variable>> clonedBlockCompositeArgumentsMap;

            // Create blank placeholders in the cloned block's composite to prevent carrying over any old
            // type information. The type information of the block's placeholders should be derived
            // afresh from the mappings
            for (auto cloneeCompositeInput : cloneeCompositeInputs)
            {
                if (IsArgument(cloneeCompositeInput))
                    cloneeCompositeReplacements.insert({ cloneeCompositeInput, PlaceholderVariable() });
            }

            // When cloning the block, we need to replace any Parameter/Constants inside the block with
            // the correspondind replacements if any
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                auto cloneeInput = cloneeInputs[i];
                auto clonedInput = inputs[i];
                if ((cloneeInput != clonedInput) && (cloneeInput.IsParameter() || cloneeInput.IsConstant()))
                {
                    auto iter = std::find(cloneeCompositeInputs.begin(), cloneeCompositeInputs.end(), cloneeInput);
                    if (iter != cloneeCompositeInputs.end())
                    {
                        auto cloneeCompositeInput = *iter;
                        Variable replacement = clonedInput;
                        if (IsArgument(replacement))
                        {
                            replacement = PlaceholderLike(inputs[i]);
                            clonedBlockCompositeArgumentsMap.push_back({ replacement, inputs[i] });
                        }

                        cloneeCompositeReplacements.insert({ cloneeCompositeInput, replacement });
                    }
                }
            }

            // We will not have the block's internal composite create new clones of Parameters/Constants since
            // in the case we want to really clone, they have been cloned as part of cloning the inputs of the
            // block and will be handled through the replacements
            auto clonedComposite = cloneeComposite->Clone(ParameterCloningMethod::Share, cloneeCompositeReplacements);

            auto clonedCompositeInputs = clonedComposite->Inputs();
            std::unordered_map<Variable, Variable> cloneeToClonedBlockCompositeArgumentsMap;
            for (size_t i = 0; i < cloneeCompositeInputs.size(); ++i)
            {
                if (IsArgument(cloneeCompositeInputs[i]))
                    cloneeToClonedBlockCompositeArgumentsMap.insert({ cloneeCompositeInputs[i], clonedCompositeInputs[i] });
            }

            auto cloneeBlockCompositeArgumentsMap = blockFunction->BlockArgumentsMapping();
            for (auto cloneeArgumentMapping : cloneeBlockCompositeArgumentsMap)
                clonedBlockCompositeArgumentsMap.push_back({ cloneeToClonedBlockCompositeArgumentsMap.at(cloneeArgumentMapping.first), cloneeToClonedInputMap.at(cloneeArgumentMapping.second) });

            clonedFunction = MakeSharedObject<BlockFunction>(std::move(clonedComposite), clonedBlockCompositeArgumentsMap, blockFunction->OpName(), Dictionary(blockFunction->Attributes()), blockFunction->Name());
            auto clonedFunctionInputs = clonedFunction->Inputs();
            if (clonedFunctionInputs != inputs)
                LogicError("Block Function '%S': Inputs '%S' of the new clone do not match the cloned inputs '%S' of the clonee Block Function.",
                    clonedFunction->AsString().c_str(),
                    NamedListString(clonedFunctionInputs).c_str(),
                    NamedListString(inputs).c_str());
        }

        return clonedFunction;
    }

    FunctionPtr Function::Clone(ParameterCloningMethod parameterCloneMethod, const std::unordered_map<Variable, Variable>& replacements) const
    {
        return CloneImpl(parameterCloneMethod, replacements, CloneFunction);
    }

    FunctionPtr Function::CloneFlattened(ParameterCloningMethod parameterCloneMethod) const
    {
        return CloneImpl(parameterCloneMethod, {}, FlattenFunction);
    }

    FunctionPtr Function::CloneImpl(
        ParameterCloningMethod parameterCloneMethod,
        const std::unordered_map<Variable, Variable>& replacements,
        std::function<FunctionPtr(const FunctionPtr&, const std::vector<Variable>&)> clone) const
    {
        const CompositeFunction* compositeFunction = dynamic_cast<const CompositeFunction*>(this);
        if (compositeFunction == nullptr)
            LogicError("Function '%S': Currently only cloning of composite functions is supported.", AsString().c_str());

        auto compositeRootFunction = compositeFunction->RootFunction();

        // Handle the scenario when the root Function outputs themselves are specified to be replaced.
        auto compositeRootFunctionOutputs = compositeRootFunction->RawOutputs();
        std::vector<Variable> rootFunctionOutputReplacements;
        for (auto output : compositeRootFunctionOutputs)
        {
            if (replacements.find(output) != replacements.end())
                rootFunctionOutputReplacements.push_back(replacements.at(output));
        }

        if (!rootFunctionOutputReplacements.empty())
        {
            if (rootFunctionOutputReplacements.size() != compositeRootFunctionOutputs.size())
                InvalidArgument("Function '%S': Clone replacements contain some of the root Function's outputs but not all.", AsString().c_str());

            if (rootFunctionOutputReplacements.size() == 1)
                return rootFunctionOutputReplacements[0];
            else
            {
                std::unordered_set<FunctionPtr> owners;
                for (auto replacementOutput : rootFunctionOutputReplacements)
                    owners.insert(replacementOutput.Owner());

                if ((owners.size() == 1) && *owners.begin())
                    return AsComposite(*owners.begin());
                else
                    return Combine(rootFunctionOutputReplacements);
            }
        }

        std::unordered_map<const Function*, FunctionPtr> cloneMap;
        std::unordered_map<Variable, Variable> leafVariablesCloneMap;
        std::unordered_map<Variable, Variable> placeholderReplacements;
        auto clonedRootFunction = Function::Clone(compositeRootFunction, parameterCloneMethod, replacements, cloneMap, leafVariablesCloneMap, placeholderReplacements, clone);

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
                            auto visitedFunctionOutputs = visitedFunction->RawOutputs();
                            for (auto visitedFunctionOutput : visitedFunctionOutputs)
                                cloningReplacementsForPlaceholderReplacement[visitedFunctionOutput] = GetCorrespondingOutputVariableFromClone(visitedFunctionOutput, visitedFunction, cloneMap.at(visitedFunction.get()));
                        }
                    }

                    if (!cloningReplacementsForPlaceholderReplacement.empty())
                    {
                        auto replacementToClone = AsComposite(varPair.second.Owner());
                        auto replacementClone = replacementToClone->CloneImpl(parameterCloneMethod, cloningReplacementsForPlaceholderReplacement, clone);
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

        auto clonedComposite = AsComposite(clonedRootFunction, compositeFunction->Name());
        clonedComposite->ReplacePlaceholders(placeholderReplacements);
        return clonedComposite;
    }


    /*virtual*/ void Function::RestoreFromCheckpoint(const Dictionary& modelDictionary)
    {
        CompositeFunction* compositeFunction = dynamic_cast<CompositeFunction*>(this);
        if (compositeFunction == nullptr)
            InvalidArgument("Primitive Function '%S' instance cannot be restored from a checkpoint.", this->AsString().c_str());

        auto restoredFunction = Function::Deserialize(modelDictionary, DeviceDescriptor::CPUDevice());
        //TODO (backcompat): when loading a stale model we can still pass this test
        // by patching up restored functions on the fly during deserialization (e.g., by
        // inserting an extra input for the sample count in case of BatchNorm).
        if (!Internal::AreEquivalent(shared_from_this(), restoredFunction))
            InvalidArgument("Function '%S' being restored is not equivalent (isomorphic) to the Function '%S' loaded from checkpoint.",
                this->AsString().c_str(), restoredFunction->AsString().c_str());

        auto inputs = Inputs();
        auto restoredInputs = restoredFunction->Inputs();

        assert(inputs.size() == restoredInputs.size());

        for (int i = 0; i < inputs.size(); i++)
        {
            assert(inputs[i].Kind() == restoredInputs[i].Kind());
            if (!inputs[i].IsConstant() && !inputs[i].IsParameter())
                continue;
            assert(Internal::AreEquivalent(inputs[i], restoredInputs[i]));
            inputs[i].Value()->CopyFrom(*(restoredInputs[i].Value().get()));
        }

        auto restoredCompositeFunction = dynamic_cast<const CompositeFunction*>(restoredFunction.get());
        compositeFunction->CopyState(*restoredCompositeFunction);
    }

    /*static*/ FunctionPtr Function::Deserialize(const Dictionary& modelDictionary, const CNTK::DeviceDescriptor& device)
    {
        return CompositeFunction::Deserialize(modelDictionary, device);
    }

    std::wstring Function::AsString(bool doNotInferOutputs) const
    {
        wstringstream wss;
        bool first = true;
        if (IsComposite())
            wss << "Composite(" << RootFunction()->OpName() << "): ";
        else
            wss << OpName() <<": ";
        bool reverse = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
        for (auto arg : Arguments(reverse))
            wss << (first ? (first = false, "") : ", ") << arg.AsString();

        wss << " -> ";
        if (doNotInferOutputs && m_outputs.empty())
            wss << "Unknown";
        else
        {
            first = true;
            for (auto out : Outputs())
                wss << (first ? (first = false, "") : ", ") << out.AsString();
        }
        return wss.str();
    }


    void Function::SetAttribute(const std::wstring& name, const DictionaryValue& value)
    {
        Function* functionPtr = !(this->IsComposite()) ? this : this->RootFunction().get();
        PrimitiveFunction* primitiveFunctionPtr = dynamic_cast<PrimitiveFunction*>(functionPtr);

        if (primitiveFunctionPtr == nullptr)
        {
            // Possibly, a udf...
            LogicError("SetAttribute cannot be invoked on an instance of function '%S'.", AsString().c_str());
        }

        if (name == PrimitiveFunction::AttributeNameDropoutRate)
        {
            double dropout;
            if (value.ValueType() == DictionaryValue::Type::Float)
                dropout = value.Value<float>();
            else
                dropout = value.Value<double>();

            primitiveFunctionPtr->SetDropoutRate(dropout);
        }
        else if (name == PrimitiveFunction::AttributeNameRngSeed)
        {
            size_t seed;
            if (value.ValueType() == DictionaryValue::Type::Int)
                seed = value.Value<int>();
            else
                seed = value.Value<size_t>();

            primitiveFunctionPtr->SetRandomSeed(seed);
        }
        else
        {
            LogicError("SetAttribute: '%S' is not supported (this attribute cannot be updated).", name.c_str());
        }
    }

    Dictionary& Function::GetCustomAttributes()
    {
        if (!m_attributes.Contains(PrimitiveFunction::AttributeNameCustomAttributes))
        {
            ResetCustomAttributes();
        }
        return m_attributes[PrimitiveFunction::AttributeNameCustomAttributes].Value<Dictionary>();
    }

    void Function::ResetCustomAttributes()
    {
        m_attributes[PrimitiveFunction::AttributeNameCustomAttributes] = Dictionary();
    }

    FunctionPtr NullaryOp(PrimitiveOpType op, Dictionary&& opConfig, const std::wstring& name)
    {
        std::vector<Variable> operands{};
        return AsComposite(MakeSharedObject<PrimitiveFunction>(op, operands, std::move(opConfig), name), name);
    }

    FunctionPtr UnaryOp(PrimitiveOpType op, const Variable& operand, Dictionary&& opConfig, const std::wstring& name)
    {
        std::vector<Variable> operands = { operand };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(op, operands, std::move(opConfig), name), name);
    }

    FunctionPtr Negate(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Negate, operand, Dictionary(), name);
    }

    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::StableSigmoid, operand, Dictionary(), name);
    }

    FunctionPtr Atanh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Atanh, operand, Dictionary(), name);
    }

    FunctionPtr Tanh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Tanh, operand, Dictionary(), name);
    }

    FunctionPtr Asin(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Asin, operand, Dictionary(), name);
    }

    FunctionPtr Sin(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Sin, operand, Dictionary(), name);
    }

    FunctionPtr Acos(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Acos, operand, Dictionary(), name);
    }

    FunctionPtr Cos(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Cos, operand, Dictionary(), name);
    }

    FunctionPtr Cosh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Cosh, operand, Dictionary(), name);
    }

    FunctionPtr Asinh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Asinh, operand, Dictionary(), name);
    }

    FunctionPtr Sinh(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Sinh, operand, Dictionary(), name);
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

    FunctionPtr Softmax(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        if (!axis.IsStaticAxis() && (axis != Axis::AllStaticAxes()))
            LogicError("Softmax: support only static axes.");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;

        if (((operand.Shape().Rank() == 1) && (axis.StaticAxisIndex() == 0)) ||
            (axis == Axis::AllStaticAxes()))
        {
            return UnaryOp(PrimitiveOpType::Softmax, operand, std::move(additionalProperties), name);
        }
        else
        {
            auto operandPlaceholder = PlaceholderVariable();
            auto operandDelta = operandPlaceholder - ReduceMax(operandPlaceholder, axis);
            auto expOperandDelta = Exp(operandDelta);
            auto result = ElementDivide(expOperandDelta, ReduceSum(expOperandDelta, axis));

            return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"Softmax", name);
        }
    }

    FunctionPtr LogSoftmax(const Variable& operand, const std::wstring& name)
    {
        return LogSoftmax(operand, Axis(0), name);
    }

    FunctionPtr LogSoftmax(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        if (!axis.IsStaticAxis() && !axis.IsBatchAxis())
            LogicError("Softmax: only batch and static axes are supported.");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;

        auto operandPlaceholder = PlaceholderVariable();

        auto result = operandPlaceholder - Log(ReduceSum(Exp(operandPlaceholder), axis));

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"LogSoftmax", name);
    }

    FunctionPtr Hardmax(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Hardmax, operand, Dictionary(), name);
    }

    FunctionPtr HardSigmoid(const Variable& operand, float alpha, float beta, const std::wstring& name)
    {
        // f(x) = max(0,min(alpha*x+beta,1))
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAlpha] = alpha;
        additionalProperties[PrimitiveFunction::AttributeNameBeta] = beta;

        auto alphaConstant = Constant::Scalar(operand.GetDataType(), alpha);
        auto betaConstant = Constant::Scalar(operand.GetDataType(), beta);
        auto one = Constant::Scalar(operand.GetDataType(), 1.0);
        auto zero = Constant::Scalar(operand.GetDataType(), 0.0);
        auto operandPlaceholder = PlaceholderVariable();

        auto result = Plus(ElementTimes(operandPlaceholder, alphaConstant), betaConstant);

        result = ElementSelect(Less(result, one), result, one);
        result = ElementSelect(Greater(result, zero), result, zero);

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"HardSigmoid", name);
    }

    FunctionPtr TopK(const Variable& operand, size_t k, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = Axis(0);
        additionalProperties[PrimitiveFunction::AttributeNameNumItems] = k;
        return UnaryOp(PrimitiveOpType::TopK, operand, std::move(additionalProperties), name);
    }


    FunctionPtr TopK(const Variable& operand, size_t k, const Axis& axis, const std::wstring& name)
    {
        if (!axis.IsStaticAxis())
            LogicError("TopK operation only supports a single static axis.");

        if (axis.StaticAxisIndex() == 0)
            return TopK(operand, k, name);
        else
        {
            auto additionalProperties = Dictionary();
            additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
            additionalProperties[PrimitiveFunction::AttributeNameNumItems] = k;

            auto operandPlaceholder = PlaceholderVariable();
            auto firstAxis = Axis(0);
            auto swapped = TransposeAxes(operandPlaceholder, firstAxis, axis);
            auto topkSwapped = TopK(swapped, k, name);
            auto outputs = topkSwapped->Outputs();
            auto topkValues = TransposeAxes(outputs[0], firstAxis, axis);
            auto topkIndices = TransposeAxes(outputs[1], firstAxis, axis);
            auto result = Combine({ topkValues , topkIndices });
            return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"TopK", name);
        }
    }

    FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis1] = axis1;
        additionalProperties[PrimitiveFunction::AttributeNameAxis2] = axis2;
        return UnaryOp(PrimitiveOpType::TransposeAxes, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Transpose(const Variable& operand, const std::vector<Axis>& permutation, const std::wstring& name)
    {
        //Check all the axes
        if (!std::all_of(permutation.begin(), permutation.end(), [](const Axis& a) { return a.IsStaticAxis(); }))
            LogicError("Transpose: Permutation vector must only contain static axes.");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(permutation);
        return UnaryOp(PrimitiveOpType::TransposeAxes, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Transpose(const Variable& operand, const std::wstring& name)
    {
        if (operand.Shape().Rank() != 2)
            InvalidArgument("Transpose called with operand '%S'; only 2D operands are supported.", operand.AsString().c_str());

        return TransposeAxes(operand, Axis(0), Axis(1), name);
    }

    FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::wstring& name)
    {
        std::vector<int> strides(axis.size(), 1);
        return Slice(operand, axis, beginIndex, endIndex, strides, name);
    }

    FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::vector<int>& strides, const std::wstring& name)
    {
        if (std::all_of(axis.cbegin(), axis.cend(), [](Axis axis) { return axis.IsStaticAxis(); }))
            return Internal::Slice(operand, axis, beginIndex, endIndex, strides, name);

        LogicError("Slice: Invalid axis argument provided. Slice along the dynamic batch axis is currently unsupported. To slice a sequence along its ordered dynamic axis use Sequence::Slice.");
    }

    FunctionPtr RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNumSamples] = numSamples;
        additionalProperties[PrimitiveFunction::AttributeNameAllowDuplicates] = allowDuplicates;

        if (seed == SentinelValueForAutoSelectRandomSeed)
            seed = Internal::GenerateRandomSeed(true);

        additionalProperties[PrimitiveFunction::AttributeNameRngSeed] = size_t(seed);
        additionalProperties[PrimitiveFunction::AttributeNameRngOffset] = size_t(0);

        return UnaryOp(PrimitiveOpType::RandomSample, operand, std::move(additionalProperties), name);
    }

    FunctionPtr RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNumSamples] = numSamples;
        additionalProperties[PrimitiveFunction::AttributeNameAllowDuplicates] = allowDuplicates;

        if (seed == SentinelValueForAutoSelectRandomSeed)
            seed = Internal::GenerateRandomSeed(true);

        additionalProperties[PrimitiveFunction::AttributeNameRngSeed] = size_t(seed);
        additionalProperties[PrimitiveFunction::AttributeNameRngOffset] = size_t(0);

        return UnaryOp(PrimitiveOpType::RandomSampleInclusionFrequency, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Dropout(const Variable& operand, double dropoutRate, unsigned long seed, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameDropoutRate] = dropoutRate;

        if (seed == SentinelValueForAutoSelectRandomSeed)
            seed = Internal::GenerateRandomSeed(true);

        additionalProperties[PrimitiveFunction::AttributeNameRngSeed] = size_t(seed);
        additionalProperties[PrimitiveFunction::AttributeNameRngOffset] = size_t(0);

        return UnaryOp(PrimitiveOpType::Dropout, operand, std::move(additionalProperties), name);
    }

    Dictionary CreateRandomDistributionAttributes(const wstring& type, const std::vector<double>& args, unsigned long seed)
    {
        auto additionalProperties = Dictionary();

        if (seed == SentinelValueForAutoSelectRandomSeed)
            seed = Internal::GenerateRandomSeed(true);
        additionalProperties.Add(
            PrimitiveFunction::AttributeNameRandomDistributionType, type,
            PrimitiveFunction::AttributeNameRandomDistributionArgs, AsDictionaryValueVector(args),
            PrimitiveFunction::AttributeNameRngSeed, size_t(seed),
            PrimitiveFunction::AttributeNameRngOffset, size_t(0));
        return additionalProperties;
    }

    Dictionary CreateRandomDistributionAttributes(const wstring& type, const std::vector<double>& args, unsigned long seed, const NDShape& shape, DataType dataType)
    {
        auto additionalProperties = CreateRandomDistributionAttributes(type, args, seed);
        additionalProperties.Add(
            PrimitiveFunction::AttributeNameNewShape, shape,
            PrimitiveFunction::AttributeNameNewDataType, static_cast<int>(dataType));
        return additionalProperties;
    }

    FunctionPtr UniformRandom(const NDShape& shape, DataType dataType, double low, double high, unsigned long seed, const std::wstring& name)
    {
        if (low >= high)
            LogicError("UniformRandom: low end of the range (%g) must be < high end of the range (%g)", low, high);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeUniform, { low, high }, seed, shape, dataType);
        return NullaryOp(PrimitiveOpType::RandomDistribution, std::move(additionalProperties), name);
    }

    FunctionPtr UniformRandomLike(const Variable& operand, double low, double high, unsigned long seed, const std::wstring& name)
    {
        if (low >= high)
            LogicError("UniformRandomLike: low end of the range (%g) must be < high end of the range (%g)", low, high);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeUniform, { low, high }, seed);
        return UnaryOp(PrimitiveOpType::RandomDistribution, operand, std::move(additionalProperties), name);
    }

    FunctionPtr NormalRandom(const NDShape& shape, DataType dataType, double mean, double stdev, unsigned long seed, const std::wstring& name)
    {
        if (stdev < 0)
            LogicError("NormalRandom: standard deviation (%g) must be non-negative", stdev);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeNormal, { mean, stdev }, seed, shape, dataType);
        return NullaryOp(PrimitiveOpType::RandomDistribution, std::move(additionalProperties), name);
    }

    FunctionPtr NormalRandomLike(const Variable& operand, double mean, double stdev, unsigned long seed, const std::wstring& name)
    {
        if (stdev < 0)
            LogicError("NormalRandomLike: standard deviation (%g) must be non-negative", stdev);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeNormal, { mean, stdev }, seed);
        return UnaryOp(PrimitiveOpType::RandomDistribution, operand, std::move(additionalProperties), name);
    }

    FunctionPtr ToBatch(const Variable& operand, const std::wstring& name)
    {
        if (operand.DynamicAxes().size() > 0)
            LogicError("ToBatch: the input should not have dynamic axis.");

        if (operand.Shape().Dimensions().size() == 0)
            LogicError("ToBatch: the input can not be scalar.");

        return UnaryOp(PrimitiveOpType::ToBatch, operand, Dictionary(), name);
    }

    FunctionPtr UnpackBatch(const Variable& operand, const std::wstring& name)
    {
        if (operand.DynamicAxes().size() > 1)
            LogicError("UnpackBatch: only support input with batch axis and no sequence axis.");

        return UnaryOp(PrimitiveOpType::UnpackBatch, operand, Dictionary(), name);
    }

    FunctionPtr Pad(const Variable& operand, PaddingMode mode, const std::vector<size_t>& head, const std::vector<size_t>& foot, double constantValue, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePaddingHead] = AsDictionaryValueVector(head);
        additionalProperties[PrimitiveFunction::AttributeNamePaddingFoot] = AsDictionaryValueVector(foot);
        additionalProperties[PrimitiveFunction::AttributeNamePaddingMode] = (size_t)mode;
        additionalProperties[PrimitiveFunction::AttributeNamePaddingConstantValue] = constantValue;
        return UnaryOp(PrimitiveOpType::Pad, operand, std::move(additionalProperties), name);
    }

    FunctionPtr GumbelRandom(const NDShape& shape, DataType dataType, double loc, double scale, unsigned long seed, const std::wstring& name)
    {
        if (scale < 0)
            LogicError("GumbelRandom: scale (%g) must be non-negative", scale);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeGumbel, { loc, scale }, seed, shape, dataType);
        return NullaryOp(PrimitiveOpType::RandomDistribution, std::move(additionalProperties), name);
    }

    FunctionPtr GumbelRandomLike(const Variable& operand, double loc, double scale, unsigned long seed, const std::wstring& name)
    {
        if (scale < 0)
            LogicError("GumbelRandomLike: scale (%g) must be non-negative", scale);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeGumbel, { loc, scale }, seed);
        return UnaryOp(PrimitiveOpType::RandomDistribution, operand, std::move(additionalProperties), name);
    }

    FunctionPtr BernoulliRandom(const NDShape& shape, DataType dataType, double mean, unsigned long seed, const std::wstring& name)
    {
        if (mean < 0 || mean > 1)
            LogicError("BernoulliRandom: mean (%g) must be between 0 and 1", mean);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeBernoulli, { mean }, seed, shape, dataType);
        return NullaryOp(PrimitiveOpType::RandomDistribution, std::move(additionalProperties), name);
    }

    FunctionPtr BernoulliRandomLike(const Variable& operand, double mean, unsigned long seed, const std::wstring& name)
    {
        if (mean < 0 || mean > 1)
            LogicError("BernoulliRandomLike: mean (%g) must be between 0 and 1", mean);
        Dictionary additionalProperties = CreateRandomDistributionAttributes(Microsoft::MSR::CNTK::RandomDistributionTypeBernoulli, { mean }, seed);
        return UnaryOp(PrimitiveOpType::RandomDistribution, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Flatten(const Variable& operand, const std::wstring& name)
    {
        // sanitize_axis applies Axis(-axis - 1), thus 0 -> -1: which means the last one of operand.Shape().
        Axis axis(-1);
        return Flatten(operand, axis, name);
    }

    FunctionPtr Flatten(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        int cntk_index;
        int onnx_axis;

        // We need to express in onnx axis system to help ONNX conversion.
        if (axis.IsStaticAxis())
        {
            if (axis.StaticAxisIndex() < 0)
            {
                // python shape [2,3,4,5], cntk_py_index = 1 (point at 3). 
                // in python, sanitize_axis applies Axis(-cntk_py_index - 1) so axis = -2
                // in cpp shape becomes [5,4,3,2], axis(-2) is still pointing to 3 (from the last)
                // With ONNX Flatten op, result shall be: [#][[2], [3,4,5]]. thus onnx_axis = cntk_py_index + 1 = 2 (point to 3)
                // for CNTK reshape, cntk_index shall point to the one after 3 (2): cntk_index = axis + 1
                // cntk_index (-1) needs to be converted to positive by rank + cntk_index = 3
                int cntk_py_index = -axis.StaticAxisIndex() - 1;
                onnx_axis = cntk_py_index + 1;
                cntk_index = axis.StaticAxisIndex() + 1;
                cntk_index += operand.Shape().Rank();
            }
            else
            {
                // in this case shape is the same as in python: [2,3,4,5]
                // that is: cntk_py_index = 1, points to 3
                // onnx_axis = 2, points to 3 in [#][[2], [3,4,5]]
                // cntk_index = 1, points to 3 in [2,3,4,5]
                int cntk_py_index = axis.StaticAxisIndex();
                onnx_axis = cntk_py_index + 1;
                cntk_index = axis.StaticAxisIndex();
            }
        }
        else if (axis.IsBatchAxis())
        {
            // expected result: [[batch],[flatten sample]]([[#][2,3,4,5]])
            cntk_index = 0;
            onnx_axis = 1;
        }
        else
        {
            LogicError("Flatten: accept only static and batch axes.");
        }

        if (cntk_index > operand.Shape().Rank())
        {
            LogicError("Flatten: unsupported axis (operand.Shape().Rank() = %d, axis = %s).",
                (int)operand.Shape().Rank(), ToString(axis.AsString()).c_str());
        }

        size_t dim0 = cntk_index == 0 ? 1 : operand.Shape().SubShape(0, cntk_index).TotalSize();
        size_t dim1 = cntk_index == operand.Shape().Rank() ? 1 : operand.Shape().SubShape(cntk_index).TotalSize();

        NDShape newShape({ dim0, dim1 });

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = Axis(onnx_axis);

        auto operandPlaceholder = PlaceholderVariable();

        auto result = Reshape(operandPlaceholder, newShape, name);

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"Flatten", name);
    }

    FunctionPtr Reshape(const Variable& operand, const NDShape& replacementShape, const Axis& beginAxis, const Axis& endAxis, const std::wstring& name)
    {
        if (!beginAxis.IsStaticAxis() || !endAxis.IsStaticAxis())
            LogicError("Reshape: operation does not support reshaping dynamic axis");

        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNewShape] = replacementShape;
        additionalProperties[PrimitiveFunction::AttributeNameBeginAxis] = beginAxis;
        additionalProperties[PrimitiveFunction::AttributeNameEndAxis] = endAxis;

        return UnaryOp(PrimitiveOpType::Reshape, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Squeeze(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::Squeeze, operand, {}, name);

        // TODO: this code is needed for ONNX converter because ONNX requires squeeze axis. However, unit test failed with this code.
        // Need further investigation.
        //auto additionalProperties = Dictionary();
        //additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(GetSqueezableAxes(operand.Shape()));
        //return UnaryOp(PrimitiveOpType::Squeeze, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Squeeze(const Variable& operand, const std::vector<Axis>& axes, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(axes);
        return UnaryOp(PrimitiveOpType::Squeeze, operand, std::move(additionalProperties), name);
    }

    FunctionPtr ExpandDims(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable();
        auto result = Reshape(operandPlaceholder, NDShape({ 1 }), axis, axis);
        return AsBlock(std::move(result), { { operandPlaceholder, operand }}, L"ExpandDims", name);
    }

    FunctionPtr ZerosLike(const Variable& operand, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameFillValue] = 0.0;

        return UnaryOp(PrimitiveOpType::ConstantOp, operand, std::move(additionalProperties), name);
    }

    FunctionPtr OnesLike(const Variable& operand, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameFillValue] = 1.0;

        return UnaryOp(PrimitiveOpType::ConstantOp, operand, std::move(additionalProperties), name);
    }

    std::vector<Variable> AutoBroadcastSequence(PrimitiveOpType op, const Variable& left, const Variable& right, bool autoBroadcast)
    {
        auto left_axis = left.DynamicAxes();
        int left_num_seqs = (int)std::count_if(left_axis.begin(), left_axis.end(), [](Axis a) {return a.IsSequenceAxis(); });
        auto right_axis = right.DynamicAxes();
        int right_num_seqs = (int)std::count_if(right_axis.begin(), right_axis.end(), [](Axis a) {return a.IsSequenceAxis(); });

        vector<Variable> result;
        if ( autoBroadcast &&
            left_axis.size() > 0 &&
            right_axis.size() > 0 &&
            (left_num_seqs + right_num_seqs) == 1)
        {
            if (left_num_seqs == 1)
            {
                auto new_right = CNTK::ReconcileDynamicAxes(right, left);
                result.push_back(left);
                result.push_back(new_right);
            }
            else
            {
                auto new_left = CNTK::ReconcileDynamicAxes(left, right);
                result.push_back(new_left);
                result.push_back(right);

            }
        }
        else
        {
            result.push_back(left);
            result.push_back(right);
        }

        return result;

    }

    FunctionPtr BinaryOp(PrimitiveOpType op, const Variable& leftOperand, const Variable& rightOperand, Dictionary&& opConfig, const std::wstring& name, bool autoBroadcast = true)
    {
        std::vector<Variable> operands = AutoBroadcastSequence(op, leftOperand, rightOperand, autoBroadcast);
        return AsComposite(MakeSharedObject<PrimitiveFunction>(op, operands, std::move(opConfig), name), name);
    }

    FunctionPtr ElementAnd(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();
        auto zero = Constant::Scalar(leftOperand.GetDataType(), 0.0);
        auto result = Greater(ElementTimes(
            Greater(leftOperandPlaceholder, zero),
            Greater(rightOperandPlaceholder, zero)), zero);
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"And", name);
    }

    FunctionPtr ElementNot(const Variable& operand, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable();
        auto result = Plus(
            Negate(Greater(operandPlaceholder, Constant::Scalar(operand.GetDataType(), 0.0))),
            Constant::Scalar(operand.GetDataType(), 1.0));
        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"Not", name);
    }

    FunctionPtr ElementOr(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();
        auto zero = Constant::Scalar(leftOperand.GetDataType(), 0.0);
        auto result = Greater(Plus(
            Greater(leftOperandPlaceholder, zero),
            Greater(rightOperandPlaceholder, zero)), zero);
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"Or", name);
    }

    FunctionPtr ElementXor(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();
        auto zero = Constant::Scalar(leftOperand.GetDataType(), 0.0);
        auto result = NotEqual(
            Greater(leftOperandPlaceholder, zero),
            Greater(rightOperandPlaceholder, zero));
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"Xor", name);
    }

    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Plus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::LogPlus, leftOperand, rightOperand, Dictionary(), name);
    }

    FunctionPtr Pow(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Pow, leftOperand, rightOperand, Dictionary(), name);
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
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();
        auto result = ElementTimes(leftOperandPlaceholder, Reciprocal(rightOperandPlaceholder), name);
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"ElementDivide", name);
    }

    FunctionPtr ElementMax(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();

        auto result = ElementSelect(Greater(leftOperandPlaceholder, rightOperandPlaceholder),
            leftOperandPlaceholder,
            rightOperandPlaceholder);
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"ElementMax", name);
    }

    FunctionPtr ElementMin(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        auto leftOperandPlaceholder = PlaceholderVariable();
        auto rightOperandPlaceholder = PlaceholderVariable();

        auto result = ElementSelect(Less(leftOperandPlaceholder, rightOperandPlaceholder),
            leftOperandPlaceholder,
            rightOperandPlaceholder);
        return AsBlock(std::move(result), { { leftOperandPlaceholder, leftOperand },{ rightOperandPlaceholder, rightOperand } }, L"ElementMin", name);
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

    FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, size_t shiftWindow, size_t numberOfNegativeSamples, const std::wstring& name)
    {
        std::vector<Variable> operands = {leftOperand, rightOperand, Constant::Scalar((float) shiftWindow),  Constant::Scalar((float) numberOfNegativeSamples) };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::CosDistanceWithNegativeSamples, operands, Dictionary(), name), name);
    }

    FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name)
    {
        auto predictionPlaceholder = PlaceholderVariable(L"prediction");
        auto labelPlaceholder = PlaceholderVariable(L"targets");
        Constant onePlusEps = Constant::Scalar(1.0f+1e-6f);
        Constant one = Constant::Scalar(1.0f);
        Constant eps = Constant::Scalar(1e-6f);
        auto compositeBinaryCrossEntropy = Negate(Plus(ElementTimes(labelPlaceholder,Log(eps + predictionPlaceholder)), ElementTimes(Minus(one, labelPlaceholder), Log(Minus(onePlusEps, predictionPlaceholder)))));
        return AsBlock(std::move(compositeBinaryCrossEntropy), { { predictionPlaceholder, prediction },{ labelPlaceholder, targets } }, L"BinaryCrossEntropy", name);
    }

    FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name)
    {
        std::vector<Variable> operands = { prediction, targets, weights };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Logistic, operands, Dictionary(), name), name);
    }

    FunctionPtr NCELoss(const Variable& weights, const Variable& biases, const Variable& inputs, const Variable& labels, const Constant& noiseWeights, size_t numSamples, bool allowDuplicates, unsigned long seed, const std::wstring& name)
    {
        auto inputsPlaceholder = PlaceholderVariable(L"inputs");
        auto labelsPlaceholder = PlaceholderVariable(L"labels");

        auto noiseWeightsShape = noiseWeights.Shape();
        if (noiseWeightsShape.Rank() != 1)
            InvalidArgument("NCELoss: noiseWeights must be a vector");

        auto numClasses = noiseWeightsShape[0];

        if (!weights.IsPlaceholder())
        {
            auto weightsShape = weights.Shape();
            if (weightsShape.Rank() != 2)
                InvalidArgument("NCELoss: weights must have two axes");
            if (weightsShape[1] != numClasses)
                InvalidArgument("NCELoss: the second axis of weights is of length %zd but it is expected to be the same length as the noiseWeights %zd", weightsShape[1], numClasses);
        }

        if (!biases.IsPlaceholder())
        {
            auto biasesShape = biases.Shape();
            if (biasesShape.Rank() != 2)
                InvalidArgument("NCELoss: biases must have two axes");
            if (biasesShape[0] != 1)
                InvalidArgument("NCELoss: the first axis of biases is of length %zd but it is expected to be of length 1", biasesShape[0]);
            if (biasesShape[1] != numClasses)
                InvalidArgument("NCELoss: the first axis of biases is of length %zd but it is expected to be the same length as the noiseWeights %zd", biasesShape[1], numClasses);
        }

        if (!inputs.IsPlaceholder())
        {
            auto inputsShape = inputs.Shape();
            if (inputsShape.Rank() != 1)
                InvalidArgument("NCELoss: inputs must be a vector");
            if (!weights.IsPlaceholder())
            {
                auto weightsShape = weights.Shape();
                if (weightsShape[0] == NDShape::InferredDimension)
                {
                    //Create a function that will result in performing the correct inference
                    //First make the right shape for a constant by starting with a {1} and appending the input shape
                    //You'd think that AppendShape appends in place but you'd be wrong.
                    auto constShape = NDShape({ 1 }).AppendShape(inputsShape);
                    //Next make a constant. The exact datatype does not matter as we will not call forward on the resulting function.
                    auto allZero = Constant(constShape, 0.0f);
                    //Finally make a function we will not use. This will infer the right shape for the weights.
                    auto unused = Times(allZero, weights);
                }
                else if (weightsShape[0] != inputsShape[0])
                    InvalidArgument("NCELoss: the second axis of weights is of length %zd but it is expected to be the same length as the inputs %zd", weightsShape[0], inputsShape[0]);
            }
        }

        if (!labels.IsPlaceholder())
        {
            auto labelsShape = labels.Shape();
            if (labelsShape.Rank() != 1)
                InvalidArgument("NCELoss: labels must be a vector");
            if (labelsShape[0] != numClasses)
                InvalidArgument("NCELoss: the shape of the label (%zd) does not agree with the shape of noiseWeights (%zd)", labelsShape[0], numClasses);
            if (!labels.IsSparse())
                Warning("NCELoss: label is not sparse; gradients will be dense and operations will be slow");
        }

        auto nSamples = Constant::Scalar((float)numSamples);
        auto noiseDistribution = ElementDivide(noiseWeights, ReduceSum(noiseWeights, Axis::AllStaticAxes()));
        auto unnormalizedNoisePrior = ElementTimes(nSamples, noiseDistribution);
        auto logUnnormalizedNoisePrior = Log(unnormalizedNoisePrior);
        auto unormalizedNoisePriorEntropy = ReduceSum(ElementTimes(unnormalizedNoisePrior, logUnnormalizedNoisePrior), Axis::AllStaticAxes());
        auto inclusionProbability = RandomSampleInclusionFrequency(noiseDistribution, numSamples, allowDuplicates, seed);
        auto importanceWeights = ElementDivide(noiseDistribution, inclusionProbability);
        auto reshapedLogNoisePrior = Reshape(logUnnormalizedNoisePrior, NDShape{ { 1, NDShape::InferredDimension } });
        auto combinedFunction = Combine({ unormalizedNoisePriorEntropy, reshapedLogNoisePrior, importanceWeights, noiseDistribution });
        auto outputs = combinedFunction->Outputs();
        auto outputMap = std::unordered_map<Variable, ValuePtr>{ { outputs[0], nullptr}, { outputs[1], nullptr }, { outputs[2], nullptr }, { outputs[3], nullptr } };
        combinedFunction->Forward({}, outputMap, noiseWeights.Value()->Device(), {}, {});
        auto noisePriorEntropy = Constant(outputMap.at(outputs[0])->Data(), L"noisePriorEntropy");
        auto logNoisePrior = Constant(outputMap.at(outputs[1])->Data(), L"logNoisePrior");
        auto importances = Constant(outputMap.at(outputs[2])->Data(), L"importanceWeights");
        auto noise = Constant(outputMap.at(outputs[3])->Data(), L"noise");


        auto inferredVectorShape = NDShape{ {NDShape::InferredDimension} };
        auto negativeSamples = RandomSample(noise, numSamples, allowDuplicates, seed, L"negativeSamples");
        auto selectedImportanceWeights = TransposeTimes(negativeSamples, importances, L"sampledImportanceWeights");
        auto negativeWeights = Times(weights, negativeSamples, L"negativeWeights");
        auto negativeBiases = Times(biases, negativeSamples, L"negativeBiases");
        auto logitsOfNegatives = Plus(TransposeTimes(negativeWeights, inputsPlaceholder), Reshape(negativeBiases, inferredVectorShape), L"negativeLogits");
        auto positiveWeights = Times(weights, labelsPlaceholder, L"positiveWeights");
        auto positiveBiases = Times(biases, labelsPlaceholder, L"positiveBiases");
        auto logitsOfPositives = Plus(ReduceSum(ElementTimes(inputsPlaceholder, positiveWeights), Axis::AllStaticAxes()), Reshape(positiveBiases, {}), L"positiveLogits");

        auto lossOnNegatives = Minus(ElementTimes(nSamples, ReduceSum(ElementTimes(selectedImportanceWeights, LogAddExp(logitsOfNegatives, Reshape(Times(logNoisePrior, negativeSamples), inferredVectorShape))), Axis::AllStaticAxes())), noisePriorEntropy);
        auto lossOnPositives = Minus(LogAddExp(logitsOfPositives, Reshape(Times(logNoisePrior, labelsPlaceholder), {})), logitsOfPositives, L"lossOnPositives");
        auto loss = lossOnPositives + lossOnNegatives;

        return AsBlock(std::move(loss), { { inputsPlaceholder, inputs }, { labelsPlaceholder, labels} }, L"NCE", name);
    }

    FunctionPtr DepthToSpace(const Variable& input, size_t blockSize, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameBlockSize] = blockSize;

        auto inputPlaceholder = PlaceholderVariable(L"input");

        if (!input.IsPlaceholder())
        {
            NDShape inputShape = input.Shape();
            if (inputShape.Rank() != 3)
                LogicError("DepthToSpace: Input operand (shape: %S) must be a 3-dimensional tensor, e.g. a 2D image with channels.", inputShape.AsString().c_str());
            if (inputShape[2] % (blockSize*blockSize) != 0)
                LogicError("DepthToSpace: Number of channels in the operand (%zu) must be divisible by (blocksize x blocksize), i.e., (%zu x %zu).", inputShape[2], blockSize, blockSize);
        }

        FunctionPtr inputView = Reshape(inputPlaceholder, { blockSize, blockSize, NDShape::InferredDimension }, Axis(2), Axis(3));
        std::vector<Axis> axisShufflePermutation({ Axis(2), Axis(0), Axis(3), Axis(1), Axis(4) });
        auto shuffleOut = Transpose(inputView, axisShufflePermutation);
        auto merge23Out = Reshape(shuffleOut, { NDShape::InferredDimension }, Axis(2), Axis(4));
        auto merge01Out = Reshape(merge23Out, { NDShape::InferredDimension }, Axis(0), Axis(2));

        return AsBlock(std::move(merge01Out), { { inputPlaceholder, input } }, std::move(additionalProperties), L"DepthToSpace", name);
    }

    FunctionPtr SpaceToDepth(const Variable& input, size_t blockSize, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameBlockSize] = blockSize;

        auto inputPlaceholder = PlaceholderVariable(L"input");

        if (!input.IsPlaceholder())
        {
            NDShape inputShape = input.Shape();
            if (inputShape.Rank() != 3)
                LogicError("SpaceToDepth: Input operand (shape: %S) must be a 3-dimensional tensor, e.g. a 2D image with channels.", inputShape.AsString().c_str());
            if ((inputShape[0] % blockSize != 0) || (inputShape[1] % blockSize != 0))
                LogicError("SpaceToDepth: All spatial dimensions in the operand (%zu x %zu) must be divisible by blocksize (%zu).", inputShape[0], inputShape[1], blockSize);
        }

        FunctionPtr reshape01out = Reshape(inputPlaceholder, { blockSize, NDShape::InferredDimension }, Axis(0), Axis(1));
        FunctionPtr reshape23out = Reshape(reshape01out, { blockSize, NDShape::InferredDimension }, Axis(2), Axis(3));
        std::vector<Axis> axisShufflePermutation({ Axis(1), Axis(3), Axis(0), Axis(2), Axis(4) });
        auto shuffleOut = Transpose(reshape23out, axisShufflePermutation);
        auto merge234Out = Reshape(shuffleOut, { NDShape::InferredDimension }, Axis(2), Axis::EndStaticAxis());

        return AsBlock(std::move(merge234Out), { { inputPlaceholder, input } }, std::move(additionalProperties), L"SpaceToDepth", name);
    }

    FunctionPtr LambdaRank(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name)
    {
        std::vector<Variable> operands = { prediction, gains, groupId };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::LambdaRank, operands, Dictionary(), name), name);
    }

    FunctionPtr NDCGAt1(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name)
    {
        std::vector<Variable> operands = { prediction, gains, groupId };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::NDCG, operands, Dictionary(), name), name);
    }

    FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name)
    {
        auto predictionPlaceholder = PlaceholderVariable(L"prediction");
        auto targetPlaceholder = PlaceholderVariable(L"target");

        auto difference = Minus(predictionPlaceholder, targetPlaceholder);
        auto squaredDifference = ElementTimes(difference, difference);
        auto squaredErrorComposite = Internal::ReduceElements(squaredDifference, PrimitiveFunction::InternalSumReductionOpName, Axis::AllStaticAxes());

        return AsBlock(std::move(squaredErrorComposite), { { predictionPlaceholder, prediction }, { targetPlaceholder, targets } }, L"SquaredError", name);
    }

    FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name)
    {
        auto predictionPlaceholder = PlaceholderVariable(L"prediction");
        auto labelPlaceholder = PlaceholderVariable(L"label");
        FunctionPtr compositeCEWithSoftmax;
        if (axis == Axis(0))
            compositeCEWithSoftmax = Minus(ReduceLogSum(predictionPlaceholder, axis), TransposeTimes(labelPlaceholder, predictionPlaceholder));
        else
            compositeCEWithSoftmax = Minus(ReduceLogSum(predictionPlaceholder, axis), ReduceSum(ElementTimes(labelPlaceholder, predictionPlaceholder), axis));

        return AsBlock(std::move(compositeCEWithSoftmax), { { predictionPlaceholder, prediction }, { labelPlaceholder, labels } }, L"CrossEntropyWithSoftmax", name);
    }

    FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name)
    {
        if (topN == 0)
            InvalidArgument("ClassificationError: The topN argument must be > 0.");

        if (topN == 1)
        {
            auto predictionPlaceholder = PlaceholderVariable(L"prediction");
            auto labelPlaceholder = PlaceholderVariable(L"label");

            FunctionPtr classificationErrorComposite;
            if (axis == Axis(0))
                classificationErrorComposite = Minus(Constant::Scalar(prediction.GetDataType(), 1.0), TransposeTimes(labelPlaceholder, Hardmax(predictionPlaceholder)));
            else
            {
                auto axMax = ReduceMax(predictionPlaceholder, axis);
                auto pred = Equal(predictionPlaceholder, axMax);
                auto wrongPred = NotEqual(labelPlaceholder, pred);
                auto axErr = ReduceSum(wrongPred, axis);
                auto capErr = GreaterEqual(axErr, Constant::Scalar(prediction.GetDataType(), 1.0));
                classificationErrorComposite = ReduceMean(capErr, Axis::AllStaticAxes());
            }

            return AsBlock(std::move(classificationErrorComposite), { { predictionPlaceholder, prediction }, { labelPlaceholder, labels } }, L"ClassificationError", name);
        }
        else
        {
            if (axis != Axis(0))
                LogicError("ClassificationError: The topN feature is not supported along a specific axis.");

            std::vector<Variable> operands = { prediction, labels, Constant::Scalar((float)topN) };
            return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::ClassificationError, operands, Dictionary(), name), name);
        }
    }

    FunctionPtr EditDistanceError(const Variable& prediction, const Variable& labels, float subPen, float delPen, float insPen, bool squashInputs, const vector<size_t>& tokensToIgnore, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameSubstitutionPenalty] = subPen;
        additionalProperties[PrimitiveFunction::AttributeNameDeletionPenalty] = delPen;
        additionalProperties[PrimitiveFunction::AttributeNameInsertionPenalty] = insPen;
        additionalProperties[PrimitiveFunction::AttributeNameSquashInputs] = squashInputs;
        additionalProperties[PrimitiveFunction::AttributeNameTokensToIgnore] = AsDictionaryValueVector(tokensToIgnore);

        return BinaryOp(PrimitiveOpType::EditDistanceError, prediction, labels, std::move(additionalProperties), name);
    }

    FunctionPtr LatticeSequenceWithSoftmax(const Variable& labels, const Variable& prediction, const Variable& scaledLogLikelihood, const Variable& lattice, const std::wstring& symListPath, const std::wstring& phonePath, const std::wstring& stateListPath, const std::wstring& transProbPath, const std::wstring& latticeConfigPath, float hSmoothingWeight, float frameDropThresh, bool doReferenceAlign, bool seqGammarUsesMBR, float seqGammarAMF, float seqGammarLMF, float seqGammarBMMIFactor, float seqGammarWordPen, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameSymListPath] = symListPath;
        additionalProperties[PrimitiveFunction::AttributeNamePhonePath] = phonePath;
        additionalProperties[PrimitiveFunction::AttributeNameStateListPath] = stateListPath;
        additionalProperties[PrimitiveFunction::AttributeNameTransProbPath] = transProbPath;
        additionalProperties[PrimitiveFunction::AttributeNameLatticeConfigPath] = latticeConfigPath;
        additionalProperties[PrimitiveFunction::AttributeNameHSmoothingWeight] = hSmoothingWeight;
        additionalProperties[PrimitiveFunction::AttributeNameFrameDropThresh] = frameDropThresh;
        additionalProperties[PrimitiveFunction::AttributeNameDoReferenceAlign] = doReferenceAlign;
        additionalProperties[PrimitiveFunction::AttributeNameSeqGammarUsesMBR] = seqGammarUsesMBR;
        additionalProperties[PrimitiveFunction::AttributeNameSeqGammarAMF] = seqGammarAMF;
        additionalProperties[PrimitiveFunction::AttributeNameSeqGammarLMF] = seqGammarLMF;
        additionalProperties[PrimitiveFunction::AttributeNameSeqGammarBMMIFactor] = seqGammarBMMIFactor;
        additionalProperties[PrimitiveFunction::AttributeNameSeqGammarWordPen] = seqGammarWordPen;
        std::vector<Variable> operands = { labels, prediction, scaledLogLikelihood, lattice };

        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::LatticeSequenceWithSoftmax, operands, std::move(additionalProperties), name), name);
    }

    FunctionPtr ForwardBackward(const Variable& graph, const Variable& features, size_t blankTokenId, int delayConstraint, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameBlankTokenId] = blankTokenId;
        additionalProperties[PrimitiveFunction::AttributeNameDelayConstraint] = delayConstraint;

        return BinaryOp(PrimitiveOpType::ForwardBackward, graph, features, std::move(additionalProperties), name);
    }

    FunctionPtr LabelsToGraph(const Variable& labels, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::LabelsToGraph, labels, Dictionary(), name);
    }

    FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::PastValue, operand, initialState, std::move(additionalProperties), name, false);
    }

    FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameOffset] = DictionaryValue(offset);
        return BinaryOp(PrimitiveOpType::FutureValue, operand, initialState, std::move(additionalProperties), name, false);
    }

    FunctionPtr OneHotOp(const Variable& operand, size_t numClass, bool outputSparse, Axis& axis, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameNumClass] = numClass;
        additionalProperties[PrimitiveFunction::AttributeNameOneHotOutputSparse] = outputSparse;
        additionalProperties[PrimitiveFunction::AttributeNameOneHotAxis] = axis;
        return UnaryOp(PrimitiveOpType::OneHot, operand, std::move(additionalProperties), name);
    }

    FunctionPtr GatherOp(const Variable& indices, const Variable& reference, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Gather, indices, reference, Dictionary(), name);
    }

    FunctionPtr GatherOp(const Variable& indices, const Variable& reference, const Axis& axis, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;

        if (!axis.IsStaticAxis())
            LogicError("Gather operation only supports a single static axis.");

        if (axis.StaticAxisIndex() == -1)
            return BinaryOp(PrimitiveOpType::Gather, indices, reference, std::move(additionalProperties), name);
        else
        {
            auto indPlaceholder = PlaceholderVariable();
            auto refPlaceholder = PlaceholderVariable();
            auto lastAxis = Axis(-1);
            auto swapped = TransposeAxes(refPlaceholder, lastAxis, axis);
            auto gatherSwapped = GatherOp(indPlaceholder, swapped);
            auto result = TransposeAxes(gatherSwapped, lastAxis, axis);
            return AsBlock(std::move(result), { { refPlaceholder, reference },{ indPlaceholder, indices } }, std::move(additionalProperties), L"GatherOp", name);
        }
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

    FunctionPtr ReduceProd(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalProdReductionOpName, axis, name);
    }

    //multiple axes reduction below:

    FunctionPtr ReduceSum(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceLogSum(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalLogSumReductionOpName, axis, name);
    }

    FunctionPtr ReduceMean(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMeanReductionOpName, axis, name);
    }

    FunctionPtr ReduceMax(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMaxReductionOpName, axis, name);
    }

    FunctionPtr ReduceMin(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalMinReductionOpName, axis, name);
    }

    FunctionPtr ReduceProd(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalProdReductionOpName, axis, name);
    }

    FunctionPtr ReduceFunctionAsBlock(const Variable& operand, const std::vector<Axis>& axes, bool keepDims,
        const std::function<FunctionPtr(const Variable&, const std::vector<Axis>& axes)> func,
        const std::wstring opName, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(axes);
        additionalProperties[PrimitiveFunction::AttributeNameReductionKeepDimensions] = keepDims;
        auto operandPlaceholder = PlaceholderVariable(L"operand");
        auto result = func(operandPlaceholder, axes);
        if (!keepDims)
        {
            // Output shape is not available before replacing operandPlaceholder with operand.
            // But we need to know the output shape in order to squeeze it. 
            // Therefore we have to manually calculate the expected shape.
            NDShape expectedShape = operand.Shape();
            for (const Axis& ax : axes)
            {
                auto axis = NormalizeStaticAxis(const_cast<Axis &>(ax), expectedShape.Rank());
                if (!axis.IsStaticAxis())
                    LogicError("ReduceOp: can only reduce on static axes.");
                auto idx = axis.StaticAxisIndex();
                expectedShape[idx] = 1;
            }

            result = Reshape(result, GetSqueezedShape(expectedShape, axes));
        }

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), opName, name);
    }

    FunctionPtr ReduceL1(const Variable& operand, const std::vector<Axis>& axes, bool keepDims, const std::wstring& name)
    {
        auto func = [](const Variable& placeholder, const std::vector<Axis>& axes) { return ReduceSum(Abs(placeholder), axes); };
        auto f = ReduceFunctionAsBlock(operand, axes, keepDims, func, L"ReduceL1", name);
        return f;
    }

    FunctionPtr ReduceL2(const Variable& operand, const std::vector<Axis>& axes, bool keepDims, const std::wstring& name)
    {
        auto func = [](const Variable& placeholder, const std::vector<Axis>& axes) { return Sqrt(ReduceSumSquare(placeholder, axes)); };
        return ReduceFunctionAsBlock(operand, axes, keepDims, func, L"ReduceL2", name);
    }

    FunctionPtr ReduceSumSquare(const Variable& operand, const std::vector<Axis>& axes, bool keepDims, const std::wstring& name)
    {
        auto func = [](const Variable& placeholder, const std::vector<Axis>& axes)
        { return ReduceSum(ElementTimes(placeholder, placeholder), axes); };
        return ReduceFunctionAsBlock(operand, axes, keepDims, func, L"ReduceSumSquare", name);
    }

    FunctionPtr ImageScaler(const Variable& operand, float scale, std::vector<float> biases, const std::wstring& name)
    {
        if (operand.Shape().Rank() != 3)
            LogicError("ImageScaler: incorrect operand shape: %S", operand.Shape().AsString().c_str());

        size_t channels = operand.Shape()[2];
        if (channels != biases.size())
            LogicError("ImageScaler: number of biases (%d) does not equal channels of the image (%d)", (int)biases.size(), (int)(channels));

        auto additionalProperties = Dictionary();
        additionalProperties[L"Scaler"] = scale;
        additionalProperties[L"Biases"] = AsDictionaryValueVector(biases);

        auto operandPlaceholder = PlaceholderVariable();

        Constant constantScalar = Constant::Scalar(operand.GetDataType(), scale);
        FunctionPtr scaledImage = ElementTimes(operandPlaceholder, constantScalar);

        std::vector<Variable> biasConstants;
        for (int i = 0; i < channels; i++)
        {
            Constant constantBias = Constant::Scalar(operand.GetDataType(), biases[i]);
            biasConstants.push_back(constantBias);
        }

        FunctionPtr constantBiases = Splice(biasConstants, Axis(0));
        NDShape shape({ 1, 1, channels });
        FunctionPtr constantBiasesReshaped = Reshape(constantBiases, shape);

        FunctionPtr result = Plus(scaledImage, constantBiasesReshaped, name);

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"ImageScaler", name);
    }

    FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const Variable& mean, const Variable& invStdDev, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable(L"operand");
        auto meanPlaceholder = PlaceholderVariable(L"mean");
        auto invStdDevPlaceholder = PlaceholderVariable(L"invStdDev");
        return AsBlock(std::move(ElementTimes(Minus(operandPlaceholder, meanPlaceholder), invStdDevPlaceholder)), { { operandPlaceholder, operand },{ meanPlaceholder, mean },{ invStdDevPlaceholder, invStdDev } }, L"PerDimMeanVarianceNormalize", name);
    }

    FunctionPtr MeanVarianceNormalization(const Variable& operand, double epsilon, const bool useStatsAcrossChannels, const bool doVarianceScaling, const std::wstring& name)
    {
        Dictionary additionalAttributes;
        additionalAttributes[PrimitiveFunction::AttributeNameUseStatsAcrossChannels] = useStatsAcrossChannels;
        additionalAttributes[PrimitiveFunction::AttributeNameDoVarianceScaling] = doVarianceScaling;
        additionalAttributes[PrimitiveFunction::AttributeNameEpsilon] = epsilon;

        if (epsilon < 0)
            InvalidArgument("Input argument epsilon must be non-negative.");
        auto operandPlaceholder = PlaceholderVariable(L"operand");
        size_t operandRank = operand.Shape().Rank();
        size_t numAxesToReduce;
        if (operandRank < 1)
            InvalidArgument("The rank of the operand must be >= 1.");
        else if (operandRank < 2)
            numAxesToReduce = operandRank; // Operand's a vector, useStatsAcrossChannels is ignored and mean is computed over the vector.
        else
            numAxesToReduce = useStatsAcrossChannels ? operandRank : operandRank - 1; // Assuming last dim to be the channel dim.

        std::vector<Axis> axesToReduce(numAxesToReduce);
        for (size_t i = 0; i < numAxesToReduce; ++i)
            axesToReduce[i] = Axis(i);
        FunctionPtr operandMeanRemoved = Minus(operandPlaceholder, ReduceMean(operandPlaceholder, axesToReduce));
        if (!doVarianceScaling)
        {
            return AsBlock(std::move(operandMeanRemoved), { { operandPlaceholder, operand } }, std::move(additionalAttributes), L"MeanVarianceNormalization", name);
        }
        else
        {
            return AsBlock(std::move(ElementDivide(operandMeanRemoved, 
                Plus(Sqrt(ReduceMean(Square(operandMeanRemoved), axesToReduce)), 
                    Constant({}, operand.GetDataType(), epsilon, DeviceDescriptor::UseDefaultDevice(), L"mvn_epsilon")))),
                { { operandPlaceholder, operand } }, std::move(additionalAttributes),
                L"MeanVarianceNormalization", name);
        }
    }

    FunctionPtr Convolution(const Variable& convolutionMap,
        const Variable& operand,
        const NDShape& strides,
        const std::vector<bool>& sharing,
        const std::vector<bool>& autoPadding,
        const NDShape& dilation,
        size_t reductionRank,
        size_t groups,
        size_t maxTempMemSizeInSamples,
        const std::wstring& name)
    {
        if ((reductionRank != 0) && (reductionRank != 1))
            LogicError("reductionRank: must be 1 or 0.");
        if (groups == 0)
            LogicError("groups: Must be strictly positive, i.e. groups > 0.");

        if (reductionRank == 0)
        {
            if (groups > 1)
                LogicError("groups: groups > 1 is not supported when reductionRank is 0.");
            return Internal::SpatialConvolution(convolutionMap, operand, strides, sharing, autoPadding, dilation,
                maxTempMemSizeInSamples, name);
        }
        else
        {
            return Internal::Convolution(convolutionMap, operand, strides, sharing, autoPadding, dilation, false,
                { 0 }, groups, maxTempMemSizeInSamples, name);
        }
    }

    FunctionPtr ConvolutionTranspose(const Variable& convolutionMap,
        const Variable& operand,
        const NDShape& strides,
        const std::vector<bool>& sharing,
        const std::vector<bool>& autoPadding,
        const NDShape& outputShape,
        const NDShape& dilation,
        size_t reductionRank,
        size_t maxTempMemSizeInSamples,
        const std::wstring& name)
    {
        if ((reductionRank != 0) && (reductionRank != 1))
            LogicError("reductionRank: must be 1 or 0.");

        size_t groups = 1;
        if (reductionRank == 1)
            return Internal::Convolution(convolutionMap,
                operand,
                strides,
                sharing,
                autoPadding,
                dilation,
                true,
                outputShape,
                groups,
                maxTempMemSizeInSamples,
                name);
        else
        {
            auto operandPlaceholder = PlaceholderVariable();
            auto inputRank = static_cast<int>(operand.Shape().Rank());
            auto filterRank = static_cast<int>(convolutionMap.Shape().Rank());
            auto padding = autoPadding;
            auto expandedStrides = strides;

            if (!((filterRank == inputRank) || ((filterRank - inputRank) == 1)))
                LogicError("convolutionMap: Invalid shape, convolutionMap must have the same rank as the input or greater by 1.");

            auto weights = Reshape(convolutionMap, { 1 }, Axis(filterRank), Axis(filterRank));
            if ((filterRank - inputRank) == 1)
                --filterRank;

            if (padding.size() == filterRank)
                padding.push_back(false);

            if (expandedStrides.Rank() == filterRank)
                expandedStrides = expandedStrides.AppendShape({ 1 });

            auto operandReshape = Reshape(operandPlaceholder, { 1 }, Axis(filterRank), Axis(filterRank));
            auto result = Internal::Convolution(weights,
                operandReshape,
                expandedStrides,
                sharing,
                padding,
                dilation,
                true,
                outputShape,
                groups,
                maxTempMemSizeInSamples,
                name);
            return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"ConvolutionTranspose", name);
        }
    }

    FunctionPtr ROIPooling(const Variable& operand,
        const Variable& rois,
        PoolingType poolingType,
        const NDShape& roiOutputShape,
        double spatialScale,
        const std::wstring& name/* = L""*/)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePoolingType] = (size_t)poolingType;
        additionalProperties[PrimitiveFunction::AttributeNameROIOutputShape] = roiOutputShape;
        additionalProperties[PrimitiveFunction::AttributeNameSpatialScale] = spatialScale;
        return BinaryOp(PrimitiveOpType::ROIPooling, operand, rois, std::move(additionalProperties), name);
    }

    FunctionPtr Pooling(const Variable& operand,
        PoolingType poolingType,
        const NDShape& poolingWindowShape,
        const NDShape& strides,
        const std::vector<bool>& autoPadding,
        const bool ceilOutDim,
        const bool includePad,
        const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePoolingType] = (size_t)poolingType;
        additionalProperties[PrimitiveFunction::AttributeNamePoolingWindowShape] = poolingWindowShape;
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = NDShape({0});
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = NDShape({0});
        additionalProperties[PrimitiveFunction::AttributeNameCeilOutDim] = ceilOutDim;
        additionalProperties[PrimitiveFunction::AttributeNameIncludePad] = includePad;

        return UnaryOp(PrimitiveOpType::Pooling, operand, std::move(additionalProperties), name);
    }

    FunctionPtr Unpooling(const Variable& operand,
        const Variable& poolingInput,
        PoolingType unpoolingType,
        const NDShape& poolingWindowShape,
        const NDShape& strides,
        const std::vector<bool>& autoPadding,
        const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNamePoolingType] = (size_t)unpoolingType;
        additionalProperties[PrimitiveFunction::AttributeNameUnpoolingWindowShape] = poolingWindowShape;
        additionalProperties[PrimitiveFunction::AttributeNameStrides] = strides;
        additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
        additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = NDShape({0});
        additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = NDShape({0});

        std::vector<Variable> operands = { operand, poolingInput};
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Unpooling, operands, std::move(additionalProperties), name), name);
    }

    FunctionPtr BatchNormalization(const Variable& operand,
        const Variable& scale,
        const Variable& bias,
        const Variable& runningMean,
        const Variable& runningInvStd,
        const Variable& runningCount,
        bool spatial,
        double normalizationTimeConstant,
        double blendTimeConstant,
        double epsilon,
        bool useCuDNNEngine,
        bool disableRegularization,
        const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameSpatial] = spatial;
        additionalProperties[PrimitiveFunction::AttributeNameNormalizationTimeConstant] = normalizationTimeConstant;
        additionalProperties[PrimitiveFunction::AttributeNameBlendTimeConstant] = blendTimeConstant;
        additionalProperties[PrimitiveFunction::AttributeNameEpsilon] = epsilon;
        additionalProperties[PrimitiveFunction::AttributeNameUseCuDNNEngine] = useCuDNNEngine;
        additionalProperties[PrimitiveFunction::AttributeNameDisableRegularization] = disableRegularization;

        std::vector<Variable> operands = { operand, scale, bias, runningMean, runningInvStd, runningCount };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::BatchNormalization,
            operands,
            std::move(additionalProperties),
            name),
            name);
    }

    FunctionPtr LocalResponseNormalization(const Variable& operand, size_t depthRadius, double bias, double alpha, double beta, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameDepthRadius] = depthRadius;
        additionalProperties[PrimitiveFunction::AttributeNameBias] = bias;
        additionalProperties[PrimitiveFunction::AttributeNameAlpha] = alpha;
        additionalProperties[PrimitiveFunction::AttributeNameBeta] = beta;

        auto operandPlaceholder = PlaceholderVariable();
        auto operandSquare = Square(operandPlaceholder);
        operandSquare = Reshape(operandSquare, { NDShape::InferredDimension, 1 }, Axis(2), Axis(3));
        auto weights = Constant({ 1, 1, 2 * depthRadius + 1, 1 }, operand.GetDataType(), alpha / (2 * depthRadius + 1));
        auto convResult = Convolution(weights, operandSquare);
        convResult = Reshape(convResult, { NDShape::InferredDimension }, Axis(2), Axis(4));
        auto denom = Exp(ElementTimes(Constant::Scalar(operand.GetDataType(), beta), Log(Plus(Constant::Scalar(operand.GetDataType(), bias), convResult))));

        auto result = ElementDivide(operandPlaceholder, denom);
        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"LocalResponseNormalization", name);
    }

    FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name)
    {
        std::vector<Variable> operands = { operand, min, max };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Clip, operands, Dictionary(), name), name);
    }

    FunctionPtr ElementSelect(const Variable& condition, const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name)
    {
        // TODO: If the condition is a scalar constant, we can just pass-through the appropriate operand
        std::vector<Variable> operands = { condition, leftOperand, rightOperand };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Select, operands, Dictionary(), name), name);
    }

    FunctionPtr Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;

        std::vector<Variable> operandsCopy = operands;
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Splice, operandsCopy, std::move(additionalProperties), name), name);
    }

    FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name)
    {
        return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::Combine, operands, Dictionary(), name), name);
    }

    FunctionPtr Mean(const std::vector<Variable>& operands, const std::wstring& name)
    {
        int count = operands.size();
        if (count == 0)
        {
            LogicError("Mean: none operand provided.");
        }

        std::vector<std::pair<Variable, Variable>> argumentsMap;
        auto placeholder = PlaceholderVariable();
        argumentsMap.push_back(std::pair<Variable, Variable>(placeholder, operands[0]));
        FunctionPtr result = placeholder;
        for (int i = 1; i < count; i++)
        {
            placeholder = PlaceholderVariable();
            argumentsMap.push_back(std::pair<Variable, Variable>(placeholder, operands[i]));
            result = Plus(result, placeholder);
        }

        Constant divider = Constant::Scalar(operands[0].GetDataType(), static_cast<double>(operands.size()));
        result = ElementDivide(result, divider);

        return AsBlock(std::move(result), argumentsMap, L"Mean", name);
    }

    FunctionPtr Sum(const std::vector<Variable>& operands, const std::wstring& name)
    {
        int count = operands.size();
        if (count == 0)
        {
            LogicError("Sum: no operand provided.");
        }

        std::vector<std::pair<Variable, Variable>> argumentsMap;
        auto placeholder = PlaceholderVariable();
        argumentsMap.push_back(std::pair<Variable, Variable>(placeholder, operands[0]));
        FunctionPtr result = placeholder;
        for (int i = 1; i < count; i++)
        {
            placeholder = PlaceholderVariable();
            argumentsMap.push_back(std::pair<Variable, Variable>(placeholder, operands[i]));
            result = Plus(result, placeholder);
        }

        return AsBlock(std::move(result), argumentsMap, L"Sum", name);
    }

    FunctionPtr Alias(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::NoOp, operand, Dictionary(), name);
    }

    FunctionPtr AsBlock(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockOpName, const std::wstring& blockName)
    {
        return AsBlock(std::move(composite), argumentsMap, Dictionary(), blockOpName, blockName);
    }

    FunctionPtr AsBlock(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, Dictionary&& attributes, const std::wstring& blockOpName, const std::wstring& blockName)
    {
        if (!composite->IsComposite())
            InvalidArgument("Composite argument '%S' to AsBlock is not a composite Function.", composite->AsString().c_str());

        return AsComposite(MakeSharedObject<BlockFunction>(std::move(composite), argumentsMap, blockOpName, std::move(attributes), blockName), blockName);
    }

    FunctionPtr AsComposite(const FunctionPtr& rootFunction, const std::wstring& name)
    {
        return rootFunction->IsComposite() ? rootFunction : CompositeFunction::Create(rootFunction, name);
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

    FunctionPtr ELU(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::ELU, operand, Dictionary(), name);
    }

    FunctionPtr SELU(const Variable& operand, double gamma, double alpha, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameGamma] = gamma;
        additionalProperties[PrimitiveFunction::AttributeNameAlpha] = alpha;

        auto operandPlaceholder = PlaceholderVariable();
        auto lessThanZero = Less(operandPlaceholder, Constant::Scalar(operand.GetDataType(), 0.0));
        auto result = ElementSelect(lessThanZero,
            ElementTimes(Constant::Scalar(operand.GetDataType(), alpha), ELU(operandPlaceholder)),
            operandPlaceholder);
        result = ElementTimes(Constant::Scalar(operand.GetDataType(), gamma), result);
        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"SELU", name);
    }

    FunctionPtr LeakyReLU(const Variable& operand, double alpha, const std::wstring& name)
    {
        auto additionalProperties = Dictionary();
        additionalProperties[PrimitiveFunction::AttributeNameAlpha] = alpha;

        auto operandPlaceholder = PlaceholderVariable();
        auto lessThanZero = Less(operandPlaceholder, Constant::Scalar(operand.GetDataType(), 0.0));
        auto result = ElementSelect(lessThanZero,
            ElementTimes(Constant::Scalar(operand.GetDataType(), alpha), operandPlaceholder),
            operandPlaceholder);

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, std::move(additionalProperties), L"LeakyReLU", name);
    }

    FunctionPtr PReLU(const Variable& alpha, const Variable& operand, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable();
        auto lessThanZero = Less(operandPlaceholder, Constant::Scalar(operand.GetDataType(), 0.0));
        auto result = ElementSelect(lessThanZero,
            ElementTimes(alpha, operandPlaceholder),
            operandPlaceholder);

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"PReLU", name);
    }

    FunctionPtr Softplus(const Variable& operand, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable();
        auto result = LogAddExp(operandPlaceholder, Constant::Scalar(operand.GetDataType(), 0.0));

        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"Softplus", name);
    }

    FunctionPtr Softsign(const Variable& operand, const std::wstring& name)
    {
        auto operandPlaceholder = PlaceholderVariable();
        auto result = ElementDivide(operandPlaceholder, Plus(Abs(operandPlaceholder), Constant::Scalar(operand.GetDataType(), 1.0)));
        return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"Softsign", name);
    }

    FunctionPtr Argmax(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalArgmaxReductionOpName, axis, name);
    }

    FunctionPtr Argmin(const Variable& operand, const Axis& axis, const std::wstring& name)
    {
        return Internal::ReduceElements(operand, PrimitiveFunction::InternalArgminReductionOpName, axis, name);
    }

    FunctionPtr StopGradient(const Variable& operand, const std::wstring& name)
    {
        return UnaryOp(PrimitiveOpType::StopGradient, operand, Dictionary(), name);
    }

    FunctionPtr Assign(Variable& refOperand, const Variable& operand, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::Assign, refOperand, operand, Dictionary(), name);
    }

    FunctionPtr ReconcileDynamicAxes(const Variable& operand, const Variable& axesAsOperand, const std::wstring& name)
    {
        // TODO: In V1 graph generation, ReconcileDynamicAxis() should be treated like a no-op if the axis is known to be the same.
        //       E.g. used for seq2seq.
        return BinaryOp(PrimitiveOpType::ReconcileDynamicAxis, operand, axesAsOperand, Dictionary(), name, false);
    }

    FunctionPtr ToSequence(const Variable& operand, const std::wstring& sequenceAxisNamePrefix, const std::wstring& name)
    {
        Dictionary additionalAttributes;
        additionalAttributes[PrimitiveFunction::AttributeNameSequenceAxisNamePrefix] = sequenceAxisNamePrefix;
        return UnaryOp(PrimitiveOpType::ToSequence, operand, std::move(additionalAttributes), name);
    }

    FunctionPtr ToSequence(const Variable& operand, const Variable& sequenceLengths, const std::wstring& sequenceAxisNamePrefix, const std::wstring& name)
    {
        Dictionary additionalAttributes;
        additionalAttributes[PrimitiveFunction::AttributeNameSequenceAxisNamePrefix] = sequenceAxisNamePrefix;
        return BinaryOp(PrimitiveOpType::ToSequence, operand, sequenceLengths, std::move(additionalAttributes), name, false);
    }

    const static std::wstring DefaultToSequenceSequenceAxisNamePrefix = L"ToSequence_";
    FunctionPtr ToSequenceLike(const Variable& operand, const Variable& dynamicAxesLike, const std::wstring& name)
    {
        return BinaryOp(PrimitiveOpType::ToSequenceLike, operand, dynamicAxesLike, Dictionary(), name, false);
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
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            return AsBlock(std::move(Internal::IsWithin(operandPlaceholder, 1)), { { operandPlaceholder, operand} }, L"Sequence::IsFirst", name);
        }

        FunctionPtr IsLast(const Variable& operand, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            return AsBlock(std::move(Internal::IsWithin(operandPlaceholder, -1)), { { operandPlaceholder, operand } }, L"Sequence::IsLast", name);
        }

        FunctionPtr Slice(const Variable& operand, int beginIndex, int endIndex, const std::wstring& name)
        {
            VerifyIsSequence(operand);

            if ((beginIndex == 0) && (endIndex == 0))
                return operand;

            auto operandPlaceholder = PlaceholderVariable(L"operand");

            auto beginFlagsLambda = [beginIndex, operandPlaceholder]() {
                return (beginIndex > 0) ? Minus(Constant::Scalar(1.0f), Internal::IsWithin(operandPlaceholder, beginIndex)) : Internal::IsWithin(operandPlaceholder, beginIndex);
            };

            auto endFlagsLambda = [endIndex, operandPlaceholder]() {
                return (endIndex > 0) ? Internal::IsWithin(operandPlaceholder, endIndex) : Minus(Constant::Scalar(1.0f), Internal::IsWithin(operandPlaceholder, endIndex));
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

            auto sliceComposite = Internal::Gather(operandPlaceholder, flags, std::make_pair(multiplicativeFactor, sliceLength));
            return AsBlock(std::move(sliceComposite), { { operandPlaceholder, operand } }, L"Sequence::Slice", name);
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
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto conditionPlaceholder = PlaceholderVariable(L"condition");
            return AsBlock(std::move(Internal::Gather(operandPlaceholder, conditionPlaceholder)), { { operandPlaceholder, operand }, { conditionPlaceholder, condition } }, L"Sequence::Gather", name);
        }

        FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto conditionPlaceholder = PlaceholderVariable(L"condition");
            return AsBlock(std::move(Internal::Gather(operandPlaceholder, conditionPlaceholder, newDerivedSequenceAxisScalingAndAdditiveFactor)), { { operandPlaceholder, operand },{ conditionPlaceholder, condition } }, L"Sequence::Gather", name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto conditionPlaceholder = PlaceholderVariable(L"condition");
            return AsBlock(std::move(Internal::Scatter(operandPlaceholder, conditionPlaceholder)), { { operandPlaceholder, operand }, { conditionPlaceholder, condition } }, L"Sequence::Scatter", name);
        }

        FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto conditionPlaceholder = PlaceholderVariable(L"condition");
            return AsBlock(std::move(Internal::Scatter(operandPlaceholder, conditionPlaceholder, newDerivedSequenceAxisScalingAndAdditiveFactor)), { { operandPlaceholder, operand },{ conditionPlaceholder, condition } }, L"Sequence::Scatter", name);
        }

        FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto broadcastAsPlaceholder = PlaceholderVariable(L"broadcastAs");
            return AsBlock(ReconcileDynamicAxes(operandPlaceholder, broadcastAsPlaceholder), { { operandPlaceholder, operand }, { broadcastAsPlaceholder, broadcastAs } }, L"Sequence::BroadcastAs", name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, bool keepReducedDimensions, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto unpackedSequence = Unpack(operandPlaceholder, ReductionIdentityValue(reductionOpName), /*suppressMaskOutput =*/ true, name);
            auto reductionResult = Internal::ReduceElements(unpackedSequence, reductionOpName, Axis(-1), keepReducedDimensions);

            return AsBlock(std::move(reductionResult), { { operandPlaceholder, operand } }, L"Sequence::ReduceElements", name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::wstring& name)
        {
            return  ReduceElements(operand, reductionOpName, /*keepReducedDimensions =*/ false, name);
        }

        FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name)
        {
            return ReduceElements(operand, PrimitiveFunction::InternalSumReductionOpName, name);
        }

        FunctionPtr ReduceMax(const Variable& operand, const std::wstring& name)
        {
            return ReduceElements(operand, PrimitiveFunction::InternalMaxReductionOpName, name);
        }

        FunctionPtr Softmax(const Variable& operand, const std::wstring& name)
        {
            auto operandPlaceholder = PlaceholderVariable(L"operand");
            auto operandDelta = operandPlaceholder - BroadcastAs(ReduceMax(operandPlaceholder, L""), operandPlaceholder);
            auto expOperandDelta = Exp(operandDelta);
            auto denominator = BroadcastAs(ReduceSum(expOperandDelta), operandPlaceholder);
            return AsBlock(ElementDivide(expOperandDelta, denominator), { { operandPlaceholder, operand } }, L"Sequence::Softmax", name);
        }

        FunctionPtr Unpack(const Variable& operand, double paddingValue, bool supressMaskOutput, const std::wstring& name)
        {
            Dictionary additionalAttributes;
            additionalAttributes[PrimitiveFunction::AttributeNameSequenceUnpackPaddingValue] = paddingValue;
            additionalAttributes[PrimitiveFunction::AttributeNameSequenceUnpackSuppressMaskOutput] = supressMaskOutput;
            return UnaryOp(PrimitiveOpType::UnpackSequence, operand, std::move(additionalAttributes), name);
        }
    }

    // Creates an instance of crop node with explicitly specified crop offsets.
    // nodeInput: input node to be cropped.
    // nodeReferent: input node which determines the spatial size of output.
    // offsetX, offsetY: offset values in pixel which determine the position of crop rectangle.
    FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, size_t offsetX, size_t offsetY, const std::wstring& name)
    {
        std::vector<Variable> operands = { nodeInput, nodeReferent };
        Dictionary additionalAttributes;
        additionalAttributes[PrimitiveFunction::AttributeNameOffset] = DictionaryValue({ offsetX, offsetY });
        return AsComposite(MakeSharedObject<PrimitiveFunction>(
            PrimitiveOpType::Crop,
            operands, std::move(additionalAttributes), name), name);
    }

    // Creates an instance of crop node with automatically computed crop offsets.
    // nodeInput: input node to be cropped.
    // nodeReferent: input node which determines the spatial size of output.
    FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, const std::wstring& name)
    {
        std::vector<Variable> operands = { nodeInput, nodeReferent };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(
            PrimitiveOpType::Crop,
            operands, Dictionary(), name), name);
    }

    // Creates an instance of crop node with automatically computed crop offsets and specified ancestor nodes.
    // This is used in cases when input nodes do not have common ancestor in the network.
    // nodeInput: input node to be cropped.
    // nodeReferent: input node which determines the spatial size of output.
    // ancestorInput: ancestor of nodeInput.
    // ancestorReferent: ancestor of nodeReferent which is treated as equal to ancestorInput for the purpose of computing crop offsets.
    FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, const Variable& ancestorInput, const Variable& ancestorReferent, const std::wstring& name)
    {
        std::vector<Variable> operands = { nodeInput, nodeReferent, ancestorInput, ancestorReferent };
        return AsComposite(MakeSharedObject<PrimitiveFunction>(
            PrimitiveOpType::Crop,
            operands, Dictionary(), name), name);
    }

    FunctionPtr Cast(const Variable& nodeInput, DataType outputType, const std::wstring& name)
    {
        std::vector<Variable> operands = { nodeInput };
        Dictionary additionalAttributes;
        additionalAttributes.Add(
            PrimitiveFunction::AttributeNameNewDataType, static_cast<int>(outputType));
        return AsComposite(MakeSharedObject<PrimitiveFunction>(
            PrimitiveOpType::Cast,
            operands, std::move(additionalAttributes), name), name);
    }

    namespace Internal
    {
        FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name)
        {
            Sequence::VerifyIsSequence(operand);

            if (offset == 0)
                InvalidArgument("Sequence::IsWithin: The offset cannot be 0.");

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
            return AsComposite(MakeSharedObject<PrimitiveFunction>(PrimitiveOpType::ScatterPacked, operands, Dictionary(), name), name);
        }

        FunctionPtr ZeroesWithDynamicAxesLike(const Variable& operand)
        {
            return ReconcileDynamicAxes(Constant::Scalar(0.0f), operand/*acts as layout input*/);
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

        FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::vector<int>& strides, const std::wstring& name)
        {
            auto additionalProperties = Dictionary();

            assert(axis.size() > 0 && axis.size() == beginIndex.size() && axis.size() == endIndex.size() && strides.size() == axis.size());
            if (axis.size() == 1)
            {
                additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis[0];
                additionalProperties[PrimitiveFunction::AttributeNameBeginIndex] = beginIndex[0];
                additionalProperties[PrimitiveFunction::AttributeNameEndIndex] = endIndex[0];
                additionalProperties[PrimitiveFunction::AttributeNameSliceStrides] = strides[0];
            }
            else
            {
                additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(axis);
                additionalProperties[PrimitiveFunction::AttributeNameBeginIndexVec] = AsDictionaryValueVector(beginIndex);
                additionalProperties[PrimitiveFunction::AttributeNameEndIndexVec] = AsDictionaryValueVector(endIndex);
                additionalProperties[PrimitiveFunction::AttributeNameSliceStridesVec] = AsDictionaryValueVector(strides);
            }
            return UnaryOp(PrimitiveOpType::Slice, operand, std::move(additionalProperties), name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, bool keepReducedDimensions, const std::wstring& name)
        {
            if (axis.IsStaticAxis() ||
                (axis == Axis::AllStaticAxes()) ||
                (axis == Axis::AllAxes()) ||
                (axis == Axis::DefaultBatchAxis()) ||
                ((reductionOpName == PrimitiveFunction::InternalSumReductionOpName) && (axis == Axis::OperandSequenceAxis())))
            {
                auto additionalProperties = Dictionary();
                additionalProperties[PrimitiveFunction::AttributeNameAxis] = axis;
                additionalProperties[PrimitiveFunction::AttributeNameReductionOpName] = reductionOpName;
                additionalProperties[PrimitiveFunction::AttributeNameReductionKeepDimensions] = keepReducedDimensions;
                return UnaryOp(PrimitiveOpType::ReduceElements, operand, std::move(additionalProperties), name);
            }

            LogicError("ReduceElements: operand %S; Invalid axis argument provided. To reduce an operand along its ordered dynamic axis use Sequence::ReduceElements.",
                operand.AsString().c_str());
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name)
        {
            bool keepReducedDimensions = true;
            if (axis == Axis::AllStaticAxes() || axis == Axis::AllAxes())
                keepReducedDimensions = false;

            return ReduceElements(operand, reductionOpName, axis, keepReducedDimensions, name);
        }

        FunctionPtr ComposeReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, bool keepReducedDimensions, const std::wstring& name)
        {
            if (
                std::any_of(axes.begin(), axes.end(),
                    [](const Axis& axis) {
                return axis.IsStaticAxis() ||
                    (axis == Axis::AllStaticAxes()) ||
                    (axis == Axis::AllAxes()) ||
                    (axis == Axis::DefaultBatchAxis()); })
                || ((reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                    && std::any_of(axes.begin(), axes.end(),
                        [](const Axis& axis) {return axis == Axis::OperandSequenceAxis(); }))
                    )
            {
                auto additionalProperties = Dictionary();
                additionalProperties[PrimitiveFunction::AttributeNameAxisVec] = AsDictionaryValueVector(axes);
                additionalProperties[PrimitiveFunction::AttributeNameReductionOpName] = reductionOpName;
                additionalProperties[PrimitiveFunction::AttributeNameReductionKeepDimensions] = keepReducedDimensions;
                return UnaryOp(PrimitiveOpType::ReduceElements, operand, std::move(additionalProperties), name);
            }

                LogicError("ReduceElements: operand %S; Invalid axis argument provided. To reduce an operand along its ordered dynamic axis use Sequence::ReduceElements.",
                    operand.AsString().c_str());
        }

        ///
        /// Compose reduction operation along multiple axes.
        /// Compose reduction along multiple axes to perform reduction over static axes first, then sequence axis and then batch axis.
        ///
        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, bool keepReducedDimensions, const std::wstring& name)
        {
            //if axes is empty, raise error:
            if (axes.empty())
                LogicError("ReduceElements: operand %S; Empty axes argument provided.", operand.AsString().c_str());

            //if reduce over all axes, we directly compose the reduction node:
            for (auto &axis : axes)
                if (axis == Axis::AllAxes() || axis == Axis::AllStaticAxes())
                    return  ComposeReduceElements(operand, reductionOpName, axes, keepReducedDimensions, name);

            std::vector<Axis> static_axes;
            std::vector<Axis> sequence_axes;
            std::vector<Axis> batch_axes;
            bool static_axes_reduced = false;
            for (auto& axis : axes)
            {
                if (axis.IsBatchAxis())
                {
                    batch_axes.push_back(axis);
                }
                else if (axis.IsSequenceAxis())
                {
                    sequence_axes.push_back(axis);
                }
                else if (axis.IsStaticAxis() || (axis == Axis::AllStaticAxes()))
                {
                    if (axis == Axis::AllStaticAxes())
                        static_axes_reduced = true;
                    if (static_axes_reduced && static_axes.size() > 1)
                        LogicError("ReduceElements: operand %S; additional static axes are provided when doing reduction over AllStaticAxes.", operand.AsString().c_str());
                    static_axes.push_back(axis);
                }
            }
            auto operand_placeholder = PlaceholderVariable(L"reduction_operand");

            FunctionPtr res = operand_placeholder;
            if (!static_axes.empty())
            {
                res = ComposeReduceElements(res, reductionOpName, static_axes, keepReducedDimensions, name + L"_static_axes_subop");
            }
            if (!sequence_axes.empty())
            {
                res = CNTK::Sequence::ReduceElements(res, reductionOpName, keepReducedDimensions, name + L"_sequence_axes_subop");
            }
            if (!batch_axes.empty())
            {
                res = ComposeReduceElements(res, reductionOpName, batch_axes, keepReducedDimensions, name + L"_batch_axes_subop");
            }
            return AsBlock(std::move(res), { { operand_placeholder, operand }}, L"MultiAxisReduce", name);
        }

        FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, const std::wstring& name)
        {
            bool keepReducedDimensions = true;
            if (std::any_of(axes.begin(), axes.end(),
                [](const Axis& axis) {return axis == Axis::AllStaticAxes() || axis == Axis::AllAxes(); }))
                keepReducedDimensions = false;

            return ReduceElements(operand, reductionOpName, axes, keepReducedDimensions, name);
        }

        FunctionPtr Convolution(const Variable& convolutionMap,
            const Variable& operand,
            const NDShape& strides,
            const std::vector<bool>& sharing,
            const std::vector<bool>& autoPadding,
            const NDShape& dilation,
            bool transpose,
            const NDShape& outputShape,
            size_t groups,
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
            additionalProperties[PrimitiveFunction::AttributeNameDilation] = dilation;
            additionalProperties[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
            additionalProperties[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
            additionalProperties[PrimitiveFunction::AttributeNameLowerPad] = NDShape({0});
            additionalProperties[PrimitiveFunction::AttributeNameUpperPad] = NDShape({0});
            additionalProperties[PrimitiveFunction::AttributeNameTranspose] = transpose;
            additionalProperties[PrimitiveFunction::AttributeNameOutputShape] = outputShape;
            additionalProperties[PrimitiveFunction::AttributeNameKernelShape] = NDShape({0});
            additionalProperties[PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples] = maxTempMemSizeInSamples;
            additionalProperties[PrimitiveFunction::AttributeNameGroups] = groups;

            return BinaryOp(PrimitiveOpType::Convolution, convolutionMap, operand, std::move(additionalProperties), name);
        }

        FunctionPtr SpatialConvolution(const Variable& convolutionMap,
            const Variable& operand,
            const NDShape& strides,
            const std::vector<bool>& sharing,
            const std::vector<bool>& autoPadding,
            const NDShape& dilation,
            size_t maxTempMemSizeInSamples,
            const std::wstring& name)
        {
            // reductionRank is assumed 0.
            auto operandPlaceholder = PlaceholderVariable();
            auto inputRank = static_cast<int>(operand.Shape().Rank());
            auto filterRank = static_cast<int>(convolutionMap.Shape().Rank());
            auto padding = autoPadding;
            auto expandedStrides = strides;

            if ((filterRank - inputRank) == 1)
                --filterRank;
            else if (filterRank != inputRank)
                LogicError("convolutionMap: Invalid shape, convolutionMap must have the same rank as the input or greater by 1.");

            if (padding.size() == filterRank)
                padding.push_back(false);

            if (expandedStrides.Rank() == filterRank)
                expandedStrides = expandedStrides.AppendShape({ 1 });

            auto weights = Reshape(convolutionMap, { 1 }, Axis(filterRank), Axis(filterRank));
            auto operandReshape = Reshape(operandPlaceholder, { 1 }, Axis(filterRank), Axis(filterRank));
            auto result = Internal::Convolution(weights,
                operandReshape,
                expandedStrides,
                sharing,
                padding,
                dilation,
                false,
                { 0 },
                PrimitiveFunction::convolutionOpDefaultValueForGroups,
                maxTempMemSizeInSamples,
                name);
            return AsBlock(std::move(result), { { operandPlaceholder, operand } }, L"Convolution", name);
        }
    }
}
