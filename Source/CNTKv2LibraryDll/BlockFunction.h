//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveFunction.h"
#include "Utils.h"
#include "Variable.h"

namespace CNTK
{
    class BlockFunction final : public PrimitiveFunction
    {
    public:
        BlockFunction(FunctionPtr&& composite,
                      const std::vector<std::pair<Variable, Variable>>& argumentsMap, // [composite's Placeholder] -> actual input it should pretend to be
                      const std::wstring& blockOpName, Dictionary&& attributes,
                      const std::wstring& blockName = std::wstring(), const std::wstring& uid = GenerateUid(PrimitiveOpType::Block))
            : PrimitiveFunction(PrimitiveOpType::Block, DetermineInputs(composite, argumentsMap, blockName), std::move(attributes), blockName, uid),
            m_composite(composite), m_blockOpName(blockOpName)
        {
        }

        // special version for InvokeGraph(). Defined in AutoBatch.cpp for now.
        BlockFunction(const std::shared_ptr<CompositeFunction>& composite, std::vector<Variable>& argumentList, std::vector<Variable>&& operands, bool isBasicBlock, bool determineShapes, std::wstring&& blockName);
        Variable FinalizeInvoke(const std::vector<Variable>& argumentList, bool shapeIsKnown);

        // special short-circuited constructor private to auto-batcher
        // This must not be used for anything else.
        BlockFunction(const FunctionPtr& composite, std::vector<Variable>&& inputs, bool isBasicBlock, std::wstring&& blockOpName, std::wstring&& name) :
            PrimitiveFunction(PrimitiveOpType::Block, std::move(inputs), Dictionary(), std::move(name)),
            m_composite(composite), m_blockOpName(move(blockOpName)), m_compositeIsShared(true), m_isBasicBlock(isBasicBlock)
        {
        }

        bool IsBasicBlock() const
        {
            if (!m_compositeIsShared)
                LogicError("IsBasicBlock can only be called on a block created by Invoke()");
            return m_isBasicBlock;
        }

        virtual const std::wstring& OpName() const override { return m_blockOpName; }

        const FunctionPtr& Composite() const { return m_composite; }

        // Mapping from each argument of the composite underlying the block to the corresponding Variable it is mapped to
        std::vector<std::pair<Variable, Variable>> CompositeArgumentsMap() const
        {
            std::vector<std::pair<Variable, Variable>> argumentsMap;
            auto arguments = m_composite->Arguments();
            for (auto argument : arguments)
            {
                //if (BlockFunctionPlaceholderMapping(argument) == Variable())
                //    LogicError("BlockFunction '%S' with OpName '%S' does not have a mapping for argument '%S'.", AsString().c_str(), OpName().c_str(), argument.AsString().c_str());

                argumentsMap.push_back({ argument, BlockFunctionPlaceholderMapping(argument) });
            }

            // Now sort the mapping by the order of occurence of the argument mapping in the block's inputs
            auto blockInputs = Inputs();
            std::unordered_map<Variable, size_t> inputIndices;
            for (size_t i = 0; i < blockInputs.size(); ++i)
                inputIndices.insert({ blockInputs[i], i });

            std::stable_sort(argumentsMap.begin(), argumentsMap.end(), [&inputIndices](const std::pair<Variable, Variable>& first, const std::pair<Variable, Variable>& second) {
                return inputIndices.at(first.second) < inputIndices.at(second.second);
            });

            return argumentsMap;
        }

        // Mapping from each output of the block to the corresponding  output of underlying composite
        std::unordered_map<Variable, Variable> CompositeOutputsMap() const
        {
            std::unordered_map<Variable, Variable> outputsMap;
            const auto& outputs = RawOutputs();
            for (auto output : outputs)
            {
                //if (BlockFunctionOutputMapping(output) == Variable())
                //    LogicError("BlockFunction '%S' with OpName '%S' does not have a mapping for output '%S'", AsString().c_str(), OpName().c_str(), output.AsString().c_str());

                outputsMap[output] = BlockFunctionOutputMapping(output);
            }

            return outputsMap;
        }

        // determine for a Placeholder in m_composite which actual value (in m_inputs) it should pretend to be
        // Will fail if no mapping has been set up.
        const Variable& BlockFunctionPlaceholderMapping(const /*Placeholder*/Variable& argument) const
        {
            if (!argument.IsPlaceholder())
                LogicError("GetPlaceholderMapping can only be used for Placeholders.");
            if (!m_compositeIsShared) // not shared: pretend value is stored in m_blockFunctionVariableMapping
            {
                if (argument.m_dataFields->m_compositeArgumentIndex != VariableFields::m_compositeArgumentIndexUndefined)
                    LogicError("m_compositeArgumentIndex should not be used when !m_compositeIsShared");
                if (!argument.m_dataFields->More().m_blockFunctionVariableMapping.m_dataFields)
                    LogicError("BlockFunction '%S' with OpName '%S' does not have a mapping for argument '%S'.", AsString().c_str(), OpName().c_str(), argument.AsString().c_str());
                return argument.m_dataFields->More().m_blockFunctionVariableMapping;
            }
            else // shared composite: pretend value is found in block's m_inputs, at index m_compositeArgumentIndex
            {
                if (argument.m_dataFields->HasMore() && argument.m_dataFields->m_more->m_blockFunctionVariableMapping.m_dataFields)
                    LogicError("m_blockFunctionVariableMapping should not be set when m_compositeIsShared");
                if (argument.m_dataFields->m_compositeArgumentIndex == VariableFields::m_compositeArgumentIndexUndefined)
                    LogicError("BlockFunction '%S' with OpName '%S' does not have a mapping for shared-composite argument '%S'.", AsString().c_str(), OpName().c_str(), argument.AsString().c_str());
                if (argument.m_dataFields->m_compositeArgumentIndex >= m_inputs.size())
                    LogicError("m_compositeArgumentIndex out of bounds??");
                return m_inputs[argument.m_dataFields->m_compositeArgumentIndex];
            }
        }

        // determine for an Output in this->m_outputs which outputs of m_composite it should pretend to be
        // Will fail if no mapping has been set up.
        const Variable& BlockFunctionOutputMapping(const /*Output*/Variable& output) const
        {
            if (!output.IsOutput())
                LogicError("BlockFunctionOutputMapping: Must only be called on OutputVariables");
            if (BlockFunctionOutputMapping(output) == Variable())
                LogicError("BlockFunction '%S' with OpName '%S' does not have a mapping for output '%S'", AsString().c_str(), OpName().c_str(), output.AsString().c_str());
            return output.m_dataFields->More().m_blockFunctionVariableMapping;
        }

    protected:
        virtual void OnPlaceholdersReplaced(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                            std::unordered_set<Variable>& replacedPlaceholders) override
        {
            // Substitute any placeholder replacements in the arguments map
            auto arguments = m_composite->Arguments();
            std::unordered_map<Variable, Variable> blockCompositePlaceholderReplacements;
            for (auto argument : arguments)
            {
                // check whether the Variable that this composite Placeholder pretends to be has been replaced
                // This is a change of the block's m_inputs.
                if (replacedPlaceholders.find(BlockFunctionPlaceholderMapping(argument)) != replacedPlaceholders.end())
                {
                    auto replacement = placeholderReplacements.at(BlockFunctionPlaceholderMapping(argument));
                    if (IsArgument(replacement))
                    {
                        // It has changed, and not to a Constant or Parameter.
                        if (!m_compositeIsShared) // (if shared then we only store an index, which does not change)
                            argument.m_dataFields->More().m_blockFunctionVariableMapping = replacement;
                        if (BlockFunctionPlaceholderMapping(argument) == replacement)
                            LogicError("BlockFunction::OnPlaceholdersReplaced inputs incorrectly updated upon replacing placeholders");
                    }
                    else
                        blockCompositePlaceholderReplacements.insert({ argument,  replacement });
                }
            }

            m_composite->ReplacePlaceholders(blockCompositePlaceholderReplacements);
        }

    private:
        /*static*/ std::vector<Variable> DetermineInputs(const FunctionPtr& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockName) const
        {
            // The m_inputs of a BlockFunction are...
            std::unordered_map<Variable, Variable> argumentsMappingAsMap; // [composite's Placeholder] -> actual input it should pretend to be
            for (auto argumentMapping : argumentsMap)
            {
                auto wasInserted = argumentsMappingAsMap.insert(argumentMapping).second;
                if (!wasInserted)
                    InvalidArgument("Multiple mappings provided for argument '%S' of the Block composite '%S'", argumentMapping.first.AsString().c_str(), composite->AsString().c_str());
            }

            std::vector<Variable> blockFunctionInputs;  // the return value of this function is built here
            auto compositeInputs = composite->Inputs(); // (this is an expensive operation for composites, including a full traversal and a copy+shared_ptr of the inputs array)
            std::vector<Variable> unmappedArguments;
            // compositeInputs includes all leaves. That is, it includes both Placeholders and enclosed Parameters/Constants.
            // We include Parameters/Constants in the inputs, so that all Parameters/Constants can be found when traversing
            // a graph, without having to step inside BlockFunctions' composites.
            for (auto compositeInput : compositeInputs)
            {
                assert(!compositeInput.IsOutput());

                if (compositeInput.IsConstant() || compositeInput.IsParameter())
                    blockFunctionInputs.push_back(compositeInput);
                else
                {
                    if (!compositeInput.IsPlaceholder())
                    {
                        InvalidArgument("The composite implementing Block '%S' has an argument '%S' which is not a placeholder. "
                                        "All arguments of the composite underlying a Block must be placeholders",
                                        blockName.c_str(), compositeInput.AsString().c_str());
                    }

                    // Verify that a mapping was provided for each Placeholder in the composite
                    if (argumentsMappingAsMap.find(compositeInput) == argumentsMappingAsMap.end())
                        unmappedArguments.push_back(compositeInput);
                }
            }

            if (!unmappedArguments.empty())
            {
                InvalidArgument("%zu of the Placeholders '%S' of the underlying composite Function of Block '%S' have not been mapped when encapsulating the composite as a Block.",
                                unmappedArguments.size(), NamedListString(unmappedArguments).c_str(), blockName.c_str());
            }

            // We now append the mapped arguments of the composite to the block inputs in the order of the map
            // instead of the original order they appear in the composite itself
            for (auto argumentMapping : argumentsMap)
            {
                if (!m_compositeIsShared)
                    argumentMapping.first.m_dataFields->More().m_blockFunctionVariableMapping = argumentMapping.second; // composite Placeholder remembers its actual input
                else
                    LogicError("DetermineInputs is not supposed to be called when composite is not shared. That'd be a different constructor.");
                    //argumentMapping.first.m_dataFields->m_compositeArgumentIndex = blockFunctionInputs.size(); // for shared composite (Dynamite), we remember the index instead
                blockFunctionInputs.push_back(argumentMapping.second);
            }

            return blockFunctionInputs; // this goes into m_inputs
        }

        OutputsVectorType InferOutputs() override
        {
            // Note: This is not used for the case of static-block invocation in Dynamite.
            std::vector<Variable> outputs;
            // We determine the outputs by replacing the arguments of the composite with new placeholders with updated 
            // shape etc. information matching the corresponding mapped input
            auto currentArguments = m_composite->Arguments(); // (this is an expensive operation, requiring a full traversal and a full copy+shared_ptr of the inputs array)
            std::unordered_map<Variable, Variable> replacementMap;
            for (auto currentArgument : currentArguments) // note: it is ensured that currentArguments only includes Placeholders (no Inputs or Outputs)
            {
                auto currentArgumentMapping = BlockFunctionPlaceholderMapping(currentArgument); // this was remembered in the constructor
                auto newArgument = PlaceholderLike(currentArgumentMapping);
                newArgument.m_dataFields->More().m_blockFunctionVariableMapping = currentArgument.m_dataFields->More().m_blockFunctionVariableMapping; // == currentArgumentMapping or, if shared composite, null
                newArgument.m_dataFields->m_compositeArgumentIndex = currentArgument.m_dataFields->m_compositeArgumentIndex;

                replacementMap.insert({ currentArgument, newArgument });
            }

            m_composite->ReplacePlaceholders(replacementMap);

            assert(outputs.empty());
            const auto& compositeOutputs = m_composite->RawOutputs();
            for (const auto& compositeOutput : compositeOutputs)
            {
                auto output = OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), compositeOutput.DynamicAxes(), compositeOutput.NeedsGradient(), Name());
                output.m_dataFields->More().m_blockFunctionVariableMapping = compositeOutput;

                outputs.push_back(output);
            }

            return OutputsVectorType(std::move(outputs)); // note: not super-efficient, but only used for static composites, so it's OK for now
        }

    private:
        FunctionPtr m_composite;
        std::wstring m_blockOpName;

        // Dynamite:
        // In BlockFunctions created via Invoke(), the composite is shared across multiple invocations.
        // Therefore we cannot use Placeholder::m_blockFunctionVariableMapping to store the redirect
        // to the actual argument to be used in place of the Placeholder.
        // Instead, we use Placeholder::m_compositeArgumentIndex. The following conceptual equivalence
        // should hold: plVar->More().m_blockFunctionVariableMapping === m_inputs[plVar->m_compositeArgumentIndex].
        // TODO: Can we switch BlockFunction to this at large?
        bool m_compositeIsShared = false; // true for Dynamite
        bool m_isBasicBlock = false;      // if true then keep as primitive operation for all batching decisions

        // Increasing s_serializationVersion every time we add more ops allows us to print 
        // a more meaningful message when trying to load a new model with a stale binary. 
        static const size_t s_serializationVersion = 1;
    };
}
