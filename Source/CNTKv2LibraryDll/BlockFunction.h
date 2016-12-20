//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveFunction.h"

namespace CNTK
{
    class BlockFunction final : public PrimitiveFunction
    {
    public:
        BlockFunction(FunctionPtr&& composite, const std::unordered_map<Variable, Variable>& argumentsMap, const std::wstring& blockOpName, Dictionary&& attributes, const std::wstring& blockName = L"", const std::wstring& uid = GenerateUid(PrimitiveOpType::Block))
            : PrimitiveFunction(DetermineInputs(composite, argumentsMap, blockName), DetermineOutputs(composite, this, blockName), std::move(attributes), blockName, uid),
            m_composite(composite), m_blockOpName(blockOpName), m_compositeArgumentsMap(argumentsMap)
        {
            auto updatedOutputs = GetOutputVariables(true);
            auto currentOutputs = Outputs();
            for (size_t i = 0; i < currentOutputs.size(); ++i)
            {
                auto newOutputVar = updatedOutputs[i];
                auto currentOutputVar = currentOutputs[i];
                Function::ValidateOrUpdateOutput(currentOutputVar, newOutputVar, true);
                currentOutputVar.m_dataFields->m_name = newOutputVar.Name();
            }

            auto compositeOutputs = composite->Outputs();
            for (size_t i = 0; i < currentOutputs.size(); ++i)
                m_compositeOutputsMap.insert({ currentOutputs[i], compositeOutputs[i] });
        }

        virtual const std::wstring& OpName() const override
        {
            return m_blockOpName;
        }

        const FunctionPtr& Composite() const { return m_composite; }
        const std::unordered_map<Variable, Variable>& CompositeArgumentsMap() const { return m_compositeArgumentsMap; }
        const std::unordered_map<Variable, Variable>& CompositeOutputsMap() const { return m_compositeOutputsMap; }

    protected:
        virtual void OnPlaceholdersReplaced(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                            std::unordered_set<Variable>& replacedPlaceholders) override
        {
            // Substitute any placeholder replacements in the arguments map
            for (auto argMapping : m_compositeArgumentsMap)
            {
                if (replacedPlaceholders.find(argMapping.second) != replacedPlaceholders.end())
                    m_compositeArgumentsMap[argMapping.first] = placeholderReplacements.at(argMapping.second);
            }
        }

    private:
        static std::vector<Variable> DetermineInputs(const FunctionPtr& composite, const std::unordered_map<Variable, Variable>& argumentsMap, const std::wstring& blockName)
        {
            std::vector<Variable> blockFunctionInputs;
            auto compositeInputs = composite->Inputs();
            std::vector<Variable> unmappedArguments;
            for (auto compositeInput : compositeInputs)
            {
                assert(!compositeInput.IsOutput());

                if (compositeInput.IsConstant() || compositeInput.IsParameter())
                    blockFunctionInputs.push_back(compositeInput);
                else
                {
                    if (!compositeInput.IsPlaceholder())
                    {
                        InvalidArgument("The composite implementing block (%S) has an argument (%S) which is not a placeholder. "
                            "All arguments of the composite underlying a block must be placeholders",
                            blockName.c_str(), compositeInput.Name().c_str());
                    }

                    // Verify that a mapping was provided for each argument of the composite
                    if (argumentsMap.find(compositeInput) == argumentsMap.end())
                        unmappedArguments.push_back(compositeInput);
                }
            }

            if (!unmappedArguments.empty())
            {
                auto unmappedArgumentsNames = NamedListString(unmappedArguments);
                InvalidArgument("%d arguments (%S) of the underlying composite Function of block (%S) have not been mapped when encapsulating the composite as a block", (int)unmappedArguments.size(), unmappedArgumentsNames.c_str(), blockName.c_str());
            }

            // We now append the mapped arguments of the composite to the block inputs in the order of the map
            // instead of the original order they appear in the composite itself
            for (auto argumentMapping : argumentsMap)
                blockFunctionInputs.push_back(argumentMapping.second);

            return blockFunctionInputs;
        }

        virtual std::vector<Variable> GetOutputVariables(bool inferDimensions) override
        {
            // We determine the outputs by replacing the arguments of the composite with new placeholders with updated 
            // shape etc. information matching the corresponding mapped input
            std::unordered_map<Variable, Variable> newArgumentsMap;
            std::unordered_map<Variable, Variable> replacementMap;
            for (auto argMapping : m_compositeArgumentsMap)
            {
                auto newArgument = PlaceholderVariable(argMapping.second.Shape(), argMapping.second.GetDataType(), argMapping.second.Name(), argMapping.second.DynamicAxes());
                newArgumentsMap.insert({ newArgument, argMapping.second });
                replacementMap.insert({ argMapping.first, newArgument });
            }

            m_composite->ReplacePlaceholders(replacementMap);
            m_compositeArgumentsMap = std::move(newArgumentsMap);

            std::vector<Variable> blockFunctionOutputs;
            auto compositeOutputs = m_composite->Outputs();
            for (auto compositeOutput : compositeOutputs)
                blockFunctionOutputs.push_back(OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), this, compositeOutput.DynamicAxes(), Name()));

            return blockFunctionOutputs;
        }

        static std::vector<Variable> DetermineOutputs(const FunctionPtr& composite, Function* owner, const std::wstring& blockName)
        {
            std::vector<Variable> blockFunctionOutputs;
            auto compositeOutputs = composite->Outputs();
            for (auto compositeOutput : compositeOutputs)
                blockFunctionOutputs.push_back(OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), owner, compositeOutput.DynamicAxes(), blockName));

            return blockFunctionOutputs;
        }

    private:
        FunctionPtr m_composite;
        std::wstring m_blockOpName;

        // Mapping from each argument of the composite underlying the block
        // to the corresponding Variable it is mapped to
        std::unordered_map<Variable, Variable> m_compositeArgumentsMap;

        // Mapping from each output of the block to the corresponding
        // output of underlying composite
        std::unordered_map<Variable, Variable> m_compositeOutputsMap;

        // Increasing s_serializationVersion every time we add more ops allows us to print 
        // a more meaningful message when trying to load a new model with a stale binary. 
        static const size_t s_serializationVersion = 1;
    };
}
