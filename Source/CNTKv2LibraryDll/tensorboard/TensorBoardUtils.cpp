//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "tensorboard/TensorBoardUtils.h"

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "tensorboard/tensorboard.pb.h"
#pragma warning(pop)

#include "Utils.h"

namespace CNTK
{
    namespace Internal
    {
        typedef google::protobuf::Map<std::string, tensorflow::AttrValue> AttrMap;
        typedef std::pair<Variable, std::wstring> VariableWithScope;

        template<typename Ptr>
        static const std::wstring& GetName(const Ptr& obj)
        {
            return (obj->Name().empty()) ? obj->Uid() : obj->Name();
        }

        template<typename Ptr>
        static std::wstring GetScopedName(const std::wstring& scope, const Ptr& obj)
        {
            return (scope.empty()) ? GetName(obj) : scope + L"/" + GetName(obj);
        }

        static tensorflow::DataType GetDataType(DataType dataType)
        {
            switch (dataType)
            {
            case DataType::Float:
                return tensorflow::DT_FLOAT;
            case DataType::Double:
                return tensorflow::DT_DOUBLE;
            default:
                NOT_IMPLEMENTED;
            }
        }

        static void PopulateShape(const NDShape& src, tensorflow::TensorShapeProto& dst)
        {
            for (auto dimension : src.Dimensions())
            {
                dst.add_dim()->set_size(dimension);
            }
        }

        static void PopulateShapeAttr(const NDShape& src, AttrMap& dst)
        {
            auto& attr = dst["shape"];
            PopulateShape(src, *attr.mutable_shape());
        }

        static void PopulateDataTypeAttr(const DataType& src, AttrMap& dst)
        {
            auto& attr = dst["dtype"];
            attr.set_type(GetDataType(src));
        }

        static void PopulateOutputShapesAttr(const std::vector<Variable>& src, AttrMap& dst)
        {
            auto& attr = dst["_output_shapes"];
            auto list = attr.mutable_list();
            for (auto output : src)
            {
                PopulateShape(output.Shape(), *list->add_shape());
            }
        }

        static void PopulateNodeDef(
            const std::wstring& name,
            const std::wstring& opName,
            DataType dataType,
            const std::vector<Variable>& outputs,
            tensorflow::NodeDef& dst)
        {
            dst.set_name(ToString(name));
            dst.set_op(ToString(opName));

            PopulateDataTypeAttr(dataType, *dst.mutable_attr());
            PopulateOutputShapesAttr(outputs, *dst.mutable_attr());
            PopulateShapeAttr(outputs[0].Shape(), *dst.mutable_attr());
        }

        static void PopulateNodeDef(const std::wstring& scope, const FunctionPtr& src, tensorflow::NodeDef& dst)
        {
            PopulateNodeDef(GetScopedName(scope, src), src->OpName(), src->Output().GetDataType(), src->Outputs(), dst);
        }

        static void PopulateNodeDef(const std::wstring& scope, const Variable& src, tensorflow::NodeDef& dst)
        {
            // Constant nodes in TensorBoard have special meaning, so need to set the expected name.
            std::wstring opName = (src.Kind() == VariableKind::Constant) ? L"Const" : VariableKindName(src.Kind());
            PopulateNodeDef(GetScopedName(scope, &src), opName, src.GetDataType(), { src }, dst);
            // TODO: set attrs["value"] for Constant - how to get the value?
        }

        // Forward-declaration.
        static tensorflow::NodeDef* CreateFunctionNode(
            const FunctionPtr& src,
            tensorflow::GraphDef& dst,
            std::unordered_map<FunctionPtr, tensorflow::NodeDef*>& functionNodes,
            std::unordered_map<Variable, tensorflow::NodeDef*>& variableNodes,
            std::unordered_multimap<tensorflow::NodeDef*, VariableWithScope>& placeholders,
            std::unordered_map<Variable, VariableWithScope>& placeholderInputs,
            const std::wstring& scope);

        static tensorflow::NodeDef* CreateVariableNode(
            const Variable& src,
            tensorflow::GraphDef& dst,
            std::unordered_map<FunctionPtr, tensorflow::NodeDef*>& functionNodes,
            std::unordered_map<Variable, tensorflow::NodeDef*>& variableNodes,
            std::unordered_multimap<tensorflow::NodeDef*, VariableWithScope>& placeholders,
            std::unordered_map<Variable, VariableWithScope>& placeholderInputs,
            const std::wstring& scope)
        {
            auto iter = variableNodes.find(src);
            if (iter != variableNodes.end())
            {
                return iter->second;
            }

            tensorflow::NodeDef* result = nullptr;
            if (src.IsOutput())
            {
                result = CreateFunctionNode(src.Owner(), dst, functionNodes, variableNodes, placeholders,
                                            placeholderInputs, scope);
            }
            else
            {
                result = dst.add_node();
                PopulateNodeDef(scope, src, *result);
            }

            variableNodes.emplace(src, result);
            return result;
        }

        static tensorflow::NodeDef* CreateFunctionNode(
            const FunctionPtr& src,
            tensorflow::GraphDef& dst,
            std::unordered_map<FunctionPtr, tensorflow::NodeDef*>& functionNodes,
            std::unordered_map<Variable, tensorflow::NodeDef*>& variableNodes,
            std::unordered_multimap<tensorflow::NodeDef*, VariableWithScope>& placeholders,
            std::unordered_map<Variable, VariableWithScope>& placeholderInputs,
            const std::wstring& scope)
        {
            auto iter = functionNodes.find(src);
            if (iter != functionNodes.end())
            {
                return iter->second;
            }

            // Create a function node.
            tensorflow::NodeDef* functionNode = nullptr;
            if (src->IsBlock())
            {
                std::wstring newScope = GetScopedName(scope, src);
                functionNode = CreateFunctionNode(src->BlockRoot(), dst, functionNodes,
                                                  variableNodes, placeholders, placeholderInputs, newScope);
            }
            else
            {
                functionNode = dst.add_node();
                PopulateNodeDef(scope, src, *functionNode);
            }

            // Note that we map the block function to its root node here (on purpose).
            functionNodes.emplace(src, functionNode);

            // Make sure that all of the function's input nodes are created and registered as the 
            // function node's inputs.
            for (const auto& input : src->Inputs())
            {
                tensorflow::NodeDef* inputNode = nullptr;
                if (!input.IsPlaceholder())
                {
                    // We don't create placeholders immediately, just register them so we can later trace the actual
                    // input.
                    inputNode = CreateVariableNode(input, dst, functionNodes, variableNodes, placeholders,
                                                   placeholderInputs, scope);
                }

                if (!src->IsBlock())
                {
                    if (inputNode != nullptr)
                    {
                        functionNode->add_input(inputNode->name());
                    }
                    else
                    {
                        placeholders.emplace(functionNode, std::make_pair(input, scope));
                    }
                }
            }

            if (src->IsBlock())
            {
                // Remember placeholder->input mapping, so that we can later use it to map function to their inputs
                // directly (avoiding placeholder chains).
                for (const auto& mapping : src->BlockArgumentsMapping())
                {
                    placeholderInputs.emplace(mapping.first, std::make_pair(mapping.second, scope));
                }
            }

            return functionNode;
        }

        void CreateTensorBoardGraph(const FunctionPtr& src, tensorflow::GraphDef& dst)
        {
            // For each function/variable visited, contains a matching tensorflow node.
            std::unordered_map<FunctionPtr, tensorflow::NodeDef*> functionNodes;
            std::unordered_map<Variable, tensorflow::NodeDef*> variableNodes;

            // For each (function, placeholder input) found, contains (tensorflow_node, (placeholder, scope)).
            std::unordered_multimap<tensorflow::NodeDef*, VariableWithScope> placeholders;
            // For each placeholder found, contains a (placeholder, (replacement variable, scope)).
            std::unordered_map<Variable, VariableWithScope> placeholderInputs;

            // Create all nodes in the graph, except placeholders.
            CreateFunctionNode(src, dst, functionNodes, variableNodes, placeholders, placeholderInputs, L"");

            // For each function that had a placeholder as its input, add a link to the actual input if it was
            // found.
            for (auto placeholder : placeholders)
            {
                // Follow the placeholder chain until the end.
                VariableWithScope* finalValue = &placeholder.second;

                do
                {
                    auto nextInput = placeholderInputs.find(finalValue->first);
                    if (nextInput == placeholderInputs.end())
                    {
                        break;
                    }

                    finalValue = &nextInput->second;
                } while (true);

                // Create a node for the finalValue and add it as an input of the function.
                tensorflow::NodeDef* inputNode = CreateVariableNode(
                    finalValue->first, dst, functionNodes, variableNodes, placeholders, placeholderInputs,
                    finalValue->second);
                placeholder.first->add_input(inputNode->name());
            }
        }
    }
}