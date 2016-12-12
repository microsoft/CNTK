//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma warning(disable: 4800 4267 4244)
#define _CRT_SECURE_NO_WARNINGS
#include "CNTKLibrary.h"
#include <istream>
#include <ostream>
#include <string>
#include <locale>         // std::wstring_convert
#include <codecvt>        // std::codecvt_utf8
#include <vector>
#include <limits>

#ifdef _MSC_VER
#include <io.h> // _close()
#endif

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "GraphIr.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/arena.h>
#pragma warning(pop)

#pragma warning(disable : 4100) // unreferenced formal parameter

extern std::string EncodeBase64(const char *buf, size_t len);
extern std::vector<char> DecodeBase64(const std::string str);

namespace GRAPHIR
{
    using namespace ::CNTK;
    using namespace ::google::protobuf;
    namespace proto = ::graphIR;

    std::string ToString(const std::wstring& wstring)
    {
#ifdef _MSC_VER
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        return converter.to_bytes(wstring);
#else
        const auto length = wstring.length() * sizeof(std::wstring::value_type) + 1;
        char buf[length];
        const auto res = std::wcstombs(buf, wstring.c_str(), sizeof(buf));
        return (res >= 0) ? buf : "";
#endif
    }

    std::wstring ToWString2(const std::string& string)
    {
#ifdef _MSC_VER
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        return converter.from_bytes(string);
#else
        const auto length = string.length() + 1;
        wchar_t buf[length];
        const auto res = std::mbstowcs(buf, string.c_str(), sizeof(buf));
        return (res >= 0) ? buf : L"";
#endif
    }

    class Serializer
    {
        friend const proto::Graph* Serialize(const FunctionPtr& modelFuncPtr);
        friend const FunctionPtr Deserialize(const proto::Graph* graph);

    private:
        static proto::Graph* CreateGraphProto(const Dictionary& src, std::unordered_map<std::wstring, NDShape> outputShapes, Arena* arena = nullptr);
        static proto::Node* CreateFunctionProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::Node* CreateVariableProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::IOArg* CreateIOArgProto(const std::wstring& src, Arena* arena);
        static void UpdateConnectionShapes(proto::Graph* dst, const std::wstring& uip, const NDShape& shape);

        static Dictionary Serializer::CreateGraphDictionary(const proto::Graph& src);
        
        static void Copy(std::string prefix, const DictionaryValue& src, proto::Node& dst);
        static void Copy(std::string prefix, const proto::Node& src, DictionaryValue& dst);

        template <typename T>
        static void CopyData(const NDArrayView& src, RepeatedField<T>* dst)
        {
            auto size = src.Shape().TotalSize();
            if (size > std::numeric_limits<int>::max())
            {
                InvalidArgument("NDArrayView is too big to fit in a protobuf.");
            }
            dst->Resize((int)size, T());
            const T* buffer = src.DataBuffer<T>();
            memcpy(dst->mutable_data(), buffer, (int)size * sizeof(T));
        }

        template <typename T>
        static void CopyData(const RepeatedField<T>& src, NDArrayView* dst)
        {
            auto size = src.size();
            assert(size == dst->Shape().TotalSize());;
            T* buffer = dst->WritableDataBuffer<T>();
            memcpy(buffer, src.data(), size * sizeof(T));
        }

        template <typename FunctionType>
        static void Traverse(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions, const FunctionType& functor)
        {
            visitedFunctions.insert(rootFunction);
            functor(rootFunction);

            std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
            for (const auto& rootInput : rootFunctionInputs)
            {
                if (rootInput.IsOutput() && visitedFunctions.find(rootInput.Owner()) == visitedFunctions.end())
                {
                    const auto& function = rootInput.Owner();
                    Traverse(function, visitedFunctions, functor);
                }
            }
        }

        static void CollectOutputShapes(FunctionPtr evalFunc, std::unordered_map<std::wstring, NDShape> & outputShapes)
        {
            std::unordered_set<FunctionPtr> functions;
            Traverse(evalFunc->RootFunction(), functions,
                [&outputShapes](const FunctionPtr& f)
            {
                fprintf(stderr, "now at %S opcode %S\n", f->Uid().c_str(), f->OpName().c_str());

                auto index = 0;
                for (auto output : f->Outputs())
                {
                    auto outputUid = f->Uid() + L"_Output_" + std::to_wstring(index++);
                    auto shape = output.Shape();

                    outputShapes[outputUid] = shape;
                }
            });
        }
    };

    proto::InitArg* CreateInitArgProto(const NDArrayView& arrayview)
    {
        proto::InitArg* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::InitArg>(arena) :*/ new proto::InitArg();

        const auto& shape2 = arrayview.Shape();
        assert(arrayview.GetStorageFormat() == StorageFormat::Dense);

        auto data = arrayview.DataBuffer<float>();
        auto count = shape2.TotalSize();
        auto base64data = EncodeBase64((char *)data, count * sizeof(float));

        dst->set_dbytes(sizeof(float));
        dst->set_data_base64(base64data);

        return dst;
    }

    /*static*/ void Serializer::UpdateConnectionShapes(proto::Graph* dst, const std::wstring& uip, const NDShape& shape)
    {
        auto iuip = ToString(uip);

        // ...find any node referencing the variable
        for (auto n = 0; n < dst->nodes_size(); n++)
        {
            auto node = dst->mutable_nodes(n);

            // ...see if this node is consuming the variable as input
            for (auto n2 = 0; n2 < node->inputs_size(); n2++)
            {
                auto ioArg = node->mutable_inputs(n2);
                if (ioArg->name() == iuip)
                {
                    printf("  node %s consumes input %s\n", node->name().c_str(), iuip.c_str());

                    assert(shape.Rank() != ioArg->shape_size() || !shape.Rank());

                    // ...append the dimensions of the shape to the ioArg constructed previously
                    for (auto n3 = 0; n3 < shape.Rank(); n3++)
                    {
                        ioArg->add_shape((unsigned int)shape[n3]);
                    }

                    assert(shape.Rank() == ioArg->shape_size());
                }
            }

            // ...check if this node is supposed to provide this data
            //    (in this case we must add the name to the outputs list)
            auto outPrefix = node->name() + "_Output_";
            if (iuip.substr(0, outPrefix.size()) == outPrefix)
            {
                // this in an out variable for this node
                printf(" current node %s is out source for %s\n", node->name().c_str(), iuip.c_str());

                // try to find link in outputs
                auto found = false;
                for (auto n4 = 0; !found && n4 < node->outputs_size(); n4++)
                {
                    found |= node->outputs(n4).name() == iuip;
                }

                if (!found)
                {
                    auto ioArg2 = CreateIOArgProto(uip, nullptr);

                    // ...append the dimensions of the shape to the ioArg constructed previously
                    // note: no need to add the data here since we are generating it in the node.
                    for (auto n5 = 0; n5 < shape.Rank(); n5++)
                    {
                        ioArg2->add_shape((unsigned int)shape[n5]);
                    }

                    printf("  adding new output %s\n", iuip.c_str());
                    node->mutable_outputs()->AddAllocated(ioArg2);
                }
            }
        }
    }

    /*static*/ proto::Graph* Serializer::CreateGraphProto(const Dictionary& src, std::unordered_map<std::wstring, NDShape> outputShapes, Arena* arena)
    {
        proto::Graph* dst = /*(arena != nullptr) ?
            Arena::CreateMessage<proto::Graph>(arena) :*/ new proto::Graph();

        auto graphInfo = new graphIR::GraphInfo();
        graphInfo->set_framework_name("CNTK");
        graphInfo->set_framework_version("2.0beta3.0"); // TODO: call cntk function to retrieve version string
        graphInfo->set_graph_version("0.3");
        graphInfo->set_description("Exported by the Graph Ir Exporter from CNTK");
        graphInfo->set_model_name(ToString(src[L"name"].Value<std::wstring>()));

        (*graphInfo->mutable_attrs())["type"] = ToString(src[L"type"].Value<std::wstring>());
        (*graphInfo->mutable_attrs())["root"] = ToString(src[L"root"].Value<std::wstring>());
        (*graphInfo->mutable_attrs())["version"] = std::to_string(src[L"version"].Value<size_t>());
        (*graphInfo->mutable_attrs())["uid"] = ToString(src[L"uid"].Value<std::wstring>());
        dst->set_allocated_graph_info(graphInfo);

        // PassFindComputationalNodes
        auto pfunctions = src[L"primitive_functions"].Value<std::vector<DictionaryValue>>();
        for (auto funct : pfunctions)
        {
            auto value = funct.Value<Dictionary>();
            printf("function: %S\n", value[L"uid"].Value<std::wstring>().c_str());

            dst->mutable_nodes()->AddAllocated(CreateFunctionProto(value, arena));
        }

        // PassFindVariableNodes
        auto pinputs = src[L"inputs"].Value<std::vector<DictionaryValue>>();
        for (auto input : pinputs)
        {
            auto value = input.Value<Dictionary>();
            auto kind  = value[L"kind"].Value<size_t>();
            auto uid   = value[L"uid"].Value<std::wstring>();

            printf("input: %S\n", uid.c_str());

            //if (kind == (size_t)VariableKind::Constant || kind == (size_t)VariableKind::Parameter)
            {
                // add the parameter as a node
                dst->mutable_nodes()->AddAllocated(CreateVariableProto(value, arena));
            }
        }

        // PassUpdateTensorShapes
        for (auto input : pinputs)
        {
            auto value = input.Value<Dictionary>();
            auto shape = value[L"shape"].Value<NDShape>();
            auto uid = value[L"uid"].Value<std::wstring>();

            printf("input: %S\n", uid.c_str());

            UpdateConnectionShapes(dst, uid, shape);
        }

        return dst;
    }

    /*static*/ proto::IOArg* Serializer::CreateIOArgProto(const std::wstring& src, Arena* arena)
    {
        proto::IOArg* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::IOArg>(arena) :*/ new proto::IOArg();

        dst->set_name(ToString(src));
        dst->set_dtype("fp32");
        dst->set_dbytes(sizeof(float));

        // Note: adding the shape will be added in the second pass, integrating the variables.
        return dst;
    }

    /*static*/ proto::Node* Serializer::CreateFunctionProto(const Dictionary& src, Arena* arena)
    {
        proto::Node* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::Node>(arena) :*/ new proto::Node();

        // setting main properties
        dst->set_name(ToString(src[L"uid"].Value<std::wstring>()));
        dst->set_op(std::to_string(src[L"op"].Value<size_t>())); // TODO: map op to name

        auto &ext = *dst->mutable_ext_attrs();
        ext["version"]  = std::to_string(src[L"version"].Value<size_t>());
        ext["type"]     = ToString(src[L"type"].Value<std::wstring>());
        ext["name"]     = ToString(src[L"name"].Value<std::wstring>());

        Copy("attributes", src[L"attributes"], *dst);

        auto inputs = src[L"inputs"].Value<std::vector<DictionaryValue>>();
        dst->mutable_inputs()->Reserve((int)inputs.size());
        for (auto input : inputs)
        {
            auto value = input.Value<std::wstring>();
            printf("input: %S\n", value.c_str());

            dst->mutable_inputs()->AddAllocated(CreateIOArgProto(value, arena));
        }

        // note: outputs will be added in a second phase
        return dst;
    }

    /*static*/ proto::Node* Serializer::CreateVariableProto(const Dictionary& src, Arena* arena)
    {
        proto::Node* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::Node>(arena) :*/ new proto::Node();

        auto shape = src[L"shape"].Value<NDShape>();
        auto uid   = src[L"uid"].Value<std::wstring>();
        auto name  = src[L"name"].Value<std::wstring>();
        auto kind  = src[L"kind"].Value<size_t>();

        // setting main properties
        dst->set_name(ToString(uid));
        dst->set_op("Parameter");

        // TODO: use serializer, not replicate functionality here.
        auto &ext = *dst->mutable_ext_attrs();
        ext["version"]        = std::to_string(src[L"version"].Value<size_t>());
        ext["type"]           = ToString(src[L"type"].Value<std::wstring>());
        ext["kind"]           = std::to_string(kind);
        ext["data_type"]      = std::to_string(src[L"data_type"].Value<size_t>());
        ext["is_sparse"]      = std::to_string(src[L"is_sparse"].Value<bool>());
        ext["name"]           = ToString(name);
        ext["needs_gradient"] = std::to_string(src[L"needs_gradient"].Value<bool>());

        auto axis = src[L"dynamic_axis"].Value<std::vector<DictionaryValue>>();
        Copy("dynamic_axis", axis, *dst);

        printf("output: %S\n", uid.c_str());

        auto ioarg = CreateIOArgProto(uid, arena);
        // ...append the dimensions of the shape to the ioArg constructed previously
        for (auto n3 = 0; n3 < shape.Rank(); n3++)
        {
            ioarg->add_shape((unsigned int)shape[n3]);
        }

        dst->mutable_outputs()->AddAllocated(ioarg);

        // now check if the data is already part of the serialized stream
   //     assert(kind == (int)VariableKind::Constant || kind == (int)VariableKind::Parameter);
        if (src.Contains(L"value"))
        {
            const auto& arrayview = src[L"value"].Value<NDArrayView>();

            auto initArg = CreateInitArgProto(arrayview);
            (*dst->mutable_init_attrs())[ToString(uid)] = *initArg;
            printf("  This is a parameter i need to embed: %p\n", initArg->data_base64().c_str());
        }


        // note: outputs will be added in a second phase
        return dst;
    }

    /*static*/ Dictionary Serializer::CreateGraphDictionary(const proto::Graph& src)
    {
        const auto& graphInfo = src.graph_info();
        assert(graphInfo.framework_name() == "CNTK");
        assert(graphInfo.framework_version() == "2.0beta3.0"); // TODO: call cntk function to retrieve version string
        assert(graphInfo.graph_version() == "0.3");

        auto& root = *(new Dictionary());
        root[L"type"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("type"));
        root[L"root"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("root"));
        root[L"uid"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("uid"));
        root[L"version"]  = (size_t)atoi(src.graph_info().attrs().at("version").c_str()); // TODO check
        root[L"name"] = GRAPHIR::ToWString2(graphInfo.model_name());

        // create the list of primitive functions
        std::vector<DictionaryValue> primitiveFunctions;
        std::vector<DictionaryValue> inputVariables;
        for (auto& node : src.nodes())
        {
            Dictionary subNode;
            auto& ext = node.ext_attrs();

            auto op = node.op();
            subNode[L"uid"] = ToWString2(node.name());
            subNode[L"type"] = ToWString2(ext.at("type"));
            subNode[L"name"] = ToWString2(ext.at("name"));
            subNode[L"version"] = (size_t)(ext.at("version")[0] - '0');

            if (op == "Parameter")
            {
                subNode[L"kind"] = (size_t)atoi(ext.at("kind").c_str());
                subNode[L"data_type"] = (size_t)atoi(ext.at("data_type").c_str());
                subNode[L"is_sparse"] = (bool)atoi(ext.at("is_sparse").c_str());
                subNode[L"needs_gradient"] = (bool)atoi(ext.at("needs_gradient").c_str());

                DictionaryValue axisvector;
                Copy("dynamic_axis", node, axisvector);
                subNode[L"dynamic_axis"] = axisvector;

                for (const auto& output : node.outputs())
                {
                    const auto& outputName = output.name();
               
                    std::vector<size_t> shape; // NDShape = CreateNDShapeFromProto(input.shape());
                    for (auto& dim : output.shape())
                    {
                        shape.push_back(dim);
                    }
                    subNode[L"shape"] = (NDShape)shape;

                    auto x = node.init_attrs().find(outputName);
                    if (x != node.init_attrs().end())
                    {
                        const auto& initData = node.init_attrs().at(outputName).data_base64();

                        auto dataBuffer = DecodeBase64(initData);
                        NDArrayView av(::CNTK::DataType::Float, (NDShape)shape, &dataBuffer[0], dataBuffer.size(), DeviceDescriptor::CPUDevice());

                        subNode[L"value"] = av;
                    }
                }

                inputVariables.push_back(subNode);
            }
            else
            {
                subNode[L"op"] = (size_t)atoi(op.c_str());

                DictionaryValue attributes;
                Copy("attributes", node, attributes);
                subNode[L"attributes"] = attributes;

                std::vector<DictionaryValue> inputs;
                for (auto input : node.inputs())
                {
                    inputs.push_back(ToWString2(input.name()));
                }

                subNode[L"inputs"] = inputs;

                primitiveFunctions.push_back(subNode);
            }
        }

        root[L"primitive_functions"] = primitiveFunctions;
        root[L"inputs"] = inputVariables;

        return root;
    }

    /*static*/ void Serializer::Copy(std::string prefix, const DictionaryValue& src, proto::Node& dst)
    {
        size_t n = 0;
        auto valueType = src.ValueType();
        switch (valueType)
        {
        case DictionaryValue::Type::None:
            break;
        case DictionaryValue::Type::Bool:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<bool>() ? 1 : 0);
            break;
        case DictionaryValue::Type::Int:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<int>());
            break;
        case DictionaryValue::Type::SizeT:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<size_t>());
            break;
        case DictionaryValue::Type::Float:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<float>());
            break;
        case DictionaryValue::Type::Double:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<double>());
            break;
        case DictionaryValue::Type::String:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + ToString(src.Value<std::wstring>());
            break;
        case DictionaryValue::Type::Axis:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#Axis");
            (*dst.mutable_attrs())[prefix + ".Axis.static_axis_idx"] = std::to_string(src.Value<Axis>().StaticAxisIndex(false));
            (*dst.mutable_attrs())[prefix + ".Axis.name"] = ToString(src.Value<Axis>().Name());
            (*dst.mutable_attrs())[prefix + ".Axis.is_ordered_dynamic_axis"] = std::to_string(src.Value<Axis>().IsOrdered() ? 1 : 0);
            break;
        case DictionaryValue::Type::Vector:
            (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + std::to_string(src.Value<std::vector<DictionaryValue>>().size());
            n = 0;
            for (auto node : src.Value<std::vector<DictionaryValue>>())
            {
                Copy(prefix + ".Vector." + std::to_string(n++), node, dst);
            }
            break;
        case DictionaryValue::Type::Dictionary:
            {
                std::string keys = "";
                for (auto node : src.Value<Dictionary>())
                {
                    keys += ToString(node.first) + ",";

                    Copy(prefix + ".Dictionary." + ToString(node.first), node.second, dst);
                }
                (*dst.mutable_attrs())[prefix] = std::to_string((unsigned int)valueType) + std::string("#") + keys;
            }
            break;
        ////case DictionaryValue::Type::NDShape:
        ////    (*dst.mutable_attrs()).set_allocated_nd_shape_value(CreateProto(src.Value<NDShape>(), arena));
        ////    break;
        ////case DictionaryValue::Type::NDArrayView:
        ////    (*dst.mutable_attrs()).set_allocated_nd_array_view_value(CreateProto(src.Value<NDArrayView>(), arena));
        ////    break;
        default:
            NOT_IMPLEMENTED
        }
    }


    /*static*/ void Serializer::Copy(std::string prefix, const proto::Node& src, DictionaryValue& dst)
    {
        auto wprefix = ToWString2(prefix);
        auto line = src.attrs().at(prefix);
        auto idx = line.find('#');

        auto valueType = (DictionaryValue::Type)atoi(line.substr(0, idx).c_str());
        auto valueValue = line.substr(idx + 1);

        switch (valueType)
        {
        case DictionaryValue::Type::None:
            break;
        case DictionaryValue::Type::Bool:
            dst = atoi(valueValue.c_str()) != 0;
            break;
        case DictionaryValue::Type::Int:
            dst = (int)atoi(valueValue.c_str());
            break;
        case DictionaryValue::Type::SizeT:
            dst = (size_t)atoi(valueValue.c_str());
            break;
        case DictionaryValue::Type::String:
            dst = ToWString2(valueValue);
            break;
        case DictionaryValue::Type::Float:
            {
                float floatValue;
                std::sscanf(valueValue.c_str(), "%f", &floatValue);
                dst = floatValue;
            }
            break;
        case DictionaryValue::Type::Double:
            {
                double doubleValue;
                std::sscanf(valueValue.c_str(), "%lf", &doubleValue);
                dst = doubleValue;
            }
            break;
        case DictionaryValue::Type::Vector:
            {
                std::vector<DictionaryValue> v;
                for (auto n = 0; n < atoi(valueValue.c_str()); n++)
                {
                    DictionaryValue vv;
                    Copy(prefix + ".Vector." + std::to_string(n), src, vv);
                    v.push_back(vv);
                }
                dst = v;
            }
            break;
        case DictionaryValue::Type::Axis:
            {
                size_t static_axis_idx = atoi(src.attrs().at(prefix + ".Axis.static_axis_idx").c_str());

                if (!Axis(static_axis_idx).IsDynamicAxis())
                {
                    dst = Axis(static_axis_idx);
                }
                else
                {
                    auto axisname = ToWString2(src.attrs().at(prefix + ".Axis.name"));
                    auto is_ordered_dynamic_axis = atoi(src.attrs().at(prefix + ".Axis.is_ordered_dynamic_axis").c_str()) ? true : false;

                    dst = Axis(axisname, is_ordered_dynamic_axis);
                }
            }
            break;
        case DictionaryValue::Type::Dictionary:
            {
                Dictionary dict;
                size_t wstart = 0;
                size_t wend = valueValue.find(',', wstart + 1);
                while (wend != string::npos)
                {
                    auto dictkey = valueValue.substr(wstart, wend - wstart);

                    DictionaryValue dictvalue;
                    Copy(prefix + ".Dictionary." + dictkey, src, dictvalue);

                    dict[ToWString2(dictkey)] = dictvalue;

                    wstart = wend + 1;
                    wend = valueValue.find(',', wstart + 1);
                }

                dst = dict;
            }
            break;
            ////case DictionaryValue::Type::NDShape:
            ////    (*dst.mutable_attrs()).set_allocated_nd_shape_value(CreateProto(src.Value<NDShape>(), arena));
            ////    break;
            ////case DictionaryValue::Type::NDArrayView:
            ////    (*dst.mutable_attrs()).set_allocated_nd_array_view_value(CreateProto(src.Value<NDArrayView>(), arena));
            ////    break;
        default:
            NOT_IMPLEMENTED
        }
    }

    static void SetUTF8Locale()
    {
#ifndef _MSC_VER
        if (std::setlocale(LC_ALL, "C.UTF-8") == nullptr)
        {
            std::setlocale(LC_ALL, "en_US.UTF-8");
        }
#endif
    }

    static void UnsetUTF8Locale()
    {
#ifndef _MSC_VER
        std::setlocale(LC_ALL, "");
#endif
    }

    struct UsingUTF8
    {
        UsingUTF8() { SetUTF8Locale(); }
        ~UsingUTF8() { UnsetUTF8Locale(); }
    };
  

    const proto::Graph* Serialize(const FunctionPtr& modelFuncPtr)
    {
        UsingUTF8 locale;
        Arena arena;

        // first let cntk serialize the function into its state
        auto dictionary = modelFuncPtr->Serialize();

        // second, get the output shapes of those nodes, not in the state data.
        // TODO: make cntk export this information as well
        std::unordered_map<std::wstring, NDShape> outputShapes;
        Serializer::CollectOutputShapes(modelFuncPtr, outputShapes);

        // third, create the graphir state
        proto::Graph* proto = Serializer::CreateGraphProto(dictionary, outputShapes, &arena);
        return proto;
    }

    const FunctionPtr Deserialize(const proto::Graph* graph)
    {
        UsingUTF8 locale;

        // first, get the cntk serialized state of the function
        auto state = Serializer::CreateGraphDictionary(*graph);

        // second, let cntk deserialize the state into a function
        auto result = Function::Deserialize(state, DeviceDescriptor::CPUDevice());

        return result;
    }
}
