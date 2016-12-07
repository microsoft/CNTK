//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <istream>
#include <ostream>
#include <string>
#include <locale>         // std::wstring_convert
#include <codecvt>        // std::codecvt_utf8
#include <vector>
#include <limits>

#ifdef _MSC_VER
#include <io.h>
#endif

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "GraphId.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/arena.h>
#pragma warning(pop)

#pragma warning(disable : 4100) // unreferenced formal parameter

#define ToWstring(a) std::to_wstring(a)
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
        friend const FunctionPtr Deserialize(const proto::Graph* graph, const FunctionPtr& modelFuncPtr);

    private:
        static proto::Graph* CreateGraphProto(const Dictionary& src, std::unordered_map<std::wstring, NDShape> outputShapes, Arena* arena = nullptr);
        static proto::Node* CreateNodeProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::Node* CreateVariableProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::IOArg* CreateIOArgProto(const std::wstring& src, Arena* arena);
        static void UpdateConnectionShapes(proto::Graph* dst, const std::wstring& uip, const NDShape& shape);

        static Dictionary Serializer::CreateGraphDictionary(const proto::Graph& src, const FunctionPtr& templateGraph);

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
        graphInfo->set_graph_version("0.2");
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

            dst->mutable_nodes()->AddAllocated(CreateNodeProto(value, arena));
        }

        // PassFindVariableNodes
        auto pinputs = src[L"inputs"].Value<std::vector<DictionaryValue>>();
        for (auto input : pinputs)
        {
            auto value = input.Value<Dictionary>();
            auto kind  = value[L"kind"].Value<size_t>();
            auto uid   = value[L"uid"].Value<std::wstring>();

            printf("input: %S\n", uid.c_str());

            if (kind == (size_t)VariableKind::Constant || kind == (size_t)VariableKind::Parameter)
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

    /*static*/ proto::Node* Serializer::CreateNodeProto(const Dictionary& src, Arena* arena)
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

        auto &ext = *dst->mutable_ext_attrs();
        ext["version"]  = std::to_string(src[L"version"].Value<size_t>());
        ext["kind"]     = std::to_string(kind);
        ext["type"]     = ToString(src[L"type"].Value<std::wstring>());
        ext["name"]     = ToString(name);

        printf("output: %S\n", uid.c_str());

        auto ioarg = CreateIOArgProto(uid, arena);
        // ...append the dimensions of the shape to the ioArg constructed previously
        for (auto n3 = 0; n3 < shape.Rank(); n3++)
        {
            ioarg->add_shape((unsigned int)shape[n3]);
        }

        dst->mutable_outputs()->AddAllocated(ioarg);

        // now check if the data is already part of the serialized stream
        assert(kind == (int)VariableKind::Constant || kind == (int)VariableKind::Parameter);
        {
            const auto& arrayview = src[L"value"].Value<NDArrayView>();

            auto initArg = CreateInitArgProto(arrayview);
            (*dst->mutable_init_attrs())[ToString(uid)] = *initArg;
            printf("  This is a parameter i need to embed: %p\n", initArg->data_base64().c_str());
        }


        // note: outputs will be added in a second phase
        return dst;
    }

    /*static*/ Dictionary Serializer::CreateGraphDictionary(const proto::Graph& src, const FunctionPtr& templateGraph)
    {
        const auto& graphInfo = src.graph_info();
        assert(graphInfo.framework_name() == "CNTK");
        assert(graphInfo.framework_version() == "2.0beta3.0"); // TODO: call cntk function to retrieve version string
        assert(graphInfo.graph_version() == "0.2");

        auto& root = *(new Dictionary());
        root[L"type"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("type"));
        root[L"root"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("root"));
        root[L"uid"] = GRAPHIR::ToWString2(src.graph_info().attrs().at("uid"));
        root[L"version"]  = (size_t)atoi(src.graph_info().attrs().at("version").c_str()); // TODO check
        root[L"name"] = GRAPHIR::ToWString2(graphInfo.model_name());

        // create the list of primitive functions
        std::vector<DictionaryValue> primitiveFunctions;
        root[L"primitive_functions"] = primitiveFunctions;
        for (auto& node : src.nodes())
        {
            Dictionary primitiveNode;
            
            auto& ext = node.ext_attrs();
            primitiveNode[L"uid"]     = ToWString2(node.name());
            primitiveNode[L"op"]      = (size_t)atoi(node.op().c_str());
            primitiveNode[L"version"] = (size_t)(ext.at("version")[0] - '0');
            primitiveNode[L"type"]    = ToWString2(ext.at("type"));
            primitiveNode[L"name"]    = ToWString2(ext.at("name"));

            primitiveFunctions.push_back(primitiveNode);
        }

        // create the list of variables and parameters
        std::unordered_map<std::string, DictionaryValue> inputValues;
        for (auto& node : src.nodes())
        {
            Dictionary variableNode;

            for (const auto& input : node.inputs())
            {
                const auto& name = input.name();
                if (name.find("_Output_") != std::string::npos)
                {
                    // value is output of another computation
                    // node, safe to skip
                    continue;
                }

                if (inputValues.find(name) != inputValues.end())
                {
                    // value already recorded,
                    // safe to skip.
                    continue;
                }

                std::vector<size_t> shape; // NDShape = CreateNDShapeFromProto(input.shape());
                for (auto& dim : input.shape())
                {
                    shape.push_back(dim);
                }

                auto x = node.init_attrs().find(name);
                if (x != node.init_attrs().end())
                {
                    const auto& initData = node.init_attrs().at(name).data_base64();

                    auto dataBuffer = DecodeBase64(initData);
                    NDArrayView av(::CNTK::DataType::Float, (NDShape)shape, &dataBuffer[0], dataBuffer.size(), DeviceDescriptor::CPUDevice());

                    variableNode[L"value"] = av;
                }

                variableNode[L"uid"] = GRAPHIR::ToWString2(name);
                variableNode[L"shape"] = (NDShape)shape;
                variableNode[L"kind"] = (size_t)VariableKind::Input;

                // update list
                inputValues[name] = variableNode;
            }
        }

        std::vector<DictionaryValue> inputVariables;
        for (const auto&tuple : inputValues)
        {
            inputVariables.push_back(tuple.second);
        }

        root[L"inputs"] = inputVariables;

        return root;
    }







    bool ParseMessage(io::CodedInputStream& input, Message& msg)
    {
        input.SetTotalBytesLimit(INT_MAX, INT_MAX);
        return msg.ParseFromCodedStream(&input) && input.ConsumedEntireMessage();
    }

    void ReadFromFile(std::wstring filename, Message& msg)
    {
        auto fd = 0; //TODO GetFileDescriptor(filename, true);
        {
            io::FileInputStream raw_input(fd);
            io::CodedInputStream coded_input(&raw_input);
            if (!ParseMessage(coded_input, msg)) 
            {
                RuntimeError("Failed to parse protobuf %s from file %ls.", 
                             msg.GetTypeName().c_str(), filename.c_str());
            }
        }
#ifdef _MSC_VER
        _close(fd);
#else
        close(fd);
#endif
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

        auto dictionary = modelFuncPtr->Serialize();

        std::unordered_map<std::wstring, NDShape> outputShapes;
        Serializer::CollectOutputShapes(modelFuncPtr, outputShapes);

        proto::Graph* proto = Serializer::CreateGraphProto(dictionary, outputShapes, &arena);
        return proto;
    }

    const FunctionPtr Deserialize(const proto::Graph* graph, const FunctionPtr& modelFuncPtr)
    {
        UsingUTF8 locale;

        const auto& templateGraph = modelFuncPtr->Serialize();

        auto state = Serializer::CreateGraphDictionary(*graph, modelFuncPtr);
        //proto::Dictionary proto;
        //stream >> proto;
        //dictionary.m_dictionaryData->reserve(proto.data_size());
        //for (const auto& kv : proto.data())
        //{
        //    Serializer::Copy(kv.second, dictionary[ToWString(kv.first)]);
        //}

        auto result = Function::Deserialize(state, DeviceDescriptor::CPUDevice());

        return result;
    }
}
