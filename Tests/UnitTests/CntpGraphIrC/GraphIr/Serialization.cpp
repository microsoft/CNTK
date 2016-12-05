//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
//#include "Utils.h"
#include <istream>
#include <ostream>
#include <string>
#include <locale>         // std::wstring_convert
#include <codecvt>        // std::codecvt_utf8#include <vector>
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

    std::wstring ToWString(const std::string& string)
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
        friend const google::protobuf::Message* Serialize(const FunctionPtr& modelFuncPtr);
        friend const FunctionPtr Deserialize(const FunctionPtr& modelFuncPtr, google::protobuf::Message* graph);

    private:
        static proto::Graph* CreateGraphProto(const Dictionary& src, std::unordered_map<std::wstring, NDShape> outputShapes, Arena* arena = nullptr);
        static proto::Node* CreateNodeProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::IOArg* CreateIOArgProto(const std::wstring& src, Arena* arena);

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
        dst->set_allocated_graph_info(graphInfo);

        // PassFindComputationalNodes
        auto pfunctions = src[L"primitive_functions"].Value<std::vector<DictionaryValue>>();
        for (auto funct : pfunctions)
        {
            auto value = funct.Value<Dictionary>();
            printf("function: %S\n", value[L"uid"].Value<std::wstring>().c_str());

            dst->mutable_nodes()->AddAllocated(CreateNodeProto(value, arena));
        }

        // PassConnectVariablesNodes
        // ...we iterate on the variables, i.e., external inputs
        auto pinputs = src[L"inputs"].Value<std::vector<DictionaryValue>>();
        for (auto input : pinputs)
        {
            auto value = input.Value<Dictionary>();
            auto shape = value[L"shape"].Value<NDShape>();
            auto iuip = ToString(value[L"uid"].Value<std::wstring>());
            auto kind = value[L"kind"].Value<size_t>();

            // print name of variable we're going to connect
            printf("variables : %s, name %S\n", iuip.c_str(), value[L"name"].Value<std::wstring>().c_str());

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

                        // ...append the dimensions of the shape to the ioArg constructed previously
                        for (auto n3 = 0; n3 < shape.Rank(); n3++)
                        {
                            ioArg->add_shape((unsigned int)shape[n3]);
                        }

                        // now check if the data is already part of the serialized stream
                        if (kind == (int)VariableKind::Constant || kind == (int)VariableKind::Parameter)
                        {
                            const auto& arrayview = value[L"value"].Value<NDArrayView>();

                            auto initArg = CreateInitArgProto(arrayview);
                            (*node->mutable_init_attrs())[iuip] = *initArg;
                            printf("  This is a parameter i need to embed: %s\n", initArg->data_base64().c_str());
                        }
                    }
                }

                // check if this node is supposed to provide this data
                auto outPrefix = node->name() + "_Output_";
                if (iuip.substr(0, outPrefix.size()) == outPrefix)
                {
                    // this in an out variable for this node
                    printf(" current node %s is out source for %s\n", node->name().c_str(), iuip.c_str());

                    // try to find node
                    auto found = false;
                    for (auto n4 = 0; !found && n4 < node->outputs_size(); n4++)
                    {
                        found |= node->outputs(n4).name() == iuip;
                    }

                    if (!found)
                    {
                        auto ioArg2 = CreateIOArgProto(value[L"uid"].Value<std::wstring>(), arena);

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

        // PassConnectComputationNodes
        // we provide output references for data shared between two nodes that
        // are not covered by a Variable
        for (auto n = 0; n < dst->nodes_size(); n++)
        {
            auto node = dst->mutable_nodes(n);

            // for every input
            for (auto n1 = 0; n1 < node->inputs_size(); n1++)
            {
                const auto& input = node->inputs(n1);
                const auto& postfix = input.name().find("_Output_", 0);

                // if not ending on output_, ignore it.
                if (postfix == string::npos)
                {
                    continue;
                }

                // find the owner of this data buffer
                graphIR::Node *node2 = nullptr;
                auto node2name = input.name().substr(0, postfix);
                for (auto n2 = 0; n2 < dst->nodes_size(); n2++)
                {
                    auto node3 = dst->mutable_nodes(n2);
                    if (node3->name() == node2name)
                    {
                        node2 = node3;
                        break;
                    }
                }

                // there has to be an owner of an output buffer
                assert(node2 != nullptr);

                // now check if the owner knows that he is the owner
                auto hasOutput = false;
                for (auto n3 = 0; !hasOutput &&  (n3 < node2->outputs_size()); n3++)
                {
                    const auto& output = node2->outputs(n3);
                    hasOutput |= output.name() == input.name();
                }

                // owner does not know about connection, so
                // lets add it here.
                if (!hasOutput)
                {
                    auto ioArg3 = CreateIOArgProto(ToWString(input.name()), arena);

                    // note: for these direct links between two nodes
                    //       we don't have the shape data in the static description
                    //       and thus assume the shape is not yet set.
                    assert(input.shape().size() == 0);

                    // ...append the dimensions of the shape to the ioArg constructed previously
                    // note: no need to add the data here since we are generating it in the node.
                    const auto& shape = outputShapes[ToWString(input.name())];

                    for (auto n6 = 0; n6 < shape.Rank(); n6++)
                    {
                        ioArg3->add_shape((unsigned int)shape[n6]);
                    }

                    printf("  DIREKT LINK adding new output %s for node %s\n", input.name().c_str(), node2->name().c_str());
                    node2->mutable_outputs()->AddAllocated(ioArg3);
                }
            }
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
   
    std::istream& operator>>(std::istream& stream, Message& msg)
    {
        //io::IstreamInputStream isistream(&stream);
        //io::CodedInputStream input(&isistream);
        //if (!ParseMessage(input, msg))
        //{
        //     RuntimeError("Failed to parse protobuf %s from the input stream.",
        //                  msg.GetTypeName().c_str());
        //}
        return stream;
    }

    const google::protobuf::Message* Serialize(const FunctionPtr& modelFuncPtr)
    {
        UsingUTF8 locale;
        Arena arena;

        auto dictionary = modelFuncPtr->Serialize();

        std::unordered_map<std::wstring, NDShape> outputShapes;
        Serializer::CollectOutputShapes(modelFuncPtr, outputShapes);

        proto::Graph* proto = Serializer::CreateGraphProto(dictionary, outputShapes, &arena);
        return proto;
    }

    const FunctionPtr Deserialize(const FunctionPtr& modelFuncPtr, google::protobuf::Message* graph)
    {
        UsingUTF8 locale;
        //proto::Dictionary proto;
        //stream >> proto;
        //dictionary.m_dictionaryData->reserve(proto.data_size());
        //for (const auto& kv : proto.data())
        //{
        //    Serializer::Copy(kv.second, dictionary[ToWString(kv.first)]);
        //}
        return nullptr;
    }
}
