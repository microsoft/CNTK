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

    template <typename FunctionType>
    void Traverse(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions, const FunctionType& functor)
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

    void CollectOutputShapes(FunctionPtr evalFunc, std::unordered_map<std::wstring, NDShape> & outputShapes)
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


    class Serializer
    {
        friend const google::protobuf::Message* Serialize(const FunctionPtr& modelFuncPtr);
        friend const Dictionary* Derialize(const proto::Graph&);

    private:
        static proto::Graph* CreateGraphProto(const Dictionary& src, std::unordered_map<std::wstring, NDShape> outputShapes, Arena* arena = nullptr);
        static proto::Node* CreateNodeProto(const Dictionary& src, Arena* arena = nullptr);
        static proto::IOArg* CreateIOArgProto(const std::wstring& src, Arena* arena);

        static Dictionary* CreateFromProto(const proto::Graph& src);

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
        graphInfo->set_graph_version("0.1");
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
        // ...we iterate on the variables
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
                            ioArg->add_shape(shape[n3]);
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
                            ioArg2->add_shape(shape[n5]);
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
        // iterating over nodes
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
                        ioArg3->add_shape(shape[n6]);
                    }

                    printf("  DIREKT LINK adding new output %s for node %s\n", input.name().c_str(), node2->name().c_str());
                    node2->mutable_outputs()->AddAllocated(ioArg3);
                }
            }
        }


        //dst->set_version(src.s_version);
        //for (const auto& kv : src)
        //{
        //    Copy(kv.second, dst->mutable_data()->operator[](ToString(kv.first)), arena);
        //}

        return dst;
    }

    /*static*/ proto::IOArg* Serializer::CreateIOArgProto(const std::wstring& src, Arena* arena)
    {
        proto::IOArg* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::IOArg>(arena) :*/ new proto::IOArg();

        dst->set_name(ToString(src));
        dst->set_dtype("fp32");
        dst->set_dbytes(sizeof(float));

        // TODO: add shape
        // this will be added in the second pass, integrating the variables.

        return dst;
    }

    /*static*/ proto::Node* Serializer::CreateNodeProto(const Dictionary& src, Arena* arena)
    {
        proto::Node* dst = /*(arena != nullptr) ?
                            Arena::CreateMessage<proto::Node>(arena) :*/ new proto::Node();

        // setting main properties
        dst->set_name(ToString(src[L"uid"].Value<std::wstring>()));
        dst->set_op(std::to_string(src[L"op"].Value<size_t>()));

        auto &ext = *dst->mutable_ext_attrs();
        ext["version"]  = std::to_string(src[L"version"].Value<size_t>());
        ext["type"]     = ToString(src[L"type"].Value<std::wstring>());
        ext["name"]     = ToString(src[L"name"].Value<std::wstring>());

        // TODO: inputs from list, outputs from searching, attributes
        auto inputs = src[L"inputs"].Value<std::vector<DictionaryValue>>();
        dst->mutable_inputs()->Reserve((int)inputs.size());
        for (auto input : inputs)
        {
            auto value = input.Value<std::wstring>();
            printf("input: %S\n", value.c_str());

            dst->mutable_inputs()->AddAllocated(CreateIOArgProto(value, arena));
        }

        return dst;
    }
    
    /*static*/ Dictionary* Serializer::CreateFromProto(const proto::Graph& src)
    {
        Dictionary* dst = new Dictionary();

        //for (const auto& kv : src.data())
        //{
        //    Copy(kv.second, dst->operator[](ToWString(kv.first)));
        //}

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
        CollectOutputShapes(modelFuncPtr, outputShapes);

        proto::Graph* proto(Serializer::CreateGraphProto(dictionary, outputShapes, &arena));
        return proto;
    }

    const Dictionary* Derialize(const proto::Graph& graph)
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

#if 0

class Serializer
{
    friend std::ostream& operator<<(std::ostream&, const Dictionary&);
    friend std::istream& operator >> (std::istream&, Dictionary&);
    friend std::ostream& operator<<(std::ostream&, const DictionaryValue&);
    friend std::istream& operator >> (std::istream&, DictionaryValue&);

    friend class Dictionary;
    friend class DictionaryValue;

private:
    static proto::DictionaryValue* CreateProto(const DictionaryValue& src, Arena* arena = nullptr);
    static proto::Dictionary* CreateProto(const Dictionary& src, Arena* arena = nullptr);
    static proto::Vector* CreateProto(const std::vector<DictionaryValue>& src, Arena* arena = nullptr);
    static proto::NDArrayView* CreateProto(const NDArrayView& src, Arena* arena = nullptr);
    static proto::Axis* CreateProto(const Axis& src, Arena* arena = nullptr);
    static proto::NDShape* CreateProto(const NDShape& src, Arena* arena = nullptr);

    static Dictionary* CreateFromProto(const proto::Dictionary& src);
    static std::vector<DictionaryValue>* CreateFromProto(const proto::Vector& src);
    static NDArrayView* CreateFromProto(const proto::NDArrayView& src);
    static Axis* CreateFromProto(const proto::Axis& src);
    static NDShape* CreateFromProto(const proto::NDShape& src);

    static void Copy(const DictionaryValue& src, proto::DictionaryValue& dst, Arena* arena = nullptr);
    static void Copy(const proto::DictionaryValue& src, DictionaryValue& dst);

    static proto::NDArrayView::DataType ToProtoType(DataType type)
    {
        if (!proto::NDArrayView::DataType_IsValid((int)type))
        {
            InvalidArgument("NDArrayView::DataType is invalid.");
        }
        return proto::NDArrayView_DataType(type);
    }

    static DataType FromProtoType(proto::NDArrayView::DataType type)
    {
        if (!proto::NDArrayView::DataType_IsValid(type))
        {
            InvalidArgument("NDArrayView::DataType is invalid.");
        }
        return DataType(type);
    }

    static proto::NDArrayView::StorageFormat ToProtoType(StorageFormat type)
    {
        if (!proto::NDArrayView::StorageFormat_IsValid((int)type))
        {
            InvalidArgument("NDArrayView::StorageFormat is invalid.");
        }
        return proto::NDArrayView_StorageFormat(type);
    }

    static StorageFormat FromProtoType(proto::NDArrayView::StorageFormat type)
    {
        if (!proto::NDArrayView::StorageFormat_IsValid((int)type))
        {
            InvalidArgument("NDArrayView::StorageFormat is invalid.");
        }
        return StorageFormat(type);
    }

    static proto::DictionaryValue::Type ToProtoType(DictionaryValue::Type type)
    {
        if (!proto::DictionaryValue::Type_IsValid((int)type))
        {
            InvalidArgument("DictionaryValue::Type is invalid.");
        }
        return  proto::DictionaryValue_Type(type);
    }

    static DictionaryValue::Type FromProtoType(proto::DictionaryValue::Type type)
    {
        if (!proto::DictionaryValue::Type_IsValid((int)type))
        {
            InvalidArgument("DictionaryValue::Type is invalid.");
        }
        return DictionaryValue::Type(type);
    }

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

};

/*static*/ proto::NDShape* Serializer::CreateProto(const NDShape& src, Arena* arena)
{
    proto::NDShape* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::NDShape>(arena) : new proto::NDShape();
    auto size = src.Rank();
    dst->mutable_shape_dim()->Reserve((int)size);
    for (auto i = 0; i < size; i++)
    {
        dst->add_shape_dim(src[i]);
    }
    return dst;
}

/*static*/ NDShape* Serializer::CreateFromProto(const proto::NDShape& src)
{
    auto size = src.shape_dim_size();
    NDShape* dst = new NDShape(size);
    for (auto i = 0; i < size; i++)
    {
        dst->operator[](i) = size_t(src.shape_dim()[i]);
    }
    return dst;
}

/*static*/ proto::Axis* Serializer::CreateProto(const Axis& src, Arena* arena)
{
    proto::Axis* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::Axis>(arena) : new proto::Axis();
    dst->set_static_axis_idx(src.StaticAxisIndex(false));
    dst->set_name(ToString(src.Name()));
    dst->set_is_ordered_dynamic_axis(src.IsOrdered());
    return dst;
}

/*static*/ Axis* Serializer::CreateFromProto(const proto::Axis& src)
{
    if (!Axis(src.static_axis_idx()).IsDynamicAxis())
    {
        return new Axis(src.static_axis_idx());
    }
    else
    {
        return new Axis(ToWString(src.name()), src.is_ordered_dynamic_axis());
    }
}

/*static*/ proto::NDArrayView* Serializer::CreateProto(const NDArrayView& src, Arena* arena)
{
    proto::NDArrayView* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::NDArrayView>(arena) : new proto::NDArrayView();
    dst->set_data_type(ToProtoType(src.GetDataType()));
    dst->set_allocated_shape(CreateProto(src.Shape(), arena));
    dst->set_storage_format(ToProtoType(src.GetStorageFormat()));
    if (src.GetDataType() == DataType::Float)
    {
        CopyData<float>(src, dst->mutable_float_values()->mutable_value());
    }
    else if (src.GetDataType() == DataType::Double)
    {
        CopyData<double>(src, dst->mutable_double_values()->mutable_value());
    }
    return dst;
}

/*static*/ NDArrayView* Serializer::CreateFromProto(const proto::NDArrayView& src)
{
    if (!proto::NDArrayView::DataType_IsValid(src.data_type()) ||
        !proto::NDArrayView::StorageFormat_IsValid(src.storage_format()))
    {
        return nullptr;
    }

    std::unique_ptr<NDShape> shape(CreateFromProto(src.shape()));
    auto dataType = FromProtoType(src.data_type());
    auto storageFormat = FromProtoType(src.storage_format());
    NDArrayView* dst = new NDArrayView(dataType, storageFormat, *shape, DeviceDescriptor::CPUDevice());

    if (dataType == DataType::Float)
    {
        CopyData<float>(src.float_values().value(), dst);
    }
    else if (dataType == DataType::Double)
    {
        CopyData<double>(src.double_values().value(), dst);
    }
    return dst;
}

/*static*/ proto::Vector* Serializer::CreateProto(const std::vector<DictionaryValue>& src, Arena* arena)
{
    proto::Vector* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::Vector>(arena) : new proto::Vector();
    dst->mutable_value()->Reserve((int)src.size());
    for (const auto& value : src)
    {
        dst->mutable_value()->AddAllocated(CreateProto(value, arena));
    }
    return dst;
}

/*static*/ std::vector<DictionaryValue>* Serializer::CreateFromProto(const proto::Vector& src)
{
    std::vector<DictionaryValue>* dst = new std::vector<DictionaryValue>(src.value_size());
    for (auto i = 0; i < src.value_size(); ++i)
    {
        Copy(src.value()[i], dst->at(i));
    }
    return dst;
}

/*static*/ proto::Dictionary* Serializer::CreateProto(const Dictionary& src, Arena* arena)
{
    proto::Dictionary* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::Dictionary>(arena) : new proto::Dictionary();
    dst->set_version(src.s_version);
    for (const auto& kv : src)
    {
        Copy(kv.second, dst->mutable_data()->operator[](ToString(kv.first)), arena);
    }
    return dst;
}

/*static*/ Dictionary* Serializer::CreateFromProto(const proto::Dictionary& src)
{
    Dictionary* dst = new Dictionary();
    for (const auto& kv : src.data())
    {
        Copy(kv.second, dst->operator[](ToWString(kv.first)));
    }
    return dst;
}

/*static*/ proto::DictionaryValue* Serializer::CreateProto(const DictionaryValue& src, Arena* arena)
{
    proto::DictionaryValue* dst = (arena != nullptr) ?
        Arena::CreateMessage<proto::DictionaryValue>(arena) : new proto::DictionaryValue();
    dst->set_version(src.s_version);
    Copy(src, *dst, arena);
    return dst;
}

/*static*/ void Serializer::Copy(const DictionaryValue& src, proto::DictionaryValue& dst, Arena* arena)
{
    auto valueType = src.ValueType();
    dst.set_value_type(ToProtoType(valueType));
    dst.set_version(src.s_version);
    switch (valueType)
    {
    case DictionaryValue::Type::None:
        break;
    case DictionaryValue::Type::Bool:
        dst.set_bool_value(src.Value<bool>());
        break;
    case DictionaryValue::Type::Int:
        dst.set_int_value(src.Value<int>());
        break;
    case DictionaryValue::Type::SizeT:
        dst.set_size_t_value(src.Value<size_t>());
        break;
    case DictionaryValue::Type::Float:
        dst.set_float_value(src.Value<float>());
        break;
    case DictionaryValue::Type::Double:
        dst.set_double_value(src.Value<double>());
        break;
    case DictionaryValue::Type::String:
        dst.set_string_value(ToString(src.Value<std::wstring>()));
        break;
    case DictionaryValue::Type::NDShape:
        dst.set_allocated_nd_shape_value(CreateProto(src.Value<NDShape>(), arena));
        break;
    case DictionaryValue::Type::Axis:
        dst.set_allocated_axis_value(CreateProto(src.Value<Axis>(), arena));
        break;
    case DictionaryValue::Type::Vector:
        dst.set_allocated_vector_value(CreateProto(src.Value<std::vector<DictionaryValue>>(), arena));
        break;
    case DictionaryValue::Type::Dictionary:
        dst.set_allocated_dictionary_value(CreateProto(src.Value<Dictionary>(), arena));
        break;
    case DictionaryValue::Type::NDArrayView:
        dst.set_allocated_nd_array_view_value(CreateProto(src.Value<NDArrayView>(), arena));
        break;
    default:
        NOT_IMPLEMENTED
    }
}

/*static*/ void Serializer::Copy(const proto::DictionaryValue& src, DictionaryValue& dst)
{
    auto valueType = src.value_type();

    if (!proto::DictionaryValue::Type_IsValid(valueType))
    {
        return;
    }

    dst.m_valueType = FromProtoType(valueType);
    switch (valueType)
    {
    case proto::DictionaryValue::None:
        break;
    case proto::DictionaryValue::Bool:
        dst.m_data.m_boolean = src.bool_value();
        break;
    case proto::DictionaryValue::Int:
        dst.m_data.m_int = src.int_value();
        break;
    case proto::DictionaryValue::SizeT:
        dst.m_data.m_sizeT = src.size_t_value();
        break;
    case proto::DictionaryValue::Float:
        dst.m_data.m_float = src.float_value();
        break;
    case proto::DictionaryValue::Double:
        dst.m_data.m_double = src.double_value();
        break;
    case proto::DictionaryValue::String:
        dst.m_data.m_ptr = new std::wstring(ToWString(src.string_value()));
        break;
    case proto::DictionaryValue::NDShape:
        dst.m_data.m_ptr = CreateFromProto(src.nd_shape_value());
        break;
    case proto::DictionaryValue::Axis:
        dst.m_data.m_ptr = CreateFromProto(src.axis_value());
        break;
    case proto::DictionaryValue::Vector:
        dst.m_data.m_ptr = CreateFromProto(src.vector_value());
        break;
    case proto::DictionaryValue::Dictionary:
        dst.m_data.m_ptr = CreateFromProto(src.dictionary_value());
        break;
    case proto::DictionaryValue::NDArrayView:
        dst.m_data.m_ptr = CreateFromProto(src.nd_array_view_value());
        break;
    }
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

#endif