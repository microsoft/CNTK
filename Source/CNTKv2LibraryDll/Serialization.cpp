//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <limits>

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510)
#include "CNTK.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#pragma warning(pop)

#if defined(_MSC_VER) || defined(_CODECVT_H)
#include <codecvt>
#else
#include <cstdlib>
#include <clocale>
#endif

namespace CNTK
{
    class Serializer
    {
        friend std::ostream& operator<<(std::ostream&, const Dictionary&);
        friend std::istream& operator>>(std::istream&, Dictionary&);
        friend std::ostream& operator<<(std::ostream&, const DictionaryValue&);
        friend std::istream& operator>>(std::istream&, DictionaryValue&);

    private:
        static proto::DictionaryValue* CreateProto(const DictionaryValue& src);
        static proto::Dictionary* CreateProto(const Dictionary& src);
        static proto::Vector* CreateProto(const std::vector<DictionaryValue>& src);
        static proto::NDArrayView* CreateProto(const NDArrayView& src);
        static proto::Axis* CreateProto(const Axis& src);
        static proto::NDShape* CreateProto(const NDShape& src);

        static Dictionary* CreateFromProto(const proto::Dictionary& src);
        static std::vector<DictionaryValue>* CreateFromProto(const proto::Vector& src);
        static NDArrayView* CreateFromProto(const proto::NDArrayView& src);
        static Axis* CreateFromProto(const proto::Axis& src);
        static NDShape* CreateFromProto(const proto::NDShape& src);

        static void Copy(const DictionaryValue& src, proto::DictionaryValue& dst);
        static void Copy(const proto::DictionaryValue& src, DictionaryValue& dst);

        static std::string ToString(const std::wstring& wstring)
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

        static std::wstring ToWString(const std::string& string)
        {
#ifdef _MSC_VER
            std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
            return converter.from_bytes(string);
#else
            const auto length = string.length() + 1;
            wchar_t buf[length];
            const auto res = std::mbstowcs(buf, string.c_str(),  sizeof(buf));
            return (res >= 0) ? buf : L"";
#endif
        }

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
        static void CopyData(const NDArrayView& src, ::google::protobuf::RepeatedField<T>* dst)
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
        static void CopyData(const ::google::protobuf::RepeatedField<T>& src, NDArrayView* dst)
        {
            auto size = src.size();
            assert(size == dst->Shape().TotalSize());;
            T* buffer = dst->WritableDataBuffer<T>();
            memcpy(buffer, src.data(), size * sizeof(T));
        }

    };

    // TODO: use arenas for message allocations
    /*static*/ proto::NDShape* Serializer::CreateProto(const NDShape& src)
    {
        proto::NDShape* dst = new proto::NDShape();
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

    /*static*/ proto::Axis* Serializer::CreateProto(const Axis& src)
    {
        proto::Axis* dst = new proto::Axis();
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

    /*static*/ proto::NDArrayView* Serializer::CreateProto(const NDArrayView& src)
    {
        proto::NDArrayView* dst = new proto::NDArrayView();
        dst->set_data_type(ToProtoType(src.GetDataType()));
        dst->set_allocated_shape(CreateProto(src.Shape()));
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

    /*static*/ proto::Vector* Serializer::CreateProto(const std::vector<DictionaryValue>& src)
    {
        proto::Vector* dst = new proto::Vector();
        dst->mutable_value()->Reserve((int)src.size());
        for (const auto& value : src)
        {
            dst->mutable_value()->AddAllocated(CreateProto(value));
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

    /*static*/ proto::Dictionary* Serializer::CreateProto(const Dictionary& src)
    {
        proto::Dictionary* dst = new proto::Dictionary();
        dst->set_version(src.s_version);
        for (const auto& kv : src)
        {
            Copy(kv.second, dst->mutable_data()->operator[](ToString(kv.first)));
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

    /*static*/ proto::DictionaryValue* Serializer::CreateProto(const DictionaryValue& src)
    {
        proto::DictionaryValue* dst = new proto::DictionaryValue();
        dst->set_version(src.s_version);
        Copy(src, *dst);
        return dst;
    }

    /*static*/ void Serializer::Copy(const DictionaryValue& src, proto::DictionaryValue& dst)
    {
        auto valueType = src.ValueType();
        dst.set_value_type(ToProtoType(valueType));
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
            dst.set_allocated_nd_shape_value(CreateProto(src.Value<NDShape>()));
            break;
        case DictionaryValue::Type::Axis:
            dst.set_allocated_axis_value(CreateProto(src.Value<Axis>()));
            break;
        case DictionaryValue::Type::Vector:
            dst.set_allocated_vector_value(CreateProto(src.Value<std::vector<DictionaryValue>>()));
            break;
        case DictionaryValue::Type::Dictionary:
            dst.set_allocated_dictionary_value(CreateProto(src.Value<Dictionary>()));
            break;
        case DictionaryValue::Type::NDArrayView:
            dst.set_allocated_nd_array_view_value(CreateProto(src.Value<NDArrayView>()));
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
   

    std::istream& operator>>(std::istream& stream, ::google::protobuf::Message& msg)
    {
        google::protobuf::io::IstreamInputStream isistream(&stream);
        google::protobuf::io::CodedInputStream input(&isistream);
        input.SetTotalBytesLimit(INT_MAX, INT_MAX);
        msg.ParseFromCodedStream(&input);
        return stream;
    }

    // TODO: Add read/write to/from file and use FileInput/OutputStream
    std::ostream& operator<<(std::ostream& stream, const Dictionary& dictionary)
    {
        SetUTF8Locale();
        std::unique_ptr<proto::Dictionary> proto(Serializer::CreateProto(dictionary));
        proto->SerializeToOstream(&stream);
        UnsetUTF8Locale();
        return stream;
    }

    std::istream& operator>>(std::istream& stream, Dictionary& dictionary)
    {
        SetUTF8Locale();
        proto::Dictionary proto;
        stream >> proto;
        dictionary.m_dictionaryData->reserve(proto.data_size());
        for (const auto& kv : proto.data())
        {
            Serializer::Copy(kv.second, dictionary[Serializer::ToWString(kv.first)]);
        }
        UnsetUTF8Locale();
        return stream;
    }

    std::ostream& operator<<(std::ostream& stream, const DictionaryValue& value)
    {
        SetUTF8Locale();
        std::unique_ptr<proto::DictionaryValue> proto(Serializer::CreateProto(value));
        proto->SerializeToOstream(&stream);
        UnsetUTF8Locale();
        return stream;
    }

    std::istream& operator>>(std::istream& stream, DictionaryValue& value)
    {
        SetUTF8Locale();
        proto::DictionaryValue proto;
        stream >> proto;
        Serializer::Copy(proto, value);
        UnsetUTF8Locale();
        return stream;
    }
}