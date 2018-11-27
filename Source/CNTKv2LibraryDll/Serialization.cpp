//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include <istream>
#include <ostream>
#include <string>
#include <vector>
#include <limits>

#ifdef _MSC_VER
#include <io.h>
#endif

#pragma warning(push)
#pragma warning(disable : 4800 4267 4610 4512 4100 4510 4505)
#include "CNTK.pb.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/arena.h>
#pragma warning(pop)

namespace CNTK
{

    using namespace ::google::protobuf;

    static const uint32 MAGIC_NUMBER = 0x636e746bU;
    static const uint32 BLOCK_SIZE = 8 << 10; // 8Kb;

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

    // Variation on EncodeFloat/EncodeDouble/DecodeFloat/DecodeDouble from
    // ::google::protobuf::internal::WireFormatLite.
    template <typename SRC, typename DST>
    inline DST Encode(SRC value) 
    {
        union { SRC src; DST dst; };
        src = value;
        return dst;
    }
    
    class RenewableCodedStream 
    {
    public:
        RenewableCodedStream(io::ZeroCopyInputStream& input) 
            : m_input(input)
        {
            Renew();
        }

        template <typename T>
        inline bool Read(T* value) 
        {
            auto size = sizeof(T);

            if (m_codedInputPtr->CurrentPosition() > INT_MAX - size)
                Renew();
            
            if (size == sizeof(uint32)) {
                uint32 tmp;
                if (!m_codedInputPtr->ReadLittleEndian32(&tmp))
                    return false;
                *value = Encode<uint32, T>(tmp);
                return true;
            }

            if (size == sizeof(uint64)) {
                uint64 tmp;
                if (!m_codedInputPtr->ReadLittleEndian64(&tmp))
                    return false;
                *value = Encode<uint64, T>(tmp);
                return true;
            }

            return false;
        }

    private:
        void Renew()
        {
            delete m_codedInputPtr.release();
            m_codedInputPtr = make_unique<io::CodedInputStream>(&m_input);
            m_codedInputPtr->SetTotalBytesLimit(INT_MAX, INT_MAX);
        }

        io::ZeroCopyInputStream& m_input;
        std::unique_ptr<io::CodedInputStream> m_codedInputPtr;
    };


    class Serializer
    {
        friend std::ostream& operator<<(std::ostream&, const Dictionary&);
        friend std::ostream& operator<<(std::ostream&, const DictionaryValue&);

        friend std::istream& operator>>(std::istream&, Dictionary&);
        friend std::istream& operator>>(std::istream&, DictionaryValue&);

        friend class Dictionary;
        friend class DictionaryValue;

        Serializer(const Dictionary& dict);
        Serializer(const DictionaryValue& dict);

        Serializer() = default;

    private:
        proto::DictionaryValue* CreateProto(const DictionaryValue& src, Arena* arena = nullptr);
        proto::Dictionary* CreateProto(const Dictionary& src, Arena* arena = nullptr);
        proto::Vector* CreateProto(const std::vector<DictionaryValue>& src, Arena* arena = nullptr);
        proto::NDArrayView* CreateProto(const NDArrayView& src, Arena* arena = nullptr);
        proto::Axis* CreateProto(const Axis& src, Arena* arena = nullptr);
        proto::NDShape* CreateProto(const NDShape& src, Arena* arena = nullptr);

        void Copy(const DictionaryValue& src, proto::DictionaryValue& dst, Arena* arena = nullptr);

        void CopyNDArrayViewDataToProtos();
        void WriteNDArrayViewData(io::CodedOutputStream& output);

        std::ostream& Write(std::ostream& stream);
        void Write(const std::wstring& filename);
        void Write(io::ZeroCopyOutputStream& stream);


        bool Read(std::istream& stream, Dictionary& dict);
        bool Read(std::istream& stream, DictionaryValue& value);

        bool Read(const std::wstring& filename, Dictionary& dict);
        bool Read(const std::wstring& filename, DictionaryValue& value);

        bool Read(std::wstring filename, const std::function<bool(io::ZeroCopyInputStream& input)>& callback);
        bool Read(std::istream& stream, const std::function<bool(io::ZeroCopyInputStream& input)>& callback);

        bool ReadNDArrayViewData(io::ZeroCopyInputStream& input);

        size_t GetTotalByteSize()
        {
            return m_byteSize + m_proto->ByteSizeLong();
        }

        bool FitsIntoProtobuf()
        {
            return GetTotalByteSize() < static_cast<size_t>(INT_MAX);
        }

        Dictionary* CreateFromProto(const proto::Dictionary& src);
        std::vector<DictionaryValue>* CreateFromProto(const proto::Vector& src);
        NDArrayView* CreateFromProto(const proto::NDArrayView& src);
        Axis* CreateFromProto(const proto::Axis& src);
        NDShape* CreateFromProto(const proto::NDShape& src);

        void Copy(const proto::Dictionary& src, Dictionary& dst);
        void Copy(const proto::DictionaryValue& src, DictionaryValue& dst);

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

        template <typename SrcT, typename DstT = SrcT>
        static void CopyData(const NDArrayView& src, RepeatedField<DstT>* dst)
        {
            auto size = src.Shape().TotalSize();
            dst->Resize((int)size, DstT());
            const SrcT* buffer = src.DataBuffer<SrcT>();
            if (std::is_same<SrcT, DstT>::value)
                memcpy(dst->mutable_data(), buffer, (int)size * sizeof(DstT));
            else
                for (size_t i = 0; i < size; i++)
                    dst->mutable_data()[i] = (DstT)buffer[i];
        }

        static void WriteInt8Data(const NDArrayView& src, io::CodedOutputStream& output)
        {
            // Write raw bytes.
            auto size = src.Shape().TotalSize();
            const int8_t* buffer = src.DataBuffer<int8_t>();
            output.WriteRaw(buffer, size);
        }

        static void WriteInt16Data(const NDArrayView& src, io::CodedOutputStream& output)
        {
            auto size = src.Shape().TotalSize();
            const int16_t* buffer = src.DataBuffer<int16_t>();
            for (auto i = 0; i < size; i++)
            {
                auto value = buffer[i];
                output.WriteVarint32SignExtended(Encode<int16_t, int16_t>(value));
            }
        }

        template <typename T>
        static void WriteData(const NDArrayView& src, io::CodedOutputStream& output)
        {
            auto size = src.Shape().TotalSize();
            const T* buffer = src.DataBuffer<T>();
            auto tSize = sizeof(T);
            for (auto i = 0; i < size; i++) 
            {
                auto value = buffer[i];
                if (tSize <= sizeof(uint32))
                {
                    output.WriteLittleEndian32(Encode<T, uint32>((float)value));
                }
                else
                {
                    output.WriteLittleEndian64(Encode<T, uint64>(value));
                }
            }
        }

        template <typename SrcT, typename DstT = SrcT>
        static bool ReadData(RenewableCodedStream& input, NDArrayView& dst)
        {
            auto size = dst.Shape().TotalSize();
            DstT* buffer = dst.WritableDataBuffer<DstT>();
            for (auto i = 0; i < size; i++)
            {
                SrcT value;
                if (!input.Read<SrcT>(&value))
                    return false;
                buffer[i] = (DstT)value;
            }
            return true;
        }

        static bool ReadInt8Data(io::ZeroCopyInputStream& input, NDArrayView& dst)
        {
            const void* temp;
            int readSize;
            size_t totalSize = 0;
            bool success;
            do {
                success = input.Next(&temp, &readSize);
                totalSize += readSize;
            } while (success && readSize == 0);

            if (!success)
                return false;
            
            auto size = dst.Shape().TotalSize();
            if (totalSize != size)
                return false;

            int8_t* buffer = dst.WritableDataBuffer<int8_t>();
            memcpy(buffer, temp, size);
            return true;
        }

        template <typename SrcT, typename DstT = SrcT>
        static void CopyData(const RepeatedField<SrcT>& src, NDArrayView* dst)
        {
            auto size = src.size();
            assert(size == dst->Shape().TotalSize());;
            DstT* buffer = dst->WritableDataBuffer<DstT>();
            if (std::is_same<SrcT, DstT>::value)
                memcpy(buffer, src.data(), size * sizeof(SrcT));
            else
            {
                for (size_t i = 0; i < size; i++)
                    buffer[i] = (DstT)src.data()[i];
            }
        }

        static void CopyInt8Data(const std::string& src, NDArrayView* dst)
        {
            auto size = src.length();
            assert(size == dst->Shape().TotalSize());
            auto* buffer = dst->WritableDataBuffer<int8_t>();
            memcpy(buffer, src.data(), size * sizeof(int8_t));
        }

        UsingUTF8 m_locale;
        Arena m_arena;
        Message* m_proto;
        std::vector<std::pair<NDArrayView*, proto::NDArrayView*>> m_arrayViews;
        size_t m_byteSize {0};
    };


    Serializer::Serializer(const Dictionary& dict) 
    {
        m_proto = CreateProto(dict, &m_arena);
    }

    Serializer::Serializer(const DictionaryValue& value)
    {
        m_proto = CreateProto(value, &m_arena);
    }
  
    void Serializer::CopyNDArrayViewDataToProtos()
    {
        for (auto& pair : m_arrayViews) 
        {
            const auto& src = *(pair.first);
            auto dst = pair.second;
            if (src.GetDataType() == DataType::Float)
            {
                CopyData<float>(src, dst->mutable_float_values()->mutable_value());
            }
            else if (src.GetDataType() == DataType::Double)
            {
                CopyData<double>(src, dst->mutable_double_values()->mutable_value());
            }
            else if (src.GetDataType() == DataType::Float16)
            {
                CopyData<float16, float>(src, dst->mutable_float_values()->mutable_value());
            }
            else if (src.GetDataType() == DataType::Int8)
            {
                // Directly copy the data as a byte array.
                auto size = src.Shape().TotalSize();
                const int8_t* buffer = src.DataBuffer<int8_t>();
                dst->mutable_bytes_value()->set_value(buffer, size);
            }
            else if (src.GetDataType() == DataType::Int16)
            {
               CopyData<int16_t, int32>(src, dst->mutable_sint32_values()->mutable_value());
            }
        }
    }

    void Serializer::WriteNDArrayViewData(io::CodedOutputStream& output) 
    {
        for (auto& pair : m_arrayViews)
        {
            const auto& src = *(pair.first);
            if (src.GetDataType() == DataType::Float)
            {
                WriteData<float>(src, output);
            }
            else if (src.GetDataType() == DataType::Double)
            {
                WriteData<double>(src, output);
            }
            else if (src.GetDataType() == DataType::Float16)
            {
                WriteData<float16>(src, output);
            }
            else if (src.GetDataType() == DataType::Int8)
            {
                WriteInt8Data(src, output);
            }
            else if (src.GetDataType() == DataType::Int16)
            {
                WriteInt16Data(src, output);
            }
        }
    }

    bool Serializer::ReadNDArrayViewData(io::ZeroCopyInputStream& input)
    {
        if (m_arrayViews.size() == 0)
            return true;

        RenewableCodedStream wrapper(input);
        for (auto& pair : m_arrayViews)
        {
            auto& dst = *(pair.first);
            if (dst.GetDataType() == DataType::Float)
            {
                if (!ReadData<float>(wrapper, dst))
                    return false;
            }
            else if (dst.GetDataType() == DataType::Double)
            {
                if (!ReadData<double>(wrapper, dst))
                    return false;
            }
            else if (dst.GetDataType() == DataType::Float16)
            {
                if (!ReadData<float, float16>(wrapper, dst))
                    return false;
            }
            else if (dst.GetDataType() == DataType::Int8)
            {
                if (!ReadInt8Data(input, dst))
                    return false;
            }
            else if (dst.GetDataType() == DataType::Int16)
            {
                if (!ReadData<int16_t, int16_t>(wrapper, dst))
                     return false;
            }
        }
        return true;
    }

    proto::NDShape* Serializer::CreateProto(const NDShape& src, Arena* arena)
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

    NDShape* Serializer::CreateFromProto(const proto::NDShape& src)
    {
        auto size = src.shape_dim_size();
        NDShape* dst = new NDShape(size);
        for (auto i = 0; i < size; i++)
        {
            dst->operator[](i) = size_t(src.shape_dim()[i]);
        }
        return dst;
    }

    proto::Axis* Serializer::CreateProto(const Axis& src, Arena* arena)
    {
        proto::Axis* dst = (arena != nullptr) ? 
            Arena::CreateMessage<proto::Axis>(arena) : new proto::Axis();
        dst->set_static_axis_idx(src.StaticAxisIndex(false));
        dst->set_name(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(src.Name())));
        dst->set_is_ordered_dynamic_axis(src.IsOrdered());
        return dst;
    }

    Axis* Serializer::CreateFromProto(const proto::Axis& src)
    {
        if (!Axis(src.static_axis_idx()).IsDynamicAxis())
        {
            return new Axis(src.static_axis_idx());
        }
        else
        {
            return new Axis(Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(src.name()), src.is_ordered_dynamic_axis());
        }
    }

    proto::NDArrayView* Serializer::CreateProto(const NDArrayView& src, Arena* arena)
    {
        proto::NDArrayView* dst = (arena != nullptr) ? 
            Arena::CreateMessage<proto::NDArrayView>(arena) : new proto::NDArrayView();
        dst->set_data_type(ToProtoType(src.GetDataType()));
        dst->set_allocated_shape(CreateProto(src.Shape(), arena));
        dst->set_storage_format(ToProtoType(src.GetStorageFormat()));

        m_arrayViews.push_back({const_cast<NDArrayView*>(&src), dst });
        
        auto numElements = src.Shape().TotalSize();
        auto dataSize = DataTypeSize(src.GetDataType());
        if (numElements > SIZE_MAX / dataSize) 
            RuntimeError("Bytes size of NDArrayView exceeds %zu.", SIZE_MAX);
        m_byteSize += numElements * dataSize;

        return dst;
    }

    NDArrayView* Serializer::CreateFromProto(const proto::NDArrayView& src)
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
            if (src.float_values().value().size() == shape->TotalSize())
                CopyData<float>(src.float_values().value(), dst);
            else 
                m_arrayViews.push_back({ dst, nullptr });
        }
        else if (dataType == DataType::Double)
        {
            if (src.double_values().value().size() == shape->TotalSize())
                CopyData<double>(src.double_values().value(), dst);
            else
                m_arrayViews.push_back({ dst, nullptr });
        }
        else if(dataType == DataType::Float16)
        {
            if (src.float_values().value().size() == shape->TotalSize())
                CopyData<float, float16>(src.float_values().value(), dst);
            else
                m_arrayViews.push_back({ dst, nullptr });
        }
        else if (dataType == DataType::Int8)
        {
            if (src.bytes_value().value().size() == shape->TotalSize())
                CopyInt8Data(src.bytes_value().value(), dst);
            else
                m_arrayViews.push_back({ dst, nullptr });
        }
        else if (dataType == DataType::Int16)
        {
            if (src.sint32_values().value().size() == shape->TotalSize())
                 CopyData<int32, int16_t>(src.sint32_values().value(), dst);
            else
                 m_arrayViews.push_back({ dst, nullptr });
        }
        return dst;
    }

    proto::Vector* Serializer::CreateProto(const std::vector<DictionaryValue>& src, Arena* arena)
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

    std::vector<DictionaryValue>* Serializer::CreateFromProto(const proto::Vector& src)
    {
        std::vector<DictionaryValue>* dst = new std::vector<DictionaryValue>(src.value_size());
        for (auto i = 0; i < src.value_size(); ++i)
        {
            Copy(src.value()[i], dst->at(i));
        }
        return dst;
    }

    proto::Dictionary* Serializer::CreateProto(const Dictionary& src, Arena* arena)
    {
        proto::Dictionary* dst = (arena != nullptr) ? 
            Arena::CreateMessage<proto::Dictionary>(arena) : new proto::Dictionary();
        dst->set_version(src.s_version);
        for (const auto& kv : src)
        {
            Copy(kv.second, dst->mutable_data()->operator[](Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(kv.first))), arena);
        }
        return dst;
    }

    Dictionary* Serializer::CreateFromProto(const proto::Dictionary& src)
    {
        Dictionary* dst = new Dictionary();
        for (const auto& kv : src.data())
        {
            Copy(kv.second, dst->operator[](Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(kv.first)));
        }
        return dst;
    }

    proto::DictionaryValue* Serializer::CreateProto(const DictionaryValue& src, Arena* arena)
    {
        proto::DictionaryValue* dst = (arena != nullptr) ? 
            Arena::CreateMessage<proto::DictionaryValue>(arena) : new proto::DictionaryValue();
        dst->set_version(src.s_version);
        Copy(src, *dst, arena);
        return dst;
    }

    void Serializer::Copy(const DictionaryValue& src, proto::DictionaryValue& dst, Arena* arena)
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
            dst.set_string_value(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(src.Value<std::wstring>())));
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

    void Serializer::Copy(const proto::Dictionary& src, Dictionary& dst)
    {
        dst.m_dictionaryData->reserve(src.data_size());
        for (const auto& kv : src.data())
        {
            Serializer::Copy(kv.second, dst[Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(kv.first)]);
        }
    }

    void Serializer::Copy(const proto::DictionaryValue& src, DictionaryValue& dst)
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
            dst.m_data.m_ptr = new std::wstring(Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(src.string_value()));
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

    void Serializer::Write(io::ZeroCopyOutputStream& stream) {
        io::CodedOutputStream output(&stream);

        // Protobufs have a hard limit on the maximum message size(INT_MAX = 2GBs). 
        // Check if we fit into a single protobuf message.
        if (FitsIntoProtobuf())
        {
            CopyNDArrayViewDataToProtos();
            m_proto->SerializeToCodedStream(&output);
        }
        else
        {
            // If we don't, pull the metadata apart from the actual payload (NDArrayView content)
            // and store the payload separately, outside of the protobuf.
            // Prefix the metadata protobuf with a magic number and its bytes size.
            output.WriteLittleEndian32(MAGIC_NUMBER);
            output.WriteLittleEndian32(m_proto->ByteSize());
            m_proto->SerializeToCodedStream(&output);
            WriteNDArrayViewData(output);
        }
    }

    std::ostream& Serializer::Write(std::ostream& stream)
    {
        io::OstreamOutputStream output(&stream);
        Write(output);
        return stream;
    }

    void Serializer::Write(const std::wstring& filename)
    {
        auto fd = GetFileDescriptor(filename, false);
        {
            io::FileOutputStream output(fd);
            Write(output);
        }
#ifdef _MSC_VER
        _close(fd);
#else
        close(fd);
#endif
    }

    bool ParseMessage(io::ZeroCopyInputStream& input, Message& msg)
    {
        uint32 prefix = 0, limit = INT_MAX;;
        const void* temp;
        int size;
        bool success;
        do {
            success = input.Next(&temp, &size);
        } while (success && size == 0);
    
        if (!success)
            return false;
        
        if (size >= (sizeof(prefix) + sizeof(limit))) 
        {
            io::CodedInputStream::ReadLittleEndian32FromArray(reinterpret_cast<const uint8*>(temp), &prefix);
        }

        // the message is only prefixed with a magic number + message length,
        // if its size exceeds 2GBs.
        if (prefix == MAGIC_NUMBER) 
        {
            io::CodedInputStream::ReadLittleEndian32FromArray(
                reinterpret_cast<const uint8*>(temp) + sizeof(prefix), &limit);

            input.BackUp(size - sizeof(prefix) - sizeof(limit));
        }
        else 
            input.BackUp(size);

        io::CodedInputStream codedInput(&input);
        codedInput.SetTotalBytesLimit(limit, limit);
        return msg.ParseFromCodedStream(&codedInput) && codedInput.ConsumedEntireMessage();
    }

    bool Serializer::Read(std::istream& stream, Dictionary& dict)
    {
        m_proto = Arena::CreateMessage<proto::Dictionary>(&m_arena);
        return Read(stream, [this, &dict](io::ZeroCopyInputStream& input) {
            Copy(*dynamic_cast<proto::Dictionary*>(m_proto), dict);
            return ReadNDArrayViewData(input);
        });
    }

    bool Serializer::Read(std::istream& stream, DictionaryValue& value)
    {
        m_proto = Arena::CreateMessage<proto::DictionaryValue>(&m_arena);
        return Read(stream, [this, &value](io::ZeroCopyInputStream& input) {
            Copy(*dynamic_cast<proto::DictionaryValue*>(m_proto), value);
            return ReadNDArrayViewData(input);
        });
    }

    bool Serializer::Read(const std::wstring& filename, Dictionary& dict)
    {
        m_proto = Arena::CreateMessage<proto::Dictionary>(&m_arena);
        return Read(filename, [this, &dict](io::ZeroCopyInputStream& input) {
            Copy(*dynamic_cast<proto::Dictionary*>(m_proto), dict);
            return ReadNDArrayViewData(input);
        });
    }

    bool Serializer::Read(const std::wstring& filename, DictionaryValue& value)
    {
        m_proto = Arena::CreateMessage<proto::DictionaryValue>(&m_arena);
        return Read(filename, [this, &value](io::ZeroCopyInputStream& input) {
            Copy(*dynamic_cast<proto::DictionaryValue*>(m_proto), value);
            return ReadNDArrayViewData(input);
        });
    }

    bool Serializer::Read(std::wstring filename, const std::function<bool(io::ZeroCopyInputStream& input)>& callback)
    {
        bool result;
        auto fd = GetFileDescriptor(filename, true);
        {
            io::FileInputStream input(fd, BLOCK_SIZE);
            result = ParseMessage(input, *m_proto);
            result = result && callback(input);
        }
#ifdef _MSC_VER
        _close(fd);
#else
        close(fd);
#endif
        return result;
    }

    bool Serializer::Read(std::istream& stream, const std::function<bool(io::ZeroCopyInputStream& input)>& callback)
    {
        io::IstreamInputStream input(&stream, BLOCK_SIZE);
        if (ParseMessage(input, *m_proto))
        {
            return callback(input);
        }
        return false;
    }

    std::ostream& operator<<(std::ostream& stream, const Dictionary& dictionary)
    {
        return Serializer(dictionary).Write(stream);
    }

    std::ostream& operator<<(std::ostream& stream, const DictionaryValue& value)
    {
        return Serializer(value).Write(stream);
    }

    void Dictionary::Save(const std::wstring& filename)
    {
        Serializer(*this).Write(filename);
    }

    void DictionaryValue::Save(const std::wstring& filename)
    {
        Serializer(*this).Write(filename);
    }

    std::istream& operator>>(std::istream& stream, Dictionary& dictionary)
    {
        if (!Serializer(dictionary).Read(stream, dictionary)) 
            RuntimeError("Failed to parse Dictionary from the input stream.");
        return stream;
    }

    std::istream& operator>>(std::istream& stream, DictionaryValue& value)
    {
        if (!Serializer(value).Read(stream, value)) 
            RuntimeError("Failed to parse DictionaryValue from the input stream.");
        return stream;
    }

    /*static*/ Dictionary Dictionary::Load(const std::wstring& filename)
    {
        Dictionary dictionary;
        if (!Serializer().Read(filename, dictionary))
            RuntimeError("Failed to parse Dictionary from file (%ls).", filename.c_str());
        return dictionary;
    }

    /*static*/ DictionaryValue DictionaryValue::Load(const std::wstring& filename)
    {
        DictionaryValue dictionaryValue;
        if (!Serializer().Read(filename, dictionaryValue))
            RuntimeError("Failed to parse DictionaryValue from file (%ls).", filename.c_str());
        return dictionaryValue;
    }
}
