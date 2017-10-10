#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456)

#include <cctype>
#include <iterator>
#include <iostream>
#include <sstream>

#include "constants.h"
#include "proto/onnx/protobuf/graph.pb.h"
#include "utils.h"

namespace ONNXIR
{
    namespace Utils
    {
        std::unordered_map<std::string, TypeProto>& OpUtils::GetTypeStrToProtoMap()
        {
            static std::unordered_map<std::string, TypeProto>* typeStrToProtoMap =
                new std::unordered_map<std::string, TypeProto>();
            return *typeStrToProtoMap;
        }

        PTYPE OpUtils::ToType(const TypeProto& p_type)
        {
            auto typeStr = ToString(p_type);
            if (GetTypeStrToProtoMap().find(typeStr) == GetTypeStrToProtoMap().end())
            {
                GetTypeStrToProtoMap()[typeStr] = p_type;
            }
            return &(GetTypeStrToProtoMap().find(typeStr)->first);
        }

        PTYPE OpUtils::ToType(const std::string& p_type)
        {
            TypeProto type;
            FromString(p_type, type);
            return ToType(type);
        }

        const TypeProto& OpUtils::ToTypeProto(const PTYPE& p_type)
        {
            auto it = GetTypeStrToProtoMap().find(*p_type);
            if (it != GetTypeStrToProtoMap().end())
            {
                return it->second;
            }
            else
            {
                throw std::invalid_argument("PTYPE not found: " + *p_type);
            }
        }

        std::string OpUtils::ToString(const TypeProto& p_type)
        {
            switch (p_type.value_case())
            {
            case TypeProto::ValueCase::kTensorType:
                return ToString(p_type.tensor_type().elem_type());
            case TypeProto::ValueCase::kSparseTensorType:
                return "sparse(" + ToString(p_type.sparse_tensor_type().elem_type()) + ")";
            case TypeProto::ValueCase::kSeqType:
                return "seq(" + ToString(p_type.seq_type().elem_type()) + ")";
            case TypeProto::ValueCase::kTupleType:
            {
                int size = p_type.tuple_type().elem_type_size();
                std::string tuple_str("tuple(");
                for (int i = 0; i < size - 1; i++)
                {
                    tuple_str = tuple_str + ToString(p_type.tuple_type().elem_type(i)) + ",";
                }
                tuple_str += ToString(p_type.tuple_type().elem_type(size - 1));
                tuple_str += ")";
                return tuple_str;
            }
            case TypeProto::ValueCase::kMapType:
            {
                std::string map_str("map(");
                map_str = map_str + ToString(p_type.map_type().key_type()) + ","
                    + ToString(p_type.map_type().value_type()) + ")";
                return map_str;
            }
            case TypeProto::ValueCase::kHandleType:
                return "handle";
            default:
                throw std::invalid_argument("Unknown TypeProto");
            }
        }

        std::string OpUtils::ToString(const TensorProto::DataType& p_type)
        {
            TypesWrapper& t = TypesWrapper::GetTypesWrapper();
            switch (p_type)
            {
            case TensorProto::DataType::TensorProto_DataType_BOOL:
                return t.c_bool;
            case TensorProto::DataType::TensorProto_DataType_STRING:
                return t.c_string;
            case TensorProto::DataType::TensorProto_DataType_FLOAT16:
                return t.c_float16;
            case TensorProto::DataType::TensorProto_DataType_FLOAT:
                return t.c_float;
            case TensorProto::DataType::TensorProto_DataType_DOUBLE:
                return t.c_double;
            case TensorProto::DataType::TensorProto_DataType_INT8:
                return t.c_int8;
            case TensorProto::DataType::TensorProto_DataType_INT16:
                return t.c_int16;
            case TensorProto::DataType::TensorProto_DataType_INT32:
                return t.c_int32;
            case TensorProto::DataType::TensorProto_DataType_INT64:
                return t.c_int64;
            case TensorProto::DataType::TensorProto_DataType_UINT8:
                return t.c_uint8;
            case TensorProto::DataType::TensorProto_DataType_UINT16:
                return t.c_uint16;
            case TensorProto::DataType::TensorProto_DataType_UINT32:
                return t.c_uint32;
            case TensorProto::DataType::TensorProto_DataType_UINT64:
                return t.c_uint64;
            case TensorProto::DataType::TensorProto_DataType_COMPLEX64:
                return t.c_complex64;
            case TensorProto::DataType::TensorProto_DataType_COMPLEX128:
                return t.c_complex128;
            }

            throw std::invalid_argument("Unknown DataType");
        }


        void OpUtils::FromString(const std::string& p_src, TypeProto& p_type)
        {
            StringRange s(p_src);
            s.LAndRStrip();
            p_type.Clear();

            if (s.LStrip("seq("))
            {
                s.RStrip(")");
                FromString(std::string(s.Data(), s.Size()), *p_type.mutable_seq_type()->mutable_elem_type());
            }
            else if (s.LStrip("tuple("))
            {
                s.RStrip(")");
                std::istringstream types(std::string(s.Data(), s.Size()));
                std::string type;
                while (std::getline(types, type, ','))
                {
                    FromString(type, *p_type.mutable_tuple_type()->mutable_elem_type()->Add());
                }
            }
            else if (s.LStrip("map("))
            {
                size_t key_size = s.Find(',');
                StringRange k(s.Data(), key_size);
                std::string key = std::string(k.Data(), k.Size());
                s.LStrip(key_size);
                s.LStrip(",");
                size_t val_size = s.Find(')');
                StringRange v(s.Data(), val_size);
                std::string val = std::string(v.Data(), v.Size());

                TensorProto::DataType key_type;
                FromString(key, key_type);
                TensorProto::DataType val_type;
                FromString(val, val_type);
                p_type.mutable_map_type()->set_key_type(key_type);
                p_type.mutable_map_type()->set_value_type(val_type);
            }
            else if (s.LStrip("handle"))
            {
                p_type.mutable_handle_type();
            }
            else if (s.LStrip("sparse("))
            {
                s.RStrip(")");
                TensorProto::DataType e;
                FromString(std::string(s.Data(), s.Size()), e);
                p_type.mutable_sparse_tensor_type()->set_elem_type(e);
            }
            else
            {
                // dense tensor
                TensorProto::DataType e;
                FromString(std::string(s.Data(), s.Size()), e);
                p_type.mutable_tensor_type()->set_elem_type(e);
            }
        }

        bool OpUtils::IsValidDataTypeString(const std::string& p_dataType)
        {
            TypesWrapper& t = TypesWrapper::GetTypesWrapper();
            return (t.GetAllowedDataTypes().find(p_dataType) != t.GetAllowedDataTypes().end());
        }

        void OpUtils::FromString(const std::string& p_typeStr, TensorProto::DataType& p_type)
        {
            if (!IsValidDataTypeString(p_typeStr))
            {
                throw std::invalid_argument("Unknown DataType: " + p_typeStr);
            }

            TypesWrapper& t = TypesWrapper::GetTypesWrapper();
            if (p_typeStr == t.c_bool)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_BOOL;
            }
            else if (p_typeStr == t.c_float)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_FLOAT;
            }
            else if (p_typeStr == t.c_float16)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_FLOAT16;
            }
            else if (p_typeStr == t.c_double)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_DOUBLE;
            }
            else if (p_typeStr == t.c_int8)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT8;
            }
            else if (p_typeStr == t.c_int16)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT16;
            }
            else if (p_typeStr == t.c_int32)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT32;
            }
            else if (p_typeStr == t.c_int64)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_INT64;
            }
            else if (p_typeStr == t.c_string)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_STRING;
            }
            else if (p_typeStr == t.c_uint8)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UINT8;
            }
            else if (p_typeStr == t.c_uint16)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UINT16;
            }
            else if (p_typeStr == t.c_uint32)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UINT32;
            }
            else if (p_typeStr == t.c_uint64)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UINT64;
            }
            else if (p_typeStr == t.c_complex64)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_COMPLEX64;
            }
            else if (p_typeStr == t.c_complex128)
            {
                p_type = TensorProto::DataType::TensorProto_DataType_COMPLEX128;
            }
            else
            {
                p_type = TensorProto::DataType::TensorProto_DataType_UNDEFINED;
            }
        }

        StringRange::StringRange()
            : m_data(""), m_size(0)
        {}

        StringRange::StringRange(const char* p_data, size_t p_size)
            : m_data(p_data), m_size(p_size)
        {}

        StringRange::StringRange(const std::string& p_str)
            : m_data(p_str.data()), m_size(p_str.size())
        {}

        StringRange::StringRange(const char* p_data)
            : m_data(p_data), m_size(strlen(p_data))
        {}

        const char* StringRange::Data() const
        {
            return m_data;
        }

        size_t StringRange::Size() const
        {
            return m_size;
        }

        bool StringRange::Empty() const
        {
            return m_size == 0;
        }

        char StringRange::operator[](size_t p_idx) const
        {
            return m_data[p_idx];
        }

        void StringRange::Reset()
        {
            m_data = "";
            m_size = 0;
        }

        void StringRange::Reset(const char* p_data, size_t p_size)
        {
            m_data = p_data;
            m_size = p_size;
        }

        void StringRange::Reset(const std::string& p_str)
        {
            m_data = p_str.data();
            m_size = p_str.size();
        }

        bool StringRange::StartsWith(const StringRange& p_str) const
        {
            return ((m_size >= p_str.m_size) && (memcmp(m_data, p_str.m_data, p_str.m_size) == 0));
        }

        bool StringRange::EndsWith(const StringRange& p_str) const
        {
            return ((m_size >= p_str.m_size) &&
                (memcmp(m_data + (m_size - p_str.m_size), p_str.m_data, p_str.m_size) == 0));
        }

        bool StringRange::LStrip() {
            size_t count = 0;
            const char* ptr = m_data;
            while (count < m_size && isspace(*ptr)) {
                count++;
                ptr++;
            }

            if (count > 0)
            {
                return LStrip(count);
            }
            return false;
        }
        bool StringRange::LStrip(size_t p_size)
        {
            if (p_size <= m_size)
            {
                m_data += p_size;
                m_size -= p_size;
                return true;
            }
            return false;
        }

        bool StringRange::LStrip(StringRange p_str)
        {
            if (StartsWith(p_str)) {
                return LStrip(p_str.m_size);
            }
            return false;
        }

        bool StringRange::RStrip() {
            size_t count = 0;
            const char* ptr = m_data + m_size - 1;
            while (count < m_size && isspace(*ptr)) {
                ++count;
                --ptr;
            }

            if (count > 0)
            {
                return RStrip(count);
            }
            return false;
        }

        bool StringRange::RStrip(size_t p_size)
        {
            if (m_size >= p_size)
            {
                m_size -= p_size;
                return true;
            }
            return false;
        }

        bool StringRange::RStrip(StringRange p_str)
        {
            if (EndsWith(p_str)) {
                return RStrip(p_str.m_size);
            }
            return false;
        }

        bool StringRange::LAndRStrip()
        {
            return LStrip() || RStrip();
        }

        size_t StringRange::Find(const char p_ch) const
        {
            size_t idx = 0;
            while (idx < m_size)
            {
                if (m_data[idx] == p_ch)
                {
                    return idx;
                }
                idx++;
            }
            return std::string::npos;
        }
    }
}

#pragma warning(pop)