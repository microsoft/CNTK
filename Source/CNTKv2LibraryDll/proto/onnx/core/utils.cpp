#include <cctype>
#include <iterator>
#include <iostream>
#include <sstream>

#include "constants.h"
#include "utils.h"
#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456 4189 4996 4503)
#include "proto/onnx/protobuf/onnx-ml.pb.h"
#pragma warning(pop)

using namespace onnx;

namespace ONNXIR
{
    namespace Utils
    {
        std::unordered_map<std::string, TypeProto>& OpUtils::GetTypeStrToProtoMap()
        {
            static std::unordered_map<std::string, TypeProto> map;
            return map;
        }

        std::mutex& OpUtils::GetTypeStrLock()
        {
            static std::mutex lock;
            return lock;
        }

        PTYPE OpUtils::ToType(const TypeProto& p_type)
        {
            auto typeStr = ToString(p_type);
            std::lock_guard<std::mutex> lock(GetTypeStrLock());
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
            std::lock_guard<std::mutex> lock(GetTypeStrLock());
            auto it = GetTypeStrToProtoMap().find(*p_type);
            assert(it != GetTypeStrToProtoMap().end());
            return it->second;
        }

        std::string OpUtils::ToString(const TypeProto& p_type, const std::string& left, const std::string& right)
        {
            switch (p_type.value_case())
            {
            case TypeProto::ValueCase::kTensorType:
            {
                if (p_type.tensor_type().has_shape()
                    && p_type.tensor_type().shape().dim_size() == 0)
                {
                    // Scalar case.
                    return left + ToDataTypeString(p_type.tensor_type().elem_type()) + right;
                }
                else
                {
                    return left + "tensor(" + ToDataTypeString(p_type.tensor_type().elem_type()) + ")" + right;
                }
            }
            case TypeProto::ValueCase::kSequenceType:
                return ToString(p_type.sequence_type().elem_type(), left + "seq(", ")" + right);
            case TypeProto::ValueCase::kMapType:
            {
                std::string map_str = "map(" + ToDataTypeString(p_type.map_type().key_type()) + ",";
                return ToString(p_type.map_type().value_type(), left + map_str, ")" + right);
            }
            default:
                assert(false);
                return "";
            }
        }

        std::string OpUtils::ToDataTypeString(const TensorProto::DataType& p_type)
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

            assert(false);
            return "";
        }

        void OpUtils::FromString(const std::string& p_src, TypeProto& p_type)
        {
            StringRange s(p_src);
            p_type.Clear();

            if (s.LStrip("seq"))
            {
                s.ParensWhitespaceStrip();
                return FromString(std::string(s.Data(), s.Size()), *p_type.mutable_sequence_type()->mutable_elem_type());
            }
            else if (s.LStrip("map"))
            {
                s.ParensWhitespaceStrip();
                size_t key_size = s.Find(',');
                StringRange k(s.Data(), key_size);
                std::string key = std::string(k.Data(), k.Size());
                s.LStrip(key_size);
                s.LStrip(",");
                StringRange v(s.Data(), s.Size());
                TensorProto::DataType key_type;
                FromDataTypeString(key, key_type);
                p_type.mutable_map_type()->set_key_type(key_type);
                return FromString(std::string(v.Data(), v.Size()), *p_type.mutable_map_type()->mutable_value_type());
            }
            else if (s.LStrip("tensor"))
            {
                s.ParensWhitespaceStrip();
                TensorProto::DataType e;
                FromDataTypeString(std::string(s.Data(), s.Size()), e);
                p_type.mutable_tensor_type()->set_elem_type(e);
            }
            else
            {
                // Scalar
                TensorProto::DataType e;
                FromDataTypeString(std::string(s.Data(), s.Size()), e);
                TypeProto::Tensor* t = p_type.mutable_tensor_type();
                t->set_elem_type(e);
                // Call mutable_shape() to initialize a shape with no dimension.
                t->mutable_shape();
            }
        }

        bool OpUtils::IsValidDataTypeString(const std::string& p_dataType)
        {
            TypesWrapper& t = TypesWrapper::GetTypesWrapper();
            const auto& allowedSet = t.GetAllowedDataTypes();
            return (allowedSet.find(p_dataType) != allowedSet.end());
        }

        void OpUtils::SplitStringTokens(StringRange& p_src, std::vector<StringRange>& p_tokens)
        {
            int parens = 0;
            p_src.RestartCapture();
            while (p_src.Size() > 0)
            {
                if (p_src.StartsWith(","))
                {
                    if (parens == 0)
                    {
                        p_tokens.push_back(p_src.GetCaptured());
                        p_src.LStrip(",");
                        p_src.RestartCapture();
                    }
                    else
                    {
                        p_src.LStrip(",");
                    }
                }
                else if (p_src.LStrip("("))
                {
                    parens++;
                }
                else if (p_src.LStrip(")"))
                {
                    parens--;
                }
                else
                {
                    p_src.LStrip(1);
                }
            }
            p_tokens.push_back(p_src.GetCaptured());
        }

        void OpUtils::FromDataTypeString(const std::string& p_typeStr, TensorProto::DataType& p_type)
        {
            assert(IsValidDataTypeString(p_typeStr));

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
                assert(false);
            }
        }

        StringRange::StringRange()
            : m_data(""), m_size(0), m_start(m_data), m_end(m_data)
        {}

        StringRange::StringRange(const char* p_data, size_t p_size)
            : m_data(p_data), m_size(p_size), m_start(m_data), m_end(m_data)
        {
            assert(p_data != nullptr);
            LAndRStrip();
        }

        StringRange::StringRange(const std::string& p_str)
            : m_data(p_str.data()), m_size(p_str.size()), m_start(m_data), m_end(m_data)
        {
            LAndRStrip();
        }

        StringRange::StringRange(const char* p_data)
            : m_data(p_data), m_size(strlen(p_data)), m_start(m_data), m_end(m_data)
        {
            LAndRStrip();
        }

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
            m_start = m_end = m_data;
        }

        void StringRange::Reset(const char* p_data, size_t p_size)
        {
            m_data = p_data;
            m_size = p_size;
            m_start = m_end = m_data;
        }

        void StringRange::Reset(const std::string& p_str)
        {
            m_data = p_str.data();
            m_size = p_str.size();
            m_start = m_end = m_data;
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
                m_end += p_size;
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
            bool l = LStrip();
            bool r = RStrip();
            return l || r;
        }

        void StringRange::ParensWhitespaceStrip()
        {
            LStrip();
            LStrip("(");
            LAndRStrip();
            RStrip(")");
            RStrip();
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

        void StringRange::RestartCapture()
        {
            m_start = m_data;
            m_end = m_data;
        }

        StringRange StringRange::GetCaptured()
        {
            return StringRange(m_start, m_end - m_start);
        }
    }
}
