#ifndef ONNXIR_UTILS_H
#define ONNXIR_UTILS_H

#include <unordered_map>
#include <unordered_set>
#include <string>

class TensorProto;
class TypeProto;

namespace ONNXIR
{
    typedef const std::string* PTYPE;

    namespace Utils
    {
        class OpUtils
        {
        public:
            static PTYPE ToType(const TypeProto& p_type);
            static PTYPE ToType(const std::string& p_type);
            static const TypeProto& ToTypeProto(const PTYPE& p_type);
            static std::string ToString(const TypeProto& p_type);
            static std::string ToString(const TensorProto::DataType& p_type);
            static void FromString(const std::string& p_src, TypeProto& p_type);
            static void FromString(const std::string& p_src, TensorProto::DataType& p_type);
            static bool IsValidDataTypeString(const std::string &p_dataType);
        private:
            static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();
        };

        class StringRange
        {
        public:
            StringRange();
            StringRange(const char* p_data, size_t p_size);
            StringRange(const std::string& p_str);
            StringRange(const char* p_data);
            const char* Data() const;
            size_t Size() const;
            bool Empty() const;
            char operator[](size_t p_idx) const;
            void Reset();
            void Reset(const char* p_data, size_t p_size);
            void Reset(const std::string& p_str);
            bool StartsWith(const StringRange& p_str) const;
            bool EndsWith(const StringRange& p_str) const;
            bool LStrip();
            bool LStrip(size_t p_size);
            bool LStrip(StringRange p_str);
            bool RStrip();
            bool RStrip(size_t p_size);
            bool RStrip(StringRange p_str);
            bool LAndRStrip();
            size_t Find(const char p_ch) const;

        private:
            const char* m_data;
            size_t m_size;
        };
    }
}

#endif // ! ONNXIR_UTILS_H
