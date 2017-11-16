#ifndef ONNXIR_UTILS_H
#define ONNXIR_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#pragma warning(push)
#pragma warning(disable : 4800 4610 4512 4510 4267 4127 4125 4100 4456 4189 4996)
#include "proto/onnx/protobuf/graph.pb.h"
#pragma warning(pop)

namespace ONNXIR
{
    typedef const std::string* PTYPE;
    namespace Utils
    {
        class StringRange;

        class OpUtils
        {
        public:
            static PTYPE ToType(const TypeProto& p_type);
            static PTYPE ToType(const std::string& p_type);
            static const TypeProto& ToTypeProto(const PTYPE& p_type);
            static std::string ToString(const TypeProto& p_type, const std::string& left = "", const std::string& right = "");
            static std::string ToDataTypeString(const TensorProto::DataType& p_type);
            static std::string ToAttrTypeString(const ValueProto& p_value, const std::string& left = "", const std::string& right = "");
            static void FromString(const std::string& p_src, TypeProto& p_type);
            static void FromDataTypeString(const std::string& p_src, TensorProto::DataType& p_type);
            static bool IsValidDataTypeString(const std::string &p_dataType);
            static void SplitStringTokens(StringRange& p_src, std::vector<StringRange>& p_tokens);
        private:
            static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();
            // Returns lock used for concurrent updates to TypeStrToProtoMap.
            static std::mutex& GetTypeStrLock();
        };

        // Simple class which contains pointers to external string buffer and a size.
        // This can be used to track a "valid" range/slice of the string.
        // Caller should ensure StringRange is not used after external storage has
        // been freed.
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
            void ParensWhitespaceStrip();
            size_t Find(const char p_ch) const;

            // These methods provide a way to return the range of the string
            // which was discarded by LStrip(). i.e. We capture the string
            // range which was discarded.
            StringRange GetCaptured();
            void RestartCapture();

        private:
            // m_data + size tracks the "valid" range of the external string buffer.
            const char* m_data;
            size_t m_size;

            // m_start and m_end track the captured range.
            // m_end advances when LStrip() is called.
            const char* m_start;
            const char* m_end;
        };

        // Use this to avoid compiler warnings about unused variables. E.g., if
        // a variable is only used in an assert when compiling in Release mode.
        // Adapted from https://stackoverflow.com/questions/15763937/unused-parameter-in-c11
        template<typename... Args>
        void Ignore(Args&&...) {}
    }
}

#endif // ! ONNXIR_UTILS_H
