#include <string>
#include <unordered_set>

namespace ONNXIR
{
    static const std::string c_noOp = "NoOp";
    static const std::string c_constantOp = "Constant";
    static const std::string c_constantValue = "value";

    // Singleton wrapper around allowed data types.
    // This implements construct on first use which is needed to ensure
    // static objects are initialized before use. Ops registration does not work
    // properly without this.
    class TypesWrapper
    {
    public:
        static TypesWrapper& GetTypesWrapper();

        // DataType strings. These should match the DataTypes defined in Data.proto
        const std::string c_float16 = "float16";
        const std::string c_float = "float";
        const std::string c_double = "double";
        const std::string c_int8 = "int8";
        const std::string c_int16 = "int16";
        const std::string c_int32 = "int32";
        const std::string c_int64 = "int64";
        const std::string c_uint8 = "uint8";
        const std::string c_uint16 = "uint16";
        const std::string c_uint32 = "uint32";
        const std::string c_uint64 = "uint64";
        const std::string c_complex64 = "complex64";
        const std::string c_complex128 = "complex128";
        const std::string c_string = "string";
        const std::string c_bool = "bool";
        std::unordered_set<std::string>& GetAllowedDataTypes();
        ~TypesWrapper() = default;
        TypesWrapper(const TypesWrapper&) = delete;
        void operator=(const TypesWrapper&) = delete;
    private:
        TypesWrapper() = default;
    };
}