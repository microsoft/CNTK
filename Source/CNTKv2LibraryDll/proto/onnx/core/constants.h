#include <string>
#include <unordered_set>
#include <vector>

namespace ONNXIR
{
    static const std::string c_noOp = "NoOp";
    static const std::string c_constantOp = "Constant";
    static const std::string c_constantValue = "value";
    static const std::string c_onnxDomain = "";
    static const std::string c_mlDomain = "ai.onnx.ml";
    static const std::string c_msDomain = "com.microsoft";

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
        const std::string c_undefined = "undefined";

        std::unordered_set<std::string>& GetAllowedDataTypes();
        ~TypesWrapper() = default;
        TypesWrapper(const TypesWrapper&) = delete;
        void operator=(const TypesWrapper&) = delete;
    private:
        TypesWrapper() = default;
    };

    // Singleton class used to help initialize static objects related to type strings.
    // This is not strictly needed but allows common rich type strings to be defined here along
    // side the data type strings above in TypesWrapper.
    class TypeStringsInitializer
    {
    public:
        static TypeStringsInitializer& InitializeTypeStrings();
        ~TypeStringsInitializer() = default;
        TypeStringsInitializer(const TypeStringsInitializer&) = delete;
        void operator=(const TypeStringsInitializer&) = delete;
    private:
        TypeStringsInitializer();
        // Common string representations of TypeProto. These are used to pre-initialize
        // typeStringToProto map. Note: some of these strings may have already been initialized in
        // the map via op registration depending on static initialization order.
        const std::vector<std::string> m_commonTypeStrings = { "tensor(float16)", "tensor(float)",
            "tensor(double)", "tensor(int8)", "tensor(int16)", "tensor(int32)",
            "tensor(int64)", "tensor(uint8)", "tensor(uint16)", "tensor(uint32)",
            "tensor(uint64)", "tensor(complex64)", "tensor(complex128)", "tensor(string)",
            "tensor(bool)" };
    };
}
