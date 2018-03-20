#include "constants.h"
#include "utils.h"

namespace ONNXIR
{
    TypesWrapper& TypesWrapper::GetTypesWrapper()
    {
        static TypesWrapper types;
        return types;
    }

    std::unordered_set<std::string>& TypesWrapper::GetAllowedDataTypes()
    {
        static std::unordered_set<std::string> allowedDataTypes = {
            c_float16, c_float, c_double,
            c_int8, c_int16, c_int32, c_int64,
            c_uint8, c_uint16, c_uint32, c_uint64,
            c_complex64, c_complex128,
            c_string, c_bool };
        return allowedDataTypes;
    }

    TypeStringsInitializer& TypeStringsInitializer::InitializeTypeStrings()
    {
        static TypeStringsInitializer initTypes;
        return initTypes;
    }

    TypeStringsInitializer::TypeStringsInitializer()
    {
        // Initialize TypeStrToProtoMap using common type strings.
        for (const auto& t : m_commonTypeStrings)
        {
            Utils::OpUtils::ToType(t);
        }
    }

    // This ensures all static objects related to type strings get initialized.
    // TypeStringsInitializer constructor populates TypeStrToProtoMap with common type strings.
    // TypesWrapper() gets instantiated via call to OpUtils::FromString()
    // which calls GetTypesWrapper().
    // Note: due to non-deterministic static initialization order, some of the type strings
    // may have already been added via Op Registrations which use those type strings.
    static TypeStringsInitializer& _typeStrings = TypeStringsInitializer::InitializeTypeStrings();
}
