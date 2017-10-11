#include "constants.h"

namespace ONNXIR
{

    TypesWrapper& TypesWrapper::GetTypesWrapper()
    {
        static TypesWrapper* types = new TypesWrapper();
        return *types;
    }

    std::unordered_set<std::string>& TypesWrapper::GetAllowedDataTypes()
    {
        static std::unordered_set<std::string>* allowedDataTypes =
            new std::unordered_set<std::string>({
            c_float16, c_float, c_double,
            c_int8, c_int16, c_int32, c_int64,
            c_uint8, c_uint16, c_uint32, c_uint64,
            c_complex64, c_complex128,
            c_string, c_bool });
        return *allowedDataTypes;
    }
}