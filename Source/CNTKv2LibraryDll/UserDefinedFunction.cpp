//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "UserDefinedFunction.h"
#include "UserFunctionFactory.h"
#include "Serialization.h"
#include "PrimitiveFunction.h"
#include "CompositeFunction.h"


namespace CNTK
{

    static Internal::UDFDeserializeCallbackWrapperPtr s_SWIGCallbackWrapper;

    void Internal::RegisterUDFDeserializeCallbackWrapper(UDFDeserializeCallbackWrapperPtr callbackPtr) 
    {
        s_SWIGCallbackWrapper = callbackPtr;
    }

    static const std::wstring s_nativeUDFTypeValue = L"NativeUserDefinedFunction" ;

    static std::unordered_map<std::wstring, std::pair<std::wstring, std::wstring>> s_deserializedUDFsRegistry;

    Function::Function(const std::vector<Variable>& inputs, const std::wstring& name)
        : Function(inputs, {}, name)
    {}

    Function::Function(const std::vector<Variable>& inputs, const Dictionary& functionConfig, const std::wstring& name)
        : Function(inputs, functionConfig, nullptr, name, Internal::GenerateUid(L"UserDefinedFunction"))
    {}

    /*static*/ FunctionPtr Function::DeserializeNativeImpl(const std::vector<Variable>& inputs, const std::wstring& name,  const Dictionary& dict)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { userDefinedStateKey, udfModuleNameKey, udfFactoryMethodNameKey, opKey };
        ValidateDictionary<PrimitiveFunction>(dict, s_requiredDictionaryKeys, s_nativeUDFTypeValue, s_serializationVersion);

        auto state = dict[userDefinedStateKey].Value<Dictionary>();
        auto opName = dict[opKey].Value<wstring>();
        auto moduleName = dict[udfModuleNameKey].Value<wstring>();
        auto methodName = dict[udfFactoryMethodNameKey].Value<wstring>();

        FunctionPtr udf = nullptr;

        auto callback = Function::GetUDFDeserializeCallback(opName);
        if (callback != nullptr) 
        {
            udf = callback->operator()(inputs, name, state);
        }
        else 
        {
            Microsoft::MSR::CNTK::Plugin plugin;
            auto factoryMethod = (UserFunctionFactoryMethodType)(plugin.Load(moduleName, Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(methodName)), /*isCNTKPlugin =*/ false));
            udf = FunctionPtr(factoryMethod(inputs.data(), inputs.size(), &state, name.c_str()));
        }

        if (udf == nullptr) 
        {
            RuntimeError("Unable to reconstruct the native UserFunction with op name '%S'", opName.c_str());
        }

        s_deserializedUDFsRegistry[opName] = { moduleName, methodName };
        
        return udf;
    }

    /*virtual*/ std::wstring Function::ModuleName() const
    {
        auto it = s_deserializedUDFsRegistry.find(OpName());
        if (it != s_deserializedUDFsRegistry.end())
        {
            auto moduleAndMethodPair = it->second;
            return moduleAndMethodPair.first;
        }
        
        // this op name was never registered in the s_deserializedUDFsRegistry (which only happens during the deserialization),
        // then use user factory as a fallback (this udf must have been registed, so that an instance could be created).
        return s_userFunctionFactory->GetModuleName(OpName());
    }

    /*virtual*/ std::wstring Function::DeserializeMethodName() const
    {
        auto it = s_deserializedUDFsRegistry.find(OpName());
        if (it != s_deserializedUDFsRegistry.end())
        {
            auto moduleAndMethodPair = it->second;
            return moduleAndMethodPair.second;
        }

        // this op name was never registered in the s_deserializedUDFsRegistry (which only happens during the deserialization),
        // then use user factory as a fallback (this udf must have been registed, so that an instance could be created).
        return Function::s_userFunctionFactory->GetFactoryMethodName(OpName());
    }

    Dictionary Function::SerializeNativeImpl() const
    {
        Dictionary dict;
        dict[userDefinedStateKey] = Serialize();
        dict[udfModuleNameKey] = ModuleName();
        dict[udfFactoryMethodNameKey] = DeserializeMethodName();
        dict[opKey] = OpName();
        dict[versionKey] = s_serializationVersion;
        dict[typeKey] = s_nativeUDFTypeValue;
        return dict;
    }

    static const std::wstring s_userDefinedFunctionTypeValue = L"UserDefinedFunction";

    /*static*/ bool UDFUtils::IsUDF(const FunctionPtr& f)
    {
        return (dynamic_cast<const PrimitiveFunction*>(f.get()) == nullptr) &&
            (dynamic_cast<const CompositeFunction*>(f.get()) == nullptr);
    }

    /*static*/ bool UDFUtils::IsUDF(const Dictionary& dict)
    {
        return (dict.Contains(typeKey) && dict[typeKey].Value<std::wstring>() == s_userDefinedFunctionTypeValue);
    }

    /*static*/ bool UDFUtils::IsNativeUDF(const Dictionary& dict)
    {
        assert(IsUDF(dict));
        return (dict.Contains(nativeUDFKey) && dict[nativeUDFKey].Value<bool>() == true);
    }

    /*static*/ Dictionary UDFUtils::Serialize(const FunctionPtr& f)
    {
        Dictionary dict = SerializeCommonFunctionAttributes(*f, s_serializationVersion, s_userDefinedFunctionTypeValue);
        bool native = f->IsNative();
        dict[nativeUDFKey] = native;
        dict[userDefinedStateKey] = (native) ? f->SerializeNativeImpl() : f->Serialize();
        return dict;
    }

    /*static*/ FunctionPtr UDFUtils::Deserialize(const Dictionary& dict,
        const unordered_map<std::wstring, Variable>& uidToVariableMap,
        const DeviceDescriptor& device)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, uidKey, inputsKey, userDefinedStateKey };
        ValidateDictionary<PrimitiveFunction>(dict, s_requiredDictionaryKeys, s_userDefinedFunctionTypeValue, s_serializationVersion);

        const auto& uid = dict[uidKey].Value<std::wstring>();
        std::wstring name = L"";
        if (dict.Contains(nameKey))
            name = dict[nameKey].Value<std::wstring>();

        auto inputs = GetInputVariables(dict, uidToVariableMap, s_serializationVersion);

        auto state = dict[userDefinedStateKey].Value<Dictionary>();

        FunctionPtr udf;

        if (IsNativeUDF(dict))
        {
            udf = Function::DeserializeNativeImpl(inputs, name, state);
        }
        else if (s_SWIGCallbackWrapper != nullptr)
        {
            // If we're being called from SWIG, the actual deserializer should be registered by
            // the target language CNTK implementation (i.e., cntk_py for Python)
            udf = s_SWIGCallbackWrapper->operator()(inputs, name, state);
        }

        if (udf == nullptr)
        {
            RuntimeError("Unable to reconstruct a user-defined function (name = %S, uid = %S). "
                "Please make sure to specify a valid UDF deserializer.", name.c_str(), uid.c_str());
        }

        // Restore the original uid, which other functions in the graph depend on
        // (their inputs refer to the uids of this UDF outputs, which are generated base on the uid of this UDF).
        udf->m_uid = uid;

        return udf;
    }
}
