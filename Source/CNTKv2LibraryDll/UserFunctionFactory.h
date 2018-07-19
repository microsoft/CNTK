//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Basics.h"

namespace CNTK
{
    typedef Function* (*UserFunctionFactoryMethodType)(const Variable* operands, size_t numOperands, const Dictionary* attributes, const wchar_t* name);
    class UserFunctionFactory : public std::enable_shared_from_this<UserFunctionFactory>
    {
    public:

        bool IsRegistered(const std::wstring& uniqueOpName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            return IsRegisteredImpl(uniqueOpName);
        }
        void Register(const std::wstring& uniqueOpId, const std::wstring& moduleName, const std::wstring& factoryMethodName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (IsRegisteredImpl(uniqueOpId))
                InvalidArgument("UserFunction with op-id '%S' is already registered. All UserFunction op-ids must be unique.", uniqueOpId.c_str());

            m_userFunctionFactoryMethodMap[uniqueOpId] = std::make_shared<UserFunctionFactoryMethodInfo>(moduleName, factoryMethodName);
        }

        std::wstring GetModuleName(const std::wstring& uniqueOpName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!IsRegisteredImpl(uniqueOpName))
                InvalidArgument("UserFunction with op name '%S' has not been registered.", uniqueOpName.c_str());

            return m_userFunctionFactoryMethodMap.at(uniqueOpName)->m_moduleName;
        }

        std::wstring GetFactoryMethodName(const std::wstring& uniqueOpName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!IsRegisteredImpl(uniqueOpName))
                InvalidArgument("UserFunction with op name '%S' has not been registered.", uniqueOpName.c_str());

            return m_userFunctionFactoryMethodMap.at(uniqueOpName)->m_factoryMethodName;
        }
        FunctionPtr CreateInstance(const std::wstring& opId, const std::vector<Variable>& inputs, const Dictionary& functionConfig, const std::wstring& userFunctionInstanceName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (!IsRegisteredImpl(opId))
                InvalidArgument("UserFunction with op-id '%S' has not been registered.", opId.c_str());

            return std::shared_ptr<Function>(m_userFunctionFactoryMethodMap.at(opId)->m_factoryMethod(inputs.data(), inputs.size(), &functionConfig, userFunctionInstanceName.c_str()));
        }

    private:
        bool IsRegisteredImpl(const std::wstring& uniqueOpName)
        {
            return (m_userFunctionFactoryMethodMap.find(uniqueOpName) != m_userFunctionFactoryMethodMap.end());
        }
        struct UserFunctionFactoryMethodInfo : public std::enable_shared_from_this<UserFunctionFactoryMethodInfo>
        {
            UserFunctionFactoryMethodInfo(const std::wstring& moduleName, const std::wstring& factoryMethodName)
                : m_moduleName(moduleName), m_factoryMethodName(factoryMethodName)
            {
                m_factoryMethod = (UserFunctionFactoryMethodType)(m_module.Load(m_moduleName, Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(m_factoryMethodName)), /*isCNTKPlugin =*/ false));
            }

            std::wstring m_moduleName;
            std::wstring m_factoryMethodName;
            Microsoft::MSR::CNTK::Plugin m_module;
            UserFunctionFactoryMethodType m_factoryMethod;
        };
        typedef std::shared_ptr<UserFunctionFactoryMethodInfo> UserFunctionFactoryMethodInfoPtr;

        std::mutex m_mutex;
        std::unordered_map<std::wstring, UserFunctionFactoryMethodInfoPtr> m_userFunctionFactoryMethodMap;
    };
}
