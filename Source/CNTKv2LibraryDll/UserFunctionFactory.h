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
    class UserFunctionFactory : public std::enable_shared_from_this<UserFunctionFactory>
    {
        typedef Function* (*UserFunctionFactoryMethodType)(const Variable* operands, size_t numOperands, const Dictionary* attributes, const wchar_t* name);

    public:
        void Register(const std::wstring& uniqueOpName, const std::wstring& moduleName, const std::wstring& factoryMethodName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_userFunctionFactoryMethodMap.find(uniqueOpName) != m_userFunctionFactoryMethodMap.end())
                InvalidArgument("UserFunction with op name '%S' is already registered. All UserFunction op names must be unique.", uniqueOpName.c_str());

            m_userFunctionFactoryMethodMap[uniqueOpName] = std::make_shared<UserFunctionFactoryMethodInfo>(moduleName, factoryMethodName);
        }

        FunctionPtr CreateInstance(const std::wstring& opName, const std::vector<Variable>& inputs, const Dictionary& functionConfig, const std::wstring& userFunctionInstanceName)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_userFunctionFactoryMethodMap.find(opName) == m_userFunctionFactoryMethodMap.end())
                InvalidArgument("UserFunction with op name '%S' has not been registered.", opName.c_str());

            return std::shared_ptr<Function>(m_userFunctionFactoryMethodMap.at(opName)->m_factoryMethod(inputs.data(), inputs.size(), &functionConfig, userFunctionInstanceName.c_str()));
        }

    private:
        struct UserFunctionFactoryMethodInfo : public std::enable_shared_from_this<UserFunctionFactoryMethodInfo>
        {
            UserFunctionFactoryMethodInfo(const std::wstring& moduleName, const std::wstring& factoryMethodName)
                : m_moduleName(moduleName), m_factoryMethodName(factoryMethodName)
            {
                m_factoryMethod = (UserFunctionFactoryMethodType)(m_module.Load(m_moduleName, msra::strfun::utf8(m_factoryMethodName), /*isCNTKPlugin =*/ false));
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
