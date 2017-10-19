//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <fstream>

namespace CNTK
{
    ///
    /// Class that stores a std::wstring, but most if the time is empty, so that it is cheaper to have one extra redirection.
    ///
    class OptionalString
    {
        std::wstring* m_string;
    public:
        OptionalString()                       : m_string(nullptr)                        { }
        OptionalString(const std::wstring&  s) : m_string(new std::wstring(s))            { }
        OptionalString(      std::wstring&& s) : m_string(new std::wstring(std::move(s))) { }
        OptionalString& operator=(const std::wstring& s) { auto newString = s.empty() ? nullptr : new std::wstring(s); if (m_string) delete m_string; m_string = newString; return *this; }
        operator const std::wstring&() const { static const std::wstring s_emptyString; return m_string ? *m_string : s_emptyString; }
        operator bool() const { return m_string && !m_string->empty(); }
        const wchar_t* c_str() const { return m_string ? m_string->c_str() : L""; }
    };
    struct VariableFields final : public std::enable_shared_from_this<VariableFields>
    {
        friend class CompositeFunction;
        friend class AutoBatch;

        // variable type
        NDShape m_shape; // TODO: use fixed vector up to 6 or so (vector is 24 bytes)
        ::CNTK::DataType m_dataType;
        bool m_needsGradient;
        bool m_isSparse;

        // main object type
        // Located down here so that it combines neatly to 8 bytes with the surrounding members.
        VariableKind m_varKind;

        // Dynamite
        static const unsigned int m_compositeArgumentIndexUndefined = UINT_MAX;
        unsigned int m_compositeArgumentIndex = m_compositeArgumentIndexUndefined;     // if this is a Placeholder that is an argument of a dynamically invoked composite, then this its position in the parameter list (otherwise undefined)
        mutable Internal::AutoBatchRedirection m_redirection; // Function redirection, e.g. into batched output
        Internal::AutoBatchConsumers m_consumers;             // set of consumers of this value. Used differently in forward (notification) and backward (inverted graph).
        mutable size_t m_visitedTag = 0;                      // used for tree traversal
        mutable size_t m_cseVisitedTag = 0;                   // used for common sub-expression elimination
        mutable uintptr_t m_valueAddrForHash = 0;             // cached address of m_value[0,...], divided by sizeof. Used as hash.

        // value
        NDArrayViewPtr m_value;
        NDArrayViewPtr m_gradient;

        // graph
        std::weak_ptr<PrimitiveFunction> m_ownerFunction; // OutputVariables only: Primitive that owns this output.
        OptionalString m_name;

        // debugging aid for identifying objects
        unsigned int m_uniqueIdForDebugging = GetUniqueId(); static unsigned int GetUniqueId() { static unsigned int id = 0; return ++id; }

        // less commonly used fields are tucked away in a separate struct.
        // The assumption is that the vast majority of variables do not have this. Saves > 128 bytes (2 cache rows) for each Variable.
        struct MoreVariableFields
        {
            // type
            std::vector<Axis> m_dynamicAxes;

            // graph
            mutable std::wstring m_uid;
            Variable m_blockFunctionVariableMapping;

            // lazy initialization
            std::unique_ptr<std::once_flag> m_initValueFlag;
            std::unique_ptr<ParameterInitializer> m_valueInitializer;
            std::unique_ptr<DeviceDescriptor> m_valueInitializationDevice;

            // static computation
            std::atomic<size_t> m_valueTimeStamp = 0;
        };
        mutable std::unique_ptr<MoreVariableFields> m_more;
        bool HasMore() const { return m_more.get() != nullptr; }
        MoreVariableFields& LazyGetMore() const
        {
            if (!HasMore()) // TODO: This should be made thread-safe.
                m_more.reset(new MoreVariableFields());
            return *m_more;
        }
        MoreVariableFields& More() { return LazyGetMore(); }
        const MoreVariableFields& More() const { return LazyGetMore(); }

        VariableFields(const NDShape& shape, VariableKind varType, ::CNTK::DataType type, const std::weak_ptr<PrimitiveFunction>& ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
            : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value),
              m_needsGradient(needsGradient)/*, m_dynamicAxes(dynamicAxes)*/, m_isSparse(isSparse), m_name(name)/*, m_uid(uid), m_valueTimeStamp(0)*/
        {
            if (value && (type != value->GetDataType()))
                InvalidArgument("The DataType of the Parameter/Constant Variable '%S' does not match the DataType of the associated Value", AsString().c_str());

            if (!uid.empty())
                More().m_uid = uid;

            // Validate that each of the dynamic axes are unique
            if (!dynamicAxes.empty())
            {
                More().m_dynamicAxes = dynamicAxes;
                std::unordered_set<Axis> uniqueDynamicAxis;
                for (auto& currentDynamicAxis : dynamicAxes)
                {
                    auto retVal = uniqueDynamicAxis.insert(currentDynamicAxis);
                    if (!retVal.second)
                        InvalidArgument("Dynamic axis named %S is specified more than once for Variable '%S'", currentDynamicAxis.Name().c_str(), AsString().c_str());
                }
            }

            if (m_varKind == VariableKind::Input)
            {
                for (auto dim : m_shape.Dimensions())
                {
                    if (dim == 0)
                        InvalidArgument("Variable '%S' has invalid shape '%S'.", AsString().c_str(), m_shape.AsString().c_str());
                }
            }

            if ((m_varKind == VariableKind::Parameter) || (m_varKind == VariableKind::Constant))
            {
                if (m_shape.HasFreeDimension())
                    InvalidArgument("Parameter/Constant '%S' has invalid shape '%S'; it is illegal for a Parameter/Constant to have a FreeDimension.", AsString().c_str(), m_shape.AsString().c_str());
            }
        }

        // simple version used during cloning
        VariableFields(NDShape&& shape, VariableKind varType, ::CNTK::DataType type, bool needsGradient, bool isSparse)
            : m_shape(std::move(shape)), m_varKind(varType), m_dataType(type),
              m_needsGradient(needsGradient), m_isSparse(isSparse)/*, m_valueTimeStamp(0)*/
        {
        }

        std::wstring AsString() const;
        std::shared_ptr<VariableFields> Clone() const;
        PrimitiveFunctionPtr Owner() const; // (can't be a const& since we lock the weak pointer)
        bool OwnerIs(const Function* f) const;
        const std::wstring& Uid() const;

        CNTK_API void SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device);

    private:
        // Disallow copy and move construction and assignment
        VariableFields(const VariableFields&) = delete; VariableFields& operator=(const VariableFields& other) = delete; VariableFields(VariableFields&&) = delete; VariableFields& operator=(VariableFields&&) = delete;
    };
}
