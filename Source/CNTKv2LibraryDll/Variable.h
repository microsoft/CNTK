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
    struct VariableFields final : public std::enable_shared_from_this<VariableFields>
    {
        friend class CompositeFunction;
        friend class AutoBatch;

        VariableKind m_varKind;

        // graph
        std::weak_ptr<PrimitiveFunction> m_ownerFunction; // OutputVariables only: Primitive that owns this output.
        std::wstring m_name;
        mutable std::wstring m_uid;
        Variable m_blockFunctionVariableMapping;

        // variable type
        NDShape m_shape;
        ::CNTK::DataType m_dataType;
        bool m_needsGradient;
        std::vector<Axis> m_dynamicAxes; // TODO: make this a shared_ptr?
        bool m_isSparse;

        // value
        NDArrayViewPtr m_value;
        NDArrayViewPtr m_gradient;

        // computation
        std::atomic<size_t> m_valueTimeStamp;

        // Dynamite
        struct Redirection // redirect this value to a different owner function. Also allow for lazy Index operation.
        {
            PrimitiveFunctionPtr m_functionHolder;  // holds shared_ptr to owner if created anew
            size_t m_index;                         // and we take this slice on the way (SIZE_MAX if none)  --TODO: replace by m_sliceBegin/End
            PrimitiveFunction* m_function = (PrimitiveFunction*)-1;          // ...for now use these instead, until we are ready to switch; m_functionHolder becomes m_functionHolder
            size_t m_sliceBegin, m_sliceEnd;        // slice out these items (applied to last dimension). Do nothing if m_sliceEnd==SIZE_MAX.  --TODO: think this through more
            size_t m_depthHint = 0;                 // this redirection skipped a Barrier with this depthHint
            operator bool() const { return m_function != nullptr; } // allows for "if (m_redirection)"
        };
        mutable Redirection m_redirection;
        struct Consumers : public std::pair<std::pair<PrimitiveFunction*, size_t>, std::vector<std::pair<PrimitiveFunction*, size_t>>>
        {
            Consumers() { first.first = (PrimitiveFunction*)-1; } // this initialization can be removed once this is debugged (or once we replaced this horrible construct)
            size_t size() const { return (first.first ? 1 : 0) + second.size(); }
            void clear() { first.first = nullptr; second.clear(); }
            void push_back(PrimitiveFunction* f, size_t i)
            {
                if (!first.first) // optimized for main case of 1 consumer. No std::vector in that case.
                    first = std::move(std::make_pair(f, i)); // note: we don't need i for forward; can optimize
                else
                    second.emplace_back(std::make_pair(f, i));
            }
            template<class F>
            void ForAll(const F& f)
            {
                if (first.first)
                    f(first);
                for (auto& c : second) // all other consumers
                    f(c);
            }
        };
        Consumers m_consumers;
        mutable size_t m_visitedTag = 0; // used for tree traversal

        // lazy initialization
        std::unique_ptr<std::once_flag> m_initValueFlag;
        std::unique_ptr<ParameterInitializer> m_valueInitializer;
        std::unique_ptr<DeviceDescriptor> m_valueInitializationDevice;

        VariableFields(const NDShape& shape, VariableKind varType, ::CNTK::DataType type, const std::weak_ptr<PrimitiveFunction>& ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
            : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value),
              m_needsGradient(needsGradient), m_dynamicAxes(dynamicAxes), m_isSparse(isSparse), m_name(name), m_uid(uid), m_valueTimeStamp(0)
        {
            if (value && (type != value->GetDataType()))
                InvalidArgument("The DataType of the Parameter/Constant Variable '%S' does not match the DataType of the associated Value", AsString().c_str());

            // Validate that each of the dynamic axes are unique
            if (!dynamicAxes.empty())
            {
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
