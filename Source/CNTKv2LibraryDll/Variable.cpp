//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Function.h"

namespace CNTK
{
    /*static*/ const std::vector<Axis> Variable::DefaultInputVariableDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };

    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
    }

    FunctionPtr Variable::Owner() const 
    {
        if (m_dataFields->m_ownerFunction != nullptr)
            return m_dataFields->m_ownerFunction->shared_from_this();
        else
            return nullptr;
    }

    Variable::operator FunctionPtr() const
    {
        auto varOwner = Owner();
        if (varOwner)
            return CompositeFunction::Create(varOwner, varOwner->Name());
        else
            return Internal::Combine({ *this });
    }

    /*static*/ Parameter Parameter::UniformInitParameter(const NDShape& shape, DataType type, double range, unsigned long seed, const DeviceDescriptor& device, const std::wstring& name)
    {
        switch (type)
        {
        case DataType::Float:
            return Parameter(NDArrayView::RandomUniform<float>(shape, -range, range, seed, device), name);
        case DataType::Double:
            return Parameter(NDArrayView::RandomUniform<double>(shape, -range, range, seed, device), name);
        default:
            InvalidArgument("Parameter construction: Unsupported DataType %s", DataTypeName(type));
        }
    }

    /*static*/ Parameter Parameter::NormalInitParameter(const NDShape& shape, DataType type, double stdDev, unsigned long seed, const DeviceDescriptor& device, const std::wstring& name)
    {
        switch (type)
        {
        case DataType::Float:
            return Parameter(NDArrayView::RandomNormal<float>(shape, 0, stdDev, seed, device), name);
        case DataType::Double:
            return Parameter(NDArrayView::RandomNormal<double>(shape, 0, stdDev, seed, device), name);
        default:
            InvalidArgument("Parameter construction: Unsupported DataType %s", DataTypeName(type));
        }
    }
}
