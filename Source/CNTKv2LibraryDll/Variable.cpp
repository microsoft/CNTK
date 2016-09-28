//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Function.h"
#include "InputAndParamNodes.h"

namespace CNTK
{
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

    NDArrayViewPtr Variable::Value() const
    {
        if (m_dataFields->m_value == nullptr)
        {
            assert(m_dataFields->m_valueInitializer);
            assert(m_dataFields->m_valueInitializationDevice);

            switch (GetDataType())
            {
            case DataType::Float:
            {
                m_dataFields->m_value = CreateValueFromParameterInitializer<float>(Shape(), *m_dataFields->m_valueInitializer, *m_dataFields->m_valueInitializationDevice);
                break;
            }
            case DataType::Double:
            {
                m_dataFields->m_value = CreateValueFromParameterInitializer<double>(Shape(), *m_dataFields->m_valueInitializer, *m_dataFields->m_valueInitializationDevice);
                break;
            }
            default:
                LogicError("Unsupported DataType %s", DataTypeName(GetDataType()));
                break;
            }
        }

        assert(m_dataFields->m_value != nullptr);
        return m_dataFields->m_value;
    }

    static const std::wstring InitializerTypeAttributeName = L"initializerType";
    static const std::wstring OutputRankAttributeName = L"outputRank";
    static const std::wstring FilterRankAttributeName = L"filterRank";
    static const std::wstring ScaleAttributeName = L"scale";
    static const std::wstring RandomSeedAttributeName = L"randomSeed";
    static const std::wstring KernelWidthAttributeName = L"kernelWidth";
    static const std::wstring KernelHeightAttributeName = L"kernelHeight";

    ParameterInitializer UniformInitializer(double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::UniformInitializerTypeName;
        initConfig[ScaleAttributeName] = scale;
        initConfig[RandomSeedAttributeName] = (size_t)seed;

        return initConfig;
    }

    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, int outputRank, int filterRank, double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = initializerTypeName;
        initConfig[OutputRankAttributeName] = (size_t)outputRank;
        initConfig[FilterRankAttributeName] = (size_t)filterRank;
        initConfig[ScaleAttributeName] = scale;
        initConfig[RandomSeedAttributeName] = (size_t)seed;

        return initConfig;
    }

    ParameterInitializer GaussianInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GaussianInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer XavierInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::XavierInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer GlorotUniformInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotUniformInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer GlorotNormalInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotNormalInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer HeUniformInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeUniformInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer HeNormalInitializer(int outputRank, int filterRank, double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeNormalInitializerTypeName, outputRank, filterRank, scale, seed);
    }

    ParameterInitializer BilinearInitializer(size_t kernelWidth, size_t kernelHeight)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::BilinearInitializerTypeName;
        initConfig[KernelWidthAttributeName] = kernelWidth;
        initConfig[KernelHeightAttributeName] = kernelHeight;

        return initConfig;
    }

    template <typename ElementType>
    /*static*/ NDArrayViewPtr Variable::CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device)
    {
        auto dataType = AsDataType<ElementType>();
        auto value = MakeSharedObject<NDArrayView>(dataType, shape, device);
        auto valueMatrix = value->template GetWritableMatrix<ElementType>();
        auto initializerType = initConfig[InitializerTypeAttributeName].Value<std::wstring>();
        if (initializerType == Microsoft::MSR::CNTK::BilinearInitializerTypeName)
        {
            auto kernelWidth = initConfig[KernelWidthAttributeName].Value<size_t>();
            auto kernelHeight = initConfig[KernelHeightAttributeName].Value<size_t>();

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitBilinear(*valueMatrix, AsTensorShape(shape), kernelWidth, kernelHeight, AsCNTKImplDeviceId(device));
        }
        else
        {
            auto randomSeed = (unsigned long)initConfig[RandomSeedAttributeName].Value<size_t>();
            auto scale = initConfig[ScaleAttributeName].Value<double>();
            int outputRank = DefaultParamInitOutputRank, filterRank = DefaultParamInitFilterRank;
            if (initializerType != Microsoft::MSR::CNTK::UniformInitializerTypeName)
            {
                outputRank = (int)initConfig[OutputRankAttributeName].Value<size_t>();
                filterRank = (int)initConfig[FilterRankAttributeName].Value<size_t>();
            }

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitRandom(*valueMatrix, AsTensorShape(shape), initializerType, randomSeed, (ElementType)scale, filterRank, outputRank, false, AsCNTKImplDeviceId(device));
        }

        return value;
    }
}
