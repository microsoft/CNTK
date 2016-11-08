//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Serialization.h"
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
            return Combine({ *this });
    }

    NDArrayViewPtr Variable::Value() const
    {
        if (!IsConstant() && !IsParameter())
            LogicError("Only Variables of kind Parameter and Constant have a Value!");

        if (m_dataFields->m_initValueFlag)
        {
            std::call_once(*m_dataFields->m_initValueFlag, [=]{
                assert(m_dataFields->m_value == nullptr);
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

                m_dataFields->m_valueInitializer = nullptr;
                m_dataFields->m_valueInitializationDevice = nullptr;
            });
        }

        assert(m_dataFields->m_value != nullptr);
        return m_dataFields->m_value;
    }

    static const std::wstring InitializerTypeAttributeName = L"initializerType";
    static const std::wstring OutputRankAttributeName = L"outputRank";
    static const std::wstring FilterRankAttributeName = L"filterRank";
    static const std::wstring ValueAttributeName = L"value";
    static const std::wstring ScaleAttributeName = L"scale";
    static const std::wstring RandomSeedAttributeName = L"randomSeed";
    static const std::wstring KernelWidthAttributeName = L"kernelWidth";
    static const std::wstring KernelHeightAttributeName = L"kernelHeight";

    void Variable::VariableFields::SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device)
    {
        if (m_value != nullptr)
            LogicError("Value initialization config cannot be set if a value already exists");

        assert(!m_valueInitializer);
        assert(!m_valueInitializationDevice);

        m_initValueFlag.reset(new std::once_flag());
        m_valueInitializer.reset(new ParameterInitializer(initializationConfig));
        m_valueInitializationDevice.reset(new DeviceDescriptor(device));
    }

    namespace Internal
    {
        static std::atomic<unsigned long> s_fixedRandomSeed(0);
        void SetFixedRandomSeed(unsigned long fixedRandomSeed)
        {
            s_fixedRandomSeed.store(fixedRandomSeed);
        }
    }

    static std::atomic<unsigned long> s_currentRandomSeed(1);

    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, int outputRank, int filterRank, double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = initializerTypeName;
        initConfig[OutputRankAttributeName] = outputRank;
        initConfig[FilterRankAttributeName] = filterRank;
        initConfig[ScaleAttributeName] = scale;

        auto currentFixedRandomSeed = Internal::s_fixedRandomSeed.load();
        if (currentFixedRandomSeed != 0)
            seed = currentFixedRandomSeed;

        initConfig[RandomSeedAttributeName] = (size_t)seed;

        return initConfig;
    }

    ParameterInitializer ConstantInitializer(double value)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::ConstantInitializerTypeName;
        initConfig[ValueAttributeName] = value;

        return initConfig;
    }

    ParameterInitializer UniformInitializer(double scale, unsigned long seed)
    {
        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = Microsoft::MSR::CNTK::UniformInitializerTypeName;
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

    ParameterInitializer RandomInitializerWithRank(const ParameterInitializer& initializer, int outputRank, int filterRank)
    {
        ParameterInitializer newInitializerWithRanks = initializer;

        // 'initializer' must be a random initializer
        auto initializerType = initializer[InitializerTypeAttributeName].Value<std::wstring>();
        if ((initializerType != Microsoft::MSR::CNTK::UniformInitializerTypeName) &&
            (initializerType != Microsoft::MSR::CNTK::BilinearInitializerTypeName) &&
            (initializerType != Microsoft::MSR::CNTK::ConstantInitializerTypeName))
        {
            int oldOutputRank = initializer[OutputRankAttributeName].Value<int>();
            int oldFilterRank = initializer[FilterRankAttributeName].Value<int>();

            if ((oldOutputRank != SentinelValueForInferParamInitRank) && (oldOutputRank != outputRank))
                InvalidArgument("Output rank of a non-uniform random initialier cannot be overridden if it has been already specified!");

            if ((oldFilterRank != SentinelValueForInferParamInitRank) && (oldFilterRank != filterRank))
                InvalidArgument("Filer rank of a non-uniform random initialier cannot be overridden if it has been already specified!");

            newInitializerWithRanks[OutputRankAttributeName] = outputRank;
            newInitializerWithRanks[FilterRankAttributeName] = filterRank;
        }

        return newInitializerWithRanks;
    }

    Variable::Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
        : m_dataFields(MakeSharedObject<VariableFields>(shape, varType, dataType, ownerFunction, value, needsGradient, dynamicAxes, isSparse, name, uid))
    {}

    template <typename ElementType>
    /*static*/ NDArrayViewPtr Variable::CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device)
    {
        auto dataType = AsDataType<ElementType>();
        auto value = MakeSharedObject<NDArrayView>(dataType, shape, device);
        auto valueMatrix = value->template GetWritableMatrix<ElementType>();
        auto initializerType = initConfig[InitializerTypeAttributeName].Value<std::wstring>();
        if (initializerType == Microsoft::MSR::CNTK::ConstantInitializerTypeName)
        {
            auto constantInitValue = initConfig[ValueAttributeName].Value<double>();
            valueMatrix->SetValue((ElementType)constantInitValue);
        }
        else if (initializerType == Microsoft::MSR::CNTK::BilinearInitializerTypeName)
        {
            auto kernelWidth = initConfig[KernelWidthAttributeName].Value<size_t>();
            auto kernelHeight = initConfig[KernelHeightAttributeName].Value<size_t>();

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitBilinear(*valueMatrix, AsTensorShape(shape), kernelWidth, kernelHeight, AsCNTKImplDeviceId(device));
        }
        else
        {
            auto randomSeed = (unsigned long)initConfig[RandomSeedAttributeName].Value<size_t>();
            if (randomSeed == SentinelValueForAutoSelectRandomSeed)
                randomSeed = s_currentRandomSeed++;

            auto scale = initConfig[ScaleAttributeName].Value<double>();
            int outputRank = DefaultParamInitOutputRank, filterRank = DefaultParamInitFilterRank;
            if (initializerType != Microsoft::MSR::CNTK::UniformInitializerTypeName)
            {
                outputRank = initConfig[OutputRankAttributeName].Value<int>();
                filterRank = initConfig[FilterRankAttributeName].Value<int>();

                if (outputRank == SentinelValueForInferParamInitRank)
                    outputRank = DefaultParamInitOutputRank;

                if (filterRank == SentinelValueForInferParamInitRank)
                    filterRank = DefaultParamInitFilterRank;

                if ((filterRank + outputRank) > shape.Rank())
                    InvalidArgument("Sum of filter rank (%d) and output rank (%d) of the parameter initializer cannot exceed the Parameter's rank(%d)", filterRank, outputRank, (int)shape.Rank());
            }

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitRandom(*valueMatrix, AsTensorShape(shape), initializerType, randomSeed, (ElementType)scale,
                                                                              filterRank, outputRank, /*initOnCPUOnly=*/true,
                                                                              AsCNTKImplDeviceId(device));
        }

        return value;
    }

    static const std::wstring s_variableTypeValue = L"Variable";

    /*virtual*/ Dictionary Variable::Serialize() const
    {
        if (IsOutput())
        {
            LogicError("Output variables cannot be saved");
        }
        Dictionary dict;

        dict[versionKey] = CurrentVersion();
        dict[typeKey] = s_variableTypeValue;
        dict[uidKey] = Uid();
        dict[kindKey] = static_cast<size_t>(Kind());
        dict[dataTypeKey] = static_cast<size_t>(GetDataType());
        const auto& dynamicAxis = DynamicAxes();
        vector<DictionaryValue> dictionaryValueVector; 
        dictionaryValueVector.reserve(dynamicAxis.size());
        for (const auto& axis : dynamicAxis)
        {
            dictionaryValueVector.push_back(axis);
        }
        dict[dynamicAxisKey] = dictionaryValueVector;
        dict[isSparseKey] = IsSparse();
        dict[nameKey] = Name();
        dict[needsGradientKey] = NeedsGradient();
        dict[shapeKey] = Shape();
        if (IsParameter() || IsConstant())
        {
            NDArrayView* value = Value().get();
            if (value == nullptr)
            {
                LogicError("Uninitialized Parameter variable cannot be saved");
            }

            // TODO: add a dictionary value constructor with an rvalue parameter.
            dict[valueKey] = DictionaryValue(*value);
        }
        return dict;
    }

    /*static*/ Variable Variable::Deserialize(const Dictionary& dict, const CNTK::DeviceDescriptor& device)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, uidKey, kindKey, dataTypeKey, dynamicAxisKey, isSparseKey, nameKey, needsGradientKey, shapeKey };

        size_t version = ValidateDictionary<Variable>(dict, s_requiredDictionaryKeys, s_variableTypeValue, s_serializationVersion);

        const auto& uid = dict[uidKey].Value<std::wstring>();

        VariableKind kind = VariableKind(dict[kindKey].Value<std::size_t>());
        if (kind != VariableKind::Constant &&
            kind != VariableKind::Input &&
            kind != VariableKind::Parameter &&
            kind != VariableKind::Placeholder)
        {
            LogicError("Unexpected variable '%ls':'%u' (%s).",
                       kindKey.c_str(),
                       static_cast<std::underlying_type<VariableKind>::type>(kind),
                       GetVersionsString<Variable>(s_serializationVersion, version).c_str());
        }
        
        DataType dataType = DataType(dict[dataTypeKey].Value<std::size_t>());
        if (dataType != DataType::Unknown &&
            dataType != DataType::Float &&
            dataType != DataType::Double)
        {
            LogicError("Unexpected variable '%ls':'%u' (%s).", 
                       dataTypeKey.c_str(), 
                       static_cast<std::underlying_type<DataType>::type>(dataType),
                       GetVersionsString<Variable>(s_serializationVersion, version).c_str());
        }
        
        const vector<DictionaryValue>& dictionaryValueVector = dict[dynamicAxisKey].Value<vector<DictionaryValue>>();
        vector<Axis> dynamicAxis;
        dynamicAxis.reserve(dictionaryValueVector.size());
        for (const auto& dictionaryValue : dictionaryValueVector)
        {
            dynamicAxis.push_back(dictionaryValue.Value<Axis>());
        }

        bool isSparse = dict[isSparseKey].Value<bool>();
        const auto& name = dict[nameKey].Value<std::wstring>();
        bool needsGradient = dict[needsGradientKey].Value<bool>();
        const auto& shape = dict[shapeKey].Value<NDShape>();

        if (kind == VariableKind::Constant || kind == VariableKind::Parameter)
        {
            auto& value = dict[valueKey].Value<NDArrayView>();

            // TODO: this copying here is redundant, value should be moved from the dictionary to the variable.
            // Also, the correct device should be used upfront when deserializing NDArrayView.
            Variable var(shape, kind, dataType, nullptr, value.DeepClone(device, kind == VariableKind::Constant), needsGradient, dynamicAxis, isSparse, name, uid);
            if (var.IsParameter())
                return Parameter(var);
            else
                return Constant(var);
        }

        return Variable(shape, kind, dataType, nullptr, nullptr, needsGradient, dynamicAxis, isSparse, name, uid);
    }
}
