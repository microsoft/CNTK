//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "CompositeFunction.h"
#include "Serialization.h"
#include "InputAndParamNodes.h"

namespace CNTK
{
    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
    }

    const NDShape& Variable::Shape() const
    {
        return m_dataFields->m_shape; 
    }

    const std::vector<Axis>& Variable::DynamicAxes() const
    {
        return m_dataFields->m_dynamicAxes; 
    }

    VariableKind Variable::Kind() const 
    {
        return m_dataFields->m_varKind; 
    }

    bool Variable::IsSparse() const
    {
        return m_dataFields->m_isSparse; 
    }

    const std::wstring& Variable::Name() const
    {
        return m_dataFields->m_name; 
    }

    const std::wstring& Variable::Uid() const
    {
        return m_dataFields->m_uid; 
    }
    
    DataType Variable::GetDataType() const
    {
        return m_dataFields->m_dataType; 
    }

    bool Variable::NeedsGradient() const
    {
        return m_dataFields->m_needsGradient; 
    }

    Variable Variable::Clone() const
    {
        Variable clonedVariable;
        clonedVariable.m_dataFields = m_dataFields->Clone();

        return clonedVariable;
    }

    const Variable& Variable::BlockFunctionVariableMapping() const
    {
        return m_dataFields->m_blockFunctionVariableMapping;
    }

    FunctionPtr Variable::Owner() const 
    {
        if (m_dataFields->m_ownerFunction != nullptr)
            return m_dataFields->m_ownerFunction->shared_from_this();
        else
            return nullptr;
    }

    Variable Variable::CompositePreservingCopy() const
    {
        // We have to preserve the whole subgraph.
        Variable result;
        result.m_outputComposite = (FunctionPtr)(*this);
        result.m_dataFields = m_dataFields;
        return result;
    }

    void Variable::SetOwner(Function* ownerFunction)
    {
        if (Kind() != VariableKind::Output)
            LogicError("Variable::SetOwner: Owner can only be set for Output Variables!");

        if (m_dataFields->m_ownerFunction != nullptr)
            LogicError("Variable::SetOwner: An Output Variable whose owner has previously been set, cannot be reset!");

        m_dataFields->m_ownerFunction = ownerFunction;
    }

    Variable::operator FunctionPtr() const
    {
        auto varOwner = Owner();
        if (varOwner)
            return AsComposite(varOwner, varOwner->Name());
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

    void Variable::SetValue(const NDArrayViewPtr& value)
    {
        if (!IsParameter())
            LogicError("Variable::SetValue can be only invoked on a Parameter variable!");
        else if (GetDataType() != value->GetDataType()) 
            LogicError("Variable::SetValue: 'source' and 'destination' have different data types!");
        else if (Shape() != value->Shape() && (AsTensorShape(Shape()) != AsTensorShape(value->Shape())))
            LogicError("Variable::SetValue: 'source' and 'destination' have different shapes!");

        bool alreadySet = false;
        if (m_dataFields->m_initValueFlag)
        {
            // In the case of lazy initialization, try to avoid the redundant call to the initializer. 
            std::call_once(*m_dataFields->m_initValueFlag, [=, &value, &alreadySet] {
                // If the variable hasn't been initialized yet, clone the content of the supplied value and delete the initializer.
                m_dataFields->m_value = value->DeepClone(*m_dataFields->m_valueInitializationDevice, false);
                m_dataFields->m_valueInitializer = nullptr;
                m_dataFields->m_valueInitializationDevice = nullptr;
                alreadySet = true;
            });
        }

        assert(m_dataFields->m_value != nullptr);
        if (!alreadySet)
        {
            // alreadySet is false, the lambda above wasn't called and the variable has been initialized before,
            // get a pointer to its value and simply copy the content of the supplied value.
            m_dataFields->m_value->CopyFrom(*value);
        }
    }

    static const std::wstring InitializerTypeAttributeName = L"initializerType";
    static const std::wstring OutputRankAttributeName = L"outputRank";
    static const std::wstring FilterRankAttributeName = L"filterRank";
    static const std::wstring ValueAttributeName = L"value";
    static const std::wstring ScaleAttributeName = L"scale";
    static const std::wstring RandomSeedAttributeName = L"randomSeed";
    static const std::wstring KernelWidthAttributeName = L"kernelWidth";
    static const std::wstring KernelHeightAttributeName = L"kernelHeight";

    void VariableFields::SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device)
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

    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, double scale, int outputRank, int filterRank, unsigned long seed)
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

    ParameterInitializer NormalInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::NormalInitializerTypeName, scale, outputRank, filterRank, seed);
    }

    ParameterInitializer XavierInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::XavierInitializerTypeName, scale, outputRank, filterRank, seed);
    }

    ParameterInitializer GlorotUniformInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotUniformInitializerTypeName, scale, outputRank, filterRank, seed);
    }

    ParameterInitializer GlorotNormalInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::GlorotNormalInitializerTypeName, scale, outputRank, filterRank, seed);
    }

    ParameterInitializer HeUniformInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeUniformInitializerTypeName, scale, outputRank, filterRank, seed);
    }

    ParameterInitializer HeNormalInitializer(double scale, int outputRank, int filterRank, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::HeNormalInitializerTypeName, scale, outputRank, filterRank, seed);
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

    Variable::Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
        : m_dataFields(MakeSharedObject<VariableFields>(shape, varType, dataType, nullptr, value, needsGradient, dynamicAxes, isSparse, name, uid))
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
        const auto& dynamicAxes = DynamicAxes();
        vector<DictionaryValue> dictionaryValueVector; 
        dictionaryValueVector.reserve(dynamicAxes.size());
        for (const auto& axis : dynamicAxes)
            dictionaryValueVector.push_back(axis);

        dict[dynamicAxisKey] = dictionaryValueVector;
        dict[isSparseKey] = IsSparse();
        if (!Name().empty())
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
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, uidKey, kindKey, dataTypeKey, dynamicAxisKey, isSparseKey, needsGradientKey, shapeKey };

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
        std::wstring name = L"";
        if (dict.Contains(nameKey))
            name = dict[nameKey].Value<std::wstring>();
        bool needsGradient = dict[needsGradientKey].Value<bool>();
        const auto& shape = dict[shapeKey].Value<NDShape>();

        if (kind == VariableKind::Constant || kind == VariableKind::Parameter)
        {
            auto& value = dict[valueKey].Value<NDArrayView>();

            // TODO: this copying here is redundant, value should be moved from the dictionary to the variable.
            // Also, the correct device should be used upfront when deserializing NDArrayView.
            Variable var(shape, kind, dataType, value.DeepClone(device, kind == VariableKind::Constant), needsGradient, dynamicAxis, isSparse, name, uid);
            if (var.IsParameter())
                return Parameter(var);
            else
                return Constant(var);
        }

        return Variable(shape, kind, dataType, nullptr, needsGradient, dynamicAxis, isSparse, name, uid);
    }

    Parameter::Parameter(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device, const std::wstring& name)
        : Variable(shape, VariableKind::Parameter, dataType, nullptr, true, {}, name, Internal::GenerateUid(VariableKind::Parameter))
    {
        m_dataFields->SetValueInitialization(initializer, device);
    }

    size_t Parameter::CurrentValueTimeStamp() const
    {
        return m_dataFields->m_valueTimeStamp.load(); 
    }

    void Parameter::RecordValueUpdate()
    {
        m_dataFields->m_valueTimeStamp++;
    }

    Constant::Constant(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device, const std::wstring& name)
        : Variable(shape, VariableKind::Constant, dataType, nullptr, false, {}, name, Internal::GenerateUid(VariableKind::Constant))
    {
        m_dataFields->SetValueInitialization(initializer, device);
    }
}
