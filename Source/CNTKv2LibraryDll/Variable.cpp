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
    Variable::Variable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid) :
        InternalVariable(shape, varType, dataType, value, needsGradient, dynamicAxes, isSparse, /*isVolatile=*/false, name, uid)
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }
    Variable::Variable(NDShape&& shape, VariableKind varType, ::CNTK::DataType dataType, bool needsGradient, bool isSparse) :
        InternalVariable(std::move(shape), varType, dataType, needsGradient, isSparse, /*isVolatile=*/false)
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }

    Variable::Variable(const FunctionPtr& function) :
        Variable(function->Output())
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }

    // move-constructor variant, for Dynamite only
    Variable::Variable(FunctionPtr&& function) :
#ifdef DYNAMITE_ONLY // It's OK if user-held Variables are no outputs of composites as long as the graph is acyclic. That is always true in Dynamite.
        Variable(CompositePreservingCopy(*this, move(function)))
        // Note: Detour via static needed to allow for a sequence point between function->RawOutputs() and move(function).
#else
        InternalVariable(function->Output())
#endif
    {
    }
    /*static*/ Variable Variable::CompositePreservingCopy(const InternalVariable& other, ConstFunctionPtr&& composite)
    {
        const auto& output = composite->RawOutputs().front();
        return Variable((const InternalVariable&)output, move(composite), PrimitiveFunctionPtr());
        //return output.CompositePreservingCopy(move(composite));
    }

    // clear out a Variable when it is no longer needed, as to release associated resources
    // This is purely an optimization for use by Dynamite only.
    void Variable::Reset()
    {
        m_outputComposite.reset();
        m_acyclicOutputPrimitiveReference.reset();
        m_shapeDims = nullptr;
        InternalVariable::Reset();
    }
    void InternalVariable::Reset()
    {
        m_dataFields.reset();
    }

    Variable::Variable(const InternalVariable& other, const ConstFunctionPtr& composite, const ConstPrimitiveFunctionPtr& primitive) :
        InternalVariable(other), m_outputComposite(composite), m_acyclicOutputPrimitiveReference(primitive)
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }
    Variable::Variable(const InternalVariable& other, ConstFunctionPtr&& composite, const ConstPrimitiveFunctionPtr& primitive) :
        InternalVariable(other), m_outputComposite(std::move(composite)), m_acyclicOutputPrimitiveReference(primitive)
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }

    const NDShape& InternalVariable::Shape() const
    {
        return m_dataFields->m_shape; 
    }

    static const std::vector<Axis> c_noDynamicAxes;
    const std::vector<Axis>& InternalVariable::DynamicAxes() const
    {
        const auto& fields = *m_dataFields;
        return fields.HasMore() ? fields.m_more->m_dynamicAxes : c_noDynamicAxes;
    }

    VariableKind InternalVariable::Kind() const
    {
        return m_dataFields->m_varKind; 
    }

    bool InternalVariable::IsSparse() const
    {
        return m_dataFields->m_isSparse; 
    }

    const std::wstring& InternalVariable::Name() const
    {
        return m_dataFields->m_name.get(); 
    }

    void InternalVariable::DebugUpdateName(const std::wstring& newName)
    {
        m_dataFields->m_name = newName;
    }

    const std::wstring& InternalVariable::Uid() const
    {
        return m_dataFields->Uid();
    }
    DataType InternalVariable::GetDataType() const
    {
        return m_dataFields->m_dataType; 
    }

    bool InternalVariable::NeedsGradient() const
    {
        return m_dataFields->m_needsGradient; 
    }

    bool InternalVariable::IsVolatile() const
    {
        return m_dataFields->m_isVolatile;
    }

    InternalVariable InternalVariable::Clone() const
    {
        InternalVariable clonedVariable;
        clonedVariable.m_dataFields = m_dataFields->Clone();

        return clonedVariable;
    }

    Variable Variable::Clone() const
    {
        InternalVariable clonedVariable = InternalVariable::Clone();
        return Variable(clonedVariable, m_outputComposite, m_acyclicOutputPrimitiveReference);

        //clonedVariable.m_dataFields = m_dataFields->Clone();
        //return clonedVariable;
    }

    //const Variable& Variable::BlockFunctionVariableMapping() const
    //{
    //    return m_dataFields->More().m_blockFunctionVariableMapping;
    //}

    PrimitiveFunctionPtr InternalVariable::OutputOwner() const
    {
        if (!IsOutput())
            LogicError("OutputOwner: Must only be called on OutputVariables");
        auto owner = m_dataFields->Owner();
        if (!owner)
            LogicError("OutputOwner: Got an OutputVariable without owner??");
        return owner;
    }

    FunctionPtr InternalVariable::Owner() const
    {
        return m_dataFields->Owner();
    }

    bool InternalVariable::OwnerIs(const Function* f) const
    {
        return m_dataFields->OwnerIs(f);
    }

//    Variable Variable::CompositePreservingCopy(const ConstFunctionPtr& composite) const
//    {
//        //return Variable((const InternalVariable&)*this, composite, m_acyclicOutputPrimitiveReference);
//#if 1
//        // TODO: This breakpoint was never hit. Is it ever called? If not, remove.
//        return CompositePreservingCopy(move(std::shared_ptr<const Function>(composite))); // will call the move version below
//#else
//        // We have to preserve the whole subgraph.
//        Variable result;
//        // This must copy all data members except m_outputComposite.
//        result.m_outputComposite = composite;
//        result.m_dataFields = m_dataFields;
//        result.m_acyclicOutputPrimitiveReference = m_acyclicOutputPrimitiveReference;
//        result.m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
//        return result;
//#endif
//    }
//
//    // TODO: Move to header.
//    Variable Variable::CompositePreservingCopy(ConstFunctionPtr&& composite) const
//    {
//        return Variable((const InternalVariable&)*this, std::move(composite), m_acyclicOutputPrimitiveReference);
//        //// We have to preserve the whole subgraph.
//        //Variable result;
//        //// This must copy all data members except m_outputComposite.
//        //result.m_outputComposite = move(composite);
//        //result.m_dataFields = m_dataFields;
//        //result.m_acyclicOutputPrimitiveReference = m_acyclicOutputPrimitiveReference;
//        //result.m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
//        //return result;
//    }
//
//    Variable Variable::NonCompositePreservingCopy() const
//    {
//        return Variable((const InternalVariable&)*this, ConstFunctionPtr(), m_acyclicOutputPrimitiveReference);
//#if 1
//        //Variable result;
//        //// This must copy all data members except m_outputComposite.
//        //result.m_dataFields = m_dataFields;
//        //result.m_acyclicOutputPrimitiveReference = m_acyclicOutputPrimitiveReference;
//        //result.m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
//        //return result;
//#else
//        Variable copy = *this;
//        copy.m_outputComposite = nullptr;
//        return copy;
//#endif
//    }

    // downcast from InternalVariable to Variable
    // This is needed to allow Parameter and Constant to be passed as Variable.
    // Outputs, on the other hand, should always be cast while implanting a ref-count holder.
    //Variable::Variable(const InternalVariable& other) :
    //    InternalVariable(other)
    //{
    //    if (IsOutput())
    //        LogicError("Variable '%S' from InternalVariable: Should not be applied to Outputs.", AsString().c_str());
    //    m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    //}
    //
    //Variable::Variable(InternalVariable&& other) :
    //    InternalVariable(std::move(other))
    //{
    //    if (IsOutput())
    //        LogicError("Variable '%S' from InternalVariable: Should not be applied to Outputs.", AsString().c_str());
    //    m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    //}

    // special version. Use this for all places where Variable and InternalVariable cannot be easily disentangled.
    // This bypasses the IsOutput check. In the future, there should be no call to this.
    Variable::Variable(const InternalVariable& other, bool) :
        InternalVariable(other)
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }
    Variable::Variable(InternalVariable&& other, bool) :
        InternalVariable(std::move(other))
    {
        m_shapeDims = (decltype(m_shapeDims))&m_dataFields->m_shape.Dimensions().front();
    }
    Variable::Variable(const Parameter& other) : Variable(other, true) { }
    Variable::Variable(Parameter&& other) : Variable(std::move(other), true) { }
    Variable::Variable(const class Constant& other) : Variable(other, true) { }
    Variable::Variable(Constant&& other) : Variable(std::move(other), true) { }

    void InternalVariable::SetOwner(const std::weak_ptr<PrimitiveFunction>& ownerFunction)
    {
        if (Kind() != VariableKind::Output)
            LogicError("Variable '%S' SetOwner: Owner can only be set for Output Variables", AsString().c_str());

        if (!OwnerIs(nullptr))
            LogicError("Variable '%S' SetOwner: An Output Variable whose owner has previously been set, cannot be reset.", AsString().c_str());

        m_dataFields->m_ownerFunction = ownerFunction;
    }

    // simplified version for internal use only
    void InternalVariable::SetOwner(std::weak_ptr<PrimitiveFunction>&& ownerFunction)
    {
        m_dataFields->m_ownerFunction = move(ownerFunction);
    }

    Variable::operator FunctionPtr() const
    {
        // Note: This function does not get executed in dynamic networks.
        auto varOwner = Owner();
        if (varOwner)
            return AsComposite(varOwner, varOwner->Name());
        else
            return Combine({ *this });
    }

    NDArrayViewPtr InternalVariable::Value() const
    {
        if (IsInput() || IsPlaceholder())
        //if (!IsConstant() && !IsParameter())
            LogicError("Variable '%S' Value(): Only Variables of kind Parameter and Constant have a Value.", AsString().c_str());

        auto* more = m_dataFields->m_more.get();
        if (more && more->m_initValueFlag)
        {
            std::call_once(*more->m_initValueFlag, [=]{
                assert(more->m_value == nullptr);
                assert(more->m_valueInitializer);
                assert(more->m_valueInitializationDevice);

                switch (GetDataType())
                {
                case DataType::Float:
                {
                    m_dataFields->m_value = CreateValueFromParameterInitializer<float>(Shape(), *more->m_valueInitializer, *more->m_valueInitializationDevice);
                    break;
                }
                case DataType::Double:
                {
                    m_dataFields->m_value = CreateValueFromParameterInitializer<double>(Shape(), *more->m_valueInitializer, *more->m_valueInitializationDevice);
                    break;
                }
                default:
                    LogicError("Variable '%S' Value(): Unsupported DataType %s", AsString().c_str(), DataTypeName(GetDataType()));
                    break;
                }

                more->m_valueInitializer = nullptr;
                more->m_valueInitializationDevice = nullptr;
            });
        }

        // compute a knowable value if possible
        if (!m_dataFields->m_value)
        {
#if 0
            OutputOwner()->Forward();
#else
            OutputOwner()->BatchedForward();
#endif
        }

        assert(m_dataFields->m_value != nullptr);
        return m_dataFields->m_value;
    }

    void InternalVariable::Backward(std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr>& gradients, double beta) const
    {
        OutputOwner()->BatchedBackward(gradients, beta);
    }

    void InternalVariable::SetValue(const NDArrayViewPtr& value)
    {
        if (!(IsParameter() || IsConstant()))
            LogicError("Variable '%S' SetValue(): Can only be invoked on a Parameter or Constant variable.", AsString().c_str());
        else if (GetDataType() != value->GetDataType()) 
            LogicError("Variable '%S' SetValue(): 'source' and 'destination' have different data types.", AsString().c_str());
        if (Shape() != value->Shape() && (AsTensorShape(Shape()) != AsTensorShape(value->Shape())))
#if 1       // for expedience we just bypass the check --TODO: check whether non-inferred dimensions match, then update the inferred ones
            m_dataFields->m_shape = value->Shape();
#else
            LogicError("Variable '%S' SetValue(): 'source' shape '%S' differs 'destination' shape '%S'.", AsString().c_str(), value->Shape().AsString().c_str(), Shape().AsString().c_str());
#endif

        bool alreadySet = false;
        auto* more = m_dataFields->m_more.get();
        if (more && more->m_initValueFlag)
        {
            // In the case of lazy initialization, try to avoid the redundant call to the initializer. 
            std::call_once(*more->m_initValueFlag, [=, &value, &alreadySet] {
                // If the variable hasn't been initialized yet, clone the content of the supplied value and delete the initializer.
                m_dataFields->m_value = value->DeepClone(*more->m_valueInitializationDevice, false);
                more->m_valueInitializer = nullptr;
                more->m_valueInitializationDevice = nullptr;
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

    std::wstring InternalVariable::AsString() const
    {
        return m_dataFields->AsString();
    }

    static const std::wstring InitializerTypeAttributeName = L"initializerType";
    static const std::wstring OutputRankAttributeName = L"outputRank";
    static const std::wstring FilterRankAttributeName = L"filterRank";
    static const std::wstring ValueAttributeName = L"value";
    static const std::wstring ScaleAttributeName = L"scale";
    static const std::wstring RandomSeedAttributeName = L"randomSeed";
    static const std::wstring KernelWidthAttributeName = L"kernelWidth";
    static const std::wstring KernelHeightAttributeName = L"kernelHeight";

    std::wstring VariableFields::AsString() const
    {
        std::wstringstream wss;
        wss << VariableKindName(m_varKind) << "('";
        if (!m_name.empty())
            wss << m_name.c_str();
        else
            wss << Uid();
        bool reverse = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
        const auto& dynamicAxes = m_more ? m_more->m_dynamicAxes : c_noDynamicAxes;
        if (reverse)
            wss << "', " << DynamicAxesAsString(dynamicAxes, reverse) << ", " << m_shape.AsString() << ")";
        else
            wss << "', " << m_shape.AsString() << ", " << DynamicAxesAsString(dynamicAxes, reverse) << ")";
        return wss.str();
    }

    PrimitiveFunctionPtr VariableFields::Owner() const
    {
        if (IsObjectExpired(m_ownerFunction))
            LogicError("The owner function of Variable '%S' is unexpectedly expired.", AsString().c_str());

        auto ownerFunctionPtr = m_ownerFunction.lock();
#if 1
        return ownerFunctionPtr;
#else
        if (ownerFunctionPtr != nullptr)
            return ownerFunctionPtr->shared_from_this(); // TODO: it's already a shared_ptr... so why shared_from_this()?
        else
            return nullptr;
#endif
    }

    bool VariableFields::OwnerIs(const Function* f) const
    {
        auto ownerFunctionPtr = m_ownerFunction.lock();
        return ownerFunctionPtr.get() == f;
    }

    const std::wstring& VariableFields::Uid() const
    {
        // we create the uid string lazily, since we don't look at it for most nodes
        auto& uid = More().m_uid;
        if (uid.empty())
        {
            if (m_varKind == VariableKind::Output && !IsObjectExpired(m_ownerFunction) && Owner() && Owner()->IsPrimitive())
            {
                // in case of a primitive function, set uid of output vars to owner function uid + "_Output_" + output index.
                auto owner = Owner();
                auto outputs = owner->Outputs(); // TODO: RawOutputs() is sufficient here, but unfortunately not accessible
                size_t i;
                for (i = 0; i < outputs.size(); i++)
                    if (outputs[i].m_dataFields.get() == this)
                        break;
                uid = owner->Uid() + L"_" + VariableKindName(m_varKind) + L"_" + std::to_wstring(i);
            }
            else
                // otherwise just use the kind
                uid = Internal::GenerateUid(m_varKind);
        }
        return uid;
    }

    VariableFieldsPtr VariableFields::Clone() const
    {
        if (Owner() != nullptr)
            InvalidArgument("Output variable '%S' cannot be cloned.", AsString().c_str());

        // Note: We do not clone m_blockFunctionVariableMapping
        // TODO: Do VariableFields really need to use MakeSharedObject()? Can they transfer ownership across a DLL boundary?
        auto clone = MakeSharedObject1<VariableFields>(m_shape,
            m_varKind,
            m_dataType,
            m_ownerFunction,
            (m_value) ? m_value->DeepClone() : nullptr,
            m_needsGradient,
            HasMore() ? m_more->m_dynamicAxes : c_noDynamicAxes,
            m_isSparse,
            m_isVolatile,
            m_name.get(),
            std::wstring()/*Internal::GenerateUid(m_varKind)*/);

        if (HasMore() && m_more->m_valueInitializer)
            clone->SetValueInitialization(*m_more->m_valueInitializer, *m_more->m_valueInitializationDevice);

        return clone;
    }

    void VariableFields::SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device)
    {
        if (m_value != nullptr)
            LogicError("Variable '%S': Value initialization config cannot be set if a value already exists", AsString().c_str());

        assert(!m_valueInitializer);
        assert(!m_valueInitializationDevice);

        More();
        m_more->m_initValueFlag.reset(new std::once_flag());
        m_more->m_valueInitializer.reset(new ParameterInitializer(initializationConfig));

        if (m_more->m_valueInitializer->Contains(RandomSeedAttributeName)) {
            auto& seed = m_more->m_valueInitializer->operator[](RandomSeedAttributeName);
            if ((unsigned long)seed.Value<size_t>() == SentinelValueForAutoSelectRandomSeed)
                seed.Value<size_t>() = Internal::GenerateRandomSeed();
        }

        m_more->m_valueInitializationDevice.reset(new DeviceDescriptor(device));
    }

    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, double scale, unsigned long seed) 
    {
        if (scale <= 0) 
            InvalidArgument("CreateInitializer: scale value for initializer '%S' cannot be 0.", 
                initializerTypeName.c_str());

        Dictionary initConfig;
        initConfig[InitializerTypeAttributeName] = initializerTypeName;
        initConfig[ScaleAttributeName] = scale;
        // Initializers are sometimes created as default arguments in python.
        // If the value for an automatically-selected seed is assigned here, 
        // subsequent calls to SetFixedRandomSeed will be ignored.
        initConfig[RandomSeedAttributeName] = (size_t)seed;        
        return initConfig;
    }
    
    static ParameterInitializer CreateInitializer(const std::wstring& initializerTypeName, double scale, int outputRank, int filterRank, unsigned long seed)
    {
        if (scale <= 0)
            InvalidArgument("CreateInitializer: scale value for initializer '%S' cannot be 0.", 
                initializerTypeName.c_str());

        auto initConfig = CreateInitializer(initializerTypeName, scale, seed);
        initConfig[OutputRankAttributeName] = outputRank;
        initConfig[FilterRankAttributeName] = filterRank;
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
        return CreateInitializer(Microsoft::MSR::CNTK::UniformInitializerTypeName, scale, seed);
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

    ParameterInitializer TruncatedNormalInitializer(double scale, unsigned long seed)
    {
        return CreateInitializer(Microsoft::MSR::CNTK::TruncNormalInitializerTypeName, scale, seed);
    }

    InternalVariable::InternalVariable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, bool isVolatile, const std::wstring& name, const std::wstring& uid) :
        m_dataFields(MakeSharedObject1<VariableFields>(shape, varType, dataType, std::weak_ptr<PrimitiveFunction>(), value, needsGradient, dynamicAxes, isSparse, isVolatile, name, uid))
    {}

    InternalVariable::InternalVariable(NDShape&& shape, VariableKind varType, CNTK::DataType dataType, bool needsGradient, bool isSparse, bool isVolatile) :
        m_dataFields(MakeSharedObject1<VariableFields>(std::move(shape), varType, dataType, needsGradient, isSparse, isVolatile))
    {}

    // the others are default. They must be defined nevertheless, because of the incomplete type w.r.t. strong_shared_ptr in external uses
    InternalVariable::InternalVariable() = default;
    InternalVariable::~InternalVariable() = default;
    InternalVariable::InternalVariable(const InternalVariable&) = default;
    InternalVariable::InternalVariable(InternalVariable&&) = default;
    InternalVariable& InternalVariable::operator=(const InternalVariable&) = default;
    InternalVariable& InternalVariable::operator=(InternalVariable&&) = default;

    // the others are default. They must be defined nevertheless, because of the incomplete type w.r.t. strong_shared_ptr in external uses
    Variable::Variable() = default;
    Variable::~Variable() = default;
    Variable::Variable(const Variable&) = default;
    Variable::Variable(Variable&&) = default;
    Variable& Variable::operator=(const Variable&) = default;
    Variable& Variable::operator=(Variable&&) = default;

    template <typename ElementType>
    /*static*/ NDArrayViewPtr InternalVariable::CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device)
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
            // using this in place on an assert, which is ignored in Release mode.
            if (randomSeed == SentinelValueForAutoSelectRandomSeed) {
                RuntimeError("Unexpected 'auto-select' placeholder. At this point the seed should have a fixed value.");
            }

            auto scale = initConfig[ScaleAttributeName].Value<double>();
            int outputRank = DefaultParamInitOutputRank, filterRank = DefaultParamInitFilterRank;
            if (initializerType != Microsoft::MSR::CNTK::UniformInitializerTypeName && 
                initializerType != Microsoft::MSR::CNTK::TruncNormalInitializerTypeName)
            {
                outputRank = initConfig[OutputRankAttributeName].Value<int>();
                filterRank = initConfig[FilterRankAttributeName].Value<int>();

                if (outputRank == SentinelValueForInferParamInitRank)
                    outputRank = DefaultParamInitOutputRank;

                if (filterRank == SentinelValueForInferParamInitRank)
                    filterRank = DefaultParamInitFilterRank;

                if ((filterRank + outputRank) > shape.Rank())
                    InvalidArgument("Sum of filter rank (%d) and output rank (%d) of the parameter initializer cannot exceed the Parameter shape '%S' rank (%d)", filterRank, outputRank, shape.AsString().c_str(), (int)shape.Rank());
            }

            Microsoft::MSR::CNTK::LearnableParameter<ElementType>::InitRandom(*valueMatrix, AsTensorShape(shape), initializerType, randomSeed, (ElementType)scale,
                                                                              filterRank, outputRank, /*initOnCPUOnly=*/true,
                                                                              AsCNTKImplDeviceId(device));
        }

        return value;
    }

    static const std::wstring s_variableTypeValue = L"Variable";

    /*virtual*/ Dictionary InternalVariable::Serialize() const
    {
        if (IsOutput())
            LogicError("Variable '%S': Output variables cannot be saved.", AsString().c_str());

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
                LogicError("Uninitialized Parameter variable '%S' cannot be saved.", AsString().c_str());

            // TODO: add a dictionary value constructor with an rvalue parameter.
            dict[valueKey] = DictionaryValue(*value);
        }
        
        return dict;
    }

    /*static*/ InternalVariable InternalVariable::Deserialize(const Dictionary& dict, const CNTK::DeviceDescriptor& device)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, uidKey, kindKey, dataTypeKey, dynamicAxisKey, isSparseKey, needsGradientKey, shapeKey };

        size_t version = ValidateDictionary<InternalVariable>(dict, s_requiredDictionaryKeys, s_variableTypeValue, s_serializationVersion);

        const auto& uid = dict[uidKey].Value<std::wstring>();

        VariableKind kind = VariableKind(dict[kindKey].Value<std::size_t>());
        if (kind != VariableKind::Constant &&
            kind != VariableKind::Input &&
            kind != VariableKind::Parameter &&
            kind != VariableKind::Placeholder)
        {
            LogicError("Unexpected variable kind '%ls':'%u' (%s).",
                       kindKey.c_str(),
                       static_cast<std::underlying_type<VariableKind>::type>(kind),
                       GetVersionsString<InternalVariable>(s_serializationVersion, version).c_str());
        }
        
        DataType dataType = DataType(dict[dataTypeKey].Value<std::size_t>());
        if (dataType != DataType::Unknown &&
            dataType != DataType::Float &&
            dataType != DataType::Double)
        {
            LogicError("Unexpected variable datatype '%ls':'%u' (%s).", 
                       dataTypeKey.c_str(), 
                       static_cast<std::underlying_type<DataType>::type>(dataType),
                       GetVersionsString<InternalVariable>(s_serializationVersion, version).c_str());
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
            InternalVariable var(shape, kind, dataType, value.DeepClone(device, value.IsReadOnly()), needsGradient, dynamicAxis, isSparse, /*isVolatile=*/false, name, uid);
            if (var.IsParameter())
                return Parameter(var);
            else
                return Constant(var);
        }

        return InternalVariable(shape, kind, dataType, nullptr, needsGradient, dynamicAxis, isSparse, /*isVolatile=*/false, name, uid);
    }

    Parameter::Parameter(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device, const std::wstring& name)
        : InternalVariable(shape, VariableKind::Parameter, dataType, nullptr, true, {}, name, Internal::GenerateUid(VariableKind::Parameter))
    {

        m_dataFields->SetValueInitialization(initializer, device);
    }

    size_t InternalVariable::CurrentValueTimeStamp() const
    {
        if (!IsParameter() && !IsConstant())
            LogicError("Variable '%S' CurrentValueTimeStamp: Variable must be a Parameter or Constant", AsString().c_str());
        return m_dataFields->More().m_valueTimeStamp.load(); 
    }

    unsigned int InternalVariable::UniqueIdForDebugging() const
    {
        return m_dataFields->m_uniqueIdForDebugging;
    }

    void Parameter::RecordValueUpdate()
    {
        m_dataFields->More().m_valueTimeStamp++;
    }

#if 0
    void Parameter::TieValueWith(const Parameter& other)
    {
        // This is a bad hack, for debugging only.
        other.Value(); // flush initializer
        m_dataFields->m_dataType = other.m_dataFields->m_dataType;
        m_dataFields->m_initValueFlag.reset();
        m_dataFields->m_valueInitializer.reset();
        m_dataFields->m_valueInitializationDevice.reset();
        m_dataFields->m_shape = other.m_dataFields->m_shape;
        m_dataFields->m_value = other.m_dataFields->m_value;
        RecordValueUpdate();
    }
#endif

    Constant::Constant(const NDShape& shape, DataType dataType, bool isVolatile, const ParameterInitializer& initializer, const DeviceDescriptor& device, const std::wstring& name)
        : InternalVariable(shape, VariableKind::Constant, dataType, nullptr, /*needsGradient=*/false, {}, /*isSparse =*/ false, isVolatile, name, std::wstring())// Internal::GenerateUid(VariableKind::Constant))
    {
        m_dataFields->SetValueInitialization(initializer, device);
    }

    Constant Constant::CloneAs(DataType dataType) const
    {
        if (dataType != DataType::Double)
            InvalidArgument("Constant::Clone: Cannot clone Constant '%S' with DataType '%s' to DataType '%s'.", AsString().c_str(), DataTypeName(GetDataType()), DataTypeName(dataType));

        auto originalConstantValue = Value();
        NDArrayViewPtr newConstantValue;
        if (dataType == originalConstantValue->GetDataType())
        {
            newConstantValue = originalConstantValue;
            LogicError("This code is untested. Verfy it works, then delete this error message.");
        }
        else // if DataType is different then convert it on the CPU, and reassign it back
        {
            auto constantValueCPU = originalConstantValue->DeepClone(DeviceDescriptor::CPUDevice(), true);
            newConstantValue = CloneAsDataType(constantValueCPU, dataType, true);
        }
        return Constant(newConstantValue->DeepClone(originalConstantValue->Device(), originalConstantValue->IsReadOnly()), Name());
    }

    void Constant::RecordValueUpdate()
    {
        m_dataFields->More().m_valueTimeStamp++;
    }

    void Constant::SetValue(const NDArrayViewPtr& value)
    {
        InternalVariable::SetValue(value);
        RecordValueUpdate();
    }
}
