#include "Learner.h"
#include "TensorView.h"
#include "Utils.h"

#define UPDATE_FUNCTION(type)                                                      \
     if(dtype == AsDataType<type>())                                                      \
     {                                                                            \
         PreProcess<type>(learnableParameter, gradient, parameter, trainingSampleCount);                  \
         Update<type>(learnableParameter, smoothedGradient, gradient, parameter, trainingSampleCount);  \
         PostProcess<type>(learnableParameter, gradient, parameter, trainingSampleCount);         \
         return; \
     }

#define UPDATE_FUNCTION_BODY              \
     auto dtype = smoothedGradient->Data()->GetDataType(); \
     UPDATE_FUNCTION(float)           \
     UPDATE_FUNCTION(double)          \
     NOT_IMPLEMENTED; 
     

using namespace ::CNTK::_Internal;
using namespace Microsoft::MSR::CNTK;
using namespace std;


namespace CNTK 
{

template <typename ElementType>
/*static*/ shared_ptr<const Matrix<ElementType>> LearnerBase::GetMatrix(const NDArrayViewPtr arrayView)
{
    return arrayView->GetMatrix<ElementType>();
}

template <typename ElementType>
/*static*/ shared_ptr<Matrix<ElementType>> LearnerBase::GetWritableMatrix(NDArrayViewPtr arrayView)
{
    return arrayView->GetWritableMatrix<ElementType>();
}

template <typename ElementType>
/*static*/ const TensorView<ElementType>* LearnerBase::GetTensorView(const NDArrayViewPtr arrayView)
{
    return arrayView->GetTensorView<ElementType>();
}

/*static*/ bool LearnerBase::HasNan(const ValuePtr& value, const char* name)
{
    auto data = value->Data();
    switch (data->GetDataType())
    {
    case DataType::Float:
        return data->GetMatrix<float>()->HasNan(name);
    case DataType::Double:
        return data->GetMatrix<double>()->HasNan(name);
    default:
        LogicError("Unsupported DataType %s", DataTypeName(data->GetDataType()));
        break;
    }
}

/*static*/ void LearnerBase::Print(const ValuePtr& value, const char* msg)
{
    auto data = value->Data();
    switch (data->GetDataType())
    {
    case DataType::Float:
        data->GetMatrix<float>()->Print(msg);
        break;
    case DataType::Double:
        data->GetMatrix<double>()->Print(msg);
        break;
    default:
        LogicError("Unsupported DataType %s", DataTypeName(data->GetDataType()));
        break;
    }
}

template <typename ElementType>
void LearnerBase::ClipGradient(Matrix<ElementType>& gradient, size_t actualMBSize) const
{
    if (m_clippingThresholdPerSample != std::numeric_limits<double>::infinity())
    {
        double maxGradientPerMB = m_clippingThresholdPerSample * actualMBSize;
        if (m_gradientClippingWithTruncation)
            gradient.InplaceTruncate(ElementType(maxGradientPerMB));
        else
        {
            // norm2 normalized
            double gradientNorm = gradient.FrobeniusNorm();
            if (gradientNorm > maxGradientPerMB)
            {
                double normFactor = maxGradientPerMB / gradientNorm;
                gradient *= ElementType(normFactor);
            }
        }
    }
}

template <typename ElementType>
void LearnerBase::PreProcess(const Variable& learnableParameter, const ValuePtr&  gradient, 
    const ValuePtr& parameter,  size_t actualMBSize) const
{
    UNUSED(learnableParameter);

    auto gradientMatrix = gradient->Data()->GetWritableMatrix<ElementType>();
    
    // clipping gradients to prevent outliers
    ClipGradient<ElementType>(*gradientMatrix, actualMBSize);

    // L2 regularizer
    if (m_L2RegWeight > 0)
    {
        // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
        auto weight = ElementType(m_L2RegWeight * actualMBSize);
        auto parameterMatrix = parameter->Data()->GetWritableMatrix<ElementType>();
        Matrix<ElementType>::ScaleAndAdd(weight, *parameterMatrix, *gradientMatrix);
    }
}

template <typename ElementType>
void LearnerBase::PostProcess(const Variable& learnableParameter, const ValuePtr&  gradient, 
    const ValuePtr& parameter, size_t actualMBSize) const
{
    auto parameterMatrix = parameter->Data()->GetWritableMatrix<ElementType>();
    if (m_GaussianNoiseInjectStd > 0)
    {
        auto gradientMatrix = gradient->Data()->GetWritableMatrix<ElementType>();

        Matrix<ElementType> sgdUpdateNoise((DEVICEID_TYPE) parameterMatrix->GetDeviceId());

        // get the gradient structure since gradient is sparse
        sgdUpdateNoise.SetValue(*gradientMatrix);

        // reset its value to random
        sgdUpdateNoise.SetGaussianRandomValue(ElementType(0.0), ElementType(m_GaussianNoiseInjectStd));

         Matrix<ElementType>::ScaleAndAdd(ElementType(1.0), sgdUpdateNoise, *parameterMatrix);
    }

     // L1 regularizer with proximal gradient descent method
    if (m_L1RegWeight > 0)
    {
        auto learningRate = ElementType(ParameterDependentLearningRate(learnableParameter));
        // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
        auto weight = ElementType(learningRate * m_L1RegWeight * actualMBSize);
        parameter->Data()->GetWritableMatrix<ElementType>()->InplaceSoftThreshold(weight);
    }
}

template <typename ElementType>
/*static*/ TensorView<ElementType>* LearnerBase::GetWritableTensorView(NDArrayViewPtr arrayView)
{
    return arrayView->GetWritableTensorView<ElementType>();
}

LearnerBase::LearnerBase(const _SimpleSet<Variable>& parameters, const Learner::AdditionalParameters& additionalParameters)
    : Learner(parameters),
    m_learningRatePerSample(0.0),
    m_momentumPerSample(0.0),
    m_L1RegWeight(additionalParameters.l1RegWeight),
    m_L2RegWeight(additionalParameters.l2RegWeight),
    m_GaussianNoiseInjectStd(additionalParameters.gaussianNoiseInjectStd),
    m_gradientClippingWithTruncation(additionalParameters.gradientClippingWithTruncation),
    m_clippingThresholdPerSample(additionalParameters.clippingThresholdPerSample),
    m_sampleCount(0)
{
    const unordered_set<Variable>& parameterSet = parameters;
    for (auto parameter : parameterSet)
    {
        // TODO: using the same device to allocate data for all smoothed gradients. Is this correct? 
        // Should the device be specified on the per-parameter basis?
        if (parameter.GetDataType() == DataType::Float)
        {
            m_smoothedGradients.Insert(parameter, new Value(new NDArrayView(0.0f, parameter.Shape(), additionalParameters.device))); 
        }
        else
        {
            m_smoothedGradients.Insert(parameter, new Value(new NDArrayView(0.0, parameter.Shape(), additionalParameters.device)));
        }

        if (additionalParameters.learningRateMultipliers.Contains(parameter))
        {
             m_learningRateMultipliers.Insert(parameter, additionalParameters.learningRateMultipliers[parameter]);
        }
        else
        {
             m_learningRateMultipliers.Insert(parameter, 1.0);
        }
    }
}

/* virtual */ bool LearnerBase::Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameters,
    const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients,
    size_t trainingSampleCount) /* override */
{
    // make sure trainingSampleCount is a valid value
    assert(trainingSampleCount > 0);

    for (const auto& learnableParameter : Parameters())
    {
        auto smoothedGradient = m_smoothedGradients[learnableParameter];
        auto gradient = gradients[learnableParameter];
        auto parameter = parameters[learnableParameter];

#if DUMPOUTPUT
            LOGPRINTF(stderr, "Update_%ls\n", learnableParameter.Name().c_str());
#endif

#ifdef _DEBUG
        if (HasNan(smoothedGradient, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
            LogicError("%ls has NaNs in smoothedGradient.", learnableParameter.Name().c_str());
#endif
        
#if DUMPOUTPUT
        LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
            m_learningRatePerSample, m_momentumPerSample, trainingSampleCount);
        LOGPRINTF(stderr, "GradUpdateType()=%ls, GradientUpdateNoiseStd()=%0.8f\n",
            LearnerType().c_str(), m_GaussianNoiseInjectStd);
        Print(gradient, "Gradient Update");
        Print(smoothedGradient, "Smoothed Gradient Input");
#endif

        Update(learnableParameter, smoothedGradient, gradient, parameter, trainingSampleCount);

#if DUMPOUTPUT
        Print(parameter, "Parameter Update");
#endif

#ifdef _DEBUG
        if (HasNan(parameter, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
            LogicError("%ls has NaNs in parameter values after parameter update.",  learnableParameter.Name().c_str());
#endif
    }
    m_sampleCount += trainingSampleCount;
    return false;
}

/* virtual */ Dictionary LearnerBase::GetCheckpointState() const /* override */ 
{
    Dictionary checkpoint;

    for (const auto& learnableParameter : Parameters())
    {
        if (checkpoint.Contains(learnableParameter.Name()))
        {
            // TODO: check uniqueness in the constructor?
            LogicError("Parameter names must be unique");
        }
        auto smoothedGradient = m_smoothedGradients[learnableParameter];

        // TODO: could also store things like dimensions, element size, format, etc.
        checkpoint[learnableParameter.Name()] = SerializeToVector(smoothedGradient->Data());
    }
    return checkpoint;
}

/* virtual */ void LearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint) /* override */
{
    for (const auto& learnableParameter : Parameters())
    {
        if (!checkpoint.Contains(learnableParameter.Name()))
        {
            LogicError("Checkpoint does not contain state for parameter %ls", learnableParameter.Name().c_str());
        }
        auto smoothedGradient = m_smoothedGradients[learnableParameter];

        const DictionaryValue& state = checkpoint[learnableParameter.Name()];

        DeserializeFromVector(smoothedGradient->Data(), state.GetValue<_Internal::_SimpleVector<DictionaryValue>>());
    }
}

Learners::SGDLearner::SGDLearner(const _SimpleSet<Variable>& parameters, bool useNesterovAcceleration, 
    const Learner::AdditionalParameters& additionalParameters)
    : LearnerBase(parameters, additionalParameters),
    m_useNesterovAcceleration(useNesterovAcceleration)
{
}

/*virtual*/ void Learners::SGDLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const /*override*/
{
    UPDATE_FUNCTION_BODY;
}

template <typename ElementType>
void Learners::SGDLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const
{
    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    //const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

    auto learningRate = ElementType(ParameterDependentLearningRate(learnableParameter));
   
    smoothedGradientMatrix->NormalGrad(*gradientMatrix, *parameterMatrix, 
        learningRate, ElementType(m_momentumPerSample), m_useNesterovAcceleration);
}


Learners::AdaGradLearner::AdaGradLearner(const _SimpleSet<Variable>& parameters, bool needAveMultiplier,
    const Learner::AdditionalParameters& additionalParameters)
    : LearnerBase(parameters, additionalParameters),
    m_needAveMultiplier(needAveMultiplier)
{
}

/*virtual*/ void Learners::AdaGradLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const /*override*/
{
    UPDATE_FUNCTION_BODY;
}

template <typename ElementType>
void Learners::AdaGradLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const
{
    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    auto learningRate = ElementType(ParameterDependentLearningRate(learnableParameter));

    auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
    Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
}


Learners::FSAdaGradLearner::FSAdaGradLearner(const _SimpleSet<Variable>& parameters,
    const Learner::AdditionalParameters& additionalParameters)
    : LearnerBase(parameters, additionalParameters)
{
}

/*virtual*/ void Learners::FSAdaGradLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const /*override*/
{
    UPDATE_FUNCTION_BODY;
}

template <typename ElementType>
void Learners::FSAdaGradLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const
{

    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    //const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

    auto learningRate = ElementType(ParameterDependentLearningRate(learnableParameter));

    smoothedGradientMatrix->FSAdagrad(trainingSampleCount, *gradientMatrix, *parameterMatrix,
        learningRate, ElementType(m_momentumPerSample));
}


Learners::RmsPropLearner::RmsPropLearner(const _SimpleSet<Variable>& parameters, RMSPropInfo info, 
    bool needAveMultiplier, const Learner::AdditionalParameters& additionalParameters)
    : LearnerBase(parameters, additionalParameters),
    m_info(info),
    m_needAveMultiplier(needAveMultiplier)
{
}

/*virtual*/ void Learners::RmsPropLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const /*override*/
{
    UPDATE_FUNCTION_BODY;
}

template <typename ElementType>
void Learners::RmsPropLearner::Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
    const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const
{
    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    auto learningRate = ElementType(ParameterDependentLearningRate(learnableParameter));

    auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix, 
        ElementType(m_info.gamma), ElementType(m_info.inc), 
        ElementType(m_info.max), ElementType(m_info.dec), 
        ElementType(m_info.min), m_needAveMultiplier);
    Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
}

LearnerPtr _SGDLearner(const _SimpleSet<Variable>& parameters, bool useNesterovAcceleration,
    const Learner::AdditionalParameters& additionalParameters)
{
    return new Learners::SGDLearner(parameters, useNesterovAcceleration, additionalParameters);
}

LearnerPtr _AdaGradLearner(const _SimpleSet<Variable>& parameters, bool needAveMultiplier,
    const Learner::AdditionalParameters& additionalParameters)
{
    return new Learners::AdaGradLearner(parameters, needAveMultiplier, additionalParameters);
}

LearnerPtr _FSAdaGradLearner(const _SimpleSet<Variable>& parameters, const Learner::AdditionalParameters& additionalParameters)
{
    return new Learners::FSAdaGradLearner(parameters, additionalParameters);
}

LearnerPtr _RmsPropLearner(const _SimpleSet<Variable>& parameters, 
    double gamma, double inc, double dec, double max, double min, bool needAveMultiplier,
    const Learner::AdditionalParameters& additionalParameters)
{
    return new Learners::RmsPropLearner(parameters, { gamma, inc, dec, max, min }, needAveMultiplier, additionalParameters);
}


// Explicit template instantiations
template CNTK_API void Learner::GetSmoothedGradients<float>(unordered_map<Variable, shared_ptr<Matrix<float>>>& list);
template CNTK_API void Learner::GetSmoothedGradients<double>(unordered_map<Variable, shared_ptr<Matrix<double>>>& list);
template shared_ptr<Matrix<float>> LearnerBase::GetWritableMatrix<float>(const NDArrayViewPtr arrayView);
template shared_ptr<Matrix<double>> LearnerBase::GetWritableMatrix<double>(const NDArrayViewPtr arrayView);

}