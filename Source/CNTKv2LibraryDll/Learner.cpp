#include "Learner.h"
#include "TensorView.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK 
{

static double MomentumPerMB(double momentumPerSample, size_t minibatchSize)
{
    return pow(momentumPerSample, minibatchSize);
}

LearnerBase::LearnerBase(const std::unordered_set<Variable>& parameters, 
    double learningRatePerSample, double momentumPerSample)
    : Learner(parameters),
    m_learningRatePerSample(learningRatePerSample),
    m_momentumPerSample(momentumPerSample),
    m_sampleCount(0)
{
    for (auto parameter : parameters)
    {
        if (parameter.DataType() == DataType::Float)
        {
            m_smoothedGradients[parameter] = new Value(new NDArrayView(0.0f, parameter.Shape()));
        }
        else
        {
            m_smoothedGradients[parameter] = new Value(new NDArrayView(0.0, parameter.Shape()));
        }
    }
}

/* virtual */ bool LearnerBase::Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameters,
    const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients,
    size_t trainingSampleCount) /* override */
{
    for (auto parameterVar : Parameters())
    {
        auto smoothedGradient = m_smoothedGradients[parameterVar];
        auto gradient = gradients[parameterVar];
        auto parameter = parameters[parameterVar];

        Update(parameterVar.DataType(), smoothedGradient, gradient, parameter, trainingSampleCount);
    }
    m_sampleCount += trainingSampleCount;
    return false;
}

SGD::SGD(const std::unordered_set<Variable>& parameters, double learningRatePerSample,
    double momentumPerSample, bool useNesterovAcceleration)
    : LearnerBase(parameters, learningRatePerSample, momentumPerSample),
    m_useNesterovAcceleration(useNesterovAcceleration)
{
}

template <typename ElementType>
void SGD::Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
    const ValuePtr parameter, size_t trainingSampleCount) const
{
    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

    smoothedGradientMatrix->NormalGrad(*gradientMatrix, *parameterMatrix, 
        ElementType(m_learningRatePerSample), ElementType(momentum), m_useNesterovAcceleration);
}


AdaGrad::AdaGrad(const std::unordered_set<Variable>& parameters, double learningRatePerSample,
    bool needAveMultiplier)
    : LearnerBase(parameters, learningRatePerSample),
    m_needAveMultiplier(needAveMultiplier)
{
}

template <typename ElementType>
void AdaGrad::Update(const ValuePtr smoothedGradient, const ValuePtr gradient, 
    const ValuePtr parameter, size_t trainingSampleCount) const
{
    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
    Matrix<ElementType>::ScaleAndAdd(ElementType(-m_learningRatePerSample / aveMultiplier), *gradientMatrix, *parameterMatrix);
}


FSAdaGrad::FSAdaGrad(const std::unordered_set<Variable>& parameters, double learningRatePerSample,
    double momentumPerSample)
    : LearnerBase(parameters, learningRatePerSample, momentumPerSample)
{
}

template <typename ElementType>
void FSAdaGrad::Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
    const ValuePtr parameter, size_t trainingSampleCount) const
{

    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

    smoothedGradientMatrix->FSAdagrad(trainingSampleCount, *gradientMatrix, *parameterMatrix,
        ElementType(m_learningRatePerSample), ElementType(momentum));
}



RmsProp::RmsProp(const std::unordered_set<Variable>& parameters, double learningRatePerSample, 
    RMSPropInfo info, bool needAveMultiplier)
    : LearnerBase(parameters, learningRatePerSample),
    m_info(info),
    m_needAveMultiplier(needAveMultiplier)
{
}

template <typename ElementType>
void RmsProp::Update(const ValuePtr smoothedGradient, const ValuePtr gradient, 
    const ValuePtr parameter, size_t trainingSampleCount) const
{
    UNUSED(trainingSampleCount);

    auto smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradient->Data());
    auto gradientMatrix = GetWritableMatrix<ElementType>(gradient->Data());
    auto parameterMatrix = GetWritableMatrix<ElementType>(parameter->Data());

    auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix, 
        ElementType(m_info.gamma), ElementType(m_info.inc), 
        ElementType(m_info.max), ElementType(m_info.dec), 
        ElementType(m_info.min), m_needAveMultiplier);
    Matrix<ElementType>::ScaleAndAdd(ElementType(-m_learningRatePerSample / aveMultiplier), *gradientMatrix, *parameterMatrix);
}

LearnerPtr SGDLearner(const std::unordered_set<Variable>& parameters, 
    double learningRatePerSample, double momentumPerSample, bool useNesterovAcceleration)
{
    return new SGD(parameters, learningRatePerSample, momentumPerSample, useNesterovAcceleration);
}

LearnerPtr AdaGradLearner(const std::unordered_set<Variable>& parameters, 
    double learningRatePerSample, bool needAveMultiplier)
{
    return new AdaGrad(parameters, learningRatePerSample, needAveMultiplier);
}

LearnerPtr FSAdaGradLearner(const std::unordered_set<Variable>& parameters,
    double learningRatePerSample, double momentumPerSample)
{
    return new FSAdaGrad(parameters, learningRatePerSample, momentumPerSample);
}

LearnerPtr RmsPropLearner(const std::unordered_set<Variable>& parameters, 
    double learningRatePerSample, double gamma, double inc, double dec, double max, double min, bool needAveMultiplier)
{
    return new RmsProp(parameters, learningRatePerSample, { gamma, inc, dec, max, min }, needAveMultiplier);
}

}