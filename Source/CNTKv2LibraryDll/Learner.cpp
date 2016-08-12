//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Learner.h"
#include "TensorView.h"
#include "Utils.h"

#define UPDATE_FUNCTION                                                                                       \
    switch (smoothedGradientValue->GetDataType())                                                             \
    {                                                                                                         \
    case DataType::Float:                                                                                     \
        Update<float>(parameter, gradientValue, smoothedGradientValue, trainingSampleCount);                  \
        break;                                                                                                \
    case DataType::Double:                                                                                    \
        Update<double>(parameter, gradientValue, smoothedGradientValue, trainingSampleCount);                 \
        break;                                                                                                \
    default:                                                                                                  \
        NOT_IMPLEMENTED;                                                                                      \
    }

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace CNTK
{
    template <typename ElementType>
    /*static*/ shared_ptr<const Matrix<ElementType>> LearnerBase::GetMatrix(const NDArrayViewPtr& arrayView)
    {
        return arrayView->GetMatrix<ElementType>();
    }

    template <typename ElementType>
    /*static*/ shared_ptr<Matrix<ElementType>> LearnerBase::GetWritableMatrix(const NDArrayViewPtr& arrayView)
    {
        return arrayView->GetWritableMatrix<ElementType>();
    }

    template <typename ElementType>
    /*static*/ const TensorView<ElementType>* LearnerBase::GetTensorView(const NDArrayViewPtr& arrayView)
    {
        return arrayView->GetTensorView<ElementType>();
    }

    /*static*/ bool LearnerBase::HasNan(const NDArrayViewPtr& value, const char* name)
    {
        switch (value->GetDataType())
        {
        case DataType::Float:
            return value->GetMatrix<float>()->HasNan(name);
        case DataType::Double:
            return value->GetMatrix<double>()->HasNan(name);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(value->GetDataType()));
        }
    }

    /*static*/ void LearnerBase::Print(const NDArrayViewPtr& value, const char* msg)
    {
        switch (value->GetDataType())
        {
        case DataType::Float:
            value->GetMatrix<float>()->Print(msg);
            break;
        case DataType::Double:
            value->GetMatrix<double>()->Print(msg);
            break;
        default:
            LogicError("Unsupported DataType %s", DataTypeName(value->GetDataType()));
        }
    }

    // Clipping gradients to prevent outliers,
    template <typename ElementType>
    void LearnerBase::ClipGradient(Matrix<ElementType>& gradient, size_t actualMBSize) const
    {
        if (m_additionalOptions.gradientClippingThresholdPerSample != numeric_limits<double>::infinity())
        {
            double maxGradientPerMB = m_additionalOptions.gradientClippingThresholdPerSample * actualMBSize;
            if (m_additionalOptions.gradientClippingWithTruncation)
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

    // Performs additional preprocessing before calling the update method 
    // (gradient clipping and L2 regularization depending on the additional learning parameters).
    template <typename ElementType>
    void LearnerBase::PreProcess(const NDArrayViewPtr& parameterValue, const NDArrayViewPtr& gradientValue, size_t actualMBSize) const
    {
        const auto& gradientMatrix = gradientValue->GetWritableMatrix<ElementType>();

        // clipping gradients to prevent outliers
        ClipGradient<ElementType>(*gradientMatrix, actualMBSize);

        // L2 regularizer
        if (m_additionalOptions.l2RegularizationWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            auto weight = ElementType(m_additionalOptions.l2RegularizationWeight * actualMBSize);
            const auto& parameterMatrix = parameterValue->GetWritableMatrix<ElementType>();
            Matrix<ElementType>::ScaleAndAdd(weight, *parameterMatrix, *gradientMatrix);
        }
    }

    // Performs additional postprocessing after the update method has been executed
    // (noise injection and L1 regularization specified by the additional learning parameters).
    template <typename ElementType>
    void LearnerBase::PostProcess(const Parameter& parameter, const NDArrayViewPtr& gradientValue, size_t actualMBSize) const
    {
        const auto& parameterValue = parameter.Value();
        const auto& parameterMatrix = parameterValue->GetWritableMatrix<ElementType>();
        if (m_additionalOptions.gaussianNoiseInjectionStdDev > 0)
        {
            const auto& gradientMatrix = gradientValue->GetWritableMatrix<ElementType>();

            Matrix<ElementType> sgdUpdateNoise((DEVICEID_TYPE)parameterMatrix->GetDeviceId());

            // get the gradient structure since gradient is sparse
            sgdUpdateNoise.SetValue(*gradientMatrix);

            auto noiseStdDev = ElementType(m_additionalOptions.gaussianNoiseInjectionStdDev);

            // reset its value to random
            sgdUpdateNoise.SetGaussianRandomValue(ElementType(0.0), noiseStdDev);

            Matrix<ElementType>::ScaleAndAdd(ElementType(1.0), sgdUpdateNoise, *parameterMatrix);
        }

        // L1 regularizer with proximal gradient descent method
        if (m_additionalOptions.l1RegularizationWeight > 0)
        {
            auto learningRate = ElementType(ParameterDependentLearningRate(parameter));
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            auto weight = ElementType(learningRate * m_additionalOptions.l1RegularizationWeight * actualMBSize);
            parameterValue->GetWritableMatrix<ElementType>()->InplaceSoftThreshold(weight);
        }
    }

    template <typename ElementType>
    /*static*/ TensorView<ElementType>* LearnerBase::GetWritableTensorView(const NDArrayViewPtr& arrayView)
    {
        return arrayView->GetWritableTensorView<ElementType>();
    }

    LearnerBase::LearnerBase(const unordered_set<Parameter>& parameters, 
                             const LearningRatesPerSample& learningRates,
                             const LearningRateMultipliers& multipliers,
                             bool allocateSmoothGradients /* = true */)
        : Learner(parameters),
        m_learningRates(learningRates),
        m_sampleCount(0),
        m_minibatchCount(0)
    {
        LearningRateMultipliers::const_iterator it;
        const unordered_set<Parameter>& parameterSet = parameters;
        for (const auto& parameter : parameterSet)
        {
            if (multipliers.size() > 0 && (it = multipliers.find(parameter)) != multipliers.end())
            {
                 m_learningRateMultipliers.insert(make_pair(parameter, it->second));
            }
            else
            {
                m_learningRateMultipliers.insert(make_pair(parameter, 1.0));
            }

            if (!allocateSmoothGradients)
            {
                continue;
            }
                
            NDArrayViewPtr view;
            if (parameter.GetDataType() == DataType::Float)
            {
                view = MakeSharedObject<NDArrayView>(0.0f, parameter.Shape(), parameter.Value()->Device());
            }
            else
            {
                view = MakeSharedObject<NDArrayView>(0.0, parameter.Shape(), parameter.Value()->Device());
            }

            m_smoothedGradientValues.insert(make_pair(parameter, view));
        }
    }

    /*virtual*/ bool LearnerBase::Update(const unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) /*override*/
    {
        // make sure trainingSampleCount is a valid value
        assert(trainingSampleCount > 0);

        for (const auto& parameter : Parameters())
        {
            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            const auto& gradientValue = gradientValues.at(parameter);
// TODO: make this a runtime parameter.
#if DUMPOUTPUT
            LOGPRINTF(stderr, "Update_%ls\n", parameter.Name().c_str());
#endif

#ifdef _DEBUG
            if (HasNan(smoothedGradientValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in smoothedGradient.", parameter.Name().c_str());
#endif

#if DUMPOUTPUT
            // TODO: replace m_momentumPerSample with momentum:
            // const double momentum = MomentumPerMB(momentumPerSample, actualMBSize);
            LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
                        m_learningRatePerSample, m_momentumPerSample, trainingSampleCount);
            LOGPRINTF(stderr, "GradUpdateType()=%s, GradientUpdateNoiseStd()=%0.8f\n",
                        LearnerType().c_str(), m_GaussianNoiseInjectStd);
            Print(gradientValue, "Gradient Update");
            Print(smoothedGradientValue, "Smoothed Gradient Input");
#endif
            UPDATE_FUNCTION;

#if DUMPOUTPUT
            Print(parameterValue, "Parameter Update");
#endif

#ifdef _DEBUG
            const auto& parameterValue = parameter.Value();
            if (HasNan(parameterValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in parameter values after parameter update.", parameter.Name().c_str());
#endif
        }
        m_sampleCount += trainingSampleCount;
        m_minibatchCount++;
        return false;
    }

    template <typename ElementType>
    void LearnerBase::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        const auto& parameterValue = parameter.Value();
        PreProcess<ElementType>(parameterValue, gradientValue, trainingSampleCount);
        Update(parameter, gradientValue, smoothedGradientValue, trainingSampleCount);
        PostProcess<ElementType>(parameter, gradientValue, trainingSampleCount);
    }

    string LearnerBase::LearnerType() const
    {
        auto name = typeid(*this).name(); 
        if (strncmp(name, "class ", 6) == 0)
        {
            // On Windows, the type name contains "class" prefix. 
            // Return the actual name, omitting the prefix.
            return &name[6];
        } 
        return name;
    }

    /*virtual*/ Dictionary LearnerBase::GetCheckpointState() const /*override*/
    {
        Dictionary checkpoint;

        checkpoint[L"checkpointVersion"] = checkpointVersion;
        checkpoint[L"sampleCount"] = m_sampleCount;
        checkpoint[L"minibatchCount"] = m_minibatchCount;

        // TODO: should we also save learning rate schedule into the checkpoint?
        // If that is the case, need to be able to override this method in subclasses
        // and save momentum schedule as well.

        for (const auto& parameter : Parameters())
        {
            // TODO: parameter name is not guaranteed to be unique. Instead, all serializable objects
            // need to expose "UId" property -- a persistent unique internal name.
            // Switch to UId as soon as it's available.
            if (checkpoint.Contains(parameter.Name()))
            {
                LogicError("Parameter names must be unique");
            }

            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            checkpoint[parameter.Name()] = *smoothedGradientValue;
        }
        return checkpoint;
    }

    /*virtual*/ void LearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        m_sampleCount = checkpoint[L"sampleCount"].GetValue<size_t>();
        m_minibatchCount = checkpoint[L"minibatchCount"].GetValue<size_t>();

        for (const auto& parameter : Parameters())
        {
            if (!checkpoint.Contains(parameter.Name()))
            {
                LogicError("Checkpoint does not contain state for parameter %ls", parameter.Name().c_str());
            }

            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            const NDArrayView& checkpointedValue = checkpoint[parameter.Name()].GetValue<NDArrayView>();
            assert(smoothedGradientValue->GetDataType() == checkpointedValue.GetDataType());
            assert(smoothedGradientValue->Shape() == checkpointedValue.Shape());
            smoothedGradientValue->CopyFrom(checkpointedValue);
        }
    }

    /*virtual*/ void LearnerSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        const auto& parameterValue = parameter.Value();
        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue);

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));
        auto momentum = ElementType(MomentumPerMB(m_momentums[m_sampleCount], trainingSampleCount));

        // TODO: break up the NormalGrad into 3 different functions, each with its own set of parameters
        // (one for vanilla SGD, the other for momentum SGD, and the third one for NAG).
        smoothedGradientMatrix->NormalGrad(*gradientMatrix, *parameterMatrix,
                                           learningRate, momentum, m_useNesterovAcceleration);
    }

    LearnerAdaGrad::LearnerAdaGrad(const unordered_set<Parameter>& parameters, 
                                   const LearningRatesPerSample& learningRates,
                                   const LearningRateMultipliers& multipliers,
                                   bool needAveMultiplier)
        : LearnerBase(parameters, learningRates, multipliers), 
        m_needAveMultiplier(needAveMultiplier)
    {
    }

    /*virtual*/ void LearnerAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        UNUSED(trainingSampleCount);

        const auto& parameterValue = parameter.Value();
        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue);

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));

        auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
    }

    LearnerFSAdaGrad::LearnerFSAdaGrad(const unordered_set<Parameter>& parameters, 
                                       const LearningRatesPerSample& learningRates, 
                                       const MomentumsPerSample& momentums,
                                       const LearningRateMultipliers& multipliers)
        : LearnerMomentumSGD(parameters, learningRates, momentums, multipliers, /*allocateSmoothGradients*/ false)
    {
        for (const auto& parameter : parameters)
        {  
            
            // TODO: refactor.
            NDArrayViewPtr view;
            if (parameter.GetDataType() == DataType::Float)
            {
                auto matrix = GetMatrix<float>(parameter.Value());
                auto shape = NDShape({ matrix->GetNumRows(), 2 * matrix->GetNumCols() });
                view = MakeSharedObject<NDArrayView>(0.0f, shape, parameter.Value()->Device());
            }
            else
            {
                auto matrix = GetMatrix<double>(parameter.Value());
                auto shape = NDShape({ matrix->GetNumRows(), 2 * matrix->GetNumCols() });
                view = MakeSharedObject<NDArrayView>(0.0, shape, parameter.Value()->Device());
            }

            m_smoothedGradientValues.insert(make_pair(parameter, view));
        }
    }

    /*virtual*/ void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        UNUSED(trainingSampleCount);

        const auto& parameterValue = parameter.Value();
        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue);

        //const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));
        auto momentum = ElementType(MomentumPerMB(m_momentums[m_sampleCount], trainingSampleCount));
        smoothedGradientMatrix->FSAdagrad(trainingSampleCount, *gradientMatrix, *parameterMatrix, learningRate, momentum);
    }

    LearnerRMSProp::LearnerRMSProp(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates,
                                   double gamma, double inc, double dec, double max, double min,
                                   const LearningRateMultipliers& multipliers, bool needAveMultiplier)
                                   : LearnerBase(parameters, learningRates, multipliers, /*allocateSmoothGradients*/ false),
                                   m_gamma(gamma), m_inc(inc), m_dec(dec), m_max(max), m_min(min),
                                   m_needAveMultiplier(needAveMultiplier)
    {
        for (const auto& parameter : parameters)
        {  
            
            // TODO: refactor.
            NDArrayViewPtr view;
            size_t factor = 3;
            if (needAveMultiplier && parameter.Value()->Device().Type() == DeviceKind::GPU)
            {
                factor = 4;
            }

            if (parameter.GetDataType() == DataType::Float)
            {
                auto matrix = GetMatrix<float>(parameter.Value());
                auto shape = NDShape({ matrix->GetNumRows(), factor * matrix->GetNumCols() });
                view = MakeSharedObject<NDArrayView>(0.0f, shape, parameter.Value()->Device());
            }
            else
            {
                auto matrix = GetMatrix<double>(parameter.Value());
                auto shape = NDShape({ matrix->GetNumRows(), factor * matrix->GetNumCols() });
                view = MakeSharedObject<NDArrayView>(0.0, shape, parameter.Value()->Device());
            }

            m_smoothedGradientValues.insert(make_pair(parameter, view));
        }
    }

    /*virtual*/ void LearnerRMSProp::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerRMSProp::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        UNUSED(trainingSampleCount);

        const auto& parameterValue = parameter.Value();
        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue);

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));

        auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix,
                                                             ElementType(m_gamma), ElementType(m_inc),
                                                             ElementType(m_max), ElementType(m_dec),
                                                             ElementType(m_min), m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
    }

    // Explicit template instantiations
    template shared_ptr<Matrix<float>> LearnerBase::GetWritableMatrix<float>(const NDArrayViewPtr& arrayView);
    template shared_ptr<Matrix<double>> LearnerBase::GetWritableMatrix<double>(const NDArrayViewPtr& arrayView);
    
    LearnerPtr SGDLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates,
                          const LearningRateMultipliers& multipliers)
    {
        return MakeSharedObject<LearnerSGD>(parameters, learningRates, multipliers);
    }

    LearnerPtr MomentumSGDLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates, const MomentumsPerSample& momentums,
                                  const LearningRateMultipliers& multipliers)
    {
        return MakeSharedObject<LearnerMomentumSGD>(parameters, learningRates, momentums, multipliers);
    }

    LearnerPtr NesterovLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates, const MomentumsPerSample& momentums,
                               const LearningRateMultipliers& multipliers)
    {
        return MakeSharedObject<LearnerNesterov>(parameters, learningRates, momentums, multipliers);
    }

    LearnerPtr AdaGradLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates,
                              const LearningRateMultipliers& multipliers, bool needAveMultiplier)
    {
        return MakeSharedObject<LearnerAdaGrad>(parameters, learningRates, multipliers, needAveMultiplier);
    }

    LearnerPtr FSAdaGradLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates, const MomentumsPerSample& momentums,
                                const LearningRateMultipliers& multipliers)
    {
        return MakeSharedObject<LearnerFSAdaGrad>(parameters, learningRates, momentums, multipliers);
    }

    LearnerPtr RMSPropLearner(const unordered_set<Parameter>& parameters, const LearningRatesPerSample& learningRates,
                              double gamma, double inc, double dec, double max, double min, 
                              const LearningRateMultipliers& multipliers, bool needAveMultiplier)
    {
        return MakeSharedObject<LearnerRMSProp>(parameters, learningRates, gamma, inc, dec, max, min, multipliers, needAveMultiplier);
    }
}
