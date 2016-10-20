//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Learner.h"
#include "TensorView.h"
#include "Utils.h"
#include "Serialization.h"

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
            auto learningRate = ElementType(LearningRate());
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

    LearnerBase::LearnerBase(const vector<Parameter>& parameters, 
                             const LearningRatesPerSample& learningRates,
                             AdditionalLearningOptions additionalOptions,
                             bool allocateSmoothGradients /* = true */)
        : Learner(parameters, learningRates),
        m_minibatchCount(0),
        m_additionalOptions(additionalOptions)
    {
        std::unordered_set<Parameter> uniqueParameters(parameters.begin(), parameters.end());

        if (uniqueParameters.size() != parameters.size()) 
        {
            LogicError("Learner parameters contain duplicates.");
        }
        
        for (const auto& parameter : parameters)
        {
            if (!allocateSmoothGradients)
            {
                continue;
            }
                
            NDArrayViewPtr view = AllocateNDArrayView(parameter, parameter.Shape());
            m_smoothedGradientValues.insert(make_pair(parameter, view));
        }
    }

    /*static*/ NDArrayViewPtr LearnerBase::AllocateNDArrayView(const Parameter& parameter, const NDShape& shape) 
    {
        if (parameter.GetDataType() == DataType::Float)
        {
            return MakeSharedObject<NDArrayView>(float(0.0), shape, parameter.Value()->Device());
        }
        else
        {
            return MakeSharedObject<NDArrayView>(0.0, shape, parameter.Value()->Device());
        }
    }

    /*static*/ NDShape LearnerBase::GetMatrixShape(const Parameter& parameter)
    {
        if (parameter.GetDataType() == DataType::Float)
        {
           auto matrix = GetMatrix<float>(parameter.Value());
           return { matrix->GetNumRows(), matrix->GetNumCols() };
        }
        else
        {
           auto matrix = GetMatrix<double>(parameter.Value());
           return { matrix->GetNumRows(), matrix->GetNumCols() };
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
            LOGPRINTF(stderr, "Update_%ls\n", parameter.Uid().c_str());
#endif

#ifdef _DEBUG
            if (HasNan(smoothedGradientValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in smoothedGradient.", parameter.Uid().c_str());
#endif

#if DUMPOUTPUT
            auto learningRate = ElementType(LearningRate());
            auto momentum = ElementType(MomentumValueForMB(m_momentumValues[m_sampleCount], trainingSampleCount));
            LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
                        learningRate, momentum, trainingSampleCount);
            LOGPRINTF(stderr, "GradUpdateType()=%s, GradientUpdateNoiseStd()=%0.8f\n",
                      LearnerType().c_str(), m_additionalOptions.gaussianNoiseInjectionStdDev);
            Print(gradientValue, "Gradient Update");
            Print(smoothedGradientValue, "Smoothed Gradient Input");
#endif
            UPDATE_FUNCTION;

#if DUMPOUTPUT
            Print(parameter.Value(), "Parameter Update");
#endif

#ifdef _DEBUG
            const auto& parameterValue = parameter.Value();
            if (HasNan(parameterValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in parameter values after parameter update.", parameter.Uid().c_str());
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
        return Typename(this);
        } 

    static const std::wstring s_learnerTypeValue = L"Learner";

    /*virtual*/ Dictionary LearnerBase::Serialize() const /*override*/
    {
        Dictionary checkpoint;

        checkpoint[versionKey] = CurrentVersion();
        checkpoint[typeKey] = s_learnerTypeValue;
        checkpoint[sampleCountKey] = m_sampleCount;
        checkpoint[minibatchCountKey] = m_minibatchCount;
        checkpoint[learningRateScheduleKey] = m_learningRateSchedule.Serialize();

        // TODO: should we also save momentum schedule into the checkpoint?
        // If that is the case, need to be able to override this method in subclasses,
        // TODO: we now store mapping from UID to Parameter value in the checkpoint,
        // if uids turn out to be too fragile, this can be easily changed to a vector
        // of parameters, so that in restore we don't have to lookup based on uids, 
        // and simply restore parameters one by one in the original (dfs) order.
        for (const auto& parameter : Parameters())
        {
            if (checkpoint.Contains(parameter.Uid()))
            {
                LogicError("Parameter uids must be unique");
            }

            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            checkpoint[parameter.Uid()] = *smoothedGradientValue;
        }

        return checkpoint;
    }

    /*virtual*/ void LearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, sampleCountKey, minibatchCountKey, learningRateScheduleKey };
        
        ValidateDictionary<LearnerBase>(checkpoint, s_requiredDictionaryKeys, s_learnerTypeValue, CurrentVersion());

        m_sampleCount = checkpoint[sampleCountKey].Value<size_t>();
        m_minibatchCount = checkpoint[minibatchCountKey].Value<size_t>();
        // TODO: which learning rate schedule should take precedence here? 
        // The one given at construction time or the one loaded from a checkpoint?
        m_learningRateSchedule = TrainingParameterSchedule<double>::Deserialize(checkpoint[learningRateScheduleKey].Value<Dictionary>());

        auto parameters = Parameters();

        for (const auto& parameter : parameters)
        {
            if (!checkpoint.Contains(parameter.Uid()))
            {
                LogicError("Checkpoint does not contain state for parameter %ls", parameter.Uid().c_str());
            }

            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);

            const NDArrayView& checkpointedValue = checkpoint[parameter.Uid()].Value<NDArrayView>();
            
            if (smoothedGradientValue->GetDataType() != checkpointedValue.GetDataType())
            {
                LogicError("A value restored from a checkpoint for the smoothed gradient data type for parameter %ls does not match the expected value",
                           parameter.Uid().c_str());
            }

            if (smoothedGradientValue->Shape() != checkpointedValue.Shape())
            {
                LogicError("A value restored from a checkpoint for the smoothed gradient shape for parameter %ls does not match the expected value",
                           parameter.Uid().c_str());
            }

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

        auto learningRate = ElementType(LearningRate());
        auto momentum = ElementType(MomentumValueForMB(m_momentumValues[m_sampleCount], trainingSampleCount));

        // TODO: break up the NormalGrad into 3 different functions, each with its own set of parameters
        // Also, come up with a better name for NormalGrad (Default? Regular? Plain?).
        // (one for vanilla SGD, the other for momentum SGD, and the third one for NAG).
        smoothedGradientMatrix->NormalGrad(*gradientMatrix, *parameterMatrix,
                                           learningRate, momentum, m_useNesterovAcceleration);
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

        auto learningRate = ElementType(LearningRate());

        auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
    }

    LearnerFSAdaGrad::LearnerFSAdaGrad(const vector<Parameter>& parameters,
                                       const LearningRatesPerSample& learningRates, 
                                       const MomentumValuesPerSample& momentumValues,
                                       const double targetAdagradAvDenom,
                                       const size_t adagradT,
                                       AdditionalLearningOptions additionalOptions)
        : LearnerMomentumSGD(parameters, learningRates, momentumValues, additionalOptions, /*allocateSmoothGradients*/ false),
        m_targetAdagradAvDenom(targetAdagradAvDenom),
        m_adagradT(adagradT)
    {
        for (const auto& parameter : parameters)
        {  
            auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, {shape[0], 2 * shape[1]});
            m_smoothedGradientValues.insert(make_pair(parameter, view));
            m_smoothedCounts.insert(make_pair(parameter, 0.0));
        }
    }

    /*virtual*/ void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        const auto& parameterValue = parameter.Value();
        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue);
        
        auto learningRate = LearningRate();
        auto momentum = MomentumValueForMB(m_momentumValues[m_sampleCount], trainingSampleCount);

        const double varMomentum = (exp(-1.0 * trainingSampleCount / m_adagradT));
        double& smoothedCount = m_smoothedCounts.at(parameter); 

        smoothedGradientMatrix->FSAdagradUpdate(trainingSampleCount, *gradientMatrix, *parameterMatrix, smoothedCount, learningRate, m_targetAdagradAvDenom, momentum, varMomentum);
    }

    LearnerRMSProp::LearnerRMSProp(const vector<Parameter>& parameters, 
                                   const LearningRatesPerSample& learningRates,
                                   double gamma, double inc, double dec, double max, double min,
                                   bool needAveMultiplier,
                                   AdditionalLearningOptions additionalOptions)
    : LearnerBase(parameters, learningRates, additionalOptions, /*allocateSmoothGradients*/ false),
    m_gamma(gamma), m_inc(inc), m_dec(dec), m_max(max), m_min(min), m_needAveMultiplier(needAveMultiplier)
    {
        for (const auto& parameter : parameters)
        {  
            // When needAveMultiplier == true, CPU and GPU implementations of RMSProp require different number of columns.
            size_t factor = 3;
            if (needAveMultiplier && parameter.Value()->Device().Type() == DeviceKind::GPU)
            {
                factor = 4;
            }

            auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, {shape[0], factor * shape[1]});

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

        auto learningRate = ElementType(LearningRate());

        auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix,
                                                             ElementType(m_gamma), 
                                                             ElementType(m_inc),
                                                             ElementType(m_max), 
                                                             ElementType(m_dec),
                                                             ElementType(m_min), 
                                                             m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
    }

    // Explicit template instantiations
    template shared_ptr<Matrix<float>> LearnerBase::GetWritableMatrix<float>(const NDArrayViewPtr& arrayView);
    template shared_ptr<Matrix<double>> LearnerBase::GetWritableMatrix<double>(const NDArrayViewPtr& arrayView);
    
    LearnerPtr SGDLearner(const vector<Parameter>& parameters,
                          const LearningRatesPerSample& learningRates,
                          AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerSGD>(parameters, learningRates, additionalOptions);
    }

    LearnerPtr MomentumSGDLearner(const vector<Parameter>& parameters,
                                  const LearningRatesPerSample& learningRates,
                                  const MomentumValuesPerSample& momentumValues,
                                  AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerMomentumSGD>(parameters, learningRates, momentumValues, additionalOptions);
    }

    LearnerPtr NesterovLearner(const vector<Parameter>& parameters,
                               const LearningRatesPerSample& learningRates,
                               const MomentumValuesPerSample& momentumValues,
                               AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerNesterov>(parameters, learningRates, momentumValues, additionalOptions);
    }

    LearnerPtr FSAdaGradLearner(const vector<Parameter>& parameters,
                                const LearningRatesPerSample& learningRates,
                                const MomentumValuesPerSample& momentumValues,
                                const double targetAdagradAvDenom /*= 0.0025*/,
                                const size_t adagradT /*= 2 * 3600 * 100*/,
                                AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerFSAdaGrad>(parameters, learningRates, momentumValues, targetAdagradAvDenom, adagradT, additionalOptions);
    }

    LearnerPtr AdaGradLearner(const vector<Parameter>& parameters,
                              const LearningRatesPerSample& learningRates,
                              bool needAveMultiplier /*= true*/,
                              AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerAdaGrad>(parameters, learningRates, needAveMultiplier, additionalOptions);
    }

    LearnerPtr RMSPropLearner(const vector<Parameter>& parameters,
                              const LearningRatesPerSample& learningRates,
                              double gamma, double inc, double dec, double max, double min, 
                              bool needAveMultiplier /*= true*/,
                              AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerRMSProp>(parameters, learningRates, gamma, inc, dec, max, min, needAveMultiplier, additionalOptions);
    }
}
