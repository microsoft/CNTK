//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Learner.h"
#include "TensorView.h"
#include "Utils.h"
#include "Serialization.h"

#define DISPATCH_TO_TYPED_UPDATE_FUNCTION                                                                     \
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

#define GET_WRITABLE_MATRICES                                                                                 \
    const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue);               \
    const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);                               \
    const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameter.Value());

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

    void LearnerBase::ResetSmoothedGradients()
    {
        for(auto v : m_smoothedGradientValues)
        {
            if (v.second->GetDataType() == DataType::Float)
                v.second->SetValue(0.0f);
            else if (v.second->GetDataType() == DataType::Double)
                v.second->SetValue(0.0);
            else
                LogicError("Unsupported DataType %s", DataTypeName(v.second->GetDataType()));
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
            const auto weight = m_additionalOptions.l2RegularizationWeight * actualMBSize;
            const auto& parameterMatrix = parameterValue->GetWritableMatrix<ElementType>();
            Matrix<ElementType>::ScaleAndAdd(ElementType(weight), *parameterMatrix, *gradientMatrix);
        }
    }

    // Performs additional postprocessing after the update method has been executed
    // (noise injection and L1 regularization specified by the additional learning parameters).
    template <typename ElementType>
    void LearnerBase::PostProcess(const Parameter& parameter, const NDArrayViewPtr& gradientValue, size_t actualMBSize) const
    {
        const auto& parameterValue = parameter.Value();
        const auto& parameterMatrix = parameterValue->GetWritableMatrix<ElementType>();
        const auto gaussianNoiseInjectionStdDev = GetCurrentTrainingParameterValue(m_additionalOptions.gaussianNoiseInjectionStdDev);
        if (gaussianNoiseInjectionStdDev > 0)
        {
            const auto& gradientMatrix = gradientValue->GetWritableMatrix<ElementType>();

            Matrix<ElementType> sgdUpdateNoise((DEVICEID_TYPE)parameterMatrix->GetDeviceId());

            // get the gradient structure since gradient is sparse
            sgdUpdateNoise.SetValue(*gradientMatrix);

            const auto noiseStdDev = gaussianNoiseInjectionStdDev;

            // reset its value to random
            sgdUpdateNoise.SetGaussianRandomValue(ElementType(0.0), ElementType(noiseStdDev));

            Matrix<ElementType>::ScaleAndAdd(ElementType(1.0), sgdUpdateNoise, *parameterMatrix);
        }

        // L1 regularizer with proximal gradient descent method
        if (m_additionalOptions.l1RegularizationWeight > 0)
        {
            const auto learningRate = LearningRate(actualMBSize);
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            const auto weight = learningRate * m_additionalOptions.l1RegularizationWeight * actualMBSize;
            parameterValue->GetWritableMatrix<ElementType>()->InplaceSoftThreshold(ElementType(weight));
        }
    }

    template <typename ElementType>
    /*static*/ TensorView<ElementType>* LearnerBase::GetWritableTensorView(const NDArrayViewPtr& arrayView)
    {
        return arrayView->GetWritableTensorView<ElementType>();
    }

    LearnerBase::LearnerBase(const vector<Parameter>& parameters,
                             const LearningRateSchedule& learningRateSchedule,
                             AdditionalLearningOptions additionalOptions,
                             bool allocateSmoothGradients /* = true */)
                             : Learner(parameters, learningRateSchedule),
                             m_minibatchCount(0),
                             m_additionalOptions(additionalOptions)
    {
        std::unordered_set<Parameter> uniqueParameters(parameters.begin(), parameters.end());

        if (uniqueParameters.size() != parameters.size())
        {
            LogicError("Learner parameters contain duplicates.");
        }

        if (allocateSmoothGradients)
        {
            for (const auto& parameter : parameters)
            {
                NDArrayViewPtr view = AllocateNDArrayView(parameter, parameter.Shape());
                m_smoothedGradientValues.emplace(parameter, view);
            }
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
            return{ matrix->GetNumRows(), matrix->GetNumCols() };
        }
        else
        {
            auto matrix = GetMatrix<double>(parameter.Value());
            return{ matrix->GetNumRows(), matrix->GetNumCols() };
        }
    }

    /*virtual*/ bool LearnerBase::Update(unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd) /*override*/
    {
        if (LearningRate(trainingSampleCount) == 0.0)
        {
            return false;
        }

        // make sure trainingSampleCount is a valid value
        if (trainingSampleCount == 0)
        {
            InvalidArgument("Learner::Update(): cannot perform an update with an empty minibatch.");
        }

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
            const auto learningRate = LearningRate(trainingSampleCount);
            const auto momentum = MomentumValueForMB(trainingSampleCount);
            LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
                      learningRate, momentum, trainingSampleCount);
            LOGPRINTF(stderr, "GradUpdateType()=%s, GradientUpdateNoiseStd()=%0.8f\n",
                      LearnerType().c_str(), m_additionalOptions.gaussianNoiseInjectionStdDev);
            Print(gradientValue, "Gradient Update");
            Print(smoothedGradientValue, "Smoothed Gradient Input");
#endif
            DISPATCH_TO_TYPED_UPDATE_FUNCTION;

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
        if (sweepEnd)
        {
            m_sweepCount++;
        }

        return true;
    }

    template <typename ElementType>
    void LearnerBase::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                             const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        const auto& parameterValue = parameter.Value();
        PreProcess<ElementType>(parameterValue, gradientValue, trainingSampleCount);
        Update(parameter, gradientValue, smoothedGradientValue, trainingSampleCount);
        PostProcess<ElementType>(parameter, gradientValue, trainingSampleCount);

        auto paramRef = parameter;
        paramRef.RecordValueUpdate();
    }

    string LearnerBase::LearnerType() const
    {
        return Typename(this);
    }

    static const std::wstring s_learnerTypeValue = L"Learner";

    /*virtual*/ Dictionary LearnerBase::CreateCheckpoint() /*override*/
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

        const auto parameters = Parameters();

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

    LearnerSGD::LearnerSGD(const std::vector<Parameter>& parameters, 
                           const LearningRateSchedule& learningRateSchedule, 
                           AdditionalLearningOptions additionalOptions,
                           bool allocateSmoothGradients)
                           : LearnerBase(parameters, learningRateSchedule, additionalOptions, allocateSmoothGradients)
    {
        if (!allocateSmoothGradients)
        {
            // the vanilla sgd does not need the smooth gradients per se, 
            // insert dummy nd views instead.
            for (const auto& parameter : parameters)
            {
                m_smoothedGradientValues.emplace(parameter, AllocateNDArrayView(parameter, {}));
            }
        }
    }

    /*virtual*/ void LearnerSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                        const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                            const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        UNUSED(smoothedGradientValue);
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue);
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameter.Value());
        const auto learningRate = ElementType(LearningRate(trainingSampleCount));

        parameterMatrix->SGDUpdate(*gradientMatrix, learningRate);
    }

    double LearnerMomentumSGD::MomentumValueForMB(const MomentumSchedule& schedule, size_t minibatchSize) const
    {
        double currentMomentum = GetCurrentTrainingParameterValue(schedule);
        if (schedule.Unit() == MomentumSchedule::UnitType::Minibatch)
        {
            return currentMomentum;
        }
        return std::pow(currentMomentum, minibatchSize);
    }

    /*virtual*/ void LearnerMomentumSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                                const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerMomentumSGD::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                    const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES;

        const auto learningRate = ElementType(LearningRate(trainingSampleCount));
        const auto momentum = ElementType(MomentumValueForMB(trainingSampleCount));

        parameterMatrix->MomentumSGDUpdate(*gradientMatrix, *smoothedGradientMatrix,
                                           learningRate, momentum, UseUnitGainMomentum());
    }

    /*virtual*/ void LearnerNesterov::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                             const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerNesterov::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                 const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES;

        const auto learningRate = ElementType(LearningRate(trainingSampleCount));
        const auto momentum = ElementType(MomentumValueForMB(trainingSampleCount));

        parameterMatrix->NesterovAcceleratedMomentumSGDUpdate(*gradientMatrix, *smoothedGradientMatrix,
                                                              learningRate, momentum, UseUnitGainMomentum());
    }

    LearnerAdaGrad::LearnerAdaGrad(const std::vector<Parameter>& parameters,
                                   const LearningRateSchedule& learningRateSchedule,
                                   bool needAveMultiplier,
                                   AdditionalLearningOptions additionalOptions)
                                   : LearnerBase(parameters, learningRateSchedule, additionalOptions, /*allocateSmoothGradients*/ false),
                                   m_needAveMultiplier(needAveMultiplier)
    {
        for (const auto& parameter : parameters)
        {
            // When needAveMultiplier == true, CPU and GPU implementations of LearnerAdaGrad require different number of columns.
            size_t factor = 1;
            if (needAveMultiplier && parameter.Value()->Device().Type() == DeviceKind::GPU)
            {
                factor = 2;
            }

            const auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, { shape[0], factor * shape[1] });

            m_smoothedGradientValues.emplace(parameter, view);
        }
    }

    /*virtual*/ void LearnerAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                            const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES

        const auto learningRate = LearningRate(trainingSampleCount);

        const auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);
    }

    /*static*/ const double LearnerFSAdaGrad::s_targetAdagradAvDenom = 1.0;

    LearnerFSAdaGrad::LearnerFSAdaGrad(const vector<Parameter>& parameters,
                                       const LearningRateSchedule& learningRateSchedule,
                                       const MomentumSchedule& momentumSchedule,
                                       bool unitGain,
                                       const MomentumSchedule& varianceMomentumSchedule,
                                       AdditionalLearningOptions additionalOptions)
                                       : LearnerMomentumSGD(parameters, learningRateSchedule, momentumSchedule, 
                                                            unitGain, additionalOptions, /*allocateSmoothGradients*/ false),
                                       m_varianceMomentumSchedule(varianceMomentumSchedule)
    {
        for (const auto& parameter : parameters)
        {
            const auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, { shape[0], 2 * shape[1] });
            m_smoothedGradientValues.emplace(parameter, view);
            m_smoothedCounts.emplace(parameter, 0.0);
        }
    }

    /*virtual*/ void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                              const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerFSAdaGrad::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                  const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES;

        const auto learningRate = LearningRate(trainingSampleCount);
        const auto momentum = MomentumValueForMB(trainingSampleCount);

        const auto varMomentum = VarianceMomentumValueForMB(trainingSampleCount);

        double& smoothedCount = m_smoothedCounts.at(parameter);

        smoothedGradientMatrix->FSAdagradUpdate(trainingSampleCount, *gradientMatrix, *parameterMatrix, smoothedCount, learningRate, 
                                                s_targetAdagradAvDenom, momentum, varMomentum, UseUnitGainMomentum());
    }

    LearnerAdam::LearnerAdam(const vector<Parameter>& parameters,
        const LearningRateSchedule& learningRateSchedule,
        const MomentumSchedule& momentumSchedule,
        bool unitGain,
        const MomentumSchedule& varianceMomentumSchedule,
        AdditionalLearningOptions additionalOptions)
        : LearnerMomentumSGD(parameters, learningRateSchedule, momentumSchedule,
            unitGain, additionalOptions, /*allocateSmoothGradients*/ false),
        m_varianceMomentumSchedule(varianceMomentumSchedule)
    {
        for (const auto& parameter : parameters)
        {
            const auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, { shape[0], 2 * shape[1] });
            m_smoothedGradientValues.emplace(parameter, view);
            m_smoothedCounts.emplace(parameter, 0.0);
        }
    }

    /*virtual*/ void LearnerAdam::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue,
        const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerAdam::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue,
        const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES;

        const auto learningRate = LearningRate(trainingSampleCount);
        const auto momentum = MomentumValueForMB(trainingSampleCount);

        const auto varMomentum = VarianceMomentumValueForMB(trainingSampleCount);

        double& smoothedCount = m_smoothedCounts.at(parameter);

        smoothedGradientMatrix->AdamUpdate(*gradientMatrix, *parameterMatrix, smoothedCount, learningRate,
            momentum, varMomentum, UseUnitGainMomentum());
    }

    LearnerRMSProp::LearnerRMSProp(const vector<Parameter>& parameters,
                                   const LearningRateSchedule& learningRateSchedule,
                                   double gamma, double inc, double dec, double max, double min,
                                   bool needAveMultiplier,
                                   AdditionalLearningOptions additionalOptions)
                                   : LearnerBase(parameters, learningRateSchedule, additionalOptions, /*allocateSmoothGradients*/ false),
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

            const auto shape = GetMatrixShape(parameter);
            NDArrayViewPtr view = AllocateNDArrayView(parameter, { shape[0], factor * shape[1] });

            m_smoothedGradientValues.emplace(parameter, view);
        }
    }

    /*virtual*/ void LearnerRMSProp::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                            const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const /*override*/
    {
        DISPATCH_TO_TYPED_UPDATE_FUNCTION;
    }

    template <typename ElementType>
    void LearnerRMSProp::Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, 
                                const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const
    {
        GET_WRITABLE_MATRICES;

        const auto learningRate = LearningRate(trainingSampleCount);

        const auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix,
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
                          const LearningRateSchedule& learningRateSchedule,
                          AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerSGD>(parameters, learningRateSchedule, additionalOptions);
    }

    LearnerPtr MomentumSGDLearner(const vector<Parameter>& parameters,
                                  const LearningRateSchedule& learningRateSchedule,
                                  const MomentumSchedule& momentumSchedule,
                                  bool unitGain,
                                  AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerMomentumSGD>(parameters, learningRateSchedule, momentumSchedule, unitGain, additionalOptions);
    }

    LearnerPtr NesterovLearner(const vector<Parameter>& parameters,
                               const LearningRateSchedule& learningRateSchedule,
                               const MomentumSchedule& momentumSchedule,
                               bool unitGain,
                               AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerNesterov>(parameters, learningRateSchedule, momentumSchedule, unitGain, additionalOptions);
    }

    LearnerPtr AdamLearner(const vector<Parameter>& parameters,
                           const LearningRateSchedule& learningRateSchedule,
                           const MomentumSchedule& momentumSchedule,
                           bool unitGain, /*=true*/
                           const MomentumSchedule& varianceMomentumSchedule, /*= MomentumAsTimeConstantSchedulePerSample(2 * 3600 * 100)*/
                           bool lowMemory, /*= true*/
                           AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        // TODO: Due to history reason, the legacy AdamLearner using FSAdaGrad implementation instead of the original paper implementation.
        //      To keep interface backward compatible, the new adam will be enabled only when lowMemory is false.
        if (!lowMemory)
        {
            return MakeSharedObject<LearnerAdam>(parameters, learningRateSchedule, momentumSchedule, unitGain, varianceMomentumSchedule, additionalOptions);
        }
        else
        {
            return MakeSharedObject<LearnerFSAdaGrad>(parameters, learningRateSchedule, momentumSchedule, unitGain, varianceMomentumSchedule, additionalOptions);
        }
    }

    LearnerPtr AdaGradLearner(const vector<Parameter>& parameters,
                              const LearningRateSchedule& learningRateSchedule,
                              bool needAveMultiplier /*= true*/,
                              AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerAdaGrad>(parameters, learningRateSchedule, needAveMultiplier, additionalOptions);
    }

    LearnerPtr RMSPropLearner(const vector<Parameter>& parameters,
                              const LearningRateSchedule& learningRateSchedule,
                              double gamma, double inc, double dec, double max, double min,
                              bool needAveMultiplier /*= true*/,
                              AdditionalLearningOptions additionalOptions /*= AdditionalLearningOptions()*/)
    {
        return MakeSharedObject<LearnerRMSProp>(parameters, learningRateSchedule, gamma, inc, dec, max, min, needAveMultiplier, additionalOptions);
    }
}
