//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Learner.h"
#include "TensorView.h"
#include "Utils.h"

#define UPDATE_FUNCTION                                                                                     \
    if (!(Update<float>(parameter, smoothedGradientValue, gradientValue, parameterValue, trainingSampleCount) ||  \
        Update<double>(parameter, smoothedGradientValue, gradientValue, parameterValue, trainingSampleCount)))    \
        NOT_IMPLEMENTED;

#define CHECK_DATA_TYPE(valuePtr)                                       \
    if (valuePtr->Data()->GetDataType() != AsDataType<ElementType>()) \
    { return false; }                                   


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
        const auto& data = value->Data();
        switch (data->GetDataType())
        {
        case DataType::Float:
            return data->GetMatrix<float>()->HasNan(name);
        case DataType::Double:
            return data->GetMatrix<double>()->HasNan(name);
        default:
            LogicError("Unsupported DataType %s", DataTypeName(data->GetDataType()));
        }
    }

    /*static*/ void LearnerBase::Print(const ValuePtr& value, const char* msg)
    {
        const auto& data = value->Data();
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
    bool LearnerBase::PreProcess(const ValuePtr& gradientValue,const ValuePtr& parameterValue, size_t actualMBSize) const
    {
        CHECK_DATA_TYPE(gradientValue);

        const auto& gradientMatrix = gradientValue->Data()->GetWritableMatrix<ElementType>();

        // clipping gradients to prevent outliers
        ClipGradient<ElementType>(*gradientMatrix, actualMBSize);

        // L2 regularizer
        if (m_additionalOptions.l2RegularizationWeight > 0)
        {
            // multiply by actualMBSize so that it's invariant to minibatch size since learning rate is per sample
            auto weight = ElementType(m_additionalOptions.l2RegularizationWeight * actualMBSize);
            const auto& parameterMatrix = parameterValue->Data()->GetWritableMatrix<ElementType>();
            Matrix<ElementType>::ScaleAndAdd(weight, *parameterMatrix, *gradientMatrix);
        }

        return true;
    }

    // Performs additional postprocessing after the update method has been executed
    // (noise injection and L1 regularization specified by the additional learning parameters).
    template <typename ElementType>
    bool LearnerBase::PostProcess(const Variable& parameter, const ValuePtr& gradientValue,
                                    const ValuePtr& parameterValue, size_t actualMBSize) const
    {
        CHECK_DATA_TYPE(gradientValue);

        const auto& parameterMatrix = parameterValue->Data()->GetWritableMatrix<ElementType>();
        if (m_additionalOptions.gaussianNoiseInjectionStdDev > 0)
        {
            const auto& gradientMatrix = gradientValue->Data()->GetWritableMatrix<ElementType>();

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
            parameterValue->Data()->GetWritableMatrix<ElementType>()->InplaceSoftThreshold(weight);
        }

        return true;
    }

    template <typename ElementType>
    /*static*/ TensorView<ElementType>* LearnerBase::GetWritableTensorView(NDArrayViewPtr arrayView)
    {
        return arrayView->GetWritableTensorView<ElementType>();
    }

    LearnerBase::LearnerBase(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
        : Learner(parameters),
        m_learningRatePerSample(0.0),
        m_sampleCount(0)
    {
        const unordered_set<Variable>& parameterSet = parameters;
        for (const auto& parameter : parameterSet)
        {
            // TODO: using the same device to allocate data for all smoothed gradients. Is this correct?
            // Should the device be specified on the per-parameter basis?
            NDArrayViewPtr view;
            if (parameter.GetDataType() == DataType::Float)
            {
                view = MakeSharedObject<NDArrayView>(0.0f, parameter.Shape(), device);
            }
            else
            {
                view = MakeSharedObject<NDArrayView>(0.0f, parameter.Shape(), device);
            }

            m_smoothedGradientValues.insert(make_pair(parameter, MakeSharedObject<Value>(view)));
            m_additionalOptions.learningRateMultipliers.insert(make_pair(parameter, 1.0));
        }
    }

    void LearnerBase::ResetSmoothedGradients()
    {
        for (const auto& parameter : Parameters())
        {
            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            const auto& data = smoothedGradientValue->Data();
            switch (data->GetDataType())
            {
            case DataType::Float:
                data->SetValue(0.0f);
                break;
            case DataType::Double:
                data->SetValue(0.0);
                break;
            default:
                LogicError("Unsupported DataType %s", ::CNTK::DataTypeName(data->GetDataType()));
            }
        }
    }

    /*virtual*/ bool LearnerBase::Update(const unordered_map<Variable, ValuePtr>& parameterValues,
                                            const unordered_map<Variable, const ValuePtr>& gradientValues,
                                            size_t trainingSampleCount) /*override*/
    {
        // make sure trainingSampleCount is a valid value
        assert(trainingSampleCount > 0);

        for (const auto& parameter : Parameters())
        {
            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);
            const auto& gradientValue = gradientValues.at(parameter);
            const auto& parameterValue = parameterValues.at(parameter);

// TODO: make this a runtime parameter.
#if DUMPOUTPUT
            LOGPRINTF(stderr, "Update_%ls\n", parameter.Name().c_str());
#endif

#ifdef _DEBUG
            if (HasNan(smoothedGradientValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in smoothedGradient.", parameter.Name().c_str());
#endif

#if DUMPOUTPUT
            LOGPRINTF(stderr, "learnRatePerSample=%0.8f, momentum=%0.8f, actualMBSize=%ld\n",
                        m_learningRatePerSample, m_momentumPerSample, trainingSampleCount);
            LOGPRINTF(stderr, "GradUpdateType()=%s, GradientUpdateNoiseStd()=%0.8f\n",
                        LearnerType().c_str(), m_GaussianNoiseInjectStd);
            Print(gradientValue, "Gradient Update");
            Print(smoothedGradientValue, "Smoothed Gradient Input");
#endif
            if (!(PreProcess<float>(gradientValue, parameterValue, trainingSampleCount) ||  
                    PreProcess<double>(gradientValue, parameterValue, trainingSampleCount)))    
                    NOT_IMPLEMENTED;

            Update(parameter, smoothedGradientValue, gradientValue, parameterValue, trainingSampleCount);

            if (!(PostProcess<float>(parameter, gradientValue, parameterValue, trainingSampleCount) ||  
                PostProcess<double>(parameter, gradientValue, parameterValue, trainingSampleCount)))    
                NOT_IMPLEMENTED;

#if DUMPOUTPUT
            Print(parameterValue, "Parameter Update");
#endif

#ifdef _DEBUG
            if (HasNan(parameterValue, "TrainOneEpoch/UpdateWeights/Learner::Update(): "))
                LogicError("%ls has NaNs in parameter values after parameter update.", parameter.Name().c_str());
#endif
        }
        m_sampleCount += trainingSampleCount;
        return false;
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

            // Potentially, could store things like dimensions, element size, format, etc., but
            // that seems to be redundant, since all of that is passed in the constructor.
            checkpoint[parameter.Name()] = SerializeToVector(smoothedGradientValue->Data());
        }
        return checkpoint;
    }

    /*virtual*/ void LearnerBase::RestoreFromCheckpoint(const Dictionary& checkpoint) /*override*/
    {
        for (const auto& parameter : Parameters())
        {
            if (!checkpoint.Contains(parameter.Name()))
            {
                LogicError("Checkpoint does not contain state for parameter %ls", parameter.Name().c_str());
            }
            const auto& smoothedGradientValue = m_smoothedGradientValues.at(parameter);

            const DictionaryValue& state = checkpoint[parameter.Name()];

            const auto& data = smoothedGradientValue->Data();

            DeserializeFromVector(data, state.GetValue<vector<DictionaryValue>>());
        }
    }

    /*virtual*/ void LearnerSGD::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                        const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    bool LearnerSGD::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                            const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const
    {
        CHECK_DATA_TYPE(smoothedGradientValue);
        UNUSED(trainingSampleCount);

        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue->Data());
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue->Data());
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue->Data());

        const auto& learningRate = ElementType(ParameterDependentLearningRate(parameter));

        // TODO: break up the NormalGrad into 3 different functions, each with its own set of parameters
        // (one for vanilla SGD, the other for momentum SGD, and the third one for NAG).
        smoothedGradientMatrix->NormalGrad(*gradientMatrix, *parameterMatrix,
                                            learningRate, ElementType(m_momentumPerSample), m_useNesterovAcceleration);
            
        return true;
    }

    LearnerAdaGrad::LearnerAdaGrad(const unordered_set<Variable>& parameters, bool needAveMultiplier, const DeviceDescriptor& device)
        : LearnerBase(parameters, device),
        m_needAveMultiplier(needAveMultiplier)
    {
    }

    /*virtual*/ void LearnerAdaGrad::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                            const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    bool LearnerAdaGrad::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const
    {
        CHECK_DATA_TYPE(smoothedGradientValue);
        UNUSED(trainingSampleCount);

        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue->Data());
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue->Data());
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue->Data());

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));

        auto aveMultiplier = smoothedGradientMatrix->Adagrad(*gradientMatrix, m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);

        return true;
    }

    LearnerFSAdaGrad::LearnerFSAdaGrad(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
        : LearnerMomentumSGD(parameters, device)
    {
    }

    /*virtual*/ void LearnerFSAdaGrad::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    bool LearnerFSAdaGrad::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                  const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const
    {
        CHECK_DATA_TYPE(smoothedGradientValue);
        UNUSED(trainingSampleCount);

        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue->Data());
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue->Data());
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue->Data());

        //const double momentum = MomentumPerMB(m_momentumPerSample, trainingSampleCount);

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));

        smoothedGradientMatrix->FSAdagrad(trainingSampleCount, *gradientMatrix, *parameterMatrix,
                                            learningRate, ElementType(m_momentumPerSample));

        return true;
    }

    LearnerRMSProp::LearnerRMSProp(const unordered_set<Variable>& parameters,
                                    double gamma, double inc, double dec, double max, double min,
                                    bool needAveMultiplier, const DeviceDescriptor& device)
                                    : LearnerBase(parameters, device),
                                    m_gamma(gamma), m_inc(inc), m_dec(dec), m_max(max), m_min(min),
                                    m_needAveMultiplier(needAveMultiplier)
    {
    }

    /*virtual*/ void LearnerRMSProp::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                            const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const /*override*/
    {
        UPDATE_FUNCTION;
    }

    template <typename ElementType>
    bool LearnerRMSProp::Update(const Variable& parameter, const ValuePtr& smoothedGradientValue,
                                const ValuePtr& gradientValue, const ValuePtr& parameterValue, size_t trainingSampleCount) const
    {
        CHECK_DATA_TYPE(smoothedGradientValue);
        UNUSED(trainingSampleCount);

        const auto& smoothedGradientMatrix = GetWritableMatrix<ElementType>(smoothedGradientValue->Data());
        const auto& gradientMatrix = GetWritableMatrix<ElementType>(gradientValue->Data());
        const auto& parameterMatrix = GetWritableMatrix<ElementType>(parameterValue->Data());

        auto learningRate = ElementType(ParameterDependentLearningRate(parameter));

        auto aveMultiplier = smoothedGradientMatrix->RmsProp(*gradientMatrix,
                                                                ElementType(m_gamma), ElementType(m_inc),
                                                                ElementType(m_max), ElementType(m_dec),
                                                                ElementType(m_min), m_needAveMultiplier);
        Matrix<ElementType>::ScaleAndAdd(ElementType(-learningRate / aveMultiplier), *gradientMatrix, *parameterMatrix);

        return true;
    }

    // Explicit template instantiations
    template shared_ptr<Matrix<float>> LearnerBase::GetWritableMatrix<float>(const NDArrayViewPtr arrayView);
    template shared_ptr<Matrix<double>> LearnerBase::GetWritableMatrix<double>(const NDArrayViewPtr arrayView);
    
    LearnerPtr SGDLearner(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerSGD>(parameters, device);
    }

    LearnerPtr MomentumSGDLearner(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerMomentumSGD>(parameters, device);
    }

    LearnerPtr NesterovLearner(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerNesterov>(parameters, device);
    }

    LearnerPtr AdaGradLearner(const unordered_set<Variable>& parameters, bool needAveMultiplier, const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerAdaGrad>(parameters, needAveMultiplier, device);
    }

    LearnerPtr FSAdaGradLearner(const unordered_set<Variable>& parameters, const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerFSAdaGrad>(parameters, device);
    }

    LearnerPtr RMSPropLearner(const unordered_set<Variable>& parameters,
                                double gamma, double inc, double dec, double max, double min, bool needAveMultiplier,
                                const DeviceDescriptor& device)
    {
        return MakeSharedObject<LearnerRMSProp>(parameters, gamma, inc, dec, max, min, needAveMultiplier, device);
    }
    
}