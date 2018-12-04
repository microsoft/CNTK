//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <numeric>
#include <functional>

namespace CNTK 
{
    // An abstract base class at the root of the standard learners hierarchy
    // It implements most of the learner functionality, except for the actual update function,
    // and adds a few pre-/postprocessing methods (which are invoked before and after the update).
    class LearnerBase : public Learner
    {
    public:
//        virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd = false) override;
        virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd) override;

        virtual Dictionary CreateCheckpoint() override;

        virtual size_t CurrentVersion() const override final { return s_serializationVersion; }

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        virtual void ResetSmoothedGradients() override;

        virtual void SetNeedToUpdateMasterParameter() override { m_masterParameterUpdated = false; }

    protected:
        LearnerBase(const std::vector<Parameter>& parameters,
            const LearningRateSchedule& learningRateSchedule,
            AdditionalLearningOptions additionalOptions);

        void AllocateSmoothedGradients(const std::vector<Parameter>& parameters, size_t factor, size_t fp16Factor = 1);

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) = 0;

        // Allows derived class may override this to perform per-minibatch update actions
        virtual void UpdateOnMinibatch(size_t /*trainingSampleCount*/) {}

        std::string LearnerType() const;

        // Returns current learning rate.
        double LearningRate(size_t minibatchSize) const
        {
            auto learningRate = Learner::LearningRate();
            if (IsCompatibleMode(m_learningRateSchedule))
            {
                if (IsCompatibleMode())
                    //learner is in compatible mode, the gradients are already mean gradient so the learning rate are directly applied
                    return learningRate;
                else
                    //learner is not in compatible mode, the gradients are not mean gradient so the learning rate need to be scaled to per sample rate to simulate per minibatch rate
                    return learningRate / (double)minibatchSize;
            }
            else 
            {
                std::size_t ref_mbsize = m_learningRateSchedule.GetMinibatchSize();
                assert(ref_mbsize > 0);
                if (IsCompatibleMode())
                    //learner is in compatible mode, the gradients are already mean gradient so the learning rate needs to be scaled to match the encountered minibatch size
                    return learningRate  * ((double) minibatchSize / (double) ref_mbsize);
                else
                    //learner is not in compatible mode, the gradients are not mean gradient so the learning rate need to scaled to per sample rate
                    return learningRate / ref_mbsize;
            }

            return learningRate;
        }

        void ReportTrainingParameterValue(const TrainingParameterSchedule<double>& schedule, const std::wstring& name) const;

        // A map cointaining hyperparameter names and corresponging values that's used to track and report changes 
        // in hyperparameter values.
        mutable std::map <std::wstring, double> m_trainingParametersMap;

        std::unordered_map<Parameter, NDArrayViewPtr> m_smoothedGradientValues;

        bool m_masterParameterUpdated; // whether the master copy of parameters are updated

        mutable size_t m_noiseInjectionSeed;

        // The following four static protected methods expose private methods of NDArrayView class
        // (which declares LearnerBase as friend class), so that they are available to subclasses.
        template <typename ElementType>
        static std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(const NDArrayViewPtr& arrayView);

        template <typename ElementType>
        static std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(const NDArrayViewPtr& arrayView);

        template <typename ElementType>
        static const Microsoft::MSR::CNTK::TensorView<ElementType>* GetTensorView(const NDArrayViewPtr& arrayView);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::TensorView<ElementType>* GetWritableTensorView(const NDArrayViewPtr& arrayView);

        template <typename ElementType>
        void ClipGradient(Microsoft::MSR::CNTK::Matrix<ElementType>& gradient, size_t actualMBSize) const;

        // Performs additional preprocessing before calling the update method 
        // (gradient clipping and L2 regularization depending on the additional learning parameters).
        template <typename ElementType>
        void PreProcess(const NDArrayViewPtr& parameterValue, const NDArrayViewPtr& gradientValue, size_t actualMBSize) const;

        // Performs additional postprocessing after the update method has been executed
        // (noise injection and L1 regularization specified by the additional learning parameters).
        template <typename ElementType>
        void PostProcess(const Parameter& parameter, const NDArrayViewPtr& gradientValue, size_t actualMBSize) const;

        // Returns an NDArrayView with the required shape, with the same data type as parameter value
        // and allocated on the same device.
        static NDArrayViewPtr AllocateSmoothedGradientFor(const Parameter& parameter, size_t factor, size_t fp16Factor = 1);

        // Retrieves the shape of the matrix corresponding to the parameter value.
        static NDShape GetMatrixShape(const Parameter& parameter);

    private:
        // Templatized update function, it invokes preprocess and postprocess using the provided
        // template parameter and also invokes virtual Update method implemented in one of the subclasses.
        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount);

        // TODO: make these functions friends of NDViewArray and move to Utils?
        static bool HasNan(const NDArrayViewPtr& value, const char* name);
        static void Print(const NDArrayViewPtr& value, const char* msg);

        // Version history:
        // 1 -- initial version.
        // 2 -- instead of storing smoothed gradients as a map<parameter_uid, smoothed_grad_value>.
        // 3 -- adding sweep count into the checkpoints
        // save them as a vector in the same order as the order of parameters this learner is responsible for.
        static const size_t s_serializationVersion = 3;
    };

    // Vanilla gradient descent optimization algorithm.
    class LearnerSGD final : public LearnerBase
    {
    public:
        LearnerSGD(const std::vector<Parameter>& parameters,
                   const LearningRateSchedule& learningRateSchedule,
                   AdditionalLearningOptions additionalOptions)
                   : LearnerBase(parameters, learningRateSchedule, additionalOptions)
        {
            AllocateSmoothedGradients(parameters, 0);
        }

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    // SGD optimization with momentum. 
    class LearnerMomentumSGD : public LearnerBase
    {
    public:
        LearnerMomentumSGD(const std::vector<Parameter>& parameters,
                           const LearningRateSchedule& learningRateSchedule,
                           const MomentumSchedule& momentumSchedule,
                           bool unitGain,
                           AdditionalLearningOptions additionalOptions,
                           size_t smoothGradientFactor)
                           : LearnerBase(parameters, learningRateSchedule, additionalOptions),
                           m_momentumSchedule(momentumSchedule), 
                           m_unitGain(unitGain)
        {
            AllocateSmoothedGradients(parameters, smoothGradientFactor, 2);
        }

        // returns current per-minibatch momentum value.
        virtual double MomentumValueForMB(size_t minibatchSize) const
        {
            return MomentumValueForMB(m_momentumSchedule, minibatchSize);
        }

    protected:
        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;

        template <typename ElemType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        void UpdateHalf(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        // returns current per-minibatch momentum value from the provided schedule.
        double MomentumValueForMB(const MomentumSchedule& schedule, size_t minibatchSize) const;

        // Return true if the update should use classic momentum and 
        // false if the unit-gain momentum should be used instead.
        bool UseUnitGainMomentum() const
        {
            return m_unitGain;
        }

        ///Return the unit gain factor. Note that the unit gain factor should not be scaled according to the minibatch size. See explanation in the Update(...) function.
        template <typename ElementType>
        ElementType UnitGainFactor(size_t minibatchSize) const
        {
            //TODO: Preliminary study shows that the unitgain factor should use the raw momentum instead of the scaled momentum as the following: 
            //      ElementType momentum = (ElementType)GetCurrentTrainingParameterValue(m_momentumSchedule);
            //However, further investigation over the perfs are needed.
            ElementType momentum = ElementType(MomentumValueForMB(minibatchSize));
            return UseUnitGainMomentum() ? ElementType(1.0) - momentum : ElementType(1.0);
        }

    private:
        MomentumSchedule m_momentumSchedule;
        bool m_unitGain;
    };

    // Nesterov's accelerated SGDLearnerBase descent. 
    class LearnerNesterov : public LearnerMomentumSGD
    {
    public:

        LearnerNesterov(const std::vector<Parameter>& parameters,
                        const LearningRateSchedule& learningRateSchedule,
                        const MomentumSchedule& momentumSchedule,
                        bool unitGain,
                        AdditionalLearningOptions additionalOptions)
                        : LearnerMomentumSGD(parameters, learningRateSchedule, momentumSchedule, unitGain, additionalOptions, 1)
        {}

    protected:
        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
        void UpdateHalf(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerAdaGrad : public LearnerBase
    {
    public:
        LearnerAdaGrad(const std::vector<Parameter>& parameters,
                       const LearningRateSchedule& learningRateSchedule,
                       bool needAveMultiplier,
                       AdditionalLearningOptions additionalOptions);

    protected:
        bool m_needAveMultiplier;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerAdaDelta : public LearnerBase
    {
    public:
        LearnerAdaDelta(
            const std::vector<Parameter>& parameters,
            const LearningRateSchedule& learningRateSchedule,
            double rho, double epsilon,
            AdditionalLearningOptions additionalOptions);

    protected:
        // If a gradient is sparse, we skip updating columns with zero gradients. This means some 
        // columns will receive their updates when their gradient is non-zero. The only exception
        // is that once every s_SyncInterval updates we will make sure all columns are up to date. 
        static const int s_SyncInterval;

        double m_rho;
        double m_epsilon;
        // If a gradient is sparse, we will maintain a timestamp per column with the last time that column was updated
        std::unordered_map<Parameter, NDArrayViewPtr> m_lastUpdateTime;
        // If a gradient is sparse we will use the current time and the timestamp to determine how to apply a bunch of delayed updates for this column.
        // This allows us to skip updating many columns when the gradients are sparse.
        std::unordered_map<Parameter, int> m_currentTime;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;

        template <typename GradType, typename AccumType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount);

        virtual Dictionary CreateCheckpoint() override;
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;
    };

    class LearnerFSAdaGrad : public LearnerMomentumSGD
    {
    public:

        LearnerFSAdaGrad(const std::vector<Parameter>& parameters,
                         const LearningRateSchedule& learningRateSchedule,
                         const MomentumSchedule& momentumSchedule,
                         bool unitGain,
                         const MomentumSchedule& varianceMomentumSchedule,
                         AdditionalLearningOptions additionalOptions);

        virtual Dictionary CreateCheckpoint() override;

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        virtual void ResetSmoothedGradients() override;

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;
        virtual void UpdateOnMinibatch(size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

    private:
        static const double s_targetAdagradAvDenom;
        double m_targetAdagradAvDenom_x_sqrtAdagradSqrFrames;

        // returns current per-minibatch variance momentum value.
        double VarianceMomentumValueForMB(size_t minibatchSize) const
        {
            return MomentumValueForMB(m_varianceMomentumSchedule, minibatchSize);
        }

        double m_smoothedCount;
        MomentumSchedule m_varianceMomentumSchedule;
    };

    class LearnerAdam : public LearnerMomentumSGD
    {
    public:

        LearnerAdam(const std::vector<Parameter>& parameters,
            const LearningRateSchedule& learningRateSchedule,
            const MomentumSchedule& momentumSchedule,
            bool unitGain,
            const MomentumSchedule& varianceMomentumSchedule,
            double epsilon,
            bool adamax,
            AdditionalLearningOptions additionalOptions);

        virtual Dictionary CreateCheckpoint() override;

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        virtual void ResetSmoothedGradients() override;

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;
        virtual void UpdateOnMinibatch(size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

    private:

        // returns current per-minibatch variance momentum value.
        double VarianceMomentumValueForMB(size_t minibatchSize) const
        {
            //TODO: According to my preliminary analysis, the second momentum variance scaling is different from momentum scaling; need to double check -- yuqing tang
            return MomentumValueForMB(m_varianceMomentumSchedule, minibatchSize);
        }

        double m_smoothedCount;
        MomentumSchedule m_varianceMomentumSchedule;
        double m_epsilon;
        bool m_adamax;
    };

    class LearnerRMSProp : public LearnerBase
    {
    public:

        LearnerRMSProp(const std::vector<Parameter>& parameters,
                       const LearningRateSchedule& learningRateSchedule,
                       double gamma, double inc, double dec, double max, double min,
                       bool needAveMultiplier,
                       AdditionalLearningOptions additionalOptions);

        virtual Dictionary CreateCheckpoint() override;

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

        virtual void ResetSmoothedGradients() override;

    protected:

        double m_gamma;
        double m_inc;
        double m_dec;
        double m_max;
        double m_min;
        bool m_needAveMultiplier;
        double m_smoothedCount;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;
        virtual void UpdateOnMinibatch(size_t trainingSampleCount) override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };


    class LearnerUniversal : public LearnerBase
    {
        std::unordered_map<Parameter, Variable> m_parameter_gradient_map;
        FunctionPtr m_update_func;

    public:
        LearnerUniversal(const std::vector<Parameter>& parameters, const ParameterUpdateFunctor& func);

        LearnerUniversal(const std::vector<Parameter>& parameters, const std::vector<Variable>& gradients, FunctionPtr updateFunc);
    
        //virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd = false) override;
        virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd) override;

    private:
        void ValidateInput(const std::vector<Parameter>& parameters, const std::vector<Variable>& gradients, FunctionPtr updateFunc);


    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) override;
    };
}
