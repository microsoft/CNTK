//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <numeric>

namespace CNTK 
{
    // A collection of additional options that are applicable for all standard learners 
    // (after these options are set, they retain their value for the entire lifespan of a learner).
    struct AdditionalLearningOptions
    {
        double l1RegularizationWeight = 0.0;
        double l2RegularizationWeight = 0.0;
        double gaussianNoiseInjectionStdDev = 0.0;
        bool gradientClippingWithTruncation = true;
        double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
        std::unordered_map<Parameter, double> learningRateMultipliers;
    };

    // An abstract base class at the root of the standard learners hierarchy
    // It implements most of the learner functionality, except for the actual update function,
    // and adds a few pre-/postprocessing methods (which are invoked before and after the update).
    class LearnerBase : public Learner
    {
    public:
        virtual bool Update(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) override final;

        virtual Dictionary GetCheckpointState() const override final;

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override final;

        void SetAdditionalOptions(const AdditionalLearningOptions& additionalOptions)
        {
            m_additionalOptions = additionalOptions;
        }

        // TODO: should this be called ResetMomentum?
        // needed for BlockMomemtumSGD to reset SGD momentum after aggregation.
        void ResetSmoothedGradients();

        // TODO: move learning rate and momentum scheduling and adjustment functionality 
        // inside the learner and drop these setters.
        void SetLearningRate(double value) { m_learningRatePerSample = value; }

    protected:
        LearnerBase(const std::unordered_set<Parameter>& parameters);

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const = 0;

        double ParameterDependentLearningRate(const Parameter& parameter) const
        {
            return m_learningRatePerSample * m_additionalOptions.learningRateMultipliers.at(parameter);
        }

        std::string LearnerType() const;

        double m_learningRatePerSample;

        AdditionalLearningOptions m_additionalOptions;

        std::unordered_map<Parameter, NDArrayViewPtr> m_smoothedGradientValues;

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

    private:
        // Templatized update function, it invokes preprocess and postprocess using the provided
        // template parameter and also invokes virtual Update method implemented in one of the subclasses.
        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        // TODO: make these functions friends of NDViewArray and move to Utils?
        static bool HasNan(const NDArrayViewPtr& value, const char* name);
        static void Print(const NDArrayViewPtr& value, const char* msg);

        size_t m_sampleCount;
    };

    // Vanilla gradient descent optimization algorithm.
    class LearnerSGD : public LearnerBase
    {
    public:
        LearnerSGD(const std::unordered_set<Parameter>& parameters, double learningRatePerSample = 0)
            : LearnerBase(parameters), m_momentumPerSample(0.0), m_useNesterovAcceleration(false)
        {
            SetLearningRate(learningRatePerSample);
        }

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        double m_momentumPerSample;
        bool m_useNesterovAcceleration;
    };

    // SGD optimization with momentum. 
    class LearnerMomentumSGD : public LearnerSGD
    {
    public:
        LearnerMomentumSGD(const std::unordered_set<Parameter>& parameters)
            : LearnerSGD(parameters)
        {}

        void SetMomentum(double value) { m_momentumPerSample = value; }
    };

    // Nesterov's accelerated SGDLearnerBase descent. 
    class LearnerNesterov : public LearnerSGD
    {
    public:

        LearnerNesterov(const std::unordered_set<Parameter>& parameters)
            : LearnerSGD(parameters)
        {
            m_useNesterovAcceleration = true;
        }
    };

    class LearnerAdaGrad : public LearnerBase
    {
    public:

        LearnerAdaGrad(const std::unordered_set<Parameter>& parameters, bool needAveMultiplier);

    protected:
        bool m_needAveMultiplier;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerFSAdaGrad : public LearnerMomentumSGD
    {
    public:

        LearnerFSAdaGrad(const std::unordered_set<Parameter>& parameters);

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerRMSProp : public LearnerBase
    {
    public:

        LearnerRMSProp(const std::unordered_set<Parameter>& parameters,
                       double gamma, double inc, double dec, double max, double min, bool needAveMultiplier);

    protected:

        double m_gamma;
        double m_inc;
        double m_dec;
        double m_max;
        double m_min;
        bool m_needAveMultiplier;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };
}
