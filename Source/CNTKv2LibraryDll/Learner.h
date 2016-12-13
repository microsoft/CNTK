//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <numeric>

namespace CNTK 
{
    // TODO: Move this to Trainer along with Pre-, PostProcess and ClipGradient.
    // A collection of additional options that are applicable for all standard learners 
    // (after these options are set, they retain their value for the entire lifespan of a learner).
    struct AdditionalLearningOptions
    {
        double l1RegularizationWeight = 0.0;
        double l2RegularizationWeight = 0.0;
        double gaussianNoiseInjectionStdDev = 0.0;
        bool gradientClippingWithTruncation = true;
        double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
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

    protected:
        LearnerBase(const std::vector<Parameter>& parameters, 
                    const LearningRatesPerSample& learningRates,
                    bool allocateSmoothGradients = true,
                    double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                    bool gradientClippingWithTruncation = true);

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const = 0;

        std::string LearnerType() const;

        LearningRatesPerSample m_learningRates;

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

        // Returns an NDArrayView with the required shape, with the same data type as parameter value
        // and allocated on the same device.
        static NDArrayViewPtr AllocateNDArrayView(const Parameter& parameter, const NDShape& shape);

        // Retrieves the shape of the matrix corresponding to the parameter value.
        static NDShape GetMatrixShape(const Parameter& parameter);

        size_t m_sampleCount;
        size_t m_minibatchCount;

    private:
        // Templatized update function, it invokes preprocess and postprocess using the provided
        // template parameter and also invokes virtual Update method implemented in one of the subclasses.
        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        // TODO: make these functions friends of NDViewArray and move to Utils?
        static bool HasNan(const NDArrayViewPtr& value, const char* name);
        static void Print(const NDArrayViewPtr& value, const char* msg);

        static const size_t checkpointVersion = 1;
    };

    // Vanilla gradient descent optimization algorithm.
    class LearnerSGD : public LearnerBase
    {
    public:
        LearnerSGD(const std::vector<Parameter>& parameters, 
                   const LearningRatesPerSample& learningRates, 
                   bool allocateSmoothGradients = true,
                   double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                   bool gradientClippingWithTruncation = true)
                   : LearnerBase(parameters, learningRates, allocateSmoothGradients, clippingThresholdPerSample, gradientClippingWithTruncation),
            m_momentums(0.0), 
            m_useNesterovAcceleration(false)
        {}

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;

        // TODO: Move m_momentums to LearnerMomentumSGD as soon as NormalGrad is refactored.
        MomentumsPerSample m_momentums;
        bool m_useNesterovAcceleration;
    };

    // SGD optimization with momentum. 
    class LearnerMomentumSGD : public LearnerSGD
    {
    public:
        LearnerMomentumSGD(const std::vector<Parameter>& parameters,
                           const LearningRatesPerSample& learningRates,
                           const MomentumsPerSample& momentums,
                           bool allocateSmoothGradients = true,
                           double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                           bool gradientClippingWithTruncation = true)
                           : LearnerSGD(parameters, learningRates, allocateSmoothGradients, clippingThresholdPerSample, gradientClippingWithTruncation)
        {
            m_momentums = momentums;
        }
    };

    // Nesterov's accelerated SGDLearnerBase descent. 
    class LearnerNesterov : public LearnerMomentumSGD
    {
    public:

        LearnerNesterov(const std::vector<Parameter>& parameters,
                        const LearningRatesPerSample& learningRates,
                        const MomentumsPerSample& momentums,
                        double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                        bool gradientClippingWithTruncation = true)
                        : LearnerMomentumSGD(parameters, learningRates, momentums, true, clippingThresholdPerSample, gradientClippingWithTruncation)
        {
            m_useNesterovAcceleration = true;
        }
    };

    class LearnerAdaGrad : public LearnerBase
    {
    public:

        LearnerAdaGrad(const std::vector<Parameter>& parameters,
                       const LearningRatesPerSample& learningRates,
                       bool needAveMultiplier,
                       double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                       bool gradientClippingWithTruncation = true);

    protected:
        bool m_needAveMultiplier;

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerFSAdaGrad : public LearnerMomentumSGD
    {
    public:

        LearnerFSAdaGrad(const std::vector<Parameter>& parameters,
                         const LearningRatesPerSample& learningRates,
                         const MomentumsPerSample& momentums,
                         double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                         bool gradientClippingWithTruncation = true);

    protected:

        virtual void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const override;

        template <typename ElementType>
        void Update(const Parameter& parameter, const NDArrayViewPtr& gradientValue, const NDArrayViewPtr& smoothedGradientValue, size_t trainingSampleCount) const;
    };

    class LearnerRMSProp : public LearnerBase
    {
    public:

        LearnerRMSProp(const std::vector<Parameter>& parameters,
                       const LearningRatesPerSample& learningRates,
                       double gamma, double inc, double dec, double max, double min,
                       bool needAveMultiplier,
                       double clippingThresholdPerSample = std::numeric_limits<double>::infinity(),
                       bool gradientClippingWithTruncation = true);

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
