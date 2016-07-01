#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{
    class LearnerBase : public Learner
    {
    public:
        
        virtual Dictionary GetCheckpointState() const override;

        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

    protected:
        LearnerBase(const _Internal::_SimpleSet<Variable>& parameters, const Learner::AdditionalParameters& additionalParameters);

        virtual bool Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameters,
            const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients,
            size_t trainingSampleCount) override;

        virtual void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
            const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const = 0;

        double ParameterDependentLearningRate(const Variable& parameter) const
        {
            return m_learningRatePerSample * m_learningRateMultipliers[parameter];
        }

        virtual std::wstring LearnerType() = 0;

        double m_learningRatePerSample;
        double m_momentumPerSample;

        double m_L1RegWeight;
        double m_L2RegWeight;
        double m_GaussianNoiseInjectStd;

        bool m_gradientClippingWithTruncation;
        double m_clippingThresholdPerSample;

        size_t m_sampleCount;

         _Internal::_SimpleSet<Variable>  m_parameters;
         
         _Internal::_SimpleMap<Variable, double> m_learningRateMultipliers;

         _Internal::_SimpleMap<Variable, ValuePtr> m_smoothedGradients;


        template <typename ElementType>
        static std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(const NDArrayViewPtr arrayView);

        template <typename ElementType>
        static std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(NDArrayViewPtr arrayView);

        template <typename ElementType>
        static const Microsoft::MSR::CNTK::TensorView<ElementType>* GetTensorView(const NDArrayViewPtr arrayView);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::TensorView<ElementType>* GetWritableTensorView(NDArrayViewPtr arrayView);

        template <typename ElementType>
        void ClipGradient(Microsoft::MSR::CNTK::Matrix<ElementType>& gradient, size_t actualMBSize) const;

        template <typename ElementType>
        void PreProcess(const Variable& learnableParameter, const ValuePtr&  gradient, const ValuePtr& parameter, size_t actualMBSize) const;

        template <typename ElementType>
        void PostProcess(const Variable& learnableParameter, const ValuePtr&  gradient, const ValuePtr& parameter, size_t actualMBSize) const;

        virtual void SetLearningRate(double value) override { m_learningRatePerSample = value; }
        virtual void SetMomentum(double value) override { m_momentumPerSample = value; }

        virtual void ResetSmoothedGradients() override;

    private:
        // TODO: make these functions friends of NDViewArray and move to Utils?
        static bool HasNan(const ValuePtr& value, const char* name);
        static void Print(const ValuePtr& value, const char* msg);
    };

    namespace Learners
    {
        class SGDLearner : public LearnerBase
        {
        public:

            SGDLearner(const _Internal::_SimpleSet<Variable>& parameters, bool useNesterovAcceleration,
                const Learner::AdditionalParameters& additionalParameters);

        protected:

            bool m_useNesterovAcceleration;

            virtual void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const override;

            template <typename ElementType>
            void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"SGD Learner"; }
        };

        class AdaGradLearner : public LearnerBase
        {
        public:

            AdaGradLearner(const _Internal::_SimpleSet<Variable>& parameters, bool needAveMultiplier,
                const Learner::AdditionalParameters& additionalParameters);

        protected:
            bool m_needAveMultiplier;

            virtual void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const override;

            template <typename ElementType>
            void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"AdaGrad Learner"; }
        };

        class FSAdaGradLearner : public LearnerBase
        {
        public:

            FSAdaGradLearner(const _Internal::_SimpleSet<Variable>& parameters, 
                const Learner::AdditionalParameters& additionalParameters);

        protected:

            virtual void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const override;

            template <typename ElementType>
            void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"FSAdaGrad Learner"; }
        };

        struct RMSPropInfo
        {
            double gamma;
            double inc;
            double dec;
            double max;
            double min;
        };

        class RmsPropLearner : public LearnerBase
        {
        public:

            RmsPropLearner(const _Internal::_SimpleSet<Variable>& parameters, RMSPropInfo info, 
                bool needAveMultiplier, const Learner::AdditionalParameters& additionalParameters);

        protected:

            double m_rmsGamma;
            RMSPropInfo m_info;
            bool m_needAveMultiplier;

            virtual void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const override;

            template <typename ElementType>
            void Update(const Variable& learnableParameter, const ValuePtr& smoothedGradient, 
                const ValuePtr& gradient, const ValuePtr&  parameter, size_t trainingSampleCount) const;

            virtual std::wstring LearnerType() override { return L"RmsProp Learner"; }
        };
    }
}