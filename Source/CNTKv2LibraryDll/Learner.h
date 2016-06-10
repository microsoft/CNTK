#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{

#define DECLARE_UPDATE_FUNCTION(type)                                                      \
     if (dtype == GetDataType<type>())                                                     \
     {                                                                                     \
         return Update<type>(smoothedGradient, gradient, parameter, trainingSampleCount);  \
     }

#define DECLARE_UPDATE_FUNCTIONS           \
     DECLARE_UPDATE_FUNCTION(float)        \
     DECLARE_UPDATE_FUNCTION(double)       \
     NOT_IMPLEMENTED; 

    class LearnerBase : public Learner
    {
    protected:
        LearnerBase(const std::unordered_set<Variable>& parameters,
            double learningRatePerSample = 0.0, double momentumPerSample = 0.0);

        virtual bool Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameters,
            const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients,
            size_t trainingSampleCount) override;

        // Member function templates cannot be virtual, using this as a workaround.
        virtual void Update(const DataType dtype, const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const = 0;

        _Internal::_SimpleMap<Variable, ValuePtr> m_smoothedGradients;
        
        double m_learningRatePerSample;
        double m_momentumPerSample;

        size_t m_sampleCount;

        // TODO: the following methods are needed for backwards compatibility until sgd.cpp is updated to v2.
#pragma region _temporary_back_compat
        virtual double GetLearningRate() const { return m_learningRatePerSample; }
        virtual double GetMomentum() const override { return m_momentumPerSample; }
        virtual void SetLearningRate(double value) override { m_learningRatePerSample = value; }
        virtual void SetMomentum(double value) override { m_momentumPerSample = value; }
        virtual _Internal::_SimpleVector<ValuePtr>  SmoothedGradients() const override
        {
            return m_smoothedGradients.Values();
        }
#pragma endregion _temporary_back_compat
    };

    class SGD : public LearnerBase
    {
    public:

        SGD(const std::unordered_set<Variable>& parameters, double learningRatePerSample, 
            double momentumPerSample, bool useNesterovAcceleration = false);

    protected:

        bool m_useNesterovAcceleration;

        
        virtual void Update(const DataType dtype, const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const override
        {
            DECLARE_UPDATE_FUNCTIONS;
        }


        template <typename ElementType>
        void Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const;
    };

    class AdaGrad : public LearnerBase
    {
    public:

        AdaGrad(const std::unordered_set<Variable>& parameters, double learningRatePerSample, 
            bool needAveMultiplier = true);

    protected:
        bool m_needAveMultiplier;


        virtual void Update(const DataType dtype, const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const override
        {
            DECLARE_UPDATE_FUNCTIONS;
        }

        template <typename ElementType>
        void Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const;
    };

    class FSAdaGrad : public LearnerBase
    {
    public:

        FSAdaGrad(const std::unordered_set<Variable>& parameters, double learningRatePerSample,
            double momentumPerSample);

    protected:

        virtual void Update(const DataType dtype, const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const override
        {
            DECLARE_UPDATE_FUNCTIONS;
        }


        template <typename ElementType>
        void Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const;
    };

    struct RMSPropInfo
    {
        double gamma;
        double inc;
        double dec;
        double max;
        double min;
    };

    class RmsProp : public LearnerBase
    {
    public:

        RmsProp(const std::unordered_set<Variable>& parameters, double learningRatePerSample, 
            RMSPropInfo info, bool needAveMultiplier = true);

    protected:

        double m_rmsGamma;
        RMSPropInfo m_info;
        bool m_needAveMultiplier;

        virtual void Update(const DataType dtype, const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const override
        {
            DECLARE_UPDATE_FUNCTIONS;
        }

        template <typename ElementType>
        void Update(const ValuePtr smoothedGradient, const ValuePtr gradient,
            const ValuePtr  parameter, size_t trainingSampleCount) const;
    };

   
}