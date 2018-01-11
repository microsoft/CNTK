// experimental emulator of Marian API on top of CNTK

#pragma once

#ifndef __MARIAN_CNTK
#define __MARIAN_CNTK

#include "CNTKLibrary.h"
#include "Models.h" // for the helper functions
#include "Layers.h" // for some functions missing in main CNTK, e.g. LogSoftmax
#include <vector>
#include <algorithm>
#include <memory>
#include <map>

namespace marian
{
    template<typename T>
    using Ptr = std::shared_ptr<T>;
    template <class T, typename... Args> Ptr<T> New(Args&&... args) { return Ptr<T>(new T(std::forward<Args>(args)...)); }
    template <class T> Ptr<T> New(Ptr<T> p) { return Ptr<T>(p); }

    class Expr : public CNTK::Variable
    {
        typedef CNTK::Variable Base;
    public:
        Expr(const CNTK::Variable& v) : Base(v) { }
        Expr(CNTK::Variable&& v) : Base(std::move(v)) { }
        // ...assignments
        //std::cout << "Epoch: " << epoch << " Cost: " << cost->scalar()
        CNTK::NDArrayViewPtr val() const { return Value(); }
        float scalar() const { return val()->AsScalar<float>(); }
        // Marian accesses members by arrow, not dot. This is a bit of a hack.
        Expr* operator->() { return this; }
        void dump() const { Value()->LogToFile(Name()); }
    };

    // helpers for mapping stuff to and from CNTK
    typedef std::vector<int> Shape; // TODO: We could use Marian's Shape class directly
    namespace mappers
    {
        CNTK::NDShape ToNDShape(const Shape& shape)
        {
            // order of axes is reverse in Marian
            return CNTK::NDShape(shape.rbegin(), shape.rend());
        }
        CNTK::Axis ToCNTKAxis(const Expr& x, int axisIndex)
        {
            const auto& viewShape = x.Shape();
            auto rank = viewShape.Rank();
            // TODO: negative axes
            if (axisIndex >= rank)
                CNTK::InvalidArgument("marian::ToCNTKAxis: axis out of range");
            return CNTK::Axis(rank - 1 - (size_t)axisIndex);
        }
    }

    static inline Expr affine(const Expr& x, const Expr& W, const Expr& b) { Expr y = CNTK::Times(W, x) + b; return Alias(y, L"Times(" + W.Name() + L"," + x.Name() + L")+(" + b.Name() + L")"); }
    static inline Expr tanh(const Expr& x) { return CNTK::Tanh(x, L"Tanh(" + x.Name() + L")"); }
    static inline Expr mean(const Expr& x, int axisIndex)
    {
        auto axis = mappers::ToCNTKAxis(x, axisIndex);
        return CNTK::ReduceMean(x, axis, L"ReduceMean(" + x.Name() + L",Axis(" + std::to_wstring(axis.StaticAxisIndex()) + L"))");
    }
    // o = unnormalized log prob; y = label as an index, not one-hot
    // o: (3,120); y: (120,)
    static inline Expr cross_entropy(const Expr& o, const Expr& y)
    {
        auto numClasses = o.Shape()[0];
        auto yOneHot = CNTK::OneHotOp(y, numClasses, /*outputSparse=*/true, CNTK::Axis(0));
        return Alias(Dynamite::CrossEntropyWithSoftmax(o, yOneHot, CNTK::Axis(0)), L"CrossEntropyWithSoftmax(" + o.Name() + L",OneHot(" + y.Name() + L",)" + std::to_wstring(numClasses) + L")");
    }
    static inline Expr logsoftmax(const Expr& x) { return Dynamite::LogSoftmax(x, CNTK::Axis(0), L"LogSoftmax(" + x.Name() + L",Axis(0))"); }

    namespace inits
    {
        CNTK::ParameterInitializer from_vector(const std::vector<float>& inputData)
        {
            return CNTK::Dictionary( // initializers are just dictionaries
                L"from_vector",
                // wrap the CPU-side buffer in an NDArrayView object (by pointer, no data is copied)
                CNTK::NDArrayView(CNTK::DataType::Float, CNTK::NDShape{ inputData.size() },
                                  (void*)inputData.data(), inputData.size() * sizeof(float),
                                  CNTK::DeviceDescriptor::CPUDevice(), /*readOnly=*/true)
            );
        }
        // TODO: check that the scaling is the same
        CNTK::ParameterInitializer uniform() { return CNTK::UniformInitializer(0.1); }
        static CNTK::ParameterInitializer zeros = CNTK::ConstantInitializer(0);
    }

    namespace Config
    {
        // TODO: need an equivalent for gcc
        __declspec(selectany) size_t seed;
    };

    static CNTK::ParameterInitializer init; // to allow to say init=...
    static int axis;

    class ExpressionGraph
    {
    public:
        ExpressionGraph() {}
        void clear() { }
        void reserveWorkspaceMB(size_t) { }
        // TODO: what is Marian's device id of the CPU?
        void setDevice(size_t device = 0)
        {
            Dynamite::SetCurrentDevice(CNTK::DeviceDescriptor::GPUDevice((unsigned int)device));
        }
        size_t getDevice() { return Dynamite::CurrentDevice().Id(); }
        void setInference(bool inference) { m_inferenceOnly = inference; }
        Expr constant(const Shape& npShape, const CNTK::ParameterInitializer& init)
        {
            auto viewShape = mappers::ToNDShape(npShape); // convert to CNTK's column-major viewShape
            if (init.Contains(L"from_vector"))
            {
                // BUGBUG: This keeps a reference to the vector, not a copy, which only works inside a single expression, if at all.
                const auto& initData = init[L"from_vector"].Value<CNTK::NDArrayView>();
                if (initData.Shape().TotalSize() != viewShape.TotalSize())
                    CNTK::InvalidArgument("marian::constant: vector size does not match viewShape");
                // copy the supplied CPU buffer, which may be a temporary, to a GPU-side NDArrayView
                return CNTK::Constant(initData.AsShape(viewShape)->DeepClone(Dynamite::CurrentDevice(), /*readOnly=*/true),
                                      /*isVolatile=*/m_inferenceOnly);
            }
            CNTK::InvalidArgument("BUGBUG: no public Constant() from ParameterInitializer?");
        }
        // TODO: namespace; lots more
        Expr param(const std::string& name, const Shape& shape, const CNTK::ParameterInitializer& init)
        {
            auto viewShape = mappers::ToNDShape(shape); // convert to CNTK's column-major viewShape
            auto iter = m_allParametersMap.find(name);
            if (iter == m_allParametersMap.end()) // case 1: create a new parameter
            {
                if (init.Contains(L"from_vector")) // our fake 
                {
                    // BUGBUG: This keeps a reference to the vector, not a copy, which only works inside a single expression, if at all.
                    const auto& initData = init[L"from_vector"].Value<CNTK::NDArrayView>();
                    if (initData.Shape().TotalSize() != viewShape.TotalSize())
                        CNTK::InvalidArgument("marian::constant: vector size does not match viewShape");
                    // copy the supplied CPU buffer, which may be a temporary, to a GPU-side NDArrayView
                    auto initVal = initData.AsShape(viewShape)->DeepClone(Dynamite::CurrentDevice(), /*readOnly=*/false);
                    auto p = CNTK::Parameter(initVal);
                    m_allParametersMap.insert(std::make_pair(name, p));
                    m_allParameters.push_back(p);
                    m_allGradients[p] = nullptr;
                    return p;
                }
                else
                {
                    auto p = CNTK::Parameter(viewShape, CNTK::DataType::Float, init, Dynamite::CurrentDevice(), std::wstring(name.begin(), name.end())); // copy it (possibly to a different device)
                    m_allParametersMap.insert(std::make_pair(name, p));
                    m_allParameters.push_back(p);
                    m_allGradients[p] = nullptr;
                    return p;
                }
            }
            else  // case 2: retrieve an existing parameter
            {
                const auto& p = iter->second;
                if (p.Shape() != viewShape)
                    CNTK::InvalidArgument("marian::param: Requested shape for existing parameter '%s' does not match original shape", name.c_str());
                return p;
            }
        }
        // forward/backward
        void forward() { }
        void forwardNext() { }
        // BREAKING CHANGE: must pass the root for backprop
        void backward(const Expr& root) { backprop(root); }
        void backprop(const Expr& root)
        {
            root.Backward(m_allGradients);
        }
    private:
        std::map<std::string, CNTK::Parameter> m_allParametersMap;
        std::vector<CNTK::Parameter> m_allParameters;
        bool m_inferenceOnly = false;
        friend class OptimizerWrapper;
        std::unordered_map<CNTK::Parameter, CNTK::NDArrayViewPtr> m_allGradients;
    };

    enum AlgorithmType { Sgd, Adam };
    class OptimizerWrapper
    {
    public:
        // TODO: more parameters; schedules, lots more
        OptimizerWrapper(float eta, AlgorithmType algorithmType)
        {
            switch (algorithmType)
            {
            case AlgorithmType::Sgd:
                m_LazyCreateLearner = [=](const Ptr<ExpressionGraph>& graph)
                {
                    return CNTK::SGDLearner(graph->m_allParameters,
                                            CNTK::LearningRateSchedule(std::vector<double>{ eta }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1)/*,
                                            AdditionalLearningOptions additionalOptions = AdditionalLearningOptions()*/);
                };
                break;
            case AlgorithmType::Adam:
                m_LazyCreateLearner = [=](const Ptr<ExpressionGraph>& graph)
                {
                    return CNTK::AdamLearner(graph->m_allParameters,
                                            CNTK::LearningRateSchedule(std::vector<double>{ eta }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            CNTK::MomentumSchedule(std::vector<double>{ 0.9 }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            /*unitGain=*/true,
                                            CNTK::MomentumSchedule(std::vector<double>{ 0.999 }, CNTK::TrainingParameterSchedule<float>::FullDataSweep, 1),
                                            /*epsilon=*/1e-8, /*adamax=*/false/*
                                            AdditionalLearningOptions additionalOptions = AdditionalLearningOptions()*/);
                };
                break;
            }
        }
        // TODO: sample count?
        void update(const Ptr<ExpressionGraph>& graph)
        {
            if (!m_learner)
                m_learner = m_LazyCreateLearner(graph);
            // sample count 1 disables all rescaling, and expects the gradient as the user wants it
            m_learner->Update(graph->m_allGradients, /*trainingSampleCount=*/1);
        }
    private:
        std::function<CNTK::LearnerPtr(const Ptr<ExpressionGraph>&)> m_LazyCreateLearner;
        CNTK::LearnerPtr m_learner;
    };
    template<AlgorithmType algorithmType>
    Ptr<OptimizerWrapper> Optimizer(float eta)
    {
        return New<OptimizerWrapper>(eta, algorithmType);
    }
}

#endif // __MARIAN_CNTK
