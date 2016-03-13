#pragma once

#include "Function.h"

namespace CNTK
{
    // Abstraction for updating a specified set of parameters of a model
    // TODO: Should learners also have support for callbacks to enable controlling learning
    class Learner
    {
    protected:
        // Gradient based learner objects
        Learner(std::unordered_set<Variable> parameters);

        // Learners that are not gradient based but instead require one or more output values
        // of the model for updating the parameters. Examples are BatchNormalization nodes' 
        // smoothed mean/stdDev values, initial state of PastValue or FutureValue nodes etc.
        Learner(std::unordered_set<Variable> parameters, std::unordered_map<Variable, std::list<Variable>> updateDependencies);

    public:
        std::unordered_set<Variable> Parameters() const;

        virtual void Checkpoint(std::ostream outCheckpointStream);
        virtual void RestoreFromCheckpoint(std::istream inCheckpointStream);

        // TODO: Additional training info needed to be passed besides sampleCount?
        virtual std::unordered_map<Variable, Value> UpdateParameters(std::unordered_map<Variable, Value> params,
                                                                     std::unordered_map<Variable, Value> updateDependencyValues,
                                                                     size_t trainingSampleCount);
    };

    // Builtin learners
    Learner MomentumLearner(std::unordered_set<Variable> parameterVariables, double momentumTimeConstant);
    Learner AdaGradLearner(std::unordered_set<Variable> parameterVariables, double momentumTimeConstant, double gaussianNoiseInjectStd);
    Learner BNLearner(Function BNNode);
    Learner PastValueLearner(std::unordered_set<Function> pastValueNodes);
    Learner FutureValueLearner(std::unordered_set<Function> futureValueNodes);

    // Parallel learners
    Learner ModelAveragingLearner(size_t syncFrequency, Learner baseLearner);
    Learner QuantizedGradientsSGDLearner(size_t numGradientBits, bool doubleBuffering, Learner baseLearner);
}
