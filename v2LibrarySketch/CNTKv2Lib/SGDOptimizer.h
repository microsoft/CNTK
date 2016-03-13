#pragma once

#include "Function.h"
#include "Learner.h"
#include "Reader.h"
#include "Model.h"
#include "TrainingControl.h"

namespace CNTK
{
    class SGDOptimizer
    {
    public:
        SGDOptimizer(Model model, std::unordered_set<Learner> paramLearners);

        // Optimizes the model parameters using the specified "inputs" values for model Inputs
        // Uses specified 'trainingCriterion' node as the loss function and returns 
        // the computed values for all specified 'outputs' variables 
        std::unordered_map<Variable, Value> TrainMB(std::unordered_map<Variable, Value> inputs,
                                                    Function trainingCriterion,
                                                    std::unordered_set<Variable> outputs);

        // Trains the model for an entire trainign corpus fed by the specified 'reader'
        void TrainCorpus(Reader reader, Function trainingCriterion, TrainingControl driver);

        void Checkpoint(std::ostream outModelStream, std::ostream outCheckpointStream);
        void RestoreFromCheckpoint(std::istream inModelStream, std::istream inCheckpointStream);
    };
}
