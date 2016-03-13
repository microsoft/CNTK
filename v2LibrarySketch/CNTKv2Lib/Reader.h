#pragma once

#include "Function.h"

namespace CNTK
{
    class Reader
    {
    public:
        virtual std::unordered_map<Variable, Value> GetInputs(std::unordered_set<Variable> inputSet);

        virtual void Checkpoint(std::ostream outCheckpointStream);
        virtual void RestoreFromCheckpoint(std::istream inCheckpointStream);
    };

    // Builtin readers
    Reader TextReader(/*Reader configuration parameters*/);
}
