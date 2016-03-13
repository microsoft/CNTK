#pragma once

#include "Function.h"
#include <unordered_map>

namespace CNTK
{
    // An object that represents a collection of root Nodes and all parameter inputs of the 
    // graph comprised of specified root nodes
    class Model : public Function
    {
    public:
        Model(std::unordered_set<Function> rootNodes, std::unordered_map<Variable, Value> parameterValues);
        Model(std::wstring modelPath);

        void Save(std::wstring modelPath);

        // Compute values corresponding to specified variables of the model given supplied "inputs" values for the model's 
        // non-parameter Input variables
        // TODO: Support passing in pre-existing Value objects to write the results into
        std::unordered_map<Variable, Value> Evaluate(std::unordered_set<Variable> evalSet, std::unordered_map<Variable, Value> inputs);

        std::unordered_map<Variable, Value> Parameters();
    };
}
