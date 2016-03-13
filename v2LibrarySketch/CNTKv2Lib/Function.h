#pragma once

#include <set>
#include <unordered_set>
#include <unordered_map>
#include "Variable.h"
#include "Value.h"

namespace CNTK
{
    typedef void* BackpropState;

    // Represents a differentiable function that may be composed
    // of an underlying graph of other Function objects and Input variables
    // TODO: Should this be called Function instead?
    class Function
    {
    protected:
        Function(std::unordered_set<Variable> inputs, std::unordered_set<Variable> outputs);

    public:
        // Bind an input variable for the Function to a newly specified variable
        // which can be an Input, Constant or Computed variable
        void SetInput(Variable inputVar, Variable newVar);

        // Computes the values of speficied "evalSet" variables using provided "inputs" values
        // for the Input variables of the node.
        virtual std::unordered_map<Variable, Value> Evaluate(std::unordered_set<Variable> evalSet,
                                                             std::unordered_map<Variable, Value> inputs);

        // Computes the values of speficied "evalSet" variables using provided "inputs" values
        // for the Input variables of the node. Note that this method retains all intermediate 
        // variable values that may be needed during backpropagation of gradients from the outputs
        // to any of the inputs of the Function, in a subsequent ComputeGradients call. The backpropSate
        // value returned serves a cookie to identify of the preserved state in the Function corresponds
        // to the same computation that is back propagated in a later call to "ComputeGradients"
        virtual std::unordered_map<Variable, Value> Evaluate(std::unordered_set<Variable> evalSet,
                                                             std::unordered_map<Variable, Value> inputs,
                                                             BackpropState& backpropState);

        // Computes the gradients corresponding to the specified set of input variables in "gradientComputeSet" using the supplied "rootGradientValues" for one or
        // more of the output variables of the Function.
        // The "state" parameter is an optional cookie from an previous Evaluate call on the Function with the same inputs that this gradient backPropagation corresponds to.
        // In case of an mismatch between the supplied 'state' and the internal intermediate values of the Function, the forward values required for backPropagation 
        // are recomputed
        virtual std::unordered_map<Variable, Value> ComputeGradients(BackpropState state,
                                                                     std::unordered_set<Variable> gradientComputeSet,
                                                                     std::unordered_map<Variable, Value> rootGradientValues);

        virtual void Save(std::ostream outStream);

        // TODO: How to load an externally defined node?

        // First output variable
        Variable Output();
        const std::wstring& Name() const;

        // All output variables whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Outputs(const std::wstring& nameFilterRegex);

        // All input vars (leaf descendants) whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Inputs(const std::wstring& nameFilterRegex);

        // All vars (including intermediate values) whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Variables(const std::wstring& nameFilterRegex);

        std::unordered_set<Function> Children(const std::wstring& nameFilterRegex);

        // TODO: A bevy of helper methods to reflect on the Function's underlying network structure
        // and modify the structure. Also provide the ability to "Visit" the graph
    };

    // Built-in ComputationNodes
    Function Times(Variable leftOperand, Variable rightOperand, const std::wstring& name = L"");
    Function Plus(Variable leftOperand, Variable rightOperand, const std::wstring& name = L"");
    Function ReLU(Variable operand, const std::wstring& name = L"");
    Function Sigmoid(Variable operand, const std::wstring& name = L"");
    Function Tanh(Variable operand, const std::wstring& name = L"");
    Function CrossEntropyWithSoftmax(Variable output, Variable labels, const std::wstring& name = L"");
    Function PredictionError(Variable prediction, Variable labels, const std::wstring& name = L"");
    Function Exp(Variable operand, const std::wstring& name = L"");
    Function PastValue(Variable initialState, Variable operand, const std::wstring& name = L"");
    Function Scale(Variable leftOperand, Variable rightOperand, const std::wstring& name = L"");
    Function DiagTimes(Variable leftOperand, Variable rightOperand, const std::wstring& name = L"");
    Function ElementTimes(Variable leftOperand, Variable rightOperand, const std::wstring& name = L"");
}
