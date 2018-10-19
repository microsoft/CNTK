#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Internals/ComputationGraphAlgorithms.h"
#include "core/graph/graph.h"

namespace CNTK
{
    class ScanLoopState
    {
    public:
        ScanLoopState(const Variable initialState, onnxruntime::NodeArg *initialStateNodeArg, const Variable stateOutput, int delay) :
            m_initialState(initialState),
            m_initialStateNodeArg(initialStateNodeArg),
            m_stateOutput(stateOutput),
            m_delay(delay),
            m_hasInitializer(false)
        {}

        Variable m_initialState;
        onnxruntime::NodeArg *m_initialStateNodeArg;
        Variable m_stateOutput;
        onnx::TensorProto m_initialStateTensor;
        bool m_hasInitializer;
        int m_delay;
    };

    class ScanLoop
    {
    public:
        ScanLoop(const std::vector<Variable> &inputs, const std::vector<Variable> &outputs,
            const std::vector<Variable> &scanInputs, const std::vector<Variable> &scanOutputs, const std::vector<FunctionPtr> &body) :
            m_inputs(inputs),
            m_outputs(outputs),
            m_scanInputs(scanInputs),
            m_scanOutputs(scanOutputs),
            m_body(body),
            m_scanOpCreated(false)
        {}

        std::vector<Variable> m_inputs, m_outputs, m_scanInputs, m_scanOutputs;
        std::vector<FunctionPtr> m_body;
        std::vector<ScanLoopState> scanLoopStates;
        std::vector<FunctionPtr> m_visited;
        bool m_scanOpCreated;
    };

    // Implementation of a graph based on ComputationNodes.
    class CNTKModelGraph : public DirectedGraph<FunctionPtr>
    {
        std::vector<FunctionPtr> m_roots;

    public:
        CNTKModelGraph(const std::vector<FunctionPtr>& roots) : m_roots(roots) {}

        std::vector<FunctionPtr> Predecessors(const FunctionPtr& node) const override
        {
            std::vector<FunctionPtr> predecessors;
            for (auto &input : node->Inputs())
            {
                if (input.Owner())
                    predecessors.push_back(input.Owner());
            }
            return predecessors;
        }

        const std::vector<FunctionPtr>& Roots() const override
        {
            return m_roots;
        }
    };

    std::wstring ToString(FunctionPtr f)
    {
        return L"( " + f->Name() + L": " + f->Uid() + L")";
    }

    void BuildLoops(const std::vector<FunctionPtr>& roots,
        std::vector<ScanLoop> &scanLoops)
    {
        std::vector<StrongComponent<FunctionPtr>> loops;
        CNTKModelGraph cntkModelGraph(roots);
        loops = StrongComponents<FunctionPtr>(cntkModelGraph);

        // Sort nodes inside the strong components in the evaluation order.
        std::function<bool(const FunctionPtr&)> delay
            = [](const FunctionPtr& f) 
        { 
            if (f->OpName() == L"PastValue")
                return 1;
            if (f->OpName() == L"FutureValue")
                return -1;
            else
                return 0;
        };

        EvaluationSort(cntkModelGraph, delay, loops);

        // Attributes:
        // body: N+M inputs, N+K outputs, N is the # of states, M inputs, K outputs)
        // directions(M):
        // num_scan_inputs(M):
        //      
        // Inputs:
        // sequence_length:
        //      max sequence length if not specified
        // initial_state_and_scan_inputs(N + M):
        //      initial_states are constant attributes from step functions 
        //      scan_inputs are input to this loop body with sequence axis
        //
        // Outputs:
        // final_state_and_scan_outputs(N + K):
        //      final_state: ?
        //      scan_outputs are outputs from this loop body with sequence axis

        std::vector<std::vector<Variable>> loopinputs, loopoutputs, scaninputs, scanoutputs;
        loopinputs.resize(loops.size());
        loopoutputs.resize(loops.size());
        scaninputs.resize(loops.size());
        scanoutputs.resize(loops.size());
        bool nestedSearchInsideBlockFunction = false;
        std::vector<FunctionPtr> visited;
        for (auto &root : roots)
        {
            root->PreorderTraverse([&root, &loops, &loopinputs, &loopoutputs, &scaninputs, &scanoutputs, &visited](const FunctionPtr& function) {
                if (std::find(visited.begin(), visited.end(), function) != visited.end())
                    return;

                for (int l = 0; l < loops.size(); l++)
                {
                    const StrongComponent<FunctionPtr> &loop = loops[l]; 
                    std::vector<Variable> &inputs = loopinputs[l];
                    std::vector<Variable> &outputs = loopoutputs[l];
                    const std::vector<FunctionPtr> &nodes = loop.Nodes();
                    if (std::find(nodes.begin(), nodes.end(), function) != nodes.end())
                    {
                        // if a function is part of a loop, any its inputs that are not from the loop body 
                        // is an input.
                        for (auto &input : function->Inputs())
                        {
                            if (!input.Owner() || (input.Owner() && std::find(nodes.begin(), nodes.end(), input.Owner()) == nodes.end()))
                            {
                                if (std::find(inputs.begin(), inputs.end(), input) == inputs.end())
                                {
                                    inputs.push_back(input);
                                    if (input.DynamicAxes().size() == 2)
                                        scaninputs[l].push_back(input);
                                }
                            }
                        }
                    }
                    else 
                    {
                        // if a function is not part of the loop and any of its inputs is from the loop
                        // that input variable is an output from the loop
                        for (auto &input : function->Inputs())
                        {
                            if (input.Owner() && std::find(nodes.begin(), nodes.end(), input.Owner()) != nodes.end())
                            {
                                if (std::find(outputs.begin(), outputs.end(), input) == outputs.end())
                                {
                                    outputs.push_back(input);
                                    if (input.DynamicAxes().size() == 2)
                                        scanoutputs[l].push_back(input);
                                }
                            }
                        }
                    }
                }
            }, nestedSearchInsideBlockFunction);

            // a corner case: if root src is in the loop body, it shall be an output as well. 
            for (int l = 0; l < loops.size(); l++)
            {
                const StrongComponent<FunctionPtr> &loop = loops[l];
                if (std::find(loop.Nodes().begin(), loop.Nodes().end(), root) != loop.Nodes().end())
                    for (auto output : root->Outputs())
                        if (std::find(scanoutputs[l].begin(), scanoutputs[l].end(), output) == scanoutputs[l].end())
                            scanoutputs[l].push_back(output);
            }
        }

        std::vector<std::vector<FunctionPtr>> loopstepfunctions;
        std::vector<std::vector<Variable>> loopStates;
        std::vector<bool> filterOutBlockRNNs(loops.size(), false);
        loopstepfunctions.resize(loops.size());
        for (int l = 0; l < loops.size(); l++)
        {
            const StrongComponent<FunctionPtr> &loop = loops[l];
            const std::vector<FunctionPtr> &nodes = loop.Nodes();
            for (auto &f : nodes)
            {
                if (f->OpName() == L"PastValue" || f->OpName() == L"FutureValue")
                    loopstepfunctions[l].push_back(f);
                else if (f->OpName() != L"LSTM" && f->OpName() != L"GRU" && f->OpName() != L"RNNStep")
                    filterOutBlockRNNs[l] = true;
            }
        }

        for (int l = 0; l < loops.size(); l++)
        {
            if (filterOutBlockRNNs[l])
            {
                ScanLoop scanLoop(loopinputs[l], loopoutputs[l], scaninputs[l], scanoutputs[l], loops[l].Nodes());
                scanLoops.push_back(scanLoop);
            }
        }
    }
}