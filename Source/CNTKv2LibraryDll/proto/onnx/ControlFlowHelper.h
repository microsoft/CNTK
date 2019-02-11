#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Internals/ComputationGraphAlgorithms.h"
#include "core/graph/graph.h"
#include "Operators.h"
#include <utility>

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    class ScanLoopState
    {
    public:
        ScanLoopState(const Variable &initialState, onnxruntime::NodeArg *initialStateNodeArg, const Variable &stateOutput, int delay) :
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

    std::wstring ToString(FunctionPtr f)
    {
        return L"( " + f->Name() + L": " + f->Uid() + L")";
    }

    bool IsStepFunction(FunctionPtr f)
    {
        return f->OpName() == L"PastValue" || f->OpName() == L"FutureValue";
    }

    class ScanLoop
    {
    public:
        ScanLoop(const std::vector<Variable> &inputs, const std::vector<Variable> &outputs,
            const std::vector<Variable> &scanInputs, const std::vector<Variable> &scanOutputs, const std::vector<FunctionPtr> &body,
            const std::vector<FunctionPtr> &loopstepfunctions) :
            m_inputs(inputs),
            m_outputs(outputs),
            m_scanInputs(scanInputs),
            m_scanOutputs(scanOutputs),
            m_body(body),
            m_loopstepfunctions(loopstepfunctions),
            m_scanOpCreated(false)
        {
            Validate();
            // collect 
            // collect nodes in RNN ops as part of the body
            for (auto &f : m_body)
            {
                if (ONNX::Operators::IsRNNOp(ToLegacyString(ToUTF8(f->OpName()))))
                {
                    CollectInternalNodes(f, m_rnnInternalBodies, true);
                    
                }
                else if (f->IsBlock())
                {
                    // RNNs and other block functions that would be traversed into by CreateNode
                    // shall be treated the same at here and below of mapping block outputs to underlying variables.
                    CollectInternalNodes(f, m_blockInternalBodies, true);
                }
            }

            // if RNN is in the loop, we want to map scan outputs that are from LSTM 
            // to LSTM block underlying variable
            for (auto &rnn : this->m_rnnInternalBodies)
            {
                FunctionPtr rnnF = rnn.first;
                MapScanOutputFromBlockOutoutsToUnderlyingVariable(rnnF);
            }

            for (auto &block : this->m_blockInternalBodies)
            {
                FunctionPtr blockF = block.first;
                MapScanOutputFromBlockOutoutsToUnderlyingVariable(blockF);
            }
        }

        void Validate()
        {
            for (auto output : m_outputs)
            {
                if (output.Owner() && IsStepFunction(output.Owner()))
                {
                    fprintf(stderr, "Warning: a loop final state is consumed by an outside op. This is not supported. It has sequence dimension mismatch.");
                }
            }
        }

        bool IsOuterScopeInput(Variable v) const
        {
            if (std::find(this->m_inputs.cbegin(), this->m_inputs.cend(), v) == this->m_inputs.cend())
                return false;
            if (std::find(this->m_scanInputs.cbegin(), this->m_scanInputs.cend(), v) != this->m_scanInputs.cend())
                return false;

            for (auto a : this->scanLoopStates)
                if (a.m_initialState == v)
                    return false;
            return true;
        }

        bool IsInBody(const FunctionPtr src)
        {
            if (std::find(this->m_body.begin(), this->m_body.end(), src) != this->m_body.end())
                return true;
            for (auto &rnn : this->m_rnnInternalBodies)
            {
                if (std::find(rnn.second.begin(), rnn.second.end(), src) != rnn.second.end())
                    return true;
            }

            for (auto &block : this->m_blockInternalBodies)
            {
                if (std::find(block.second.begin(), block.second.end(), src) != block.second.end())
                    return true;
            }
            return false;
        }

        void MapScanOutputFromBlockOutoutsToUnderlyingVariable(const FunctionPtr blkF)
        {
            if (CNTK::ONNX::Operators::IsBlockFnNotConvertedThroughBlockRoot(blkF))
                return;

            BlockFunction* block = dynamic_cast<BlockFunction *>(blkF.get());
            std::unordered_map<Variable, Variable> bm = block->CompositeOutputsMap();
            for (auto &blockOutput : blkF->Outputs())
            {
                for (int i = 0; i < m_scanOutputs.size(); i++)
                {
                    if (m_scanOutputs[i] == blockOutput)
                    {
                        if (bm.find(blockOutput) == bm.end())
                            LogicError("cannot map PastValue/Future's input to LSTM underlying output");
                        m_scanOutputs[i] = bm[blockOutput];
                    }
                }
            }
        }

        static void CollectInternalNodes(FunctionPtr src, 
            std::unordered_map<FunctionPtr, std::vector<FunctionPtr>> &internalBodies, bool collectRecursively = false)
        {
            FunctionPtr br = src->BlockRoot();
            std::vector<FunctionPtr> internalBody;
            br->PreorderTraverse([&internalBody](const FunctionPtr& function) {
                internalBody.push_back(function);
            }, false);
            internalBodies.insert(std::make_pair(src, internalBody));
            if (collectRecursively)
            {
                for (auto f : internalBody)
                {
                    if (f->IsBlock())
                        CollectInternalNodes(f, internalBodies, collectRecursively);
                }
            }
        }

        std::vector<Variable> m_inputs, m_outputs, m_scanInputs, m_scanOutputs;
        std::vector<FunctionPtr> m_loopstepfunctions;
        std::vector<FunctionPtr> m_body;
        std::vector<std::string> initializerAsInput;
        std::unordered_map<FunctionPtr, std::vector<FunctionPtr>> m_rnnInternalBodies;
        std::unordered_map<FunctionPtr, std::vector<FunctionPtr>> m_blockInternalBodies;
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

    void AddScanOutputVariable(std::vector<Variable>& scanoutput, Variable output)
    {
        // in case output is 
        if (output.Owner() && IsStepFunction(output.Owner()))
        {
            // if scan output is from a step function, we need to replace it with the step function's
            // input. Otherwise it will collid with the final state output and produce wrong numbers.
            // By doing this, onnx test cannot map CNTK's output to ONNX model's outputs. 
            // Test case generated will fail lotus because the scan output is mappted to final state output.
            // Before we can workout anything better, we post a warning here.
            fprintf(stderr, "Warning: The model has a scan op with output colliding with a final state. The scan output is replaced. Generated onnxruntime test case may not pass because of this.");
            scanoutput.push_back(output.Owner()->Inputs()[0]);
        }
        else
            scanoutput.push_back(output);
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
                                        AddScanOutputVariable(scanoutputs[l], input);
                                }
                            }
                        }
                    }
                }
            }, nestedSearchInsideBlockFunction);

            // corner case: 
            // if root src is in the loop body, it shall be an output as well. 
            for (int l = 0; l < loops.size(); l++)
            {
                const StrongComponent<FunctionPtr> &loop = loops[l];
                if (std::find(loop.Nodes().begin(), loop.Nodes().end(), root) != loop.Nodes().end())
                    for (auto output : root->Outputs())
                        if (std::find(loopoutputs[l].begin(), loopoutputs[l].end(), output) == loopoutputs[l].end())
                        {
                            loopoutputs[l].push_back(output);
                            if (output.HasSequenceAxis())
                                AddScanOutputVariable(scanoutputs[l], output);
                        }
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
                if (IsStepFunction(f))
                    loopstepfunctions[l].push_back(f);
                else if (f->OpName() != L"LSTM" && f->OpName() != L"GRU" && f->OpName() != L"RNNStep")
                    filterOutBlockRNNs[l] = true;
            }
        }

        for (int l = 0; l < loops.size(); l++)
        {
            if (filterOutBlockRNNs[l])
            {
                ScanLoop scanLoop(loopinputs[l], loopoutputs[l], scaninputs[l], scanoutputs[l], loops[l].Nodes(),
                    loopstepfunctions[l]);
                scanLoops.push_back(scanLoop);
            }
        }
    }
}