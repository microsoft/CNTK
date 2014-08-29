//
// <copyright file="PTaskExecutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "PTaskComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// PTaskNodeEvaluator
// Process the Network Description Language into a Computation Network useable
// by PTaskExecutionEngine.
template <typename ElemType>
class PTaskNodeEvaluator : public NDLNodeEvaluator<ElemType>
{
public:
    // Constructor - create evaluator
    PTaskNodeEvaluator(PTaskComputationNetwork<ElemType>& cn)
        : m_net(cn)
    { }

    // Evaluate - evaluate a node and translate into underlying 
    // node - node we are evaluating
    // baseName - base name for all symbols at this level
    virtual void Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const int pass)
    {
        // constants don't need to be evaluated, they just translate into numbers...
        if (node->GetType() == ndlTypeConstant 
            || node->GetType() == ndlTypeArray) // currently arrays only used for node lists, in the future may be used for other things
            return;

        // get the parameters
        std::vector<NDLNode<ElemType>*> parameter = node->GetParameters();
        if (parameter.size() < 1)
        {
            Error("Node with no parameters, %s\n", node->GetName().c_str());
        }

        // get the name for the symbol to be used by CN nodes
        std::wstring name = msra::strfun::utf16(node->GetName());
        if (!baseName.empty())
        {
            name = baseName + L"." + name;
        }

        if (node->GetValue() == "InputValue")
        {
            if (pass > 0)
                return;

            // get dimensions of input
            size_t rows = parameter[0]->GetScalar();

            // check for second dimension, otherwise default to 1
            size_t cols = 1;
            if (parameter.size() > 1)
            {
                cols = parameter[1]->GetScalar();
            }

            ComputationNodePtr input = m_net.CreateInputNode(name, rows, cols);
            node->SetEvalValue(input);
        }
        else if (node->GetValue() == "LearnableParameter")
        {
            // get dimensions of input
            size_t rows = parameter[0]->GetScalar();

            // check for second dimension, otherwise default to 1
            size_t cols = 1;
            if (parameter.size() > 1)
            {
                cols = parameter[1]->GetScalar();
            }

            if (pass == 0)
            {

                bool needGradient = true;
                ComputationNodePtr nodePtr = m_net.CreateLearnableParameter(name, rows, cols);
                node->SetEvalValue(nodePtr);

                nodePtr->NeedGradient() = needGradient;
            }
            else
            {
                static int randomSeed = 1;
                ComputationNodePtr nodePtr = (ComputationNodePtr)m_net.GetNodeFromName(name);

                bool init = true;
                bool uniformInit=true;
                if (init)
                {
                    InitLearnableParameters(nodePtr, cols, randomSeed++, uniformInit);
                }
            }
        }
        else if (node->GetValue() == "ConstantScalarParameter")
        {
            if (pass > 0)
                return;

            size_t rows = 1;
            size_t cols = 1;
            bool needGradient = false;
            bool init = false;
            ElemType val = parameter[0]->GetScalar();
            ComputationNodePtr nodePtr = m_net.CreateLearnableParameter(name, rows, cols);
            node->SetEvalValue(nodePtr);

            nodePtr->NeedGradient() = needGradient;
            nodePtr->FunctionValues().SetValue(val);
        }
        else
        {
            ComputationNodePtr nodePtr = NULL;
            if (pass == 0)
            {
                nodePtr = m_net.CreateComputationNode(node->GetValue(), name);
                node->SetEvalValue(nodePtr);
            }

            std::vector<void*> inputs = EvaluateParameters(node, baseName, pass);

            if (pass == 0)
            {
                switch (inputs.size())
                {
                case 1:
                    nodePtr->AttachInputs(ComputationNodePtr(inputs[0]));
                    break;
                case 2:
                    nodePtr->AttachInputs(ComputationNodePtr(inputs[0]), ComputationNodePtr(inputs[1]));
                    break;
                case 3:
                    nodePtr->AttachInputs(ComputationNodePtr(inputs[0]), ComputationNodePtr(inputs[1]), ComputationNodePtr(inputs[2]));
                    break;
                default:
                    Error("Invalid number of parameters name = '%s' call = '%s'\n", node->GetName().c_str(), node->GetValue().c_str());
                }
            }
        }

    }

    virtual ~PTaskNodeEvaluator()
    {

    }

private:
    PTaskComputationNetwork<ElemType>& m_net;
    typedef ComputationNode<ElemType>* ComputationNodePtr;

    void InitLearnableParameters(ComputationNodePtr node, const size_t inputSize, ULONG randomSeed, bool uniformInit)
    {
        ElemType initValueScale = (ElemType)1.0;
        if (uniformInit)
        {
            ElemType randRange = (ElemType)0.05; //initValueScale/sqrt(inputSize);
            node->FunctionValues().SetUniformRandomValue(-randRange, randRange, randomSeed);
        }
        else
        {
            ElemType randInitstd = (ElemType)0.2 * initValueScale/sqrt((ElemType)inputSize);
            node->FunctionValues().SetGaussianRandomValue(0,randInitstd,randomSeed);
        }
    }
};

    template class PTaskComputationNetwork<float>; 
    template class PTaskComputationNetwork<double>;

// PTaskExecutionEngine
template <typename ElemType>
class PTaskExecutionEngine : public IExecutionEngine<ElemType>
{
public:
    PTaskExecutionEngine()
    {
        m_nodeEvaluator = new PTaskNodeEvaluator<ElemType>(m_computationNetwork);
    }

    virtual ~PTaskExecutionEngine()
    {

    }

    ComputationNetwork<ElemType>& GetComputationNetwork()
    {
        return m_computationNetwork;
    }

    NDLNodeEvaluator<ElemType>& GetNodeEvaluator()
    {
        return *m_nodeEvaluator;
    }

private:
    PTaskComputationNetwork<ElemType> m_computationNetwork;
    PTaskNodeEvaluator<ElemType>* m_nodeEvaluator;

};
    template class PTaskExecutionEngine<float>; 
    template class PTaskExecutionEngine<double>;

}}}