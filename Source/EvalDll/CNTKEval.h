//
// <copyright file="CNTKEval.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// CNTKEval.h - Include file for the CNTK Evaluation DLL
#pragma once

#include <string>
#include <map>
#include <vector>

#include "Eval.h"
#include "EvalReader.h"
#include "EvalWriter.h"

#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CNTKEval : public IEvaluateModel<ElemType>
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    EvalReader<ElemType>* m_reader;
    EvalWriter<ElemType>* m_writer;
    ConfigParameters m_config;
    ComputationNetworkPtr m_net;
    std::map<std::wstring, size_t> m_dimensions;
    size_t m_start;

public:
    // constructor
    CNTKEval()
        : m_reader(nullptr), m_net(nullptr)
    {
    }

    // LoadModel - load a model from the specified path
    // modelFileName - file holding the model to load
    virtual void LoadModel(const std::wstring& modelFileName);

    // GetNodeDimensions - Get the node dimensions of the specified nodes
    // dimensions - map from name of node to dimension of the node
    // nodeGroup - type of node we are requesting (input/output/specified)
    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup);

    // StartEvaluateMinibatchLoop - Prepare network for Evaluate() calls.
    // ouputNodeName - name of node that will be evaluated
    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName);

    // Evaluate - Evalute using the model with the given inputs and outputs
    // inputs - map from node name to input vector
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs);

    virtual void Init(const std::string& config);
    virtual void Destroy();
    virtual void ResetState();
};
}
}
}