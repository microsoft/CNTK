//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKEval.cpp : Defines the exported functions for the CNTK DLL.
//

#include "stdafx.h"
#define EVAL_EXPORTS // creating the exports here
#include "Eval.h"
#include "Actions.h"
#include "CNTKEval.h"
#include "CPUMatrix.h" // for SetNumThreads()
#include "SimpleOutputWriter.h"
#include "NDLNetworkBuilder.h"
#ifdef LEAKDETECT
#include <vld.h> // leak detection
#endif
#include "BestGpu.h"
#include "MPIWrapper.h"

// TODO: Temporary mechanism to enable memory sharing for
// node output value matrices. This will go away when the
// sharing is ready to be enabled by default
bool g_shareNodeValueMatrices = false;

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void EVAL_API GetEval(IEvaluateModel<ElemType>** peval)
{
    *peval = new CNTKEval<ElemType>();
}

extern "C" EVAL_API void GetEvalF(IEvaluateModel<float>** peval)
{
    GetEval(peval);
}
extern "C" EVAL_API void GetEvalD(IEvaluateModel<double>** peval)
{
    GetEval(peval);
}

template <class ElemType>
void CNTKEval<ElemType>::Init(const std::string& config)
{
    m_start = 0;
    m_config.Parse(config);
    size_t nThreads = m_config("numCPUThreads", "1");
    CPUMatrix<ElemType>::SetNumThreads(nThreads);

    g_shareNodeValueMatrices = m_config(L"shareNodeValueMatrices", false);
}

// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <class ElemType>
void CNTKEval<ElemType>::Destroy()
{
    // cleanup everything
    m_net.reset();
    delete m_reader;
    delete m_writer;
    delete this;
}

// CreateNetwork - create a network based on the network description
// networkDescription - network description
template <class ElemType>
void CNTKEval<ElemType>::CreateNetwork(const std::string& networkDescription)
{
    ConfigParameters config;
    config.Parse(networkDescription);

    std::vector<wstring> outputNodeNames;
    m_net = GetModelFromConfig<ConfigParameters, ElemType>(config, outputNodeNames);
    
    if (m_net == nullptr)
    {
        LogicError("Unable to construct network from description");
    }
}

// GetNodeDimensions - Get the node dimensions of the specified nodes
// dimensions - map from name of node to dimension of the node, will be appended to for Input/Output scenarios
// nodeGroup - type of node we are requesting (input/output/specified)
// NOTE: when nodeGroup==specified the dimensions map is expected to be populated with the string names of the nodes requested, dimensions will be modified return the current value.
template <class ElemType>
void CNTKEval<ElemType>::GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup)
{
    if (m_net == NULL)
    {
        for (auto iter = dimensions.begin(); iter != dimensions.end(); iter++)
            iter->second = 0;
        return;
    }

    const auto& outputNodes = m_net->OutputNodes();
    switch (nodeGroup)
    {
    case nodeInput:
    {
        if (outputNodes.size() == 0)
        {
            LogicError("No Output nodes found: Cannot determine Input node dimensions due to lack of Output nodes.\n(are 'outputNodeNames' and/or 'OutputNodes' properly defined in the configuration file?)");
        }

        auto& nodes = m_net->InputNodes(outputNodes[0]);
        for (auto& node : nodes)
        {
            std::wstring name = node->NodeName();
            size_t size = node->GetSampleMatrixNumRows();
            dimensions[name] = size;
        }
        break;
    }
    case nodeOutput:
    {
        const auto& nodes = outputNodes;
        for (auto& node : nodes)
        {
            std::wstring name = node->NodeName();
            size_t size = node->GetSampleMatrixNumRows();
            dimensions[name] = size;
        }
        break;
    }
    case nodeSpecified:
        for (auto iter = dimensions.begin(); iter != dimensions.end(); iter++)
        {
            auto node = m_net->GetNodeFromName(iter->first);
            iter->second = node->GetSampleMatrixNumRows();
        }
        break;
    }
}

// StartEvaluateMinibatchLoop - Prepare network for Evaluate() calls.
// ouputNodeName - name of node that will be evaluated
template <class ElemType>
void CNTKEval<ElemType>::StartEvaluateMinibatchLoop(const std::wstring& outputNodeName)
{
    m_net->StartEvaluateMinibatchLoop(m_net->GetNodeFromName(outputNodeName));
}

// Evaluate - Evalute using the model with the given inputs and outputs
// inputs - map from node name to input vector
// outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
template <class ElemType>
void CNTKEval<ElemType>::Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs)
{
    size_t minibatchSize = m_config(L"minibatchSize", (size_t) 10240);
    // get the evaluation names from the output string
    vector<wstring> outNodeNames;

    ConfigParameters config;
    // config["deviceId"] = to_string(m_net->GetDeviceId());

    // create the reader if necessary
    if (m_reader == nullptr)
    {
        m_reader = new EvalReader<ElemType>(config);
    }

    // now set the data in the reader
    GetNodeDimensions(m_dimensions, nodeInput);
    m_reader->SetData(&inputs, &m_dimensions);
    m_reader->SetBoundary(m_start);
    
    // create the writer if necessary
    if (m_writer == nullptr)
    {
        m_writer = new EvalWriter<ElemType>(config);
    }
    // now set the data in the writer
    GetNodeDimensions(m_dimensions, nodeOutput);
    m_writer->SetData(&outputs, &m_dimensions);

    // call the evaluator
    SimpleOutputWriter<ElemType> eval(m_net);
    eval.WriteOutput(*m_reader, minibatchSize, *m_writer, outNodeNames);
}

// Evaluate - Evalute using the model with the given inputs and outputs
// outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
template <class ElemType>
void CNTKEval<ElemType>::Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs)
{
    // get the evaluation names from the output string
    vector<wstring> outNodeNames;

    ConfigParameters config;

    // create the writer if necessary
    if (m_writer == nullptr)
    {
        m_writer = new EvalWriter<ElemType>(config);
    }

    // now set the data in the writer
    GetNodeDimensions(m_dimensions, nodeOutput);
    m_writer->SetData(&outputs, &m_dimensions);

    // call the evaluator
    SimpleOutputWriter<ElemType> eval(m_net);
    eval.WriteOutput(*m_writer, outNodeNames);
}

// ResetState - Reset the cell state when we get start of an utterance
template <class ElemType>
void CNTKEval<ElemType>::ResetState()
{
    m_start = 1 - m_start;
}

// instantiate all the combinations we expect to be used
template class CNTKEval<double>;
template class CNTKEval<float>;
} } }
