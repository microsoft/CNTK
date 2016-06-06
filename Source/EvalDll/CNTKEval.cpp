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
#include "DataDeserializer.h"
#include "SequencePacker.h"
#include "NoRandomizer.h"
#include "HeapMemoryProvider.h"
#include "InputAndParamNodes.h"

// TODO: Temporary mechanism to enable memory sharing for
// node output value matrices. This will go away when the
// sharing is ready to be enabled by default
bool g_shareNodeValueMatrices = false;

namespace Microsoft { namespace MSR { namespace CNTK {


template <typename ElemType>
void CNTKEvalBase<ElemType>::Init(const std::string& config)
{
    m_config.Parse(config);
    size_t nThreads = m_config("numCPUThreads", "1");
    CPUMatrix<ElemType>::SetNumThreads(nThreads);
    g_shareNodeValueMatrices = m_config(L"shareNodeValueMatrices", false);
}


// CreateNetwork - create a network based on the network description
// networkDescription - network description
template <typename ElemType>
void CNTKEvalBase<ElemType>::CreateNetwork(const std::string& networkDescription)
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


// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template <typename ElemType>
void CNTKEvalBase<ElemType>::Destroy()
{
    // cleanup everything
    m_net.reset();
}


// ----------------------------------------------------------------------------
// Basic interface
// ----------------------------------------------------------------------------

template <typename ElemType>
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

// GetNodeDimensions - Get the node dimensions of the specified nodes
// dimensions - map from name of node to dimension of the node, will be appended to for Input/Output scenarios
// nodeGroup - type of node we are requesting (input/output/specified)
// NOTE: when nodeGroup==specified the dimensions map is expected to be populated with the string names of the nodes requested, dimensions will be modified return the current value.
template <typename ElemType>
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
template <typename ElemType>
void CNTKEval<ElemType>::StartEvaluateMinibatchLoop(const std::wstring& outputNodeName)
{
    m_net->StartEvaluateMinibatchLoop(m_net->GetNodeFromName(outputNodeName));
}

// Evaluate - Evalute using the model with the given inputs and outputs
// inputs - map from node name to input vector
// outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
template <typename ElemType>
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
template <typename ElemType>
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


template <typename ElemType>
void CNTKEval<ElemType>::Destroy()
{
    CNTKEvalBase<ElemType>::Destroy();
    delete m_reader;
    delete m_writer;
    delete this;
}

// instantiate all the combinations we expect to be used
template class CNTKEval<double>;
template class CNTKEval<float>;

// ----------------------------------------------------------------------------
// Extended interface
// ----------------------------------------------------------------------------

template<typename ElemType>
VariableLayout CNTKEvalExtended<ElemType>::ToVariableLayout(const ComputationNodeBasePtr n) 
{
    auto matrix = dynamic_pointer_cast<Matrix<ElemType>>(n->ValuePtr());
    return VariableLayout
    {
        /* name */          n->GetName(),
        /* type */          sizeof(ElemType) == sizeof(float) ? VariableLayout::Float32 : VariableLayout::Float64,
        /* storage */       matrix ? matrix->GetMatrixType() == MatrixType::DENSE ? VariableLayout::Dense :
                                matrix->GetMatrixType() == MatrixType::SPARSE ? VariableLayout::Sparse : 
                                VariableLayout::Undetermined :
                                VariableLayout::Undetermined,
        /* dimension */     n->GetSampleLayout().GetNumElements(),
        /* dynamic axis */  wstring(n->GetMBLayout() ? n->GetMBLayout()->GetAxisName() : L"*")
    };
}


template<typename ElemType>
void CNTKEvalExtended<ElemType>::StartForwardEvaluation(std::vector<wstring> outputNodeNames)
{
    m_scopedNetworkOperationMode = make_shared<ScopedNetworkOperationMode>(m_net, NetworkOperationMode::inferring);
    // allocate memory for forward computation
    m_outputNodes  = m_net->OutputNodesByName(outputNodeNames);
    m_inputNodes = m_net->InputNodesForOutputs(outputNodeNames);
    // allocate memory for forward computation
    m_net->AllocateAllMatrices({}, m_outputNodes, nullptr);
    m_net->StartEvaluateMinibatchLoop(m_outputNodes);
    m_inputMatrices = DataReaderHelpers::RetrieveInputMatrices(m_inputNodes);
} 

template<typename ElemType>
VariableSchema CNTKEvalExtended<ElemType>::GetOutputSchema() const
{
    VariableSchema schema;
    for (const auto& n : m_net->OutputNodes())
    {
        schema.push_back(ToVariableLayout(n));
    }
    return schema;
}

template<typename ElemType>
VariableSchema CNTKEvalExtended<ElemType>::GetInputSchema() const
{
    VariableSchema inputLayouts;
    auto nodes = m_inputNodes;
    if (nodes.size() == 0)
    {
        // Default to all nodes
        nodes = m_net->InputNodesForOutputs({});
    }

    for (const auto& n : nodes)
    {
        inputLayouts.push_back(ToVariableLayout(n));
    }
    return inputLayouts;
}

template<typename ElemType>
void CNTKEvalExtended<ElemType>::ForwardPass(const Variables<ElemType>& inputs, Variables<ElemType>& output)
{
    if (inputs.size() != (size_t)std::distance(m_inputMatrices.begin(), m_inputMatrices.end()))
    {
        RuntimeError("Expected %d inputs, but got %d", (int)std::distance(m_inputMatrices.begin(), m_inputMatrices.end()), (int)inputs.size());
    }

    int i = 0;
    for (auto& input : m_inputMatrices)
    {
        VariableBuffer<ElemType> buffer = inputs[i];
        shared_ptr<Matrix<ElemType>> matrix = dynamic_pointer_cast<Matrix<ElemType>>(input.second.matrix);
        auto type = matrix->GetMatrixType();
        int numRows = input.second.sampleLayout.GetNumElements();

        if (type == MatrixType::DENSE)
        {
            if (buffer.m_buffer.size() % numRows != 0)
            {
                RuntimeError("Input %ls: Expected input data to be a multiple of %ld, but it is %ld", m_inputNodes[i]->GetName().c_str(), numRows, buffer.m_buffer.size());
            }
            if (buffer.m_buffer.size() == 0)
            {
                RuntimeError("Input %ls: Expected at least one element.", m_inputNodes[i]->GetName().c_str());
            }
        }
        else if (type == MatrixType::SPARSE)
        {
            if (buffer.m_colIndices.size() < 2)
            {
                RuntimeError("Input %ls: Expected at least one element.", m_inputNodes[i]->GetName().c_str());
            } 
            if (buffer.m_colIndices[0] != 0)
            {
                RuntimeError("Input %ls: First element of column indices must be 0", m_inputNodes[i]->GetName().c_str());
            }
            if (buffer.m_colIndices[buffer.m_colIndices.size()-1] != buffer.m_indices.size())
            {
                RuntimeError("Input %ls: Last element of column indices must be equal to the size of indices (%ld), but was %d", m_inputNodes[i]->GetName().c_str(), buffer.m_indices.size(), buffer.m_colIndices[buffer.m_colIndices.size() - 1]);
            }
        }

        int numCols = type == MatrixType::DENSE ? buffer.m_buffer.size() / numRows : buffer.m_colIndices.size() - 1;
        assert(numCols >= 1);
        input.second.pMBLayout->Init(1, numCols);
        input.second.pMBLayout->AddSequence(0, 0, 0, numCols);
       
        if (type == MatrixType::DENSE)
        {
            matrix->SetValue(numRows, numCols, matrix->GetDeviceId(), buffer.m_buffer.data(), matrixFlagNormal);
        }
        else if (type == MatrixType::SPARSE)
        {
            // In the sparse case the m_data layout is identical to CUDA's CSC layout
            // (see http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc).
            matrix->SetMatrixFromCSCFormat(buffer.m_colIndices.data(), buffer.m_indices.data(), buffer.m_buffer.data(), buffer.m_buffer.size(), numRows, numCols);
        }

        ++i;
    }

    ComputationNetwork::BumpEvalTimeStamp(m_inputNodes);
    
    for (int i = 0; i < m_outputNodes.size(); ++i)
    {
        auto node = m_outputNodes[i];
        m_net->ForwardProp(node);
        shared_ptr<Matrix<ElemType>> outputMatrix = dynamic_pointer_cast<Matrix<ElemType>>(node->ValuePtr());
        auto pMBLayout = node->GetMBLayout();
        if (!pMBLayout) 
        {
            pMBLayout = make_shared<MBLayout>();
            pMBLayout->InitAsFrameMode(1); // treat this as if we have one single sample
        }

        const auto& seq = pMBLayout->GetAllSequences();
        if (seq.size() != 1)
        {
            RuntimeError("Only 1 sequence supported by this API"); // TODO
        }
        std::vector<ElemType>& vec = output[i].m_buffer;
        
        vec.resize(outputMatrix->GetNumElements());
        ElemType* data = const_cast<ElemType*>(vec.data());
        size_t numElements = outputMatrix->GetNumElements();
        outputMatrix->CopyToArray(data, numElements);
    }
}

template <typename ElemType>
void CNTKEvalExtended<ElemType>::Destroy()
{
    CNTKEvalBase<ElemType>::Destroy();
    delete this;
}

template <typename ElemType>
void EVAL_API GetEvalExtended(IEvaluateModelExtended<ElemType>** peval)
{
    *peval = new CNTKEvalExtended<ElemType>();
}

extern "C" EVAL_API void  GetEvalExtendedF(IEvaluateModelExtended<float>** peval)
{
    GetEvalExtended(peval);
}
extern "C" EVAL_API void GetEvalExtendedD(IEvaluateModelExtended<double>** peval)
{
    GetEvalExtended(peval);
}

template class CNTKEvalExtended<double>;
template class CNTKEvalExtended<float>;
} } }
