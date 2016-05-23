//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKEval.h - Include file for the CNTK Evaluation DLL
// 
// NOTICE: This interface is a public interface for evaluating models in CNTK. 
//         Changes to this interface may affect other projects, such as Argon and LatGen,
//         and therefore need to be communicated with such groups.
//
#pragma once

#include <string>
#include <map>
#include <vector>

#include "Eval.h"
#include "EvalReader.h"
#include "EvalWriter.h"

#include "ComputationNetwork.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
class CNTKEvalBase : public IEvaluateModelBase<ElemType>
{
protected:
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    ConfigParameters m_config;
    ComputationNetworkPtr m_net;

    // constructor
    CNTKEvalBase() : m_net(nullptr) { }
public:

    // CreateNetwork - create a network based on the network description
    // networkDescription - network description
    virtual void CreateNetwork(const std::string& networkDescription);
    virtual void Init(const std::string& config);
    virtual void Destroy();
};

// ------------------------------------------------------------------------
// Basic interface
// ------------------------------------------------------------------------
template <typename ElemType>
class CNTKEval : public CNTKEvalBase<ElemType>, public IEvaluateModel<ElemType>
{
    EvalReader<ElemType>* m_reader;
    EvalWriter<ElemType>* m_writer;
    std::map<std::wstring, size_t> m_dimensions;
    size_t m_start;
public:
    CNTKEval() : CNTKEvalBase<ElemType>(), m_reader(nullptr), m_writer(nullptr) {}

    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup);

    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName);

    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs);

    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs);

    virtual void Destroy() override;

    virtual void CreateNetwork(const std::string& networkDescription) override
    {
        CNTKEvalBase<ElemType>::CreateNetwork(networkDescription);
    }
    
    virtual void Init(const std::string& config) override
    {
        CNTKEvalBase<ElemType>::Init(config);
        m_start = 0;
    }

    virtual void ResetState() override
    {
        m_start = 1 - m_start;
    }
};



// ------------------------------------------------------------------------
// Extended interface
// ------------------------------------------------------------------------
template <typename ElemType>
class CNTKEvalExtended : public CNTKEvalBase<ElemType>, public IEvaluateModelExtended<ElemType>
{
    virtual VariableSchema GetOutputSchema() const override;

    virtual void StartForwardEvaluation(std::vector<wstring> outputs) override;

    virtual VariableSchema GetInputSchema() const override;

    virtual void ForwardPass(const Variables<ElemType>& inputs, Variables<ElemType>& output) override;

    virtual void Destroy() override;

    virtual void CreateNetwork(const std::string& networkDescription) override
    {
        CNTKEvalBase<ElemType>::CreateNetwork(networkDescription);
    }

    virtual void Init(const std::string& config) override
    {
        CNTKEvalBase<ElemType>::Init(config);
    }
private:
    static VariableLayout ToVariableLayout(const ComputationNodeBasePtr n);
    std::vector<ComputationNodeBasePtr> m_outputNodes;
    std::shared_ptr<ScopedNetworkOperationMode> m_scopedNetworkOperationMode;
    std::vector<ComputationNodeBasePtr> m_inputNodes;
    StreamMinibatchInputs m_inputMatrices;
};
} } }
