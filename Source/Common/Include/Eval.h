//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the EVAL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// EVAL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#if defined(EVAL_EXPORTS)
#define EVAL_API __declspec(dllexport)
#elif defined(EVAL_LOCAL)
#define EVAL_API
#else
#define EVAL_API __declspec(dllimport)
#endif
#else
#define EVAL_API
#endif

#include <map>
#include <vector>
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

enum NodeGroup
{
    nodeInput,  // an input node
    nodeOutput, // an output node
    nodeSpecified
};

// IEvaluateModel - interface used by decoders and other components that need just evaluator functionality in DLL form
// NOTICE: This interface is a public interface for evaluating models in CNTK. 
//         Changes to this interface may affect other projects, such as Argon and LatGen,
//         and therefore need to be communicated with such groups.
template <class ElemType>
class IEvaluateModel // Evaluate Model Interface
{
public:
    virtual void Init(const std::string& config) = 0;
    virtual void Destroy() = 0;

    virtual void CreateNetwork(const std::string& networkDescription) = 0;
    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup) = 0;
    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName) = 0;
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs) = 0;
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs) = 0;
    virtual void ResetState() = 0;
};

// GetEval - get a evaluator type from the DLL
// since we have 2 evaluator types based on template parameters, exposes 2 exports
// could be done directly with the templated name, but that requires mangled C++ names
template <class ElemType>
void EVAL_API GetEval(IEvaluateModel<ElemType>** peval);
extern "C" EVAL_API void GetEvalF(IEvaluateModel<float>** peval);
extern "C" EVAL_API void GetEvalD(IEvaluateModel<double>** peval);

// Data Reader class
// interface for clients of the Data Reader
// mirrors the IEvaluateModel interface, except the Init method is private (use the constructor)
template <class ElemType>
class Eval : public IEvaluateModel<ElemType>, protected Plugin
{
private:
    IEvaluateModel<ElemType>* m_eval; // evaluation class pointer

    void GetEvalClass(const std::string& config);

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy();

public:
    // EvaluateModel Constructor
    // config - configuration information:
    // deviceId=auto ( can be [0,all,cpu,0:2:3,auto] define accellerators (GPUs) to use, or the CPU
    // modelPath=c:\models\model.dnn (model path, if not specified, must call LoadModel() method before Evaluate()
    // minibatchSize=1024 (minibatch size used during evaluation if < passed data size)
    Eval(const std::string& config);
    virtual ~Eval();

    // CreateNetwork - create a network based on the network description
    // networkDescription - network description
    virtual void CreateNetwork(const std::string& networkDescription);

    // GetNodeDimensions - Get the node dimensions of the specified nodes
    // dimensions - map from name of node to dimension of the node
    // nodeGroup - type of node we are requesting (input/output/specified)
    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup);

    // StartEvaluateMinibatchLoop - Prepare network for Evaluate() calls.
    // ouputNodeName - name of node that will be evaluated
    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName);

    // Evaluate - Evaluate using the model with the given inputs and outputs
    // inputs - map from node name to input vector
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs);

    // Evaluate - Evaluate using the network without input, and provide the outputs
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs);

    virtual void Init(const std::string& config);
    virtual void ResetState();
};
} } }
