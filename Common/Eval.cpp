//
// <copyright file="Eval.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// Eval.cpp : Defines the exported functions for the DLL application.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#define EVAL_LOCAL
#include "Eval.h"
#include "basetypes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
std::string GetEvalName(ElemType)
{std::string empty; return empty;}

template<> std::string GetEvalName(float) {std::string name = "GetEvalF"; return name;}
template<> std::string GetEvalName(double) {std::string name = "GetEvalD"; return name;}

template<class ElemType>
void Eval<ElemType>::Init(const std::string& /*config*/)
{
    throw std::logic_error("Init shouldn't be called, use constructor");
    // not implemented, calls the underlying class instead
}


// Destroy - cleanup and remove this class
// NOTE: this destroys the object, and it can't be used past this point
template<class ElemType>
void Eval<ElemType>::Destroy()
{
    m_eval->Destroy();
}

// Eval Constructor
template<class ElemType>
void Eval<ElemType>::GetEvalClass(const std::string& config)
{
    typedef void (*GetEvalProc)(IEvaluateModel<ElemType>** peval);

    // initialize just in case
    m_hModule = NULL;
    m_eval = NULL;
    m_dllName = L"CNTKEval";
    // get the name for the dll we want to use, default to CNTKEval.dll
    std::string::size_type found = config.find("evaluator=");
    if (found != std::string::npos)
    {
        std::string::size_type end = config.find_first_of("\n \t", found);
        if (end != std::string::npos)
        {
            m_dllName = msra::strfun::utf16(config.substr(found, end-found));
        }
    }
    m_dllName += L".dll";
    m_hModule = LoadLibrary(m_dllName.c_str());
    if (m_hModule == NULL)
    {
        std::string message = "Eval not found: ";
        message += msra::strfun::utf8(m_dllName);
        throw std::runtime_error(message);
    }

    // create a variable of each type just to call the proper templated version
    ElemType elemType = ElemType();
    GetEvalProc getEvalProc = (GetEvalProc)GetProcAddress(m_hModule, GetEvalName(elemType).c_str());
    getEvalProc(&m_eval);
}

// Eval Constructor
// options - [in] string  of options (i.e. "-windowsize:11 -addenergy") data reader specific 
template<class ElemType>
Eval<ElemType>::Eval(const std::string& config)
{
    GetEvalClass(config);
    m_eval->Init(config);
}


// destructor - cleanup temp files, etc. 
template<class ElemType>
Eval<ElemType>::~Eval()
{
    // free up resources
    if (m_eval != NULL)
    {
        m_eval->Destroy();
        m_eval = NULL;
    }
    if (m_hModule != NULL)
    {
        FreeLibrary(m_hModule);
        m_hModule = NULL;
    }
}

// LoadModel - load a model from the specified path
// modelFileName - file holding the model to load
template<class ElemType>
void Eval<ElemType>::LoadModel(const std::wstring& modelFileName)
{
    m_eval->LoadModel(modelFileName);
}

// GetNodeDimensions - Get the node dimensions of the specified nodes
// dimensions - map from name of node to dimension of the node
// nodeGroup - type of node we are requesting (input/output/specified)
template<class ElemType>
void Eval<ElemType>::GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup)
{
    m_eval->GetNodeDimensions(dimensions, nodeGroup);
}

// Evaluate - Evalute using the model with the given inputs and outputs
// inputs - map from node name to input vector
// outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will happen during evaluation
template<class ElemType>
void Eval<ElemType>::Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs)
{
    m_eval->Evaluate(inputs, outputs);
}

// ResetState - Reset the cell state when we get the start of an utterance
template<class ElemType>
void Eval<ElemType>::ResetState()
{
	m_eval->ResetState();
}

//The explicit instantiation
template class Eval<double>; 
template class Eval<float>;

}}}