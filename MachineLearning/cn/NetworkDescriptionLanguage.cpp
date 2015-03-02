//
// <copyright file="NetworkDescriptionLanguage.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// NetworkDescriptionLanguage.cpp : Code used to interpret the Network Description Language.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "NetworkDescriptionLanguage.h"
#include "SynchronousExecutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// DuplicateNode - Duplicate a node in a macro as needed (it might already exist)
// node - node we are duplicating
// return - the new duplicated node if it didn't exist, or the previously duplicated node if it already did
template <typename ElemType>
NDLNode<ElemType>* NDLScript<ElemType>::DuplicateNode(NDLNode<ElemType>* node)
{
    NDLNode<ElemType>* newNode = node->Copy();
    m_children.push_back(newNode);
    newNode->SetParentScript(this);
    return newNode;
}

template <typename ElemType>
NDLScript<ElemType>::NDLScript(const NDLScript& copyMe) : ConfigParser(copyMe)
{
    m_baseName = copyMe.m_baseName;
    m_scriptString = copyMe.m_scriptString;
    m_macroNode = copyMe.m_macroNode;
    m_noDefinitions = copyMe.m_noDefinitions; // no definitions can be made in this script, interpret all macro/function names as calls
    m_definingMacro = false; // not defining when expanding macros (only reason to call this method
    m_cn = copyMe.m_cn; // computation network to use for backup symbol lookup. Used for MEL where NDL and network nodes are mixed

    // script lines in parsed node order
    for (NDLNode<ElemType>* node : copyMe.m_script)
    {
        // duplicate this node
        NDLNode<ElemType>* newNode = DuplicateNode(node);
        AddSymbol(newNode->GetName(), newNode);

        // now get the parameters to the functions added
        ConfigValue value = newNode->GetParamString();
        ParseParameters(newNode, value, true /*createNew*/);

        // add it to the new script
        m_script.push_back(newNode);
    }

    // now search the symbol table for other symbols that haven't been copied yet
    // this happens for constants defined in macros and such
    for (std::pair<std::string, NDLNode<ElemType>*> pair : copyMe.m_symbols)
    {
        // if we can't find the symbol in the copied symbol table, copy it here
        if (m_symbols.find(pair.first) == end(m_symbols))
        {
            // duplicate this node
            NDLNode<ElemType>* newNode = DuplicateNode(pair.second);
            AddSymbol(pair.first, newNode);
            // anything that takes parameters should be evaluated in the script loop
            assert(newNode->GetParamString().empty());
        }
    }
    // NOTE: the child nodes get populated as the nodes are duplicated in the loop above
    // we shouldn't try to duplicate them separately
}

// copy constructor, creates a new disconnected copy of this node
// doesn't copy everything, so use for macro expansion only (it's private)
// copyMe - node to copy
template <typename ElemType>
NDLNode<ElemType>::NDLNode(const NDLNode<ElemType>& copyMe)
{
    m_name = copyMe.m_name; // value on the left of the equals
    m_value = copyMe.m_value; // value on the right of the equals (CN node name, or value)
    m_parent = copyMe.m_parent; // parent script
    m_type = copyMe.m_type; //type of node
    m_paramString = copyMe.m_paramString; // parameter of a function/array
    m_paramMacro = copyMe.m_paramMacro; // parameter of a macro (the variables used in the macro definition)
    // don't copy over the parameters, they will be reparsed after the copy
    //m_parameters = copyMe.m_parameters; // copy over the parameters straight

    m_eval = nullptr; // pointer to an arbitrary eval structure
    // script for macro calls, need to expand the macro for each call
    // if it's not expanded the evalValue will be overwitten on multiple calls to a macro
    m_script = (copyMe.m_script) ? new NDLScript<ElemType>(*copyMe.m_script) : nullptr;
}
template <typename ElemType>
NDLScript<ElemType>::NDLScript(const NDLScript&& moveMe) : ConfigParser(move(moveMe))
{
    m_baseName = move(moveMe.m_baseName);
    m_scriptString = move(moveMe.m_scriptString);
    m_script = move(moveMe.m_script); // script lines in parsed node order, macros will have definition followed by body
    m_symbols = move(moveMe.m_symbols); // symbol table
    m_macroNode = move(moveMe.m_macroNode); // set when interpretting a macro definition
    m_noDefinitions = move(moveMe.m_noDefinitions); // no definitions can be made in this script, interpret all macro/function names as calls
    m_definingMacro = move(moveMe.m_definingMacro);
    m_children = move(moveMe.m_children); // child nodes. Note that m_script nodes may not be children of this object, they include macro nodes
    m_cn = move(moveMe.m_cn); // computation network to use for backup symbol lookup. Used for MEL where NDL and network nodes are mixed
}

// EqualInsensitive - check to see if two nodes are equal 
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::wstring& string1, const std::wstring& string2, const wchar_t* alternate/*=NULL*/)
{
    bool equal = !_wcsnicmp(string1.c_str(), string2.c_str(), string1.size()) && string1.size()==string2.size();

    if (!equal && alternate != NULL)
        equal = !_wcsnicmp(string1.c_str(), alternate, string1.size()) && string1.size()==wcslen(alternate);

    if (equal)
        string1 = string2;

    return equal;
}

// ++ operator for this enum, so loops work
NDLPass &operator++(NDLPass &ndlPass) {
  assert(ndlPass != ndlPassMax);
  ndlPass = static_cast<NDLPass>(ndlPass + 1);
  return ndlPass;
}

// CheckFunction - check to see if we match a function name
// string1 - [in,out] string to compare, if comparision is equal and at least half the full node name will replace with full node name
// allowUndeterminedVariable - [out] set to true if undetermined variables (symbols yet to be defined) are allowed here
// return - true if function name found
template <typename ElemType>
bool CheckFunction(std::string& p_nodeType, bool* allowUndeterminedVariable)
{
    std::wstring nodeType = msra::strfun::utf16(p_nodeType);
    bool ret = false;
    if (allowUndeterminedVariable)
        *allowUndeterminedVariable = true; // be default we allow undetermined variables
    if (EqualInsensitive(nodeType, InputValue<ElemType>::TypeName(), L"Input"))
        ret = true;   
    else if (EqualInsensitive(nodeType, SparseInputValue<ElemType>::TypeName(), L"SparseInput"))
        ret = true; 
    else if (EqualInsensitive(nodeType, LearnableParameter<ElemType>::TypeName(), L"Parameter"))
        ret = true;   
    //else if (EqualInsensitive(nodeType, SparseLearnableParameter<ElemType>::TypeName(), L"SparseParameter"))
    //    ret = true;  
    else if (EqualInsensitive(nodeType, L"Constant", L"Const"))
        ret = true;   
    else if (EqualInsensitive(nodeType, L"ImageInput", L"Image"))
        ret = true;   
    else if (EqualInsensitive(nodeType, SumElementsNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ScaleNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, TimesNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ElementTimesNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, DiagTimesNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, CosDistanceNode<ElemType>::TypeName(), L"CosDist"))
        ret = true;
    else if (EqualInsensitive(nodeType, KhatriRaoProductNode<ElemType>::TypeName(), L"ColumnwiseCrossProduct"))
        ret = true;
    else if (EqualInsensitive(nodeType, PlusNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MinusNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, NegateNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, RectifiedLinearNode<ElemType>::TypeName(), L"ReLU"))
        ret = true;
    else if (EqualInsensitive(nodeType, SigmoidNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, TanhNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ExpNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, LogNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, CosineNode<ElemType>::TypeName(), L"Cos"))
        ret = true;
    else if (EqualInsensitive(nodeType, SoftmaxNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, LogSoftmaxNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SquareErrorNode<ElemType>::TypeName(), L"SE"))
        ret = true;
    else if (EqualInsensitive(nodeType, CrossEntropyWithSoftmaxNode<ElemType>::TypeName(), L"CEWithSM"))
        ret = true;
    else if (EqualInsensitive(nodeType, CrossEntropyNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName(), L"CBCEWithSM"))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL1RegNode<ElemType>::TypeName(), L"L1Reg"))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL2RegNode<ElemType>::TypeName(), L"L2Reg"))
        ret = true;
    else if (EqualInsensitive(nodeType, PerDimMeanVarNormalizationNode<ElemType>::TypeName(),L"PerDimMVNorm"))
        ret = true;            
    else if (EqualInsensitive(nodeType, PerDimMeanVarDeNormalizationNode<ElemType>::TypeName(),L"PerDimMVDeNorm"))
        ret = true;            
    else if (EqualInsensitive(nodeType, ErrorPredictionNode<ElemType>::TypeName(), L"ClassificationError"))
        ret = true;    
    else if (EqualInsensitive(nodeType, DropoutNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MeanNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, InvStdDevNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ConvolutionNode<ElemType>::TypeName(), L"Convolve"))
        ret = true;   
    else if (EqualInsensitive(nodeType, MaxPoolingNode<ElemType>::TypeName()))
        ret = true;   
    else if (EqualInsensitive(nodeType, AveragePoolingNode<ElemType>::TypeName()))
        ret = true;   
    else if (EqualInsensitive(nodeType, DelayNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, RowSliceNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, LookupTableNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, GMMLogLikelihoodNode<ElemType>::TypeName(), L"GMMLL"))
        ret = true;
    else if (EqualInsensitive(nodeType, CosDistanceWithNegativeSamplesNode<ElemType>::TypeName(), L"CosWithNegSamples"))
        ret = true;

    // return the actual node name in the parameter if we found something
    if (ret)
    {
        p_nodeType = msra::strfun::utf8(nodeType);
    }
    return ret;
}

template <typename ElemType>
NDLScript<ElemType> NDLScript<ElemType>::s_global("global");

// declare the static variables from the classes
template<> NDLScript<float> NDLScript<float>::s_global{};
template<> NDLScript<double> NDLScript<double>::s_global{};

template<> int NDLNode<float>::s_nameCounter = 0;
template<> int NDLNode<double>::s_nameCounter = 0;

template class NDLNode<float>;
template class NDLNode<double>;

template class NDLScript<float>;
template class NDLScript<double>;

}}}
