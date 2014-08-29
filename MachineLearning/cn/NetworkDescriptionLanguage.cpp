//
// <copyright file="NetworkDescriptionLanguage.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// NetworkDescriptionLanguage.cpp : Code used to interpret the Network Description Language.
//
#include "NetworkDescriptionLanguage.h"
#include "SynchronousExecutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// NOTE: We changed the behavior to require complete match
// EqualInsensitive - check to see if two nodes are equal up to the length of the first string (must be at least half as long as actual node name)
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::wstring& string1, const std::wstring& string2, wchar_t* alternate/*=NULL*/)
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
    else if (EqualInsensitive(nodeType, SparseLearnableParameter<ElemType>::TypeName(), L"SparseParameter"))
        ret = true;  
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
    else if (EqualInsensitive(nodeType, LogNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, CosineNode<ElemType>::TypeName(), L"Cos"))
        ret = true;
    else if (EqualInsensitive(nodeType, SoftmaxNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SquareErrorNode<ElemType>::TypeName(), L"SE"))
        ret = true;
    else if (EqualInsensitive(nodeType, CrossEntropyWithSoftmaxNode<ElemType>::TypeName(), L"CEWithSM"))
        ret = true;
    else if (EqualInsensitive(nodeType, ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName(), L"CBCEWithSM"))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL1RegNode<ElemType>::TypeName(), L"L1Reg"))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL2RegNode<ElemType>::TypeName(), L"L2Reg"))
        ret = true;
    else if (EqualInsensitive(nodeType, PerDimMeanVarNormalizationNode<ElemType>::TypeName(),L"PerDimMVNorm"))
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
NDLScript<float> NDLScript<float>::s_global;
NDLScript<double> NDLScript<double>::s_global;

int NDLNode<float>::s_nameCounter=0;
int NDLNode<double>::s_nameCounter=0;

    template class NDLScript<float>; 
    template class NDLScript<double>;

}}}