//
// <copyright file="NetworkDescriptionLanguage.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// NetworkDescriptionLanguage.cpp : Code used to interpret the Network Description Language.
//
#include "NetworkDescriptionLanguage.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// EqualInsensitive - check to see if two nodes are equal up to the length of the first string (must be at least half as long as actual node name)
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::wstring& string1, const std::wstring& string2)
{
    bool equal = !_wcsnicmp(string1.c_str(), string2.c_str(), string1.size());

    // don't allow partial matches that are less than half the string
    if (equal && string1.size() < string2.size()/2)
    {
        equal = false;
    }

    // if we have a (partial) match replace with the full name
    if (equal && wcscmp(string1.c_str(), string2.c_str()))
    {
        string1 = string2;
    }
    return equal;
}

// CheckFunction - check to see if we match a function name
// string1 - [in,out] string to compare, if comparision is equal and at least half the full node name will replace with full node name
// return - true if function name found
template <typename ElemType>
bool CheckFunction(std::string& p_nodeType)
{
    std::wstring nodeType = msra::strfun::utf16(p_nodeType);
    bool ret = false;
    if (EqualInsensitive(nodeType, NegateNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, RectifiedLinearNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SigmoidNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, TanhNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, LogNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SoftmaxNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SumNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, ScaleNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, TimesNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, PlusNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MinusNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, SquareErrorNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, CrossEntropyWithSoftmaxNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL1RegNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MatrixL2RegNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, PerDimMeanVarNormalizationNode<ElemType>::TypeName()))
        ret = true;            
    else if (EqualInsensitive(nodeType, ErrorPredictionNode<ElemType>::TypeName()))
        ret = true;    
    else if (EqualInsensitive(nodeType, DropoutNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MeanNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, InvStdDevNode<ElemType>::TypeName()))
        ret = true;
    else if (EqualInsensitive(nodeType, MaxPoolingNode<ElemType>::TypeName()))
        ret = true;   
    return ret;
}

}}}
