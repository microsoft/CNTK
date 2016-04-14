//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NetworkDescriptionLanguage.cpp : Code used to interpret the Network Description Language.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "NetworkDescriptionLanguage.h"
#include "NDLNetworkBuilder.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "ReshapingNodes.h"
#include "SpecialPurposeNodes.h"
#include "TrainingNodes.h"
#include "PreComputeNodes.h"
#include "EvaluationNodes.h"

using namespace std;

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
NDLScript<ElemType>::NDLScript(const NDLScript& copyMe)
    : ConfigParser(copyMe)
{
    m_baseName = copyMe.m_baseName;
    m_scriptString = copyMe.m_scriptString;
    m_macroNode = copyMe.m_macroNode;
    m_noDefinitions = copyMe.m_noDefinitions; // no definitions can be made in this script, interpret all macro/function names as calls
    m_definingMacro = false;                  // not defining when expanding macros (only reason to call this method
    m_cn = copyMe.m_cn;                       // computation network to use for backup symbol lookup. Used for MEL where NDL and network nodes are mixed

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
    m_name        = copyMe.m_name;        // value on the left of the equals
    m_value       = copyMe.m_value;       // value on the right of the equals (CN node name, or value)
    m_parent      = copyMe.m_parent;      // parent script
    m_type        = copyMe.m_type;        // type of node
    m_paramString = copyMe.m_paramString; // parameter of a function/array
    m_paramMacro  = copyMe.m_paramMacro;  // parameter of a macro (the variables used in the macro definition)
    // don't copy over m_parameters, they will be reparsed after the copy

    m_eval = nullptr; // pointer to an arbitrary eval structure
    // script for macro calls, need to expand the macro for each call
    // if it's not expanded the evalValue will be overwitten on multiple calls to a macro
    m_script = (copyMe.m_script) ? new NDLScript<ElemType>(*copyMe.m_script) : nullptr;
}
template <typename ElemType>
NDLScript<ElemType>::NDLScript(const NDLScript&& moveMe)
    : ConfigParser(move(moveMe))
{
    m_baseName      = move(moveMe.m_baseName);
    m_scriptString  = move(moveMe.m_scriptString);
    m_script        = move(moveMe.m_script);        // script lines in parsed node order, macros will have definition followed by body
    m_symbols       = move(moveMe.m_symbols);       // symbol table
    m_macroNode     = move(moveMe.m_macroNode);     // set when interpretting a macro definition
    m_noDefinitions = move(moveMe.m_noDefinitions); // no definitions can be made in this script, interpret all macro/function names as calls
    m_definingMacro = move(moveMe.m_definingMacro);
    m_children      = move(moveMe.m_children);      // child nodes. Note that m_script nodes may not be children of this object, they include macro nodes
    m_cn            = move(moveMe.m_cn);            // computation network to use for backup symbol lookup. Used for MEL where NDL and network nodes are mixed
}

// EqualInsensitive - check to see if two nodes are equal
// string1 - [in,out] string to compare, if comparision is equal insensitive but not sensitive, will replace with sensitive version
// string2 - second string to compare
// alternate - alternate naming of the string
// return - true if strings are equal insensitive and modifies string1 to sensitive version if different
bool EqualInsensitive(std::wstring& string1, const std::wstring& string2, const wchar_t* alternate /*=NULL*/)
{
    bool equal = EqualCI(string1, string2) ||
                 (alternate && EqualCI(string1, alternate));

    if (equal)
        string1 = string2;

    return equal;
}

// ++ operator for this enum, so loops work
NDLPass& operator++(NDLPass& ndlPass)
{
    assert(ndlPass != ndlPassMax);
    ndlPass = static_cast<NDLPass>(ndlPass + 1);
    return ndlPass;
}

// CheckFunction - check to see if we match a function name
// string1 - [in,out] string to compare, if comparision is equal and at least half the full node name will replace with full node name
// allowUndeterminedVariable - [out] set to true if undetermined variables (symbols yet to be defined) are allowed here
// return - true if function name found
bool CheckFunction(std::string& p_nodeType, bool* allowUndeterminedVariable)
{
    if (allowUndeterminedVariable)
        *allowUndeterminedVariable = true; // be default we allow undetermined variables

    wstring nodeType = msra::strfun::utf16(p_nodeType);
    bool ret = false;
         if (EqualInsensitive(nodeType, OperationNameOf(AbsNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(AveragePoolingNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(BatchNormalizationNode))) ret = true;
#ifdef COMING_SOON
    else if (EqualInsensitive(nodeType, OperationNameOf(CRFNode), L"CRF")) ret = true;
#endif
    else if (EqualInsensitive(nodeType, OperationNameOf(ClassBasedCrossEntropyWithSoftmaxNode), L"CBCEWithSM")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ConvolutionNode), L"Convolve")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(PoolingNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(CosDistanceNode), L"CosDist")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(CosDistanceWithNegativeSamplesNode), L"CosWithNegSamples")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(CosineNode), L"Cos")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(CrossEntropyNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(CrossEntropyWithSoftmaxNode), L"CEWithSM")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(DiagTimesNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(DiagonalNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(DropoutNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(DummyCriterionNode), L"DummyCriterion")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ElementTimesNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ErrorPredictionNode), L"ClassificationError")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ExpNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(FutureValueNode))) ret = true;
#ifdef COMING_SOON
    else if (EqualInsensitive(nodeType, OperationNameOf(GMMLogLikelihoodNode), L"GMMLL")) ret = true;
#endif
    else if (EqualInsensitive(nodeType, OperationNameOf(HardmaxNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(InputValue), L"Input")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(InvStdDevNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(KhatriRaoProductNode), L"ColumnwiseCrossProduct")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(LearnableParameter), L"Parameter")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(LogNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(LogSoftmaxNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(LogisticNode), L"Logistic")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(LookupTableNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(MatrixL1RegNode), L"L1Reg")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(MatrixL2RegNode), L"L2Reg")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(MaxPoolingNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(MeanNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(MinusNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(NegateNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(PastValueNode), L"Delay")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(PerDimMeanVarDeNormalizationNode), L"PerDimMVDeNorm")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(PerDimMeanVarNormalizationNode), L"PerDimMVNorm")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(PlusNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ReciprocalNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(RectifiedLinearNode), L"ReLU")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(ReshapeNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(RowRepeatNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(RowStackNode))) ret = true;
#ifdef COMING_SOON
    else if (EqualInsensitive(nodeType, OperationNameOf(SequenceDecoderNode), L"SEWithSM")) ret = true;
#endif
    else if (EqualInsensitive(nodeType, OperationNameOf(SequenceWithSoftmaxNode), L"SEWithSM")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SigmoidNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SinNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SoftmaxNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SparseInputValue), L"SparseInput")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SqrtNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SquareErrorNode), L"SE")) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SumColumnElementsNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(SumElementsNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(TanhNode))) ret = true;
    else if (EqualInsensitive(nodeType, OperationNameOf(TimesNode))) ret = true;
    //else if (EqualInsensitive(nodeType, OperationNameOf(TransposeDimensionsNode))) ret = true; // not supported from NDL, use Transpose()
    else if (EqualInsensitive(nodeType, OperationNameOf(TransposeTimesNode))) ret = true;
    // legacy names:
    else if (EqualInsensitive(nodeType, L"ColumnElementTimes")) ret = true;
    else if (EqualInsensitive(nodeType, L"Constant", L"Const")) ret = true;
    else if (EqualInsensitive(nodeType, L"ImageInput", L"Image")) ret = true;
    else if (EqualInsensitive(nodeType, L"ImageParameter")) ret = true;
    else if (EqualInsensitive(nodeType, L"RowElementTimes")) ret = true;
    else if (EqualInsensitive(nodeType, L"RowSlice")) ret = true;
    else if (EqualInsensitive(nodeType, L"Scale")) ret = true;
    else if (EqualInsensitive(nodeType, L"SparseImageInput", L"SparseImage")) ret = true;
    else if (EqualInsensitive(nodeType, L"Transpose")) ret = true;

    // return the actual node name in the parameter if we found something
    if (ret)
        p_nodeType = msra::strfun::utf8(nodeType);
    return ret;
}

template <typename ElemType>
NDLScript<ElemType> NDLScript<ElemType>::s_global("global");

// declare the static variables from the classes
template <>
NDLScript<float> NDLScript<float>::s_global{};
template <>
NDLScript<double> NDLScript<double>::s_global{};

template <>
int NDLNode<float>::s_nameCounter = 0;
template <>
int NDLNode<double>::s_nameCounter = 0;

template class NDLNode<float>;
template class NDLNode<double>;

template class NDLScript<float>;
template class NDLScript<double>;
} } }
