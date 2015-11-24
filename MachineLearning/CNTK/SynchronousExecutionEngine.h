//
// <copyright file="SynchronousExecutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "IExecutionEngine.h"
#include "ComputationNetwork.h"
#include "ComputationNetworkBuilder.h"
#include "fileutil.h"   // for fexists()

namespace Microsoft { namespace MSR { namespace CNTK {

// SynchronousNodeEvaluator
// Process the Network Description Language into a Computation Network useable
// by SynchronousExecutionEngine.
template <typename ElemType>
class SynchronousNodeEvaluator : public NDLNodeEvaluator<ElemType>
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
public:
    // Constructor - create evaluator
    SynchronousNodeEvaluator(ComputationNetworkPtr cn) : m_net(cn)
    { }

    // Evaluate - evaluate a node and translate into underlying 
    // node - node we are evaluating
    // baseName - base name for all symbols at this level
    // pass - NDLPass through the evaluation (0-initial, 1-resolve variables, 2-final)
    virtual void Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass);

#ifdef LATER
    // EvaluateDotName - Evaluate a dot name and resolve to target node
    // node - NDLNode of the script
    // nodeParam - NDLNode parameter we are evaluating
    // baseName - name of the base node
    // pass - which pass through the NDL nodes
    // returns: the node that is the evaluated parameter
    virtual NDLNode<ElemType>* EvaluateDotName(NDLNode<ElemType>* node, NDLNode<ElemType>* nodeParam, const std::wstring& baseNameP, const NDLPass pass)

    {
        if (pass > ndlPassInitial && evaluateNode)
        {
            std::string name = nodeParam->GetName();
            std::wstring wname = msra::strfun::utf16(name);
            if (nodeParam->GetType() == ndlTypeDotParameter)
            {
                // When we see a variable of the form "A.B" in a macro, we need to resolve it to an actual node, by first constructing it's
                // fully-qualified name. There are 2 possibilities: 
                // 1) "A" was defined locally within the macro.  In this case, we must find the fully-qualified name of the node that this macro
                //    call is being assigned to (eg, "C" in the example "C=Macro(X)"), and concatenate it's name with "A.B" (eg, "C.A.B").
                // 2) "A" was passed in as a parameter to a macro.  In this case, we must find the fully-qualified name of the node that
                //    was passed in as "A", and replace the "A" and "A.B" with this name.

                // Consider the following example:
                // NdlBLob=[
                //      P=MacroCall1(...)
                //      C=MacroCall2(P) 
                // ]
                // # MacroDefinition
                // MacroCall2(X)
                // { 
                //      A=MacroCall3(...)
                //      D=Times(A.B,X.B)}
                // }
                // 

                // In this example, in the call D=Times(A.B,X.B), we need to resolve A.B and X.B appropriately.
                // Specifically, "A.B" must be resolved to the fully qualified name "C.A.B", whereas "X.B" must be resolved to the fully qualified name "P.B".
                // We then use this fully-qualified name to look up this node in the model (using "m_net->GetNodeFromName").

                std::size_t firstDotPos = name.find_first_of(".");
                if (firstDotPos == std::string::npos)
                {
                    LogicError("nodeParam of type \"ndlTypeDotParameter\" doesn't have a dot in its name: %s", name.c_str());
                }

                std::string nameBeforeDot = name.substr(0, firstDotPos);
                std::string nameAfterDot = name.substr(firstDotPos + 1, name.size() - (firstDotPos + 1));

                // look up if "nameBeforeDot" was a parameter to the macro.
                NDLNode<ElemType>* resolvedParam = nodeParam->GetParentScript()->FindSymbol(nameBeforeDot);
                if (resolvedParam != nullptr && resolvedParam->GetType() == ndlTypeMacroCall)
                {
                    // if "nameBeforeDot" was a parameter to the macro, builds it's fully qualified name by
                    // replacing "nameBeforeDot" with the fully qualified name of the node passed in as the parameter.
                    NDLScript<ElemType>* parentScript = resolvedParam->GetParentScript();
                    baseName = parentScript->GetBaseName();
                    std::wstring resolvedParamName = msra::strfun::utf16(resolvedParam->GetName());
                    wname = baseName.empty() ?
                        resolvedParamName + L"." + msra::strfun::utf16(nameAfterDot) :
                        baseName + L"." + resolvedParamName + L"." + msra::strfun::utf16(nameAfterDot);
                }
                else if (!baseName.empty())
                {
                    // else, "nameBeforeDot" wasn't a parameter to the macro, so treat it as a local variable.
                    wname = baseName + L"." + wname;
                }
            }
            else if (!baseName.empty())
            {
                wname = baseName + L"." + wname;
            }

            // fully qualified names can be looked up in the model
            if (m_net->NodeNameExist(wname))
            {
                void* np = (void*)m_net->GetNodeFromName(wname);
                nodeParam->SetEvalValue(np);
            }
            // NOTE: there is a bug here, we allow an abbreviated node reference (i.e. L1.BFF) based on return values in NDL 
            // when the actual full node reference that the computational network uses would be L1.BFF.FF.P, so that is what CN sees
            // can we do the normal find symbol here to allow abbreviated node references?

            // if we still didn't get a value, throw an error
            if (nodeParam->GetEvalValue() == nullptr)
            {
                LogicError("Dot name could not be resolved '%s': should have a node named '%ls' in computational network\n", nodeParam->GetName().c_str(), name.c_str());
            }
        }
        return nodeParam;
    }
#endif

    // EvaluateParameter - Evaluate a parameter of a call
    // node - NDLNode of the script
    // nodeParam - NDLNode parameter we are evaluating
    // baseName - name of the base node
    // pass - which pass through the NDL nodes
    // returns: the node that is the evaluated parameter
    virtual NDLNode<ElemType>* EvaluateParameter(NDLNode<ElemType>* node, NDLNode<ElemType>* nodeParam, const std::wstring& baseNameP, const NDLPass pass )
    {
        // get the parent script that includes the symbol table we are interested in
        NDLScript<ElemType>* script = node->GetParentScript();
        wstring baseName = baseNameP;
        if (script == NULL)
        {
            std::wstring name = baseName + L"." + msra::strfun::utf16(node->GetName());
            LogicError("no script for a parameter node in call to %ls\n", name.c_str());
        }

        // evaluate the parameter if we haven't yet, or if we are in the resolve pass (need to set the inputs)
        bool evaluateNode = nodeParam->GetEvalValue() == NULL || pass == ndlPassResolve;
        switch (nodeParam->GetType())
        {
        // if the node is a parameter then look it up in the symbol table
        case ndlTypeUndetermined: // an undetermined parameter needs to be looked up again in the symbol table
        case ndlTypeParameter:
        {
            // lookup the parameter
            NDLNode<ElemType>* nodeResolve = script->FindSymbol(nodeParam->GetName());

            // if we have resolved the name, no need to continue evaluation
            if (!(pass == ndlPassResolve && nodeResolve && nodeParam->GetEvalValue() == nullptr))
            {
                break;
            }
            if (pass > ndlPassInitial && evaluateNode && nodeResolve)
            {
                std::string name = nodeResolve->GetName();
                // we need to start from the parent script, because that is the namespace of the parameter being passed in
                NDLScript<ElemType>* parentScript = nodeResolve->GetParentScript();
                nodeResolve = parentScript->FindSymbol(name);

                // if we still didn't get a value
                if (nodeResolve == nullptr || nodeResolve->GetEvalValue() == nullptr)
                {
                    // check for the fully quantified name in the computation network
                    // this is needed for MEL processing, since CN nodes names can be used as parameters in MEL
                    std::wstring wname = msra::strfun::utf16(name);
                    if (m_net->NodeNameExist(wname))
                    {
                        void* np = (void*)m_net->GetNodeFromName(wname).get();
                        // if we don't have a resolve node, it's because the name didn't exist in NDL
                        if (!nodeResolve)
                            nodeResolve = nodeParam;
                        nodeResolve->SetEvalValue(np);
                    }
                    else
                    {
                        RuntimeError("Parameter name could not be resolved '%s'\n", name.c_str());
                    }
                }
            }
            nodeParam = nodeResolve;
            break;
        }
        case ndlTypeFunction:
            if (evaluateNode)
                Evaluate(nodeParam, baseName, pass);
            break;
        case ndlTypeMacroCall:
            if (evaluateNode)
                nodeParam->EvaluateMacro(*this, baseName, pass);
            break;
        // constants and variables are good as is
        case ndlTypeConstant:
        case ndlTypeVariable:
                break;
        // everything else is illegal as a parameter
        default:
            {
                std::wstring name = baseName + L"." + msra::strfun::utf16(node->GetName());
                RuntimeError("Invalid parameter (macro definitions and arrays not allowed), see call to %ls\n", name.c_str());
            }
            break;
        }
        return nodeParam;
    }

    // EvaluateParameters - Evaluate the parameters of a call
    // node - NDLNode we are evaluating paramters for
    // baseName - baseName for the current node
    // nodeParamStart - starting parameter that contains a node
    // nodeParamCount - ending parameter that contains a node
    // pass - NDL pass we are evaluating
    // returns: vector of eval pointers, which are ComputationNodePtr for CNEvaluator
    virtual std::vector<void*> EvaluateParameters(NDLNode<ElemType>* node, const wstring& baseName, int nodeParamStart, int nodeParamCount, const NDLPass pass)
    {
        std::vector<void*> inputs;
        std::vector<NDLNode<ElemType>*> parameter = node->GetParameters();
        ConfigArray paramString = node->GetParamString();

        if (parameter.size() < 1)
        {
            return inputs;
        }
        if (nodeParamStart + nodeParamCount > parameter.size())
            LogicError("EvaluateParmeters: nodeParamters specified that do not exist");
        size_t numChildren = nodeParamCount;
        for (size_t i=0; i < numChildren;++i)
        {
            int index = i+nodeParamStart;
            NDLNode<ElemType>* nodeParam = parameter[index];
            std::wstring paramS = paramString[index];

            // default base is same as current
            std::wstring baseSymbol = baseName;

            NDLNode<ElemType>* nodeResult = EvaluateParameter(node, nodeParam, baseSymbol, pass);
            // look for a prefix here and set baseName appropriately

            if (pass == ndlPassResolve)
            {
                void* np = nodeResult->GetEvalValue();
                assert(np != nullptr);
                inputs.push_back((void*)np);
            }
            else if (pass == ndlPassInitial) // for initial pass we are only interested in resolved nodes (to get constant values)
            {
                inputs.push_back((void*)nodeResult);
            }
            // NOTE: in final pass inputs are always NULL
        }

        // now return the vector
        return inputs;
    }

    // ProcessOptionalParameters - Process the optional parameters of a node
    virtual void ProcessOptionalParameters(NDLNode<ElemType>* node)
    {
        vector<NDLNode<ElemType>*> params = node->GetParameters(true); // get all the optional parameters only
        auto compNode = ComputationNode<ElemType>::FromVoidPtr(node->GetEvalValue());
        std::string empty;

        // loop through all the optional parameters processing them as necessary
        for (NDLNode<ElemType>* param : params)
        {
            // make sure it's a "tag" optional parameter, that's all we process currently
            if (_stricmp(param->GetName().c_str(), "tag"))
                continue;

            std::string value = param->GetValue();
            if (!_stricmp(value.c_str(), "feature"))
            {
                SetOutputNode(m_net->FeatureNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "label"))
            {
                SetOutputNode(m_net->LabelNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "criteria"))
            {
                SetOutputNode(m_net->FinalCriterionNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "multiseq"))
            {
                fprintf(stderr, "'multiseq' tag is defunct.\n");
            }
            else if (!_strnicmp(value.c_str(), "eval", 4)) // only compare the first 4 characters
            {
                SetOutputNode(m_net->EvaluationNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "output"))
            {
                SetOutputNode(m_net->OutputNodes(), compNode);
            }
        }

    }

    // SetOutputNode - Set the output node, checks to see if it already exists first
    // nodeGroup - group vector to add to
    // compNode - computation node to add
    // TODO: It seems that this is also applied to other tyoes of nodes, so the name of this function is wrong.
    static void SetOutputNode(std::vector<ComputationNodeBasePtr> & nodeGroup, ComputationNodePtr compNode)
    {
        for (const auto & node : nodeGroup)
        {
            if (node == compNode)
                return;
        }
        nodeGroup.push_back(compNode);
    }

    // FindSymbol - Search the nodes for a fully quantified symbol
    // symbol - name of the symbol fully quantified name with "dots"
    // returns - pointer to the matching EvalValue for that node, of NULL if not found
    virtual void* FindSymbol(const wstring& symbol)
    {
        if (m_net->NodeNameExist(symbol))
            return m_net->GetNodeFromName(symbol).get();
        return nullptr;
    }

    virtual ~SynchronousNodeEvaluator()
    {
    }

private:
    ComputationNetworkPtr m_net;
    void operator=(const SynchronousNodeEvaluator&);
};

// SynchronousExecutionEngine
// TODO JC Refactor eligible methods and members into abstract base class.
template <typename ElemType>
class SynchronousExecutionEngine : public IExecutionEngine<ElemType>
{
public:
    SynchronousExecutionEngine(DEVICEID_TYPE deviceId=AUTOPLACEMATRIX, unsigned long randomSeedOffset=0)
    {
        m_computationNetwork = make_shared<ComputationNetwork>(deviceId);
        m_computationNetwork->SetRandomSeedOffset(randomSeedOffset);
        m_nodeEvaluator = new SynchronousNodeEvaluator<ElemType>(m_computationNetwork);
    }

    SynchronousExecutionEngine(ComputationNetworkPtr computationNetwork)
    {
        m_computationNetwork = computationNetwork;
        m_nodeEvaluator = new SynchronousNodeEvaluator<ElemType>(m_computationNetwork);
    }

    virtual ~SynchronousExecutionEngine()
    { 
        delete m_nodeEvaluator;
    }

    ComputationNetworkPtr GetComputationNetwork()
    {
        return m_computationNetwork;
    }

    NDLNodeEvaluator<ElemType>& GetNodeEvaluator()
    {
        return *m_nodeEvaluator;
    }

private:
    ComputationNetworkPtr m_computationNetwork;
    SynchronousNodeEvaluator<ElemType>* m_nodeEvaluator;
protected:
    // Copy constructor, should never be called.
    SynchronousExecutionEngine(const SynchronousExecutionEngine<ElemType>& /*deepCopyFrom*/) 
    {         
        LogicError("'SynchronousExecutionEngine(const SynchronousExecutionEngine<ElemType>& deepCopyFrom)' should never be called.");
    } 

    // Assignment operator, should never be called.
    SynchronousExecutionEngine<ElemType>& operator=(const SynchronousExecutionEngine<ElemType>& /*deepCopyFrom*/) 
    {            
        LogicError("'SynchronousExecutionEngine<ElemType>& operator=(const SynchronousExecutionEngine<ElemType>& deepCopyFrom)' should never be called.");
    } 
};

}}}
