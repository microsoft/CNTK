//
// <copyright file="SynchronousExecutionEngine.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include "IExecutionEngine.h"
#include "ComputationNetwork.h"
#include "fileutil.h"   // for fexists()

namespace Microsoft { namespace MSR { namespace CNTK {

// SynchronousNodeEvaluator
// Process the Network Description Language into a Computation Network useable
// by SynchronousExecutionEngine.
template <typename ElemType>
class SynchronousNodeEvaluator : public NDLNodeEvaluator<ElemType>
{
public:
    // Constructor - create evaluator
    SynchronousNodeEvaluator(ComputationNetwork<ElemType>& cn) : m_net(cn)
    { }

    // Evaluate - evaluate a node and translate into underlying 
    // node - node we are evaluating
    // baseName - base name for all symbols at this level
    // pass - NDLPass through the evaluation (0-initial, 1-resolve variables, 2-final)
    virtual void Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass)
    {
        // constants don't need to be evaluated, they just translate into numbers...
        if (node->GetType() == ndlTypeConstant 
            || node->GetType() == ndlTypeArray)
            return;

        // setup the node parameters, where they start in the parameter list, and how many there are
        // this is needed for the ndlPassResolve step to hookup all the inputs
        int nodeParamStart = 0;
        int nodeParamCount = 0;

        // get the parameters
        std::vector<NDLNode<ElemType>*> parameter = node->GetParameters();

        // get the name for the symbol to be used by CN nodes
        std::wstring name = msra::strfun::utf16(node->GetName());
        if (!baseName.empty())
        {
            name = baseName + L"." + name;
        }

        std::wstring cnNodeType = msra::strfun::utf16(node->GetValue());

        ComputationNodePtr nodePtr = nullptr;

        // get the node pointer for the node, should be stored in the EvalValue;
        if (pass > ndlPassInitial) 
        {
            nodePtr = (ComputationNodePtr)node->GetEvalValue();
            if (nodePtr == nullptr)
            {
                nodePtr = (ComputationNodePtr)m_net.GetNodeFromName(name);
                node->SetEvalValue(nodePtr);
            }
        }
        
        if (InputValue<ElemType>::TypeName() == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ws should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                // first look for this node already existing in the network
                if (m_net.NodeNameExist(name))
                    nodePtr = m_net.GetNodeFromName(name);
                else
                    nodePtr = m_net.CreateInputNode(name, rows, cols);
            }
        }
        else if (SparseInputValue<ElemType>::TypeName() == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ws should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                // first look for this node already existing in the network
                if (m_net.NodeNameExist(name))
                    nodePtr = m_net.GetNodeFromName(name);
                else
                    nodePtr = m_net.CreateSparseInputNode(name, rows, cols);
            }
        }
        else if (cnNodeType == L"ImageInput")
        {
            if (parameter.size() < 3 || parameter.size() > 4)
                RuntimeError("%ws should have 3 or 4 parameters[imageWidth, imageHeight, imageChannels, [numImages=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;

                nodePtr = m_net.CreateInputNode(name, imageWidth, imageHeight, imageChannels, numImages);
            }
        }
        else if (LearnableParameter<ElemType>::TypeName() == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ws should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "true");

                nodePtr = m_net.CreateLearnableParameter(name, rows, cols);

                nodePtr->NeedGradient() = needGradient;
            }
            else if (pass == ndlPassFinal)
            {
                static int randomSeed = 1;
                std::string initString = node->GetOptionalParameter("init", "uniform");
                ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                ElemType value = node->GetOptionalParameter("value", "0");
                
                msra::strfun::tolower_ascii (initString);
                if (initString == "fixedvalue")
                    nodePtr->FunctionValues().SetValue(value);
                else if (initString == "uniform")
                    m_net.InitLearnableParameters(nodePtr, true, randomSeed++, initValueScale);
                else if (initString == "gaussian")
                    m_net.InitLearnableParameters(nodePtr, false, randomSeed++, initValueScale);
                else if (initString == "fromfile")
                {
                    std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                    if (initFromFilePath == "")
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    if(initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size()-1] == '\"')
                        // remove the opening and closing double quotes
                        initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size()-2);
                    if(!fexists(initFromFilePath))
                        RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                    m_net.InitLearnableParametersFromFile(nodePtr, initFromFilePath);
                }
                else
                    RuntimeError("init must be one of the values of [uniform|gaussian|fixedvalue]");
            }
        }
        else if (SparseLearnableParameter<ElemType>::TypeName() == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ws should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "true");

                nodePtr = m_net.CreateSparseLearnableParameter(name, rows, cols);

                nodePtr->NeedGradient() = needGradient;
            }
            else if (pass == ndlPassFinal)
            {
                static int randomSeed = 1;
                std::string initString = node->GetOptionalParameter("init", "uniform");
                ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                ElemType value = node->GetOptionalParameter("value", "0");
                
                msra::strfun::tolower_ascii(initString);
                if (initString == "fixedvalue")
                    nodePtr->FunctionValues().SetValue(value);
                else if (initString == "uniform")
                    m_net.InitLearnableParameters(nodePtr, true, randomSeed++, initValueScale);
                else if (initString == "gaussian")
                    m_net.InitLearnableParameters(nodePtr, false, randomSeed++, initValueScale);
                else if (initString == "fromfile")
                {
                    std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                    if (initFromFilePath == "")
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    if(initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size()-1] == '\"')
                        // remove the opening and closing double quotes
                        initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size()-2);
                    if(!fexists(initFromFilePath))
                        RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                    m_net.InitLearnableParametersFromFile(nodePtr, initFromFilePath);
                }
                else
                    RuntimeError("init must be one of the values of [uniform|gaussian|fixedvalue]");
            }
        }
        else if (cnNodeType == L"Constant")
        {
            if (parameter.size() != 1)
                RuntimeError("Constant should have 1 fixed parameter [val] and two optional parameters [rows=[1|yourvalue], cols=[1|yourvalue]].");

            if (pass == ndlPassInitial)
            {
                size_t rows = node->GetOptionalParameter("rows", "1");
                size_t cols = node->GetOptionalParameter("cols", "1");

                nodePtr = m_net.CreateLearnableParameter(name, rows, cols);
                nodePtr->NeedGradient() = false;
            }
            else if (pass == ndlPassFinal || nodePtr->FunctionValues().GetNumElements() != 0)
            {
                ElemType val = parameter[0]->GetScalar();
                nodePtr->FunctionValues().SetValue(val);
            }
        }
        else if (cnNodeType == RowSliceNode<ElemType>::TypeName())
        {

            // setup the parameter position of children so we can hook them up later
            nodeParamCount = 1;
            // parameters are (rows, [cols], inputNode)
            nodeParamStart = parameter.size() > 2?2:1;
            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t start_index = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t num_rows = ((NDLNode<ElemType>*)params[1])->GetScalar();

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                nodePtr = m_net.RowSlice(NULL, start_index, num_rows, name);
                nodePtr->NeedGradient() = needGradient;

            }
        }
        else if (cnNodeType == DelayNode<ElemType>::TypeName())
        {
            // setup the parameter position of children so we can hook them up later
            nodeParamCount = 1;
            // parameters are (rows, [cols], delayNode)
            nodeParamStart = parameter.size() > 2?2:1;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                // if we have three parameters the second is columns
                size_t cols = parameter.size() > 2 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                float defaultHiddenActivity = node->GetOptionalParameter("defaultHiddenActivity", "0.1");
                nodePtr = m_net.Delay(NULL, defaultHiddenActivity, rows, cols, name);
                size_t delayTime = node->GetOptionalParameter("delayTime","1");
                ((DelayNode<ElemType>*)nodePtr)->SetDelay(delayTime);

                nodePtr->NeedGradient() = needGradient;
            }
        }    
        else if (cnNodeType == ConvolutionNode<ElemType>::TypeName())
        {
            if (parameter.size() != 7)
                RuntimeError("%ws should have 7 fixed parameters[weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels,horizontalSubsample, verticalSubsample] and two optional parameters [zeroPadding = [false|yourvalue], maxTempMemSizeInSamples = [0|yourvalue]].", cnNodeType.c_str());

            // setup the parameter position of children so we can hook them up later
            nodeParamCount = 2;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                int id = 2; // skip weightNode and inputValueNode

                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size()-id, pass);
                id = 0; // reset counter because the params array starts at zero
                size_t kernelWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t kernelHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t outputChannels = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
            
                assert (id == 5);

                //optional
                bool zeroPadding = node->GetOptionalParameter("zeroPadding", "false");
                size_t maxTempMemSizeInSamples = node->GetOptionalParameter("maxTempMemSizeInSamples", "0");


                nodePtr = m_net.Convolution(NULL, NULL, kernelWidth, kernelHeight, outputChannels,
                    horizontalSubsample, verticalSubsample, zeroPadding, name, maxTempMemSizeInSamples);
            }
        }
        else if (cnNodeType == MaxPoolingNode<ElemType>::TypeName())
        {
            if (parameter.size() != 5)
                RuntimeError("%ws should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

            // setup the parameter position of children so we can hook them up later
            nodeParamCount = 1;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                int id = 1; // skip inputValueNode

                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
                id = 0; // reset counter because the params array starts at zero
                size_t windowWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t windowHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
            
                assert (id == 4);

                nodePtr = m_net.MaxPooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight, 
                            horizontalSubsample, verticalSubsample, name);
            }
        }
        else if (cnNodeType == AveragePoolingNode<ElemType>::TypeName())
        {
            if (parameter.size() != 5)
                RuntimeError("%ws should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

            // setup the parameter position of children so we can hook them up later
            nodeParamCount = 1;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                int id = 1; // skip inputValueNode

                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
                id = 0; // reset counter because the params array starts at zero
                size_t windowWidth = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t windowHeight = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t horizontalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();
                size_t verticalSubsample = ((NDLNode<ElemType>*)params[id++])->GetScalar();

                assert (id == 4);

                nodePtr = m_net.AveragePooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight, 
                            horizontalSubsample, verticalSubsample, name);
            }
        }
        else
        {

            // setup the variables for node parameter processing
            nodeParamCount = parameter.size(); // all parameters are nodes in standard nodes
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                nodePtr = m_net.CreateComputationNode(node->GetValue(), name);
            }
        }

        switch (pass)
        {
        case ndlPassInitial:
            node->SetEvalValue(nodePtr);
            // evaluate parameters
            EvaluateParameters(node, baseName, nodeParamStart, nodeParamCount, pass);
            break;
        case ndlPassResolve:
            {
            std::vector<void*> inputs = EvaluateParameters(node, baseName, nodeParamStart, nodeParamCount, pass);

            switch (inputs.size())
            {
            case 1:
                nodePtr->AttachInputs(ComputationNodePtr(inputs[0]));
                break;
            case 2:
                nodePtr->AttachInputs(ComputationNodePtr(inputs[0]), ComputationNodePtr(inputs[1]));
                break;
            case 3:
                nodePtr->AttachInputs(ComputationNodePtr(inputs[0]), ComputationNodePtr(inputs[1]), ComputationNodePtr(inputs[2]));
                break;
            case 4:
                nodePtr->AttachInputs(ComputationNodePtr(inputs[0]), ComputationNodePtr(inputs[1]), ComputationNodePtr(inputs[2]), ComputationNodePtr(inputs[3]));
                break;
            default:
                if (nodeParamCount > 0)
                    RuntimeError("Invalid number of parameters name = '%s' call = '%s'\n", node->GetName().c_str(), node->GetValue().c_str());
                break;
            }

            // process common optional parameters (like "tag");
            ProcessOptionalParameters(node);
            break;
            }
        case ndlPassFinal:
            break;
        }
    }

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
                // We then use this fully-qualified name to look up this node in the model (using "m_net.GetNodeFromName").

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
            if (m_net.NodeNameExist(wname))
            {
                void* np = (void*)m_net.GetNodeFromName(wname);
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
                    if (m_net.NodeNameExist(wname))
                    {
                        void* np = (void*)m_net.GetNodeFromName(wname);
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
            throw logic_error("EvaluateParmeters: nodeParamters specified that do not exist");
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
        ComputationNode<ElemType>* compNode = (ComputationNode<ElemType>*)node->GetEvalValue();
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
                SetOutputNode(m_net.FeatureNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "label"))
            {
                SetOutputNode(m_net.LabelNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "criteria"))
            {
                SetOutputNode(m_net.FinalCriterionNodes(), compNode);
            }
            else if (!_strnicmp(value.c_str(), "eval", 4)) // only compare the first 4 characters
            {
                SetOutputNode(m_net.EvaluationNodes(), compNode);
            }
            else if (!_stricmp(value.c_str(), "output"))
            {
                SetOutputNode(m_net.OutputNodes(), compNode);
            }
        }

    }

    // SetOutputNode - Set the output node, checks to see if it already exists first
    // nodeGroup - group vector to add to
    // compNode - computation node to add
    void SetOutputNode(std::vector<ComputationNode<ElemType>*>& nodeGroup, ComputationNode<ElemType>* compNode)
    {
        for (ComputationNodePtr node : nodeGroup)
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
        if (m_net.NodeNameExist(symbol))
            return m_net.GetNodeFromName(symbol);
        return NULL;
    }

    virtual ~SynchronousNodeEvaluator()
    {
    }

private:
    ComputationNetwork<ElemType>& m_net;
    typedef ComputationNode<ElemType>* ComputationNodePtr;
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
        m_computationNetwork = new ComputationNetwork<ElemType>(deviceId);
        m_computationNetwork->SetRandomSeedOffset(randomSeedOffset);
        m_ownNetwork = true;
        m_nodeEvaluator = new SynchronousNodeEvaluator<ElemType>(*m_computationNetwork);
    }

    SynchronousExecutionEngine(ComputationNetwork<ElemType>* computationNetwork)
    {
        m_computationNetwork = computationNetwork;
        m_ownNetwork = false;
        m_nodeEvaluator = new SynchronousNodeEvaluator<ElemType>(*m_computationNetwork);
    }

    virtual ~SynchronousExecutionEngine()
    { 
        if (m_ownNetwork)
            delete m_computationNetwork;
        delete m_nodeEvaluator;
    }

    ComputationNetwork<ElemType>& GetComputationNetwork()
    {
        return *m_computationNetwork;
    }

    NDLNodeEvaluator<ElemType>& GetNodeEvaluator()
    {
        return *m_nodeEvaluator;
    }

private:
    bool m_ownNetwork;
    ComputationNetwork<ElemType>* m_computationNetwork;
    SynchronousNodeEvaluator<ElemType>* m_nodeEvaluator;
protected:
    // Copy constructor, should never be called.
    SynchronousExecutionEngine(const SynchronousExecutionEngine<ElemType>& /*deepCopyFrom*/) 
    {         
        throw std::logic_error("'SynchronousExecutionEngine(const SynchronousExecutionEngine<ElemType>& deepCopyFrom)' should never be called.");
    } 

    // Assignment operator, should never be called.
    SynchronousExecutionEngine<ElemType>& operator=(const SynchronousExecutionEngine<ElemType>& /*deepCopyFrom*/) 
    {            
        throw std::logic_error("'SynchronousExecutionEngine<ElemType>& operator=(const SynchronousExecutionEngine<ElemType>& deepCopyFrom)' should never be called.");
    } 
};

template class SynchronousExecutionEngine<float>; 
template class SynchronousExecutionEngine<double>;

}}}