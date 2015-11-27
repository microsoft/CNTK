//
// <copyright file="SynchronousExecutionEngine.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// Note: Despite its name, this file is really about parsing NDL into an actual ComputationNetwork.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "SynchronousExecutionEngine.h"
#include "LinearAlgebraNodes.h"
#include "RecurrentNodes.h"
#include "ConvolutionalNodes.h"
#include "NonlinearityNodes.h"
#include "ReshapingNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    void SynchronousNodeEvaluator<ElemType>::Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass)
    {
        ComputationNetworkBuilder<ElemType> builder(*m_net);

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

        ComputationNodePtr nodePtr;

        // get the node pointer for the node, should be stored in the EvalValue;
        if (pass > ndlPassInitial) 
        {
            nodePtr = ComputationNode<ElemType>::FromVoidPtr(node->GetEvalValue());
            if (!nodePtr)
            {
                nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->GetNodeFromName(name));
                node->SetEvalValue(nodePtr.get());
            }
        }
        
        if (OperationNameOf(InputValue) == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                // first look for this node already existing in the network
                if (m_net->NodeNameExist(name))
                    nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->GetNodeFromName(name));
                else
                    nodePtr = builder.CreateInputNode(name, rows, cols);
            }
        }
        else if (OperationNameOf(SparseInputValue) == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                // first look for this node already existing in the network
                if (m_net->NodeNameExist(name))
                    nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->GetNodeFromName(name));
                else
                    nodePtr = builder.CreateSparseInputNode(name, rows, cols);
            }
        }
        else if (cnNodeType == L"ImageInput")
        {
            if (parameter.size() < 3 || parameter.size() > 4)
                RuntimeError("%ls should have 3 or 4 parameters[imageWidth, imageHeight, imageChannels, [numImages=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;

                nodePtr = builder.CreateInputNode(name, ImageLayoutWHC(imageWidth, imageHeight, imageChannels), numImages);
            }
        }
        else if (cnNodeType == L"SparseImageInput")
        {
            if (parameter.size() < 3 || parameter.size() > 4)
                RuntimeError("%ls should have 3 or 4 parameters[imageWidth, imageHeight, imageChannels, [numImages=1]].", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t imageWidth = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t imageHeight = ((NDLNode<ElemType>*)params[1])->GetScalar();
                size_t imageChannels = ((NDLNode<ElemType>*)params[2])->GetScalar();
                size_t numImages = parameter.size() > 3 ? ((NDLNode<ElemType>*)params[3])->GetScalar() : 1;

                nodePtr = builder.CreateSparseInputNode(name, ImageLayoutWHC(imageWidth, imageHeight, imageChannels), numImages);
            }
        }
        else if (OperationNameOf(LearnableParameter) == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "true");

                nodePtr = builder.CreateLearnableParameter(name, rows, cols);

                nodePtr->SetParameterUpdateRequired(needGradient);
            }
            else if (pass == ndlPassFinal)
            {
                static int randomSeed = 1;
                wstring initString = node->GetOptionalParameter("init", "uniform");
                ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                ElemType value = node->GetOptionalParameter("value", "0");
                bool initOnCPUOnly = node->GetOptionalParameter("initOnCPUOnly", "false");
                int forcedRandomSeed = node->GetOptionalParameter("randomSeed", "-1"/*disabled*/);

                if (!_wcsicmp(initString.c_str(), L"fixedValue"))
                    nodePtr->FunctionValues().SetValue(value);
                else if (!_wcsicmp(initString.c_str(), L"uniform"))
                    m_net->InitLearnableParameters(nodePtr, true, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, initValueScale, initOnCPUOnly);
                else if (!_wcsicmp(initString.c_str(), L"gaussian"))
                    m_net->InitLearnableParameters(nodePtr, false, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, initValueScale, initOnCPUOnly);
                else if (!_wcsicmp(initString.c_str(), L"fromFile"))
                {
                    std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                    if (initFromFilePath == "")
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    if(initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size()-1] == '\"')
                        // remove the opening and closing double quotes
                        initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size()-2);
                    if(!fexists(initFromFilePath))
                        RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                    dynamic_pointer_cast<LearnableParameter<ElemType>>(nodePtr)->InitFromFile(msra::strfun::utf16(initFromFilePath));
                }
                else
                    RuntimeError("'init' must be one of the values of [ uniform | gaussian | fixedValue ]");
            }
        }
        else if (OperationNameOf(SparseLearnableParameter) == cnNodeType)
        {
            if (parameter.size() < 1 || parameter.size() > 2)
                RuntimeError("%ls should have 1 or 2 parameters[rows, [cols=1]] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t cols = params.size() > 1 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "true");

                nodePtr = builder.CreateSparseLearnableParameter(name, rows, cols);

                nodePtr->SetParameterUpdateRequired(needGradient);
            }
            else if (pass == ndlPassFinal)
            {
                static int randomSeed = 1;
                wstring initString = node->GetOptionalParameter("init", "uniform");
                ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
                ElemType value = node->GetOptionalParameter("value", "0");
                
                if (!_wcsicmp(initString.c_str(), L"fixedValue"))
                    nodePtr->FunctionValues().SetValue(value);
                else if (!_wcsicmp(initString.c_str(), L"uniform"))
                    m_net->InitLearnableParameters(nodePtr, true, randomSeed++, initValueScale);
                else if (!_wcsicmp(initString.c_str(), L"gaussian"))
                    m_net->InitLearnableParameters(nodePtr, false, randomSeed++, initValueScale);
                else if (!_wcsicmp(initString.c_str(), L"fromFile"))
                {
                    std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                    if (initFromFilePath == "")
                        RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                    if(initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size()-1] == '\"')
                        // remove the opening and closing double quotes
                        initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size()-2);
                    if(!fexists(initFromFilePath))
                        RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                    dynamic_pointer_cast<SparseLearnableParameter<ElemType>>(nodePtr)->InitFromFile(msra::strfun::utf16(initFromFilePath));
                }
                else
                    RuntimeError("init must be one of the values of [ uniform | gaussian | fixedValue ]");
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

                nodePtr = builder.CreateLearnableParameter(name, rows, cols);
                nodePtr->SetParameterUpdateRequired(false);
            }
            else if (pass == ndlPassFinal || nodePtr->FunctionValues().GetNumElements() != 0)
            {
                ElemType val = parameter[0]->GetScalar();
                nodePtr->FunctionValues().SetValue(val);
            }
        }
        else if (cnNodeType == OperationNameOf(RowSliceNode))
        {
            if (parameter.size() != 3)
                RuntimeError("RowSlice should have three parameters. Usage: RowSlice(startRowIndex, numRows, origNodeName.");

            nodeParamCount = 1;
            nodeParamStart = 2;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t start_index = ((NDLNode<ElemType>*)params[0])->GetScalar();
                size_t num_rows = ((NDLNode<ElemType>*)params[1])->GetScalar();

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                nodePtr = builder.RowSlice(NULL, start_index, num_rows, name);
                // BUGBUG: This was probably meant to cut updates at this point. However, this will overwritten in EnumerateNodes() with values propagated upwards.
                nodePtr->SetParameterUpdateRequired(needGradient);
            }
        }
        else if (cnNodeType == OperationNameOf(RowRepeatNode))
        {
            if (parameter.size() != 2)
                RuntimeError("RowRepeat should have two parameters. Usage: RowRepeat(origNodeName, numRepeats.");

            nodeParamCount = 1;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t num_repeat = ((NDLNode<ElemType>*)params[1])->GetScalar();

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                nodePtr = builder.RowRepeat(NULL, num_repeat, name);
                nodePtr->SetParameterUpdateRequired(needGradient);
            }
        }
        else if (cnNodeType == OperationNameOf(DiagonalNode))
        {
            if (parameter.size() != 1)
                RuntimeError("Diagonal should have one parameter. Usage: Diagonal(origNodeName).");

            nodeParamCount = 1;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                nodePtr = builder.Diagonal(NULL, name);
                nodePtr->SetParameterUpdateRequired(needGradient);
            }
        }
        else if (cnNodeType == OperationNameOf(ReshapeNode))
        {
            if (parameter.size() < 2 || parameter.size() > 5)
                RuntimeError("Reshape should have two to five parameters. Usage: Reshape(origNodeName, numRows, [imageWidth=], [imageHeight=], [imageChannels=].");

            nodeParamCount = 1;
            nodeParamStart = 0;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t num_rows = ((NDLNode<ElemType>*)params[1])->GetScalar();
                size_t img_width = node->GetOptionalParameter("imageWidth", "0");
                size_t img_height = node->GetOptionalParameter("imageHeight", "0");
                size_t img_channels = node->GetOptionalParameter("imageChannels", "0");

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                nodePtr = builder.Reshape(NULL, num_rows, ImageLayoutWHC(img_width, img_height, img_channels), name);
                nodePtr->SetParameterUpdateRequired(needGradient);
            }
        }
        else if (cnNodeType == OperationNameOf(PastValueNode) || 
                 cnNodeType == OperationNameOf(FutureValueNode))
        {
            if (parameter.size() <2 || parameter.size() >3)
                RuntimeError("PastValue or FutureValue should have two to three fixed parameters. Usage: PastValue(rows, [cols], m, [timeStep=1, defaultPastValue=0.1]).");

            nodeParamCount = 1;
            nodeParamStart = parameter.size() > 2?2:1;

            if (pass == ndlPassInitial)
            {
                // evaluate only scalar parameters
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                size_t rows = ((NDLNode<ElemType>*)params[0])->GetScalar();
                // if we have three parameters the second is columns
                size_t cols = parameter.size() > 2 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

                bool needGradient = node->GetOptionalParameter("needGradient", "false");
                float defaultHiddenActivity = node->GetOptionalParameter("defaultHiddenActivity", "0.1");   // TODO: parameter should be called 'defaultHiddenActivation'

                //for backward compatibility we check timeStep first
                size_t timeStep = node->GetOptionalParameter("timeStep", "1");
                if (timeStep == 1)
                {
                    timeStep = node->GetOptionalParameter("delayTime", "1");
                }

                if (cnNodeType == OperationNameOf(PastValueNode))
                    nodePtr = builder.PastValue(NULL, defaultHiddenActivity, rows, cols, timeStep, name);
                else
                    nodePtr = builder.FutureValue(NULL, defaultHiddenActivity, rows, cols, timeStep, name);

                nodePtr->SetParameterUpdateRequired(needGradient);    // TODO: what's this for?
            }
        }    
        else if (cnNodeType == OperationNameOf(ConvolutionNode))
        {
            if (parameter.size() != 7)
                RuntimeError("%ls should have 7 fixed parameters[weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels,horizontalSubsample, verticalSubsample] and two optional parameters [zeroPadding = [false|yourvalue], maxTempMemSizeInSamples = [0|yourvalue]].", cnNodeType.c_str());

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


                nodePtr = builder.Convolution(NULL, NULL, kernelWidth, kernelHeight, outputChannels,
                                              horizontalSubsample, verticalSubsample, zeroPadding, name, maxTempMemSizeInSamples);
            }
        }
        else if (cnNodeType == OperationNameOf(MaxPoolingNode))
        {
            if (parameter.size() != 5)
                RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

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

                nodePtr = builder.MaxPooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight,
                                             horizontalSubsample, verticalSubsample, name);
            }
        }
        else if (cnNodeType == OperationNameOf(AveragePoolingNode))
        {
            if (parameter.size() != 5)
                RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample].", cnNodeType.c_str());

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

                nodePtr = builder.AveragePooling(NULL, /*inputWidth,inputHeight, channels,*/windowWidth, windowHeight,
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
                nodePtr = builder.CreateComputationNode(node->GetValue(), name);
            }
        }

        switch (pass)
        {
        case ndlPassInitial:
            node->SetEvalValue(nodePtr.get());
            // evaluate parameters
            EvaluateParameters(node, baseName, nodeParamStart, nodeParamCount, pass);
            break;
        case ndlPassResolve:
            {
            std::vector<void*> inputs = EvaluateParameters(node, baseName, nodeParamStart, nodeParamCount, pass);

            if (cnNodeType == OperationNameOf(RowStackNode)) //support variable length inputs
            {
                std::vector<ComputationNodeBasePtr> inputNodes;
                inputNodes.resize(inputs.size());
                for (int i = 0; i < inputs.size(); i++)
                    inputNodes[i] = ComputationNode<ElemType>::FromVoidPtr(inputs[i]);

                nodePtr->AttachInputs(inputNodes);
            }
            else
            {
                switch (inputs.size())
                {
                    // TODO: just use a vector attach
                case 1:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]));
                    break;
                case 2:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]), ComputationNode<ElemType>::FromVoidPtr(inputs[1]));
                    break;
                case 3:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]), ComputationNode<ElemType>::FromVoidPtr(inputs[1]), ComputationNode<ElemType>::FromVoidPtr(inputs[2]));
                    break;
                case 4:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]), ComputationNode<ElemType>::FromVoidPtr(inputs[1]), ComputationNode<ElemType>::FromVoidPtr(inputs[2]), ComputationNode<ElemType>::FromVoidPtr(inputs[3]));
                    break;
                case 5:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]), ComputationNode<ElemType>::FromVoidPtr(inputs[1]), ComputationNode<ElemType>::FromVoidPtr(inputs[2]), ComputationNode<ElemType>::FromVoidPtr(inputs[3]), ComputationNode<ElemType>::FromVoidPtr(inputs[4]));
                    break;
                case 6:
                    nodePtr->AttachInputs(ComputationNode<ElemType>::FromVoidPtr(inputs[0]), ComputationNode<ElemType>::FromVoidPtr(inputs[1]), ComputationNode<ElemType>::FromVoidPtr(inputs[2]), ComputationNode<ElemType>::FromVoidPtr(inputs[3]), ComputationNode<ElemType>::FromVoidPtr(inputs[4]), ComputationNode<ElemType>::FromVoidPtr(inputs[5]));
                    break;
                default:
                    if (nodeParamCount > 0)
                        RuntimeError("Invalid number of parameters name = '%s' call = '%s'\n", node->GetName().c_str(), node->GetValue().c_str());
                    break;
                }
            }
            // process common optional parameters (currently only "tag");
            ProcessOptionalParameters(node);
            break;
            }
        case ndlPassFinal:
            break;
        }
    }

    template class SynchronousExecutionEngine<float>;
    template class SynchronousExecutionEngine<double>;

}}}
