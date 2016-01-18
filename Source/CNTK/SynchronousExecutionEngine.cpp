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
#include "TensorShape.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

template <class ElemType>
void SynchronousNodeEvaluator<ElemType>::Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass)
{
    ComputationNetworkBuilder<ElemType> builder(*m_net);

    // constants don't need to be evaluated, they just translate into numbers...
    if (node->GetType() == ndlTypeConstant || node->GetType() == ndlTypeArray)
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

    if (OperationNameOf(InputValue) == cnNodeType || OperationNameOf(SparseInputValue) == cnNodeType)
    {
        bool isSparse = (OperationNameOf(SparseInputValue) == cnNodeType);
        if (parameter.size() < 1)
            RuntimeError("%ls should have 1 or more parameters (tensor dimensions, e.g. [vecdim] or [rows, cols]).", cnNodeType.c_str());

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t i = 0;
            auto tensorShape = ProcessTensorShapeParameters(node, params, i, /*isImage=*/false, cnNodeType);

            // first look for this node already existing in the network
            // BUGBUG: How does this set the dimensions then?
            if (m_net->NodeNameExists(name))
                nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->GetNodeFromName(name));
            else if (isSparse)
                nodePtr = builder.CreateSparseInputNode(name, tensorShape);
            else
                nodePtr = builder.CreateInputNode(name, tensorShape);
        }
    }
    else if (cnNodeType == L"ImageInput" || cnNodeType == L"SparseImageInput")
    {
        bool isSparse = (cnNodeType == L"SparseImageInput");
        if (parameter.size() < 3 || parameter.size() > 4) // we allow 4 for legacy (numImages, was ignored)
            RuntimeError("%ls should have 3 parameters[imageWidth, imageHeight, imageChannels].", cnNodeType.c_str());

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t imageWidth = ((NDLNode<ElemType>*) params[0])->GetScalar();
            size_t imageHeight = ((NDLNode<ElemType>*) params[1])->GetScalar();
            size_t imageChannels = ((NDLNode<ElemType>*) params[2])->GetScalar();
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));

            if (isSparse)
                nodePtr = builder.CreateSparseInputNode(name, ImageDimensions::AsTensorShape(imageWidth, imageHeight, imageChannels, imageLayoutKind));
            else
                nodePtr = builder.CreateInputNode(name, ImageDimensions::AsTensorShape(imageWidth, imageHeight, imageChannels, imageLayoutKind));
        }
    }
    else if (OperationNameOf(LearnableParameter) == cnNodeType || cnNodeType == L"ImageParameter")
    {
        bool isImage = (cnNodeType == L"ImageParameter");
        if (!isImage)
        {
            if (parameter.size() < 1)
                RuntimeError("%ls should have 1 or more parameters (tensor dimensions, e.g. [vecdim] or [rows, cols]) plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());
        }
        else
        {
            if (parameter.size() < 3)
                RuntimeError("%ls should have 3 parameters [imageWidth, imageHeight, imageChannels] plus other optional parameters (needGradient=[true|false], init=[uniform|gaussian|fixedvalue], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());
        }

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t i = 0;
            auto tensorShape = ProcessTensorShapeParameters(node, params, i, isImage, cnNodeType);
            if (isImage)
                tensorShape.AppendInPlace(3, 1); // this goes into the column dimension
            bool needGradient = node->GetOptionalParameter("needGradient", "true");

            nodePtr = builder.CreateLearnableParameter(name, tensorShape);
            nodePtr->SetParameterUpdateRequired(needGradient);
        }
        else if (pass == ndlPassFinal)
        {
            static int randomSeed = 1;
            wstring initString = node->GetOptionalParameter("init", "uniform");
            ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
            ElemType value = node->GetOptionalParameter("value", "0");
            bool initOnCPUOnly = node->GetOptionalParameter("initOnCPUOnly", "false");
            int forcedRandomSeed = node->GetOptionalParameter("randomSeed", "-1" /*disabled*/);

            if (!_wcsicmp(initString.c_str(), L"fixedValue"))
                nodePtr->Value().SetValue(value);
            else if (!_wcsicmp(initString.c_str(), L"uniform"))
                m_net->InitLearnableParameters(nodePtr, true, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long) forcedRandomSeed, initValueScale, initOnCPUOnly);
            else if (!_wcsicmp(initString.c_str(), L"gaussian"))
                m_net->InitLearnableParameters(nodePtr, false, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long) forcedRandomSeed, initValueScale, initOnCPUOnly);
            else if (!_wcsicmp(initString.c_str(), L"fromFile"))
            {
                std::string initFromFilePath = node->GetOptionalParameter("initFromFilePath", "");
                if (initFromFilePath == "")
                    RuntimeError("initFromFilePath must be set when using \"fromFile\" initialization method");
                if (initFromFilePath[0] == '\"' && initFromFilePath[initFromFilePath.size() - 1] == '\"')
                    // remove the opening and closing double quotes
                    initFromFilePath = initFromFilePath.substr(1, initFromFilePath.size() - 2);
                if (!fexists(initFromFilePath))
                    RuntimeError("File pointed to by initFromFilePath does not exist: %s", initFromFilePath.c_str());
                dynamic_pointer_cast<LearnableParameter<ElemType>>(nodePtr)->InitFromFile(msra::strfun::utf16(initFromFilePath));
            }
            else
                RuntimeError("'init' must be one of the values of [ uniform | gaussian | fixedValue ]");
        }
    }
#if 0 // not functional at present
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
                    nodePtr->Value().SetValue(value);
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
#endif
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
        else if (pass == ndlPassFinal || nodePtr->Value().GetNumElements() != 0)
        {
            ElemType val = parameter[0]->GetScalar();
            nodePtr->Value().SetValue(val);
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
            size_t start_index = ((NDLNode<ElemType>*) params[0])->GetScalar();
            size_t num_rows = ((NDLNode<ElemType>*) params[1])->GetScalar();

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
            size_t num_repeat = ((NDLNode<ElemType>*) params[1])->GetScalar();

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
    else if (cnNodeType == L"Reshape" /*OperationNameOf(ReshapeNode)*/)
    {
        if (parameter.size() < 2 || parameter.size() > 5)
            RuntimeError("Reshape should have two to five parameters. Usage: Reshape(origNodeName, numRows, [imageWidth=], [imageHeight=], [imageChannels=].");

        nodeParamCount = 1;
        nodeParamStart = 0;

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t num_rows = ((NDLNode<ElemType>*) params[1])->GetScalar();
            size_t img_width = node->GetOptionalParameter("imageWidth", "0");
            size_t img_height = node->GetOptionalParameter("imageHeight", "0");
            size_t img_channels = node->GetOptionalParameter("imageChannels", "0");

            bool needGradient = node->GetOptionalParameter("needGradient", "false");
            nodePtr = builder.DeprecatedReshape(NULL, num_rows, ImageDimensions::AsTensorShape(img_width, img_height, img_channels, ImageLayoutKind::HWC /*legacy*/), name); // BUGBUG: use a tensor descriptor instead
            nodePtr->SetParameterUpdateRequired(needGradient);
        }
    }
    else if (cnNodeType == OperationNameOf(PastValueNode) ||
             cnNodeType == OperationNameOf(FutureValueNode))
    {
        if (parameter.size() < 2 || parameter.size() > 3) // we allow 3 for legacy (cols parameter which is now unused)
            RuntimeError("PastValue or FutureValue should have two to three fixed parameters. Usage: PastValue(rows, input, [timeStep=1, defaultPastValue=0.1]).");
        // TODO: allow a tensor descriptor. Or allow 0 (inference). Maybe already supported--check this.

        nodeParamCount = 1;                            // number of inputs
        nodeParamStart = parameter.size() > 2 ? 2 : 1; // index of input

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t rows = ((NDLNode<ElemType>*) params[0])->GetScalar();
            // if we have three parameters the second is columns
            // ignore legacy size_t cols = parameter.size() > 2 ? ((NDLNode<ElemType>*)params[1])->GetScalar() : 1;

            //bool needGradient = node->GetOptionalParameter("needGradient", "false");  // TODO: what's this for?
            float defaultHiddenActivity = node->GetOptionalParameter("defaultHiddenActivity", "0.1"); // TODO: parameter should be called 'defaultHiddenActivation'

            // for backward compatibility we check 'timeStep' first
            size_t timeStep = node->GetOptionalParameter("timeStep", "1");
            if (timeStep == 1)
                timeStep = node->GetOptionalParameter("delayTime", "1");

            if (cnNodeType == OperationNameOf(PastValueNode))
                nodePtr = builder.PastValue(NULL, defaultHiddenActivity, rows, timeStep, name);
            else
                nodePtr = builder.FutureValue(NULL, defaultHiddenActivity, rows, timeStep, name);

            //nodePtr->SetParameterUpdateRequired(needGradient);    // TODO: what's this for?
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
            vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);
            id = 0; // reset counter because the params array starts at zero
            size_t kernelWidth = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t kernelHeight = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t outputChannels = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t horizontalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t verticalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            assert(id == 5);

            // optional
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));
            bool zeroPadding = node->GetOptionalParameter("zeroPadding", "false");
            size_t maxTempMemSizeInSamples = node->GetOptionalParameter("maxTempMemSizeInSamples", "0");

            nodePtr = builder.Convolution(NULL, NULL, kernelWidth, kernelHeight, outputChannels,
                                          horizontalSubsample, verticalSubsample, imageLayoutKind, zeroPadding, maxTempMemSizeInSamples, name);
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
            size_t windowWidth = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t windowHeight = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t horizontalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t verticalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            assert(id == 4);

            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));

            nodePtr = builder.MaxPooling(NULL, /*inputWidth,inputHeight, channels,*/ windowWidth, windowHeight,
                                         horizontalSubsample, verticalSubsample, imageLayoutKind, name);
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
            size_t windowWidth = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t windowHeight = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t horizontalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            size_t verticalSubsample = ((NDLNode<ElemType>*) params[id++])->GetScalar();
            assert(id == 4);

            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));

            nodePtr = builder.AveragePooling(NULL, /*inputWidth,inputHeight, channels,*/ windowWidth, windowHeight,
                                             horizontalSubsample, verticalSubsample, imageLayoutKind, name);
        }
    }
    else if (cnNodeType == OperationNameOf(BatchNormalizationNode))
    {
        if (parameter.size() != 5)
            RuntimeError("%ls should have 5 fixed parameters[inputValueNodeName, scale, bias, runMean, runInvStdDev].", cnNodeType.c_str());

        // setup the parameter position of children so we can hook them up later
        nodeParamCount = 5;
        nodeParamStart = 0;

        if (pass == ndlPassInitial)
        {
            int id = 5; // skip inputValueNode, scale and bias, runMean, runInvStdDev.
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);

            // Optional parameters
            bool eval = node->GetOptionalParameter("eval", "false");
            bool spatial = node->GetOptionalParameter("spatial", "false");
            double expAvgFactor = node->GetOptionalParameter("expAvgFactor", "1.0");
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "CHW"));

            nodePtr = builder.BatchNormalization(nullptr, nullptr, nullptr, nullptr, nullptr, eval, spatial, expAvgFactor, imageLayoutKind, name);
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

// ProcessTensorShapeParameters - assume positional parameters starting from position i are tensor dimensions--parse those.
// Is isImage then must be a 3D tensor, which is interpreted as (W,H,C), and optional parameter 'imageLayout' says how.
template <class ElemType>
TensorShape SynchronousNodeEvaluator<ElemType>::ProcessTensorShapeParameters(const NDLNode<ElemType>* node, const vector<void*>& params, size_t& i, bool isImage, const wstring& cnNodeType /*for error messages only*/)
{
    // gather dims
    vector<size_t> dims;
    dims.push_back(((NDLNode<ElemType>*) params[i])->GetScalar()); // first is mandatory
    for (i++; i < params.size(); i++)
        dims.push_back(((NDLNode<ElemType>*) params[i])->GetScalar());

    // turn into tensor
    TensorShape tensorShape(dims);

    // if image then interpret as W, H, C with layout according to optional imageLayout parameter
    if (isImage)
    {
        if (dims.size() != 3)
            RuntimeError("%ls should have 3 parameters [width, height, numChannels].", cnNodeType.c_str());
        ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));
        tensorShape = ImageDimensions::AsTensorShape(tensorShape[0], tensorShape[1], tensorShape[2], imageLayoutKind);
    }

    return tensorShape;
}

template class SynchronousExecutionEngine<float>;
template class SynchronousExecutionEngine<double>;
} } }
