//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Note: Despite its name, this file is really about parsing NDL into an actual ComputationNetwork.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "NDLNetworkBuilder.h"
#include "LinearAlgebraNodes.h"
#include "RecurrentNodes.h"
#include "ConvolutionalNodes.h"
#include "RNNNodes.h"
#include "NonlinearityNodes.h"
#include "ReshapingNodes.h"
#include "InputAndParamNodes.h"
#include "TensorShape.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

template <class ElemType>
void NDLNodeEvaluatorImpl<ElemType>::Evaluate(NDLNode<ElemType>* node, const wstring& baseName, const NDLPass pass)
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

            wstring dynamicAxis = node->GetOptionalParameter("dynamicAxis", "");
            // TODO: Map dynamicAxis from name to node at this point, where that node is memoized inside NDL.
            // first look for this node already existing in the network
            // BUGBUG: How does this set the dimensions then?
            if (m_net->NodeNameExists(name))
                nodePtr = dynamic_pointer_cast<ComputationNode<ElemType>>(m_net->GetNodeFromName(name));
            else if (isSparse)
                nodePtr = builder.CreateSparseInputNode(name, tensorShape, dynamicAxis);
            else
                nodePtr = builder.CreateInputNode(name, tensorShape, dynamicAxis);
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
            size_t imageWidth    = ((NDLNode<ElemType>*) params[0])->GetScalar();
            size_t imageHeight   = ((NDLNode<ElemType>*) params[1])->GetScalar();
            size_t imageChannels = ((NDLNode<ElemType>*) params[2])->GetScalar();
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));
            wstring dynamicAxis = node->GetOptionalParameter("dynamicAxis", "");

            if (isSparse)
                nodePtr = builder.CreateSparseInputNode(name, ImageDimensions::AsTensorShape(imageWidth, imageHeight, imageChannels, imageLayoutKind), dynamicAxis);
            else
                nodePtr = builder.CreateInputNode(name, ImageDimensions::AsTensorShape(imageWidth, imageHeight, imageChannels, imageLayoutKind), dynamicAxis);
        }
    }
    else if (OperationNameOf(LearnableParameter) == cnNodeType || cnNodeType == L"ImageParameter")
    {
        bool isImage = (cnNodeType == L"ImageParameter");
        if (!isImage)
        {
            if (parameter.size() < 1)
                RuntimeError("%ls should have 1 or more parameters (tensor dimensions, e.g. [vecdim] or [rows, cols]) plus other optional parameters (learningRateMultiplier=[1|0|float], init=[uniform|gaussian|fixedvalue|fromFile|heNormal|bilinear], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());
        }
        else
        {
            if (parameter.size() < 3)
                RuntimeError("%ls should have 3 or more parameters [imageWidth, imageHeight, imageChannels] plus other optional parameters (learningRateMultiplier=[1|0|float], init=[uniform|gaussian|fixedvalue|fromFile|heNormal|bilinear], initValueScale=[1|float], value=[0|float]).", cnNodeType.c_str());
        }

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t i = 0;
            auto tensorShape = ProcessTensorShapeParameters(node, params, i, isImage, cnNodeType);

            // for backward compatibility needsGradient is now subsumed by learningRateMultiplier
            bool gradientUpdateNeeded = node->GetOptionalParameter("needGradient", "true") 
                                     && node->GetOptionalParameter("needsGradient", "true")
                                     && node->GetOptionalParameter("computeGradient", "true");
            float learningRateMultiplier = node->GetOptionalParameter("learningRateMultiplier", "1");
            if (!gradientUpdateNeeded)  // if user has specified needsGradient flag to false
                learningRateMultiplier = 0.0;

            nodePtr = builder.CreateLearnableParameter(name, tensorShape);
            nodePtr->SetLearningRateMultiplier(learningRateMultiplier);
        }
        else if (pass == ndlPassFinal)
        {
            static int randomSeed = 1;
            wstring initString = node->GetOptionalParameter("init", "uniform");
            ElemType initValueScale = node->GetOptionalParameter("initValueScale", "1");
            ElemType value = node->GetOptionalParameter("value", "0");
            bool initOnCPUOnly = node->GetOptionalParameter("initOnCPUOnly", "false");
            int forcedRandomSeed = node->GetOptionalParameter("randomSeed", "-1" /*disabled*/);

            if (EqualCI(initString, L"fixedValue"))
                m_net->InitLearnableParameters(nodePtr, L"fixedValue", value);
            else if (EqualCI(initString, L"uniform"))
                m_net->InitLearnableParameters(nodePtr, L"uniform",  initValueScale, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, initOnCPUOnly);
            else if (EqualCI(initString, L"gaussian"))
                m_net->InitLearnableParameters(nodePtr, L"gaussian", initValueScale, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, initOnCPUOnly);
            else if (EqualCI(initString, L"bilinear"))
            {
                const size_t kernelWidth = node->GetOptionalParameter("kernelWidth", "0");
                const size_t kernelHeight = node->GetOptionalParameter("kernelHeight", "0");
                assert(kernelWidth > 0 && kernelHeight > 0);
                m_net->InitLearnableParametersWithBilinearFill<ElemType>(nodePtr, kernelWidth, kernelHeight);
            }
            else if (EqualCI(initString, L"fromFile"))
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
            else if (EqualCI(initString, L"heNormal"))
                m_net->InitLearnableParameters(nodePtr, L"heNormal", initValueScale, forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed, initOnCPUOnly);
            else
                RuntimeError("'init' must be one of the values of [ uniform | gaussian | fixedValue | fromFile | heNormal | bilinear]");
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
            nodePtr->SetLearningRateMultiplier(0);
        }
        else if (pass == ndlPassFinal || nodePtr->Value().GetNumElements() != 0)
        {
            ElemType val = parameter[0]->GetScalar();
            m_net->InitLearnableParameters(nodePtr, L"fixedValue", val);
        }
    }
    else if (cnNodeType == L"RowSlice") // Note: This now maps onto SliceNode which specifies the end differently.
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

            nodePtr = builder.RowSlice(NULL, start_index, num_rows, name);
        }
    }
    else if (cnNodeType == OperationNameOf(RowRepeatNode))
    {
        if (parameter.size() != 2)
            RuntimeError("RowRepeat should have two parameters. Usage: RowRepeat(origNodeName, numRepeats).");

        nodeParamCount = 1;
        nodeParamStart = 0;

        if (pass == ndlPassInitial)
        {
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
            size_t num_repeat = ((NDLNode<ElemType>*) params[1])->GetScalar();

            nodePtr = builder.RowRepeat(NULL, num_repeat, name);
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

            nodePtr = builder.Diagonal(NULL, name);
        }
    }
    else if (cnNodeType == L"Reshape" /*OperationNameOf(ReshapeNode)*/)
    {
        if (parameter.size() != 2)
            RuntimeError("Reshape should have two parameters. Usage: Reshape(origNodeName, numRows, [imageWidth=], [imageHeight=], [imageChannels=].");

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
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));

            nodePtr = builder.LegacyReshape(NULL, num_rows, ImageDimensions::AsTensorShape(img_width, img_height, img_channels, imageLayoutKind), name);
        }
    }
    else if (cnNodeType == OperationNameOf(ReconcileDynamicAxisNode))
    {
        nodeParamCount = 2;
        nodeParamStart = 0;

        if (pass == ndlPassInitial)
        {
            nodePtr = builder.ReconcileDynamicAxis(NULL, NULL, name);
        }
    }
    else if (cnNodeType == OperationNameOf(PastValueNode) ||
             cnNodeType == OperationNameOf(FutureValueNode))
    {
        if (parameter.size() < 2 || parameter.size() > 3) // we allow 3 for legacy (cols parameter which is now unused)
            RuntimeError("PastValue or FutureValue should have two to three fixed parameters. Usage: PastValue(rows, input, [timeStep=1, defaultHiddenActivity=0.1]).");
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

            float defaultHiddenActivity = node->GetOptionalParameter("defaultHiddenActivity", "0.1"); // TODO: parameter should be called 'defaultHiddenActivation'

            // for backward compatibility we check 'timeStep' first
            size_t timeStep = node->GetOptionalParameter("timeStep", "1");
            if (timeStep == 1)
                timeStep = node->GetOptionalParameter("delayTime", "1");

            if (cnNodeType == OperationNameOf(PastValueNode))
                nodePtr = builder.PastValue(NULL, defaultHiddenActivity, rows, timeStep, name);
            else
                nodePtr = builder.FutureValue(NULL, defaultHiddenActivity, rows, timeStep, name);
        }
    }
    else if (cnNodeType == OperationNameOf(ConvolutionNode) ||
             cnNodeType == OperationNameOf(PoolingNode) ||
             cnNodeType == OperationNameOf(MaxUnpoolingNode))
    {
        if (parameter.size() != 2 && parameter.size() != 3 && parameter.size() != 7)
        {
            if (cnNodeType == OperationNameOf(ConvolutionNode))
            {
                RuntimeError("%ls: unexpected parameter count. %ls supports 2 modes: \n"
                             "1. 2D convolution which takes 7 fixed parameters [weightNodeName, inputValueNodeName, kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample] \n"
                             "and two optional parameters [zeroPadding = [false|yourvalue], maxTempMemSizeInSamples = [0|yourvalue], imageLayout = \"HWC\"|\"cudnn\"]. \n"
                             "2. ND convolution which takes 3 fixed parameters [weightNodeName, inputValueNodeName, kernelShape] and \n"
                             "10 optional parameters [mapCount = [0|yourvalue], stride = [1|yourvalue], sharing = [true|yourvalue], autoPadding = [true|yourvalue], lowerPad = [0|yourvalue], upperPad = [0|yourvalue], bool transpose = [false|yourvalue], maxTempMemSizeInSamples = [0|yourvalue], imageLayout = \"cudnn\"|\"HWC\"]. \n"
                             "For ND convolution, parameters kernelShape, mapCount, stride, sharing, autoPadding, lowerPad, upperPad can be arrays, e.g. kernelShape={5, 5, 3}",
                             cnNodeType.c_str(), cnNodeType.c_str());
            }
            else if (cnNodeType == OperationNameOf(PoolingNode))
            {
                RuntimeError("%ls: unexpected parameter count. %ls 3 fixed parameters [inputValueNodeName, poolKind, kernelShape] and \n"
                             "5 optional parameters stride = [1|yourvalue], autoPadding = [true|yourvalue], lowerPad = [0|yourvalue], upperPad = [0|yourvalue], imageLayout = \"cudnn\"]. \n"
                             "Parameters kernelShape, stride, autoPadding, lowerPad, upperPad can be arrays, e.g. kernelShape={5, 5, 3}",
                             cnNodeType.c_str(), cnNodeType.c_str());
            }
            else if (cnNodeType == OperationNameOf(MaxUnpoolingNode))
            {
                RuntimeError("%ls: unexpected parameter count. %ls 3 fixed parameters [inputValueNodeName, mask, kernelShape] and \n"
                             "5 optional parameters stride = [1|yourvalue], autoPadding = [true|yourvalue], lowerPad = [0|yourvalue], upperPad = [0|yourvalue], imageLayout = \"cudnn\"]. \n"
                             "Parameters kernelShape, stride, autoPadding, lowerPad, upperPad can be arrays, e.g. kernelShape={5, 5, 3}",
                             cnNodeType.c_str(), cnNodeType.c_str());
            }
        }

        // setup the parameter position of children so we can hook them up later
        nodeParamStart = 0;
        nodeParamCount = (cnNodeType == OperationNameOf(ConvolutionNode) || cnNodeType == OperationNameOf(MaxUnpoolingNode))
                         ? 2
                         : 1;

        if (pass == ndlPassInitial)
        {
            if (parameter.size() == 2 || parameter.size() == 3)
            {
                auto reqParams = node->GetParameters(false);
                auto optParams = node->GetParameters(true);
                auto paramGetter = [reqParams, node](size_t index) -> TensorShape
                {
                    assert(index < reqParams.size());
                    auto parm = reqParams[index];
                    if (parm->GetType() != ndlTypeArray)
                        return TensorShape((size_t)parm->GetScalar());
                    auto parms = node->GetParentScript()->ParseVariable(parm->GetValue(), false)->GetParameters();
                    vector<size_t> dims(parms.size());
                    for (size_t i = 0; i < dims.size(); i++)
                        dims[i] = parms[i]->GetValue();
                    return TensorShape(dims);
                };
                auto paramResolver = [optParams, node](const char* name, size_t defaultVal) -> TensorShape
                {
                    auto res = std::find_if(begin(optParams), end(optParams), [name](const NDLNode<ElemType>* n) { return EqualCI(n->GetName(), name); });
                    if (res == end(optParams))
                        return TensorShape(defaultVal);
                    auto parm = node->GetParentScript()->ParseVariable((*res)->GetValue(), false);
                    if (parm->GetType() == ndlTypeConstant)
                        return TensorShape((size_t)parm->GetValue());
                    auto parms = parm->GetParameters();
                    vector<size_t> dims(parms.size());
                    for (size_t i = 0; i < dims.size(); i++)
                        dims[i] = parms[i]->GetValue();
                    return TensorShape(dims);
                };
                auto boolParamResolver = [&optParams, node](const char* name, bool defaultVal) -> vector<bool>
                {
                    auto res = std::find_if(begin(optParams), end(optParams), [name](const NDLNode<ElemType>* n) { return EqualCI(n->GetName(), name); });
                    if (res == end(optParams))
                        return vector<bool>{defaultVal};
                    auto parm = node->GetParentScript()->ParseVariable((*res)->GetValue(), false);
                    if (parm == nullptr)
                        return vector<bool>{(*res)->GetValue()};
                    if (parm->GetType() != ndlTypeArray)
                        return vector<bool>{parm->GetValue()};
                    auto parms = parm->GetParameters();
                    vector<bool> dims(parms.size());
                    for (size_t i = 0; i < dims.size(); i++)
                        dims[i] = parms[i]->GetValue();
                    return dims;
                };

                auto kernelShape = paramGetter(reqParams.size() - 1);
                auto mapCount = paramResolver("mapCount", 0);
                auto stride = paramResolver("stride", 1);
                auto sharing = boolParamResolver("sharing", true);
                auto autoPad = boolParamResolver("autoPadding", true);
                auto lowerPad = paramResolver("lowerPad", 0);
                auto upperPad = paramResolver("upperPad", 0);
                ImageLayoutKind imageLayout = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "CHW"));
                size_t maxTempMemSizeInSamples = node->GetOptionalParameter("maxTempMemSizeInSamples", "0");

                if (cnNodeType == OperationNameOf(MaxUnpoolingNode))
                    nodePtr = builder.MaxUnpooling(NULL, NULL, kernelShape, stride, autoPad, lowerPad, upperPad, imageLayout, name);
                else if (cnNodeType == OperationNameOf(PoolingNode))
                {
                    auto parm = node->GetParentScript()->ParseVariable(reqParams[1]->GetValue(), false);
                    auto pool = PoolKindFrom(wstring(parm->GetValue()));
                    nodePtr = builder.Pooling(NULL, pool, kernelShape, stride, autoPad, lowerPad, upperPad, imageLayout, name);
                }
                else
                {
                    bool transpose = node->GetOptionalParameter("transpose", "false");
                    nodePtr = builder.Convolution(NULL, NULL, kernelShape, mapCount, stride, sharing, 
                                                  autoPad, lowerPad, upperPad, transpose, imageLayout, maxTempMemSizeInSamples, name);
                }

            }
            else if (parameter.size() == 7)
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
                                              horizontalSubsample, verticalSubsample, imageLayoutKind, zeroPadding,
                                              maxTempMemSizeInSamples, name);
            }
            else
                assert(false);
        }
    }
    else if (cnNodeType == OperationNameOf(MaxPoolingNode))
    {
        if (parameter.size() != 5)
            RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayout = \"HWC\"|\"cudnn\"].", cnNodeType.c_str());

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
            RuntimeError("%ls should have 5 parameters[inputValueNodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayout = \"HWC\"|\"cudnn\"].", cnNodeType.c_str());

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
        if (parameter.size() < 5)
            RuntimeError("%ls should have 5 fixed parameters[inputValueNodeName, scale, bias, runMean, runVariance].", cnNodeType.c_str());

        // setup the parameter position of children so we can hook them up later
        nodeParamCount = 6;
        nodeParamStart = 0;

        if (pass == ndlPassInitial)
        {
            int id = 5; // skip inputValueNode, scale and bias, runMean, runVariance.
            // evaluate only scalar parameters
            vector<void*> params = EvaluateParameters(node, baseName, id, parameter.size() - id, pass);

            // Optional parameters
            bool spatial = node->GetOptionalParameter("spatial", "false");
            double normTimeConst = node->GetOptionalParameter("normalizationTimeConstant", "0");
            double blendTimeConst = node->GetOptionalParameter("blendTimeConstant", "0");
            double epsilon = node->GetOptionalParameter("epsilon", "0.00001");
            std::wstring bnEngineS = node->GetOptionalParameter("engine", "cntk");
            bool useCntkEngine;
            if (EqualCI(bnEngineS, L"cntk"))
                useCntkEngine = true;
            else if (EqualCI(bnEngineS, L"cudnn"))
                useCntkEngine = false;
            else
                InvalidArgument("Unsupported batch normalization engine, choose either \"cntk\"(default) or \"cudnn\".");
            ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "CHW"));

            nodePtr = builder.BatchNormalization(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, spatial, normTimeConst, blendTimeConst, epsilon, useCntkEngine, imageLayoutKind, name);

            if (parameter.size() == 5) 
            {
                // Patch up NDL network config by creating and injecting an additional input parameter for
                // BatchNormalizationNode
                ComputationNodePtr runSampleCount = builder.CreateLearnableParameter(name + L".run_sample_count", TensorShape(1));
                runSampleCount->SetLearningRateMultiplier(0);
                m_net->InitLearnableParameters(runSampleCount, L"fixedValue", 0);

                NDLNode<ElemType>* runCountNode = new NDLNode<ElemType>("runCount", ConfigValue("0"), node->GetParentScript(), ndlTypeConstant);

                runCountNode->SetEvalValue(runSampleCount.get());

                node->InsertParam(runCountNode);
            }
        }
    }
    else if (cnNodeType == OperationNameOf(CropNode))
    {
        // We expect 2 or 4 inputs.
        if (parameter.size() != 2 && parameter.size() != 4)
        {
            RuntimeError("%ls accepts inputs: [input1, input2, offsetX, offsetY] or \
                                              [input1, input2] or \
                                              [input1, input2, eqNode1, eqNode2].", cnNodeType.c_str());
        }

        if (pass == ndlPassInitial)
        {
            // In initial phase we just need to create node.
            if (parameter.size() == 4)
            {
                // Here we need to determine if 3rd and 4th parameters are offsets or equivalence nodes.
                vector<void*> params = EvaluateParameters(node, baseName, 0, parameter.size(), pass);
                // TODO: Is there a better way to discriminate?
                if (((NDLNode<ElemType>*) params[2])->GetType() == NDLType::ndlTypeConstant)
                {
                    // We have offsets given, take offsets from evaluated parameters.
                    size_t offsetX = ((NDLNode<ElemType>*) params[2])->GetScalar();
                    size_t offsetY = ((NDLNode<ElemType>*) params[3])->GetScalar();

                    // Create crop node with offsets but without inputs (will be attached later in resolve phase).
                    nodePtr = builder.Crop(nullptr, nullptr, offsetX, offsetY, name);
                }
                else
                {
                    // We have 4 node inputs (2 crop inputs and 2 equivalence node inputs).
                    nodePtr = builder.Crop(nullptr, nullptr, nullptr, nullptr, name);
                }
            }
            else
            {
                // Just two inputs, must be node inputs which will be attached in the resolve phase below.
                nodePtr = builder.Crop(nullptr, nullptr, name);
            }
            // Done processing in this phase.
            nodeParamStart = 0;
            nodeParamCount = 0;
        }
        else
        {
            // In non-initial phase we just process node inputs below, here we just set inputs of interest.
            nodeParamStart = 0;
            nodeParamCount = nodePtr->GetNumInputs();
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

        if (cnNodeType == OperationNameOf(RowStackNode)) // support variable length inputs
        {
            std::vector<ComputationNodeBasePtr> inputNodes;
            inputNodes.resize(inputs.size());
            for (int i = 0; i < inputs.size(); i++)
                inputNodes[i] = ComputationNode<ElemType>::FromVoidPtr(inputs[i]);

            nodePtr->AttachInputs(inputNodes);
        }
        else
        {
#if 1
            vector<ComputationNodeBasePtr> inputNodes;
            for (let& in : inputs)
                inputNodes.push_back(ComputationNode<ElemType>::FromVoidPtr(in));

            nodePtr->AttachInputs(inputNodes);
#else       // TODO: delete this
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
#endif
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
TensorShape NDLNodeEvaluatorImpl<ElemType>::ProcessTensorShapeParameters(const NDLNode<ElemType>* node, const vector<void*>& params, size_t& i, bool isImage, const wstring& cnNodeType /*for error messages only*/)
{
    // gather dims
    vector<size_t> dims;
    dims.push_back(((NDLNode<ElemType>*) params[i])->GetScalar()); // first is mandatory
    for (i++; i < params.size(); i++)
        dims.push_back(((NDLNode<ElemType>*) params[i])->GetScalar());

    // if image then interpret as W, H, C with layout according to optional imageLayout parameter
    // If more than 3 parameters are given, then we assume that this is for a Times operation and interpret the last 3 dimensions according to imageLayout.
    if (isImage)
    {
        if (dims.size() < 3)
            RuntimeError("%ls should have 3 or more parameters [width, height, numChannels].", cnNodeType.c_str());
        ImageLayoutKind imageLayoutKind = ImageLayoutKindFrom(node->GetOptionalParameter("imageLayout", "HWC"));
        size_t k0 = dims.size() - 3; // last 3 need to be arranged
        SmallVector<size_t> imageDims = ImageDimensions::AsTensorShape(dims[k0 + 0], dims[k0 + 1], dims[k0 + 2], imageLayoutKind).GetDims();
        for (size_t k = 0; k < 3; k++)
            dims[k0 + k] = imageDims[k];
    }

    // turn into tensor
    return TensorShape(dims);
}

template class NDLBuilderImpl<float>;
template class NDLBuilderImpl<double>;

}}}
