//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//

#include "stdafx.h"
#include "eval.h"

using namespace Microsoft::MSR::CNTK;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

/// <summary>
/// Program for demonstrating how to run model evaluations using the native evaluation interface
/// </summary>
/// <description>
/// This program is a native C++ client using the native evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation dll (EvalDLL.dll), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.
/// This client can evaluate in two modes:
///     -batchMode (default) - input samples are evaluated in batches. This evaluation has some overhead because:
///             The size of the batch is not known.
///             Network nodes timestamps are reset for each batch that forces CNTK to recompute all nodes, even those that haven't changed.
///             Certain checks are performed for each batch, e.g. whether matrices have been allocated, etc.
///     -streamMode - input samples are evaluated in minibatche of fixed size. The network is initialized prior to evaluation and the overhead of processing each minibatch is minimized.
/// 
/// To run the client in stream mode, set STREAM_MODE to true.
int _tmain(int argc, _TCHAR* argv[])
{
    bool STREAM_MODE = true;

    // Get the binary path (current working directory)
    argc = 0;
    std::wstring wapp(argv[0]);
    std::string app(wapp.begin(), wapp.end());
    std::string path = app.substr(0, app.rfind("\\"));
    
    // Load the eval library
    auto hModule = LoadLibrary(L"evaldll.dll");
    if (hModule == nullptr)
    {
        const std::wstring msg(L"Cannot find evaldll.dll library");
        const std::string ex(msg.begin(), msg.end());
        throw new std::exception(ex.c_str());
    }

    // Get the factory method to the evaluation engine
    std::string func = "GetEvalF";
    auto procAddress = GetProcAddress(hModule, func.c_str());
    auto getEvalProc = (GetEvalProc<float>)procAddress;

    // Native model evaluation instance
    IEvaluateModel<float> *model;
    getEvalProc(&model);

    // This relative path assumes launching from CNTK's binary folder
    const std::string modelWorkingDirectory = path + "\\..\\..\\Examples\\Image\\MNIST\\Data\\";
    const std::string modelFilePath = modelWorkingDirectory + "..\\Output\\Models\\01_OneHidden";

    // Load model
    model->CreateNetwork("modelPath=\"" + modelFilePath + "\"");

    // get the model's layers dimensions
    std::map<std::wstring, size_t> inDims;
    std::map<std::wstring, size_t> outDims;
    model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
    model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);
    
    // Generate dummy input values in the appropriate structure and size
    auto inputLayerName = inDims.begin()->first;
    std::vector<float> inputs;
    for (int i = 0; i < inDims[inputLayerName]; i++)
    {
        inputs.push_back(static_cast<float>(i % 255));
    }

    // Allocate the output values layer
    std::vector<float> outputs;

    // Setup the maps for inputs and output
    Layer inputLayer;
    inputLayer.insert(MapEntry(inputLayerName, &inputs));
    Layer outputLayer;
    auto outputLayerName = outDims.begin()->first;
    outputLayer.insert(MapEntry(outputLayerName, &outputs));

    // We can call the evaluate method and get back the results...
    if (STREAM_MODE) {
        //Evaluation in stream mode
        //Prepare the network
        model->PrepareForStreamMode();
        // For simpliciy, evaluate the same input sample
        for (int i = 0; i < 10; i++)
        {
            model->EvaluateStreamMode(inputLayer, outputLayer);
        }
    }
    else
    {
        model->Evaluate(inputLayer, outputLayer);
    }

    // Output the results
    for each (auto& value in outputs)
    {
        fprintf(stderr, "%f\n", value);
    }

    return 0;
}