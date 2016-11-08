//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalClient.cpp : Sample application using the evaluation interface from C++
//
#include <sys/stat.h>
#include "Eval.h"
#ifdef _WIN32
#include "Windows.h"
#endif

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
/// The CNTK evaluation library (EvalDLL.dll on Windows, and LibEval.so on Linux), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/Image/MNIST. Once the model file 01_OneHidden is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.
int main(int argc, char* argv[])
{
    // Get the binary path (current working directory)
    argc = 0;   
    std::string app = argv[0];
    std::string path; 
    IEvaluateModel<float> *model;
    size_t pos;
    int ret;

#ifdef _WIN32
    pos = app.rfind("\\");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
    const std::string modelWorkingDirectory = path + "/../../Examples/Image/GettingStarted";
#else // on Linux
    pos = app.rfind("/");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. build/cpu/release/bin/
    const std::string modelWorkingDirectory = path + "/../../../../Examples/Image/GettingStarted";
#endif
    const std::string modelFilePath = modelWorkingDirectory + "/Output/Models/01_OneHidden";

    try
    {
        struct stat statBuf;
        if (stat(modelFilePath.c_str(), &statBuf) != 0)
        {
            fprintf(stderr, "Error: The model %s does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/GettingStarted to create the model.\n", modelFilePath.c_str());
            return(1);
        }

        GetEvalF(&model);

        // Load model with desired outputs
        std::string networkConfiguration;
        // Uncomment the following line to re-define the outputs (include h1.z AND the output ol.z)
        // When specifying outputNodeNames in the configuration, it will REPLACE the list of output nodes 
        // with the ones specified.
        //networkConfiguration += "outputNodeNames=\"h1.z:ol.z\"\n";
        networkConfiguration += "modelPath=\"" + modelFilePath + "\"";
        model->CreateNetwork(networkConfiguration);

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

        // We can call the evaluate method and get back the results (single layer)...
        model->Evaluate(inputLayer, outputLayer);

        // Output the results
        fprintf(stderr, "Layer '%ls' output:\n", outputLayerName.c_str());
        for (auto& value : outputs)
        {
            fprintf(stderr, "%f\n", value);
        }

        // This pattern is used by End2EndTests to check whether the program runs to complete.
        fprintf(stderr, "Evaluation complete.\n");
        ret = 0;
    }
    catch (const std::exception& err)
    {
        fprintf(stderr, "Evaluation failed. EXCEPTION occurred: %s\n", err.what());
        ret = 1;
    }
    catch (...)
    {
        fprintf(stderr, "Evaluation failed. Unknown ERROR occurred.\n");
        ret = 1;
    }

    fflush(stderr);
    return ret;
}
