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
#include <time.h>
#include <chrono>


//Extended eval allows to pass features in batches of different size as long as each feature vector has the same dimension as the network input. 
#define EXTENDED_EVAL

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

#ifdef EXTENDED_EVAL
    IEvaluateModelExtended<float> *model;
#else
    IEvaluateModel<float> *model;
#endif

    size_t pos;
    int ret;

#ifdef _WIN32
    pos = app.rfind("\\");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
    const std::string modelWorkingDirectory = path + "/../../Examples/Image/MNIST/Data/";
#else // on Linux
    pos = app.rfind("/");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. build/cpu/release/bin/
    const std::string modelWorkingDirectory = path + "/../../../../Examples/Image/MNIST/Data/";
#endif
    const std::string modelFilePath = "D:/src1/private/data/dev/sr/src/AM/enu_ls4m_lstm/amv2/Tensor.sampling.se.67"; // "C:\\musor\\TFLSTM\\tflstm.model.0"; //;

    try
    {
        struct stat statBuf;
        if (stat(modelFilePath.c_str(), &statBuf) != 0)
        {
            fprintf(stderr, "Error: The model %s does not exist. Please follow instructions in README.md in <CNTK>/Examples/Image/MNIST to create the model.\n", modelFilePath.c_str());
            return(1);
        }

        GetEvalExtendedF(&model);

        // Load model with desired outputs
        std::string networkConfiguration;
        // Uncomment the following line to re-define the outputs (include h1.z AND the output ol.z)
        // When specifying outputNodeNames in the configuration, it will REPLACE the list of output nodes 
        // with the ones specified.
        //networkConfiguration += "outputNodeNames=\"h1.z:ol.z\"\n";
        networkConfiguration += "modelPath=\"" + modelFilePath + "\" \n";
        networkConfiguration += "numCPUThreads=1";
        model->CreateNetwork(networkConfiguration);

#ifdef EXTENDED_EVAL
        // Initialize start

        model->StartForwardEvaluation({ model->GetOutputSchema()[0].m_name });
        auto m_inputBuffer = model->GetInputSchema().CreateBuffers<float>({ 1 });
        auto m_outputBuffer = model->GetOutputSchema().CreateBuffers<float>({ 1 });
        unsigned inDim = 80; // 80;// 957;
        bool m_resetRNN = true;

        // Init input buffer:
        for (unsigned i = 0; i < inDim; i++)
            m_inputBuffer[0].m_buffer.push_back(0.0);
        // Initialize end
#else
        GetEvalF(&model);
#endif

        // get the model's layers dimensions
        std::map<std::wstring, size_t> inDims;
        std::map<std::wstring, size_t> outDims;

#ifdef EXTENDED_EVAL
        // Eval start
        auto start_time = std::chrono::high_resolution_clock::now();

        for (unsigned i = 0; i < inDim; i++)
            m_inputBuffer[0].m_buffer[i] = (float)i; //specify actual value
        for (int k = 0; k < 1000; k++)
        {
            model->ForwardPass(m_inputBuffer, m_outputBuffer, m_resetRNN);
            m_resetRNN = false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        auto g_fSenoneEvalTimeInSecond = std::chrono::duration_cast<std::chrono::milliseconds>(time).count();

        fprintf(stderr, "Time spent '%d' output:\n", (int)g_fSenoneEvalTimeInSecond);


        // Eval end
#else
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
#endif
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
