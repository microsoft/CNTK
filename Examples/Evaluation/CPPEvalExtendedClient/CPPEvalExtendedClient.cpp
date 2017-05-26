//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPPEvalExtendedClient.cpp : Sample application using the extended evaluation interface from C++
//

#include <sys/stat.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <algorithm>
#include <fstream>
#include <unordered_map>

#include "Eval.h"
#ifdef _WIN32
#include "Windows.h"
#endif

using namespace std;
using namespace Microsoft::MSR::CNTK;

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModelExtended<ElemType>**);

std::unordered_map<std::string, size_t> buildVocab(std::string filePath)
{
    std::ifstream ifs(filePath);
    size_t idx = 0;

    std::unordered_map<std::string, size_t> vocab;
    std::string line;
    while (std::getline(ifs, line))
    {
        vocab.insert(std::pair<std::string, size_t>(line, idx));
        idx += 1;
    }

    ifs.close();
    return vocab;
}

std::unordered_map<size_t, std::string> buildInvVocab(std::string filePath)
{
    std::ifstream ifs(filePath);
    size_t idx = 1;

    std::unordered_map<size_t, std::string> vocab;
    std::string line;
    while (std::getline(ifs, line))
    {
        vocab.insert(std::pair<size_t, std::string>(idx, line));
        idx += 1;
    }

    ifs.close();
    return vocab;
}

size_t word2idx(std::string word, std::unordered_map<std::string, size_t>& word2idxVocab)
{
    std::unordered_map<std::string, size_t>::iterator iter = word2idxVocab.find(word);
    if (iter == word2idxVocab.end())
    {
        throw std::runtime_error("word not found in source vocab");
    }

    return iter->second;
}


std::string idx2word(size_t idx, std::unordered_map<size_t, std::string>& idx2wordVocab)
{
    std::unordered_map<size_t, std::string>::iterator iter = idx2wordVocab.find(idx);
    if (iter == idx2wordVocab.end())
    {
        throw std::runtime_error("word index is not found in target vocab");
    }

    return iter->second;
}

void addOneHotWord(Values<float>& inputBuffers, size_t idx, VariableSchema& inputLayouts, size_t inputNode)
{
    size_t inputDim = inputLayouts[inputNode].m_numElements;
    for (size_t i = 0; i < inputDim; i++)
    {
        if (i == idx)
        {
            inputBuffers[inputNode].m_buffer.push_back(1);
        }
        else
        {
            inputBuffers[inputNode].m_buffer.push_back(0);
        }
    }
}

std::vector<std::string> feedInputVectors(std::string sentence, std::unordered_map<std::string, size_t>& word2idxVocab, Values<float>& inputBuffers, VariableSchema& inputLayouts)
{
    std::vector<std::string> words;

    // Split input sentence by space.
    char delimiters = ' ';
    size_t begin = 0;
    size_t end = sentence.find_first_of(delimiters);
    while (end != sentence.npos)
    {
        words.push_back(sentence.substr(begin, end - begin));
        begin = end + 1;
        end = sentence.find(delimiters, begin);
    }

    words.push_back(sentence.substr(begin));

    // Convert words to ids.
    std::vector<size_t> wordIds;
    for (size_t i = 0; i < words.size(); i++)
    {
        size_t id = word2idx(words[i], word2idxVocab);
        wordIds.push_back(id);
    }

    // Process the input words to construct network input vectors.
    // As the sentence begins and ends with special tag, we will ignore the first and last word.
    for (size_t i = 1; i < words.size() - 1; i++)
    {
        // Current word.
        size_t cwIdx = wordIds[i];
        addOneHotWord(inputBuffers, cwIdx, inputLayouts, 0);

        // Next word.
        size_t nwIdx = wordIds[i + 1];
        addOneHotWord(inputBuffers, nwIdx, inputLayouts, 1);

        // Previous word.
        size_t pwIdx = wordIds[i - 1];
        addOneHotWord(inputBuffers, pwIdx, inputLayouts, 2);
    }

    return words;
}

IEvaluateModelExtended<float>* SetupNetworkAndGetLayouts(std::string modelDefinition, VariableSchema& inputLayouts, VariableSchema& outputLayouts)
{
    // Native model evaluation instance
    IEvaluateModelExtended<float> *eval;

    GetEvalExtendedF(&eval);

    try
    {
        eval->CreateNetwork(modelDefinition);
    }
    catch (std::exception& ex)
    {
        fprintf(stderr, "%s\n", ex.what());
        throw;
    }
    fflush(stderr);

    // Get the model's layers dimensions
    outputLayouts = eval->GetOutputSchema();

    for (auto vl : outputLayouts)
    {
        fprintf(stderr, "Output dimension: %" PRIu64 "\n", vl.m_numElements);
        fprintf(stderr, "Output name: %ls\n", vl.m_name.c_str());
    }

    eval->StartForwardEvaluation({ outputLayouts[0].m_name });
    inputLayouts = eval->GetInputSchema();
    outputLayouts = eval->GetOutputSchema();

    return eval;
}


/// <summary>
/// Program for demonstrating how to run model evaluations using the native extended evaluation interface, also show
/// how to input sequence vectors to LSTM(RNN) network.
/// </summary>
/// <description>
/// This program is a native C++ client using the native extended evaluation interface
/// located in the <see cref="eval.h"/> file.
/// The CNTK evaluation library (EvalDLL.dll on Windows, and LibEval.so on Linux), must be found through the system's path. 
/// The other requirement is that Eval.h be included
/// In order to run this program the model must already exist in the example. To create the model,
/// first run the example in <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript. Once the model file ATIS.slot.lstm is created,
/// you can run this client.
/// This program demonstrates the usage of the Evaluate method requiring the input and output layers as parameters.
int main(int argc, char* argv[])
{
    // Get the binary path (current working directory)
    argc = 0;
    std::string app = argv[0];
    std::string path;
    size_t pos;
    int ret;

#ifdef _WIN32
    pos = app.rfind("\\");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. x64\Release
    const std::string modelBaseDir = path + "/../../Examples/LanguageUnderstanding/ATIS/BrainScript/";
    
#else // on Linux
    pos = app.rfind("/");
    path = (pos == std::string::npos) ? "." : app.substr(0, pos);

    // This relative path assumes launching from CNTK's binary folder, e.g. build/cpu/release/bin/
    const std::string modelBaseDir = path + "/../../../../Examples/LanguageUnderstanding/ATIS/BrainScript/";
#endif
    const std::string modelWorkingDirectory = modelBaseDir + "work/";

    const std::string modelFilePath = modelWorkingDirectory + "ATIS.slot.lstm";

    try
    {
        struct stat statBuf;
        if (stat(modelFilePath.c_str(), &statBuf) != 0)
        {
            fprintf(stderr, "Error: The model %s does not exist. Please follow instructions in README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript to create the model.\n", modelFilePath.c_str());
            return(1);
        }

        std::string networkConfiguration;
        networkConfiguration += "modelPath=\"" + modelFilePath + "\"";

        VariableSchema inputLayouts;
        VariableSchema outputLayouts;
        IEvaluateModelExtended<float> *eval;
        eval = SetupNetworkAndGetLayouts(networkConfiguration, inputLayouts, outputLayouts);

        vector<size_t> inputBufferSize;
        for (size_t i = 0; i < inputLayouts.size(); i++)
        {
            fprintf(stdout, "Input node name: %ls\n", inputLayouts[i].m_name.c_str());
            fprintf(stdout, "Input feature dimension: %" PRIu64 "\n", inputLayouts[i].m_numElements);
            inputBufferSize.push_back(inputLayouts[i].m_numElements);
        }

        vector<size_t> outputBufferSize;
        for (size_t i = 0; i < outputLayouts.size(); i++)
        {
            outputBufferSize.push_back(outputLayouts[i].m_numElements);
        }

        // Build source word vocab to id 
        const::string sourceVocab = modelBaseDir + "/../Data/ATIS.vocab";
        if (stat(sourceVocab.c_str(), &statBuf) != 0)
        {
            fprintf(stderr, "Error: The file '%s' does not exist.\n", sourceVocab.c_str());
            return(1);
        }
        std::unordered_map<std::string, size_t> word2idxVocab = buildVocab(sourceVocab);

        // Build id to target word vocab
        const::string targetVocab = modelBaseDir + "/../Data/ATIS.label";
        if (stat(targetVocab.c_str(), &statBuf) != 0)
        {
            fprintf(stderr, "Error: The file '%s' does not exist.\n", targetVocab.c_str());
            return(1);
        }
        std::unordered_map<size_t, std::string> idx2wordVocab = buildInvVocab(targetVocab);

        // Use the following sentence as input example.
        // One single space is used as word sperator. 
        std::string inputSequences = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";

        Values<float> inputBuffers = inputLayouts.CreateBuffers<float>(inputBufferSize);
        Values<float> outputBuffers = outputLayouts.CreateBuffers<float>(outputBufferSize);

        // Feed input sequence vectors to network
        std::vector<std::string> words = feedInputVectors(inputSequences, word2idxVocab, inputBuffers, inputLayouts);

        // Forward propagation
        eval->ForwardPass(inputBuffers, outputBuffers);

        // Get output from output layer
        auto buf = outputBuffers[0].m_buffer;
        size_t bufSize = outputBuffers[0].m_buffer.size();

        std::vector<std::string> outputs;
        size_t outputDim = outputLayouts[0].m_numElements;
        size_t outputStep = bufSize / outputDim;

        auto iter = buf.begin();
        for (size_t i = 0; i < outputStep; i++)
        {
            auto max_iter = std::max_element(iter, iter + outputDim);
            auto index = max_iter - iter;
            outputs.push_back(idx2word(index, idx2wordVocab));
            iter += outputDim;
        }

        words.erase(words.begin());
        words.pop_back();
        fprintf(stdout, "Slot tag for sentence \"%s\" is as follows:\n", inputSequences.c_str());
        for (size_t i = 0; i < outputs.size(); i++)
        {
            fprintf(stdout, "%10s -- %s\n", words[i].c_str(), outputs[i].c_str());
        }

        eval->Destroy();
       
        // This pattern is used by End2EndTests to check whether the program runs to complete.
        fprintf(stdout, "Evaluation complete.\n");
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

    fflush(stdout);
    fflush(stderr);
    return ret;
}
