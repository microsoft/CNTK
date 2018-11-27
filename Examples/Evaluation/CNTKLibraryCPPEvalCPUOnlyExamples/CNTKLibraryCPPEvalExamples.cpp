//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamples.cpp : Sample application shows how to evaluate a model using CNTK V2 API.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms.
#endif

#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "CNTKLibrary.h"

using namespace CNTK;

template <typename ElementType>
void PrintOutput(size_t, std::vector<std::vector<ElementType>>);

/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data for a single sample.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// - how to retrieve evaluation result and retrieve output data in dense format.
/// Note: The example uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// </summary>
void EvaluationSingleSampleUsingDense(const wchar_t* modelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate single sample using dense format.\n");

    // Load the model.
    // The model is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    PrintOutput<float>(outputVar.Shape().TotalSize(), outputData);
}

/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data for a batch of samples.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// - how to retrieve evaluation result and retrieve output data in dense format.
/// Note: The example uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// </summary>
void EvaluationBatchUsingDense(const wchar_t* modelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate batch of samples using dense format.\n");

    // The number of samples in the batch.
    size_t sampleCount = 3;

    // Load the model.
    // The model is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize() * sampleCount);
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map.
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    PrintOutput<float>(outputVar.Shape().TotalSize(), outputData);
}

void RunEvaluationOnSingleSample(FunctionPtr, const DeviceDescriptor&);

/// <summary>
/// The example shows
/// - how to evaluate multiple sample requests in parallel.
/// Note: The example uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// </summary>
void ParallelEvaluationExample(const wchar_t* modelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate multiple requests in parallel.\n");

    size_t threadCount = 3;

    // Load the model.
    // The model is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Run evaluation in parallel.
    std::vector<std::thread> threadList(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(RunEvaluationOnSingleSample, modelFunc->Clone(ParameterCloningMethod::Share), device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        printf("thread %d joined.\n", th);
    }
}

void RunEvaluationOnSingleSample(FunctionPtr evalInstance, const DeviceDescriptor& device)
{
    // Get input variable. The model has only one single input.
    Variable inputVar = evalInstance->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = evalInstance->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    evalInstance->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);
}


std::unordered_map<std::string, size_t> BuildVocabIndex(const wchar_t*);
std::vector<std::string> BuildSlotIndex(const wchar_t*);

/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data as sequence using one-hot vector.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// - how to retrieve evaluation result.
/// The examples uses the model trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
/// Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// The vocabularyFile specifies the vacabulary file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/query.wl
/// The labelFile specifies the label file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/slots.wl
/// </summary>
/// <param name="device">Specify on which device to run the evaluation</param>
void EvaluationSingleSequenceUsingOneHot(const wchar_t* modelFile, const wchar_t* vocabularyFile, const wchar_t* labelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate single sequence using one-hot vector.\n");

    // Load the model.
    // The model is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
    // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Read word and slot index files.
    std::unordered_map<std::string, size_t> vocabToIndex = BuildVocabIndex(vocabularyFile);
    std::vector<std::string> indexToSlots = BuildSlotIndex(labelFile);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];
    size_t vocabSize = inputVar.Shape().TotalSize();

    const char *inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
    std::vector<size_t> seqData;
    std::vector<std::string> inputWords;
    std::stringstream inputStream;
    std::string word;
    size_t index;

    // build one-hot index for the input sequence.
    inputStream.str(inputSentence);
    while (inputStream >> word)
    {
        inputWords.push_back(word);
        index = vocabToIndex.at(word);
        seqData.push_back(index);
    }

    // SeqStartFlag is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
    bool seqStartFlag = true;

    // Create input value using one-hot vector and input data map
    ValuePtr inputVal = Value::CreateSequence<float>(vocabSize, seqData, seqStartFlag, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    // output the result
    size_t outputSampleSize = outputVar.Shape().TotalSize();
    if (outputData.size() != 1)
    {
        throw("Only one sequence of slots is expected as output.");
    }
    std::vector<float> slotSeq = outputData[0];
    if (slotSeq.size() % outputSampleSize != 0)
    {
        throw("The number of elements in the slot sequence is not a multiple of sample size");
    }

    size_t numOfSlotsInOutput = slotSeq.size() / outputSampleSize;
    if (inputWords.size() != numOfSlotsInOutput)
    {
        throw("The number of input words and the number of output slots do not match");
    }
    for (size_t i = 0; i < numOfSlotsInOutput; i++)
    {
        float max = slotSeq[i * outputSampleSize];
        size_t maxIndex = 0;
        for (size_t j = 1; j < outputSampleSize; j++)
        {
            if (slotSeq[i * outputSampleSize + j] > max)
            {
                max = slotSeq[i * outputSampleSize + j];
                maxIndex = j;
            }
        }
        printf("     %10s ---- %s\n", inputWords[i].c_str(), indexToSlots[maxIndex].c_str());
    }
    printf("\n");
}

/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data as batch of sequences with variable length.
///   how to prepare data using one-hot vector format.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// The example uses the model trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
/// Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// The vocabularyFile specifies the vacabulary file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/query.wl
/// The labelFile specifies the label file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/slots.wl
/// </summary>
void EvaluationBatchOfSequencesUsingOneHot(const wchar_t* modelFile, const wchar_t* vocabularyFile, const wchar_t* labelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate batch of sequences with variable length using one-hot vector.\n");

    // Load the model.
    // The model is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
    // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Read word and slot index files.
    std::unordered_map<std::string, size_t> vocabToIndex = BuildVocabIndex(vocabularyFile);
    std::vector<std::string> indexToSlots = BuildSlotIndex(labelFile);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];
    size_t vocabSize = inputVar.Shape().TotalSize();

    std::vector<const char *> inputSentences = {
        "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS",
        "BOS flights from new york to seattle EOS"
    };

    // Prepare input data.
    std::vector<std::vector<std::string>> inputWordsList(inputSentences.size());
    // Each sample is represented by an index to the one-hot vector, so the index of the non-zero value of each sample is saved in the inner list.
    // The outer list represents sequences contained in the batch.
    std::vector<std::vector<size_t>> inputBatch;
    // SeqStartFlagBatch is used to indicate whether this sequence is a new sequence (true) or concatenating the previous sequence (false).
    std::vector<bool> seqStartFlagBatch;
    std::string word;
    size_t index;

    for (size_t seqIndex = 0; seqIndex < inputSentences.size(); seqIndex++)
    {
        std::stringstream inputStream;
        std::vector<size_t> seqData;
        // build one-hot index for the input sequences.
        inputStream.str(inputSentences[seqIndex]);
        while (inputStream >> word)
        {
            inputWordsList[seqIndex].push_back(word);
            index = vocabToIndex.at(word);
            seqData.push_back(index);
        }
        inputBatch.push_back(seqData);
        seqStartFlagBatch.push_back(true);
    }

    // Create input value representing the batch data and input data map
    ValuePtr inputVal = Value::CreateBatchOfSequences<float>(vocabSize, inputBatch, seqStartFlagBatch, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    // output the result
    size_t outputSampleSize = outputVar.Shape().TotalSize();
    if (outputData.size() != inputBatch.size())
    {
        throw("The number of sequence in output does not match that in input.");
    }
    printf("The number of sequences in the batch: %d\n", (int)outputData.size());
    for (size_t seqno = 0; seqno < outputData.size(); seqno++)
    {
        std::vector<float> slotSeq = outputData[seqno];
        printf("Sequence %d:\n", (int)seqno);

        if (slotSeq.size() % outputSampleSize != 0)
        {
            throw("The number of elements in the slot sequence is not a multiple of sample size");
        }

        size_t numOfSlotsInOutput = slotSeq.size() / outputSampleSize;
        if (inputWordsList[seqno].size() != numOfSlotsInOutput)
        {
            throw("The number of input words and the number of output slots do not match");
        }
        for (size_t i = 0; i < numOfSlotsInOutput; i++)
        {
            float max = slotSeq[i * outputSampleSize];
            size_t maxIndex = 0;
            for (size_t j = 1; j < outputSampleSize; j++)
            {
                if (slotSeq[i * outputSampleSize + j] > max)
                {
                    max = slotSeq[i * outputSampleSize + j];
                    maxIndex = j;
                }
            }
            printf("     %10s ---- %s\n", inputWordsList[seqno][i].c_str(), indexToSlots[maxIndex].c_str());
        }
        printf("\n");
    }
}

/// <summary>
/// The example shows
/// - how to prepare input data as sequence using sparse input.
/// The example uses the model trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
/// Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
/// The parameter 'modelFile' specifies the path to the model.
/// The vocabularyFile specifies the vacabulary file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/query.wl
/// The labelFile specifies the label file used by the ATIS model, e.g. <CNTK>/Examples/LanguageUnderstanding/ATIS/BrainScript/slots.wl
/// </summary>
void EvaluationSingleSequenceUsingSparse(const wchar_t* modelFile, const wchar_t* vocabularyFile, const wchar_t* labelFile, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate single sequence using sparse input.\n");

    // Load the model.
    // The model is trained by <CNTK>/Examples/LanguageUnderstanding/ATIS/Python/LanguageUnderstanding.py
    // Please see README.md in <CNTK>/Examples/LanguageUnderstanding/ATIS about how to train the model.
    FunctionPtr modelFunc = Function::Load(modelFile, device);

    // Read word and slot index files.
    std::unordered_map<std::string, size_t> vocabToIndex = BuildVocabIndex(vocabularyFile);
    std::vector<std::string> indexToSlots = BuildSlotIndex(labelFile);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];
    size_t vocabSize = inputVar.Shape().TotalSize();

    const char *inputSentence = "BOS i would like to find a flight from charlotte to las vegas that makes a stop in st. louis EOS";
    std::vector<size_t> seqData;
    std::vector<std::string> inputWords;
    std::stringstream inputStream;
    std::string word;

    // build one-hot index for the input sequence.
    inputStream.str(inputSentence);
    while (inputStream >> word)
    {
        inputWords.push_back(word);
    }

    size_t seqLen = inputWords.size();
    // For this example, only 1 non-zero value for each sample.
    size_t numNonZeroValues = seqLen * 1;
    std::vector<SparseIndexType> colStarts;
    std::vector<SparseIndexType> rowIndices;
    std::vector<float> nonZeroValues;

    size_t count = 0;
    for (; count < seqLen; count++)
    {
        // Get the index of the word
        auto nonZeroValueIndex = static_cast<SparseIndexType>(vocabToIndex[inputWords[count]]);
        // Add the sample to the sequence
        nonZeroValues.push_back(1.0);
        rowIndices.push_back(nonZeroValueIndex);
        colStarts.push_back(static_cast<SparseIndexType>(count));
    }
    colStarts.push_back(static_cast<SparseIndexType>(numNonZeroValues));

    // Create input value using one-hot vector and input data map
    ValuePtr inputVal = Value::CreateSequence<float>(vocabSize, seqLen, colStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    // output the result
    size_t outputSampleSize = outputVar.Shape().TotalSize();
    if (outputData.size() != 1)
    {
        throw("Only one sequence of slots is expected as output.");
    }
    std::vector<float> slotSeq = outputData[0];
    if (slotSeq.size() % outputSampleSize != 0)
    {
        throw("The number of elements in the slot sequence is not a multiple of sample size");
    }

    size_t numOfSlotsInOutput = slotSeq.size() / outputSampleSize;
    if (inputWords.size() != numOfSlotsInOutput)
    {
        throw("The number of input words and the number of output slots do not match");
    }
    for (size_t i = 0; i < numOfSlotsInOutput; i++)
    {
        float max = slotSeq[i * outputSampleSize];
        size_t maxIndex = 0;
        for (size_t j = 1; j < outputSampleSize; j++)
        {
            if (slotSeq[i * outputSampleSize + j] > max)
            {
                max = slotSeq[i * outputSampleSize + j];
                maxIndex = j;
            }
        }
        printf("     %10s ---- %s\n", inputWords[i].c_str(), indexToSlots[maxIndex].c_str());
    }
    printf("\n");
}

/// <summary>
/// The example shows
/// - how to load a pretrained model and evaluate an intermediate layer of its network.
/// Note: The example uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The parameter 'modelFilePath' specifies the path to the model.
/// </summary>
void EvaluateIntermediateLayer(const wchar_t* modelFilePath, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate intermediate layer =====\n");

    // Load the model.
    FunctionPtr rootFunc = Function::Load(modelFilePath, device);

    std::wstring intermediateLayerName = L"final_avg_pooling";
    FunctionPtr interLayerPrimitiveFunc = rootFunc->FindByName(intermediateLayerName);

    // The Function returned by FindByName is a primitive function.
    // For evaluation, it is required to create a composite function from the primitive function.
    FunctionPtr modelFunc = AsComposite(interLayerPrimitiveFunc);

    Variable outputVar = modelFunc->Output();
    Variable inputVar = modelFunc->Arguments()[0];

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    PrintOutput<float>(outputVar.Shape().TotalSize(), outputData);
}

/// <summary>
/// The example shows
/// - how to load a pretrained model and evaluate several nodes by combining their outputs
/// Note: The example uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The parameter 'modelFilePath' specifies the path to the model.
/// </summary>
void EvaluateCombinedOutputs(const wchar_t* modelFilePath, const DeviceDescriptor& device)
{
    printf("\n===== Evaluate combined outputs =====\n");

    // Load the model.
    FunctionPtr modelFunc = Function::Load(modelFilePath, device);

    // Get node of interest
    std::wstring intermediateLayerName = L"final_avg_pooling";
    FunctionPtr interLayerPrimitiveFunc = modelFunc->FindByName(intermediateLayerName);

    Variable poolingOutput = interLayerPrimitiveFunc->Output();

    // Create a function which combine outputs from the node "final_avg_polling" and the final layer of the model.
    FunctionPtr evalFunc = Combine( { modelFunc->Output(), poolingOutput });
    Variable inputVar = evalFunc->Arguments()[0];

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    Variable modelOutput = evalFunc->Outputs()[0];
    Variable interLayerOutput = evalFunc->Outputs()[1];

    std::unordered_map<Variable, ValuePtr> outputDataMap = { { modelOutput, nullptr }, { interLayerOutput, nullptr } };

    // Start evaluation on the device
    evalFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense outputs
    for(auto & outputVariableValuePair : outputDataMap)
    {
        auto variable = outputVariableValuePair.first;
        auto value = outputVariableValuePair.second;
        std::vector<std::vector<float>> outputData;
        value->CopyVariableValueTo(variable, outputData);
        PrintOutput<float>(variable.Shape().TotalSize(), outputData);
    }
}

std::shared_ptr<std::fstream> GetIfstream(const wchar_t *filePath)
{
    const size_t pathBufferLen = 1024;
    char pathBuffer[pathBufferLen];
    size_t writtenBytes = ::wcstombs(pathBuffer, filePath, pathBufferLen);
    if (writtenBytes == (size_t)-1)
        throw ("Unknown characters in the file path.");
    else if (writtenBytes == pathBufferLen)
        throw("The file path is too long");
    return std::make_shared<std::fstream>(pathBuffer);
}

std::unordered_map<std::string, size_t> BuildVocabIndex(const wchar_t *filePath)
{
    std::unordered_map<std::string, size_t> vocab;
    std::string str;
    size_t idx = 0;

    std::shared_ptr<std::fstream> input = GetIfstream(filePath);

    while (*input >> str)
        vocab[str] = idx++;
    return vocab;
}

std::vector<std::string> BuildSlotIndex(const wchar_t *filePath)
{
    std::shared_ptr<std::fstream> input = GetIfstream(filePath);
    std::vector<std::string> slots;
    std::string str;

    while (*input >> str)
        slots.push_back(str);
    return slots;
}

/// <summary>
/// Print out the evalaution results.
/// </summary>
template <typename ElementType>
void PrintOutput(size_t sampleSize, std::vector<std::vector<ElementType>> outputBuffer)
{
    printf("The batch contains %d sequences.\n", (int)outputBuffer.size());
    for(size_t seqNo = 0; seqNo < outputBuffer.size(); seqNo++)
    {
        auto seq = outputBuffer[seqNo];
        if (seq.size() % sampleSize != 0)
            throw("The number of elements in the sequence is not a multiple of sample size");

        printf("Sequence %d contains %d samples.\n", (int)seqNo, (int)(seq.size() / sampleSize));
        size_t sampleNo = 0;
        for(size_t i = 0; i < seq.size(); )
        {
            if (i % sampleSize == 0)
                printf("    sample %d: ", (int)sampleNo);
            printf("%f", seq[i++]);
            if (i % sampleSize == 0)
            {
                printf(".\n");
                sampleNo++;
            }
            else
                printf(", ");
        }
    }
}