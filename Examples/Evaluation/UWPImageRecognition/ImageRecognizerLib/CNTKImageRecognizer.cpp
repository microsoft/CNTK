//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "pch.h"
#include "CNTKImageRecognizer.h"

using namespace ImageRecognizerLib;
using namespace Platform;
using namespace Microsoft::MSR::CNTK;

#include "utils.inl"

std::wstring CNTKImageRecognizer::classifyImage(const uint8_t* image_data, size_t image_data_len)
{
    // Prepare the input vector and convert it to the correct color scheme (BBB ... GGG ... RRR)
    size_t resized_image_data_len = GetRequiredWidth() * GetRequiredHeight() * GetRequiredChannels();
    std::vector<uint8_t> image_data_array(resized_image_data_len);
    memcpy_s(image_data_array.data(), image_data_len, image_data, resized_image_data_len);
    std::vector<float> rawFeatures = get_features(image_data_array.data(), GetRequiredWidth(), GetRequiredHeight());

    // Prepare the input layer of the computation graph
    // Most of the work is putting rawFeatures into CNTK's data representation format
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputLayer = {};

    auto features = CNTK::Value::CreateBatch<float>(inputShape, rawFeatures, evalDevice, false);
    inputVar = model->Arguments()[0];
    inputLayer.insert({ inputVar, features });

    // Prepare the output layer of the computation graph
    // For this a NULL blob will be placed into the output layer
    // so that CNTK can place its own datastructure there
    std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputLayer = {};
    CNTK::Variable outputVar = model->Output();
    CNTK::NDShape outputShape = outputVar.Shape();
    size_t possibleClasses = outputShape.Dimensions()[0];

    std::vector<float> rawOutputs(possibleClasses);
    auto outputs = CNTK::Value::CreateBatch<float>(outputShape, rawOutputs, evalDevice, false);
    outputLayer.insert({ outputVar, NULL });

    // Evaluate the image and extract the results (which will be a [ #classes x 1 x 1 ] tensor)
    model->Evaluate(inputLayer, outputLayer, evalDevice);

    CNTK::ValuePtr outputVal = outputLayer[outputVar];
    std::vector<std::vector<float>> resultsWrapper;
    std::vector<float> results;

    outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);
    results = resultsWrapper[0];

    // Map the results to the string representation of the class
    int64_t image_class = find_class(results);
    return classNames.at(image_class);
}

uint32_t CNTKImageRecognizer::GetRequiredWidth()
{
    return (uint32_t)inputShape[0];
}

uint32_t CNTKImageRecognizer::GetRequiredHeight()
{
    return (uint32_t)inputShape[1];
}

uint32_t CNTKImageRecognizer::GetRequiredChannels()
{
    return (uint32_t)inputShape[2];
}

CNTKImageRecognizer::CNTKImageRecognizer(String^ modelFile, Platform::String^ classesFile)
{
    std::wstring w_str = std::wstring(modelFile->Data());
    model = CNTK::Function::Load(w_str, evalDevice);

    // List out all the outputs and their indexes
    // The probability output is usually listed as 'z' and is 
    // usually the last layer
    size_t z_index = model->Outputs().size() - 1;

    // Modify the in-memory model to use the z layer as the actual output
    auto z_layer = model->Outputs()[z_index];
    model = CNTK::Combine({ z_layer.Owner() });

    // Extract information about what the model accepts as input
    inputVar = model->Arguments()[0];
    // Shape contains image [width, height, depth] respectively
    inputShape = inputVar.Shape();

    // Load the class names
    w_str = std::wstring(classesFile->Data());
    classNames = read_class_names(w_str);
}

CNTKImageRecognizer^ CNTKImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile)
{
    return ref new CNTKImageRecognizer(modelFile, classesFile);
}

Windows::Foundation::IAsyncOperation<Platform::String^>^ CNTKImageRecognizer::RecognizeObjectAsync(const Platform::Array<byte>^ bytes)
{
    return concurrency::create_async([=] {
        // The data we've got is in RGBA format. We should convert it to BGR
        std::vector<uint8_t> rgb((bytes->Length / 4) * 3);
        uint8_t* rgba = bytes->Data;

        uint32_t i = 0;
        for (uint32_t j = 0; j < bytes->Length;)
        {
            uint32_t r = j++;  // R
            uint32_t g = j++;  // G
            uint32_t b = j++;  // B
            uint32_t a = j++;  // A (skipped)

            rgb[i++] = rgba[r];
            rgb[i++] = rgba[g];
            rgb[i++] = rgba[b];
        }

        auto image_class = classifyImage(rgb.data(), rgb.size());
        return ref new Platform::String(image_class.c_str());
    });
}