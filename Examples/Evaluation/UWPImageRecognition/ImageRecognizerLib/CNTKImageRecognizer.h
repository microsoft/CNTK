//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace ImageRecognizerLib
{
    public ref class CNTKImageRecognizer sealed
    {
        CNTK::DeviceDescriptor evalDevice = CNTK::DeviceDescriptor::UseDefaultDevice();
        CNTK::FunctionPtr model;
        CNTK::Variable inputVar;
        CNTK::NDShape inputShape;
        std::vector<std::wstring> classNames;
        CNTKImageRecognizer(Platform::String^ modelFile, Platform::String^ classesFile);
        std::wstring classifyImage(const uint8_t* image_data, size_t image_data_len);

    public:
        static CNTKImageRecognizer^ CNTKImageRecognizer::Create(Platform::String^ modelFile, Platform::String^ classesFile);
        Windows::Foundation::IAsyncOperation<Platform::String^>^ RecognizeObjectAsync(const Platform::Array<byte>^ bytes);
        uint32_t GetRequiredWidth();
        uint32_t GetRequiredHeight();
        uint32_t GetRequiredChannels();
    };
}
