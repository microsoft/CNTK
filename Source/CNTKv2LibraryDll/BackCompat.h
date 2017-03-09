//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <fstream>

namespace CNTK
{
    namespace Internal
    {
        FunctionPtr LoadLegacyModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice);

        inline bool IsLegacyModel(std::fstream& stream)
        {
            static const char legacyMarker[] = { 0x42, 0x00, 0x43, 0x00, 0x4e, 0x00, 0x00, 0x00 }; // L"BCN"
            static const auto size = sizeof(legacyMarker);
            char buffer[size];
            const auto position = stream.tellg();
            stream.read(buffer, size);
            stream.seekg(position);
            return (strcmp(legacyMarker, buffer) == 0);
        }

        enum class LegacyModelDataType : unsigned int
        {
            Auto, // starting from model version 7, type is encoded into the model file and need not be explicitly specified
            Float,
            Double
        };

        LegacyModelDataType DetectLegacyModelDataType(const std::wstring& modelFile);
    }
}