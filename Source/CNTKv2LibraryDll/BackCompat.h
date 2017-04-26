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

        FunctionPtr ConvertFromLegacyModel(const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& net);

        bool IsLegacyModel(std::fstream& stream);

        bool IsLegacyModel(const char *buffer, size_t bufferSize);

        enum class LegacyModelDataType : unsigned int
        {
            Auto, // starting from model version 7, type is encoded into the model file and need not be explicitly specified
            Float,
            Double
        };

        LegacyModelDataType DetectLegacyModelDataType(const std::wstring& modelFile);
    }
}