//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"
#include "Reader.h"

namespace CNTK {

enum class CropType
{
    Center = 0,         // center crop with a given size 
    RandomSide = 1,     // random scale resized with shorter side sampled from min and max (ResNet-style)
    RandomArea = 2,     // random scale resized with area size ratio between min and max (Inception-style)
    MultiView10 = 3     // 10 view crop
};

// A helper class for image specific parameters.
// A simple wrapper around CNTK ConfigParameters.
class ImageConfigHelper
{
public:
    explicit ImageConfigHelper(const Microsoft::MSR::CNTK::ConfigParameters& config);

    // Get all streams that are specified in the configuration.
    std::vector<StreamInformation> GetStreams() const;

    // Get index of the feature stream.
    size_t GetFeatureStreamId() const;

    // Get index of the label stream.
    size_t GetLabelStreamId() const;

    // Get the map file path that describes mapping of images into their labels.
    std::string GetMapPath() const;

    Microsoft::MSR::CNTK::ImageLayoutKind GetDataFormat() const
    {
        return m_dataFormat;
    }

    int GetCpuThreadCount() const
    {
        return m_cpuThreadCount;
    }

    bool ShouldRandomize() const
    {
        return m_randomize;
    }

    bool UseGrayscale() const
    {
        return m_grayscale;
    }

    CropType GetCropType() const
    {
        return m_cropType;
    }

    bool IsMultiViewCrop() const
    {
        return m_cropType == CropType::MultiView10;
    }

    static CropType ParseCropType(const std::string &src);

private:
    ImageConfigHelper(const ImageConfigHelper&) = delete;
    ImageConfigHelper& operator=(const ImageConfigHelper&) = delete;

    std::string m_mapPath;
    std::vector<StreamInformation> m_streams;
    Microsoft::MSR::CNTK::ImageLayoutKind m_dataFormat;
    int m_cpuThreadCount;
    bool m_randomize;
    bool m_grayscale;
    CropType m_cropType;
};

typedef std::shared_ptr<ImageConfigHelper> ImageConfigHelperPtr;
}
