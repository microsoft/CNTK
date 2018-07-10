//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageConfigHelper.h"
#include "StringUtil.h"
#include "ConfigUtil.h"

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

ImageConfigHelper::ImageConfigHelper(const ConfigParameters& config)
    : m_dataFormat(CHW)
{
    std::vector<std::string> featureNames = GetSectionsWithParameter("ImageReader", config, "width");
    std::vector<std::string> labelNames = GetSectionsWithParameter("ImageReader", config, "labelDim");

    // REVIEW alexeyk: currently support only one feature and label section.
    if (featureNames.size() != 1 || labelNames.size() != 1)
    {
        RuntimeError(
            "ImageReader currently supports a single feature and label stream. '%d' features , '%d' labels found.",
            static_cast<int>(featureNames.size()),
            static_cast<int>(labelNames.size()));
    }

    ConfigParameters featureSection = config(featureNames[0]);
    size_t w = featureSection("width");
    size_t h = featureSection("height");
    size_t c = featureSection("channels");

    std::string mbFmt = featureSection("mbFormat", "nchw");
    if (AreEqualIgnoreCase(mbFmt, "nhwc") || AreEqualIgnoreCase(mbFmt, "legacy"))
    {
        m_dataFormat = HWC;
    }
    else if (!AreEqualIgnoreCase(mbFmt, "nchw") || AreEqualIgnoreCase(mbFmt, "cudnn"))
    {
        RuntimeError("ImageReader does not support the sample format '%s', only 'nchw' and 'nhwc' are supported.", mbFmt.c_str());
    }

    StreamInformation features;
    features.m_id = 0;
    features.m_name = Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(featureSection.ConfigName());
    auto dims = ImageDimensions(w, h, c).AsTensorShape(m_dataFormat).GetDims();
    features.m_sampleLayout = NDShape(std::vector<size_t>(dims.begin(), dims.end()));
    features.m_storageFormat = StorageFormat::Dense;

    ConfigParameters label = config(labelNames[0]);
    size_t labelDimension = label("labelDim");

    StreamInformation labelSection;
    labelSection.m_id = 1;
    labelSection.m_name = Microsoft::MSR::CNTK::ToFixedWStringFromMultiByte(label.ConfigName());
    labelSection.m_sampleLayout = NDShape({ labelDimension });
    labelSection.m_storageFormat = StorageFormat::Dense;

    // Identify precision
    string precision = config.Find("precision", "float");
    if (AreEqualIgnoreCase(precision, "float"))
    {
        features.m_elementType = DataType::Float;
        labelSection.m_elementType = DataType::Float;
    }
    else if (AreEqualIgnoreCase(precision, "double"))
    {
        features.m_elementType = DataType::Double;
        labelSection.m_elementType = DataType::Double;
    }
    else
    {
        RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());
    }

    m_streams.push_back(features);
    m_streams.push_back(labelSection);

    m_mapPath = config(L"file");

    m_grayscale = config(L"grayscale", c == 1);
    std::string rand = config(L"randomize", "auto");


    std::string customCropValue = string(featureSection(L"customCrop", "false"));
    if (customCropValue == "true" || customCropValue == "True")
        m_customCrop = true;
    else
        m_customCrop = false;


    if (AreEqualIgnoreCase(rand, "auto"))
    {
        m_randomize = true;
    }
    else if (AreEqualIgnoreCase(rand, "none"))
    {
        m_randomize = false;
    }
    else
    {
        RuntimeError("'randomize' parameter must be set to 'auto' or 'none'");
    }

    

    m_cpuThreadCount = config(L"numCPUThreads", 0);

    m_cropType = ParseCropType(featureSection(L"cropType", ""));
}

std::vector<StreamInformation> ImageConfigHelper::GetStreams() const
{
    return m_streams;
}

size_t ImageConfigHelper::GetFeatureStreamId() const
{
    // Currently we only support a single feature/label stream, so the index is hard-wired.
    return 0;
}

size_t ImageConfigHelper::GetLabelStreamId() const
{
    // Currently we only support a single feature/label stream, so the index is hard-wired.
    return 1;
}

std::string ImageConfigHelper::GetMapPath() const
{
    return m_mapPath;
}

CropType ImageConfigHelper::ParseCropType(const std::string &src)
{
    if (src.empty() || AreEqualIgnoreCase(src, "center"))
    {
        return CropType::Center;
    }

    if (AreEqualIgnoreCase(src, "randomside"))
    {
        return CropType::RandomSide;
    }

    if (AreEqualIgnoreCase(src, "randomarea"))
    {
        return CropType::RandomArea; 
    }

    if (AreEqualIgnoreCase(src, "multiview10"))
    {
        return CropType::MultiView10;
    }

    RuntimeError("Invalid crop type: %s.", src.c_str());
}

}
