//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageConfigHelper.h"
#include "StringUtil.h"
#include "ConfigUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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

    auto features = std::make_shared<StreamDescription>();
    features->m_id = 0;
    features->m_name = msra::strfun::utf16(featureSection.ConfigName());
    features->m_sampleLayout = std::make_shared<TensorShape>(ImageDimensions(w, h, c).AsTensorShape(m_dataFormat));
    features->m_storageType = StorageType::dense;
    m_streams.push_back(features);

    ConfigParameters labelSection = config(labelNames[0]);
    size_t labelDimension = labelSection("labelDim");

    auto labels = std::make_shared<StreamDescription>();
    labels->m_id = 1;
    labels->m_name = msra::strfun::utf16(labelSection.ConfigName());
    labels->m_sampleLayout = std::make_shared<TensorShape>(labelDimension);
    labels->m_storageType = StorageType::dense;
    m_streams.push_back(labels);

    m_mapPath = config(L"file");

    m_grayscale = config(L"grayscale", c == 1);
    std::string rand = config(L"randomize", "auto");

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

    std::string type = labelSection(L"labelType", "classification");
    if (AreEqualIgnoreCase(type, "classification"))
    {
        m_labelType = LabelType::Classification;
    }
    else if (AreEqualIgnoreCase(type, "regression"))
    {
        m_labelType = LabelType::Regression;
    }
    else
    {
        RuntimeError("'labelType' parameter must be set to 'classification' or 'regression'");
    }

    // Identify precision
    string precision = config.Find("precision", "float");
    if (AreEqualIgnoreCase(precision, "float"))
    {
        m_elementType = ElementType::tfloat;
        features->m_elementType = ElementType::tfloat;
        labels->m_elementType = ElementType::tfloat;
    }
    else if (AreEqualIgnoreCase(precision, "double"))
    {
        m_elementType = ElementType::tdouble;
        features->m_elementType = ElementType::tdouble;
        labels->m_elementType = ElementType::tdouble;
    }
    else
    {
        RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());
    }

    m_cpuThreadCount = config(L"numCPUThreads", 0);

    m_cropType = ParseCropType(featureSection(L"cropType", ""));
}

std::vector<StreamDescriptionPtr> ImageConfigHelper::GetStreams() const
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

    if (AreEqualIgnoreCase(src, "random"))
    {
        return CropType::Random;
    }

    if (AreEqualIgnoreCase(src, "multiview10"))
    {
        return CropType::MultiView10;
    }

    RuntimeError("Invalid crop type: %s.", src.c_str());
}

}}}
