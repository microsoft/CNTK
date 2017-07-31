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
    std::vector<std::string> dataExtendNames = TryGetSectionsWithParameter(config, "extendMode");

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

    ConfigParameters label = config(labelNames[0]);
    size_t labelDimension = label("labelDim");

    auto labelSection = std::make_shared<StreamDescription>();
    labelSection->m_id = 1;
    labelSection->m_name = msra::strfun::utf16(label.ConfigName());
    labelSection->m_sampleLayout = std::make_shared<TensorShape>(labelDimension);
    labelSection->m_storageType = StorageType::dense;
    m_streams.push_back(labelSection);

    if (!dataExtendNames.empty())
    {
        if(dataExtendNames.size() != 1)
            InvalidArgument("ImageReader supports a single dataExtend block.");
        ConfigParameters dataExtend = config(dataExtendNames[0]);

        string extendMode = dataExtend("extendMode", "none");
        std::transform(extendMode.begin(), extendMode.end(), extendMode.begin(), ::tolower);
        if (!(extendMode == "none" || extendMode == "equidiff" || extendMode == "expand"))
            InvalidArgument("data extend mode: none, equidiff, expand");
        
        string randomFill = dataExtend("randomFill", "false");
        std::transform(randomFill.begin(), randomFill.end(), randomFill.begin(), ::tolower);
        if (!(randomFill == "true" || randomFill == "false"))
            InvalidArgument("randomFill is a bool value.");

        string useSplitRead = dataExtend("useSplitRead", "false");
        std::transform(useSplitRead.begin(), useSplitRead.end(), useSplitRead.begin(), ::tolower);
        if(!(useSplitRead == "true" || useSplitRead == "false"))
            InvalidArgument("useSplitRead is a bool value.");

        string randomData = dataExtend("randomData", "false");
        std::transform(randomData.begin(), randomData.end(), randomData.begin(), ::tolower);
        if (!(randomData == "true" || randomData == "false"))
            InvalidArgument("randomData is a bool value.");

        auto dataExtendSection = std::make_shared<AppendFuncDescription>();
        dataExtendSection->m_id = 0;
        dataExtendSection->m_name = msra::strfun::utf16(dataExtend.ConfigName());
        (dataExtendSection->m_params)["extendMode"] = extendMode;
        (dataExtendSection->m_params)["extendEpochs"] = dataExtend("extendEpochs", "0");
        (dataExtendSection->m_params)["randomFill"] = randomFill;
        (dataExtendSection->m_params)["randomData"] = randomData;
        (dataExtendSection->m_params)["useSplitRead"] = useSplitRead;
        m_appendFuncs.push_back(dataExtendSection);
    }

    m_mapPath = config(L"file");

    m_grayscale = config(L"grayscale", c == 1);
    std::string rand = config(L"randomize", "auto");

    m_splitReadEpochs = config(L"splitRead", 0);

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

    // Identify precision
    string precision = config.Find("precision", "float");
    if (AreEqualIgnoreCase(precision, "float"))
    {
        features->m_elementType = ElementType::tfloat;
        labelSection->m_elementType = ElementType::tfloat;
    }
    else if (AreEqualIgnoreCase(precision, "double"))
    {
        features->m_elementType = ElementType::tdouble;
        labelSection->m_elementType = ElementType::tdouble;
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

std::vector<AppendFuncDescriptionPtr> ImageConfigHelper::GetAppendFuncs() const
{
    return m_appendFuncs;
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

size_t ImageConfigHelper::GetDataExtendFuncId() const
{
    return 0;
}

std::string ImageConfigHelper::GetMapPath() const
{
    return m_mapPath;
}

int ImageConfigHelper::GetSplitReadEpochs() const
{
    return m_splitReadEpochs;
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

}}}
