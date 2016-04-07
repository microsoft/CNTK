//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageConfigHelper.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    std::vector<std::string> GetSectionsWithParameter(const ConfigParameters& config, const std::string& parameterName)
    {
        std::vector<std::string> sectionNames;
        for (const std::pair<std::string, ConfigParameters>& section : config)
        {
            if (section.second.ExistsCurrent(parameterName))
            {
                sectionNames.push_back(section.first);
            }
        }

        if (sectionNames.empty())
        {
            RuntimeError("ImageReader requires %s parameter.", parameterName.c_str());
        }

        return sectionNames;
    }

    ImageConfigHelper::ImageConfigHelper(const ConfigParameters& config)
        : m_dataFormat(CHW)
    {
        std::vector<std::string> featureNames = GetSectionsWithParameter(config, "width");
        std::vector<std::string> labelNames = GetSectionsWithParameter(config, "labelDim");

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

        m_mapPath = config(L"file");

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
}}}
