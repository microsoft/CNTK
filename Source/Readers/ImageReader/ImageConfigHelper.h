//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class for image specific parameters.
// A simple wrapper around CNTK ConfigParameters.
class ImageConfigHelper
{
public:
    explicit ImageConfigHelper(const ConfigParameters& config);

    // Get all streams that are specified in the configuration.
    std::vector<StreamDescriptionPtr> GetStreams() const;

    // Get index of the feature stream.
    size_t GetFeatureStreamId() const;

    // Get index of the label stream.
    size_t GetLabelStreamId() const;

    // Get the map file path that describes mapping of images into their labels.
    std::string GetMapPath() const;

    ImageLayoutKind GetDataFormat() const
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

    bool IsMultiViewCrop() const
    {
        return m_multiViewCrop;
    }

private:
    ImageConfigHelper(const ImageConfigHelper&) = delete;
    ImageConfigHelper& operator=(const ImageConfigHelper&) = delete;

    std::string m_mapPath;
    std::vector<StreamDescriptionPtr> m_streams;
    ImageLayoutKind m_dataFormat;
    int m_cpuThreadCount;
    bool m_randomize;
    bool m_multiViewCrop;
};

typedef std::shared_ptr<ImageConfigHelper> ImageConfigHelperPtr;
} } }
