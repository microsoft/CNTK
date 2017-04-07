//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <utility>
#include <string>
#include <vector>
#include "Config.h"
#include "Reader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// A helper class for HTKMLF configuration.
// Provides typed accessor to config parameters.
class ConfigHelper
{
public:
    ConfigHelper(const ConfigParameters& config) : m_config(config)
    {}

    // Gets context window for augmentation.
    std::pair<size_t, size_t> GetContextWindow();

    // Gets feature dimension.
    size_t GetFeatureDimension();

    // Gets label dimension.
    size_t GetLabelDimension();

    // Gets element type.
    // Currently both features and labels should be of the same type.
    ElementType GetElementType() const;

    // Checks feature type in the configuration.
    void CheckFeatureType();

    // Checks lables type in the configuration.
    void CheckLabelType();

    // Gets names of feature, label, hmm and lattice files from the configuration.
    void GetDataNamesFromConfig(
        std::vector<std::wstring>& features,
        std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms,
        std::vector<std::wstring>& lattices);

    // Gets mlf file paths from the configuraiton.
    std::vector<std::wstring> GetMlfPaths() const;

    // Gets utterance paths from the configuration.
    std::vector<std::string> GetSequencePaths();

    // Gets randomization window.
    size_t GetRandomizationWindow();

    // Gets randomizer type - "auto" or "block"
    std::wstring GetRandomizer();

    // Gets number of utterances per minibatch for epochs as an array.
    intargvector GetNumberOfUtterancesPerMinibatchForAllEppochs();

private:
    DISABLE_COPY_AND_MOVE(ConfigHelper);

    // Expands ... in the name of the feature path.
    void ExpandDotDotDot(std::string& featPath, const std::string& scpPath, std::string& scpDirCached);

    const ConfigParameters& m_config;
};

}}}
