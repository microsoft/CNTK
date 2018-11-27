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

namespace CNTK {

// A helper class for HTKMLF configuration.
// Provides typed accessor to config parameters.
class ConfigHelper
{
public:
    ConfigHelper(const Microsoft::MSR::CNTK::ConfigParameters& config) : m_config(config)
    {}

    // Gets context window for augmentation.
    std::pair<size_t, size_t> GetContextWindow();

    // Gets feature dimension.
    size_t GetFeatureDimension();

    // Gets label dimension.
    size_t GetLabelDimension();

    // Gets element type.
    // Currently both features and labels should be of the same type.
    DataType GetDataType() const;

    // Checks feature type in the configuration.
    void CheckFeatureType();

    // Checks lables type in the configuration.
    void CheckLabelType();

    // Returns scp root path.
    std::string GetRootPath();

    // Returns scp file path.
    std::string GetScpFilePath();

    // Returns lattice index file (collection of lattice files)
    std::string GetLatticeIndexFilePath();

    // Returns scp file dir.
    std::string GetScpDir();

    // Adjusts utterance path according to the given root path and scp directory.
    void AdjustUtterancePath(const std::string& rootPath, const string& scpDir, std::string& path);

    // Gets names of feature, label, hmm and lattice files from the configuration.
    void GetDataNamesFromConfig(
        std::vector<std::wstring>& features,
        std::vector<std::wstring>& labels,
        std::vector<std::wstring>& hmms,
        std::vector<std::wstring>& lattices);

    // Gets mlf file paths from the configuration.
    std::vector<std::wstring> GetMlfPaths() const;

    // Gets utterance paths from the configuration.
    std::vector<std::string> GetSequencePaths();

    // Gets randomization window.
    size_t GetRandomizationWindow();

    // Gets randomizer type - "auto" or "block"
    std::wstring GetRandomizer();

    // Gets "cacheIndex" config flag.
    bool GetCacheIndex() const;

    // Gets number of utterances per minibatch for epochs as an array.
    Microsoft::MSR::CNTK::intargvector GetNumberOfUtterancesPerMinibatchForAllEppochs();

private:
    DISABLE_COPY_AND_MOVE(ConfigHelper);

    // Expands ... in the name of the feature path.
    void ExpandDotDotDot(std::string& featPath, const std::string& scpPath, std::string& scpDirCached);

    const Microsoft::MSR::CNTK::ConfigParameters& m_config;
};

}
