//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"

namespace CNTK {

// Helper function to get sections that contains specified parameter.
inline std::vector<std::string> TryGetSectionsWithParameter(const Microsoft::MSR::CNTK::ConfigParameters& config, const std::string& parameterName)
{
    std::vector<std::string> sectionNames;
    for (const std::pair<std::string, Microsoft::MSR::CNTK::ConfigParameters>& section : config)
    {
        if (section.second.ExistsCurrent(parameterName))
        {
            sectionNames.push_back(section.first);
        }
    }

    return sectionNames;
}

// Helper function to get sections that contains specified parameter. Throws if the parameter does not exist.
inline std::vector<std::string> GetSectionsWithParameter(const std::string& reader, const Microsoft::MSR::CNTK::ConfigParameters& config, const std::string& parameterName)
{
    auto result = TryGetSectionsWithParameter(config, parameterName);
    if (result.empty())
    {
        RuntimeError("%s requires %s parameter.", reader.c_str(), parameterName.c_str());
    }
    return result;
}

// This class allows specifying delimiters and 3 dot patterns
// both for char and wchar_t strings.
template<class T>
struct DirectoryExpansion
{
    static const T* Delimiters() { return "/\\"; }
    static const T* Pattern() { return "..."; }
};

template<>
struct DirectoryExpansion<wchar_t>
{
    static const wchar_t* Delimiters() { return L"/\\"; }
    static const wchar_t* Pattern() { return L"..."; }
};

// Extracts the directory from the absolute path.
template<class TString>
TString ExtractDirectory(const TString& absoluteFilePath)
{
    static_assert(std::is_same<TString, std::string>::value || std::is_same<TString, std::wstring>::value, "Only string types are supported");
    using Char = typename TString::value_type;

    const auto delim = DirectoryExpansion<Char>::Delimiters();
    auto result = absoluteFilePath;
    auto pos = result.find_last_of(delim);
    if (pos != result.npos)
        result.resize(pos);
    return result;
}

// Expands the filePath using the directory if the filePath starts with ...
template<class TString>
TString Expand3Dots(const TString& filePath, const TString& directoryExpansion)
{
    static_assert(std::is_same<TString, std::string>::value || std::is_same<TString, std::wstring>::value, "Only string types are supported");
    using Char = typename TString::value_type;

    const auto extensionPattern = DirectoryExpansion<Char>::Pattern();
    size_t pos = filePath.find(extensionPattern);
    return pos == 0 ? directoryExpansion + filePath.substr(pos + 3) : filePath;
}

}
