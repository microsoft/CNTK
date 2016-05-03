//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Helper function to get sections that contains specified parameter.
inline std::vector<std::string> TryGetSectionsWithParameter(const ConfigParameters& config, const std::string& parameterName)
{
    std::vector<std::string> sectionNames;
    for (const std::pair<std::string, ConfigParameters>& section : config)
    {
        if (section.second.ExistsCurrent(parameterName))
        {
            sectionNames.push_back(section.first);
        }
    }

    return sectionNames;
}

}}}
