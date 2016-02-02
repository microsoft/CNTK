//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <locale>

namespace Microsoft { namespace MSR { namespace CNTK {

// Compares two ASCII strings ignoring the case.
// TODO: Should be moved to common CNTK library and after switching to boost, boost::iequal should be used instead.
inline bool AreEqualIgnoreCase(const std::string& s1, const std::string& s2)
{
    return std::equal(s1.begin(), s1.end(), s2.begin(), [](const char& a, const char& b)
                      {
                          return std::tolower(a) == std::tolower(b);
                      });
}
} } }
