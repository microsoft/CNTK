//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

// Compares two ASCII strings ignoring the case.
// TODO: Should switch to boost, boost::iequal should be used instead.
// TODO: we already have EqualCI() in Basics.h which does the same thing.
template<class TElement>
inline bool AreEqualIgnoreCase(
    const std::basic_string<TElement, char_traits<TElement>, allocator<TElement>>& s1,
    const std::basic_string<TElement, char_traits<TElement>, allocator<TElement> >& s2)
{
    if (s1.size() != s2.size())
    {
        return false;
    }

    return std::equal(s1.begin(), s1.end(), s2.begin(), [](const TElement& a, const TElement& b)
    {
        return std::tolower(a) == std::tolower(b);
    });
}

template<class TString>
inline bool AreEqualIgnoreCase(
    const TString& s1,
    const typename TString::value_type* s2pointer)
{
    return AreEqualIgnoreCase(s1, TString(s2pointer));
}

template<class TString>
inline bool AreEqualIgnoreCase(
    const typename TString::value_type* s1pointer,
    const TString& s2)
{
    return AreEqualIgnoreCase(TString(s1pointer), s2);
}

}}}
