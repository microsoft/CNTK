// Basics.h -- some shared generally useful pieces of code used by CNTK
//
// We also include a simple "emulation" layer for some proprietary MSVC CRT functions.

#pragma once

#ifndef _BASICS_H_
#define _BASICS_H_

#include "basetypes.h"  // TODO: gradually move over here all that's needed of basetypes.h, then remove basetypes.h.

#define TWO_PI 6.283185307f // TODO: find the official standards-confirming definition of this and use it instead

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    // string comparison class, so we do case insensitive compares
    struct nocase_compare
    {
        // std::string version of 'less' function
        // return false for equivalent, true for different
        bool operator()(const std::string& left, const std::string& right) { return _stricmp(left.c_str(), right.c_str()) < 0; }
        // std::wstring version of 'less' function, used in non-config classes
        bool operator()(const std::wstring& left, const std::wstring& right) { return _wcsicmp(left.c_str(), right.c_str()) < 0; }
    };

}}}

// ===========================================================================
// emulation of some MSVC proprietary CRT
// ===========================================================================

#ifndef _MSC_VER
static inline int _wsystem(const wchar_t *command) { return system(msra::strfun::utf8(command).c_str()); }
#endif

#endif // _BASICS_H_
