// Basics.h -- some shared generally useful pieces of code used by CNTK
//
// We also include a simple "emulation" layer for some proprietary MSVC CRT functions.

#pragma once

#ifndef _BASICS_H_
#define _BASICS_H_

#include "basetypes.h"  // TODO: gradually move over here all that's needed of basetypes.h, then remove basetypes.h.


// ===========================================================================
// emulation of some MSVC proprietary CRT
// ===========================================================================

#ifndef _MSC_VER
static inline int _wsystem(const wchar_t *command) { return system(msra::strfun::utf8(command).c_str()); }
#endif

#endif // _BASICS_H_
