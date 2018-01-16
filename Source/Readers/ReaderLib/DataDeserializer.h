//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

// Currently it is a workaround to make CNTKLibrary.h header only
// So that interfaces can be implemented by different libraries without
// binary dependency on CNTKLibrary.so/dll.
// TODO: CNTKLibrary.h should be cleaned up to allow header only dependencies.

#define CNTK_HEADERONLY_DEFINITIONS     1

namespace CNTK {

#pragma warning(push)
#pragma warning(disable : 4996)

#ifdef _MSC_VER
    template <class E>
    __declspec(noreturn) void ThrowFormatted(const char* format, ...);
#else
    template <class E>
    __attribute__((noreturn)) void ThrowFormatted(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

#pragma warning(pop)

}

namespace CNTK { namespace Internal {
    bool IsReversingTensorShapesInErrorMessagesEnabled();
}}

#undef max

// Current a stop gap to redirect to CNTK v2 interfaces/types.
#include "CNTKLibraryExperimental.h"

#include "Config.h"

using Microsoft::MSR::CNTK::ConfigParameters;